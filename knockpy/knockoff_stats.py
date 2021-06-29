import warnings
import numpy as np

# Model-fitters
import sklearn
from sklearn import linear_model, model_selection, ensemble
from group_lasso import GroupLasso, LogisticGroupLasso
from pyglmnet import GLMCV
from . import utilities
from .constants import DEFAULT_REG_VALS


def calc_mse(model, X, y):
    """ Gets MSE of a model """
    preds = model.predict(X)
    resids = (preds - y) / y.std()
    return np.sum((resids) ** 2)


def use_reg_lasso(groups):
    """ Parses whether or not to use group lasso """
    # See if we are using regular lasso...
    if groups is not None:
        p = groups.shape[0]
        m = np.unique(groups).shape[0]
        if p == m:
            return True
        else:
            return False
    else:
        return True


def parse_y_dist(y):
    """ Checks if y is binary; else assumes it is continuous """
    n = y.shape[0]
    if np.unique(y).shape[0] == 2:
        return "binomial"
    elif np.unique(y).shape[0] == n:
        return "gaussian"
    else:
        warnings.warn("Treating y data as continuous even though it may be discrete.")
        return "gaussian"


def parse_logistic_flag(kwargs):
    """ Checks whether y_dist is binomial """
    if "y_dist" in kwargs:
        if kwargs["y_dist"] == "binomial":
            return True
    return False


def combine_Z_stats(Z, groups=None, antisym="cd", group_agg="sum"):
    """
    Given Z scores (variable importances), returns (grouped) feature statistics

    Parameters
    ----------
    Z : np.ndarray
        ``(2p,)``-shaped numpy array of Z-statistics. The first p values
        correspond to true features, and the last p correspond to knockoffs
        (in the same order as the true features).
    groups : np.ndarray
        For group knockoffs, a p-length array of integers from 1 to 
        num_groups such that ``groups[j] == i`` indicates that variable `j`
        is a member of group `i`. Defaults to None (regular knockoffs). 
    antisym : str
        The antisymmetric function used to create (ungrouped) feature
        statistics. Three options: 
        - "CD" (Difference of absolute vals of coefficients),
        - "SM" (signed maximum).
        - "SCD" (Simple difference of coefficients - NOT recommended)
    group_agg : str
        For group knockoffs, specifies how to turn individual feature
        statistics into grouped feature statistics. Two options: 
        "sum" and "avg".

    Returns
    -------
    W : np.ndarray
        an array of feature statistics. This is ``(p,)``-dimensional
        for regular knockoffs and ``(num_groups,)``-dimensional for
        group knockoffs.
    """

    # Step 1: Pairwise W statistics.
    p = int(Z.shape[0] / 2)
    if Z.shape[0] != 2 * p:
        raise ValueError(
            f"Unexpected shape {Z.shape} for Z statistics (expected ({2*p},))"
        )
    if groups is None:
        groups = np.arange(1, p + 1, 1)

    antisym = str(antisym).lower()
    # Absolute coefficient differences
    if antisym == "cd":
        pair_W = np.abs(Z[0:p]) - np.abs(Z[p:])
    # Signed maxes
    elif antisym == "sm":
        inds = np.arange(0, p, 1)
        pair_W = np.maximum(np.abs(Z[inds]), np.abs(Z[inds + p]))
        pair_W = pair_W * np.sign(np.abs(Z[inds]) - np.abs(Z[inds + p]))
    # Simple coefficient differences
    elif antisym == "scd":
        pair_W = Z[0:p] - Z[p:]
    else:
        raise ValueError(f'antisym ({antisym}) must be one of "cd", "sm", "scd"')

    # Step 2: Group statistics
    m = np.unique(groups).shape[0]
    W_group = np.zeros(m)
    for j in range(p):
        W_group[groups[j] - 1] += pair_W[j]

    # If averaging...
    if group_agg == "sum":
        pass
    elif group_agg == "avg":
        group_sizes = utilities.calc_group_sizes(groups)
        W_group = W_group / group_sizes
    else:
        raise ValueError(f'group_agg ({group_agg}) must be one of "sum", "avg"')

    # Return
    return W_group


# ------------------------------ Lasso Stuff ---------------------------------------#
def calc_lars_path(X, Xk, y, groups=None, **kwargs):
    """
    Calculates locations at which X/knockoffs enter lasso 
    model when regressed on y.
    
    Parameters
    ----------
    X : np.ndarray
        the ``(n, p)``-shaped design matrix
    Xk : np.ndarray
        the ``(n, p)``-shaped matrix of knockoffs
    y : np.ndarray
        ``(n,)``-shaped response vector
    groups : np.ndarray
        For group knockoffs, a p-length array of integers from 1 to 
        num_groups such that ``groups[j] == i`` indicates that variable `j`
        is a member of group `i`. Defaults to None (regular knockoffs). 
    **kwargs
        kwargs for ``sklearn.linear_model.lars_path``

    Returns
    -------
    Z : np.ndarray
        ``(2p,)``-shaped array indicating the lasso path statistic
        for each variable. (This means the maximum lambda such that
        the lasso coefficient on variable j is nonzero.) 
     """

    # Ignore y_dist kwargs (residual)
    if "y_dist" in kwargs:
        kwargs.pop("y_dist")

    # Bind data
    p = X.shape[1]
    features = np.concatenate([X, Xk], axis=1)

    # Randomize coordinates to make sure everything is symmetric
    inds, rev_inds = utilities.random_permutation_inds(2 * p)
    features = features[:, inds]

    # By default, all variables are their own group
    if groups is None:
        groups = np.arange(1, p + 1, 1)

    # Fit
    alphas, _, coefs = linear_model.lars_path(features, y, method="lasso", **kwargs,)

    # Calculate places where features enter the model
    Z = np.zeros(2 * p)
    for i in range(2 * p):
        if (coefs[i] != 0).sum() == 0:
            Z[i] = 0
        else:
            Z[i] = alphas[np.where(coefs[i] != 0)[0][0]]

    return Z[rev_inds]


def fit_lasso(X, Xk, y, y_dist=None, use_lars=False, **kwargs):
    """
    Fits cross-validated lasso on [X, Xk] and y.
    
    Parameters
    ----------
    X : np.ndarray
        the ``(n, p)``-shaped design matrix.
    Xk : np.ndarray
        the ``(n, p)``-shaped matrix of knockoffs
    y : np.ndarray
        ``(n,)``-shaped response vector
    y_dist : str
        One of "binomial" or "gaussian"
    use_lars : bool
        If True, uses a LARS-based solver for Gaussian data.
        If False, uses a gradient based solver (default).
    **kwargs
        kwargs for sklearn model.

    Notes
    -----
    To avoid FDR control violations, this randomly permutes
    the features before fitting the lasso.

    Returns
    -------
    gl : sklearn.linear_model.LassoCV/LassoLarsCV/LogisticRegressionCV
        The sklearn model fit through cross-validation.
    inds : np.ndarray
        ``(2p,)``-dimensional array of indices representing the random
        permutation applied to the concatenation of [X, Xk] before fitting
        ``gl.``
    rev_inds : np.ndarray:
        Indices which reverse the effect of ``inds.`` In particular, if
        M is any ``(n, 2p)``-dimensional array, then ```M==M[:, inds][:, rev_inds]```
    """

    # Parse some kwargs/defaults
    max_iter = kwargs.pop("max_iter", 500)
    tol = kwargs.pop("tol", 1e-3)
    cv = kwargs.pop("cv", 5)
    if y_dist is None:
        y_dist = parse_y_dist(y)

    # Bind data
    p = X.shape[1]
    features = np.concatenate([X, Xk], axis=1)

    # Randomize coordinates to make sure everything is symmetric
    inds, rev_inds = utilities.random_permutation_inds(2 * p)
    features = features[:, inds]

    # Fit lasso
    warnings.filterwarnings("ignore")
    if y_dist == "gaussian":
        if not use_lars:
            gl = linear_model.LassoCV(
                alphas=DEFAULT_REG_VALS,
                cv=cv,
                verbose=False,
                max_iter=max_iter,
                tol=tol,
                **kwargs,
            ).fit(features, y)
        elif use_lars:
            gl = linear_model.LassoLarsCV(
                cv=cv, verbose=False, max_iter=max_iter, **kwargs,
            ).fit(features, y)
    elif y_dist == "binomial":
        gl = linear_model.LogisticRegressionCV(
            Cs=1 / DEFAULT_REG_VALS,
            penalty="l1",
            max_iter=max_iter,
            tol=tol,
            cv=cv,
            verbose=False,
            solver="liblinear",
            **kwargs,
        ).fit(features, y)
    else:
        raise ValueError(f"y_dist must be one of gaussian, binomial, not {y_dist}")
    warnings.resetwarnings()

    return gl, inds, rev_inds


def fit_ridge(X, Xk, y, y_dist=None, **kwargs):
    """
    Fits cross-validated ridge on [X, Xk] and y.

    Parameters
    ----------
    X : np.ndarray
        the ``(n, p)``-shaped design matrix
    Xk : np.ndarray
        the ``(n, p)``-shaped matrix of knockoffs
    y : np.ndarray
        ``(n,)``-shaped response vector
    y_dist : str
        One of "binomial" or "gaussian"
    **kwargs
        kwargs for sklearn model.

    Notes
    -----
    To avoid FDR control violations, this randomly permutes
    the features before fitting the ridge.

    Returns
    -------
    gl : sklearn.linear_model.RidgeCV/LogisticRegressionCV
        The sklearn model fit through cross-validation.
    inds : np.ndarray
        ``(2p,)``-dimensional array of indices representing the random
        permutation applied to the concatenation of [X, Xk] before fitting
        ``gl.``
    rev_inds : np.ndarray:
        Indices which reverse the effect of ``inds.`` In particular, if
        M is any ``(n, 2p)``-dimensional array, then ```M==M[:, inds][:, rev_inds]```
    """

    # Bind data
    p = X.shape[1]
    features = np.concatenate([X, Xk], axis=1)

    # Randomize coordinates to ensure antisymmetry
    inds, rev_inds = utilities.random_permutation_inds(2 * p)
    features = features[:, inds]

    # Fit lasso
    warnings.filterwarnings("ignore")
    if y_dist == "gaussian":
        ridge = linear_model.RidgeCV(
            alphas=DEFAULT_REG_VALS,
            store_cv_values=True,
            scoring="neg_mean_squared_error",
            **kwargs,
        ).fit(features, y)
    elif y_dist == "binomial":
        ridge = linear_model.LogisticRegressionCV(
            Cs=1 / DEFAULT_REG_VALS, penalty="l2", solver="liblinear", **kwargs,
        ).fit(features, y)
    else:
        raise ValueError(f"y_dist must be one of gaussian, binomial, not {y_dist}")
    warnings.resetwarnings()

    return ridge, inds, rev_inds


def fit_group_lasso(
    X, Xk, y, groups, use_pyglm=True, y_dist=None, group_lasso=True, **kwargs,
):
    """
    Fits cross-validated ridge on [X, Xk] and y.

    Parameters
    ----------
    X : np.ndarray
        the ``(n, p)``-shaped design matrix
    Xk : np.ndarray
        the ``(n, p)``-shaped matrix of knockoffs
    y : np.ndarray
        ``(n,)``-shaped response vector
    groups : np.ndarray
        For group knockoffs, a p-length array of integers from 1 to 
        num_groups such that ``groups[j] == i`` indicates that variable `j`
        is a member of group `i`. Defaults to None (regular knockoffs). 
    use_pyglm : bool
        When fitting the group lasso, use the pyglm package if True (default).
        Else, use the group_lasso package.
    y_dist : str
        One of "binomial" or "gaussian"
    group_lasso : bool
        If True, use a true group lasso. Else just use the sklearn ungrouped
        lasso.
    **kwargs
        kwargs for eventual (group) lasso model.

    Notes
    -----
    To avoid FDR control violations, this randomly permutes
    the features before fitting the group lasso.

    Returns
    -------
    gl : pyglm/sklearn/group_lasso model
        The model fit through cross-validation; one of many types.
    inds : np.ndarray
        ``(2p,)``-dimensional array of indices representing the random
        permutation applied to the concatenation of [X, Xk] before fitting
        ``gl.``
    rev_inds : np.ndarray:
        Indices which reverse the effect of ``inds.`` In particular, if
        M is any ``(n, 2p)``-dimensional array, then ```M==M[:, inds][:, rev_inds]```
    """

    warnings.filterwarnings("ignore")

    if y_dist is None:
        y_dist = parse_y_dist(y)

    # Bind data
    n = X.shape[0]
    p = X.shape[1]

    # By default, all variables are their own group
    if groups is None:
        groups = np.arange(1, p + 1, 1)
    m = np.unique(groups).shape[0]

    # If m == p, meaning each variable is their own group,
    # just fit a regular lasso
    if m == p or not group_lasso:
        return fit_lasso(X, Xk, y, y_dist, **kwargs)

    # Make sure variables and their knockoffs are in the same group
    # This is necessary for antisymmetry
    doubled_groups = np.concatenate([groups, groups], axis=0)

    # Randomize coordinates to make sure everything is symmetric
    inds, rev_inds = utilities.random_permutation_inds(2 * p)
    features = np.concatenate([X, Xk], axis=1)
    features = features[:, inds]
    doubled_groups = doubled_groups[inds]

    # Standardize - important for pyglmnet performance,
    # highly detrimental for group_lasso performance
    if use_pyglm:
        features = (features - features.mean()) / features.std()
        if y_dist == "gaussian":
            y = (y - y.mean()) / y.std()

    # Get regularization values for cross validation
    reg_vals = kwargs.pop("reg_vals", [(x, x) for x in DEFAULT_REG_VALS])

    # Fit pyglm model using warm starts
    if use_pyglm:

        # Change defaults forkwargs
        kwargs['max_iter'] = kwargs.pop("max_iter", 100)
        kwargs['tol'] = kwargs.pop("tol", 1e-2)
        kwargs['cv'] = kwargs.pop("cv", 5)
        kwargs['learning_rate'] = kwargs.pop("learning_rate", 2)
        l1_regs = [x[0] for x in reg_vals]
        gl = GLMCV(
            distr=y_dist,
            group=doubled_groups,
            alpha=1.0,
            reg_lambda=l1_regs,
            solver="cdfast",
            **kwargs
        )
        gl.fit(features, y)

        # Pull score, rename
        best_score = -1 * calc_mse(gl, features, y)
        best_gl = gl

    # Fit model
    if not use_pyglm:
        best_gl = None
        best_score = -1 * np.inf
        for group_reg, l1_reg in reg_vals:

            # Fit logistic/gaussian group lasso
            kwargs['tol'] = kwargs.pop("tol", 1e-2)
            if y_dist.lower() == "gaussian":
                gl = GroupLasso(
                    groups=doubled_groups,
                    group_reg=group_reg,
                    l1_reg=l1_reg,
                    **kwargs,
                )
            elif y_dist.lower() == "binomial":
                gl = LogisticGroupLasso(
                    groups=doubled_groups,
                    group_reg=group_reg,
                    l1_reg=l1_reg,
                    **kwargs,
                )
            else:
                raise ValueError(
                    f"y_dist must be one of gaussian, binomial, not {y_dist}"
                )

            gl.fit(features, y.reshape(n, 1))
            score = -1 * calc_mse(gl, features, y.reshape(n, 1))

        # Score, possibly select
        if score > best_score:
            best_score = score
            best_gl = gl

    warnings.resetwarnings()

    return best_gl, inds, rev_inds


class FeatureStatistic:
    """
    The base knockoff feature statistic class, which can wrap any
    generic prediction algorithm.

    Parameters
    ----------
    model : 
        An instance of a class with a "train" or "fit" method
        and a "predict" method.

    Attributes
    ----------
    model : 
        A (predictive) model class underlying the variable importance
        measures.
    inds : np.ndarray
        ``(2p,)``-dimensional array of indices representing the random
        permutation applied to the concatenation of [X, Xk] before fitting
        ``gl.``
    rev_inds : np.ndarray:
        Indices which reverse the effect of ``inds.`` In particular, if
        M is any ``(n, 2p)``-dimensional array, then ```M==M[:, inds][:, rev_inds]```
    score : float
        A metric of the model's performance, as defined by ``score_type``.
    score_type : str
        One of MSE, CVMSE, accuracy, or cvaccuracy. (cv stands for
        cross-validated)
    Z : np.ndarray
        a ``2p``-dimsional array of feature and knockoff importances. The
        first p coordinates correspond to features, the last p correspond
        to knockoffs.
    W : np.ndarray
        an array of feature statistics. This is ``(p,)``-dimensional
        for regular knockoffs and ``(num_groups,)``-dimensional for
        group knockoffs.
    groups : np.ndarray
        For group knockoffs, a p-length array of integers from 1 to 
        num_groups such that ``groups[j] == i`` indicates that variable `j`
        is a member of group `i`. Defaults to None (regular knockoffs).
    """

    def __init__(self, model=None):

        self.model = model  # sklearn/statsmodels/pyglmnet model
        self.inds = None  # permutation of features
        self.rev_inds = None  # reverse permutation of features
        self.score = None  # E.g. MSE/CV MSE model
        self.score_type = None  # One of MSE, CVMSE, or accuracy
        self.Z = None  # Z statistic
        self.groups = None  # Grouping of features for group knockoffs
        self.W = None  # W statistic

    def fit(
        self,
        X,
        Xk,
        y,
        groups=None,
        feature_importance="swap",
        antisym="cd",
        group_agg="avg",
        **kwargs,
    ):
        """
        Trains the model and creates feature importances.

        Parameters
        ----------
        X : np.ndarray
            the ``(n, p)``-shaped design matrix
        Xk : np.ndarray
            the ``(n, p)``-shaped matrix of knockoffs
        y : np.ndarray
            ``(n,)``-shaped response vector
        groups : np.ndarray
            For group knockoffs, a p-length array of integers from 1 to 
            num_groups such that ``groups[j] == i`` indicates that variable `j`
            is a member of group `i`. Defaults to None (regular knockoffs). 
        feature_importance : str
            Specifies how to create feature importances from ``model``. 
            Two options:
                - "swap": The default swap-statistic from 
                http://proceedings.mlr.press/v89/gimenez19a/gimenez19a.pdf.
                These are good measures of feature importance but
                slightly slower.
                - "swapint": The swap-integral defined from
                http://proceedings.mlr.press/v89/gimenez19a/gimenez19a.pdf
            Defaults to 'swap'
        antisym : str
            The antisymmetric function used to create (ungrouped) feature
            statistics. Three options: 
            - "CD" (Difference of absolute vals of coefficients),
            - "SM" (signed maximum).
            - "SCD" (Simple difference of coefficients - NOT recommended)
        group_agg : str
            For group knockoffs, specifies how to turn individual feature
            statistics into grouped feature statistics. Two options: 
            "sum" and "avg".
        **kwargs : **dict
            kwargs to pass to the 'train' or 'fit' method of the model.
        
        Returns
        -------
        W : np.ndarray
            an array of feature statistics. This is ``(p,)``-dimensional
            for regular knockoffs and ``(num_groups,)``-dimensional for
            group knockoffs.
        """

        if self.model is None:
            raise ValueError(
                "For base feature statistic class, must provide a trainable model class instance."
            )

        # Permute features to prevent FDR control violations
        n = X.shape[0]
        p = X.shape[1]
        features = np.concatenate([X, Xk], axis=1)
        self.inds, self.rev_inds = utilities.random_permutation_inds(2 * p)
        features = features[:, self.inds]

        # Train model
        if hasattr(self.model, "train"):
            self.model.train(features, y, **kwargs)
        elif hasattr(self.model, "fit"):
            self.model.fit(features, y, **kwargs)
        else:
            raise ValueError(
                f"model {self.model} must have either a 'fit' or 'train' method"
            )

        # Score using swap importances
        if feature_importance == "swap":
            self.Z = self.swap_feature_importances(features, y)
        elif feature_importance == "swapint":
            self.Z = self.swap_path_feature_importances(features, y)
        else:
            raise ValueError(f"Unrecognized feature_importance {feature_importance}")

        # Combine Z statistics
        self.groups = groups
        self.W = combine_Z_stats(
            self.Z, self.groups, antisym=antisym, group_agg=group_agg
        )
        return self.W

    def swap_feature_importances(self, features, y):
        """
        Given a model of the features and y, calculates feature importances
        as follows.

        For feature i, replace the feature with its knockoff and calculate
        the relative increase in the loss. Similarly, for knockoff i, 
        replace the knockoffs with its feature and calculate the relative
        increase in the loss.

        Parameters
        ----------
        features : np.ndarray
            ``(n, 2p)``-shaped array of concatenated features and knockoffs,
            which must be permuted by ``self.inds``.
        y : np.ndarray
            ``(n,)``-shaped response vector

        Returns
        -------
        Z_swap : np.ndarray
            ``(2p,)``-shaped array of variable importances such that
            Z_swap is in the same permuted as ``features`` initially were.
        """

        # Parse aspects of the DGP
        n = features.shape[0]
        p = int(features.shape[1] / 2)
        y_dist = parse_y_dist(y)

        # Iteratively replace columns with their knockoffs/features
        Z_swap = np.zeros(2 * p)
        for i in range(p):
            for knockoff in [0, 1]:

                # Unshuffle features and replace column
                new_features = features[:, self.rev_inds].copy()
                col = i + knockoff * p  # The column we calculate the score for
                partner = i + (1 - knockoff) * p  # its corresponding feature/knockoff
                new_features[:, col] = new_features[:, partner]
                new_features = new_features[:, self.inds]  # Reshuffle cols for model

                # Calculate loss
                Z_swap[col] = self.score_model(new_features, y, y_dist=y_dist)

        return Z_swap

    def swap_path_feature_importances(self, features, y, step_size=0.5, max_lambda=5):
        """
        Similar to ``swap_feature_importances``; see
        http://proceedings.mlr.press/v89/gimenez19a/gimenez19a.pdf
        """

        # Parse aspects of the DGP
        n = features.shape[0]
        p = int(features.shape[1] / 2)
        y_dist = parse_y_dist(y)

        # Baseline loss
        baseline_loss = self.score_model(features, y, y_dist=y_dist)

        # Iteratively replace columns with their knockoffs/features
        lambda_vals = []
        lambd = step_size
        while lambd <= max_lambda:
            lambda_vals.append(lambd)
            lambd += step_size
        lambda_vals = np.array(lambda_vals)

        # DP approach to calculating area
        Z_swap_lambd_prev = np.zeros(2 * p) + baseline_loss
        Z_swap_int = np.zeros(2 * p)
        for lambd in lambda_vals:
            for i in range(p):
                for knockoff in [0, 1]:

                    # Unshuffle features and replace column
                    new_features = features[:, self.rev_inds].copy()
                    col = i + knockoff * p  # The column we calculate the score for
                    partner = (
                        i + (1 - knockoff) * p
                    )  # its corresponding feature/knockoff

                    # Calc new col as linear interpolation btwn col and partner
                    fk_diff = new_features[:, partner] - new_features[:, col]
                    new_col = new_features[:, col] + lambd * (fk_diff)

                    # Set new column and reshuffle
                    new_features[:, col] = new_col
                    new_features = new_features[:, self.inds]  # Reshuffle

                    # Calculate area under curve
                    loss = self.score_model(new_features, y, y_dist=y_dist)
                    avg_trapezoid_height = (Z_swap_lambd_prev[col] + loss) / 2
                    Z_swap_int[col] += step_size * avg_trapezoid_height

                    # Cache for DP
                    Z_swap_lambd_prev[col] = loss

        return Z_swap_int

    def score_model(self, features, y, y_dist=None):
        """
        Computes mean-squared error of self.model on
        (features, y) when y is nonbinary, and computes
        1 - accuracy otherwise.

        Returns
        -------
        loss : float
            Either the MSE or one minus the accuracy of the model,
            depending on whether y is continuous or binary.
        """

        # Make sure model exists
        if self.model is None:
            raise ValueError("Must train self.model before calling model_training_loss")

        # Parse y distribution
        if y_dist is None:
            y_dist = parse_y_dist(y)

        # MSE for gaussian data
        if y_dist == "gaussian":
            preds = self.model.predict(features)
            loss = np.power(preds - y, 2).mean()
        # 1- accuracy for binomial data
        elif y_dist == "binomial":
            preds = self.model.predict(features)
            accuracy = (preds == y).mean()
            loss = 1 - accuracy
            # log_probs = self.model.predict_log_proba(features)
            # log_probs[log_probs == -np.inf] = -10 # Numerical errors
            # # TODO: should normalize
            # loss = -1*log_probs[:, y.astype('int32')].mean()
        else:
            raise ValueError(f"Unexpected y_dist = {y_dist}")

        return loss

    def cv_score_model(self, features, y, cv_score, logistic_flag=False):
        """
        Similar to score_model, but uses cross-validated scoring if cv_score=True.
        """

        # Possibly, compute CV MSE/Accuracy, although this
        # can be very expensive (e.g. for LARS solver)
        if cv_score:
            if isinstance(self.model, sklearn.base.RegressorMixin):

                # Parse whether to use MSE or Accuracy
                if logistic_flag:
                    self.score_type = "accuracy_cv"
                    scoring = "accuracy"
                else:
                    self.score_type = "mse_cv"
                    scoring = "neg_mean_squared_error"
                cv_scores = model_selection.cross_val_score(
                    self.model, features, y, cv=5, scoring=scoring,
                )

                # Adjust negative mse to be positive
                if scoring == "neg_mean_squared_error":
                    cv_scores = -1 * cv_scores

                # Take the mean
                self.score = cv_scores.mean()

            else:
                raise ValueError(
                    f"Model is of {type(self.model)}, must be sklearn estimator for cvscoring"
                )
        else:
            if logistic_flag:
                y_dist = "binomial"
                self.score_type = "log_likelihood"
            else:
                y_dist = "gaussian"
                self.score_type = "mse"
            self.score = -1 * self.score_model(features, y, y_dist=y_dist)


class RidgeStatistic(FeatureStatistic):
    """ Ridge statistic wrapper class """

    def __init__(self):

        super().__init__()

    def fit(
        self,
        X,
        Xk,
        y,
        groups=None,
        antisym="cd",
        group_agg="avg",
        cv_score=False,
        **kwargs,
    ):
        """
        Wraps the FeatureStatistic class but uses cross-validated Ridge
        coefficients as variable importances.

        Parameters
        ----------
        X : np.ndarray
            the ``(n, p)``-shaped design matrix
        Xk : np.ndarray
            the ``(n, p)``-shaped matrix of knockoffs
        y : np.ndarray
            ``(n,)``-shaped response vector
        groups : np.ndarray
            For group knockoffs, a p-length array of integers from 1 to 
            num_groups such that ``groups[j] == i`` indicates that variable `j`
            is a member of group `i`. Defaults to None (regular knockoffs). 
        antisym : str
            The antisymmetric function used to create (ungrouped) feature
            statistics. Three options: 
            - "CD" (Difference of absolute vals of coefficients),
            - "SM" (signed maximum).
            - "SCD" (Simple difference of coefficients - NOT recommended)
        group_agg : str
            For group knockoffs, specifies how to turn individual feature
            statistics into grouped feature statistics. Two options: 
            "sum" and "avg".
        cv_score : bool
            If true, score the feature statistic's predictive accuracy
            using cross validation.
        kwargs : dict
            Extra kwargs to pass to underlying Lasso classes

        Returns
        -------
        W : np.ndarray
            an array of feature statistics. This is ``(p,)``-dimensional
            for regular knockoffs and ``(num_groups,)``-dimensional for
            group knockoffs.
        """

        # Possibly set default groups
        n = X.shape[0]
        p = X.shape[1]
        if groups is None:
            groups = np.arange(1, p + 1, 1)

        # Check if y_dist is gaussian, binomial
        y_dist = parse_y_dist(y)
        kwargs["y_dist"] = y_dist

        # Step 1: Calculate Z stats by fitting ridge
        self.model, self.inds, self.rev_inds = fit_ridge(X=X, Xk=Xk, y=y, **kwargs,)

        # Retrieve Z statistics and save cv scores
        if y_dist == "gaussian":
            Z = self.model.coef_[self.rev_inds]
            self.score = -1 * self.model.cv_values_.mean(axis=1).min()
            self.score_type = "mse_cv"
        elif y_dist == "binomial":
            Z = self.model.coef_[0, self.rev_inds]
            self.score = self.model.scores_[1].mean(axis=0).max()
            self.score_type = "accuracy_cv"
        else:
            raise ValueError(
                f"y_dist must be one of gaussian, binomial, not {kwargs['y_dist']}"
            )

        # Combine Z statistics
        W_group = combine_Z_stats(Z, groups, antisym=antisym, group_agg=group_agg)

        # Save values for later use
        self.Z = Z
        self.groups = groups
        self.W = W_group
        return W_group


class LassoStatistic(FeatureStatistic):
    """ Lasso Statistic wrapper class """

    def __init__(self):

        super().__init__()

    def fit(
        self,
        X,
        Xk,
        y,
        groups=None,
        zstat="coef",
        use_pyglm=True,
        group_lasso=False,
        antisym="cd",
        group_agg="avg",
        cv_score=False,
        debias=False,
        Ginv=None,
        **kwargs,
    ):
        """
        Wraps the FeatureStatistic class but uses cross-validated Lasso
        coefficients or Lasso path statistics as variable importances.

        Parameters
        ----------
        X : np.ndarray
            the ``(n, p)``-shaped design matrix
        Xk : np.ndarray
            the ``(n, p)``-shaped matrix of knockoffs
        y : np.ndarray
            ``(n,)``-shaped response vector
        groups : np.ndarray
            For group knockoffs, a p-length array of integers from 1 to 
            num_groups such that ``groups[j] == i`` indicates that variable `j`
            is a member of group `i`. Defaults to None (regular knockoffs). 
        zstat : str:
            Two options for the variable importance measure:
            - If 'coef', uses to cross-validated (group) lasso coefficients.
            - If 'lars_path', uses the lambda value where each feature/knockoff
            enters the lasso path (meaning becomes nonzero).
            This defaults to coef.
        use_pyglm : bool
            When fitting the group lasso, use the pyglm package if True (default).
            Else, use the group_lasso package.
        y_dist : str
            One of "binomial" or "gaussian"
        group_lasso : bool
            If True, use a true group lasso. Else just use the sklearn ungrouped
            lasso.
        antisym : str
            The antisymmetric function used to create (ungrouped) feature
            statistics. Three options: 
            - "CD" (Difference of absolute vals of coefficients),
            - "SM" (signed maximum).
            - "SCD" (Simple difference of coefficients - NOT recommended)
        group_agg : str
            For group knockoffs, specifies how to turn individual feature
            statistics into grouped feature statistics. Two options: 
            "sum" and "avg".
        cv_score : bool
            If true, score the feature statistic's predictive accuracy
            using cross validation.
        debias : bool:
            If true, debias the lasso. See https://arxiv.org/abs/1508.02757
        Ginv : np.ndarray
            ``(2p, 2p)``-shaped precision matrix for the feature-knockoff 
            covariate distribution. This must be specified if ``debias=True``.
        kwargs : dict
            Extra kwargs to pass to underlying Lasso classes

        Returns
        -------
        W : np.ndarray
            an array of feature statistics. This is ``(p,)``-dimensional
            for regular knockoffs and ``(num_groups,)``-dimensional for
            group knockoffs.
        """

        # Possibly set default groups
        n = X.shape[0]
        p = X.shape[1]
        if groups is None:
            groups = np.arange(1, p + 1, 1)

        # Check if y_dist is gaussian, binomial, poisson
        kwargs["y_dist"] = parse_y_dist(y)

        # Step 1: Calculate Z statistics
        zstat = str(zstat).lower()
        if zstat == "coef":

            # Fit (possibly group) lasso
            gl, inds, rev_inds = fit_group_lasso(
                X=X,
                Xk=Xk,
                y=y,
                groups=groups,
                use_pyglm=use_pyglm,
                group_lasso=group_lasso,
                **kwargs,
            )

            # Parse the expected output format based on which
            # lasso package we are using
            reg_lasso_flag = use_reg_lasso(groups) or (not group_lasso)
            logistic_flag = parse_logistic_flag(kwargs)

            # Retrieve Z statistics
            if use_pyglm and not reg_lasso_flag:
                Z = gl.beta_[rev_inds]
            elif reg_lasso_flag and logistic_flag:
                if gl.coef_.shape[0] != 1:
                    raise ValueError(
                        "Unexpected shape for logistic lasso coefficients (sklearn)"
                    )
                Z = gl.coef_[0, rev_inds]
            else:
                Z = gl.coef_[rev_inds]

            # Possibly debias the lasso
            if debias:
                if Ginv is None:
                    raise ValueError(f"To debias the lasso, Ginv must be provided")
                elif logistic_flag:
                    raise ValueError(
                        f"Debiased lasso is not implemented for binomial data"
                    )
                else:
                    features = np.concatenate([X, Xk], axis=1)
                    debias_term = np.dot(Ginv, features.T)
                    debias_term = np.dot(debias_term, y - np.dot(features, Z))
                    Z = Z + debias_term / n

            # Save lasso class and reverse inds
            self.model = gl
            self.inds = inds
            self.rev_inds = rev_inds

            # Try to save cv accuracy for logistic lasso
            if isinstance(self.model, linear_model.LogisticRegressionCV):
                self.score = self.model.scores_[1].mean(axis=0).max()
                self.score_type = "accuracy_cv"
            # Save cv mse for lasso
            elif isinstance(self.model, linear_model.LassoCV):
                self.score = self.model.mse_path_.mean(axis=1).min()
                self.score_type = "mse_cv"
            # Else compute the score
            else:
                features = np.concatenate([X, Xk], axis=1)[:, inds]
                self.cv_score_model(
                    features=features,
                    y=y,
                    cv_score=cv_score,
                    logistic_flag=logistic_flag,
                )

        elif zstat == "lars_path":
            Z = calc_lars_path(X, Xk, y, groups, **kwargs)

        else:
            raise ValueError(f'zstat ({zstat}) must be one of "coef", "lars_path"')

        # Combine Z statistics
        W_group = combine_Z_stats(Z, groups, antisym=antisym, group_agg=group_agg)

        # Save values for later use
        self.Z = Z
        self.groups = groups
        self.W = W_group
        return W_group


class MargCorrStatistic(FeatureStatistic):
    """ Lasso Statistic wrapper class """

    def __init__(self):

        super().__init__()

    def fit(
        self, X, Xk, y, groups=None, **kwargs,
    ):
        """
        Wraps the FeatureStatistic class using marginal correlations 
        between X, Xk and y as variable importances.

        Parameters
        ----------
        X : np.ndarray
            the ``(n, p)``-shaped design matrix
        Xk : np.ndarray
            the ``(n, p)``-shaped matrix of knockoffs
        y : np.ndarray
            ``(n,)``-shaped response vector
        groups : np.ndarray
            For group knockoffs, a p-length array of integers from 1 to 
            num_groups such that ``groups[j] == i`` indicates that variable `j`
            is a member of group `i`. Defaults to None (regular knockoffs). 
        kwargs : dict
            Extra kwargs to pass to underlying ``combine_Z_stats``

        Returns
        -------
        W : np.ndarray
            an array of feature statistics. This is ``(p,)``-dimensional
            for regular knockoffs and ``(num_groups,)``-dimensional for
            group knockoffs.
        """

        # Calc correlations
        features = np.concatenate([X, Xk], axis=1)
        correlations = np.corrcoef(features, y.reshape(-1, 1), rowvar=False)[-1][0:-1]

        # Combine
        W = combine_Z_stats(correlations, groups, **kwargs)

        # Cache
        self.Z = correlations
        self.groups = groups
        self.W = W

        return W


class OLSStatistic(FeatureStatistic):
    """ Lasso Statistic wrapper class """

    def __init__(self):

        super().__init__()

    def fit(self, X, Xk, y, groups=None, cv_score=False, **kwargs):
        """
        Wraps the FeatureStatistic class with OLS coefs as variable importances.

        Parameters
        ----------
        X : np.ndarray
            the ``(n, p)``-shaped design matrix
        Xk : np.ndarray
            the ``(n, p)``-shaped matrix of knockoffs
        y : np.ndarray
            ``(n,)``-shaped response vector
        groups : np.ndarray
            For group knockoffs, a p-length array of integers from 1 to 
            num_groups such that ``groups[j] == i`` indicates that variable `j`
            is a member of group `i`. Defaults to None (regular knockoffs). 
        cv_score : bool
            If true, score the feature statistic's predictive accuracy
            using cross validation.
        kwargs : dict
            Extra kwargs to pass to ``combine_Z_stats``.

        Returns
        -------
        W : np.ndarray
            an array of feature statistics. This is ``(p,)``-dimensional
            for regular knockoffs and ``(num_groups,)``-dimensional for
            group knockoffs.
        """

        # Run linear regression, permute indexes to prevent FDR violations
        p = X.shape[1]
        features = np.concatenate([X, Xk], axis=1)
        inds, rev_inds = utilities.random_permutation_inds(2 * p)
        features = features[:, inds]

        lm = linear_model.LinearRegression(fit_intercept=False).fit(features, y)
        Z = lm.coef_

        # Play with shape
        Z = Z.reshape(-1)
        if Z.shape[0] != 2 * p:
            raise ValueError(
                f"Unexpected shape {Z.shape} for sklearn LinearRegression coefs (expected ({2*p},))"
            )

        # Undo random permutation
        Z = Z[rev_inds]

        # Combine with groups to create W-statistic
        W = combine_Z_stats(Z, groups, **kwargs)

        # Save
        self.model = lm
        self.inds = inds
        self.rev_inds = rev_inds
        self.Z = Z
        self.groups = groups
        self.W = W

        # Score model
        self.cv_score_model(
            features=features, y=y, cv_score=cv_score,
        )

        return W


class RandomForestStatistic(FeatureStatistic):
    def fit(
        self,
        X,
        Xk,
        y,
        groups=None,
        feature_importance="swap",
        antisym="cd",
        group_agg="sum",
        cv_score=False,
        **kwargs,
    ):
        """
        Wraps the FeatureStatistic class using a Random Forest to 
        generate variable importances.

        Parameters
        ----------
        X : np.ndarray
            the ``(n, p)``-shaped design matrix
        Xk : np.ndarray
            the ``(n, p)``-shaped matrix of knockoffs
        y : np.ndarray
            ``(n,)``-shaped response vector
        groups : np.ndarray
            For group knockoffs, a p-length array of integers from 1 to 
            num_groups such that ``groups[j] == i`` indicates that variable `j`
            is a member of group `i`. Defaults to None (regular knockoffs). 
        feature_importance : str
            Specifies how to create feature importances from ``model``. 
            Three options:
                - "sklearn": Use sklearn feature importances. These
                are very poor measures of feature importance, but
                very fast.
                - "swap": The default swap-statistic from 
                http://proceedings.mlr.press/v89/gimenez19a/gimenez19a.pdf.
                These are good measures of feature importance but
                slightly slower.
                - "swapint": The swap-integral defined from
                http://proceedings.mlr.press/v89/gimenez19a/gimenez19a.pdf
            Defaults to 'swap'
        antisym : str
            The antisymmetric function used to create (ungrouped) feature
            statistics. Three options: 
            - "CD" (Difference of absolute vals of coefficients),
            - "SM" (signed maximum).
            - "SCD" (Simple difference of coefficients - NOT recommended)
        group_agg : str
            For group knockoffs, specifies how to turn individual feature
            statistics into grouped feature statistics. Two options: 
            "sum" and "avg".
        cv_score : bool
            If true, score the feature statistic's predictive accuracy
            using cross validation. This is extremely expensive for random
            forests.
        kwargs : dict
            Extra kwargs to pass to underlying RandomForest class

        Returns
        -------
        W : np.ndarray
            an array of feature statistics. This is ``(p,)``-dimensional
            for regular knockoffs and ``(num_groups,)``-dimensional for
            group knockoffs.
        """

        # Bind data
        p = X.shape[1]
        features = np.concatenate([X, Xk], axis=1)

        # Randomize coordinates to make sure everything is symmetric
        self.inds, self.rev_inds = utilities.random_permutation_inds(2 * p)
        features = features[:, self.inds]

        # By default, all variables are their own group
        if groups is None:
            groups = np.arange(0, p, 1)
        self.groups = groups

        # Parse y_dist, initialize model
        y_dist = parse_y_dist(y)
        # Avoid future warnings
        if "n_estimators" not in kwargs:
            kwargs["n_estimators"] = 10
        if y_dist == "gaussian":
            self.model = ensemble.RandomForestRegressor(**kwargs)
        else:
            self.model = ensemble.RandomForestClassifier(**kwargs)

        # Fit model, get Z statistics
        self.model.fit(features, y)
        feature_importance = str(feature_importance).lower()
        if feature_importance == "default":
            self.Z = self.model.feature_importances_[self.rev_inds]
        elif feature_importance == "swap":
            self.Z = self.swap_feature_importances(features, y)
        elif feature_importance == "swapint":
            self.Z = self.swap_path_feature_importances(features, y)
        else:
            raise ValueError(
                f"feature_importance {feature_importance} must be one of 'swap', 'swapint', 'default'"
            )

        # Get W statistics
        self.W = combine_Z_stats(
            self.Z, self.groups, antisym=antisym, group_agg=group_agg
        )

        # Possibly score model
        self.cv_score_model(features=features, y=y, cv_score=cv_score)

        return self.W


class DeepPinkStatistic(FeatureStatistic):
    def fit(
        self,
        X,
        Xk,
        y,
        groups=None,
        feature_importance="deeppink",
        antisym="cd",
        group_agg="sum",
        cv_score=False,
        train_kwargs={"verbose": False},
        **kwargs,
    ):
        """
        Wraps the FeatureStatistic class using DeepPINK to generate
        variable importances.

        Parameters
        ----------
        X : np.ndarray
            the ``(n, p)``-shaped design matrix
        Xk : np.ndarray
            the ``(n, p)``-shaped matrix of knockoffs
        y : np.ndarray
            ``(n,)``-shaped response vector
        groups : np.ndarray
            For group knockoffs, a p-length array of integers from 1 to 
            num_groups such that ``groups[j] == i`` indicates that variable `j`
            is a member of group `i`. Defaults to None (regular knockoffs). 
        feature_importance : str
            Specifies how to create feature importances from ``model``. 
            Four options:
                - "deeppink": Use the deeppink feature importance
                defined in https://arxiv.org/abs/1809.01185
                - "unweighted": Use the Z weights from the deeppink
                paper without weighting them using the layers from
                the MLP. Deeppink usually outperforms this feature
                importance (but not always).
                - "swap": The default swap-statistic from 
                http://proceedings.mlr.press/v89/gimenez19a/gimenez19a.pdf
                - "swapint": The swap-integral defined from
                http://proceedings.mlr.press/v89/gimenez19a/gimenez19a.pdf
            Defaults to deeppink, which is often both the most powerful and 
            the most computationally efficient.
        antisym : str
            The antisymmetric function used to create (ungrouped) feature
            statistics. Three options: 
            - "CD" (Difference of absolute vals of coefficients),
            - "SM" (signed maximum).
            - "SCD" (Simple difference of coefficients - NOT recommended)
        group_agg : str
            For group knockoffs, specifies how to turn individual feature
            statistics into grouped feature statistics. Two options: 
            "sum" and "avg".
        cv_score : bool
            If true, score the feature statistic's predictive accuracy
            using cross validation. This is extremely expensive for random
            forests.
        kwargs : dict
            Extra kwargs to pass to underlying RandomForest class

        Returns
        -------
        W : np.ndarray
            an array of feature statistics. This is ``(p,)``-dimensional
            for regular knockoffs and ``(num_groups,)``-dimensional for
            group knockoffs.
        """

        # Check if kpytorch (and therefore deeppink) is available.
        utilities.check_kpytorch_available(purpose="deepPINK statistics")
        from .kpytorch import deeppink

        # Bind data
        n = X.shape[0]
        p = X.shape[1]
        features = np.concatenate([X, Xk], axis=1)

        # Randomize coordinates to make sure everything is symmetric
        self.inds = np.arange(0, 2 * p, 1)
        self.rev_inds = np.arange(0, 2 * p, 1)

        # By default, all variables are their own group
        if groups is None:
            groups = np.arange(0, p, 1)
        self.groups = groups

        # Parse y_dist, hidden_sizes, initialize model
        y_dist = parse_y_dist(y)
        if "hidden_sizes" not in kwargs:
            kwargs["hidden_sizes"] = [min(n, p)]
        self.model = deeppink.DeepPinkModel(
            p=p, inds=self.inds, rev_inds=self.inds, **kwargs
        )
        # Train model
        self.model.train()
        self.model = deeppink.train_deeppink(self.model, features, y, **train_kwargs)
        self.model.eval()

        # Get Z statistics
        feature_importance = str(feature_importance).lower()
        if feature_importance == "deeppink":
            self.Z = self.model.feature_importances()
        elif feature_importance == "unweighted":
            self.Z = self.model.feature_importances(weight_scores=False)
        elif feature_importance == "swap":
            self.Z = self.swap_feature_importances(features, y)
        elif feature_importance == "swapint":
            self.Z = self.swap_path_feature_importances(features, y)
        else:
            raise ValueError(
                f"feature_importance {feature_importance} must be one of 'deeppink', 'unweighted', 'swap', 'swapint'"
            )

        # Get W statistics
        self.W = combine_Z_stats(
            self.Z, self.groups, antisym=antisym, group_agg=group_agg
        )

        # Possibly score model
        self.cv_score_model(features=features, y=y, cv_score=cv_score)

        return self.W


def data_dependent_threshhold(W, fdr=0.10, offset=1):
    """
    Calculate data-dependent threshhold given W statistics.
    
    Parameters
    ----------
    W : np.ndarray
        p-length numpy array of feature statistics OR (p, batch_length) 
        shaped array.
    fdr : float
        desired level of false discovery rate control
    offset : int
        If offset = 0, control the modified FDR.
        If offset = 1 (default), controls the FDR exactly.

    Returns
    -------
    T : float or np.ndarray
        The data-dependent threshhold. Either a float or a (batch_length,) 
        dimensional array.
    """

    # Add dummy batch axis if necessary
    if len(W.shape) == 1:
        W = W.reshape(-1, 1)
    p = W.shape[0]
    batch = W.shape[1]

    # Sort W by absolute values
    ind = np.argsort(-1 * np.abs(W), axis=0)
    sorted_W = np.take_along_axis(W, ind, axis=0)

    # Calculate ratios
    negatives = np.cumsum(sorted_W <= 0, axis=0)
    positives = np.cumsum(sorted_W > 0, axis=0)
    positives[positives == 0] = 1  # Don't divide by 0
    ratios = (negatives + offset) / positives

    # Add zero as an option to prevent index errors
    # (zero means select everything strictly > 0)
    sorted_W = np.concatenate([sorted_W, np.zeros((1, batch))], axis=0)

    # Find maximum indexes satisfying FDR control
    # Adding np.arange is just a batching trick
    helper = (ratios <= fdr) + np.arange(0, p, 1).reshape(-1, 1) / p
    sorted_W[1:][helper < 1] = np.inf  # Never select values where the ratio > fdr
    T_inds = np.argmax(helper, axis=0) + 1
    more_inds = np.indices(T_inds.shape)

    # Find Ts
    acceptable = np.abs(sorted_W)[T_inds, more_inds][0]

    # Replace 0s with a very small value to ensure that
    # downstream you don't select W statistics == 0.
    # This value is the smallest abs value of nonzero W
    if np.sum(acceptable == 0) != 0:
        W_new = W.copy()
        W_new[W_new == 0] = np.abs(W).max()
        zero_replacement = np.abs(W_new).min(axis=0)
        acceptable[acceptable == 0] = zero_replacement[acceptable == 0]

    if batch == 1:
        acceptable = acceptable[0]

    return acceptable
