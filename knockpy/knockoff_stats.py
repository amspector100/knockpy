import warnings

import numpy as np

# Model-fitters
import sklearn
from sklearn import ensemble, linear_model, model_selection

from . import utilities
from .constants import DEFAULT_REG_VALS


def calc_mse(model, X, y):
    """Gets MSE of a model"""
    preds = model.predict(X)
    resids = (preds - y) / y.std()
    return np.sum((resids) ** 2)


def parse_y_dist(y):
    """Checks if y is binary; else assumes it is continuous"""
    n = y.shape[0]
    if np.unique(y).shape[0] == 2:
        return "binomial"
    elif np.unique(y).shape[0] == n:
        return "gaussian"
    else:
        # warnings.warn("Treating y data as continuous even though it may be discrete.")
        return "gaussian"


def parse_logistic_flag(kwargs):
    """Checks whether y_dist is binomial"""
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
            f"Unexpected shape {Z.shape} for Z statistics (expected ({2 * p},))"
        )
    if groups is None:
        groups = np.arange(1, p + 1)
    else:
        groups = utilities.preprocess_groups(groups)

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


def compute_residual_variance(X, Xk, y):
    """
    Estimates sigma2 using residual variance of y
    given [X, Xk]. Uses a penalized linear model
    if n < 2p + 10 or p >= 1500 (for efficiency).

    Returns
    -------
    sigma2 : float
        Estimated residual variance
    """
    # Use (penalized) linear model to predict y
    n, p = X.shape
    if n - 2 * p > 10 and p < 1500:
        model = linear_model.LinearRegression()
    else:
        model = linear_model.Lasso(alpha=0.1 * np.sqrt(np.log(p) / n), max_iter=100)
    # Concatenating features shows no preference between X, Xk
    features = np.concatenate([X, Xk], axis=1)
    perm_inds, _ = utilities.random_permutation_inds(2 * p)
    features = features[:, perm_inds]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(features, y)
    resid = np.power(model.predict(features) - y, 2).sum()
    return resid / max(10, n - np.sum(model.coef_ != 0))

def default_regularization(X, Xk, y):
    """
    Returns the default regularization parameter for the lasso based on 
    the heuristic from https://arxiv.org/pdf/1508.02757.
    
    """
    n, p = X.shape
    return 8 * np.sqrt(compute_residual_variance(X, Xk, y) * np.log(p) / n)


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
    alphas, _, coefs = linear_model.lars_path(
        features,
        y,
        method="lasso",
        **kwargs,
    )

    # Calculate places where features enter the model
    Z = np.zeros(2 * p)
    for i in range(2 * p):
        if (coefs[i] != 0).sum() == 0:
            Z[i] = 0
        else:
            Z[i] = alphas[np.where(coefs[i] != 0)[0][0]]

    return Z[rev_inds]


def fit_lasso(X, Xk, y, y_dist=None, alphas=None, use_lars=False, mx=True, **kwargs):
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
    alphas : float
        The regularization parameter(s) to try. Selects one
        alpha using cross-validation by default.
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
    gl : sklearn Lasso/LassoCV/LassoLarsCV/LogisticRegressionCV object
        The sklearn model fit through cross-validation.
    inds : np.ndarray
        ``(2p,)``-dimensional array of indices representing the random
        permutation applied to the concatenation of [X, Xk] before fitting
        ``gl.``
    rev_inds : np.ndarray:
        Indices which reverse the effect of ``inds.`` In particular, if
        M is any ``(n, 2p)``-dimensional array, then ```M==M[:, inds][:, rev_inds]```
    """
    n, p = X.shape

    # Parse some kwargs/defaults
    max_iter = kwargs.pop("max_iter", 500)
    tol = kwargs.pop("tol", 1e-3)
    cv = kwargs.pop("cv", 5)

    # Default regularization parameter depends on MX vs. FX
    if alphas is None and mx:
        alphas = DEFAULT_REG_VALS
    elif alphas is None:
        alphas = [default_regularization(X, Xk, y)]

    # ensure everything is in the right format
    if isinstance(alphas, float) or isinstance(alphas, int):
        alphas = [alphas]
    if y_dist is None:
        y_dist = parse_y_dist(y)

    # Bind data
    features = np.concatenate([X, Xk], axis=1)

    # Randomize coordinates to make sure everything is symmetric
    inds, rev_inds = utilities.random_permutation_inds(2 * p)
    features = features[:, inds]

    # Fit lasso
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if y_dist == "gaussian":
            if not use_lars and len(alphas) == 1:
                gl = linear_model.Lasso(
                    alpha=alphas[0], max_iter=max_iter, tol=tol, **kwargs
                ).fit(features, y)
            elif not use_lars:
                gl = linear_model.LassoCV(
                    alphas=alphas,
                    cv=cv,
                    verbose=False,
                    max_iter=max_iter,
                    tol=tol,
                    **kwargs,
                ).fit(features, y)
            elif use_lars:
                gl = linear_model.LassoLarsCV(
                    cv=cv,
                    verbose=False,
                    max_iter=max_iter,
                    **kwargs,
                ).fit(features, y)
        elif y_dist == "binomial":
            gl = linear_model.LogisticRegressionCV(
                Cs=1 / np.array(alphas),
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

    return gl, inds, rev_inds


def fit_ridge(X, Xk, y, y_dist=None, mx: bool=True, **kwargs):
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

    # Default regularization parameter depends on MX vs. FX
    if mx:
        alphas = DEFAULT_REG_VALS
    else:
        alphas = [default_regularization(X, Xk, y)]

    # Fit lasso
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if y_dist == "gaussian":
            try:
                ridge = linear_model.RidgeCV(
                    alphas=alphas,
                    store_cv_values=True,
                    scoring="neg_mean_squared_error",
                    **kwargs,
                )
            # compatability with sklearn 1.7.0+
            except TypeError:
                ridge = linear_model.RidgeCV(
                    alphas=alphas,
                    store_cv_results=True,
                    scoring="neg_mean_squared_error",
                    **kwargs,
                )
            ridge.fit(features, y)
        elif y_dist == "binomial":
            ridge = linear_model.LogisticRegressionCV(
                Cs=1 / alphas,
                penalty="l2",
                solver="liblinear",
                **kwargs,
            ).fit(features, y)
        else:
            raise ValueError(f"y_dist must be one of gaussian, binomial, not {y_dist}")

    return ridge, inds, rev_inds


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

                - "swap": The default swap-statistic from http://proceedings.mlr.press/v89/gimenez19a/gimenez19a.pdf.
                - "swapint": The swap-integral from http://proceedings.mlr.press/v89/gimenez19a/gimenez19a.pdf.
            
            Defaults to 'swap'. This is ignored for most specialized classes
            inheriting from FeatureStatistic (e.g. MLR/Lasso Statistics).
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
        X.shape[0]
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
        features.shape[0]
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
        features.shape[0]
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
                    self.model,
                    features,
                    y,
                    cv=5,
                    scoring=scoring,
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
    """
    Wraps the FeatureStatistic class but uses Ridge
    coefficients as variable importances.
    
    Parameters
    ----------
    mx : bool
        If True, the ridge is fit using cross validation. For FX knockoffs,
        this is invalid, and we use the heuristic alpha=8 * np.sqrt(hatsigma2 * np.log(p) / n).
    """

    def __init__(self, mx: bool=True, **kwargs):
        super().__init__(**kwargs)
        self.mx = mx

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
        X.shape[0]
        p = X.shape[1]
        if groups is None:
            groups = np.arange(1, p + 1, 1)

        # Check if y_dist is gaussian, binomial
        y_dist = parse_y_dist(y)
        kwargs["y_dist"] = y_dist

        # Step 1: Calculate Z stats by fitting ridge
        self.model, self.inds, self.rev_inds = fit_ridge(
            X=X,
            Xk=Xk,
            y=y,
            mx=self.mx,
            **kwargs,
        )

        # Retrieve Z statistics and save cv scores
        if y_dist == "gaussian":
            Z = self.model.coef_[self.rev_inds]
            try:
                self.score = -1 * self.model.cv_values_.mean(axis=1).min()
            except AttributeError:
                self.score = np.nan
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
    """
    Wraps the FeatureStatistic class but uses cross-validated Lasso
    coefficients or Lasso path statistics as variable importances.


    Parameters
    ----------
    mx : bool
        If True, the lasso is fit using cross validation. For FX knockoffs,
        this is invalid, and we use the heuristic alpha=8 * np.sqrt(hatsigma2 * np.log(p) / n).
    """

    def __init__(self, mx: bool=True, **kwargs):
        super().__init__(**kwargs)
        self.mx = mx

    def fit(
        self,
        X,
        Xk,
        y,
        groups=None,
        zstat="coef",
        antisym="cd",
        group_agg="avg",
        cv_score=False,
        debias=False,
        Ginv=None,
        **kwargs,
    ):
        """
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
            Extra kwargs to pass to underlying Lasso classes.

        Notes
        -----
        When mx=True, the lasso is fit using cross validation. For FX knockoffs,
        this is invalid, and we use the heuristic 
        alpha=8 * np.sqrt(hatsigma2 * np.log(p) / n).

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
        kwargs["y_dist"] = kwargs.get("y_dist", parse_y_dist(y))

        # Step 1: Calculate Z statistics
        zstat = str(zstat).lower()
        if zstat == "coef":
            # Fit (possibly group) lasso
            gl, inds, rev_inds = fit_lasso(
                X=X,
                Xk=Xk,
                y=y,
                mx=self.mx,
                **kwargs,
            )

            # Parse the expected output format
            logistic_flag = parse_logistic_flag(kwargs)

            # Retrieve Z statistics
            if logistic_flag:
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
                    raise ValueError("To debias the lasso, Ginv must be provided")
                elif logistic_flag:
                    raise ValueError(
                        "Debiased lasso is not implemented for binomial data"
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
    """Marginal correlation statistic"""

    def __init__(self):
        super().__init__()

    def fit(
        self,
        X,
        Xk,
        y,
        groups=None,
        **kwargs,
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
    """Lasso Statistic wrapper class"""

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
                f"Unexpected shape {Z.shape} for sklearn LinearRegression coefs (expected ({2 * p},))"
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
            features=features,
            y=y,
            cv_score=cv_score,
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
            groups = np.arange(1, p + 1)
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

        # Fit model, get Z statistics.
        # Note this does the randomization of features by itself.
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
            using cross validation. This is extremely expensive for deeppink.
        kwargs : dict
            Extra kwargs to pass to underlying deeppink class (in kpytorch)

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

        # The deeppink model class will shuffle statistics,
        # but for compatability we create indices anyway
        self.inds = np.arange(2 * p)
        self.rev_inds = self.inds

        # By default, all variables are their own group
        if groups is None:
            groups = np.arange(1, p + 1)
        self.groups = groups

        # Parse y_dist, hidden_sizes, initialize model
        parse_y_dist(y)
        if "hidden_sizes" not in kwargs:
            kwargs["hidden_sizes"] = [min(n, p)]
        self.model = deeppink.DeepPinkModel(p=p, **kwargs)

        # Train model
        self.model.train()
        self.model = deeppink.train_deeppink(self.model, features, y, **train_kwargs)
        self.model.eval()

        # Get Z statistics
        feature_importance = str(feature_importance).lower()
        if feature_importance == "deeppink":
            self.Z = self.model.feature_importances(weight_scores=True)
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

    # Non-batched result.
    if len(W.shape) == 1:
        # sort by abs values
        absW = np.abs(W)
        inds = np.argsort(-absW, stable="stable")
        negatives = np.cumsum(W[inds] <= 0)
        positives = np.cumsum(W[inds] > 0)
        positives[positives == 0] = 1  # Don't divide by 0
        # calc hat fdrs
        hat_fdrs = (negatives + offset) / positives
        # Maximum threshold such that hat_fdr <= nominal level
        if np.any(hat_fdrs <= fdr):
            T = absW[inds[np.where(hat_fdrs <= fdr)[0].max()]]
            if T == 0:
                T = np.min(W[W > 0])
        else:
            T = np.inf
        return T

    # batch (may be depreciated)
    else:
        return np.array(
            [
                data_dependent_threshhold(W[:, j], fdr=fdr, offset=offset)
                for j in range(W.shape[1])
            ]
        )
