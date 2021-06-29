import numpy as np
import unittest
import sklearn.naive_bayes
import sklearn.neural_network
from .context import knockpy

from knockpy import knockoff_stats as kstats
from knockpy import utilities, dgp
from knockpy.knockoff_stats import data_dependent_threshhold

DEFAULT_SAMPLE_KWARGS = {
    "coeff_size": 5,
    "method": "blockequi",
    "gamma": 0,
    "sparsity": 0.5,
}

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class KStatVal(unittest.TestCase):
    def check_kstat_fit(
        self,
        fstat,
        fstat_name,
        fstat_kwargs={},
        min_power=0.8,
        max_l2norm=9,
        seed=110,
        group_features=False,
        **sample_kwargs,
    ):
        """ fstat should be a class instance inheriting from FeatureStatistic """

        # Add defaults to sample kwargs
        if "method" not in sample_kwargs:
            sample_kwargs["method"] = "blockequi"
        if "gamma" not in sample_kwargs:
            sample_kwargs["gamma"] = 1
        if "n" not in sample_kwargs:
            sample_kwargs["n"] = 200
        if "p" not in sample_kwargs:
            sample_kwargs["p"] = 50
        if "rho" not in sample_kwargs:
            sample_kwargs["rho"] = 0.5
        if "y_dist" not in sample_kwargs:
            sample_kwargs["y_dist"] = "gaussian"
        n = sample_kwargs["n"]
        p = sample_kwargs["p"]
        rho = sample_kwargs["rho"]
        y_dist = sample_kwargs["y_dist"]

        # Create data generating process
        np.random.seed(seed)
        dgprocess = dgp.DGP()
        X, y, beta, _, corr_matrix = dgprocess.sample_data(**sample_kwargs)

        # Create groups
        if group_features:
            groups = np.random.randint(1, p + 1, size=(p,))
            groups = utilities.preprocess_groups(groups)
        else:
            groups = np.arange(1, p + 1, 1)

        # Create knockoffs
        ksampler = knockpy.knockoffs.GaussianSampler(
            X=X,
            groups=groups,
            Sigma=corr_matrix,
            verbose=False,
            S=(1 - rho) * np.eye(p),
        )
        Xk = ksampler.sample_knockoffs()
        S = ksampler.fetch_S()

        # Fit and extract coeffs/T
        fstat.fit(
            X, Xk, y, groups=groups, **fstat_kwargs,
        )
        W = fstat.W
        T = data_dependent_threshhold(W, fdr=0.2)

        # Test L2 norm
        m = np.unique(groups).shape[0]
        if m == p:
            pair_W = W
        else:
            pair_W = kstats.combine_Z_stats(fstat.Z, antisym="cd")
        l2norm = np.power(pair_W - np.abs(beta), 2)
        l2norm = l2norm.mean()
        self.assertTrue(
            l2norm < max_l2norm,
            msg=f"{fstat_name} fits {y_dist} data very poorly (l2norm = {l2norm} btwn real {beta} / fitted {pair_W} coeffs)",
        )

        # Test power for non-grouped setting.
        # (For group setting, power will be much lower.)
        selections = (W >= T).astype("float32")
        group_nnulls = utilities.fetch_group_nonnulls(beta, groups)
        power = ((group_nnulls != 0) * selections).sum() / np.sum(group_nnulls != 0)
        fdp = ((group_nnulls == 0) * selections).sum() / max(np.sum(selections), 1)
        self.assertTrue(
            power >= min_power,
            msg=f"Power {power} for {fstat_name} in equicor case (n={n},p={p},rho={rho}, y_dist {y_dist}, grouped={group_features}) should be > {min_power}. W stats are {W}, beta is {beta}",
        )


class TestFeatureStatistics(KStatVal):
    """ Tests fitting of ols, lasso, ridge, margcorr, random forest """

    def test_combine_Z_stats(self):
        """ Tests the combine_Z_stats function """

        # Fake data
        Z = np.array([-1, -2, -1, 0, 1, 0, 0, 0, 0, -1, 0, 0])
        groups = np.array([1, 1, 1, 2, 2, 2])
        W = kstats.combine_Z_stats(Z, groups)
        np.testing.assert_array_almost_equal(
            W,
            np.array([4, 0]),
            decimal=3,
            err_msg="calc_LCD function incorrectly calculates group LCD",
        )

        # Again
        Z2 = np.array([0, 1, 2, 3, -1, -2, -3, -4])
        groups2 = np.array([1, 2, 3, 4])
        W2 = kstats.combine_Z_stats(Z2, groups2, group_agg="avg")
        np.testing.assert_array_almost_equal(
            W2,
            np.array([-1, -1, -1, -1]),
            err_msg="calc_LCD function incorrectly calculates group LCD",
        )

    def test_margcorr_statistic(self):

        # Fake data (p = 5)
        n = 10000
        p = 5
        X = np.random.randn(n, p)
        knockoffs = np.random.randn(n, p)
        groups = np.array([1, 1, 2, 2, 2])

        # Calc y
        beta = np.array([1, 1, 0, 0, 0])
        y = np.dot(X, beta.reshape(-1, 1))

        # Correlations
        margcorr = kstats.MargCorrStatistic()
        W = margcorr.fit(X, knockoffs, y, groups=None)

        self.assertTrue(
            np.abs(W[0] - 1 / np.sqrt(2)) < 0.05,
            msg="marg_corr_diff statistic calculates correlations incorrectly",
        )

    def test_lars_solver_fit(self):
        """ Tests power of lars lasso solver """

        self.check_kstat_fit(
            fstat=kstats.LassoStatistic(),
            fstat_name="LARS solver",
            fstat_kwargs={"use_lars": True},
            n=150,
            p=100,
            rho=0.7,
            sign_prob=0,
            coeff_size=5,
            coeff_dist="uniform",
            sparsity=0.5,
            seed=1,
        )

    def test_lars_path_fit(self):
        """ Tests power of lars path statistic """
        # Get DGP, knockoffs, S matrix

        self.check_kstat_fit(
            fstat=kstats.LassoStatistic(),
            fstat_name="LARS path statistic",
            fstat_kwargs={"zstat": "lars_path", "antisym": "sm"},
            n=300,
            p=100,
            rho=0.7,
            sign_prob=0.5,
            coeff_size=5,
            coeff_dist="uniform",
            sparsity=0.5,
            seed=110,
            min_power=0.8,
            max_l2norm=np.inf,
        )

    def test_ols_fit(self):
        """ Good old OLS """

        self.check_kstat_fit(
            fstat=kstats.OLSStatistic(),
            fstat_name="OLS solver",
            n=150,
            p=50,
            rho=0.2,
            coeff_size=100,
            sparsity=0.5,
            seed=110,
            min_power=0.8,
        )

    def test_pyglm_group_lasso_fit(self):

        pyglm_kwargs = {
            "use_pyglm": True,
            "max_iter": 20,
            "tol": 5e-2,
            "learning_rate": 3,
            "group_lasso": True,
        }
        self.check_kstat_fit(
            fstat=kstats.LassoStatistic(),
            fstat_name="Pyglm solver",
            fstat_kwargs=pyglm_kwargs,
            n=500,
            p=200,
            rho=0.2,
            coeff_size=5,
            sparsity=0.5,
            seed=110,
            min_power=0.5,
            group_features=True,
            max_l2norm=np.inf,
        )

        # Repeat for logistic case
        self.check_kstat_fit(
            fstat=kstats.LassoStatistic(),
            fstat_name="Pyglm solver",
            fstat_kwargs=pyglm_kwargs,
            n=500,
            p=100,
            rho=0.2,
            coeff_size=5,
            sparsity=0.5,
            seed=110,
            min_power=0.5,
            group_features=True,
            y_dist="binomial",
            max_l2norm=np.inf,
        )

    def test_vanilla_group_lasso_fit(self):

        glasso_kwargs = {
            "use_pyglm": False, 
            "group_lasso": True,
        }
        self.check_kstat_fit(
            fstat=kstats.LassoStatistic(),
            fstat_name="Vanilla group lasso solver",
            fstat_kwargs=glasso_kwargs,
            n=500,
            p=200,
            rho=0.2,
            coeff_size=5,
            sparsity=0.5,
            seed=110,
            min_power=0,
            group_features=True,
            max_l2norm=np.inf,
        )

        # Repeat for logistic case
        self.check_kstat_fit(
            fstat=kstats.LassoStatistic(),
            fstat_name="Vanilla group lasso solver",
            fstat_kwargs=glasso_kwargs,
            n=500,
            p=100,
            rho=0.2,
            coeff_size=5,
            sparsity=0.5,
            seed=110,
            min_power=0,
            group_features=True,
            y_dist="binomial",
            max_l2norm=np.inf,
        )

    def test_lasso_fit(self):

        # Lasso fit for Gaussian data
        self.check_kstat_fit(
            fstat=kstats.LassoStatistic(),
            fstat_name="Sklearn lasso",
            n=200,
            p=100,
            rho=0.7,
            coeff_size=5,
            sparsity=0.5,
            seed=110,
            min_power=0.9,
            group_features=False,
            max_l2norm=np.inf,
        )

        # Repeat for grouped features
        self.check_kstat_fit(
            fstat=kstats.LassoStatistic(),
            fstat_name="Sklearn lasso",
            n=200,
            p=100,
            rho=0.7,
            coeff_size=5,
            sparsity=0.5,
            seed=110,
            min_power=0.4,
            group_features=True,
            max_l2norm=np.inf,
        )

        # Repeat for logistic features
        self.check_kstat_fit(
            fstat=kstats.LassoStatistic(),
            fstat_name="Sklearn lasso",
            n=350,
            p=100,
            rho=0.7,
            coeff_size=5,
            sparsity=0.5,
            seed=110,
            min_power=0.8,
            group_features=True,
            max_l2norm=np.inf,
        )

    def test_antisymmetric_fns(self):

        n = 100
        p = 20
        np.random.seed(110)
        dgprocess = dgp.DGP()
        X, y, beta, _, corr_matrix = dgprocess.sample_data(
            n=n, p=p, y_dist="gaussian", coeff_size=100, sign_prob=1
        )
        groups = np.arange(1, p + 1, 1)

        # These are not real, just helpful syntatically
        fake_knockoffs = np.zeros((n, p))

        # Run to make sure there are no errors for
        # different antisyms
        np.random.seed(110)
        lasso_stat = kstats.LassoStatistic()
        lasso_stat.fit(X=X, Xk=fake_knockoffs, y=y, y_dist=None, antisym="cd")
        W_cd = lasso_stat.W
        Z_cd = lasso_stat.Z
        W_cd[np.abs(W_cd) < 10] = 0
        Z_cd[np.abs(Z_cd) < 10] = 0
        np.testing.assert_array_almost_equal(
            W_cd, -1 * Z_cd[0:p], err_msg="antisym CD returns weird W stats"
        )

        # Run to make sure there are no errors for
        # different antisyms
        np.random.seed(110)
        lasso_stat = kstats.LassoStatistic()
        lasso_stat.fit(X=X, Xk=fake_knockoffs, y=y, y_dist=None, antisym="sm")
        Z_sm = lasso_stat.Z
        W_sm = lasso_stat.W
        np.testing.assert_array_almost_equal(
            W_sm,
            np.abs(Z_sm[0:p]),
            decimal=3,
            err_msg="antisym SM returns weird W stats",
        )

        # Run to make sure there are no errors for
        # different antisyms
        np.random.seed(110)
        lasso_stat = kstats.LassoStatistic()
        lasso_stat.fit(X=X, Xk=fake_knockoffs, y=y, y_dist=None, antisym="scd")
        W_scd = lasso_stat.W
        Z_scd = lasso_stat.Z
        W_scd[np.abs(W_scd) < 10] = 0
        Z_scd[np.abs(Z_scd) < 10] = 0
        np.testing.assert_array_almost_equal(
            W_scd, Z_scd[0:p], err_msg="antisym SCD returns weird W stats"
        )

    def test_cv_scoring(self):

        # Create data generating process
        n = 100
        p = 20
        np.random.seed(110)
        dgprocess = dgp.DGP()
        X, y, beta, _, corr_matrix = dgprocess.sample_data(
            n=n, p=p, y_dist="gaussian", coeff_size=100, sign_prob=1
        )
        groups = np.arange(1, p + 1, 1)

        # These are not real, just helpful syntatically
        knockoffs = np.zeros((n, p))

        # 1. Test lars cv scoring
        lars_stat = kstats.LassoStatistic()
        lars_stat.fit(
            X, knockoffs, y, use_lars=True, cv_score=True,
        )
        self.assertTrue(
            lars_stat.score_type == "mse_cv",
            msg=f"cv_score=True fails to create cross-validated scoring for lars (score_type={lars_stat.score_type})",
        )

        # 2. Test OLS cv scoring
        ols_stat = kstats.OLSStatistic()
        ols_stat.fit(
            X, knockoffs, y, cv_score=True,
        )
        self.assertTrue(
            ols_stat.score_type == "mse_cv",
            msg=f"cv_score=True fails to create cross-validated scoring for lars (score_type={lars_stat.score_type})",
        )
        self.assertTrue(
            ols_stat.score < 2,
            msg=f"cv scoring fails for ols_stat as cv_score={ols_stat.score} >= 2",
        )

        # 3. Test that throws correct error for non-sklearn backend
        def non_sklearn_backend_cvscore():
            dgprocess = dgp.DGP()
            X, y, beta, _, corr_matrix = dgprocess.sample_data(
                n=n, p=p, y_dist="binomial", coeff_size=100, sign_prob=1
            )
            groups = np.random.randint(1, p + 1, size=(p,))
            group = utilities.preprocess_groups(groups)
            pyglm_logit = kstats.LassoStatistic()
            pyglm_logit.fit(
                X,
                knockoffs,
                y,
                use_pyglm=True,
                group_lasso=True,
                groups=groups,
                cv_score=True,
            )

        self.assertRaisesRegex(
            ValueError, "must be sklearn estimator", non_sklearn_backend_cvscore
        )

    def test_debiased_lasso(self):

        # Create data generating process
        n = 200
        p = 20
        rho = 0.3
        np.random.seed(110)
        dgprocess = dgp.DGP()
        X, y, beta, _, corr_matrix = dgprocess.sample_data(
            n=n,
            p=p,
            y_dist="gaussian",
            coeff_size=100,
            sign_prob=0.5,
            method="blockequi",
            rho=rho,
        )
        groups = np.arange(1, p + 1, 1)

        # Create knockoffs
        S = (1 - rho) * np.eye(p)
        ksampler = knockpy.knockoffs.GaussianSampler(
            X=X, groups=groups, Sigma=corr_matrix, verbose=False, S=S
        )
        knockoffs = ksampler.sample_knockoffs()
        G = np.concatenate(
            [
                np.concatenate([corr_matrix, corr_matrix - S]),
                np.concatenate([corr_matrix - S, corr_matrix]),
            ],
            axis=1,
        )
        Ginv = utilities.chol2inv(G)

        # Debiased lasso - test accuracy
        dlasso_stat = kstats.LassoStatistic()
        dlasso_stat.fit(
            X, knockoffs, y, use_lars=False, cv_score=False, debias=True, Ginv=Ginv
        )
        W = dlasso_stat.W
        l2norm = np.power(W - beta, 2).mean()
        self.assertTrue(
            l2norm > 1,
            msg=f"Debiased lasso fits gauissan very poorly (l2norm = {l2norm} btwn real/fitted coeffs)",
        )

        # Test that this throws the correct errors
        # first for Ginv
        def debiased_lasso_sans_Ginv():
            dlasso_stat.fit(
                X, knockoffs, y, use_lars=False, cv_score=False, debias=True, Ginv=None
            )

        self.assertRaisesRegex(
            ValueError, "Ginv must be provided", debiased_lasso_sans_Ginv
        )

        # Second for logistic data
        y = np.random.binomial(1, 0.5, n)

        def binomial_debiased_lasso():
            dlasso_stat.fit(
                X, knockoffs, y, use_lars=False, cv_score=False, debias=True, Ginv=Ginv,
            )

        self.assertRaisesRegex(
            ValueError,
            "Debiased lasso is not implemented for binomial data",
            binomial_debiased_lasso,
        )

    def test_ridge_fit(self):

        # Ridge fit for Gaussian data
        self.check_kstat_fit(
            fstat=kstats.RidgeStatistic(),
            fstat_name="Sklearn ridge",
            n=160,
            p=100,
            rho=0.7,
            coeff_size=5,
            sparsity=0.5,
            seed=110,
            min_power=0.9,
            group_features=False,
        )

        # Repeat for grouped features
        self.check_kstat_fit(
            fstat=kstats.RidgeStatistic(),
            fstat_name="Sklearn ridge",
            n=160,
            p=100,
            rho=0.7,
            coeff_size=5,
            sparsity=0.5,
            seed=110,
            min_power=0.9,
            group_features=True,
            max_l2norm=np.inf,
        )

        # Repeat for logistic features
        self.check_kstat_fit(
            fstat=kstats.RidgeStatistic(),
            fstat_name="Sklearn ridge",
            n=150,
            p=100,
            rho=0.7,
            coeff_size=5,
            sparsity=0.5,
            seed=110,
            min_power=0.85,
            group_features=False,
            max_l2norm=9,
        )

    def test_randomforest_fit(self):

        # RF power on trunclinear data
        self.check_kstat_fit(
            fstat=kstats.RandomForestStatistic(),
            fstat_name="Random forest regression",
            n=2000,
            p=100,
            rho=0.5,
            coeff_size=10,
            sparsity=0.2,
            seed=110,
            min_power=0.5,
            group_features=False,
            cond_mean="trunclinear",
            max_l2norm=np.inf,  # L2 norm makes no sense here
        )

        # Repeat for logistic features
        self.check_kstat_fit(
            fstat=kstats.RandomForestStatistic(),
            fstat_name="Random forest classification",
            n=10000,
            p=100,
            rho=0.5,
            coeff_size=10,
            sparsity=0.2,
            seed=110,
            min_power=0.5,
            group_features=False,
            y_dist="binomial",
            cond_mean="trunclinear",
            max_l2norm=np.inf,  # L2 norm makes no sense here
        )

        # Repeat for pairwise interactions
        self.check_kstat_fit(
            fstat=kstats.RandomForestStatistic(),
            fstat_name="Random forest classification",
            n=4000,
            p=100,
            rho=0.5,
            coeff_size=1,
            sparsity=0.2,
            seed=110,
            gamma=0,
            method="blockequi",
            min_power=0.5,
            group_features=False,
            cond_mean="pairint",
            max_l2norm=np.inf,  # L2 norm makes no sense here
        )

    def test_randomforest_feature_importances(self):
        """ Just makes sure the other efature importance measures don't error """

        # Check that these feature importance scores throw
        # no errors
        self.check_kstat_fit(
            fstat=kstats.RandomForestStatistic(),
            fstat_name="Random forest regression",
            fstat_kwargs={"feature_importance": "default"},
            n=50,
            p=10,
            sparsity=1,
            min_power=0,
            max_l2norm=np.inf,
        )
        self.check_kstat_fit(
            fstat=kstats.RandomForestStatistic(),
            fstat_name="Random forest regression",
            fstat_kwargs={"feature_importance": "swapint"},
            n=50,
            p=10,
            sparsity=1,
            min_power=0,
            max_l2norm=np.inf,
        )

        # Check that correct error is thrown for bad
        # feature importance score
        def bad_feature_importance_type():
            self.check_kstat_fit(
                fstat=kstats.RandomForestStatistic(),
                fstat_name="Random forest regression",
                fstat_kwargs={"feature_importance": "undefined"},
                n=50,
                p=10,
                sparsity=1,
                min_power=0,
                max_l2norm=np.inf,
            )

        self.assertRaisesRegex(
            ValueError,
            "feature_importance undefined must be one of",
            bad_feature_importance_type,
        )

    def test_deeppink_fit(self):

        if not TORCH_AVAILABLE:
            return None

        # RF power on trunclinear data
        self.check_kstat_fit(
            fstat=kstats.DeepPinkStatistic(),
            fstat_name="Deeppink regression",
            n=1000,
            p=100,
            rho=0.5,
            coeff_size=5,
            sparsity=0.2,
            seed=110,
            min_power=0.5,
            group_features=False,
            cond_mean="cubic",
            max_l2norm=np.inf,  # L2 norm makes no sense here
        )

        # Repeat for logistic features
        self.check_kstat_fit(
            fstat=kstats.DeepPinkStatistic(),
            fstat_name="Deeppink classification",
            n=4000,
            p=100,
            rho=0.5,
            gamma=1,
            coeff_size=5,
            sparsity=0.2,
            seed=110,
            min_power=0.2,
            group_features=False,
            y_dist="binomial",
            cond_mean="cubic",
            max_l2norm=np.inf,  # L2 norm makes no sense here
        )

    def test_deeppink_feature_importances(self):

        if not TORCH_AVAILABLE:
            return None

        # Check that these feature importance scores throw
        # no errors
        self.check_kstat_fit(
            fstat=kstats.DeepPinkStatistic(),
            fstat_name="Deep pink regression",
            fstat_kwargs={"feature_importance": "unweighted"},
            n=50,
            p=10,
            sparsity=1,
            min_power=0,
            y_dist="binomial",
            max_l2norm=np.inf,
        )
        self.check_kstat_fit(
            fstat=kstats.DeepPinkStatistic(),
            fstat_name="Deep pink classification",
            fstat_kwargs={"feature_importance": "swap"},
            n=50,
            p=10,
            sparsity=1,
            min_power=0,
            y_dist="binomial",
            max_l2norm=np.inf,
        )
        self.check_kstat_fit(
            fstat=kstats.DeepPinkStatistic(),
            fstat_name="Deep pink classification",
            fstat_kwargs={"feature_importance": "swapint"},
            n=50,
            p=5,
            sparsity=1,
            min_power=0,
            y_dist="binomial",
            max_l2norm=np.inf,
        )

        # Check that correct error is thrown for bad
        # feature importance score
        def bad_feature_importance_type():
            self.check_kstat_fit(
                fstat=kstats.DeepPinkStatistic(),
                fstat_name="Deep pink regression",
                fstat_kwargs={"feature_importance": "undefined"},
                n=50,
                p=10,
                sparsity=1,
                min_power=0,
                max_l2norm=np.inf,
            )

        self.assertRaisesRegex(
            ValueError,
            "feature_importance undefined must be one of",
            bad_feature_importance_type,
        )


class TestBaseFeatureStatistic(KStatVal):
    """ Tests performance of Vanilla feature statistic """

    def test_errors(self):

        fstat = kstats.FeatureStatistic()

    def test_nbayes(self):
        """ Checks that a random sklearn class works with feature stat"""

        gnb = sklearn.naive_bayes.GaussianNB()
        for feature_imp in ["swap", "swapint"]:
            self.check_kstat_fit(
                fstat=kstats.FeatureStatistic(model=gnb),
                fstat_name="Gaussian Naive Bayes",
                fstat_kwargs={"feature_importance": feature_imp},
                n=500,
                p=20,
                sparsity=0.5,
                min_power=0.5,
                max_l2norm=np.inf,
                y_dist="binomial",
                method="blockequi",
                rho=0.2,
                gamma=0,
            )

    def test_mlp(self):
        """ Checks that MLP regressor works """

        mlp = sklearn.neural_network.MLPRegressor(
            solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(10, 2),
        )
        self.check_kstat_fit(
            fstat=kstats.FeatureStatistic(model=mlp),
            fstat_name="Sklearn multilayer perceptron",
            fstat_kwargs={"feature_importance": "swap"},
            n=300,
            p=20,
            sparsity=0.5,
            min_power=0.5,
            max_l2norm=np.inf,
            y_dist="gaussian",
            method="blockequi",
            rho=0.2,
            gamma=0,
        )


class TestDataThreshhold(unittest.TestCase):
    """ Tests data-dependent threshhold """

    def test_unbatched(self):

        W1 = np.array([1, -2, 3, 6, 3, -2, 1, 2, 5, 3, 0.5, 1, 1, 1, 1, 1, 1, 1])
        T1 = data_dependent_threshhold(W1, fdr=0.2)
        expected = np.abs(W1).min()
        self.assertTrue(
            T1 == expected,
            msg=f"Incorrect data dependent threshhold: T1 should be 0, not {T1}",
        )

        W2 = np.array([-1, -2, -3])
        T2 = data_dependent_threshhold(W2, fdr=0.3)
        self.assertTrue(
            T2 == np.inf,
            msg=f"Incorrect data dependent threshhold: T2 should be inf, not {T2}",
        )

        W3 = np.array([-5, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        T3 = data_dependent_threshhold(W3, fdr=0.2)
        self.assertTrue(
            T3 == 5,
            msg=f"Incorrect data dependent threshhold: T3 should be 5, not {T3}",
        )

    def test_batched(self):

        W1 = np.array([1] * 10)
        W2 = np.array([-2, -1, 1, 2, 3, 4, 5, 6, 7, 8])
        W3 = np.array([-1] * 10)
        combined = np.stack([W1, W2, W3]).transpose()
        Ts = data_dependent_threshhold(combined, fdr=0.2)
        expected = np.array([1, 2, np.inf])
        np.testing.assert_array_almost_equal(
            Ts,
            expected,
            err_msg=f"Incorrect data dependent threshhold (batched): Ts should be {expected}, not {Ts}",
        )

    def test_zero_handling(self):
        """ Makes sure Ts != 0 """

        W1 = np.array([1] * 10 + [0] * 10)
        W2 = np.array([-2, -1, 1, 2, 3, 4, 5, 6, 7, 8] + [0] * 10)
        W3 = np.array([-1] * 10 + [0] * 10)
        combined = np.stack([W1, W2, W3]).transpose()
        Ts = data_dependent_threshhold(combined, fdr=0.2)
        expected = np.array([1, 2, np.inf])
        np.testing.assert_array_almost_equal(
            Ts,
            expected,
            err_msg=f"Incorrect data dependent threshhold (batched): Ts should be {expected}, not {Ts}",
        )


if __name__ == "__main__":
    unittest.main()
