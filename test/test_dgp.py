import numpy as np
import scipy as sp
import unittest
import pytest
from .context import knockpy


from knockpy import dgp


class TestSampleData(unittest.TestCase):
    """ Tests sample_data function """

    def test_logistic(self):

        np.random.seed(110)

        p = 50
        dgprocess = dgp.DGP()
        X, y, beta, Q, Sigma = dgprocess.sample_data(p=p, y_dist="binomial")

        # Test outputs are binary
        y_vals = np.unique(y)
        np.testing.assert_array_almost_equal(
            y_vals,
            np.array([0, 1]),
            err_msg="Binomial flag not producing binary responses",
        )

        # Test conditional mean for a single X val - start by
        # sampling ys
        N = 5000
        X_repeated = np.repeat(X[0], N).reshape(p, N).T
        ys = dgp.sample_response(X_repeated, beta, y_dist="binomial")

        # Then check that the theoretical/empirical mean are the same
        cond_mean = 1 / (1 + np.exp(-1 * np.dot(X_repeated[0], beta)))
        emp_cond_mean = ys.mean(axis=0)
        np.testing.assert_almost_equal(cond_mean, emp_cond_mean, decimal=2)

    def test_beta_gen(self):

        # Test sparsity
        p = 100
        dgprocess = dgp.DGP()
        _, _, beta, _, _ = dgprocess.sample_data(p=p, sparsity=0.3, coeff_size=0.3,)
        self.assertTrue(
            (beta != 0).sum() == 30, msg="sparsity parameter yields incorrect sparsity"
        )
        abs_coefs = np.unique(np.abs(beta[beta != 0]))
        np.testing.assert_array_almost_equal(
            abs_coefs,
            np.array([0.3]),
            err_msg="beta generation yields incorrect coefficients",
        )

        # Test number of selections for groups
        sparsity = 0.2
        groups = np.concatenate([np.arange(0, 50, 1), np.arange(0, 50, 1)])
        dgprocess = dgp.DGP()
        _, _, beta, _, _ = dgprocess.sample_data(p=p, sparsity=sparsity, groups=groups,)

        # First, test that the correct number of features is chosen
        num_groups = np.unique(groups).shape[0]
        expected_nonnull_features = sparsity * p
        self.assertTrue(
            (beta != 0).sum() == expected_nonnull_features,
            msg="sparsity for groups chooses incorrect number of features",
        )

        # Check that the correct number of GROUPS has been chosen
        expected_nonnull_groups = sparsity * num_groups
        selected_groups = np.unique(groups[beta != 0]).shape[0]
        self.assertTrue(
            selected_groups == expected_nonnull_groups,
            msg="group sparsity parameter does not choose coeffs within a group",
        )

    def test_y_response(self):

        # Sample design matrix, beta
        # np.random.seed(100)
        n = 100000
        p = 10
        X = np.random.randn(n, p)
        beta = dgp.create_sparse_coefficients(
            p=p, sparsity=0.5, coeff_size=1, sign_prob=0.5, coeff_dist="none"
        )
        beta[0] = 1
        beta[1] = -1
        beta[2] = 0

        # Test if a feature has the expected marginal covariance w y
        def test_cov(feature, y, name, expected=1):
            ycov = (feature * y).mean()
            var = (feature ** 2).mean()
            coef = ycov / var
            self.assertTrue(
                np.abs(coef - expected) < 0.05,
                msg=f"when sampling y, {name} cond_mean yields unexpected results ({coef} vs {expected})",
            )

        # Cond mean 1: linear.
        y = dgp.sample_response(X, beta, cond_mean="linear")
        test_cov(X[:, 0], y, name="linear")

        # Cond mean 2: cubic
        y = dgp.sample_response(X, beta, cond_mean="cubic")
        feature = np.power(X[:, 0], 3) - X[:, 0]
        test_cov(feature, y, name="cubic")

        # Cond mean 3: trunclinear
        y = dgp.sample_response(X, beta, cond_mean="trunclinear")
        feature = X[:, 0] >= 1
        mean1 = y[feature].mean()
        mean2 = y[~feature].mean()
        self.assertTrue(
            np.abs(mean1 - mean2 - 1) < 0.05,
            msg=f"when sampling y, trunclinear cond_mean yields unexpected results for conditional means {mean1} vs {mean2+1}",
        )

        # Cond mean 4: pairwise interactions
        y = dgp.sample_response(X, beta, cond_mean="pairint")
        feature = X[:, 0] * X[:, 1]
        test_cov(feature, y, name="pairint", expected=-1)

        # Cond mean 5: sin
        y = dgp.sample_response(X, beta, cond_mean="cos")
        feature = X[:, 0]
        test_cov(feature, y, name="cos", expected=0)
        feature = X[:, 2]
        test_cov(feature, y, name="cos", expected=0)

        # Cond mean 6: quadratic
        y = dgp.sample_response(X, beta, cond_mean="quadratic")
        feature = X[:, 0]
        test_cov(feature, y, name="quadratic", expected=0)

    def test_coeff_dist(self):

        # Test normal
        np.random.seed(110)
        p = 1000
        dgprocess = dgp.DGP()
        _, _, beta, _, _ = dgprocess.sample_data(
            p=p, sparsity=1, coeff_size=1, coeff_dist="normal", sign_prob=0
        )
        expected = 0
        mean_est = beta.mean()
        self.assertTrue(
            np.abs(mean_est - expected) < 0.1,
            msg=f"coeff_dist (normal) mean is wrong: expected mean 1 but got mean {mean_est}",
        )

        # Test uniform
        np.random.seed(110)
        p = 1000
        dgprocess = dgp.DGP()
        _, _, beta, _, _ = dgprocess.sample_data(
            p=p, sparsity=1, coeff_size=1, coeff_dist="uniform", sign_prob=0
        )
        expected = 0.75
        mean_est = beta.mean()
        self.assertTrue(
            np.abs(mean_est - expected) < 0.1,
            msg=f"coeff_dist (uniform) mean is wrong: expected mean 1 but got mean {mean_est}",
        )
        maxbeta = np.max(beta)
        self.assertTrue(
            maxbeta <= 1,
            msg=f"coeff_dist (uniform) produces max beta abs of {maxbeta} > 1 for coeff_size = 1",
        )
        minbeta = np.min(beta)
        self.assertTrue(
            minbeta >= 0.5,
            msg=f"coeff_dist (uniform) produces min beta abs of {minbeta} 0.5 for coeff_size = 1",
        )

        # Test Value-Error
        def sample_bad_dist():
            dgprocess = dgp.DGP()
            dgprocess.sample_data(p=100, coeff_dist="bad_dist_arg")

        self.assertRaisesRegex(
            ValueError, "must be 'none', 'normal', or 'uniform'", sample_bad_dist
        )

    def test_beta_sign_prob(self):

        # Test signs of beta
        p = 100
        for sign_prob in [0, 1]:
            dgprocess = dgp.DGP()
            _, _, beta, _, _ = dgprocess.sample_data(
                p=p, sparsity=1, sign_prob=sign_prob, coeff_dist="uniform"
            )
            num_pos = (beta > 0).sum()
            expected = p * (1 - sign_prob)
            self.assertTrue(
                num_pos == expected,
                msg=f"sign_prob ({sign_prob}) fails to correctly control sign of beta",
            )

        # Test for non-iid sampling
        sparsity = 0.1
        for method in ["blockequi", "ar1"]:
            for sign_prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
                dgprocess = dgp.DGP()
                _, _, beta, _, _ = dgprocess.sample_data(
                    p=p,
                    sparsity=sparsity,
                    sign_prob=sign_prob,
                    iid_signs=False,
                    coeff_dist="uniform",
                    method=method,
                )
                num_pos = (beta > 0).sum()
                expected = np.floor(sign_prob * np.floor(sparsity * p))
                self.assertTrue(
                    num_pos == expected,
                    msg=f"For non-iid sampling, num_pos ({num_pos}) != expected ({expected})",
                )

    def test_beta_corr_signals(self):

        # Test signals are grouped together
        p = 4
        sparsity = 0.5
        expected_nn = int(sparsity * p)
        for j in range(10):
            dgprocess = dgp.DGP()
            _, _, beta, _, _ = dgprocess.sample_data(
                p=p, sparsity=0.5, corr_signals=True
            )
            nn_flags = beta != 0
            self.assertTrue(
                nn_flags.sum() == expected_nn,
                f"Corr_signals breaks sparsity (beta = {beta}, should have {expected_nn} non-nulls)",
            )
            first_nonzero = np.where(nn_flags)[0].min()
            self.assertTrue(
                nn_flags[first_nonzero + 1],
                f"Corr_signals does not produce correlated signals (beta = {beta})",
            )

    def test_partialcorr_sample(self):

        p = 50
        rho = 0.99
        dgprocess = dgp.DGP()
        _, _, _, _, V = dgprocess.sample_data(p=p, method="partialcorr", rho=rho)
        diag_diff = np.mean(np.abs(np.diag(V) - 1))
        self.assertTrue(
            diag_diff < 1e-4,
            f"Partial corr Sigma={V} for rho={rho} is not a correlation matrix",
        )
        pairwise_corr = V[0, 1]
        expected = -1 / (p - 1)
        self.assertTrue(
            np.abs(pairwise_corr - expected) < 1e-4,
            f"Partial corr pairwise_corr {pairwise_corr} deviates from expectation {expected} for rho={rho}",
        )

    def test_factor_sample(self):

        p = 50
        rank = 3
        np.random.seed(100)
        dgprocess = dgp.DGP()
        _, _, _, _, V = dgprocess.sample_data(p=p, method="factor", rank=rank)
        diag_diff = np.mean(np.abs(np.diag(V) - 1))
        self.assertTrue(
            diag_diff < 1e-4,
            f"Factor Sigma={V} for rank={rank} is not a correlation matrix",
        )
        mineig = np.linalg.eigh(V)[0].min()

    def test_blockequi_sample(self):

        # Check that defaults are correct - start w cov matrix
        _, _, beta, _, V, _ = dgp.block_equi_graph()

        # Construct expected cov matrix -  this is a different
        # construction than the actual function
        def construct_expected_V(p, groupsize, rho, gamma):

            # Construct groups with rho ingroup correlation
            block = np.zeros((groupsize, groupsize)) + rho
            block += (1 - rho) * np.eye(groupsize)
            blocks = [block for _ in range(int(p / groupsize))]
            expected = sp.linalg.block_diag(*blocks)

            # Add gamma between-group correlations
            expected[expected == 0] = gamma * rho
            return expected

        expected = construct_expected_V(p=1000, groupsize=5, rho=0.5, gamma=0)

        # Test equality with actual one
        np.testing.assert_array_almost_equal(
            V, expected, err_msg="Default blockequi cov matrix is incorrect"
        )

        # Check number of nonzero groups
        groupsize = 5
        nonzero_inds = np.arange(0, 1000, 1)[beta != 0]
        num_nonzero_groups = np.unique(nonzero_inds // 5).shape[0]
        self.assertTrue(
            num_nonzero_groups == 20,
            msg=f"Default blockequi beta has {num_nonzero_groups} nonzero groups, expected 20",
        )

        # Check number of nonzero features
        num_nonzero_features = (beta != 0).sum()
        self.assertTrue(
            num_nonzero_features == 100,
            msg=f"Default blockequi beta has {num_nonzero_features} nonzero features, expected 100",
        )

    def test_dsliu2020_sample(self):

        rho = 0.8
        n = 500
        p = 500
        dgprocess = dgp.DGP()
        _, _, beta, _, _ = dgprocess.sample_data(
            rho=rho,
            gamma=1,
            p=p,
            n=n,
            sparsity=0.1,
            method="blockequi",
            coeff_dist="dsliu2020",
        )
        self.assertTrue(
            (beta != 0).sum() == 50, f"Sparsity constraint for dsliu2020 violated"
        )

        p = 2000
        dgprocess = dgp.DGP()
        _, _, beta, _, _ = dgprocess.sample_data(
            rho=rho,
            gamma=1,
            p=p,
            n=n,
            sparsity=0.025,
            method="blockequi",
            coeff_dist="dsliu2020",
        )
        self.assertTrue(
            (beta != 0).sum() == 50, f"Sparsity constraint for dsliu2020 violated"
        )

    def test_gmliu2019_sample(self):

        n = 300
        p = 1000
        rho = 0.8
        np.random.seed(110)
        dgprocess = dgp.DGP()
        _, _, beta, _, _ = dgprocess.sample_data(
            rho=rho,
            gamma=1,
            p=p,
            n=n,
            sparsity=0.06,
            method="blockequi",
            coeff_dist="gmliu2019",
        )
        self.assertTrue(
            (beta != 0).sum() == 60, f"Sparsity constraint for gmliu2019 violated"
        )

    def test_AR1_sample(self):

        # Check that rho parameter works
        rho = 0.3
        p = 500
        dgprocess = dgp.DGP()
        _, _, _, _, Sigma = dgprocess.sample_data(p=p, method="AR1", rho=rho)
        np.testing.assert_almost_equal(
            np.diag(Sigma, k=1),
            np.array([rho for _ in range(p - 1)]),
            decimal=4,
            err_msg="Rho parameter for AR1 graph sampling fails",
        )

        # Error testing
        def ARsample():
            dgprocess = dgp.DGP()
            dgprocess.sample_data(method="AR1", rho=1.5)

        self.assertRaisesRegex(
            ValueError, "must be a correlation between -1 and 1", ARsample
        )

        # Check that a, b parameters work
        np.random.seed(110)
        a = 100
        b = 100
        dgprocess = dgp.DGP()
        _, _, _, _, Sigma = dgprocess.sample_data(p=500, method="AR1", a=a, b=b)
        mean_rho = np.diag(Sigma, k=1).mean()
        expected = a / (a + b)
        np.testing.assert_almost_equal(
            mean_rho,
            a / (a + b),
            decimal=2,
            err_msg=f"random AR1 gen has unexpected avg rho {mean_rho} vs {expected} ",
        )

        # Check that maxcorr works
        max_corr = 0.8
        dgprocess = dgp.DGP()
        _, _, _, _, Sigma = dgprocess.sample_data(
            p=500, method="AR1", a=10, b=1, max_corr=max_corr
        )
        obs_max_corr = np.max(np.diag(Sigma, 1))
        self.assertTrue(
            max_corr >= obs_max_corr - 1e-5, # float errors
            f"observed max corr {obs_max_corr} > max_corr {max_corr}"
        )

    def test_nested_AR1(self):

        # Check that a, b parameters work
        np.random.seed(110)
        a = 100
        b = 40
        dgprocess = dgp.DGP()
        _, _, _, _, Sigma = dgprocess.sample_data(
            p=500, method="nestedar1", a=a, b=b, nest_size=2, num_nests=1
        )
        mean_rho = np.diag(Sigma, k=1).mean()
        expected = a / (2 * (a + b)) + (a / (a + b)) ** 2 / 2
        np.testing.assert_almost_equal(
            mean_rho,
            expected,
            decimal=2,
            err_msg=f"random nested AR1 gen has unexpected avg rho {mean_rho}, should be ~ {expected} ",
        )

    def test_dot_corr_matrices(self):
        """ Tests wishart and uniform corr matrices """

        d = 1000
        p = 4
        dgprocess = dgp.DGP()
        _, _, _, _, Sigma = dgprocess.sample_data(p=p, d=d, method="wishart")
        np.testing.assert_almost_equal(
            Sigma,
            np.eye(p),
            decimal=1,
            err_msg=f"random Wishart generation {Sigma} unexpectedly deviates from the identity",
        )

        # Repeat for the uniform case
        dgprocess = dgp.DGP()
        _, _, _, _, Sigma = dgprocess.sample_data(p=p, d=d, method="uniformdot")
        expected = 0.25 * np.eye(p) + 0.75 * np.ones((p, p))
        np.testing.assert_almost_equal(
            Sigma,
            expected,
            decimal=1,
            err_msg=f"random unifdot generation {Sigma} unexpectedly deviates from the {expected}",
        )

    @pytest.mark.slow
    def test_dirichlet_matrices(self):
        """ Simple test that ensures there are no errors, we get corr matrix 
		with expected eigenvalues"""

        # Try one with low temp
        p = 2000
        temp = 0.1
        np.random.seed(110)
        dgprocess = dgp.DGP()
        _, _, _, Q, V = dgprocess.sample_data(p=p, temp=temp, method="dirichlet")
        np.testing.assert_almost_equal(
            np.diag(V),
            np.ones(p),
            decimal=6,
            err_msg=f"DirichletCorr generation {V} is not a correlation matrix",
        )
        min_eig = np.linalg.eigh(V)[0].min()
        self.assertTrue(
            min_eig < 0.003,
            msg=f"Minimum eigenvalue of dirichlet {min_eig} should be <=0.001 when temp={temp}",
        )

        # Try 2 with high temp
        temp = 10
        np.random.seed(110)
        dgprocess = dgp.DGP()
        _, _, _, Q, V = dgprocess.sample_data(p=p, temp=temp, method="dirichlet")
        np.testing.assert_almost_equal(
            np.diag(V),
            np.ones(p),
            decimal=6,
            err_msg=f"DirichletCorr generation {V} is not a correlation matrix",
        )
        min_eig = np.linalg.eigh(V)[0].min()
        self.assertTrue(
            min_eig > 0.001,
            msg=f"Minimum eigenvalue of dirichlet {min_eig} should be >=0.001 when temp={temp}",
        )

    def test_trueER_sample(self):
        """ ER sampling following nodewise knockoffs paper """

        # Try er = Q
        p = 500
        delta = 0.5
        np.random.seed(110)
        dgprocess = dgp.DGP()
        _, _, _, Q, V = dgprocess.sample_data(p=p, delta=delta, method="qer")

        prop_nonzero = (np.abs(Q) > 0.001).mean()
        self.assertTrue(
            abs(prop_nonzero - delta) < 0.02,
            "True (Q)ErdosRenyi sampler fails to give correct sparsity",
        )

        mean_val = (Q.sum() - np.diag(Q).sum()) / (p ** 2 - p)
        self.assertTrue(
            abs(mean_val) < 0.1,
            "True (Q)ErdosRenyi sampler fails to give correct mean val",
        )

        # Try er = V
        delta = 0.1
        np.random.seed(110)
        dgprocess = dgp.DGP()
        _, _, _, Q, V = dgprocess.sample_data(p=p, delta=delta, method="ver")
        prop_nonzero = (np.abs(V) > 0.001).mean()
        self.assertTrue(
            abs(prop_nonzero - delta) < 0.02,
            "True (V)ErdosRenyi sampler fails to give correct sparsity",
        )

        mean_val = (V.sum() - np.diag(V).sum()) / (p ** 2 - p)
        self.assertTrue(
            abs(mean_val) < 0.1,
            "True (V)ErdosRenyi sampler fails to give correct mean val",
        )

        # Try er = V with maxcorr
        max_corr = 0.1
        delta = 0.05
        dgprocess = dgp.DGP()
        _, _, _, _, V = dgprocess.sample_data(p=p, delta=delta, method="ver", max_corr=max_corr)
        np.testing.assert_array_almost_equal(
            np.diag(V),
            np.ones(p),
            err_msg=f"After setting max_corr={max_corr}, ER cov is not corr. matrix")
        hV = V - np.diag(np.diag(V)) # zero out diagonals 
        obs_max_corr = np.abs(hV).max()
        self.assertTrue(
            obs_max_corr <= max_corr + 1e-5,
            f"For VER, obs_max_corr {obs_max_corr} >= max_corr {max_corr}"
        )


    def test_tblock_sample(self):

        # Fake data --> we want the right cov matrix
        np.random.seed(110)
        n = 1000000
        p = 4
        df_t = 6
        dgprocess = dgp.DGP()
        X, _, _, Q, V = dgprocess.sample_data(
            n=n,
            p=p,
            method="blockequi",
            gamma=0,
            x_dist="blockt",
            block_size=2,
            df_t=df_t,
        )
        emp_corr = np.cov(X.T)

        # Check empirical covariance matrice
        np.testing.assert_array_almost_equal(
            V,
            emp_corr,
            decimal=2,
            err_msg=f"t-block empirical correlation matrix does not match theoretical one",
        )

    def test_t_sample(self):

        # Check that we get the right covariance matrix
        np.random.seed(110)
        n = 100000
        p = 5
        dgprocess = dgp.DGP()
        X, _, _, Q, V = dgprocess.sample_data(
            n=n, p=p, method="AR1", x_dist="ar1t", df_t=5
        )

        emp_corr = np.corrcoef(X.T)
        np.testing.assert_array_almost_equal(
            V,
            emp_corr,
            decimal=2,
            err_msg=f"ar1t empirical correlation matrix does not match theoretical one",
        )

        # Check that this fails correctly for non-ar1-method
        def non_ar1_t():
            dgprocess = dgp.DGP()
            dgprocess.sample_data(n=n, p=p, method="ver", x_dist="ar1t")

        self.assertRaisesRegex(ValueError, "should equal 'ar1'", non_ar1_t)

    def test_gibbs_sample(self):

        # Check that we get a decent correlation matrix
        # with the right type of Q matrix
        np.random.seed(110)
        n = 150000
        p = 9
        dgprocess = dgp.DGP()
        X,_,_,_,V = dgprocess.sample_data(n=n, p=p, method="ising", x_dist="gibbs",)
        gibbs_graph = dgprocess.gibbs_graph

        # Mean test
        np.testing.assert_almost_equal(
            X.mean(),
            0,
            decimal=1,
            err_msg=f"Ising sampler has unexpected mean (expected 0, got {X.mean()})",
        )
        # Test Q
        expected_edges = 4 * p - (4 * np.sqrt(p))
        num_edges = (gibbs_graph != 0).sum()
        self.assertTrue(
            num_edges == expected_edges,
            f"Gibbs grid dist has unexpected number of edges ({num_edges}, expected {expected_edges})",
        )
        # Check consistency of gibbs_graph when passed in
        dgprocess2 = dgp.DGP(gibbs_graph=gibbs_graph)
        X2,_,_,_,V2 = dgprocess2.sample_data(
            n=n, p=p, x_dist="gibbs", y_dist="binomial"
        )
        gibbs_graph2 = dgprocess2.gibbs_graph
        np.testing.assert_array_almost_equal(
            gibbs_graph,
            gibbs_graph2,
            decimal=5,
            err_msg=f"Gibbs (non-ising) sampler is not consistent when gibbs_graph passed in",
        )
        error = np.abs(V - V2).mean()
        self.assertTrue(
            error < 0.01,
            msg=f"Gibbs emp. covs are inconsistent (error {error} > 0.01) for same gibbs_graph",
        )

        # Test this works without errors for n < p
        dgprocess = dgp.DGP()
        _ = dgprocess.sample_data(n=5, p=p, x_dist="gibbs", y_dist="binomial")

    def test_xdist_error(self):

        # Check that we get an error for wrong dist
        def bad_xdist():
            dgprocess = dgp.DGP()
            dgprocess.sample_data(method="ver", x_dist="t_dist")

        self.assertRaisesRegex(ValueError, "x_dist must be one of", bad_xdist)

class TestGroupings(unittest.TestCase):

    def test_group_creation(self):

        # Random covariance matrix
        np.random.seed(110)
        dgprocess = knockpy.dgp.DGP()
        dgprocess.sample_data(p=100)
        # Check within-group correlations are less than 0.5
        Sigma = dgprocess.Sigma
        #helper_Sigma = dgprocess.Sigma.copy()
        #helper_Sigma = helper_Sigma - 1 * np.diag(np.diag(helper_Sigma))
        for cutoff in [0.1, 0.5, 0.9]:
            groups = knockpy.dgp.create_grouping(Sigma, cutoff=cutoff, method='single')
            for j in np.unique(groups):
                between_group_corrs = Sigma[groups==j][:, groups!=j]
                maxcorr = np.max(np.abs(between_group_corrs))
                self.assertTrue(
                     maxcorr <= cutoff,
                    f"Max between-group corr is {maxcorr} > cutoff {cutoff}"
                )

        groups = knockpy.dgp.create_grouping(Sigma, cutoff=1, method='single')
        np.testing.assert_array_almost_equal(
            np.sort(groups),
            np.arange(1, Sigma.shape[0] + 1),
            err_msg=f"When cutoff=1, groups are not trivial groups"
        )

if __name__ == "__main__":
    unittest.main()
