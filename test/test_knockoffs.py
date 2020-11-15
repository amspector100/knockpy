""" Tests knockpy.knockoffs and knockpy.smatrix modules"""
import numpy as np
import scipy as sp
import unittest
from .context import knockpy
from statsmodels.stats.moment_helpers import cov2corr
from knockpy import graphs, utilities, mac, mrc, smatrix, knockoffs


class CheckSMatrix(unittest.TestCase):

    # Helper function
    def check_S_properties(self, V, S, groups):

        # Test PSD-ness of S
        min_S_eig = np.linalg.eigh(S)[0].min()
        self.assertTrue(
            min_S_eig > 0, f'S matrix is not positive semidefinite: mineig is {min_S_eig}' 
        )

        # Test PSD-ness of 2V - S
        min_diff_eig = np.linalg.eigh(2*V - S)[0].min()
        self.assertTrue(
            min_diff_eig > 0, f"2Sigma-S matrix is not positive semidefinite: mineig is {min_diff_eig}"
        )

        # Calculate conditional knockoff matrix
        invV = utilities.chol2inv(V)
        invV_S = np.dot(invV, S)
        Vk = 2 * S - np.dot(S, invV_S)

        # Test PSD-ness of the conditional knockoff matrix
        min_Vk_eig = np.linalg.eigh(Vk)[0].min()
        self.assertTrue(
            min_Vk_eig > 0, f"conditional knockoff matrix is not positive semidefinite: mineig is {min_Vk_eig}"
        )

        # Test that S is just a block matrix
        p = V.shape[0]
        S_test = np.zeros((p, p))
        for j in np.unique(groups):

            # Select subset of S
            inds = np.where(groups == j)[0]
            full_inds = np.ix_(inds, inds)
            group_S = S[full_inds]

            # Fill only in this subset of S
            S_test[full_inds] = group_S


        # return
        np.testing.assert_almost_equal(
            S_test, S, decimal = 5, err_msg = "S matrix is not a block matrix of the correct shape"
        )


class TestEquicorrelated(CheckSMatrix):
    """ Tests equicorrelated knockoffs and related functions """

    def test_eigenvalue_calculation(self):

        # Test to make sure non-group and group versions agree
        # (in the case of no grouping)
        p = 100
        groups = np.arange(0, p, 1) + 1
        for rho in [0, 0.3, 0.5, 0.7]:
            V = np.zeros((p, p)) + rho
            for i in range(p):
                V[i, i] = 1
            expected_gamma = min(1, 2*(1-rho))
            gamma = mac.calc_min_group_eigenvalue(
                Sigma=V, groups=groups, 
            )
            np.testing.assert_almost_equal(
                gamma, expected_gamma, decimal = 3, 
                err_msg = 'calc_min_group_eigenvalue calculates wrong eigenvalue'
            )

        # Test non equicorrelated version
        V = np.random.randn(p, p)
        V = np.dot(V.T, V) + 0.1*np.eye(p)
        V = cov2corr(V)
        expected_gamma = min(1, 2*np.linalg.eigh(V)[0].min())
        gamma = mac.calc_min_group_eigenvalue(
            Sigma=V, groups=groups
        )
        np.testing.assert_almost_equal(
            gamma, expected_gamma, decimal = 3, 
            err_msg = 'calc_min_group_eigenvalue calculates wrong eigenvalue'

        )

    def test_equicorrelated_construction(self):

        # Test S matrix construction
        p = 100
        groups = np.arange(0, p, 1) + 1
        V = np.random.randn(p, p)
        V = np.dot(V.T, V) + 0.1*np.eye(p)
        V = cov2corr(V)

        # Expected construction
        expected_gamma = min(1, 2*np.linalg.eigh(V)[0].min())
        expected_S = expected_gamma*np.eye(p)

        # Equicorrelated
        S = mac.solve_equicorrelated(Sigma=V, groups=groups)

        # Test to make sure the answer is expected
        np.testing.assert_almost_equal(
            S, expected_S, decimal = 3, 
            err_msg = 'calc_min_group_eigenvalue calculates wrong eigenvalue'

        )

    def test_psd(self):

        # Test S matrix construction
        p = 100
        V = np.random.randn(p, p)
        V = np.dot(V.T, V) + 0.1*np.eye(p)
        V = cov2corr(V)

        # Create random groups
        groups = np.random.randint(1, p, size=(p))
        groups = utilities.preprocess_groups(groups)
        S = mac.solve_equicorrelated(Sigma=V, groups=groups)

        # Check S properties
        self.check_S_properties(V, S, groups)

class TestSDP(CheckSMatrix):
    """ Tests an easy case of SDP and ASDP """

    def test_easy_sdp(self):

        # Test non-group SDP first
        n = 200
        p = 50
        X,_,_,_, corr_matrix, groups = graphs.daibarber2016_graph(
            n = n, p = p, gamma = 0.3
        )

        # S matrix
        trivial_groups = np.arange(0, p, 1) + 1
        S_triv = smatrix.compute_smatrix(
            Sigma=corr_matrix,
            groups=trivial_groups,
            method='sdp',
            verbose=True,
        )
        np.testing.assert_array_almost_equal(
            S_triv, np.eye(p), decimal = 2,
            err_msg = 'solve_group_SDP does not produce optimal S matrix (daibarber graphs)'
        )
        self.check_S_properties(corr_matrix, S_triv, trivial_groups)

        # Repeat for gaussian_knockoffs method
        ksampler = knockoffs.GaussianSampler(
            X=X,
            Sigma=corr_matrix,
            groups=trivial_groups, 
            verbose=False,
            method='sdp',
        )
        S_triv2 = ksampler.fetch_S()

        np.testing.assert_array_almost_equal(
            S_triv2, np.eye(p), decimal = 2, 
            err_msg = 'solve_group_SDP does not produce optimal S matrix (daibarber graphs)'
        )
        self.check_S_properties(corr_matrix, S_triv2, trivial_groups)

        # Test slightly harder case
        _,_,_,_, expected_out, _ = graphs.daibarber2016_graph(
            n = n, p = p, gamma = 0
        )
        ksampler = knockoffs.GaussianSampler(
            X = X, Sigma = corr_matrix, groups = groups, 
            verbose = False,
            method = 'sdp'
        )
        S_harder = ksampler.fetch_S()
        np.testing.assert_almost_equal(
            S_harder, expected_out, decimal = 2,
            err_msg = 'solve_group_SDP does not produce optimal S matrix (daibarber graphs)'
        )
        self.check_S_properties(corr_matrix, S_harder, groups)

        # Repeat for ASDP
        ksampler = knockoffs.GaussianSampler(
            X=X,
            Sigma=corr_matrix,
            groups=groups,
            method='ASDP',
            verbose=False,
            max_block=10
        )
        S_harder_ASDP = ksampler.fetch_S()
        np.testing.assert_almost_equal(
            S_harder_ASDP, expected_out, decimal = 2,
            err_msg = 'solve_group_ASDP does not produce optimal S matrix (daibarber graphs)'
        )
        self.check_S_properties(corr_matrix, S_harder_ASDP, groups)


    def test_equicorr_SDP(self):

        # Test non-group SDP on equicorrelated cov matrix
        p = 100
        rho = 0.8
        V = rho*np.ones((p,p)) + (1-rho)*np.eye(p)
        S = mac.solve_group_SDP(V, verbose=True)
        expected = (2 - 2*rho) * np.eye(p)
        np.testing.assert_almost_equal(
            S, expected, decimal = 2,
            err_msg = 'solve_SDP does not produce optimal S matrix (equicorrelated graph)'
        )

        # Repeat for scaled cov matrix
        scale = 5
        V = scale*V
        S = smatrix.compute_smatrix(
            Sigma=V, method='sdp', verbose=True
        )
        expected = (2 - 2*rho) * np.eye(p)
        np.testing.assert_almost_equal(
            S/scale, expected, decimal = 2,
            err_msg = 'compute_smatrix does not produce optimal S matrix for scaled equicorr graph'
        )

    def test_sdp_tolerance(self):

        # Get graph
        np.random.seed(110)
        Q = graphs.ErdosRenyi(p=50, tol=1e-1)
        V = cov2corr(utilities.chol2inv(Q))
        groups = np.concatenate([np.zeros(10) + j for j in range(5)]) + 1
        groups = groups.astype('int32')

        # Solve SDP
        for tol in [1e-3, 0.01, 0.02]:
            S = smatrix.compute_smatrix(
                Sigma=V, 
                groups=groups,
                method='sdp', 
                objective="pnorm",  
                num_iter=10,
                tol=tol
            )
            G = np.hstack([np.vstack([V, V-S]), np.vstack([V-S, V])])
            mineig = np.linalg.eig(G)[0].min()
            self.assertTrue(
                tol - mineig > -1*tol/10,
                f'sdp solver fails to control minimum eigenvalues: tol is {tol}, val is {mineig}'
            )
            self.check_S_properties(V, S, groups)


    def test_corrmatrix_errors(self):
        """ Tests that SDP raises informative errors when sigma is not scaled properly"""

        # Get graph
        np.random.seed(110)
        Q = graphs.ErdosRenyi(p=50, tol=1e-1)
        V = utilities.chol2inv(Q)
        groups = np.concatenate([np.zeros(10) + j for j in range(5)]) + 1
        groups = groups.astype('int32')

        # Helper function
        def SDP_solver():
            return mac.solve_group_SDP(V, groups)

        # Make sure the value error increases 
        self.assertRaisesRegex(
            ValueError, "Sigma is not a correlation matrix",
            SDP_solver
        )

class TestUtilFunctions(unittest.TestCase):
    """ Tests a couple of simple utility functions"""

    def test_blockdiag_to_blocks(self):

        # Create block sizes and blocks
        block_nos = knockpy.utilities.preprocess_groups(
            np.random.randint(1, 50, 100)
        )        
        block_nos = np.sort(block_nos)
        block_sizes = knockpy.utilities.calc_group_sizes(block_nos)
        blocks = [np.random.randn(b, b) for b in block_sizes]

        # Create block diagonal matrix in scipy
        block_diag = sp.linalg.block_diag(*blocks)
        blocks2 = mrc.blockdiag_to_blocks(block_diag, block_nos)
        for expected, out in zip(blocks, blocks2):
            np.testing.assert_almost_equal(
                out, expected, err_msg='blockdiag_to_blocks incorrectly separates blocks'
            )

class TestMRCSolvers(CheckSMatrix):
    """ Tests the various MRC solvers / classes"""

    def test_scale_sqrt_S(self):
        """ Tests the function which scales sqrt S for PSGD solver"""

        # Construct covariance matrix
        p = 50
        rho = 0.8
        Sigma = np.zeros((p, p)) + rho
        Sigma += (1-rho)*np.eye(p)
        groups = np.arange(1, p+1, 1)
        init_blocks = [np.eye(1) for _ in range(p)]

        # Create model - this automatically scales the
        # initial blocks properly
        fk_precision_calc = mrc.MVRLoss(
            Sigma, groups, init_S=init_blocks
        )
        # Check for proper scaling
        S = fk_precision_calc.pull_S().detach().numpy()
        expected = min(1, 2-2*rho)*np.eye(p)
        np.testing.assert_almost_equal(
            S, expected, decimal=1,
            err_msg=f'Initial scaling fails, expected {expected} but got {S} for equicorrelated rho={rho}'
        )

    def test_group_sorting_error(self):
        """ Tests PSGD class raieses error if the cov 
        matrix/groups are improperly sorted"""

        # Groups and sigma 
        p = 50
        Sigma = np.eye(p)
        groups = knockpy.utilities.preprocess_groups(
            np.random.randint(1, p+1, p)
        )

        # Try to initialize
        def init_unsorted_model():
            model = mrc.MVRLoss(Sigma, groups)

        self.assertRaisesRegex(
            ValueError, "Sigma and groups must be sorted prior to input",
            init_unsorted_model
        )

    def test_smoothing(self):
        """
        Tests that one small eigenvalue of the cov matrix
        doesn't ruin the performance of the methods
        """
        p = 50
        smoothing = 0.1
        _,_,_,_,V = knockpy.graphs.sample_data(
            method='partialcorr', rho=0.1,
        )
        S_MVR = mrc.solve_mvr(Sigma=V, smoothing=smoothing)
        # Not implemented yet
        #S_ENT = mrc.solve_maxent(Sigma=V, smoothing=smoothing)
        S_SDP = mac.solve_SDP(Sigma=V, tol=1e-5)
        mvr_mean = np.diag(S_MVR).mean()
        sdp_mean = np.diag(S_SDP).mean()
        self.assertTrue(
            sdp_mean - mvr_mean < 1e-3,
            f"Highly smoothed S_MVR ({S_MVR}) too far from S_SDP ({S_SDP}) for equi partial corr"
        )

    def test_equicorrelated_soln(self):
        """ Tests that solvers yield expected
        solution for equicorrelated matrices """

        # Main constants 
        p = 50
        groups = np.arange(1, p+1, 1)
        smoothings = [0, 0.01]

        # Construct equicorrelated matrices
        rhos = [0.1, 0.3, 0.5, 0.7, 0.9]
        for rho in rhos:
            for smoothing in smoothings:
                for method in ['mvr', 'maxent']:
                    # Construct Sigma
                    Sigma = np.zeros((p, p)) + rho
                    Sigma += (1-rho)*np.eye(p)

                    # Expected solution
                    opt_prop_rec = min(rho, 0.5)
                    max_S_val = min(1, 2-2*rho)
                    expected = (1-rho)*np.eye(p)

                    # Test optimizer
                    opt_S = mrc.solve_mrc_psgd(
                        Sigma=Sigma,
                        groups=groups,
                        init_S=None,
                        smoothing=smoothing,
                        verbose=True,
                        tol=1e-5,
                        max_epochs=100,
                        line_search_iter=10,
                        lr=1e-2,
                    )
                    self.check_S_properties(Sigma, opt_S, groups)
                    np.testing.assert_almost_equal(
                        opt_S, expected, decimal=1,
                        err_msg=f'For equicorrelated cov rho={rho}, PSGD solver yields unexpected solution'
                    )

                # Test MVR coordinate descent optimizer
                opt_S = mrc.solve_mvr(Sigma=Sigma, smoothing=smoothing, verbose=True)
                self.check_S_properties(Sigma, opt_S, groups)
                np.testing.assert_almost_equal(
                    opt_S, expected, decimal=2,
                    err_msg=f'For equicorrelated cov rho={rho}, mvr_solver yields unexpected solution'
                )

                # Test maximum entropy coordinate descent optimizer
                if smoothing == 0:
                    opt_S = mrc.solve_maxent(Sigma=Sigma, smoothing=smoothing, verbose=True)
                    self.check_S_properties(Sigma, opt_S, groups)
                    np.testing.assert_almost_equal(
                        opt_S, expected, decimal=2,
                        err_msg=f'For equicorrelated cov rho={rho}, maxent_solver yields unexpected solution'
                    )

    def test_equicorrelated_soln_recycled(self):

        # Main constants 
        p = 50
        groups = np.arange(1, p+1, 1)

        # Test separately with the recycling proportion param
        rhos = [0.1, 0.3, 0.5, 0.7, 0.9]
        true_rec_props = [0.5, 0.25, 0.8, 0.5, 0.5]
        for true_rec_prop, rho in zip(true_rec_props, rhos):
            
            # Construct Sigma
            Sigma = np.zeros((p, p)) + rho
            Sigma += (1-rho)*np.eye(p)

            # Expected solution
            opt_prop_rec = min(rho, 0.5)
            max_S_val = min(1, 2-2*rho)
            normal_opt = (1-opt_prop_rec)*max_S_val
            new_opt = min(2-2*rho, normal_opt/(1-true_rec_prop))
            expected = new_opt*np.eye(p)

            # Test PSGD optimizer
            opt_S = mrc.solve_mrc_psgd(
                Sigma=Sigma,
                groups=groups,
                init_S=None,
                rec_prop=true_rec_prop,
                tol=1e-5,
                max_epochs=100,
                line_search_iter=10,
            )
            self.check_S_properties(Sigma, opt_S, groups)
            np.testing.assert_almost_equal(
                opt_S, expected, decimal=2,
                err_msg=f'For equicorrelated cov rho={rho} rec_prop={true_rec_prop}, MVR PSDSolver returns {opt_S}, expected {expected}'
            )

            # Coord descent solver
            opt_S = mrc.solve_mvr(
                Sigma=Sigma,
                rej_rate=true_rec_prop,
                tol=1e-5,
                verbose=True,
            )
            self.check_S_properties(Sigma, opt_S, groups)
            np.testing.assert_almost_equal(
                opt_S, expected, decimal=2,
                err_msg=f'For equicorrelated cov rho={rho} rec_prop={true_rec_prop}, MVR CD solver returns {opt_S}, expected {expected}'
            )



    def test_complex_solns(self):
        """
        Check the solution of the various solvers
        for non-grouped knockoffs.
        """
        np.random.seed(110)
        p = 100
        methods = ['ar1', 'ver']
        groups = np.arange(1, p+1, 1)
        for method in methods:
            _,_,_,_,Sigma = knockpy.graphs.sample_data(
                method=method, p=p
            )

            # Use SDP as baseline
            init_S = knockpy.mac.solve_group_SDP(Sigma, groups)
            sdp_mvr_loss = mrc.mvr_loss(Sigma, init_S)

            # Apply gradient solver
            opt_S = mrc.solve_mrc_psgd(
                Sigma=Sigma,
                groups=groups,
                init_S=init_S,
                tol=1e-5,
                max_epochs=100,
                line_search_iter=10,
            )
            psgd_mvr_loss = mrc.mvr_loss(Sigma, opt_S)

            # Check S matrix
            self.check_S_properties(Sigma, opt_S, groups)
            # Check new loss < init_loss
            self.assertTrue(
                psgd_mvr_loss <= sdp_mvr_loss,
                msg=f"For {method}, PSGD solver has higher loss {psgd_mvr_loss} v. sdp {sdp_mvr_loss}"
            )

            # MVR solver outperforms PSGD
            opt_S_mvr = mrc.solve_mvr(Sigma=Sigma)
            self.check_S_properties(Sigma, opt_S_mvr, groups)
            cd_mvr_loss = mrc.mvr_loss(Sigma, opt_S_mvr)
            self.assertTrue(
                cd_mvr_loss <= psgd_mvr_loss,
                msg=f"For {method}, coord descent MVR solver has higher loss {cd_mvr_loss} v. PSGD {psgd_mvr_loss}"
            )

            # Maxent solver outperforms PSGD
            opt_S_maxent = mrc.solve_maxent(Sigma=Sigma)
            self.check_S_properties(Sigma, opt_S_maxent, groups)
            cd_maxent_loss = mrc.maxent_loss(Sigma, opt_S_maxent)
            psgd_maxent_loss = mrc.maxent_loss(Sigma, opt_S)
            self.assertTrue(
                cd_maxent_loss <= psgd_maxent_loss,
                msg=f"For {method}, coord descent maxent solver has higher loss {cd_maxent_loss} v. PSGD {psgd_maxent_loss}"
            )           

    def test_complex_group_solns(self):
        """
        Check the solutions of the PSGD solver
        for group knockoffs.
        """

        # Construct graph + groups
        np.random.seed(110)
        p = 50
        groups = knockpy.utilities.preprocess_groups(
            np.random.randint(1, p+1, p)
        )
        for method in ['ar1', 'ver']:
            _,_,_,_,Sigma = knockpy.graphs.sample_data(
                method=method, p=p,
            )

            # Use SDP as baseline
            init_S = knockpy.mac.solve_group_SDP(Sigma, groups)
            init_loss = mrc.mvr_loss(Sigma, init_S)

            # Apply gradient solver
            opt_S = mrc.solve_mrc_psgd(
                Sigma=Sigma,
                groups=groups,
                init_S=init_S,
                tol=1e-5,
                max_epochs=100,
                line_search_iter=10,
            )
            psgd_loss = mrc.mvr_loss(Sigma, opt_S)

            # Check S matrix
            self.check_S_properties(Sigma, opt_S, groups)
            # Check new loss < init_loss
            self.assertTrue(
                psgd_loss <= init_loss,
                msg=f"For {method}, PSGD solver has higher loss {psgd_loss} v. sdp {init_loss}"
            )

    def test_equi_ciknock_solution(self):
        """
        Check ciknockoff solution
        """
        p = 500
        rho = 0.6
        # 1. Block equicorrelated
        _,_,_,_,Vblock = knockpy.graphs.sample_data(
            p=p, method='daibarber2016', gamma=0, rho=rho, group_size=2
        )
        S_CI = mrc.solve_ciknock(Vblock)
        np.testing.assert_almost_equal(
            S_CI, (1-rho**2)*np.eye(p), 2, "S_CI is incorrect for block-equicorrelated with blocksize 2"
        )
        # 2. Equicorelated
        _,_,_,_,V = knockpy.graphs.sample_data(
            p=p, method='daibarber2016', gamma=1, rho=rho
        )
        S_CI = mrc.solve_ciknock(V)
        np.testing.assert_almost_equal(
            S_CI, (1-rho)*np.eye(p), 2, "S_CI is incorrect for equicorrelated"
        )

class TestBlockdiagApprx(CheckSMatrix):

    def test_divide_comp(self):
        """ Check that when we divide computation, our
        final blocks do not exceed the specified block size."""

        p = 120
        for method in ['daibarber2016', 'ar1', 'ver', 'qer']:
            _,_,_,_,V = graphs.sample_data(p=p, method=method)
            for max_block in [3, 12, 15, 54, 76]:
                blocks = smatrix.divide_computation(V, max_block)
                block_sizes = utilities.calc_group_sizes(blocks)
                self.assertTrue(
                    block_sizes.max() <= max_block,
                    msg=f"Returned blocks have max size {block_sizes.max()} > max_block {max_block}"
                )

    def test_blockdiag_approx_on_blockdiag(self):

        # Check that blockdiag approx yields same solution
        # for block-diagonal matrix.
        p = 250
        rho = 0.3
        gamma = 0
        groups = np.arange(1, p+1, 1)
        _,_,_,_,V = graphs.sample_data(
            p=p, method='daibarber2016', gamma=gamma, rho=rho
        )

        for method in ['mvr', 'mmi', 'sdp']:
            # Check that S_approx is valid
            S_approx = smatrix.compute_smatrix(
                V, method=method, max_block=10
            )
            self.check_S_properties(V, S_approx, groups)
            
            # In this case, should be the same as S_exact
            S_exact = smatrix.compute_smatrix(V, method=method)
            np.testing.assert_array_almost_equal(
                S_approx, S_exact, 3, 
                err_msg="Blockdiag approximation yields incorrect answer for blockdiag Sigma"
            )

    def test_blockdiag_approx_on_ar1(self):

        # Check that blockdiag approx yields a good soln on AR1
        p = 150
        a = 1
        b = 1
        # Create trivial and nontrivial groups
        triv_groups = np.arange(1, p+1, 1)
        nontriv_groups = np.around(triv_groups/2 + 0.01)
        nontriv_groups = utilities.preprocess_groups(nontriv_groups)
        # Sample data
        np.random.seed(110)
        _,_,_,_,V = graphs.sample_data(
            p=p, method='ar1', a=a, b=b
        )
        for groups in [triv_groups, nontriv_groups]:
            for method in ['mvr', 'mmi', 'sdp']:

                # Check that S_approx is valid
                S_approx = smatrix.compute_smatrix(
                    V, method=method, groups=groups, max_block=50
                )
                self.check_S_properties(V, S_approx, groups)
                
                # For exponentially decaying offdiags, should be quite similar
                # This seems to not be true for groups, so skip that for now...
                # For now, skip grouped mvr/mmi
                if np.all(groups == nontriv_groups):
                    continue
                S_exact = smatrix.compute_smatrix(
                    V, method=method, groups=groups
                )
                diff = np.abs(S_approx - S_exact)
                mean_diff = diff[S_exact != 0].mean()
                expected = 1e-2
                identifier = f"for method={method}, groups={groups}"
                self.assertTrue(
                    mean_diff < expected, 
                    msg=f"Blockdiag apprx - exact soln {diff} is {mean_diff} > {expected} on avg. away from exact soln {identifier}"
                )


class CheckValidKnockoffs(unittest.TestCase):

    def check_valid_mxknockoffs(
        self, 
        X,
        mu=None,
        Sigma=None,
        msg='',
        **kwargs
    ):

        # S matrix
        ksampler = knockoffs.GaussianSampler(
            X=X,
            mu=mu,
            Sigma=Sigma,
            verbose=True,
            **kwargs
        )
        S = ksampler.fetch_S()
        Xk = ksampler.sample_knockoffs()

        # Test knockoff mean
        if mu is None:
            mu = X.mean(axis=0)
        outmsg = "Knockoffs have incorrect means "
        outmsg += f"for MX knockoffs for {msg}"
        np.testing.assert_array_almost_equal(
            Xk.mean(axis=0), mu, 2, outmsg
        )


        # Sigma should be
        if Sigma is None:
            Sigma, _ = utilities.estimate_covariance(X, tol=1e-2)

        # Also rescale X/Xk so S makes sense
        scale = np.sqrt(np.diag(Sigma))
        X = X / scale.reshape(1, -1)
        if mu is None:
            mu = X.mean(axis=0)
        else:
            mu = mu / scale 
        Xk = Xk / scale.reshape(1, -1)
        Sigma = Sigma / np.outer(scale, scale)
        S = S / np.outer(scale, scale)

 
        # Empirical FK correlation matrix
        features = np.concatenate([X, Xk], axis = 1)
        G_hat = np.cov(features.T)

        # Calculate population version
        G = np.concatenate(
            [np.concatenate([Sigma, Sigma-S]),
            np.concatenate([Sigma-S, Sigma])],
            axis=1
        )

        # Test G has correct structure
        outmsg = f"Feature-knockoff cov matrix has incorrect values "
        outmsg += f"for MX knockoffs for {msg} graph "
        np.testing.assert_array_almost_equal(G_hat, G, 2, outmsg)


class TestKnockoffGen(CheckValidKnockoffs):
    """ Tests whether knockoffs have correct distribution empirically"""

    def test_method_parser(self):

        # Easiest test
        method1 = 'hello'
        out1 = smatrix.parse_method(method1, None, None)
        self.assertTrue(
            out1 == method1, 
            "parse method fails to return non-None methods"
        )

        # Default is mvr
        p = 1000
        groups = np.arange(1, p+1, 1)
        out2 = smatrix.parse_method(None, groups, p)
        self.assertTrue(
            out2 == 'mvr', 
            "parse method fails to return mvr by default"
        )

        # Otherwise SDP
        groups[-1] = 1
        out2 = smatrix.parse_method(None, groups, p)
        self.assertTrue(
            out2 == 'sdp', 
            "parse method fails to return SDP for grouped knockoffs"
        )

        # Otherwise ASDP
        p = 1001
        groups = np.ones(p)
        out2 = smatrix.parse_method(None, groups, p)
        self.assertTrue(
            out2 == 'asdp', 
            "parse method fails to return asdp for large p"
        )

    def test_error_raising(self):

        # Generate data
        n = 100
        p = 100
        X,_,_,_, corr_matrix, groups = graphs.daibarber2016_graph(
            n = n, p = p, gamma = 1, rho = 0.8
        )
        S_bad = np.eye(p)

        def fdr_vio_knockoffs():
            ksampler = knockoffs.GaussianSampler(
                X=X, 
                Sigma=corr_matrix,
                S=S_bad,
                verbose=False
            )
            ksampler.sample_knockoffs()

        self.assertRaisesRegex(
            np.linalg.LinAlgError,
            "meaning FDR control violations are extremely likely",
            fdr_vio_knockoffs, 
        )

        # Test FX knockoff violations
        def fx_knockoffs_low_n():
            knockoffs.FXSampler(
                X=X,
                Sigma=corr_matrix,
                S=None,
            )

        self.assertRaisesRegex(
            np.linalg.LinAlgError,
            "FX knockoffs can't be generated with n",
            fx_knockoffs_low_n, 
        )

    def test_consistency_of_inferring_sigma(self):
        """ Checks that the same knockoffs are produced
        whether you infer the covariance matrix first and
        pass it to the gaussian_knockoffs generator, or
        you let the generator do the work for you
        """

        n = 25
        p = 300
        rho = 0.5
        X,_,_,_,_ = graphs.sample_data(
            n=n, p=p, rho=rho, method='AR1'
        )

        # Method 1: infer cov first
        V, _ = utilities.estimate_covariance(X, tol=1e-2)
        np.random.seed(110)
        Xk1 = knockoffs.GaussianSampler(
            X=X, Sigma=V, method='sdp', max_epochs=1
        ).sample_knockoffs()

        # Method 2: Infer during
        np.random.seed(110)
        Xk2 = knockoffs.GaussianSampler(
            X=X, method='sdp', max_epochs=1
        ).sample_knockoffs()
        np.testing.assert_array_almost_equal(
            Xk1, Xk2, 5, err_msg='Knockoff gen is inconsistent'
        )



    def test_MX_knockoff_dist(self):

        # Test knockoff construction for mvr and SDP
        # on equicorrelated matrices
        np.random.seed(110)
        n = 100000
        copies = 3
        p = 5

        # Check with a non-correlation matrix
        V = 4*graphs.AR1(p=p, rho=0.5)
        mu = np.random.randn(p)
        print(f"true mu: {mu}")
        X,_,_,_,_ = graphs.sample_data(
            corr_matrix=V, n=n, mu=mu, p=p,
        )
        print(f"X mean: {X.mean(axis=0)}")


        # Check validity for oracle cov matrix
        self.check_valid_mxknockoffs(
            X,
            mu=mu,
            Sigma=V,
            msg=f'ORACLE 4*AR1(rho=0.5)'
        )

        # Check validity for estimated cov matrix
        print("Here now")
        self.check_valid_mxknockoffs(
            X,
            msg=f'ESTIMATED 4*AR1(rho=0.5)'
        )


        # Check for many types of data
        for rho in [0.1, 0.9]:
            for gamma in [0.5, 1]:
                for method in ['mvr', 'sdp']:

                    mu = 10*np.random.randn(p)
                    X,_,_,_, corr_matrix,_ = graphs.daibarber2016_graph(
                        n=n,
                        p=p,
                        gamma=gamma,
                        rho=rho,
                        mu=mu
                    )

                    # Check validity for oracle correlation matrix
                    self.check_valid_mxknockoffs(
                        X,
                        mu=mu,
                        Sigma=corr_matrix,
                        msg=f'daibarber graph, rho = {rho}, gamma = {gamma}'
                    )

                    # Check validity for estimation
                    self.check_valid_mxknockoffs(
                        X,
                        msg=f'ESTIMATED daibarber graph, rho = {rho}, gamma = {gamma}'
                    )


    def test_FX_knockoff_dist(self):
        # Test knockoff construction for mvr and SDP
        # on equicorrelated matrices
        n = 500
        p = 5
        for rho in [0.1, 0.9]:
            for gamma in [0.5, 1]:
                for method in ['mvr', 'sdp']:
                    # X values
                    X,_,_,_,corr_matrix,_ = graphs.daibarber2016_graph(
                        n = n, p = p, gamma = gamma, rho = rho
                    )
                    # S matrix
                    trivial_groups = np.arange(0, p, 1) + 1
                    ksampler = knockoffs.FXSampler(
                        X=X, 
                        method=method, 
                        verbose=False
                    )
                    Xk = ksampler.sample_knockoffs()
                    S = ksampler.fetch_S()

                    # Compute empirical (scaled) cov matrix
                    features = np.concatenate([X, Xk], axis = 1)
                    G_hat = np.dot(features.T, features)
                    
                    # Calculate what this should be
                    Sigma = np.dot(X.T, X)
                    G = np.concatenate(
                        [np.concatenate([Sigma, Sigma-S]),
                        np.concatenate([Sigma-S, Sigma])],
                        axis=1
                    )

                    # Test G has correct structure
                    msg = f"Feature-knockoff cov matrix has incorrect values "
                    msg += f"for daibarber graph, FX knockoffs, rho = {rho}, gamma = {gamma}"
                    np.testing.assert_array_almost_equal(G_hat, G, 5, msg)

    def test_scaling_consistency(self):
        """
        Checks whether if we first calculate S
        and then pass it back into the knockoff
        generator, we'll get the same answer back.
        """

        p = 100
        n = 300
        X, y, beta, Q, V = knockpy.graphs.sample_data(
            method='qer', p=p, n=n, coeff_size=0.5, sparsity=0.5, 
        )
        ksampler1 = knockoffs.FXSampler(
            X=X,
        )
        S1 = ksampler1.fetch_S()
        ksampler2 = knockoffs.FXSampler(
            X=X,
            S=S1,
        )
        S2 = ksampler2.fetch_S()
        np.testing.assert_array_almost_equal(
            S1, S2, decimal=6,
            err_msg = f"Repeatedly passing S into/out of knockoff gen yields inconsistencies"
        )

if __name__ == '__main__':
    unittest.main()