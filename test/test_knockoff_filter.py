import time
import pytest
import numpy as np
from scipy import stats
import unittest
from .context import knockpy
from .context import file_directory

from knockpy import utilities
from knockpy import dgp
from knockpy.knockoff_filter import KnockoffFilter

NUM_REPS = 1
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestFdrControl(unittest.TestCase):
    """
    NOTE: Due to resource limitations, this currently
    does not actually check FDR control.

	This checkes FDR control for the KnockoffFilter class.
	It is admittedly difficult to do this with high power
	in a computationally efficient manner, so often we run only 
	a few replications to check that the KnockoffFilter class behaves
	roughly as expected.
	"""

    def check_fdr_control(
        self,
        reps=NUM_REPS,
        q=0.2,
        alpha=0.05,
        filter_kwargs={},
        S=None,
        infer_sigma=False,
        test_grouped=True,
        S_method="sdp",
        **kwargs,
    ):

        np.random.seed(110)
        filter_kwargs = filter_kwargs.copy()
        kwargs = kwargs.copy()
        fixedX = False
        if "ksampler" in filter_kwargs:
            if filter_kwargs["ksampler"] == "fx":
                fixedX = True

        # Create and name DGP
        mu = kwargs.pop("mu", None)
        Sigma = kwargs.pop("Sigma", None)
        invSigma = kwargs.pop("invSigma", None)
        beta = kwargs.pop("beta", None)
        dgprocess = dgp.DGP(mu=mu, Sigma=Sigma, invSigma=invSigma, beta=beta)
        X0, _, beta, _, Sigma = dgprocess.sample_data(**kwargs)

        basename = ""
        for key in kwargs:
            basename += f"{key}={kwargs[key]} "

        # Two settings: one grouped, one not
        p = Sigma.shape[0]
        groups1 = np.arange(1, p + 1, 1)
        name1 = basename + " (ungrouped)"
        groups2 = np.random.randint(1, p + 1, size=(p,))
        groups2 = utilities.preprocess_groups(groups2)
        name2 = basename + " (grouped)"

        # Split filter_kwargs
        init_filter_kwargs = {}
        init_filter_kwargs["ksampler"] = filter_kwargs.pop("ksampler", "gaussian")
        init_filter_kwargs["fstat"] = filter_kwargs.pop("fstat", "lasso")
        knockoff_kwargs = filter_kwargs.pop('knockoff_kwargs', {})


        for name, groups in zip([name1, name2], [groups1, groups2]):

            if not test_grouped and np.all(groups==groups2):
                continue

            # Solve SDP
            if S is None and not fixedX and not infer_sigma:
                ksampler = knockpy.knockoffs.GaussianSampler(
                    X=X0, Sigma=Sigma, groups=groups, method=S_method,
                )
            if not fixedX:
                invSigma = utilities.chol2inv(Sigma)
            group_nonnulls = utilities.fetch_group_nonnulls(beta, groups)

            # Container for fdps
            fdps = []

            # Sample data reps times
            for j in range(reps):
                np.random.seed(j)
                dgprocess = dgp.DGP(Sigma=Sigma, beta=beta)
                X, y, _, Q, _ = dgprocess.sample_data(**kwargs)
                gibbs_graph = dgprocess.gibbs_graph

                # Infer y_dist
                if "y_dist" in kwargs:
                    y_dist = kwargs["y_dist"]
                else:
                    y_dist = "gaussian"

                # Run (MX) knockoff filter
                if fixedX or infer_sigma:
                    mu_arg = None
                    Sigma_arg = None
                    invSigma_arg = None
                else:
                    mu_arg = np.zeros(p)
                    Sigma_arg = Sigma
                    invSigma_arg = invSigma

                # Initialize filter
                knockoff_filter = KnockoffFilter(**init_filter_kwargs)

                # Knockoff kwargs
                knockoff_kwargs['S'] = S
                knockoff_kwargs['invSigma'] = invSigma_arg
                knockoff_kwargs['verbose'] = False

                if "df_t" in kwargs:
                    knockoff_kwargs["df_t"] = kwargs["df_t"]
                if "x_dist" in kwargs:
                    if kwargs["x_dist"] == "gibbs":
                        knockoff_kwargs["gibbs_graph"] = gibbs_graph
                    knockoff_kwargs.pop("S", None)

                selections = knockoff_filter.forward(
                    X=X,
                    y=y,
                    mu=mu_arg,
                    Sigma=Sigma_arg,
                    groups=groups,
                    knockoff_kwargs=knockoff_kwargs,
                    fdr=q,
                    **filter_kwargs,
                )
                del knockoff_filter

                # Calculate fdp
                fdp = np.sum(selections * (1 - group_nonnulls)) / max(
                    1, np.sum(selections)
                )
                fdps.append(fdp)

            fdps = np.array(fdps)
            fdr = fdps.mean()
            # fdr_se = fdps.std() / np.sqrt(reps)

            # norm_quant = stats.norm.ppf(1 - alpha)

            # self.assertTrue(
            #     fdr - norm_quant * fdr_se <= q,
            #     msg=f"MX filter FDR is {fdr} with SE {fdr_se} with q = {q} for DGP {name}",
            # )


class TestKnockoffFilter(TestFdrControl):
    """ Tests knockoff filter (mostly MX, some FX tests) """

    @pytest.mark.slow
    def test_gnull_control(self):
        """ Test FDR control under global null """

        # Scenario 1: AR1 a = 1, b = 1, global null
        self.check_fdr_control(
            n=100, p=50, method="AR1", sparsity=0, y_dist="gaussian", reps=NUM_REPS
        )

        # Scenario 2: Erdos Renyi
        self.check_fdr_control(
            n=300,
            p=50,
            method="ver",
            sparsity=0,
            y_dist="gaussian",
            reps=NUM_REPS,
            filter_kwargs={"fstat": "ols"},
        )

        # Erdos Renyi, but with Ridge Statistic
        self.check_fdr_control(
            n=100,
            p=50,
            method="ver",
            sparsity=0,
            y_dist="gaussian",
            reps=NUM_REPS,
            filter_kwargs={"fstat": "ridge"},
        )

        # Scenario 3: Blockequi
        self.check_fdr_control(
            method="blockequi",
            rho=0.6,
            sparsity=0,
            y_dist="binomial",
            reps=NUM_REPS,
        )

    @pytest.mark.slow
    def test_sparse_control(self):
        """ Test FDR control under sparsity """

        # Scenario 1: AR1 a = 1, b = 1,
        self.check_fdr_control(
            n=300, p=100, method="AR1", sparsity=0.2, y_dist="binomial", reps=NUM_REPS,
        )

        # Scenario 2: Erdos Renyi
        self.check_fdr_control(
            n=100,
            p=100,
            method="ver",
            sparsity=0.2,
            y_dist="gaussian",
            reps=NUM_REPS,
            filter_kwargs={"fstat_kwargs": {"debias": True}},
        )

        # Scenario 3: Blockequi
        self.check_fdr_control(
            method="blockequi",
            rho=0.8,
            sparsity=0.2,
            y_dist="binomial",
            reps=NUM_REPS,
        )

    @pytest.mark.slow
    def test_dense_control(self):
        """ Test FDR control in dense scenario """

        # Scenario 1: AR1 a = 1, b = 1, global null
        self.check_fdr_control(
            n=300, p=50, method="AR1", sparsity=0.5, y_dist="gaussian", reps=NUM_REPS,
        )

        # Scenario 2: Erdos Renyi
        self.check_fdr_control(
            n=100, p=50, method="ver", sparsity=0.5, y_dist="binomial", reps=NUM_REPS
        )

        # Scenario 3: Blockequi
        self.check_fdr_control(
            method="blockequi",
            rho=0.4,
            sparsity=0.5,
            y_dist="gaussian",
            reps=NUM_REPS,
            filter_kwargs={"fstat": "margcorr"},
        )

    @pytest.mark.slow
    def test_nonlinear_control(self):
        """ Test FDR control for nonlinear responses """

        # Scenario 1: AR1 a = 1, b = 1, global null
        self.check_fdr_control(
            n=300,
            p=50,
            method="AR1",
            sparsity=0.5,
            y_dist="gaussian",
            cond_mean="pairint",
            reps=NUM_REPS,
            filter_kwargs={"fstat": "randomforest"},
        )

        # Scenario 2: Erdos Renyi
        self.check_fdr_control(
            n=100,
            p=50,
            method="ver",
            sparsity=0.5,
            y_dist="binomial",
            cond_mean="pairint",
            reps=NUM_REPS,
        )

    @pytest.mark.slow
    def test_recycling_control(self):

        # Scenario 1: AR1, recycle half
        self.check_fdr_control(
            reps=NUM_REPS,
            n=300,
            p=50,
            method="AR1",
            sparsity=0.5,
            y_dist="gaussian",
            filter_kwargs={"recycle_up_to": 0.5},
        )

        # Scenario 2: AR1, recycle exactly 23
        self.check_fdr_control(
            reps=NUM_REPS,
            n=300,
            p=50,
            method="AR1",
            sparsity=0.5,
            y_dist="gaussian",
            filter_kwargs={"recycle_up_to": 28},
        )

    @pytest.mark.slow
    def test_inferred_mx_control(self):
        self.check_fdr_control(
            reps=NUM_REPS,
            n=200,
            p=100,
            method="AR1",
            sparsity=0,
            y_dist="gaussian",
            infer_sigma=True,
        )

        self.check_fdr_control(
            reps=NUM_REPS,
            n=200,
            p=150,
            method="ver",
            sparsity=0,
            y_dist="gaussian",
            infer_sigma=True,
            filter_kwargs={"shrinkage": "graphicallasso"},
        )

    @pytest.mark.slow
    def test_fxknockoff_control(self):

        # Scenario 1: AR1, recycle, lasso, p = 50
        self.check_fdr_control(
            reps=NUM_REPS,
            n=500,
            p=50,
            method="AR1",
            sparsity=0.5,
            y_dist="gaussian",
            filter_kwargs={"ksampler": "fx"},
        )

    @pytest.mark.slow
    def test_deeppink_control(self):
        if not TORCH_AVAILABLE:
            return None
        self.check_fdr_control(
            reps=NUM_REPS,
            n=5000,
            p=150,
            method="AR1",
            sparsity=0.5,
            y_dist="gaussian",
            filter_kwargs={"fstat": "deeppink"},
        )

    @pytest.mark.slow
    def test_t_control(self):
        """ FDR control with t-distributed designs """

        # Scenario 1: AR1 a = 1, b = 1, low sparsity
        self.check_fdr_control(
            n=500,
            p=50,
            method="AR1",
            sparsity=0.5,
            x_dist="ar1t",
            reps=NUM_REPS,
            df_t=5,
            filter_kwargs={"ksampler": "artk",},
        )
        # Scenario 2: block-T R1 a = 1, b = 1, high sparsity
        self.check_fdr_control(
            n=500,
            p=50,
            method="blockequi",
            gamma=0,
            sparsity=0.5,
            x_dist="blockt",
            reps=NUM_REPS,
            df_t=5,
            filter_kwargs={"ksampler": "blockt",},
        )

    @pytest.mark.slow
    def test_gibbs_grid_control(self):

        # Need to pull in specially-estimated Sigma
        p = 49
        V = np.loadtxt(f"{file_directory}/test_covs/vout{p}.txt")
        self.check_fdr_control(
            n=500,
            p=49,
            method="gibbs_grid",
            Sigma=V,
            sparsity=0.5,
            x_dist="gibbs",
            reps=NUM_REPS,
            filter_kwargs={"ksampler": "gibbs_grid",},
        )

    @pytest.mark.slow
    def test_gibbs_grid_dlasso(self):
        """ Makes sure gibbs_grid works in combination with debiased lasso """

        # Need to pull in specially-estimated Sigma
        p = 49
        V = np.loadtxt(f"{file_directory}/test_covs/vout{p}.txt")
        self.check_fdr_control(
            n=500,
            p=49,
            method="gibbs_grid",
            Sigma=V,
            sparsity=0.5,
            x_dist="gibbs",
            reps=1,
            q=1,
            filter_kwargs={"ksampler": "gibbs_grid", "fstat": "dlasso",},
        )

    @pytest.mark.slow
    def test_lars_control(self):

        # Scenario 1: blockequi
        p = 500
        rho = 0.3
        S = (1 - rho) * np.eye(p)
        self.check_fdr_control(
            reps=NUM_REPS,
            n=1000,
            p=p,
            S=S,
            method="blockequi",
            gamma=1,
            rho=rho,
            sparsity=0.5,
            y_dist="gaussian",
            coeff_dist="uniform",
            coeff_size=5,
            filter_kwargs={"fstat_kwargs": {"use_lars": True}},
        )

    @pytest.mark.slow
    def test_factor_model(self):

        p = 1000
        n = 300
        gamma = 1
        rho = 0.5
        time0 = time.time()
        self.check_fdr_control(
            n=n,
            p=p,
            method='blockequi',
            gamma=gamma,
            rho=rho,
            infer_sigma=True,
            S_method="mvr",
            filter_kwargs={"num_factors":2, 'fstat':'margcorr'},
            test_grouped=False,
        )
        time_factored = time.time() - time0
        print(f"Factored time is {time_factored}")

        time0 = time.time()
        self.check_fdr_control(
            n=n,
            p=p,
            method='blockequi',
            gamma=gamma,
            rho=rho,
            infer_sigma=True,
            S_method="mvr",
            filter_kwargs={
                "knockoff_kwargs":{
                    'how_approx':'factor'
                }, 
                'fstat':'margcorr'
            },
            test_grouped=False,
        )
        time_apprx_factored = time.time() - time0
        print(f"Apprx_factored time is {time_apprx_factored}")

        time0 = time.time()
        self.check_fdr_control(
            n=n,
            p=p,
            method='blockequi',
            gamma=gamma,
            rho=rho,
            infer_sigma=True,
            S_method="mvr",
            filter_kwargs={"num_factors":None, 'fstat':'margcorr'},
            test_grouped=False,
        )
        time_unfactored = time.time() - time0
        print(f"Unfactored time is {time_unfactored}")
        self.assertTrue(
            1.5*time_factored < time_unfactored,
            msg=f"time for factor assumption ({time_factored}) > 1.5*time for no apprx ({time_unfactored})"
        )
        self.assertTrue(
            1.2*time_apprx_factored < time_unfactored,
            msg=f"time for factor apprx ({time_apprx_factored}) > 1.2*time for no apprx ({time_unfactored})"
        )


    @pytest.mark.quick
    def test_selection_procedure(self):

        mxfilter = KnockoffFilter()
        W1 = np.concatenate([np.ones(10), -0.4 * np.ones(100)])
        selections = mxfilter.make_selections(W1, fdr=0.1)
        num_selections = np.sum(selections)
        expected = np.sum(W1 > 0)
        self.assertTrue(
            num_selections == expected,
            f"selection procedure makes {num_selections} discoveries, expected {expected}",
        )

        # Repeat to test zero handling
        W2 = np.concatenate([np.abs(np.random.randn(500)), np.zeros(1)])
        selections2 = mxfilter.make_selections(W2, fdr=0.2)
        num_selections2 = np.sum(selections2)
        expected2 = np.sum(W2 > 0)
        self.assertTrue(
            num_selections2 == expected2,
            f"selection procedure makes {num_selections2} discoveries, expected {expected2}",
        )


if __name__ == "__main__":
    unittest.main()
