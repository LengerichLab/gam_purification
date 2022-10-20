import unittest
import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor as ebr
from .ebm import purify_ebm
from gam_purification.utils import calc_density


class TestEBM(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEBM, self).__init__(*args, **kwargs)

    def test_ebm_simple(self):
        X = np.random.uniform(-1, 1, size=(100, 2))
        Y = X[:, 0] * X[:, 1]

        ebm = ebr(interactions=1, n_jobs=1)
        ebm.fit(X, Y)
        ebm_global = ebm.explain_global()
        results_empirical = purify_ebm(
            ebm_global,
            use_density=False,
            dataset_name="toy",
            move_name="uniform",
            X_train=X,
            X_means=np.mean(X, axis=0),
            X_stds=np.std(X, axis=0),
            laplace=0,
            should_transpose=False,
        )
        # Check Mains are centered.
        for pure_main in results_empirical["mains_moved"].values():
            assert (
                np.abs(np.average(pure_main, weights=np.ones_like(pure_main))) < 1e-10
            )

        # Check Pairs are centered.
        pure_mats = results_empirical["pairs_moved"]
        for pure_mat in pure_mats.values():
            densities = np.ones_like(pure_mat)
            for i in range(pure_mat.shape[0]):
                assert (
                    np.abs(np.average(pure_mat[i, :], weights=densities[i, :])) < 1e-10
                )
            for j in range(pure_mat.shape[1]):
                assert (
                    np.abs(np.average(pure_mat[:, j], weights=densities[:, j])) < 1e-10
                )


if __name__ == "__main__":
    unittest.main()
