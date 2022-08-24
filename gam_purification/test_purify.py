import unittest
import numpy as np
from .purify import purify_row, purify_col, calc_col_means, purify

N_ROWS = 10
N_COLS = 15


class TestPurify(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPurify, self).__init__(*args, **kwargs)

    def test_purify_row(self):
        # Test Base Case of all Zeros.
        raw_mat = np.array([np.array([0, 0]), np.array([0, 0])], dtype=np.float64)
        raw_marg = np.array([0, 0], dtype=np.float64)
        densities = np.ones_like(raw_mat)
        pure_mat, pure_marg = purify_row(raw_mat.copy(), raw_marg.copy(), densities, 0)
        assert np.all(np.isclose(pure_mat, raw_mat, atol=1e-10))
        assert np.all(np.isclose(pure_marg, raw_marg, atol=1e-10))

        # Test Random Matrix.
        raw_mat = np.random.uniform(-1, 1, size=(N_ROWS, N_COLS))
        raw_marg = np.random.uniform(-1, 1, size=(N_ROWS, 1))
        densities = np.random.uniform(0, 1, size=raw_mat.shape)
        for i in range(N_ROWS):
            pure_mat, pure_marg = purify_row(
                raw_mat.copy(), raw_marg.copy(), densities, i
            )
            assert np.abs(np.average(pure_mat[i, :], weights=densities[i, :])) < 1e-10
            assert np.all(
                np.isclose(
                    pure_marg[i],
                    raw_marg[i] + np.average(raw_mat[i, :], weights=densities[i, :]),
                    atol=1e-10,
                )
            )

    def test_purify_col(self):
        # Test Base Case of all Zeros.
        raw_mat = np.array([np.array([0, 0]), np.array([0, 0])], dtype=np.float64)
        raw_marg = np.array([0, 0], dtype=np.float64)
        densities = np.ones_like(raw_mat)
        pure_mat, pure_marg = purify_col(raw_mat.copy(), raw_marg.copy(), densities, 0)
        assert np.all(np.isclose(pure_mat, raw_mat, atol=1e-10))
        assert np.all(np.isclose(pure_marg, raw_marg, atol=1e-10))

        # Test Random Matrix.
        raw_mat = np.random.uniform(-1, 1, size=(N_ROWS, N_COLS))
        raw_marg = np.random.uniform(-1, 1, size=(N_COLS, 1))
        densities = np.random.uniform(0, 1, size=raw_mat.shape)
        for j in range(N_COLS):
            pure_mat, pure_marg = purify_col(
                raw_mat.copy(), raw_marg.copy(), densities, j
            )
            assert np.abs(np.average(pure_mat[:, j], weights=densities[:, j])) < 1e-10
            assert np.all(
                np.isclose(
                    pure_marg[j],
                    raw_marg[j] + np.average(raw_mat[:, j], weights=densities[:, j]),
                    atol=1e-10,
                )
            )

    def test_purify(self):
        # Test Base Case of all Zeros.
        raw_mat = np.array([np.array([0, 0]), np.array([0, 0])], dtype=np.float64)
        densities = np.ones_like(raw_mat)
        intercept, pure_marg1, pure_marg2, pure_mat, n_iter = purify(
            raw_mat.copy(), densities=densities, tol=1e-10, randomize=False
        )
        assert np.abs(intercept) < 1e-10
        assert np.all(np.isclose(pure_mat, raw_mat, atol=1e-10))
        assert np.all(np.isclose(pure_marg1, 0, atol=1e-10))
        assert np.all(np.isclose(pure_marg2, 0, atol=1e-10))

        # Test with random matrix and uniform density.
        raw_mat = np.random.uniform(-1, 1, size=(N_ROWS, N_COLS))
        densities = np.ones_like(raw_mat)
        intercept, pure_marg1, pure_marg2, pure_mat, n_iter = purify(
            raw_mat.copy(), densities=densities, tol=1e-10, randomize=False
        )
        for i in range(N_ROWS):
            assert np.abs(np.average(pure_mat[i, :], weights=densities[i, :])) < 1e-10
            assert np.all(
                np.isclose(
                    pure_marg1[i] + intercept,
                    np.average(raw_mat[i, :], weights=densities[i, :]),
                    atol=1e-10,
                )
            )
        for j in range(N_COLS):
            assert np.abs(np.average(pure_mat[:, j], weights=densities[:, j])) < 1e-10
            assert np.all(
                np.isclose(
                    pure_marg2[j] + intercept,
                    np.average(raw_mat[:, j], weights=densities[:, j]),
                    atol=1e-10,
                )
            )
        assert n_iter == 1  # Always takes a single iteration with uniform density.

        # Test with random matrix and random density.
        # Test with random matrix and uniform density.
        raw_mat = np.random.uniform(-1, 1, size=(N_ROWS, N_COLS))
        densities = np.random.uniform(0, 100, size=raw_mat.shape)
        intercept, pure_marg1, pure_marg2, pure_mat, n_iter = purify(
            raw_mat.copy(), densities=densities, tol=1e-10, randomize=False
        )
        for i in range(N_ROWS):
            assert np.abs(np.average(pure_mat[i, :], weights=densities[i, :])) < 1e-10
        for j in range(N_COLS):
            assert np.abs(np.average(pure_mat[:, j], weights=densities[:, j])) < 1e-10
        # Can't easily predict marginal values when non-uniform density.

    def test_purify_randomize(self):
        # Randomize should not change the results
        # Test Base Case of all Zeros.
        raw_mat = np.array([np.array([0, 0]), np.array([0, 0])], dtype=np.float64)
        densities = np.ones_like(raw_mat)
        intercept, pure_marg1, pure_marg2, pure_mat, n_iter = purify(
            raw_mat.copy(), densities=densities, tol=1e-10, randomize=True
        )
        assert np.abs(intercept) < 1e-10
        assert np.all(np.isclose(pure_mat, raw_mat, atol=1e-10))
        assert np.all(np.isclose(pure_marg1, 0, atol=1e-10))
        assert np.all(np.isclose(pure_marg2, 0, atol=1e-10))

        # Test with random matrix and uniform density.
        raw_mat = np.random.uniform(-1, 1, size=(N_ROWS, N_COLS))
        densities = np.ones_like(raw_mat)
        intercept, pure_marg1, pure_marg2, pure_mat, n_iter = purify(
            raw_mat.copy(), densities=densities, tol=1e-10, randomize=True
        )
        for i in range(N_ROWS):
            assert np.abs(np.average(pure_mat[i, :], weights=densities[i, :])) < 1e-10
            assert np.all(
                np.isclose(
                    pure_marg1[i] + intercept,
                    np.average(raw_mat[i, :], weights=densities[i, :]),
                    atol=1e-10,
                )
            )
        for j in range(N_COLS):
            assert np.abs(np.average(pure_mat[:, j], weights=densities[:, j])) < 1e-10
            assert np.all(
                np.isclose(
                    pure_marg2[j] + intercept,
                    np.average(raw_mat[:, j], weights=densities[:, j]),
                    atol=1e-10,
                )
            )

        # Test with random matrix and random density.
        # Test with random matrix and uniform density.
        raw_mat = np.random.uniform(-1, 1, size=(N_ROWS, N_COLS))
        densities = np.random.uniform(0, 100, size=raw_mat.shape)
        intercept, pure_marg1, pure_marg2, pure_mat, n_iter = purify(
            raw_mat.copy(), densities=densities, tol=1e-10, randomize=True
        )
        for i in range(N_ROWS):
            assert np.abs(np.average(pure_mat[i, :], weights=densities[i, :])) < 1e-10
        for j in range(N_COLS):
            assert np.abs(np.average(pure_mat[:, j], weights=densities[:, j])) < 1e-10
        # Can't easily predict marginal values when non-uniform density.

    def test_purify_identifiable(self):
        # Test whether perturbing a model, then purifying it recovers the original model.
        def helper(randomize):
            raw_mat = np.random.uniform(
                -1, 1, size=(N_ROWS, N_COLS)
            ) * np.random.binomial(1, 0.9, size=(N_ROWS, N_COLS))
            densities = np.random.uniform(1, 100, size=(N_ROWS, N_COLS))
            intercept, m1, m2, pure_mat, n_iters = purify(
                raw_mat.copy(), densities=densities, tol=1e-10, randomize=randomize
            )
            m1_perturbed = m1.copy()
            m2_perturbed = m2.copy()
            mat_perturbed = pure_mat.copy()
            for i in range(N_ROWS):
                val = np.random.normal()
                m1_perturbed[i] += val
                mat_perturbed[i, :] -= val
            for j in range(N_COLS):
                val = np.random.normal()
                m2_perturbed[j] += val
                mat_perturbed[:, j] -= val
            intercept2, m12, m22, pure_mat2, n_iters2 = purify(
                mat_perturbed, densities=densities, tol=1e-10, randomize=randomize
            )
            m12 += m1_perturbed
            m22 += m2_perturbed

            # After centering, the purified model should be the same.
            m12 -= np.average(m12, weights=np.sum(densities, axis=1))
            m22 -= np.average(m22, weights=np.sum(densities, axis=0))

            assert np.all(np.isclose(m1, m12, atol=1e-10))
            assert np.all(np.isclose(m2, m22, atol=1e-10))
            assert np.all(np.isclose(pure_mat, pure_mat2, atol=1e-10))

        helper(False)
        helper(True)


if __name__ == "__main__":
    unittest.main()
