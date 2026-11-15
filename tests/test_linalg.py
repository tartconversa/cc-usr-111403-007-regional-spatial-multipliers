"""线性代数原语测试：特征值、行列式双路径、求解与正定分解。"""
import math
import random
import unittest

from app import linalg as la


class LinalgTests(unittest.TestCase):
    def test_diagonal_eigenvalues(self):
        d = la.eigenvalues([[2.0, 0, 0], [0, -3.0, 0], [0, 0, 1.5]])
        reals = sorted(r for r, im in d if abs(im) < 1e-9)
        self.assertEqual(reals, [-3.0, 1.5, 2.0])

    def test_symmetric_matches_jacobi(self):
        s = [[4.0, 1.0, 2.0], [1.0, 3.0, 0.5], [2.0, 0.5, 5.0]]
        gen = sorted(r for r, im in la.eigenvalues(s) if abs(im) < 1e-9)
        sym, _ = la.symmetric_eigen(s)
        for a, b in zip(gen, sym):
            self.assertAlmostEqual(a, b, places=7)

    def test_complex_conjugate_pair(self):
        r = 0.5
        root3 = math.sqrt(3.0) / 2.0
        rot = [[r, -root3], [root3, r]]
        ev = la.eigenvalues(rot)
        complex_pairs = [(re, im) for re, im in ev if abs(im) > 1e-9]
        self.assertEqual(len(complex_pairs), 2)
        for re, im in complex_pairs:
            self.assertAlmostEqual(re, 0.5, places=8)
            self.assertAlmostEqual(abs(im), root3, places=8)

    def test_trace_and_det_preserved_random(self):
        random.seed(42)
        for n in (4, 6, 9):
            a = [[random.uniform(-2, 2) for _ in range(n)] for _ in range(n)]
            ev = la.eigenvalues(a)
            self.assertAlmostEqual(sum(r for r, _ in ev),
                                   sum(a[i][i] for i in range(n)), places=6)
            ld_eig = la.log_det_from_eigen(ev)
            lu, piv = la.lu_decompose(a)
            ld_lu, _ = la.log_abs_det_lu(lu, piv)
            self.assertLess(abs(ld_eig - ld_lu), 1e-8)

    def test_logdet_two_paths_for_i_minus_rho_w(self):
        w = [[0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [0.3, 0.7, 0.0]]
        eigs = la.eigenvalues(w)
        for rho in (-0.5, 0.0, 0.3, 0.8):
            ld_eig = 0.0
            for re, im in eigs:
                zre, zim = 1 - rho * re, -rho * im
                ld_eig += 0.5 * math.log(zre * zre + zim * zim)
            am = [[(1.0 if i == j else 0.0) - rho * w[i][j] for j in range(3)]
                  for i in range(3)]
            lu, piv = la.lu_decompose(am)
            ld_lu, _ = la.log_abs_det_lu(lu, piv)
            self.assertLess(abs(ld_eig - ld_lu), 1e-10)

    def test_solve_and_inverse(self):
        random.seed(3)
        n = 7
        a = [[random.uniform(-3, 3) for _ in range(n)] for _ in range(n)]
        b = [random.uniform(-1, 1) for _ in range(n)]
        x = la.solve(a, b)
        resid = math.sqrt(sum((sum(a[i][j] * x[j] for j in range(n)) - b[i]) ** 2
                              for i in range(n)))
        self.assertLess(resid, 1e-9)
        inv = la.inverse(a)
        prod = la.matmul(a, inv)
        self.assertLess(max(abs(prod[i][j] - (1.0 if i == j else 0.0))
                            for i in range(n) for j in range(n)), 1e-9)

    def test_cholesky_spd(self):
        m = [[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]]
        l = la.cholesky(m)
        recon = la.matmul(l, la.transpose(l))
        self.assertLess(max(abs(recon[i][j] - m[i][j]) for i in range(3)
                            for j in range(3)), 1e-12)

    def test_singular_matrix_raises(self):
        singular = [[1.0, 2.0], [2.0, 4.0]]
        with self.assertRaises(la.SingularMatrixError):
            la.lu_decompose(singular)

    def test_quantile_matches_linear_interp(self):
        x = list(range(1, 11))
        self.assertEqual(la.quantile(x, [0.0, 0.5, 1.0]), [1.0, 5.5, 10.0])

    def test_chi2_sf_monotone(self):
        self.assertAlmostEqual(la.chi2_sf(0.0, 2), 1.0)
        self.assertGreater(la.chi2_sf(1.0, 2), la.chi2_sf(5.0, 2))
        self.assertGreater(la.chi2_sf(10.0, 1), 0.0)


if __name__ == '__main__':
    unittest.main()
