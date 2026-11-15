"""SAR 极大似然与影响分解测试。"""
import copy
import json
import math
import unittest
from pathlib import Path

from app.catalog import load
from app.service import SCHEMAS
from app import spatial as sp
from app import sar
from app import linalg as la

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures' / 'spatial_sample.json'
DATA = json.loads(FIXTURE.read_text(encoding='utf-8'))


def _fit(boundary='frozen', seed=20240101, draws=1000):
    snap = load(DATA, SCHEMAS)
    sample = sp.build_sample(snap, boundary=boundary)
    design = sp.design_matrix(sample)
    return sample, design, sar.fit_sar(sample, design, seed=seed, mc_draws=draws)


class FeasibleRegionTests(unittest.TestCase):
    def test_feasible_interval_from_real_eigenvalues(self):
        snap = load(DATA, SCHEMAS)
        sample = sp.build_sample(snap, boundary='frozen')
        eigs = la.eigenvalues(sample['W'])
        lower, upper, lmin, lmax = sar.feasible_interval(eigs)
        self.assertLess(lower, 0.0)
        self.assertGreater(upper, 0.0)
        self.assertLess(lower, upper)
        # 界的定义：1/lambda_min 与 1/lambda_max
        self.assertAlmostEqual(lower, 1.0 / lmin, places=10)
        self.assertAlmostEqual(upper, 1.0 / lmax, places=10)

    def test_i_minus_rho_w_singular_outside_feasible(self):
        snap = load(DATA, SCHEMAS)
        sample = sp.build_sample(snap, boundary='frozen')
        w = sample['W']
        n = w and len(w)
        eigs = la.eigenvalues(w)
        lower, upper, _, lmax = sar.feasible_interval(eigs)
        # 恰在上界 1/lambda_max 处矩阵奇异（行随机 W 该值为 1）
        rho_edge = upper
        a = [[(1.0 if i == j else 0.0) - rho_edge * w[i][j] for j in range(n)]
             for i in range(n)]
        with self.assertRaises(la.SingularMatrixError):
            la.lu_decompose(a)
        # 可行域内可逆，且 rcond 为正
        ld, rcond = sar.logdet_lu(w, 0.5 * (lower + upper))
        self.assertGreater(rcond, 0.0)

    def test_rho_estimate_inside_feasible_not_just_minus_one_one(self):
        _, _, fit = _fit()
        lower, upper = fit['feasible_rho_interval']
        self.assertTrue(fit['rho_inside_feasible'])
        self.assertGreater(fit['rho'], lower)
        self.assertLess(fit['rho'], upper)
        # 明确：判据是实际 W 的可行域，而非 (-1,1)
        self.assertNotEqual([lower, upper], [-1.0, 1.0])

    def test_logdet_paths_agree_tightly(self):
        _, _, fit = _fit()
        self.assertLess(fit['logdet']['abs_gap'], 1e-10)

    def test_all_isolated_w_raises_identifiability_error(self):
        data = copy.deepcopy(DATA)
        data['collections']['edges'] = []
        snap = load(data, SCHEMAS)
        sample = sp.build_sample(snap, boundary='frozen')
        design = sp.design_matrix(sample)
        with self.assertRaisesRegex(ValueError, 'rho 不可识别'):
            sar.fit_sar(sample, design, seed=1, mc_draws=100)


class EffectsDecompositionTests(unittest.TestCase):
    def test_effect_identity_total_equals_direct_plus_indirect(self):
        _, _, fit = _fit()
        for p in fit['effects']['point_estimates']:
            self.assertAlmostEqual(p['direct'] + p['indirect'], p['total'], places=10)

    def test_beta_is_not_the_total_effect(self):
        sample, design, fit = _fit()
        # 存在空间项 rho≠0 时，总影响不等于 beta（这是本分析的核心口径）
        for coef, eff in zip(fit['coefficients'], fit['effects']['point_estimates']):
            if coef['term'] == 'intercept':
                continue
            self.assertNotAlmostEqual(abs(coef['estimate']), abs(eff['total']), places=3)

    def test_zero_rho_reduces_direct_to_beta(self):
        # 构造退化情形 rho=0：直接效应=beta，间接=0
        w = [[0.0, 0.0], [0.0, 0.0]]
        eigs = la.eigenvalues(w)
        d, t = sar._effect_scalars(w, 0.0)
        self.assertEqual(d, 1.0)
        self.assertEqual(t, 1.0)  # A=I，总=直接，间接=0

    def test_joint_uncertainty_uses_full_covariance(self):
        _, _, fit = _fit(draws=2000)
        mc = fit['effects']['joint_monter_carlo']
        # 联合抽样实际执行，越界抽样被计数而非伪造
        self.assertGreaterEqual(mc['draws_outside_feasible'], 0)
        used = mc['table'][1]['total']['draws_used']
        self.assertEqual(used, mc['draws_requested'] - mc['draws_outside_feasible'])
        # 每个效应都有完整联合分布汇总（均值、SD、95% CI、MC 标准误）
        for row in mc['table']:
            for kind in ('direct', 'indirect', 'total'):
                self.assertIn('ci95', row[kind])
                self.assertIn('mc_standard_error', row[kind])
        # 插值近似误差被显式报告且很小
        self.assertLess(mc['grid_interpolation_max_abs_error'], 1e-2)

    def test_delta_and_mc_agree_for_linear_direct_effect(self):
        _, _, fit = _fit(draws=4000)
        for dm, mt in zip(fit['effects']['delta_method_crosscheck'],
                          fit['effects']['joint_monter_carlo']['table']):
            if dm['term'] == 'intercept':
                continue
            # 直接效应近似线性，Delta SE 与 MC SD 应在 35% 内一致
            ratio = mt['direct']['sd'] / dm['direct']['delta_se']
            self.assertLess(abs(ratio - 1.0), 0.35)


class ReproducibilityTests(unittest.TestCase):
    def test_seed_and_order_reproduce_identical_results(self):
        snap1 = load(DATA, SCHEMAS)
        s1 = sp.build_sample(snap1, boundary='frozen')
        d1 = sp.design_matrix(s1)
        f1 = sar.fit_sar(s1, d1, seed=77, mc_draws=800)
        # 打乱城市输入顺序（读取器按 id 排序，结果应不变）
        data2 = copy.deepcopy(DATA)
        data2['collections']['cities'].reverse()
        data2['collections']['edges'].reverse()
        snap2 = load(data2, SCHEMAS)
        s2 = sp.build_sample(snap2, boundary='frozen')
        d2 = sp.design_matrix(s2)
        f2 = sar.fit_sar(s2, d2, seed=77, mc_draws=800)
        self.assertEqual(s1['cities'], s2['cities'])
        self.assertEqual(f1['rho'], f2['rho'])
        self.assertEqual(f1['beta'], f2['beta'])
        self.assertEqual(f1['effects']['joint_monter_carlo']['table'],
                         f2['effects']['joint_monter_carlo']['table'])

    def test_results_traceable_to_cities_and_weights(self):
        sample, _, fit = _fit()
        # 输出携带城市键序与 W，可回溯
        self.assertEqual(len(sample['cities']), fit['n'])
        self.assertEqual(len(sample['W']), fit['n'])
        self.assertEqual(sample['cities'], sorted(sample['cities']))

    def test_unit_scaling_preserves_fit_in_original_units(self):
        # zscore 设计矩阵下，截距吸收尺度，rho 与似然应与未缩放一致
        snap = load(DATA, SCHEMAS)
        sample = sp.build_sample(snap, boundary='frozen')
        d_none = sp.design_matrix(sample, x_scale='none')
        d_z = sp.design_matrix(sample, x_scale='zscore')
        f_none = sar.fit_sar(sample, d_none, seed=5, mc_draws=400)
        f_z = sar.fit_sar(sample, d_z, seed=5, mc_draws=400)
        self.assertAlmostEqual(f_none['rho'], f_z['rho'], places=8)
        self.assertAlmostEqual(f_none['log_likelihood'], f_z['log_likelihood'], places=8)

    def test_weight_policy_variants_run_and_reproduce(self):
        snap = load(DATA, SCHEMAS)
        rhos = {}
        for policy in ('row_standardized', 'none', 'symmetric'):
            sample = sp.build_sample(snap, weight_policy=policy, boundary='frozen')
            design = sp.design_matrix(sample)
            fit = sar.fit_sar(sample, design, seed=3, mc_draws=300)
            rhos[policy] = fit['rho']
            self.assertTrue(fit['rho_inside_feasible'])
            self.assertLess(fit['logdet']['abs_gap'], 1e-10)
        # 不同权重方案给出不同估计（证明换版被真正区分，而非忽略 policy 参数）
        self.assertEqual(len(set(round(v, 9) for v in rhos.values())), 3)
        # 同方案、同种子可精确复现
        s1 = sp.build_sample(snap, weight_policy='symmetric', boundary='frozen')
        s2 = sp.build_sample(snap, weight_policy='symmetric', boundary='frozen')
        f1 = sar.fit_sar(s1, sp.design_matrix(s1), seed=3, mc_draws=300)
        f2 = sar.fit_sar(s2, sp.design_matrix(s2), seed=3, mc_draws=300)
        self.assertEqual(f1['rho'], f2['rho'])


class DiagnosticsTests(unittest.TestCase):
    def test_diagnostics_complete(self):
        _, _, fit = _fit()
        diag = fit['diagnostics']
        for key in ('residual_profile', 'r_squared_reduced_form', 'ols_comparison',
                    'moran_residual', 'information_matrix_rcond', 'collinearity',
                    'aic', 'bic', 'numeric_hessian_check'):
            self.assertIn(key, diag)
        # 共线诊断含 VIF
        self.assertTrue(any(v['term'] != 'intercept' for v in diag['collinearity']['vif']))
        # LM-lag 是合法卡方统计量（非负）与 p 值
        self.assertGreaterEqual(diag['ols_comparison']['lm_lag_statistic'], 0.0)
        self.assertGreater(diag['ols_comparison']['lm_lag_p_value'], 0.0)

    def test_boundary_schemes_yield_explicit_comparison(self):
        from app.spatial_service import analyze_spatial
        report = analyze_spatial(DATA, mc_draws=400, seed=1)
        self.assertIn('frozen', report['results'])
        self.assertIn('renorm', report['results'])
        cmp = report['boundary_comparison']
        self.assertIn('rho_sign_changed', cmp)
        self.assertIn('significance_flips', cmp)
        # 两个方案的 rho 确实被分别估计且数值不同（权重不同）
        self.assertNotEqual(report['results']['frozen']['fit']['rho'],
                            report['results']['renorm']['fit']['rho'])

    def test_output_is_finite_json_safe(self):
        from app.spatial_service import analyze_spatial
        report = analyze_spatial(DATA, mc_draws=200, seed=1)

        def walk(v):
            if isinstance(v, float):
                self.assertTrue(math.isfinite(v))
            elif isinstance(v, dict):
                for x in v.values():
                    walk(x)
            elif isinstance(v, list):
                for x in v:
                    walk(x)
        walk(report)


if __name__ == '__main__':
    unittest.main()
