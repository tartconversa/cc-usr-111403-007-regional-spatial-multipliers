"""空间滞后模型分析入口（独立于 app.service 的既有摘要，不改变其语义）。

输出包含：
- 输入身份（input_digest、data_version）与全部可复算设置（权重方案、边界、缩放、排序键、随机种子）；
- frozen（登记口径：完整全集归一化后取子矩阵，不二次归一）与 renorm（敏感性：缩减样本重标化）两套结果；
- 样本边界方案对结论影响的显式比较；
- 拟合诊断、剖面、共线诊断、直接/间接/总影响及联合不确定性；
- 结果可回溯到具体城市与权重（输出城市键序、W 行和、孤立城市清单）。
"""
import json
import math
import sys
from pathlib import Path

from .catalog import load
from . import spatial as sp
from . import sar
from .service import SCHEMAS

DEFAULTS = {'weight_policy': 'row_standardized', 'x_scale': 'none',
            'mc_draws': 2000, 'seed': 20240101, 'moran_permutations': 999}


def _finite_json(value):
    """递归确保输出不含 NaN/Infinity（与 catalog 的 allow_nan=False 口径一致）。"""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_json(v) for v in value]
    return value


def analyze_spatial(payload, weight_policy=DEFAULTS['weight_policy'],
                    x_scale=DEFAULTS['x_scale'], mc_draws=DEFAULTS['mc_draws'],
                    seed=DEFAULTS['seed'], boundaries=('frozen', 'renorm')):
    snapshot = load(payload, SCHEMAS)
    results = {}
    for boundary in boundaries:
        sample = sp.build_sample(snapshot, weight_policy=weight_policy, boundary=boundary)
        if sample['n'] < len(sample['feature_names']) + 3:
            raise ValueError('观测城市数 %d 对自由度不足，无法稳定估计 SAR' % sample['n'])
        design = sp.design_matrix(sample, intercept=True, x_scale=x_scale)
        fit = sar.fit_sar(sample, design, seed=seed, mc_draws=mc_draws)
        results[boundary] = _package(sample, design, fit, weight_policy, x_scale, seed, mc_draws)
    comparison = _compare(results) if len(boundaries) > 1 else None
    return _finite_json({
        'schema_version': 1,
        'input_digest': snapshot.digest,
        'data_version': snapshot.metadata.get('data_version'),
        'settings': {'weight_policy': weight_policy, 'x_scale': x_scale,
                     'mc_draws': mc_draws, 'seed': seed,
                     'boundaries': list(boundaries),
                     'city_order': 'sorted by id (deterministic)'},
        'boundary_policy_note': ('frozen=完整全集行归一后取观测子矩阵，不重新归一、不补零缺失 y；'
                                 'renorm=缩减样本上重新行归一，仅作敏感性对照。'),
        'results': results,
        'boundary_comparison': comparison,
        'causality_note': '结果是给定空间关系 W 下的统计依赖，不证明产业扩散的因果机制。',
    })


def _package(sample, design, fit, weight_policy, x_scale, seed, mc_draws):
    return {
        'sample': {
            'n': sample['n'], 'universe_size': sample['universe_size'],
            'cities': sample['cities'],
            'dropped_cities': sample['dropped'],
            'sample_isolates': sample['sample_isolates'],
            'universe_isolates': sample['universe_isolates'],
            'weight_row_sums': sample['row_sums'],
            'frozen_row_sum_range': [sample['frozen_row_sum_min'], sample['frozen_row_sum_max']],
            'W': sample['W'],
        },
        'design': {'names': design['names'], 'x_scale': x_scale,
                   'means': design['means'], 'sds': design['sds']},
        'fit': fit,
    }


def _compare(results):
    """量化两个边界方案的结论差异（rho、beta、三类影响的绝对/相对变化）。"""
    if 'frozen' not in results or 'renorm' not in results:
        return None
    a, b = results['frozen']['fit'], results['renorm']['fit']

    def rel(new, old):
        denom = max(abs(old), 1e-12)
        return (new - old) / denom

    rows = []
    rows.append({'quantity': 'rho', 'frozen': a['rho'], 'renorm': b['rho'],
                 'abs_diff': b['rho'] - a['rho'], 'rel_diff': rel(b['rho'], a['rho'])})
    for ca, cb in zip(a['coefficients'], b['coefficients']):
        rows.append({'quantity': 'beta:' + ca['term'], 'frozen': ca['estimate'],
                     'renorm': cb['estimate'], 'abs_diff': cb['estimate'] - ca['estimate'],
                     'rel_diff': rel(cb['estimate'], ca['estimate'])})
    for pa, pb in zip(a['effects']['point_estimates'], b['effects']['point_estimates']):
        for kind in ('direct', 'indirect', 'total'):
            rows.append({'quantity': '%s_effect:%s' % (kind, pa['term']),
                         'frozen': pa[kind], 'renorm': pb[kind],
                         'abs_diff': pb[kind] - pa[kind],
                         'rel_diff': rel(pb[kind], pa[kind])})
    rho_sign_change = (a['rho'] > 0) != (b['rho'] > 0)
    # 显著性结论是否翻转（95% CI 是否跨 0）
    sig_flips = []
    for ca, cb in zip(a['coefficients'], b['coefficients']):
        sig_a = ca['ci95'][0] > 0 or ca['ci95'][1] < 0
        sig_b = cb['ci95'][0] > 0 or cb['ci95'][1] < 0
        if sig_a != sig_b:
            sig_flips.append(ca['term'])
    return {'rows': rows, 'rho_sign_changed': rho_sign_change,
            'significance_flips': sig_flips,
            'conclusion_note': ('边界方案改变了 rho 符号或系数显著性，结论对缺资料处理敏感'
                                if rho_sign_change or sig_flips
                                else '两方案下符号与显著性结论一致，结论对该边界选择稳健')}


def main(argv=None):
    argv = sys.argv if argv is None else argv
    if len(argv) < 2:
        raise SystemExit('用法: python -m app.spatial_service <资料JSON> [mc_draws] [seed]')
    payload = json.loads(Path(argv[1]).read_text(encoding='utf-8'))
    mc_draws = int(argv[2]) if len(argv) > 2 else DEFAULTS['mc_draws']
    seed = int(argv[3]) if len(argv) > 3 else DEFAULTS['seed']
    report = analyze_spatial(payload, mc_draws=mc_draws, seed=seed)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
