"""空间权重构建与样本边界处理。

口径（见 docs/data-contract.md）：
- edges: from 受 to 影响，W[i][j] 表示 i 的邻居 j 对 i 的权重；无自边、非负。
- weight_policy=row_standardized：先按完整城市全集行和归一，零行（孤立城市）保留为零。
- 缺资料城市的处理必须由分析显式声明；读取器不隐式重标化，缺失 y 一律不补零。

本模块提供两个显式边界方案，供结论比较：
- frozen：保留完整全集归一化后的行，取观测城市子矩阵，不做任何二次归一（默认口径）。
- renorm：在缩减样本上重新按行和归一（仅作敏感性对照，不是默认行为）。
"""
from . import linalg as la

POLICIES = ('row_standardized', 'none', 'symmetric')
BOUNDARIES = ('frozen', 'renorm')


def _raw_weight_matrix(cities, edges):
    keys = [c['id'] for c in cities]
    pos = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    w = la.zeros(n)
    for e in edges:
        if e['from'] not in pos or e['to'] not in pos:
            raise ValueError('空间边端点未知')
        if e['from'] == e['to']:
            raise ValueError('空间自边无效')
        if e['weight'] < 0:
            raise ValueError('空间边权重为负')
        w[pos[e['from']]][pos[e['to']]] += e['weight']  # 重复边按权重相加
    return keys, w


def _row_standardize(w):
    out = la.copy(w)
    for row in out:
        total = sum(row)
        if total > 0.0:
            for j in range(len(row)):
                row[j] /= total
        # 零行（孤立城市）保留为零
    return out


def _symmetric_normalize(w):
    n = len(w)
    degrees = [sum(row) for row in w]
    out = la.zeros(n)
    for i in range(n):
        for j in range(n):
            if w[i][j] != 0.0:
                if degrees[i] <= 0.0 or degrees[j] <= 0.0:
                    continue  # 孤立节点无法对称归一，保持零连接
                out[i][j] = w[i][j] / (degrees[i] * degrees[j]) ** 0.5
    return out


def build_universe_w(snapshot, weight_policy='row_standardized'):
    """在完整城市全集上构建 W（在任何缺资料处理之前）。"""
    if weight_policy not in POLICIES:
        raise ValueError('未知权重方案')
    cities = sorted(snapshot.rows('cities'), key=lambda r: r['id'])
    edges = snapshot.rows('edges')
    keys, raw = _raw_weight_matrix(cities, edges)
    if weight_policy == 'row_standardized':
        w = _row_standardize(raw)
    elif weight_policy == 'symmetric':
        w = _symmetric_normalize(raw)
    else:
        w = raw
    isolates = [keys[i] for i in range(len(keys)) if sum(raw[i]) == 0.0]
    return {'cities': keys, 'W': w, 'raw_W': raw, 'isolates': isolates,
            'weight_policy': weight_policy}


def _is_observed(city):
    """缺资料判定：status 显式标记或 y/features 为 None。缺失 y 绝不补零。"""
    if city.get('status') != 'observed':
        return False
    if city.get('y') is None:
        return False
    if not isinstance(city.get('features'), dict):
        return False
    return True


def build_sample(snapshot, weight_policy='row_standardized', boundary='frozen'):
    """构建估计样本与对应的 W，显式记录样本边界。

    boundary='frozen'：取完整全集 W 的观测城市子矩阵，不重新归一（登记口径）。
    boundary='renorm'：取原始权重子矩阵后在缩减样本上重新行归一（敏感性方案）。
    两种方案都剔除缺资料城市，都不把缺失 y 补零。
    """
    if boundary not in BOUNDARIES:
        raise ValueError('未知样本边界方案')
    universe = build_universe_w(snapshot, weight_policy)
    all_cities = sorted(snapshot.rows('cities'), key=lambda r: r['id'])
    city_by_id = {c['id']: c for c in all_cities}
    observed_keys = [c['id'] for c in all_cities if _is_observed(c)]
    dropped = []
    for c in all_cities:
        if c['id'] not in set(observed_keys):
            dropped.append({'id': c['id'], 'status': c.get('status'),
                            'reason': 'missing_status_or_data'})
    pos_full = {k: i for i, k in enumerate(universe['cities'])}
    idx = [pos_full[k] for k in observed_keys]
    raw_full = universe['raw_W']
    raw_sub = [[raw_full[i][j] for j in idx] for i in idx]

    if boundary == 'frozen':
        w_full = universe['W']
        w = [[w_full[i][j] for j in idx] for i in idx]
    else:
        # 敏感性方案：缩减样本上重新行归一（对称/none 方案下退化为对应子矩阵处理）
        if weight_policy == 'row_standardized':
            w = _row_standardize(raw_sub)
        elif weight_policy == 'symmetric':
            w = _symmetric_normalize(raw_sub)
        else:
            w = raw_sub

    feature_names = None
    rows_y, rows_x = [], []
    for k in observed_keys:
        city = city_by_id[k]
        names = sorted(city['features'].keys())
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError('城市特征键不一致，无法构建设计矩阵: %s' % k)
        rows_y.append(float(city['y']))
        rows_x.append([float(city['features'][name]) for name in feature_names])

    n = len(observed_keys)
    sample_isolates = [observed_keys[i] for i in range(n) if sum(w[i]) == 0.0]
    # 冻结方案下，被保留城市的行和可能 <1（部分邻居被剔除）——这是必须显式报告的信息
    row_sums = [sum(row) for row in w]
    return {
        'cities': observed_keys,
        'dropped': dropped,
        'n': n,
        'W': w,
        'raw_W_sub': raw_sub,
        'y': rows_y,
        'X_raw': rows_x,
        'feature_names': feature_names,
        'boundary': boundary,
        'weight_policy': weight_policy,
        'universe_size': len(universe['cities']),
        'universe_isolates': universe['isolates'],
        'sample_isolates': sample_isolates,
        'row_sums': row_sums,
        'frozen_row_sum_min': min(row_sums) if row_sums else None,
        'frozen_row_sum_max': max(row_sums) if row_sums else None,
    }


def design_matrix(sample, intercept=True, x_scale='none'):
    """构建设计矩阵 X。x_scale: none/zscore；zscore 同时返回缩放参数以便复原。"""
    n, k0 = len(sample['X_raw']), len(sample['feature_names'])
    cols = list(zip(*sample['X_raw'])) if sample['X_raw'] else []
    means, sds = [], []
    scaled_cols = []
    for col in cols:
        m = la.mean(col)
        sd = la.sample_var(col) ** 0.5
        means.append(m)
        sds.append(sd)
        if x_scale == 'zscore':
            if sd == 0.0:
                raise la.SingularMatrixError('特征零方差，无法标准化: %s' % '?')
            scaled_cols.append([(v - m) / sd for v in col])
        elif x_scale == 'none':
            scaled_cols.append(list(col))
        else:
            raise ValueError('未知缩放方案')
    x = [list(row) for row in zip(*scaled_cols)] if scaled_cols else [[] for _ in range(n)]
    names = list(sample['feature_names'])
    if intercept:
        x = [[1.0] + row for row in x]
        names = ['intercept'] + names
    return {'X': x, 'names': names, 'means': means, 'sds': sds,
            'intercept': intercept, 'x_scale': x_scale}


def morans_i(w, z, permutations=999, seed=12345):
    """Moran's I 及基于条件置换的单侧/双侧参考分布。"""
    import random
    n = len(z)
    m = la.mean(z)
    dz = [v - m for v in z]
    s0 = sum(sum(row) for row in w)
    if s0 == 0.0:
        return {'I': None, 'p_value_two_sided': None, 'permutations': permutations,
                'note': 'W 总权重为零，Moran I 不可定义'}
    num = sum(w[i][j] * dz[i] * dz[j] for i in range(n) for j in range(n))
    den = sum(v * v for v in dz)
    i_obs = (n / s0) * num / den if den != 0.0 else None

    def stat(vals):
        mm = la.mean(vals)
        d = [v - mm for v in vals]
        numer = sum(w[i][j] * d[i] * d[j] for i in range(n) for j in range(n))
        denom = sum(v * v for v in d)
        return (n / s0) * numer / denom if denom != 0.0 else 0.0

    rng = random.Random(seed)
    perm_vals = []
    base = list(z)
    for _ in range(permutations):
        shuffled = base[:]
        rng.shuffle(shuffled)
        perm_vals.append(stat(shuffled))
    if i_obs is None:
        return {'I': None, 'p_value_two_sided': None, 'permutations': permutations}
    ge = sum(1 for v in perm_vals if abs(v) >= abs(i_obs) - 1e-15)
    p = (ge + 1) / (permutations + 1)
    return {'I': i_obs, 'p_value_two_sided': p, 'permutations': permutations,
            'perm_mean': la.mean(perm_vals), 'perm_sd': (la.sample_var(perm_vals) ** 0.5)}
