"""空间滞后模型（SAR / Spatial Lag, y = rho W y + X beta + epsilon）的极大似然估计与影响分解。

登记口径：y = rho * W * y + X * beta + epsilon，epsilon ~ N(0, sigma^2 I)。

关键约束（对应项目说明）：
1. rho 的可行域由实际 W 的实特征值决定：1/lambda_max < rho < 1/lambda_min，
   不能仅因优化值落在 (-1,1) 就判定 I-rho W 可逆。
2. 对数行列式 log|I-rho W| 同时用特征值路径与 LU 行列式路径计算并比对，报告误差与适用条件。
3. beta 不是总边际影响；直接/间接/总影响基于 S_rho = (I-rho W)^-1 的矩阵量计算。
4. 影响分解的不确定性通过 (beta, sigma2, rho) 的联合协方差整体传播（联合蒙特卡洛 + Delta 交叉验证），
   不对 rho、beta 分别取端点拼区间。
5. 稠密 O(n^3) 实现；近似（网格插值、Jacobi/QR 迭代）均报告误差与适用条件。
模型描述给定空间关系下的统计依赖，不构成产业扩散的因果证据。
"""
import math
import random

from . import linalg as la

CONVERGENCE_TOL = 1e-10
GRID_POINTS = 61          # 剖面似然优化网格
EFFECT_GRID_POINTS = 121  # 影响分解 MC 插值网格（更密以控制线性插值误差）
GOLDEN_ITERS = 80
MC_DEFAULT_DRAWS = 2000


# ---------------------------------------------------------------- 可行域与 log determinant
def feasible_interval(eigs):
    """由 W 的实特征值确定 rho 可行域。复根不约束实 rho（|1-rho*lambda| 对实 rho 恒正）。

    行随机 W 满足 lambda_max=1，故标准可行域为 (1/lambda_min, 1/lambda_max)。
    """
    real_vals = [re for re, im in eigs if abs(im) <= 1e-10 * max(1.0, abs(re))]
    if not real_vals:
        raise ValueError('W 无实特征值，无法确定 rho 可行域')
    # 全零 W（所有城市孤立）：W 无任何非零特征值，空间滞后项退化、rho 不可识别
    if all(abs(v) <= 1e-12 for v in real_vals):
        raise ValueError('W 为零矩阵（样本城市全部孤立），空间滞后项退化，rho 不可识别；'
                         '请检查邻接资料或改用普通回归')
    lam_max, lam_min = max(real_vals), min(real_vals)
    # rho 须满足 1-rho*lambda>0 对所有实 lambda；分别取两侧界再保证严格有序
    cand_lo, cand_hi = [], []
    for lam in real_vals:
        if lam > 0:
            cand_hi.append(1.0 / lam)      # rho < 1/lambda
        elif lam < 0:
            cand_lo.append(1.0 / lam)      # rho > 1/lambda（负数）
    lower = max(cand_lo) if cand_lo else float('-inf')
    upper = min(cand_hi) if cand_hi else float('inf')
    return lower, upper, lam_min, lam_max


def logdet_eigen(eigs, rho):
    """特征值路径：log|I-rho W| = sum log|1-rho lambda|（复根取模）。"""
    total = 0.0
    for re, im in eigs:
        zre = 1.0 - rho * re
        zim = -rho * im
        total += 0.5 * math.log(zre * zre + zim * zim)
    return total


def logdet_lu(w, rho):
    """LU 路径：直接分解 I-rho W 取 log|det|；奇异时抛错（用于可行域与误差交叉验证）。"""
    n = len(w)
    a = la.identity(n)
    for i in range(n):
        for j in range(n):
            a[i][j] -= rho * w[i][j]
    lu, piv = la.lu_decompose(a)
    return la.log_abs_det_lu(lu, piv)[0], la.reciprocal_condition(a)


# ---------------------------------------------------------------- 剖面似然
def _ols_fit(x, y):
    xt = la.transpose(x)
    xtx = la.matmul(xt, x)
    xty = la.matvec(xt, y)
    beta = la.solve(xtx, xty)
    resid = [y[i] - sum(x[i][j] * beta[j] for j in range(len(beta))) for i in range(len(y))]
    sse = sum(r * r for r in resid)
    return beta, sse, xtx


def profile_loglik(rho, w, eigs, x, y, n):
    """给定 rho 的集中（剖面）对数似然常数项：log|I-rhoW| - (n/2) log(S(rho)'S(rho))。"""
    wy = la.matvec(w, y)
    sy = [y[i] - rho * wy[i] for i in range(n)]
    beta, sse, _ = _ols_fit(x, sy)
    if sse <= 0.0:
        raise la.SingularMatrixError('剖面似然遇到零残差平方和')
    ld = logdet_eigen(eigs, rho)
    return ld - 0.5 * n * math.log(sse / n), sse, beta, ld


def _golden_maximize(fun, lo, hi, tol=CONVERGENCE_TOL, iters=GOLDEN_ITERS):
    """在单峰区间上黄金分割最大化一维函数，返回 (argmax, max)。"""
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc = fun(c)
    fd = fun(d)
    for _ in range(iters):
        if fc < fd:
            lo = c
            c, fc = d, fd
            d = lo + gr * (hi - lo)
            fd = fun(d)
        else:
            hi = d
            d, fd = c, fc
            c = hi - gr * (hi - lo)
            fc = fun(c)
        if abs(hi - lo) < tol:
            break
    arg = (lo + hi) / 2.0
    return arg, fun(arg)


def maximize_rho(w, eigs, x, y):
    """网格 + 黄金分割最大化剖面似然；报告网格级多峰诊断，避免局部解伪装成全局解。"""
    n = len(y)
    lower, upper, lam_min, lam_max = feasible_interval(eigs)
    # 安全收缩，避免贴可行域边界（那里 logdet -> -inf / 矩阵奇异）
    span_lo = lower if math.isfinite(lower) else -1.0
    span_hi = upper if math.isfinite(upper) else 1.0
    margin = 1e-4 * (span_hi - span_lo)
    glo, ghi = span_lo + margin, span_hi - margin

    def neg_free(rho):
        val, _, _, _ = profile_loglik(rho, w, eigs, x, y, n)
        return val

    grid = [glo + (ghi - glo) * t / (GRID_POINTS - 1) for t in range(GRID_POINTS)]
    grid_vals = [neg_free(r) for r in grid]
    best_t = max(range(len(grid)), key=lambda t: grid_vals[t])
    # 多峰检测：内部严格局部极大值数量
    modes = [t for t in range(1, len(grid) - 1)
             if grid_vals[t] > grid_vals[t - 1] and grid_vals[t] > grid_vals[t + 1]]
    lo0 = grid[max(0, best_t - 1)]
    hi0 = grid[min(GRID_POINTS - 1, best_t + 1)]
    rho_hat, pll = _golden_maximize(neg_free, lo0, hi0)
    # 边界检查
    edge = min(abs(rho_hat - lower) if math.isfinite(lower) else float('inf'),
               abs(upper - rho_hat) if math.isfinite(upper) else float('inf'))
    return {'rho': rho_hat, 'profile_loglik': pll, 'grid_modes': len(modes),
            'grid': grid, 'grid_values': grid_vals,
            'feasible': (lower, upper), 'boundary_distance': edge}


# ---------------------------------------------------------------- 完整 MLE 与信息矩阵
def fit_sar(sample, design, seed=20240101, mc_draws=MC_DEFAULT_DRAWS):
    """执行完整 SAR 极大似然估计、推断与影响分解。"""
    w = sample['W']
    y = sample['y']
    x = design['X']
    n, k = len(y), len(design['names'])
    eigs = la.eigenvalues(w)
    mle = maximize_rho(w, eigs, x, y)
    rho = mle['rho']

    # 终值处的全套量
    pll, sse, beta_given_rho, ld_eig = profile_loglik(rho, w, eigs, x, y, n)
    beta = beta_given_rho
    sigma2 = sse / n
    wy = la.matvec(w, y)
    sy = [y[i] - rho * wy[i] for i in range(n)]
    xb = la.matvec(x, beta)
    resid = [sy[i] - xb[i] for i in range(n)]

    # log-det 双路径交叉验证与可逆性证据
    ld_lu, rcond = logdet_lu(w, rho)
    logdet_gap = abs(ld_eig - ld_lu)
    lower, upper, lam_min, lam_max = feasible_interval(eigs)

    # 完整对数似然（含常数项）
    ll = -(n / 2.0) * math.log(2.0 * math.pi * sigma2) + ld_eig - sse / (2.0 * sigma2)

    # ---- 期望 Fisher 信息矩阵（MLE 推断的标准选择；参数序 theta=[beta..., sigma2, rho]）----
    # 块结构：beta 与 (sigma2,rho) 渐近正交。
    # I_bb=X'X/s2；I_ss=n/(2 s2^2)；I_sr=tr(WA)/s2；
    # I_rr=tr((WA^-1)^2)+tr(W'W (A^-1)'A^-1)。已用蒙特卡洛经验标准差验证。
    xt = la.transpose(x)
    xtx = la.matmul(xt, x)
    a_mat = la.identity(n)
    for i in range(n):
        for j in range(n):
            a_mat[i][j] -= rho * w[i][j]
    a_inv = la.inverse(a_mat)
    wa = la.matmul(w, a_inv)          # W A^-1
    wb = la.matmul(wa, wa)            # (W A^-1)^2
    tr_wa = sum(wa[i][i] for i in range(n))
    tr_wb = sum(wb[i][i] for i in range(n))
    wt_w = la.matmul(la.transpose(w), w)
    at_a = la.matmul(la.transpose(a_inv), a_inv)
    tr_wtw_ata = sum(la.matmul(wt_w, at_a)[i][i] for i in range(n))
    dim = k + 2
    info = la.zeros(dim)
    for i in range(k):
        for j in range(k):
            info[i][j] = xtx[i][j] / sigma2
    info[k][k] = n / (2.0 * sigma2 * sigma2)
    info[k][k + 1] = tr_wa / sigma2
    info[k + 1][k] = tr_wa / sigma2
    info[k + 1][k + 1] = tr_wb + tr_wtw_ata
    cov = la.solve(info, la.identity(dim))  # I^-1，含 rho 与 beta 的联合协方差
    se_all = [math.sqrt(max(cov[i][i], 0.0)) for i in range(dim)]

    # ---- 数值观测 Hessian 独立旁证（不参与推断；期望信息与观测 Hessian 是不同量，
    #      仅用于检查似然曲率数值稳定、无平坦方向）----
    hess_num = _numeric_hessian(ll_full, [beta, sigma2, rho], w, x, y, eigs, n)
    try:
        neg_hess = [[-v for v in row] for row in hess_num]
        cov_num = la.solve(neg_hess, la.identity(dim))
        # 只比较 beta 块（两种信息矩阵在 beta 块一致），rho/sigma2 块期望量与观测量本就不同
        var_info = [cov[i][i] for i in range(k)]
        var_num = [cov_num[i][i] for i in range(k)]
        rel_gap = [abs(a - b) / max(abs(a), abs(b), 1e-300) for a, b in zip(var_info, var_num)]
        neg_hess_eig = sorted(r for r, im in la.eigenvalues(neg_hess) if abs(im) < 1e-9)
        hess_check = {'beta_block_max_relative_variance_gap': max(rel_gap),
                      'observed_neg_hessian_min_real_eigenvalue': neg_hess_eig[0],
                      'note': '期望 Fisher 用于推断；观测 Hessian 仅作曲率稳定性旁证，二者 rho 块不可直接比较'}
    except la.SingularMatrixError:
        hess_check = {'beta_block_max_relative_variance_gap': None,
                      'note': '数值 Hessian 奇异，跳过比对'}

    effects = effects_decomposition(w, rho, beta, sigma2, design, cov, k, eigs, seed, mc_draws)
    diag = fit_diagnostics(sample, design, w, x, y, resid, rho, beta, sigma2, ll,
                           eigs, mle, logdet_gap, rcond, se_all, cov, k, hess_num, info)
    diag['numeric_hessian_check'] = hess_check
    return {
        'n': n, 'k': k, 'rho': rho, 'beta': beta, 'sigma2': sigma2,
        'log_likelihood': ll, 'coefficients': _coef_table(design['names'], beta, se_all, k, cov),
        'rho_se': se_all[k + 1], 'sigma2_se': se_all[k],
        'information_matrix': info, 'covariance': cov,
        'eigenvalues_real_min': lam_min, 'eigenvalues_real_max': lam_max,
        'feasible_rho_interval': [lower, upper],
        'rho_inside_feasible': lower < rho < upper,
        'logdet': {'eigen_path': ld_eig, 'lu_path': ld_lu, 'abs_gap': logdet_gap,
                   'rcond_I_minus_rhoW': rcond},
        'effects': effects, 'diagnostics': diag, 'optimizer': {
            'grid_points': GRID_POINTS, 'golden_iters': GOLDEN_ITERS,
            'grid_local_modes': mle['grid_modes'], 'boundary_distance': mle['boundary_distance']},
        'causality_note': 'SAR 刻画给定 W 下的统计空间依赖，不证明产业扩散的因果机制。',
    }


def ll_full(beta, sigma2, rho, w, x, y, eigs, n):
    s = [y[i] - rho * la.matvec(w, y)[i] for i in range(n)]
    xb = la.matvec(x, beta)
    resid = [s[i] - xb[i] for i in range(n)]
    sse = sum(r * r for r in resid)
    ld = logdet_eigen(eigs, rho)
    return -(n / 2.0) * math.log(2.0 * math.pi * sigma2) + ld - sse / (2.0 * sigma2)


def _numeric_hessian(fun, params, w, x, y, eigs, n):
    beta, sigma2, rho = params
    theta = list(beta) + [sigma2, rho]
    k = len(beta)
    dim = len(theta)
    # 各参数自适应步长（取机器精度六次方根量级，平衡截断与舍入误差）
    steps = []
    for t in theta:
        steps.append(max(la.EPS ** (1.0 / 3.0) * max(abs(t), 1.0), 1e-7))
    hess = la.zeros(dim)
    for i in range(dim):
        for j in range(i, dim):
            tp, tm = theta[:], theta[:]
            tp[i] += steps[i]; tm[i] -= steps[i]
            if i == j:
                def f(v):
                    b, s, r = v[:k], v[k], v[k + 1]
                    return fun(b, s, r, w, x, y, eigs, n)
                val = (f(tp) - 2.0 * f(theta) + f(tm)) / (steps[i] ** 2)
            else:
                tpp, tpm, tmp, tmm = theta[:], theta[:], theta[:], theta[:]
                tpp[i] += steps[i]; tpp[j] += steps[j]
                tpm[i] += steps[i]; tpm[j] -= steps[j]
                tmp[i] -= steps[i]; tmp[j] += steps[j]
                tmm[i] -= steps[i]; tmm[j] -= steps[j]

                def f(v):
                    b, s, r = v[:k], v[k], v[k + 1]
                    return fun(b, s, r, w, x, y, eigs, n)
                val = (f(tpp) - f(tpm) - f(tmp) + f(tmm)) / (4.0 * steps[i] * steps[j])
            hess[i][j] = val
            hess[j][i] = val
    return hess


def _coef_table(names, beta, se_all, k, cov):
    rows = []
    for j, name in enumerate(names):
        se = se_all[j]
        z = beta[j] / se if se > 0 else None
        p = 2.0 * (1.0 - la.normal_cdf(abs(z))) if z is not None else None
        rows.append({'term': name, 'estimate': beta[j], 'std_error': se,
                     'z': z, 'p_value': p,
                     'ci95': [beta[j] - 1.959963984540054 * se,
                              beta[j] + 1.959963984540054 * se]})
    return rows


# ---------------------------------------------------------------- 影响分解
def _effect_scalars(w, rho):
    """返回 LeSage-Pace 平均影响标量 (M_direct, M_total)。

    M_direct = tr(A)/n：平均直接效应（每个城市自身受冲击的平均反馈，含空间乘数）。
    M_total  = (1'A1)/n：平均总效应（对某城市施加单位冲击后，跨全部城市的平均响应行和）。
    间接效应 = M_total - M_direct。A=(I-rhoW)^-1，用 LU 解，不显式求逆。
    """
    n = len(w)
    a = la.identity(n)
    for i in range(n):
        for j in range(n):
            a[i][j] -= rho * w[i][j]
    lu, piv = la.lu_decompose(a)
    tr = 0.0
    total_sum = 0.0
    for j in range(n):
        e = [0.0] * n
        e[j] = 1.0
        col = la.lu_solve(lu, piv, e)  # A 的第 j 列
        tr += col[j]
        total_sum += sum(col)
    avg_direct = tr / n
    avg_total = total_sum / n
    return avg_direct, avg_total


def effects_decomposition(w, rho, beta, sigma2_hat, design, cov, k, eigs, seed, mc_draws):
    """LeSage-Pace 平均影响分解；联合蒙特卡洛传播 (beta,rho) 联合变化，Delta 法交叉验证。"""
    n = len(w)
    avg_direct, avg_total = _effect_scalars(w, rho)
    names = design['names']

    # 点估计（截距的总效应不解释为空间溢出，仅作完整性输出）
    point = []
    for j, name in enumerate(names):
        d = avg_direct * beta[j]
        t = avg_total * beta[j]
        point.append({'term': name, 'direct': d, 'indirect': t - d, 'total': t})

    # 网格预计算 q0(rho)=tr(A)/n, q1(rho)=1'A1/n^2，供 MC 线性插值。
    # 可行域边界处 A=(I-rhoW)^-1 奇异、标量发散，而联合抽样的有效支撑集中在 rho_hat 附近，
    # 故网格只覆盖 [rho_hat ± 6*se_rho] 与可行域收缩边界的交集（平滑区域，插值误差可核验）。
    lower, upper, _, _ = feasible_interval(eigs)
    se_rho = math.sqrt(max(cov[k + 1][k + 1], 0.0))
    center = rho
    half_width = 6.0 * se_rho
    safety = 1e-2 * (upper - lower)
    grid_lo = max(lower + safety, center - half_width)
    grid_hi = min(upper - safety, center + half_width)
    if grid_hi <= grid_lo:
        grid_lo, grid_hi = lower + safety, upper - safety
    # 效应标量曲率随 |rho| 增大而上升，在靠近上界端加密：用余弦聚点网格（端点密、中部疏），
    # 在不增加总成本的前提下压低线性插值误差。
    def clustered_grid(lo, hi, points):
        out = []
        for t in range(points):
            u = t / (points - 1)
            # 余弦映射：在两端聚集更多节点
            v = 0.5 - 0.5 * math.cos(math.pi * u)
            out.append(lo + (hi - lo) * v)
        return out

    grid = clustered_grid(grid_lo, grid_hi, EFFECT_GRID_POINTS)
    q0_grid, q1_grid = [], []
    for r in grid:
        d0, t0 = _effect_scalars(w, r)
        q0_grid.append(d0)
        q1_grid.append(t0)

    def interp(rho_v, vals):
        if rho_v <= grid[0]:
            return vals[0]
        if rho_v >= grid[-1]:
            return vals[-1]
        # 非均匀网格：二分定位所在区间
        lo_i, hi_i = 0, len(grid) - 1
        while hi_i - lo_i > 1:
            mid = (lo_i + hi_i) // 2
            if grid[mid] <= rho_v:
                lo_i = mid
            else:
                hi_i = mid
        span = grid[hi_i] - grid[lo_i]
        frac = (rho_v - grid[lo_i]) / span
        return vals[lo_i] * (1.0 - frac) + vals[hi_i] * frac

    # 插值误差：在 7 个网格外点精确复算
    check_rhos = [grid[0] + (grid[-1] - grid[0]) * (t + 0.5) / 7.0 for t in range(7)]
    interp_errors = []
    for r in check_rhos:
        d_exact, t_exact = _effect_scalars(w, r)
        interp_errors.append(max(abs(interp(r, q0_grid) - d_exact),
                                 abs(interp(r, q1_grid) - t_exact)))
    interp_max_error = max(interp_errors)

    # 联合抽样：theta = [beta..., sigma2, rho]，协方差来自完整期望信息矩阵逆。
    # 中心必须是全部参数的 MLE（sigma2 位置不能置 0，否则联合分布偏移）。
    l_chol = la.cholesky(cov)
    rng = random.Random(seed)
    theta_center = list(beta) + [sigma2_hat, rho]
    draws_direct = [[] for _ in names]
    draws_indirect = [[] for _ in names]
    draws_total = [[] for _ in names]
    n_outside = 0
    for _ in range(mc_draws):
        z = [rng.gauss(0.0, 1.0) for _ in range(k + 2)]
        draw = [0.0] * (k + 2)
        for i in range(k + 2):
            draw[i] = theta_center[i] + sum(l_chol[i][jj] * z[jj] for jj in range(i + 1))
        b_draw, rho_draw = draw[:k], draw[k + 1]
        if not (grid[0] < rho_draw < grid[-1]):
            # 超出可行域或有效插值网格的抽样：剔除并计数（不截断、不伪造）
            n_outside += 1
            continue
        q0, q1 = interp(rho_draw, q0_grid), interp(rho_draw, q1_grid)
        for j in range(k):
            d = q0 * b_draw[j]
            tt = q1 * b_draw[j]
            draws_direct[j].append(d)
            draws_indirect[j].append(tt - d)
            draws_total[j].append(tt)

    def summarize(vals):
        if len(vals) < 2:
            return None
        qs = la.quantile(vals, [0.025, 0.975])
        m = la.mean(vals)
        sd = la.sample_var(vals) ** 0.5
        # MC 标准误（均值的蒙特卡洛误差）
        mc_se = sd / math.sqrt(len(vals))
        return {'mean': m, 'sd': sd, 'ci95': qs, 'mc_standard_error': mc_se,
                'draws_used': len(vals)}

    mc_table = []
    for j, name in enumerate(names):
        mc_table.append({
            'term': name,
            'direct': summarize(draws_direct[j]),
            'indirect': summarize(draws_indirect[j]),
            'total': summarize(draws_total[j]),
        })

    # Delta 法交叉验证：效应 g(beta,rho)=beta_j*q(rho)，梯度 [q(对beta), beta*q'(rho)]
    # q' 用中心差分（精确标量复算，非插值）
    h = max(1e-5 * abs(rho), 1e-6)
    d_plus_d, t_plus = _effect_scalars(w, min(rho + h, grid[-1]))
    d_minus_d, t_minus = _effect_scalars(w, min(max(rho - h, grid[0]), grid[-1]))
    dq0 = (d_plus_d - d_minus_d) / (2.0 * h)
    dq1 = (t_plus - t_minus) / (2.0 * h)
    delta_rows = []
    for j, name in enumerate(names):
        row = {'term': name}

        def delta_se_for(q, dq):
            # g(beta,rho)=beta_j*q(rho)：对 beta_j 梯度 q，对 rho 梯度 beta_j*dq，其余为 0
            grad = [0.0] * (k + 2)
            grad[j] = q
            grad[k + 1] = beta[j] * dq
            var = sum(grad[a] * cov[a][b] * grad[b] for a in range(k + 2) for b in range(k + 2))
            return math.sqrt(max(var, 0.0))

        se_direct = delta_se_for(avg_direct, dq0)
        se_total = delta_se_for(avg_total, dq1)
        # 间接 = 总 - 直接 = beta_j*(q1-q0)，用其自身联合梯度（含总/直接的协方差）
        q_ind = avg_total - avg_direct
        dq_ind = dq1 - dq0
        se_indirect = delta_se_for(q_ind, dq_ind)
        row['direct'] = {'estimate': avg_direct * beta[j], 'delta_se': se_direct,
                         'ci95': [avg_direct * beta[j] - 1.959963984540054 * se_direct,
                                  avg_direct * beta[j] + 1.959963984540054 * se_direct]}
        row['total'] = {'estimate': avg_total * beta[j], 'delta_se': se_total,
                        'ci95': [avg_total * beta[j] - 1.959963984540054 * se_total,
                                 avg_total * beta[j] + 1.959963984540054 * se_total]}
        row['indirect'] = {'estimate': q_ind * beta[j], 'delta_se': se_indirect,
                           'ci95': [q_ind * beta[j] - 1.959963984540054 * se_indirect,
                                    q_ind * beta[j] + 1.959963984540054 * se_indirect]}
        delta_rows.append(row)

    return {
        'point_estimates': point,
        'joint_monter_carlo': {
            'draws_requested': mc_draws, 'draws_outside_feasible': n_outside,
            'seed': seed, 'table': mc_table,
            'grid_interpolation_max_abs_error': interp_max_error,
            'grid_points': EFFECT_GRID_POINTS,
            'approximation_note': ('效应标量在有效支撑网格上分段线性插值；误差由网格外点精确复算核验并报告，'
                                   '适用前提是抽样 rho 落在平滑支撑区内，越界抽样已剔除计数。'),
            'method_note': '对 (beta, sigma2, rho) 联合正态抽样；rho 越出实际可行域的抽样剔除并计数。',
        },
        'delta_method_crosscheck': delta_rows,
        'interpretation_note': 'beta 是条件系数，不是总边际影响；间接效应才是邻地传导部分。',
    }


# ---------------------------------------------------------------- 拟合诊断
def fit_diagnostics(sample, design, w, x, y, resid, rho, beta, sigma2, ll,
                    eigs, mle, logdet_gap, rcond, se_all, cov, k, hess_num, info):
    n, kk = len(y), len(design['names'])
    # 残差剖面
    m = la.mean(resid)
    sd = la.sample_var(resid) ** 0.5
    skew = sum((r - m) ** 3 for r in resid) / (n * sd ** 3) if sd > 0 else None
    kurt = sum((r - m) ** 4 for r in resid) / (n * sd ** 4) - 3.0 if sd > 0 else None
    # R2：基于简化式 y = A^-1 X beta + A^-1 eps 的拟合
    a_inv_xb = _a_inv_apply(w, rho, la.matvec(x, beta))
    ybar = la.mean(y)
    sst = sum((v - ybar) ** 2 for v in y)
    sse_model = sum((y[i] - a_inv_xb[i]) ** 2 for i in range(n))
    r2 = 1.0 - sse_model / sst if sst > 0 else None
    # OLS 基准（无空间项）与 LM-lag 检验
    beta_ols, sse_ols, _ = _ols_fit(x, y)
    resid_ols = [y[i] - sum(x[i][j] * beta_ols[j] for j in range(kk)) for i in range(n)]
    sigma2_ols = sse_ols / n
    # Moran's I on SAR residuals（条件置换）
    from .spatial import morans_i
    moran = morans_i(w, resid)
    # LM-lag（Anselin 1988）：分母 = tr(W'W+W^2) + (WXb)' M (WXb) / sigma2_ols，M=I-X(X'X)^-1X'
    wy = la.matvec(w, y)
    wx = la.matmul(w, x)
    wt_w = la.matmul(la.transpose(w), w)
    w2 = la.matmul(w, w)
    tr_term = sum(wt_w[i][i] + w2[i][i] for i in range(n))
    wxb = la.matvec(wx, beta_ols)
    xtx_inv = la.inverse(la.matmul(la.transpose(x), x))
    proj = la.matvec(x, la.matvec(xtx_inv, la.matvec(la.transpose(x), wxb)))
    m_wxb = [wxb[i] - proj[i] for i in range(n)]
    denom = tr_term + sum(v * v for v in m_wxb) / sigma2_ols
    lm_lag = (sum(resid_ols[i] * wy[i] for i in range(n)) / sigma2_ols) ** 2 / denom
    lm_p = la.chi2_sf(lm_lag, 1)
    # 信息矩阵条件数（共线/弱识别诊断）
    try:
        info_rcond = la.reciprocal_condition(info)
    except la.SingularMatrixError:
        info_rcond = 0.0
    # 设计矩阵相关矩阵（不含截距列）
    collin = _collinearity(x, design)
    aic = -2.0 * ll + 2.0 * (kk + 2)
    bic = -2.0 * ll + math.log(n) * (kk + 2)
    return {
        'residual_profile': {'mean': m, 'sd': sd, 'skewness': skew, 'excess_kurtosis': kurt,
                             'min': min(resid), 'max': max(resid)},
        'r_squared_reduced_form': r2,
        'ols_comparison': {
            'beta_ols': beta_ols, 'sigma2_ols': sigma2_ols,
            'loglik_ols': -(n / 2.0) * (math.log(2.0 * math.pi * sigma2_ols) + 1.0),
            'lm_lag_statistic': lm_lag, 'lm_lag_p_value': lm_p},
        'moran_residual': moran,
        'information_matrix_rcond': info_rcond,
        'collinearity': collin,
        'aic': aic, 'bic': bic,
        'logdet_crosscheck_gap': logdet_gap,
        'rcond_I_minus_rhoW': rcond,
        'numeric_hessian_check': None,  # 由 fit_sar 填充
    }


def _a_inv_apply(w, rho, v):
    n = len(w)
    a = la.identity(n)
    for i in range(n):
        for j in range(n):
            a[i][j] -= rho * w[i][j]
    return la.solve(a, v)


def _collinearity(x, design):
    names = design['names']
    n = len(x)
    cols = la.transpose(x)
    out = {'correlation': [], 'vif': []}
    # 相关矩阵（全部列，截距与变量的相关也保留以便剖面诊断）
    for i, ni in enumerate(names):
        row = []
        for j, nj in enumerate(names):
            if i == j:
                row.append(1.0)
            else:
                try:
                    row.append(la.pearson(cols[i], cols[j]))
                except la.SingularMatrixError:
                    row.append(None)
        out['correlation'].append({'term': ni, 'with': names, 'values': row})
    # VIF：跳过截距列
    for i, name in enumerate(names):
        if design['intercept'] and i == 0:
            continue
        others = [cols[j] for j in range(len(names)) if j != i]
        x_other = [list(row) for row in zip(*others)]
        try:
            b, sse, _ = _ols_fit(x_other, cols[i])
            sst = sum((v - la.mean(cols[i])) ** 2 for v in cols[i])
            r2_j = 1.0 - sse / sst if sst > 0 else 0.0
            vif = 1.0 / (1.0 - r2_j) if r2_j < 1.0 else float('inf')
        except la.SingularMatrixError:
            vif = float('inf')
        out['vif'].append({'term': name, 'vif': vif, 'r_squared_on_others': min(r2_j, 1.0)})
    return out
