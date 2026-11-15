"""纯标准库稠密线性代数与统计原语（无第三方依赖）。

设计原则：
- 全部使用 Python 浮点（IEEE 双精度）与列表矩阵；适用于城市横截面的小/中样本。
- 关键量提供两条独立计算路径（LU 行列式 vs 特征值求和），由调用方比对并报告误差。
- 不静默吞掉奇异情形：奇异矩阵抛出 SingularMatrixError，由上层决定降级或报错。
"""
import math
import random

EPS = 2.220446049250313e-16
_SINGULAR_TOL = 1e-12


class SingularMatrixError(ValueError):
    """矩阵在当前浮点容差下奇异。"""


# ---------------------------------------------------------------- 基础运算
def zeros(n, m=None):
    m = n if m is None else m
    return [[0.0] * m for _ in range(n)]


def identity(n):
    a = zeros(n)
    for i in range(n):
        a[i][i] = 1.0
    return a


def copy(a):
    return [row[:] for row in a]


def shape(a):
    return len(a), len(a[0])


def transpose(a):
    n, m = shape(a)
    return [[a[i][j] for i in range(n)] for j in range(m)]


def matmul(a, b):
    na, ma = shape(a)
    mb = len(b[0])
    if ma != len(b):
        raise ValueError('矩阵维度不匹配')
    out = zeros(na, mb)
    for i in range(na):
        ai = a[i]
        oi = out[i]
        for k in range(ma):
            aik = ai[k]
            if aik == 0.0:
                continue
            bk = b[k]
            for j in range(mb):
                oi[j] += aik * bk[j]
    return out


def matvec(a, x):
    n, m = shape(a)
    if m != len(x):
        raise ValueError('矩阵向量维度不匹配')
    return [sum(a[i][j] * x[j] for j in range(m)) for i in range(n)]


def vec_add(x, y, s=1.0):
    return [xi + s * yi for xi, yi in zip(x, y)]


def scale(a, s):
    if isinstance(a[0], list):
        return [[v * s for v in row] for row in a]
    return [v * s for v in a]


def frobenius_sq(a):
    return sum(v * v for row in a for v in row)


# ---------------------------------------------------------------- LU 分解
def lu_decompose(a):
    """带部分主元的 LU 分解。返回 (lu, piv)，lu 下三角（不含对角）为 L，上三角为 U，piv 为行置换。"""
    n = len(a)
    lu = copy(a)
    piv = list(range(n))
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(lu[i][k]))
        if abs(lu[p][k]) <= _SINGULAR_TOL:
            raise SingularMatrixError('LU 分解遇到奇异主元')
        if p != k:
            lu[k], lu[p] = lu[p], lu[k]
            piv[k], piv[p] = piv[p], piv[k]
        pivot = lu[k][k]
        for i in range(k + 1, n):
            lik = lu[i][k] / pivot
            lu[i][k] = lik
            rowi, rowk = lu[i], lu[k]
            for j in range(k + 1, n):
                rowi[j] -= lik * rowk[j]
    return lu, piv


def lu_solve(lu, piv, b):
    """利用 lu_decompose 的结果解线性方程组。"""
    n = len(lu)
    x = [b[piv[i]] for i in range(n)]
    # 前向代入 Ly = Pb（L 对角为 1）
    for i in range(1, n):
        s = x[i]
        row = lu[i]
        for j in range(i):
            s -= row[j] * x[j]
        x[i] = s
    # 回代 Ux = y
    for i in range(n - 1, -1, -1):
        s = x[i]
        row = lu[i]
        for j in range(i + 1, n):
            s -= row[j] * x[j]
        if abs(row[i]) <= _SINGULAR_TOL:
            raise SingularMatrixError('回代遇到奇异主元')
        x[i] = s / row[i]
    return x


def solve(a, b):
    lu, piv = lu_decompose(a)
    if isinstance(b[0], list):
        return [lu_solve(lu, piv, col) for col in transpose(b)]
    return lu_solve(lu, piv, b)


def log_abs_det_lu(lu, piv):
    """由 LU 结果计算 log|det|，并返回置换符号。"""
    n = len(lu)
    total = 0.0
    sign = 1
    counted = [False] * n
    for i in range(n):
        if piv[i] != i:
            sign = -sign
        u = lu[i][i]
        if u == 0.0:
            raise SingularMatrixError('零主元，行列式为零')
        total += math.log(abs(u))
        counted[i] = True
    return total, sign


def inverse(a):
    n = len(a)
    lu, piv = lu_decompose(a)
    cols = []
    for j in range(n):
        e = [0.0] * n
        e[j] = 1.0
        cols.append(lu_solve(lu, piv, e))
    return transpose(cols)


def reciprocal_condition(a):
    """一阶范数下的 1-范数条件数倒数估计 rcond = 1/(||A||_1 * ||A^-1||_1)。"""
    n = len(a)
    norm_a = max(sum(abs(a[i][j]) for i in range(n)) for j in range(n))
    inv_a = inverse(a)
    norm_inv = max(sum(abs(inv_a[i][j]) for i in range(n)) for j in range(n))
    if norm_a == 0.0 or norm_inv == 0.0:
        return 0.0
    return 1.0 / (norm_a * norm_inv)


# ---------------------------------------------------------------- Cholesky
def cholesky(a):
    """对称正定矩阵的 Cholesky 下三角 L，使 L L^T = A。"""
    n = len(a)
    l = zeros(n)
    for i in range(n):
        for j in range(i + 1):
            s = sum(l[i][k] * l[j][k] for k in range(j))
            if i == j:
                val = a[i][i] - s
                if val <= _SINGULAR_TOL:
                    raise SingularMatrixError('矩阵非正定，无法 Cholesky 分解')
                l[i][j] = math.sqrt(val)
            else:
                l[i][j] = (a[i][j] - s) / l[j][j]
    return l


def solve_spd(a, b):
    l = cholesky(a)
    n = len(a)

    def fwd(rhs):
        y = [0.0] * n
        for i in range(n):
            s = rhs[i] - sum(l[i][k] * y[k] for k in range(i))
            y[i] = s / l[i][i]
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = y[i] - sum(l[k][i] * x[k] for k in range(i + 1, n))
            x[i] = s / l[i][i]
        return x

    if isinstance(b[0], list):
        return transpose([fwd(col) for col in transpose(b)])
    return fwd(b)


# ---------------------------------------------------------------- 对称特征值（循环 Jacobi）
def symmetric_eigen(a, max_sweeps=100, tol=None):
    """实对称矩阵特征分解，返回 (eigenvalues, eigenvectors 按列)。仅用于信息矩阵等小矩阵。"""
    n = len(a)
    tol = EPS if tol is None else tol
    m = copy(a)
    v = identity(n)
    for _ in range(max_sweeps):
        off = math.sqrt(sum(m[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        if off <= tol * math.sqrt(frobenius_sq(m) + EPS):
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(m[p][q]) <= tol:
                    continue
                tau = (m[q][q] - m[p][p]) / (2.0 * m[p][q])
                t = (1.0 if tau >= 0 else -1.0) / (abs(tau) + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                for k in range(n):
                    mkp, mkq = m[k][p], m[k][q]
                    m[k][p] = c * mkp - s * mkq
                    m[k][q] = s * mkp + c * mkq
                for k in range(n):
                    mpk, mqk = m[p][k], m[q][k]
                    m[p][k] = c * mpk - s * mqk
                    m[q][k] = s * mpk + c * mqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    vals = [m[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: vals[i])
    vals_sorted = [vals[i] for i in order]
    vecs = transpose([[v[r][i] for r in range(n)] for i in order])
    return vals_sorted, vecs


# ---------------------------------------------------------------- 一般矩阵特征值（Hessenberg + 隐式双位移 QR）
def _hessenberg(a):
    """Householder 相似变换化为上 Hessenberg 矩阵。

    每步对列向量 x = h[k+1:n, k] 构造显式 Householder 向量 u（u0 = x0 + sign*||x||），
    用外积形式 P = I - 2 u u'/(u'u) 分别做左乘与右乘，避免就地索引歧义。
    """
    n = len(a)
    h = copy(a)
    for k in range(n - 2):
        x = [h[i][k] for i in range(k + 1, n)]
        normx = math.sqrt(sum(v * v for v in x))
        if normx <= EPS:
            continue
        sgn = 1.0 if x[0] >= 0 else -1.0
        u = x[:]
        u[0] += sgn * normx
        u_norm_sq = sum(v * v for v in u)
        if u_norm_sq <= EPS:
            continue

        # 右乘：h[:, k+1:n] <- (I - 2uu'/u'u) h[:, k+1:n]
        for i in range(n):
            dot = sum(u[r] * h[i][k + 1 + r] for r in range(len(u)))
            coef = 2.0 * dot / u_norm_sq
            for r in range(len(u)):
                h[i][k + 1 + r] -= coef * u[r]
        # 左乘：h[k+1:n, :] <- (I - 2uu'/u'u) h[k+1:n, :]
        for j in range(n):
            dot = sum(u[r] * h[k + 1 + r][j] for r in range(len(u)))
            coef = 2.0 * dot / u_norm_sq
            for r in range(len(u)):
                h[k + 1 + r][j] -= coef * u[r]
    # 清掉次对角以下的数值噪声（相似变换后应为精确零）
    for i in range(n):
        for j in range(i - 2, -1, -1):
            h[i][j] = 0.0
    return h


def _givens(a, b):
    """标准 Givens 旋转参数 (c, s)：使 [c s; -s c][a; b] = [r; 0]，即 c=a/r, s=b/r。"""
    if b == 0.0:
        return 1.0, 0.0
    if abs(b) > abs(a):
        t = a / b
        s = 1.0 / math.sqrt(1.0 + t * t)
        return s * t, s
    t = b / a
    c = 1.0 / math.sqrt(1.0 + t * t)
    return c, c * t


def eigenvalues(a, max_iter=300):
    """实矩阵的全部特征值（实数与共轭复根，以 (re, im) 对返回，复根成对给出）。

    上 Hessenberg 化后执行带位移 QR（Numerical Recipes hqr 结构）：
    零次对角分裂子块，1x1 给实根、2x2 解析给实根或共轭复根。
    适用稠密中小矩阵；收敛失败显式报错而不返回伪值。
    """
    n = len(a)
    if n == 1:
        return [(a[0][0], 0.0)]
    if n == 2:
        return _eigen_2x2(a[0][0], a[0][1], a[1][0], a[1][1])
    h = _hessenberg(a)
    wr = [0.0] * n
    wi = [0.0] * n
    nn = n - 1
    its = 0
    # 分裂容差：相对尺度为主，加绝对下限，避免小次对角在 QR 充分收敛前被误判为零
    scale = max(max(abs(v) for row in h for v in row), 1.0)
    split_tol = 1e-12 * scale
    while nn >= 1:
        its += 1
        if its > max_iter:
            raise RuntimeError('特征值 QR 迭代未收敛')
        # 找最小 l 使 h[l][l-1] 可视为 0（子块起点）
        l = nn
        while l > 0:
            if abs(h[l][l - 1]) <= max(split_tol,
                                       EPS * (abs(h[l - 1][l - 1]) + abs(h[l][l]))):
                break
            l -= 1
        if l == nn:
            wr[nn] = h[nn][nn]
            wi[nn] = 0.0
            nn -= 1
            its = 0
        elif l == nn - 1:
            pair = _eigen_2x2(h[nn - 1][nn - 1], h[nn - 1][nn],
                              h[nn][nn - 1], h[nn][nn])
            wr[nn - 1], wi[nn - 1] = pair[0]
            wr[nn], wi[nn] = pair[1]
            nn -= 2
            its = 0
        else:
            # 连续 Wilkinson 位移：底部 2x2 特征值中离 h[nn][nn] 更近者
            a11 = h[nn - 1][nn - 1]
            a22 = h[nn][nn]
            off = h[nn][nn - 1] * h[nn - 1][nn]
            tr2 = a11 + a22
            disc = (a11 - a22) * (a11 - a22) + 4.0 * off
            if disc >= 0.0:
                r = math.sqrt(disc)
                e1, e2 = (tr2 + r) / 2.0, (tr2 - r) / 2.0
                shift = e1 if abs(e1 - a22) < abs(e2 - a22) else e2
            else:
                shift = tr2 / 2.0  # 复根区域取实部；复根本身由 2x2 解析处理
            for i in range(l, nn + 1):
                h[i][i] -= shift
            # 显式 QR 步：H_shift = Q R（全尺寸 Givens 旋转），再 H_new = R Q。
            # 用完整矩阵乘法避免原地旋转的顺序/范围歧义；稠密 O(n^3)，适用城市中小样本。
            r_mat = h
            q_mat = identity(n)
            for i in range(l, nn):
                c, s = _givens(r_mat[i][i], r_mat[i + 1][i])
                g = identity(n)
                g[i][i], g[i][i + 1] = c, s
                g[i + 1][i], g[i + 1][i + 1] = -s, c
                r_mat = matmul(g, r_mat)
                q_mat = matmul(q_mat, transpose(g))  # Q = G_1^T G_2^T ...
            h = matmul(r_mat, q_mat)
            for i in range(l, nn + 1):
                h[i][i] += shift
    if nn == 0:
        wr[0] = h[0][0]
        wi[0] = 0.0
    return list(zip(wr, wi))


def _eigen_2x2(a11, a12, a21, a22):
    tr, det = a11 + a22, a11 * a22 - a12 * a21
    disc = tr * tr - 4.0 * det
    if disc >= 0.0:
        r = math.sqrt(disc)
        return [((tr + r) / 2.0, 0.0), ((tr - r) / 2.0, 0.0)]
    r = math.sqrt(-disc)
    return [(tr / 2.0, r / 2.0), (tr / 2.0, -r / 2.0)]


def log_det_from_eigen(eigs):
    """由特征值计算 log|prod(lambda)|，复根共轭成对，取模。"""
    total = 0.0
    for re, im in eigs:
        total += 0.5 * math.log(re * re + im * im)
    return total


# ---------------------------------------------------------------- 统计辅助
def mean(x):
    return sum(x) / len(x)


def sample_var(x):
    m = mean(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - 1)


def pearson(x, y):
    mx, my = mean(x), mean(y)
    sx = math.sqrt(sum((v - mx) ** 2 for v in x))
    sy = math.sqrt(sum((v - my) ** 2 for v in y))
    if sx == 0.0 or sy == 0.0:
        raise SingularMatrixError('零方差变量，无法计算相关系数')
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def quantile(x, probs):
    """线性插值分位数（与 numpy 默认 linear 方法一致）。"""
    s = sorted(x)
    n = len(s)
    out = []
    for p in probs:
        if not 0.0 <= p <= 1.0:
            raise ValueError('分位概率越界')
        h = (n - 1) * p
        lo = int(math.floor(h))
        frac = h - lo
        if lo + 1 < n:
            out.append(s[lo] * (1.0 - frac) + s[lo + 1] * frac)
        else:
            out.append(s[-1])
    return out


def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def chi2_sf(x, df):
    """正则化下不完全伽马的补，df 为整数自由度。"""
    if x <= 0.0:
        return 1.0
    a = df / 2.0
    return _gammaincc(a, x / 2.0)


def _gammaincc(a, x):
    if x < a + 1.0:
        return 1.0 - _gammainc_series(a, x)
    return _gammaincc_cf(a, x)


def _gammainc_series(a, x, itmax=200):
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(itmax):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammaincc_cf(a, x, itmax=200):
    b = x + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gaussian_draws(mean_vec, l_chol, rng):
    """由 Cholesky 下三角 L 生成多元正态抽样 z = mean + L z0。"""
    n = len(mean_vec)
    z0 = [rng.gauss(0.0, 1.0) for _ in range(n)]
    z = mean_vec[:]
    for i in range(n):
        z[i] += sum(l_chol[i][j] * z0[j] for j in range(i + 1))
    return z
