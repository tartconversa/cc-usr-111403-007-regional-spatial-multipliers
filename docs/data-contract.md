# 资料与现有计算约定

## 修订快照

schema_version为整数1，metadata为业务方案字典，collections为业务集合字典。每条记录有非空字符串id和正整数revision。
同集合相同id取最高revision；同id同revision内容不同即冲突，即便存在更高修订也拒绝。deleted=true为撤回，之后更高修订可恢复。
撤回记录不要求携带业务字段。输入顺序与相同记录重试不改变语义；非有限数字拒绝。
Snapshot.rows返回独立副本；input_digest绑定有效修订和全部metadata。输出data_version沿用metadata，不由程序自造资料版本。
既有摘要入口为app.service.inspect(payload)，输出schema_version、input_digest、data_version和summary。

## 业务资料

cities含y、features、status；edges含from、to、weight，from受to影响，非负权重且无自边。weight_policy为row_standardized，先按完整城市全集行和归一，零行保留为零。缺资料城市的处理方案由分析显式声明，不由读取器重标化。baseline.analyze只生成键序W和原始城市资料，重复边按权重相加，未知端点拒绝。

各集合必要字段：{"cities": ["y", "features", "status"], "edges": ["from", "to", "weight"]}。

## 当前能力

app.baseline.analyze(snapshot)产生已有摘要，app.service.inspect负责绑定输入身份；新增分析可以增设独立入口，已有接口及上述数据语义保持兼容。

## 空间滞后模型（SAR）

新增独立入口 app.spatial_service.analyze_spatial，不改动 app.service 的既有输出。登记模型为
y = rho·W·y + X·beta + epsilon，epsilon ~ N(0, sigma^2 I)，采用极大似然估计。

样本边界（缺资料城市）由分析显式声明，提供两套可比较方案：
- frozen（登记口径，默认）：先在完整城市全集按行和归一 W，再取观测城市子矩阵，不做二次归一；
  被保留城市因邻居被剔除而出现的行和小于 1 会原样报告。缺失 y 一律剔除，绝不补零。
- renorm（敏感性对照）：在缩减样本上对原始权重子矩阵重新行归一。两套结果的符号、显著性差异由 boundary_comparison 显式量化。

rho 可行域由实际 W 的实特征值决定：1/lambda_min < rho < 1/lambda_max（复根不约束实 rho），
不能仅凭优化值落在 (-1,1) 就判定 I-rhoW 可逆。对数行列式 log|I-rhoW| 同时经特征值路径与
LU 行列式路径计算并报告绝对误差；终值处另报 I-rhoW 的一范数条件数倒数作为可逆性证据。
全零 W（样本全部孤立）时空间项退化、rho 不可识别，程序显式报错。

影响分解按 LeSage-Pace 口径：平均直接效应 tr(A)/n·beta、平均总效应 (1'A1)/n·beta、
间接效应为二者之差，A=(I-rhoW)^-1。beta 是条件系数，不是总边际影响。不确定性通过
(beta, sigma2, rho) 的期望 Fisher 信息矩阵联合协方差整体传播（联合蒙特卡洛，固定种子），
并以 Delta 法交叉验证；不对 rho、beta 分别取端点拼区间。效应标量在有效支撑网格上分段线性
插值，网格外点精确复算报告误差，越出可行域的抽样剔除并计数。

诊断保留残差剖面、简化式 R²、OLS 基准与 LM-lag、残差 Moran's I（条件置换）、信息矩阵条件数、
相关矩阵与 VIF 共线诊断、AIC/BIC、数值 Hessian 曲率旁证。全部计算为纯标准库稠密 O(n^3) 实现，
适用城市中小样本；输出携带城市键序、W、权重方案、缩放、排序与种子，可回到具体城市及权重复算。
模型刻画给定空间关系下的统计依赖，不证明产业扩散的因果机制。
