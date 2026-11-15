# 区域工业空间滞后模型的识别与影响分解

本项目提供城市样本与空间权重边界相关资料的现有业务摘要。当前代码执行资料修订读取与基础统计，不包含扩展分析。

app/catalog.py 按集合和记录键解析修订，拒绝同修订冲突；app/baseline.py 承载当前业务计算；app/service.py 输出摘要及输入摘要哈希。
数据字段及现有接口见 [资料契约](docs/data-contract.md)。fixtures/sample.json 是合成小数据，不对应真实企业或个人。
运行环境为 Python 3.14，现有模块仅使用标准库。

读取并计算当前摘要：

    python -m app.service fixtures/sample.json

空间滞后模型分析（SAR，独立入口，不改变既有摘要语义）：

    python -m app.spatial_service fixtures/spatial_sample.json [mc_draws] [seed]

输出含 frozen（完整全集归一后取子矩阵、不重新归一、不补零）与 renorm（缩减样本重标化）两套
估计及边界方案对比、rho 可行域与 log-det 双路径误差、拟合诊断、共线诊断，以及直接/间接/总影响
的联合不确定性。方法口径、样本边界与适用条件见 [资料契约](docs/data-contract.md)。

运行当前模块检查：

    python -m unittest discover -s tests -v

检查源码编译：

    python -m compileall -q app tests
