# WSArena

WSArena（Weight-Space Arena）研究语言模型能否通过直接编辑神经网络权重来修复或改善控制策略。模型读取策略权重和开发集反馈，用结构化补丁修改参数。评测器在隐藏的分布内（ID）和分布外（OOD）episode 上衡量最后一轮编辑的效果。

**研究进展快照 · 2026-11-13（新加坡时间）**

Claude 已完成 Tier-0 和 Meta-World 主批次。GPT 已完成 Native Pilot，Tier-0 主批次已验证 6/64 个运行。

- [进度、结果与分析](RESEARCH_STATUS_ZH.md)
- [后续实验与复现计划](EXPERIMENT_ROADMAP_ZH.md)
- [冻结结果及后续复核](#结果资料)

## 目前发现

- Tier-0 的四个 Claude 臂在修复任务上均优于随机编辑。Meta-World 的两个 medium 臂也优于随机编辑，但原报告中均低于可使用专家标签的闭式重拟合。Meta-World 的脚本对照与主运行使用了不同的隐藏评测预算，同预算复评待完成。
- 在 Tier-0 的八个盲读检查点上，权重可见与盲读的平均增益差为 +0.006，按检查点聚类的 95% CI 为 [−0.115, +0.115]。区间仍容许两个方向上的效应。
- 单次运行波动较大。Tier-0 和 Meta-World 两个 medium 臂的运行间标准差约为 0.13 和 0.08。GPT Pilot 只提供接入和消耗证据，GPT 主批次仍在采集。

## 结果资料

| 文档 | 内容 |
|---|---|
| [Claude Tier-0](reports/MAIN_RESULTS.md) | 冻结主检验、次要检验与编辑机制 |
| [Claude Meta-World](reports/MAIN_RESULTS_MW.md) | 冻结主检验、对照比较与运行统计 |
| [盲读与输出上限](reports/A3_BLIND_CAP.md) | 权重可见性及 Claude 32k/64k 敏感性 |
| [盲读与重参数化 Pilot](reports/A3_A4_ANALYSIS.md) | 机制诊断与等价参数表示敏感性 |
| [附加分析](reports/MAIN_EXTRA.md) | 效应、effort、重复运行、编辑行为 |
| [检查点筛选](reports/CHECKPOINT_SCREEN.md) | 已知组效度、任务区分度和地板效应 |
| [独立复核补充](reports/REVIEW_P8_ADDENDUM.md) | 置信区间、配对检验与运行间波动 |
| [C3 Tier-0](reports/C3_ESTIMAND_CHECK_T0.md) · [C3 Meta-World](reports/C3_ESTIMAND_CHECK_MW.md) | 任务等权与检查点等权估计量对照 |

本仓库发布研究进展、实验计划及整理后的汇总结果。源码、检查点、运行日志与论文全文留待单独的可复现版本发布。
