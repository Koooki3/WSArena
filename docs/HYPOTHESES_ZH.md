# WSArena 假设卡 v0.1（轻量 S2；预注册前冻结判据）

每张卡必填：反事实增量、规模量级、KILL 阈值、区分性读数。阈值在预注册（PREREG）冻结；这里先写“候选”。主终点均为隐藏 ID 上的 ΔSR（Final-Round 轨道），单位=检查点（聚类 bootstrap）。

| ID | 假设 | 反事实增量（只用已有结果能做到什么） | KILL / 否决条件 | 区分性读数 |
|---|---|---|---|---|
| H1 | **修复能力**：在 S1/S2 上，Claude 的 ΔSR 显著大于“最强简单替代” | 简单替代 = `analytic_ridge`（特权标签的闭式解）、`gain_greedy`（单参数输出增益搜索；S2 上 +0.27）与 `random`；已有结果（MIT Kinetix、Qwen）都是单次、无对照 | 对每个模型×effort 臂：ΔSR 的 95% 聚类 CI 下界 ≤ 0，或不优于 `random`+噪声带 → 该臂 L2 KILL | 相对 `oracle_restore` 的比例；S1 的定位精确/召回（真值）；S2 上只有真正理解失败分型才会修的读数（修“overshoot”还是修“stalled”） |
| H2 | **推理强度非单调**：high 并不总优于 medium；失败类型随 effort 变化 | MIT 用 high/xhigh/max 无对照；社区转述“xhigh 耗尽输出预算” | 若 high 在 P(valid patch)、E[ΔSR\|valid]、ΔSR 三项上都不低于 medium，则“非单调”KILL（对该模型） | 把 ΔSR 分解为 P(valid)×P(improve\|valid)×E[Δ\|valid]；输出预算耗尽率；大输出上限的灵敏度子实验 |
| H3 | **增强（L3）弱于修复（L2）**（注：E1 显示自然 S3 仍可被特权闭式解改善 +0.17，故 H3 必须同时报告相对 analytic_ridge 的位置）：在 S3 上 Claude 的稳定提升显著小于 S1/S2 | 无已有对照；MIT 的 Gemma 270M 反例提示“只能移除障碍，不能添加能力” | 若 S3 的 ΔSR 与 S2 无显著差，或 S3 的 Regression 显著高于 Recovery 收益 → “增强弱”KILL；若 S3 显著为正且保留约束成立 → 记为“正常策略可被增强”（重要正结果） | Recovery−Regression 配对差；ΔJ_retention≥−ε 是否成立 |
| H4 | **理解 vs 记忆**：编辑效果对函数等价重参数化敏感 → 依赖参数排列而非功能理解 | 无 | PSI = \|ΔSR(θ) − ΔSR(T(θ))\|：若 PSI 的 95% CI 包含 0 且小于噪声带 → “对排列不敏感”（支持理解）；若显著大 → 支持“依赖排列” | 同一检查点的 θ 与 T(θ)（tanh 精确对称）各编辑一次；隐藏 episode 完全相同 |
| H5 | **可编程性随功能复杂度下降**：L2→L5 任务阶梯上 ΔSR 单调下降 | 来自 vukrosic 自报（有规则任务强、CIFAR 弱），未独立验证 | Spearman(任务级复杂度, ΔSR) 的聚类置换检验不显著，或非负 → KILL | 解析赛道 `analytic_ridge` 随复杂度的同曲线（区分“任务难”与“LLM 弱”）；c04-delay 需要估计器 |
| H6 | **模块局部性**：增益主要来自少数张量/单元（稀疏、定位） | MIT Qwen 为 0.016%；frontier-weight-edit 为 4–8 个神经元，均自报 | 若 S2 上成功补丁的 `frac_changed` 中位数 >20% 且分张量消融无集中 → KILL | 补丁因果消融（回退每个张量组）；`frac_changed`、`rel_l2`；Projection/方向消减型编辑单独标注 |

## 诚实条款对应
- T1：H1–H6 全部 KILL 时，默认报告零结果，附上界（`oracle_restore`、`analytic_ridge`）、召回、未覆盖层面（官方基准、大模型）与推翻条件。
- T2：KILL 后不放宽判据；Pilot 后新增的假设只能标“探索性”。
- 所有“显著”均指预注册的聚类 CI/置换检验；CI 含 0 ≠ 无效应（K-S6-02）。

## 与任务书 H1–H6 的对应
任务书 H1（识别缺陷）并入 H1 的“定位读数”；H2 推理强度=本 H2；H3 增强=本 H3；H4 函数等价=本 H4；H5 复杂度=本 H5；H6 模块依赖=本 H6（本期只在 Tier-0 的张量层面做，官方基准层再做 Action Head/Dynamics Head 级别）。
