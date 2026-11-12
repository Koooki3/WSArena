# WSArena 图表记录（自动生成，勿手改）

生成：2026-11-07T06:09:15+08:00 · 契约 v11 · 源码哈希 43f05e80。每张图有浅色/深色版本（png / pdf / svg）、CSV 表格视图和图内溯源脚注。
配色经 `PALETTE_VALIDATION.txt` 验证；颜色跟实体走（见 `wsarena/viz/style.py::ENTITY_SLOT`）。

| ID | 标题 | 类别 | 状态 | 图（浅色 png） | 表格视图 |
|---|---|---|---|---|---|
| FIG-01 | WSP-8 evaluation pipeline | overview | rendered | [FIG-01.png](figures/light/FIG-01.png) | [csv](figures/data/FIG-01.csv) |
| FIG-02 | Task formalisation and capability ladder | overview | rendered | [FIG-02.png](figures/light/FIG-02.png) | [csv](figures/data/FIG-02.csv) |
| FIG-03 | Layered failure attribution | overview | rendered | [FIG-03.png](figures/light/FIG-03.png) | [csv](figures/data/FIG-03.csv) |
| FIG-04 | Benchmark coverage matrix | benchmark | rendered | [FIG-04.png](figures/light/FIG-04.png) | [csv](figures/data/FIG-04.csv) |
| FIG-05 | Start-state difficulty of the checkpoints | benchmark | rendered | [FIG-05.png](figures/light/FIG-05.png) | [csv](figures/data/FIG-05.csv) |
| FIG-06 | Control arms: hidden ID gain | baselines | rendered | [FIG-06.png](figures/light/FIG-06.png) | [csv](figures/data/FIG-06.csv) |
| FIG-07 | Control arms: hidden OOD gain | baselines | rendered | [FIG-07.png](figures/light/FIG-07.png) | [csv](figures/data/FIG-07.csv) |
| FIG-08 | Gain by task, class and control arm | baselines | rendered | [FIG-08.png](figures/light/FIG-08.png) | [csv](figures/data/FIG-08.csv) |
| FIG-09 | Recovery versus Regression | baselines | rendered | [FIG-09.png](figures/light/FIG-09.png) | [csv](figures/data/FIG-09.csv) |
| FIG-10 | Evaluation noise floor | benchmark | rendered | [FIG-10.png](figures/light/FIG-10.png) | [csv](figures/data/FIG-10.csv) |
| FIG-11 | OOD shift calibration | benchmark | rendered | [FIG-11.png](figures/light/FIG-11.png) | [csv](figures/data/FIG-11.csv) |
| FIG-12 | What the unedited policies get wrong | benchmark | rendered | [FIG-12.png](figures/light/FIG-12.png) | [csv](figures/data/FIG-12.csv) |
| FIG-13 | Edit footprint versus gain | baselines | rendered | [FIG-13.png](figures/light/FIG-13.png) | [csv](figures/data/FIG-13.csv) |
| FIG-14 | Case study: one planted fault, three outcomes | case | rendered | [FIG-14.png](figures/light/FIG-14.png) | [csv](figures/data/FIG-14.csv) |
| FIG-15 | Validity and reproducibility record | process | rendered | [FIG-15.png](figures/light/FIG-15.png) | [csv](figures/data/FIG-15.csv) |
| FIG-16 | Official benchmark feasibility: Meta-World | benchmark | rendered | [FIG-16.png](figures/light/FIG-16.png) | [csv](figures/data/FIG-16.csv) |
| FIG-17 | LLM weight programming versus conventionally trained networks | baselines | rendered | [FIG-17.png](figures/light/FIG-17.png) | [csv](figures/data/FIG-17.csv) |
| FIG-18 | Robustness to out-of-distribution shift (OOD curves) | baselines | rendered | [FIG-18.png](figures/light/FIG-18.png) | [csv](figures/data/FIG-18.csv) |
| LLM-01 | Claude arms: hidden ID gain by checkpoint class | llm | rendered | [LLM-01.png](figures/light/LLM-01.png) | [csv](figures/data/LLM-01.csv) |
| LLM-02 | Decomposing the gain: valid patch × improvement × size | llm | rendered | [LLM-02.png](figures/light/LLM-02.png) | [csv](figures/data/LLM-02.csv) |
| LLM-03 | Cost versus gain | llm | rendered | [LLM-03.png](figures/light/LLM-03.png) | [csv](figures/data/LLM-03.csv) |
| LLM-04 | Gain across editing rounds | llm | rendered | [LLM-04.png](figures/light/LLM-04.png) | [csv](figures/data/LLM-04.csv) |
| LLM-05 | Where editing fails: patch status per round | llm | rendered | [LLM-05.png](figures/light/LLM-05.png) | [csv](figures/data/LLM-05.csv) |

## 图注

- **FIG-01 WSP-8 evaluation pipeline**：WSP-8 评测流程：密封基线检查点 → 编辑者读权重文本并回复一个 edits 块 → 确定性应用 → 开发集反馈（循环 R 轮）→ 冻结最终权重 → 隐藏集（ID/OOD）配对评测。服务端强制预算、工具审计与哈希校验，失败留在分母。
- **FIG-02 Task formalisation and capability ladder**：WSArena 任务实例的形式化定义，以及五级能力 L0–L4 与检查点类别、实现状态的对应（✓ 已实现并测试，○ 计划）。
- **FIG-03 Layered failure attribution**：诊断式失败归因（参照 VABAR 的诊断执行流）：每个失败归到最早能解释后续结果的层——回合层（供应商状态）、补丁层（解析与应用）、效果层（行为变化与逐 episode 失败类型）。下方计数来自 588 个脚本对照 run 的 2,352 个回合。
- **FIG-04 Benchmark coverage matrix**：检查点覆盖矩阵：任务（行）× 检查点类别（列），格内为“可用数”，斜线数字为按 SR 带筛选未能构造出的不可行数；右侧给出观测/动作维度与参数量。
- **FIG-05 Start-state difficulty of the checkpoints**：未编辑检查点的隐藏成功率（实心=ID，空心=OOD；每点一个检查点，短横线为均值）。S1/S2 起点约 0.3–0.55，S3 约 0.7–0.9；OOD 起点整体低得多——这是留给编辑的空间。
- **FIG-06 Control arms: hidden ID gain**：脚本对照臂在隐藏 ID 上的成功率增益 ΔSR（点=检查点，粗线=按检查点聚类的 bootstrap 95% CI，灰带=判定带宽 ±0.10）。闭式解在 S1 上≈神谕；S2 上不稳；同等反馈预算的随机搜索≈0。
- **FIG-07 Control arms: hidden OOD gain**：同 FIG-06，但为隐藏 OOD 上的增益。OOD 偏移已在 select 命名空间上校准；c02-avoid 的 S2 起点 OOD 成功率仍为 0（该子集无分辨力）。
- **FIG-08 Gain by task, class and control arm**：ΔSR（ID）热图：行=任务×类别，列=对照臂；蓝=提升，红=下降，灰=0；格内为均值，括号为检查点数。可以读出：解析岭回归在 c03-hold 的 S2 上显著为负（−0.36）。
- **FIG-09 Recovery versus Regression**：可靠性：Recovery=原本失败的 episode 被修好的比例；Regression=原本成功的 episode 被破坏的比例。理想点在左上角。闭式解在 S2 上同时高 Regression（0.50），而 S1 上接近理想。
- **FIG-10 Evaluation noise floor**：噪声尺度：同一检查点在 5 个互不相交的 200-episode 集合上的成功率标准差（每点一个检查点；虚线=二项上界 0.035）。判定带宽 δ=0.10 高于 2σ 的典型值，故安全。
- **FIG-11 OOD shift calibration**：OOD 偏移强度校准（只用 select 命名空间，不碰隐藏集）：横轴为偏移强度 s（0=同分布，1=v1 的最大偏移），纵轴为 S2∪S3 检查点的平均成功率；竖线=选定值，实线=ID 基线。规则：取使 OOD 成功率 ≥0.20 且 ID−OOD ≥0.15 的最大 s。
- **FIG-12 What the unedited policies get wrong**：未编辑检查点在隐藏 ID 上的失败类型构成（占失败 episode 的比例，规则分型，不是编辑者自评）。S1/S2 以 stalled_far / drift_far / collision 为主，说明故障是可诊断的。
- **FIG-13 Edit footprint versus gain**：补丁足迹：每个 run 一个点，横轴为改动参数占比（对数），纵轴为隐藏 ID 的 ΔSR。神谕只改约 1% 参数就得到最大增益；随机扰动改动最多且有害。形状=检查点类别。
- **FIG-14 Case study: one planted fault, three outcomes**：案例（c01-reach，S1 weak_axis 植入故障，隐藏 ID 前 12 个 episode）：上排=三个策略的轨迹（○起点，◆目标；实线=成功，×=失败）；下排=补丁相对基线的输出层 W3 差异。神谕只改被腐蚀的那一列；闭式解重写整个输出层。
- **FIG-15 Validity and reproducibility record**：工程与可复现性记录：关键计数（检查点、对照臂 run、独立复算、测试）与契约版本时间线（每次修订的原因，旧结果保留为证据）。
- **FIG-16 Official benchmark feasibility: Meta-World**：官方基准可行性（Meta-World）：(a) 行为克隆策略成功率随训练轮数——reach 迅速饱和，push 有可利用的过渡区；(b) 已构造的 Meta-World 检查点在 select 上的成功率与目标区间。
- **FIG-17 LLM weight programming versus conventionally trained networks**：效率前沿：横轴为墙钟时间（对数），纵轴为隐藏 ID 的 ΔSR。灰色=训练参照（方块=专家标签的行为克隆微调，三角=无标签的进化策略；点=各检查点均值）；彩色=Claude Pilot 的 run（蓝=Sonnet，橙=Opus，实心=high）。训练按 CPU 秒计，LLM 按提供方延迟计；训练便宜约 3–4 个数量级（金钱），但需要专家标签，而无标签的 ES 在同等交互预算下远不如 LLM。
- **FIG-18 Robustness to out-of-distribution shift (OOD curves)**：OOD 曲线（A7）：横轴为 OOD 偏移强度 λ（0=同分布参数，1=校准的 OOD 划分，2=两倍偏移），纵轴为编辑后的隐藏成功率（实线），灰色虚线=未编辑基线。同一批隐藏 OOD episode 上评测。基线随偏移崩溃，所以增益随偏移变大要结合基线读；Opus 的收益在 λ≈1 最高，Sonnet 在 S1 上随偏移单调下降。
- **LLM-01 Claude arms: hidden ID gain by checkpoint class**：【待数据】Claude 四个臂（Sonnet/Opus × medium/high）在隐藏 ID 上的 ΔSR（点=检查点，粗线=聚类 bootstrap 95% CI）；虚线=闭式解（特权）均值，灰带=判定带宽。
- **LLM-02 Decomposing the gain: valid patch × improvement × size**：【待数据】把 ΔSR 分解为 P(提交合法补丁)、P(改善 | 合法)、E[ΔSR | 合法]，区分“没写出补丁”“补丁无用”“补丁有用”；推理强度是否单调看这三项而不是只看总分。
- **LLM-03 Cost versus gain**：【待数据】每个 run 的输出 token（对数）与 ΔSR：同一增益下谁更省；高推理强度是否把预算花在了思考而不是补丁上。
- **LLM-04 Gain across editing rounds**：【待数据】开发集成功率随编辑轮次的变化（0=未编辑的初始报告；均值与 95% CI）：对应 VABAR 的“零样本 vs 两轮反馈后”，这里给出每一轮的曲线。
- **LLM-05 Where editing fails: patch status per round**：【待数据】每个臂各轮次的补丁状态构成（合法且改变权重 / 合法但未改 / 空块 / 无块 / 全部被跳过）：失败留在分母里，并按“最早的解释层”归类。

## 主研究图（2026-11-08，`scripts/make_main_figures.py`；浅色/深色，png/pdf/svg，数据表 `figures/data/MAIN-0x.csv`）
| ID | 标题 | 图 |
|---|---|---|
| MAIN-01 | 每个点是一个检查点：各编辑者把隐藏成功率提高了多少（Tier-0 与 Meta-World，含对照臂） | [png](figures/light/MAIN-01.png) |
| MAIN-02 | 隐藏权重（盲读）对增益的影响：8 个检查点 × 2 个模型的配对线 | [png](figures/light/MAIN-02.png) |
| MAIN-03 | 同一检查点两次独立运行的增益散点（run-to-run σ） | [png](figures/light/MAIN-03.png) |
| MAIN-04 | 逐检查点：编辑者对“单旋钮增益搜索”的胜负热图 | [png](figures/light/MAIN-04.png) |
| MAIN-05 | 为什么需要探针赛道：权重文本所需 token 随宽度的增长 | [png](figures/light/MAIN-05.png) |
