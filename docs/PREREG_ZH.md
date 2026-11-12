# WSArena：LLM 权重空间编程能力基准（Claude 先行） 预注册 v1（**§11 于 2026-11-07 冻结，早于任何主研究输出**）

> **状态：§1–§10 为设计底稿；§11 为冻结的主研究（Phase 1, Tier-0）规格，冻结提交早于首个主研究输出（核对 `git log` 与 `runs/main-v1-*` 的创建时间）。** 标 `‹待E3›`/`‹待E4›` 的数字要等成本冒烟/Pilot 实测后才能填；填完、加上分析脚本后单独 commit（需用户批准本地 git），早于任何主研究输出，用 `stat -c %w` 核对。修订只能在任何预测之前，并保留旧哈希清单。
> 先于本文件的输出：Tier-0 脚本基线（无额度；只用于校准任务带与验证 harness，不含任何 LLM 输出）。

## 1. 主问题与主比较
- **主问题**：在同一固定协议（WSP-Native，R 轮，固定反馈与预算，禁止训练/搜索）下，Claude 的 Sonnet 5.5 与 Opus 5.5（medium / high）能否在 Tier-0 控制任务上**可靠地**修复故障策略（L2，S1/S2）与增强正常策略（L3，S3），其增益相对等资源下最强的简单替代有多大？
- **主终点**：隐藏 ID 的 ΔSR（编辑后−基线，同一批 200 个隐藏 episode 配对），Final-Round 轨道，检查点为聚类单位。
- **主要比较（≤3 个，Holm 校正）**：①每臂 L2（S1+S2）ΔSR 相对 `random` 臂；②每臂 L2 相对 `analytic_ridge`；③L3（S3）与 L2 的 ΔSR 差。其余全部标“探索性”。
- **资源对齐表（K-S3-02）**：

| 臂 | 读取量 | 计算 | 训练预算 | 交互预算 | 备注 |
|---|---|---|---|---|---|
| Sonnet 5.5 / Opus 5.5 × medium/high | 全部权重文本（约 5k token）+ 反馈 | 模型自身思考（不可见量，记 CLI 报告的输出/思考 token） | 0 | R 轮 × 16 dev episode + 初始 16 | 无工具；不可选探针 |
| `null` | 0 | 0 | 0 | 同（但不使用） | 零假设下限 |
| `random`（稀疏高斯扰动，5% 参数，σ=0.5·std） | 0 | 0 | 0 | 同 | 排除“任意扰动也有增益” |
| `oracle_restore`（仅 S1） | 特权真值 | 0 | 0 | 同 | 上界，不参与排名 |
| `gain_greedy` | 0 | 0（每轮缩放 W3 一个标量） | 0 | 同（用开发集反馈做贪心接受） | **强简单替代**：S2 上 +0.27，LLM 必须先超过它 |
| `analytic_ridge` | 特权专家标签（200 条演示） | 闭式解 | 0（无梯度） | 同 | **Analytical 赛道**，单列，不与 Native 混排 |

- **等资源下最强的简单替代**：`gain_greedy`（单参数、同反馈预算，按类别取最强者）、`analytic_ridge`（特权闭式重拟合，只作参考）与 `random`。LLM 若不显著优于它们，则“LLM 的权重空间编程”这一说法在本协议内不被支持。

## 2. 数据、模型、harness
- 任务：Tier-0 `c01-reach, c02-avoid, c03-hold, c04-delay`（机制层）；官方基准层另行预注册（Kinetix/OCBench/Meta-World）。头条结论只能来自官方基准层（K-S4-04）；Tier-0 的结论限定为“机制层、玩具”（K-S2-03：玩具只检验推理，不为方向背书）。
- 检查点：`configs/tasks/ckpt_manifest.json`（冻结时的 sha256 清单），按 S1/S2/S3 分类；不可行组合如实排除并列出。
- 身份绑定：检查点 sha256、`code_version`（源码树哈希）、配置哈希、模型 id、effort、CLI 版本、系统提示哈希；缺项拒绝汇总。
- 模型与 effort：`claude-sonnet-5-5`、`claude-opus-5-5` × `--effort medium|high`；允许值与行为以隔离探针（E2）结果为准。输出上限（Pilot 后的候选）`max_output_tokens=64000`（Pilot 最大输出 28,556 token，逼近 32,000 上限；CLI 接受 64,000 与 128,000，但无法从返回值证实一定生效），并做 32k 与 64k 的灵敏度子实验（H2 辅助）。
- R=4（Pilot 候选：模型在开发集达标后提交空块，R=4 已够；更多轮数只放进消融）；每轮 16 dev episode；隐藏 ID 200 + OOD 200，种子命名空间 `hid_id`/`hid_ood`。

## 3. 噪声尺度与判定带宽（K-S3-05）
- 评测完全确定（给定权重与种子）。噪声来自 episode 抽样与 LLM 运行间方差。
- `noise_floor`：在 5 个互不相交的 200-episode 集合上对每个检查点测 SR 标准差 σ_ep：53 个检查点（契约 v6）× 5 个互不相交 200-episode 集合，ID 中位数 0.030、最大 0.060，OOD 中位数 0.028（`runs/SUMMARY/noise_floor.json`）；二项上界 √(0.25/200)=0.035。
- LLM 运行间方差：同一检查点同一臂重复运行的 ΔSR 标准差 σ_run = ‹待补：Pilot 每格仅 1 次，无法估计；主研究必须有 ≥2 次重复›。
- **判定带宽 δ = max(2σ_ep, 预注册最小关注效应 0.10)**（0.10 来自章程 §1 最小关注效应）；|ΔSR|<δ 的单个 run 记“无可分辨变化”。
- 数值地板（K-S3-06）：运行精度 float64；评测对权重的扰动敏感性在 1e-9 量级以下不改变 episode 结果（`test_reparameterize_preserves_function` 验证等价变换误差 <1e-10）；判定精度 = 0.005（SR 报告到小数点后三位，不低于 1/200）。

## 4. 价值口径（K-S3-03）
- **A 无条件方法收益**：不适用（本项目不提出方法）。
- **A′ 有条件收益**：某臂在 L2 上相对 `random` 与 `analytic_ridge` 的主终点聚类 CI 下界 > 0，且 Regression 不显著高于 Recovery 收益（选择性约束）→ 记“在 Tier-0 机制层可靠修复”（前提：官方基准层待验证）。
- **B 未测**：官方基准、大模型、GPT 侧（Codex 另报）→ 记“未知”。
- **C 认识价值**：协议、对照臂、失败分型、成本曲线本身；即使全 KILL 也产出（零结果报告 T1）。
- 全部不成立时：不再追加 Claude 侧额度；写零结果报告并留出“开放门”触发器（如 MIT benchmark 发布、新模型）。

## 5. 结果类别（穷尽）× 实践含义 × 固定措辞
| 读数 | 类别 | 改变哪项实践 | 固定措辞 |
|---|---|---|---|
| H1 某臂 L2 ΔSR CI 下界 > 0，且优于 random 与 analytic_ridge | 修复成立 | 可把该臂当“神经网络调试器”使用，进入官方基准层 | “在本协议、Tier-0 机制层上，该臂的修复增益优于最强简单替代；不外推到官方基准或大模型。” |
| H1 CI 含 0 或不优于 random/analytic | 修复不成立 | 不把转述的单次成功当能力证据 | “未发现该臂优于简单替代；不能推出不存在。” |
| H1 优于 random 但不优于 analytic_ridge | 部分（可由解析替代） | 命名为 Neural Circuit Synthesis / 解析可复现 | “增益可被闭式求解复现，不构成 LLM 特有能力的证据。” |
| H3 S3 的 ΔSR>0 且保留约束成立 | 增强成立（重要正结果） | 值得投入官方基准层的增强实验 | “在 Tier-0 的正常策略上观察到稳定增强；机制层结论。” |
| H3 S3 的 ΔSR 与零无差，或 Regression 抵消 | 增强不成立 | 把 LLM 视为修复器而非优化器 | “未观察到对正常策略的稳定增强。” |
| H2 high 在三项分解上均不低于 medium | “非单调”KILL | 默认使用高档 | “未发现推理强度非单调；仅限本模型/本任务。” |
| 方向相反（Δ<0 且 CI 上界 <0） | 有害 | 警示：编辑使策略更差 | “该臂的平均编辑使隐藏成功率下降。” |
| run 被判 invalid / 大量中断 / 完成率 <50% | 不可估计 | 修工程后重跑（新 attempt，保留旧分母） | “运行完成率不足，主终点不可估计。” |
| 不属于任何类别 | 未分类 | 无 | “无可主张的类别结论。” |

## 6. 停止规则（K-S3-07）
- 臂集：4 个 LLM 臂 + 脚本臂（null、random、random_greedy、gain_greedy、analytic_ridge、oracle_restore）；n_max（run 数）= ‹待用户额度决定›。Pilot 实测单 run 平均 $0.33（Sonnet 约 $0.18，Opus 约 $0.48）；候选：最简扩展档 ≈32 run（≈$11–16）、精简档 144 run R=4（≈$48，R=6 约 $70，按批 1 的额度/成本比估算超过一周额度，须分多周）；每检查点重复数 ≥2。
- 主研究方案候选 A/B/C 见 `docs/MAIN_STUDY_PLAN_ZH.md`（推荐 B：16 聚类、80 run、≈$26、MDE≈0.175；待用户选择后填入本节并冻结）。
- Pilot 对任务选择的约束：c01 已对 Claude 饱和（所有臂≈1.00）→ 只作健全性检查，不进主要比较；主要比较放在 c02/c03/c04/Meta-World 与 S1/S2/S3。
- 功效：用 Pilot 的 σ_run、σ_ep 做聚类 bootstrap 功效估计（模拟脚本随预注册一并提交）；达不到“检出 0.10 的差、功效 0.8”时，不增加 run，而是缩小主要比较并如实写“低功效”。
- 额度停止：5h 窗口 ≥80% 不派新批；周 ≥70% 仅收尾；≥85% 全停（全局 CLAUDE.md）。任一触发 → 保存+记 QUOTA_LOG，不改参数续跑。
- 审查停止：一轮审查只剩措辞级修订 → 停；连续两轮新阻断 → 先提交停止规则再算可行性。
- 决策路径：Pilot→（成本外推可行?）→是：冻结预注册→主研究；否：改最简档并如实声明低功效。

## 7. 态势评估预备稿（K-S3-04）
- 若 L2 成立而 L3 不成立：报告“LLM=调试器”，下一步官方基准层（Kinetix car→OCBench）只做修复。
- 若全不成立：零结果报告，开放门触发器；不换任务/改判据补救（T2）。
- 若 L2、L3 都成立：优先检查对照臂（analytic_ridge 是否同样成立）与 PSI（H4），防止“解析可复现”被误读。
- 若 Sonnet≈Opus 或 medium≈high：成本效率成为主要发现（H2）。
- 判定后只填数；新增分支注明“事后”。

## 8. 盲化、隔离与查看纪律（K-S3-12、K-S4-06）
- 编辑者看不到：隐藏集、检查点类别、真值、OOD 参数（见 `test_dev_budget_accounting_and_hidden_never_in_prompt`、`test_prompt_identical_across_classes`）。
- 冻结前不对真实模型做前向（探针/成本冒烟除外，且其输出不入主分析）。
- 运行期间监控只输出计数与耗时；查看中间回复/补丁写入 `runs/VIEW_LOG.tsv`（谁、何时、看了什么）。
- 子代理仅用于只读调研与独立复算，不读取模型回复原文之外的隐藏信息。

## 9. 判定器规格（K-S5-01）
- 严格：run.json 缺身份字段 → 拒绝汇总（`analysis.metrics.run_row` 抛错）；`tool_use` 事件 → run failed；检查点哈希不符 → 拒绝启动（`IntegrityError`）。
- 解析：edits 块解析规则固定于 `wsarena/edits.py`；跳过语句计数并回传；无 `eval/exec`。
- 重复/冲突/坏行：同一 run-id 目录已存在 verified/failed → 运行器不重跑；中断仅从保存回合重放。
- 每类失败配合成反例：见 `tests/test_protocol.py`（中断续跑、工具审计、无块、预算耗尽）、`tests/test_edits.py`、`tests/test_store_envs.py`。
- 独立复算（K-S5-02）：`checks/` 下不导入 wsarena 的重算脚本从原始输出（补丁→权重→episode）复算 ΔSR；‹待做›，由独立子代理或另一会话完成，结果并列呈现。

## 10. 偏离表（只追加）
| 日期 | 偏离 | 原因 | 方向 | 披露位置 |
|---|---|---|---|---|
| 2026-11-06 | 开发集总量 16+R×16（R=8 时 144），任务书写 ≤128 | 初始报告单独计，每轮预算一致 | 放宽（+16） | DESIGN §3、§10 |
| 2026-11-06 | 首期仅 Claude；调用走订阅 `claude -p`；GPT 侧交 Codex | 用户原话 | 范围缩小 | CHARTER §6、decisions U1/U2/U6 |
| 2026-11-06 | 先 Tier-0 机制层，官方基准层后置 | 官方基准依赖未装；先验证协议与成本 | 范围缩小 | DESIGN §10 |
| 2026-11-06 | S1 从最轻腐蚀档起搜 | 目标 SR 带难以直接命中 | 构造细节披露 | DESIGN §5.2、ckpt_manifest.json |
| 2026-11-06 | **v1→v2：S3 由“弥散噪声”改为自然训练；OOD 偏移在 select 上重新校准（网格下端细化）；v1 基线标 SUPERSEDED** | v1 的 S3 可被闭式去噪（非健康策略）；v1 的 OOD 把 S2 压到 0 | 加严（去掉人为构造；增加分辨力） | CONTRACT_CHANGELOG v6、EXPERIMENT_PLAN §8 |

## 11. 主研究 Phase 1（Tier-0）冻结规格（2026-11-07；方案 B 的落实；用户授权“按推荐方案做，不再就技术路线征求意见”）
**检查点**：只用从未被任何 LLM 看过的种子 2、3（核对：此前全部 LLM 运行——pilot-v0/v1、smoke、A3-blind、A4——只触及种子 1；种子 1 一律视为探索性，不进主分析）。16 个聚类：
c02-avoid {s1-weak_axis, s2, s3}×{seed2, seed3}（6）；c03-hold 同（6）；c04-delay {s1-weak_axis, s3}×{seed2, seed3}（4；c04 的 S2 不可行，见清单）。S1 只取 weak_axis 一种腐蚀以保持聚类同质，其余腐蚀类型列入探索性后续。清单文件：`configs/experiments/main-v1-t0.yaml`。
**协议**：WSP-Native，R=4，每轮 16 dev episode，初始报告 16，隐藏 ID 200 + OOD 200，`max_output_tokens=64000`，0 重试；契约 v11（`check_frozen.py` 必须 OK）。
**臂**：Sonnet 5.5·medium/high、Opus 5.5·medium/high（64 个 run，`main-v1-t0`）。
**附加批次（同一冻结，独立 exp_id）**：`main-v1-t0-rep`（seed2 的 8 个聚类，Sonnet·med 与 Opus·med 各再跑 1 次，估 σ_run，16 run）；`main-v1-t0-blind`（同 8 个聚类，A3 盲读，16 run，算 WRI）。
**顺序**：`order: class_interleave`（种子→类别→任务→检查点→臂），任何前缀都跨类别；每批 ≤8 个 run、串行；批前后记 `QUOTA_LOG` 读数；5h ≥80% 不派新批。
**对照臂映射（在任何 LLM 输出前固定）**：`random`=`random5`；`random_greedy`=`greedy_f02_s01`（两个贪心变体里在非 LLM 基线上平均 ΔSR 更高者：L2 0.007 vs −0.010；L3 −0.031 vs −0.117；种子 2、3）；另有 `gain_greedy`、`tensor_gain_greedy`、`analytic_ridge`（特权）、`oracle_restore`（特权，仅上界）、`null`。
**主要比较（Holm，12 个检验 = 4 臂×3）**：C1 L2(S1+S2) 的隐藏 ID ΔSR 对 `random`；C2 L2 对 `analytic_ridge`；C3 ΔSR(S3) − ΔSR(L2)。聚类 = 检查点；C1/C2 用分任务分层的配对聚类 bootstrap CI + 符号翻转检验，C3 用任务内置换 p 与分层 bootstrap CI。次要比较（Holm，12 个检验）：L2 对 `gain_greedy`、`tensor_gain_greedy`、`random_greedy`。其余一律标“探索性”。分析脚本 `scripts/analyze_main.py` 与本节同一提交冻结。
**A3/A4 结果带来的约束（pilot 种子 1，探索性，已见数据）**：盲读与看权重的 ΔSR 平均差 −0.007（12 对），重参数化 PSI 平均 0.040；说明 Tier-0 增益大量来自不依赖读权重的整体缩放/增益类编辑。因此：①所有结果按编辑机制分层报告（`wsarena.analysis.mechanism`），不只给合并数；②任何“读懂了权重”类的表述须由 `main-v1-t0-blind` 的 WRI 支持，否则只写“编辑改善了策略”；③Tier-0 头条仅限机制层（K-S4-04）。
**功效**：16 聚类，σ_diff≈0.25 时 MDE≈0.175（MAIN_STUDY_PLAN §2）；达不到的比较如实写“低功效”，不追加 run 补功效。
**输出上限灵敏度（32k 对 64k）与 Meta-World Phase 2**：另行在首个 Phase 1 批次之后、Phase 2 首个 run 之前追加预注册小节（Phase 2 先做 4 个 run 的可行性探针，不入分析）。
**KILL/停止**：同 §6；完成率 <50% → “不可估计”，不追加调用。运行期间不看中间回复；确需查看写 `runs/VIEW_LOG.tsv`。

## 12. 输出上限灵敏度（P2；2026-11-07 冻结，早于其首个 run）
- 目的：Pilot 最大输出 30,042 token 逼近 32k 上限，主研究用 64k；检验上限是否改变结论。
- 设计：`configs/experiments/main-v1-cap32k.yaml`：4 个种子-2 聚类（c02 S1、c02 S2、c03 S2、c04 S1）× Sonnet·med/Opus·med，`max_output_tokens=32000`，8 个 run；对照为 `main-v1-t0` 里同聚类同臂的 64k run（配对）。
- 读数（只此两项，均为描述性）：①因 `output_budget_exhausted` 失败的回合数（32k 对 64k）；②同聚类同臂的隐藏 ID ΔSR 差（64k − 32k）。判读：任一臂 |均差| ≥0.10 或 32k 下有 ≥2 个回合被截断 → 在报告中把 Pilot（32k）标注为“可能受截断影响”；否则一句话披露。不做显著性检验（n=8，低功效，已声明）。
- 额度：≈\$2.5；额度不足时延期，不改设计。

## 13. 主研究 Phase 2（Meta-World 官方基准层）冻结规格（2026-11-08；早于其首个主运行）
**依据**：4-run 可行性探针（`mw-probe-v1`，种子 1，不入任何分析）：4/4 verified，0 回合截断，泄漏审计通过，单 run 输出 1.2k–2.8k token，约 2.5 个 5h 额度点/run（与 Tier-0 同量级）。探针结果的效应大小未被用于任何设计选择。
**检查点（种子 2、3，从未被 LLM 看过）**：mw-push {S1 weak_axis, S2, S3}×{2,3}（6）；mw-pickplace {S1: seed2 用 bias_shift（seed2 无 weak_axis）、seed3 用 weak_axis；S2；S3}×{2,3}（6）；mw-peginsert {S1 weak_axis×{2,3}；S2 仅 seed3（seed2 不可行）；S3×{2,3}}（5）。共 17 个聚类，L2=11、L3=6。清单：`configs/experiments/main-v1-mw.yaml`。
**协议与臂**：同 §11（R=4，每轮 16 dev，隐藏 ID/OOD 各 200，`max_output_tokens=64000`，契约 v11）；臂 = Sonnet 5.5·medium、Opus 5.5·medium（Phase 1 中 high 档无稳定优势且成本更高，Phase 2 不再重复该比较）。34 个主运行；另 `main-v1-mw-rep`（seed-2 的 8 个聚类 × 2 臂 = 16 运行，估 σ_run）。顺序 `class_interleave`，串行，额度由 `configs/quota_limits.json` 与自动读数管理（U11/U12）。
**主要比较**：同 §11 的 C1/C2/C3，每个臂 3 个检验，Holm 家族 = 2 臂×3 = 6 个检验；次要比较：L2 对 `gain_greedy`、`random_greedy`（`tensor_gain_greedy` 在 Meta-World 无基线，不比）。控制臂映射同 §11（`random`=`random5`，`random_greedy`=`greedy_f02_s01`）。分析脚本 `scripts/analyze_main.py --llm main-v1-mw --seeds 2,3`（`LLM` 列表只含出现的臂）与本节同提交冻结。
**与 Phase 1 的合并**：不合并成一个总数；Phase 1 与 Phase 2 并列报告，并报告方向是否一致（一致才写“跨层稳健”）。
**预先说明的局限**：Meta-World 的 S1 腐蚀类型在 seed2 的 pickplace 上与其他格不同；peginsert 的 S2 只有一个聚类；n 小，MDE 约 0.2。

## 14. P5 规模阶梯冻结规格（2026-11-08；早于其首个运行）
**问题**：“读权重”的收益是否随网络变宽而出现或消失？Phase 1 与 A3 显示在宽度 24（c02：约 0.8k 参数）上权重依赖度平均≈0（WRI +0.006 [−0.088, +0.099]，种子 2，c02 子集见 A3_BLIND_CAP.md）。
**任务与宽度**：c02-avoid，隐藏宽度 24（已有数据）、64（约 4.7k 参数，首条提示约 1.8 万 token）、128（约 17.7k 参数，约 6 万 token）；`c02-avoid@wN` 与基础任务评测完全相同的 episode（`seed_for` 去掉 `@w…` 后缀；有检验）。检查点：每个宽度 {S1 weak_axis, S2, S3} × 种子 {2, 3} = 6 个（BC 构造，按 select 命名空间的 SR 带筛选；全部 built，无不可行项），构造脚本 `make_tasks.py --tasks c02-avoid@w64 c02-avoid@w128 --seeds 2 3`。宽度 24 的对应数据 = Phase 1 里 c02 的 6 个聚类（Sonnet·med / Opus·med 的主运行与重复运行）。
**对照臂**：`baselines-scale-v1`（脚本臂，180 个运行，无额度）：null、random5、greedy_f02_s01、gain_greedy、tensor_gain_greedy、analytic_ridge、oracle_restore，同协议同记账。
**LLM 臂**：Sonnet 5.5·medium、Opus 5.5·medium；协议同 §11（R=4，64k 输出上限，0 重试）。`scale-v1-w64`、`scale-v1-w128`（完整权重，各 12 个运行）与 `scale-v1-w64-blind`、`scale-v1-w128-blind`（盲读，各 12 个运行）；共 48 个运行。
**读数（均为描述性，n 小；每个宽度 6 个聚类）**：①每个宽度、每个臂的 ΔSR（隐藏 ID）均值与聚类 bootstrap CI；②WRI(w) = ΔSR(完整) − ΔSR(盲读)，宽度 24/64/128 三点并列（宽度 24 取 c02 的 6 个聚类，完整权重取主运行与重复运行均值）；③控制臂同一读数；④输入 token 与每运行额度点；⑤有效补丁率、截断回合数、编辑机制分布。**不做假设检验**，只报告点估计与区间；判读规则预先写定：WRI 随宽度单调上升且 128 的 CI 下界 >0 → “权重依赖度随规模出现”；三点都含 0 → “在测得范围内未见权重依赖”；其余 → “无单调趋势，不下结论”。
**顺序与额度**：先盲读（便宜），再 w64 完整，最后 w128 完整；串行，额度由 `configs/quota_limits.json`（5h 90%、周 80%）自动管理；w128 的单运行成本未知（输入约为宽度 24 的 14 倍），若周额度触线则停在 w128 并等周重置，不改设计。
**局限（预先声明）**：仅一个任务（c02）；宽度 128 仍远小于真实策略网络；n=6/宽度；Sonnet·med 与 Opus·med 两个臂；BC 检查点的 S1/S2/S3 带与宽度 24 同规则但具体网络不同。

| 2026-11-08 | **C3 的点估计与 CI/p 的估计量不一致（Codex 复核 REVIEW-01 发现）**：`analyze_main.py` 的 c3() 报告的 `diff` 是合并均值差（L3 聚类均值−L2 聚类均值），而其 bootstrap CI 与置换 p 建立在“任务等权”的任务内差上（§11 写的是任务分层）。已用同一重抽样/置换方案并列计算两种估计量（`scripts/analyze_c3_check.py`，`reports/C3_ESTIMAND_CHECK_T0.md`/`_MW.md`）：点估计差 ≤0.014，p 值同量级，所有 C3 结论的方向与显著性不变。冻结的旧报告原样保留，不替换；此后 C3 同时报告两种估计量，以任务等权为预注册估计量。 | 修正说明（不改结论） | `docs/PREREG_ZH.md` §10、reports/C3_ESTIMAND_CHECK_*.md |

## 15. P6 探针赛道冻结规格（2026-11-08；早于其首个主批运行；可行性探针 `probe-feas-v1` 的 4 个运行不入分析）
**问题**：网络大到完整权重放不进提示词（c02 宽度 256 约 24.6 万 token、宽度 512 约 98 万 token），模型只靠只读探针能否完成编辑？探针比盲读（同样看不到权重值、也不能探针）多出多少？
**实现**：`wsarena/probe.py`、`wsarena/llm/probe_gateway.py`（契约 v13；既有文件字节不变）。提示 = `prompt_variant: blind` 的首条消息，其中盲读说明被探针说明替换；每轮最多 3 个探针子回合，每子回合最多 4 条指令，全程返回总量上限 8,000 个数；`forward` 只调用网络本身，探针不能运行环境、训练或搜索；探针结果只含当前权重的函数。
**任务与检查点**：c02-avoid@w256、@w512，{S1 weak_axis, S2, S3}×种子{2,3}（每宽度 6 个，共 12 个；全部 built；契约 v14）。同一 episode（`seed_for` 去后缀）。
**臂**：Sonnet 5.5·medium、Opus 5.5·medium，各在两条赛道：`probe`（`probe-v1-w256/w512`，各 12 运行）与盲读对照 `blind`（`probe-v1-w256-blind/w512-blind`，各 12 运行，无探针）；共 48 个运行。协议同 §11（R=4，64k 输出上限请求，0 重试）。
**对照臂（脚本，`baselines-probe-v1`，无额度）**：null、random5、greedy_f02_s01、gain_greedy、tensor_gain_greedy、analytic_ridge（特权）、oracle_restore。
**读数（描述性，n=6/宽度）**：①每宽度每臂 ΔSR（隐藏 ID/OOD）与聚类 bootstrap CI；②**探针增量** = ΔSR(probe) − ΔSR(blind)（聚类配对）及 CI；③探针使用（子回合数、指令数、返回数值量、各类指令占比，来自 `probe_log.jsonl`）；④编辑机制分布；⑤成本与输出 token；⑥截断回合数（按输出上限政策）。**预先写定判读**：探针增量的聚类均值 CI 下界 >0 → “探针提供了可测收益”；CI 含 0 → “未见收益”；上界 <0 → “探针有害”。**不做假设检验。**
**赛道规则**：探针赛道独立排名，不与 Native 混排；两条赛道的数字并列只为对照。
**局限（预先声明）**：仅一个任务（c02）；两个臂、仅 medium；n=6；探针预算（3/4/8000）是人为设定，结果依赖该预算；宽度 512 的 BC 检查点 S1 腐蚀位置极稀疏，探针发现腐蚀位置的难度随宽度上升。
**顺序与额度**：主批在周额度重置后（13 日 14:00）按额度政策自动推进；先跑可行性探针，只检查“循环是否正常、每运行成本、截断”，不看效应大小。
| 2026-11-08 | **独立复核 P8 后的修正（不改任何冻结的分析与数字）**：①论文中只把 C1–C3 与次要比较族称为预注册；WRI、σ_run、尝试率明确标为探索性（F-T1）。②表中的 CI 为任务分层（条件于任务），另给未分层聚类 CI 与精确符号翻转 p（`reports/REVIEW_P8_ADDENDUM.md`）（F-S1/S3）。③新增 `scripts/check_completeness.py`，在分析前断言运行数、状态、聚类数、Holm 家族大小（F-S2）。④措辞按 F-C1–C7 收紧（见 `reports/INDEPENDENT_REVIEW_P8.md`）。 | 修正说明（不改结论） | reports/INDEPENDENT_REVIEW_P8.md |
