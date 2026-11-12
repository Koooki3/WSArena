# WSArena 技术设计 v0.1（与代码一致；2026-11-06）

> 每一节标注实现状态：**[已实现+测试]**、**[已实现，待登录后验证]**、**[计划]**。设计来源：用户粘贴的任务书（摘要见 survey/RELATED_WORK_ZH.md）、MIT 日志仓库 harness（一手）、质料库经验。与任务书不同之处在 §10 逐条列出。

## 1. 要回答的问题与能力分层
五级能力独立评分，**不给总分**：

| 级 | 问题 | 本版实现 |
|---|---|---|
| L0 读取 | 能否从原始参数推断结构与功能 | [计划] 开放问答轨道（MIT exp1 式），Pilot 不含 |
| L1 合成 | 不训练，从零写出可用网络 | [已实现] S0（全零）检查点，同一协议 |
| L2 修复 | 识别并修复明确故障 | [已实现] S1（植入，有真值）、S2（自然故障） |
| L3 增强 | 改善无单点故障的正常策略 | [已实现] S3（弥散型，SR∈[0.6,0.9]） |
| L4 泛化 | 编辑在未见环境/重参数化后仍有效 | [已实现] 隐藏 OOD 分裂；[已实现] `mlp.reparameterize`（tanh MLP 的精确等价变换）；PSI 分析 [计划] |

主实验 = L2；最大科学挑战 = L3；L0/L1 作能力诊断；L4 证明不是偶然参数排列。

## 2. 协议 WSP-Native（WEP-1）[已实现+测试]
θ_{k+1} = Apply(θ_k, P_k)，P_k 由 LLM 在看到权重文本、任务描述与反馈后写出；Apply 是确定性解释器。
- **模型侧语法**：MIT 的 `edits` 语言的兼容超集（`T[idx] = / += / -= / *= 值`，范围、步长、二维列表，另加 `RESET`）。只读**最后一个** `edits` 代码块；不可读语句被跳过并计数、其余照常应用；`|v|≤1e6`，必须是普通十进制数，**不用 eval/exec**（有“不会执行代码”的测试）；形状不可变。
- **规范化**：每轮保存补丁文本、解析结果（应用/跳过及原因）、相对上一轮与相对基线的差异（改动参数数、占比、L2、L∞）。检查点按 float64 规范哈希（sha256）。
- **三赛道**：Native（本版实现）；Analytical（解析求解器：`analytic_ridge` 对照臂；LLM 受限矩阵运算 [计划]）；Agentic（沙箱 Python [计划]）。**赛道不共享排名**。
- **不可作弊约束**：无运行时干预（只改登记的原始参数张量，架构/观测归一化/动作后处理不可改）；LLM 不能自己启动环境；梯度/优化器/搜索不可用；部署时无编辑 LLM。投影/方向消减一类编辑在分析里单独标注，不当作“电路合成”。

## 3. 轮次与预算（服务端强制）[已实现+测试]
| 项 | Pilot v0 | 主协议默认（预注册前可调） |
|---|---|---|
| 轮数 R | 4 | 8 |
| 每轮候选补丁 | 1 | 1 |
| 每轮开发评测 episode | 16（新鲜种子，每轮不同） | 16 |
| 初始报告 | 16 episode，含 1 条失败轨迹摘要 | 同 |
| 补丁修正重试 | 0 | 0 |
| 开发集总量 | 16+R·16 | 144（任务书写 128；因初始报告另计，已记偏离） |
| 前向探针 | 关（代码路径预留） | 可选（协议 §探针） |
| 回退 | `RESET`，占用该轮 | 同 |
| 隐藏测试 | ID 200 + OOD 200，**从不反馈** | 同 |
| 最终权重 | 最后一轮（Final-Round 主轨道）；开发集最优轮另列次轨道 | 同 |

种子命名空间 `envs.seed_for(task, split, i)`：`select`（检查点筛选，永不展示）/`dev`/`hid_id`/`hid_ood`，互不相交（有测试）。评测**完全确定**（给定权重+种子），所以“重复评测噪声”是 episode 抽样噪声，不是运行噪声（见 §8 噪声尺度）。

## 4. 反馈 [已实现]
每轮 JSON：`round, episodes_used, success_rate, mean_return, failures{规则分型}, diagnostic{mean_final_dist, mean_final_speed, action_saturation_rate, mean_action_jerk}, remaining_rounds`；失败分型由固定规则生成（`collision / wall / left_zone / overshoot_unsettled / near_unsettled / stalled_far / drift_far`），**不让被测模型自评**。初始报告多一条失败轨迹摘要（MIT Kinetix 实验同款信息量）。编辑解析摘要（读到几条、跳过几条）同时回传。

## 5. 任务与检查点 [已实现+测试]
### 5.1 Tier-0 任务族（自建，机制层；K-S4-04：头条不放这里）
2-D 质点力控，向量化、确定性。四级复杂度阶梯：

| 任务 | 级 | 观测 | 成功判据 |
|---|---|---|---|
| c01-reach | L2 | [e, v]（4） | 末步 \|e\|<0.08 且 \|v\|<0.2 |
| c02-avoid | L3 | [e, v, o]（6） | 同上且从未靠近圆形障碍 |
| c03-hold | L4 | [e, v, t/T]（5） | 最后 25 步全部在目标区内 |
| c04-delay | L5 | 5 步前的 [e, v] + 前两步动作（8） | 末步到达（真实状态） |

标称物理参数公开写入提示词；**OOD 参数（质量×、阻尼、推力、障碍半径、延迟+1）不披露**。网络 obs→24→24→2，tanh，float64（c01 为 866 参数；文本约 5k token）。任务阶梯映射任务书 H5 的 Level 1–5；Level 6（隐藏目标需记忆）需要循环结构，[计划]。

### 5.2 检查点四类（selection 命名空间上按 SR 带筛选）
| 类 | 构造 | 带 | 备注 |
|---|---|---|---|
| S0 | 全零 | — | L1 合成 |
| S1 | 健康 BC 策略 + 局部植入腐蚀（weak_axis/bias_shift/unit_sign_flip，**从最轻档起搜**） | 腐蚀后 SR∈[0.1,0.6]且降幅≥0.25 | 带张量/索引真值，可算定位精确率/召回率；神谕补丁=还原 |
| S2 | 在有缺陷示范（欠阻尼专家、无延迟补偿）或数据缺口上做 BC | SR∈[0.2,0.6] | 自然故障，可由失败分型诊断 |
| S3 | **自然训练**的健康策略：收敛的 BC（c02 的专家本身只有约 0.8），或 BC 训练过程中的欠训练快照（lr 1e-3，第 4–16 轮逐轮取第一个落入区间者）；**不再注入噪声** | SR∈[0.6,0.9] | 无单点故障；构造方法与轮次写入 meta。v1 的“弥散噪声”构造已废弃（可被闭式重拟合“去噪”，不算健康策略），旧清单归档为 `ckpt_manifest.v1.json` |

**c02-avoid 的专家上限约 0.81**（约 19% 碰撞来自障碍几何加执行器饱和；对 `k_rep/rep_radius/tang/kd` 做过 36 组扫描无法突破）：这是任务特性，报告时作为天花板，不再调专家。**OOD 偏移强度**由 `scripts/calibrate_ood.py` 在 `select` 命名空间上按预先写定的规则校准（最大的 s 使 S2∪S3 基线 OOD SR≥0.20 且 ID−OOD≥0.15；网格下端在第一轮发现 c01 已低于阈值后细化到 0.05），结果 c01 s=0.15、c02 s=0.65、c03 s=0.2、c04 延迟 6；残余问题：c02 的 S2 起点 OOD 成功率仍为 0.00（该任务上 OOD 对 S2 无分辨力）。

**不可行组合如实登记**（`configs/tasks/ckpt_manifest.json` 的 `infeasible` 条目），不凑数。BC 训练只用于造检查点，**编辑者看不到**。

### 5.3 官方基准层 [计划，见 EXPERIMENT_PLAN §环境]
Tier-1 Kinetix（JAX，复现 MIT exp4 的 Ramp/car 与新关卡）、Meta-World；Tier-2 OCBench（用户主基准，ocbench-gen 已有集群入口）；Tier-3/4 LIBERO/LeWAM/RoboDojo 占位。适配器接口：给定权重字典→成功/回报/失败分型，隐藏物理参数由任务生成器冻结。大网络（>~5k 参数）不能放进提示词，需 Agentic/探针赛道（沙箱+只读前向），属另一赛道。

## 6. 网关与审计
### 6.1 Gateway 契约（冻结，Claude 与 Codex 共用）[已实现+测试]
`wsarena/llm/base.py`：一回合=一次**无状态**调用（全部历史重新渲染进提示；便于重放与审计）。必须：①关闭全部供应商内置工具并**审计**；②与主机用户级指令/记忆隔离；③记录模型 id、effort、CLI 版本、系统提示哈希；④把供应商侧限额映射成 `TurnResult.status`（`rate_limited / output_budget_exhausted / no_output / refused / timeout / error`），不抛异常；⑤只写 `workdir`。`wsarena/llm/conformance.py` 是所有网关必须通过的一致性套件。
### 6.2 ClaudeCLIGateway [已实现；待登录后验证]
`claude -p --model M --effort E --tools "" --strict-mcp-config --setting-sources "" --disable-slash-commands --system-prompt <中性一句> --output-format stream-json --verbose --no-session-persistence`，空 cwd。每回合保存原始事件流；任何 `tool_use` 事件或 init 里出现工具 → `audit.tool_free=False` → **整个 run 标 failed（invalid_reason=tool_use_detected）**。`scripts/isolation_probe.py` 在首个实验前验证“看不到用户指令/记忆、无工具”，探针未通过则不开跑。**2026-11-06 探针实测发现**：CLI 仍会注入三类**恒定环境上下文**——用户邮箱（`userEmail`）、“Claude Agent SDK”系统前缀、环境块（平台/OS/工作目录）。邮箱和前缀无法去除（`--bare` 需要 API key，`CLAUDE_CODE_SIMPLE` 与订阅登录不兼容，复制凭据到隔离配置目录有令牌轮换风险，故不做）；它们对所有臂相同，记为已知环境因素。**工作目录必须中性**（旧版目录名含任务/类别/臂名会泄漏标签，已修；`test_gateway_workdir_is_neutral_and_raw_events_go_elsewhere` 防回归）。GPT 侧的 Codex CLI 有自己的环境上下文，比较时作为“供应商 CLI 环境差异”披露。
### 6.3 额度与失败处理 [已实现+测试]
- 供应商侧限额（rate/usage limit、鉴权失败、超时）→ **中断**：保存已完成回合，运行器立即停止整批，不重试、不设定时器；之后同一 `--attempt` 再启动会**重放**已保存回合（零新模型调用）。
- 输出预算耗尽/无输出/拒答/无 edits 块/全部语句被跳过 → **该轮记失败并留在分母**，不重试（T4）。
- 每批前后写 `runs/QUOTA_LOG.tsv`（用户提供读数）。

## 7. 对照臂（强制，同一记账）[已实现+测试]
`null`（空补丁）、`random`（稀疏高斯扰动；**实测过于破坏性**，ΔSR 约 −0.2…−0.44，只作“任意扰动有害”的下限）、`random_greedy`（**同等反馈预算的贪心随机搜索**：每轮在“开发集最优”权重上加稀疏扰动，是最强的简单搜索替代，两个强度档）、`oracle_restore`（S1 真值还原，上界，带真值）、`analytic_ridge`（输出层闭式岭回归，特权专家标签，**Analytical 赛道单独一行**）。主结论必须同时报告 LLM 相对这些臂的位置（K-S3-02、K-S2-06）。

## 8. 指标与统计
- **主终点**：隐藏 ID 的 ΔSR（编辑后−基线，同一批隐藏 episode 配对）；次终点：OOD ΔSR。
- **可靠性**：Recovery=P(成功|基线失败)，Regression=P(失败|基线成功)，逐 episode 配对；Selective change 约束 ΔJ_target>0 且 ΔJ_retention≥−ε（ε 预注册）。
- **编程质量**：valid-patch 率、完成率、编辑稀疏度、相对 L2、定位精确/召回（S1）、语句类型（元素/范围/缩放/RESET）；[计划] 补丁因果消融（回退各张量组看增益来源）、PSI。
- **效率**：输入/输出/思考 token（只记供应商实际字段，不估计）、耗时、CLI 报告成本（订阅下仅供参考）。
- **推断**：单位=检查点；同一检查点内的 LLM 重复先平均；按任务分层的配对聚类 bootstrap + sign-flip 置换检验；Holm 校正（`analysis/stats.py`，有测试）。**不把 episode 当独立样本。**
- **噪声尺度（K-S3-05）**：评测确定，故用二项/跨集合标准差；`scripts/noise_floor.py` [计划] 在 5 个互不相交的 200-episode 集合上测同一检查点的 SR 标准差，登记为零假设对照后才设判定带宽。

## 9. 工程结构与保障
```
wsarena/{edits,mlp,envs,tasks,store,evaluation,prompts,protocol,version}.py
wsarena/llm/{base,claude_cli,scripted,conformance}.py   wsarena/analysis/{metrics,stats}.py
scripts/{make_tasks,run_experiment,summarize,isolation_probe,wsa_clean,calibrate_*}.py
tests/（解析、存储篡改、环境确定性、探针纯度、等价重参数化、协议预算/续跑/审计/分母、网关一致性、清理安全、统计）
```
CI 完整性测试对应任务书 11.6：检查点完整性（哈希+只读+篡改检测）、补丁确定性、架构不变（形状不可变）、前向纯度、无梯度（接口层无此能力）、隔离（探针）、Rollout 配额（Budget）、隐藏数据隔离（命名空间+提示不含 hid_*）、重放可复现（续跑重放）、崩溃恢复（中断后不重复消耗）。**不适用的两项（网络隔离、沙箱）随 Agentic 赛道到来时补**。

## 10. 与任务书（ChatGPT 报告）的偏离（只记录，未经用户裁决不回退）
| 任务书 | 本实现 | 原因 |
|---|---|---|
| 模型 GPT-6 Astra + Claude Opus 5.5，effort medium/high/xhigh | 首期仅 Claude：Sonnet 5.5、Opus 5.5 × medium/high | 用户 2026-11-06 原话 |
| 调用走官方 API（Messages/Responses） | 走 Claude Code 订阅（`claude -p`）；GPT 侧由 Codex 接手（`codex exec`，同 MIT） | 用户原话；订阅无 API key |
| 开发集 ≤128 episode | 16 初始 + R×16（R=8 时 144） | 初始报告单独计，保持每轮预算一致 |
| 主实验 324 run / 2,592 回合 | **按 Pilot 实测额度成本定档（全量/精简/最简）** | Pro 额度（5h+周）约束，K-S0-05 |
| 主 Benchmark 先 Kinetix/Meta-World/OCBench | 先 Tier-0 机制层打通协议，再接官方基准 | 官方基准依赖未装；先验证协议与成本 |
| 失败轨迹由网关统一给出 | 是（初始报告含一条轨迹摘要） | 与 MIT Kinetix 同信息量 |
