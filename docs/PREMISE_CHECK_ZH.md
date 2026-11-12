# WSArena · G0 前提核验（2026-11-06）

> 范例：M2A grade-ledger（K-S0-01、K-S1-02）。证据分级：`[✔claim]` 读过一手原文并核对到主张；`[✔meta]` 只核对到存在性/元数据；`[未核实]` 只见转述。**只有 `[✔claim]` 可支撑结论或打分。**

## A. 用户硬性条件

| # | 用户的硬性条件（原话） | 可操作定义 | 核验方法（一手来源） | 当前满足 | 结论 |
|---|---|---|---|---|---|
| P1 | “先做claude相关实验（sonnet 5.5和opus 5.5，分别用中、高两个推理强度）” | 4 臂：`claude-sonnet-5-5`/`claude-opus-5-5` × `--effort medium`/`high`；每个 run 记录实际模型 id 与 effort | `claude --help` 有 `--model`、`--effort`；模型 id 来自系统环境；**实际可调用性待登录后探针验证** | 条件存在；调用 blocked | **blocked（需登录）** |
| P2 | “调用走 Claude Code 订阅额度” | 不使用 API key；所有模型调用经 `claude -p`；额度读数记 QUOTA_LOG | `env` 无 `ANTHROPIC_API_KEY`；`claude auth status` loggedIn=false | 0/1 | **blocked（需用户重新登录）** |
| P3 | “最好能直接在本机本地完成实验” | Tier-0（自建小网络+CPU 环境）本机完成；官方基准能否本机跑待实测 | 本机：RTX 4060 8 GB（桌面占 3 GB）、11 GB RAM、20 核；当前**无** torch/jax/mujoco（见 EXPERIMENT_PLAN 环境章） | Tier-0 可；官方基准未测 | 部分满足；官方基准按 smoke 结果决定 |
| P4 | “如需要更强的算力再考虑用上MLDA和rose cluster” | 仅在本机 smoke 证明不足时请示后使用 | 记忆 mlda-cluster / rose-cluster；ROSE 只用免费 QoS | 未触发 | 保留后备 |
| P5 | “用上autoresearch等科研skills” | 使用本机 research-paradigm（章程/预注册/审查/结题）与 work-handoff；未发现名为 autoresearch 的 skill | `ListSkills`/skill 清单：只有 research-paradigm、work-handoff 等；`SearchSkills` 未查（可选） | 以 research-paradigm 代替 | 已满足（如用户指另一个具体 skill 请告知） |
| P6 | “不用命名为WSArena 2.0” | 项目、包、文档均用 WSArena | 目录/包名 | 满足 | verified |

## B. 任务书（ChatGPT 报告，用户粘贴）里的事实性前提

K-S0-02：二手转述只用于定位。下表逐项给一手核对结果。

| # | 报告中的断言 | 核对 | 等级 |
|---|---|---|---|
| F1 | MIT CSAIL《Sparks of Weight-Space Programming》，2026-11-02；含 Kinetix 修复、MinAtar 合成、Qwen3.5-0.8B 修复 | 读过原文（rikiyat.github.io/blog/weight-space-programming/）；八个实验；作者 Takehi、Teoh、Zhang、Khattab | `[✔claim]` |
| F2 | Kinetix：单条失败轨迹，单次改约 3.5% 权重，1,024 个 episode 全部成功 | 原文：“In a single try with no tools… all 1,024 episodes, changing 3.5% of the weights”；LLM 只看到场景描述和一条原网络的记录轨迹，**修改后无反馈**；原文称网络 171,664 权重，按所述结构自算 171,544（差 120，疑为原文笔误，不影响结论） | `[✔claim]` |
| F3 | Qwen3.5-0.8B：改 0.016% 权重，MATH-500 7.9→48.1 | 原文表格一致；改动是从 layer 15、19 的 MLP 输出各 64 列减去一个方向（131,072 权重），**使用了 Python 沙箱与最多 50 次前向**；Gemma 3 270M 无提升 | `[✔claim]` |
| F4 | 报告称“Qwen 修复主要来自推理循环”及 Honnibal/Goedecke 批评 | 维基百科参考文献列出；批评文原文未读 | `[✔meta]` |
| F5 | MIT 用“只给前馈计算工具”设置 | 日志仓库：每回合一次 `codex exec`，内置工具全关，每回合审计 `tool_free`；有代码的实验（3/6/7/8）用 bubblewrap 沙箱、无网络；effort：exp1/2/4=high，3/5/6/7=xhigh，8=max | `[✔claim]` |
| F6 | 日志与 harness 公开：github.com/RikiyaT/lm_understand-logs | 仓库存在（commit 4d09ba6d…），含 harness/ 与各实验 README；原文称**未发布独立 benchmark/harness，正在构建** | `[✔claim]` |
| F7 | Alan Turing Institute `weights_editing` 跨 LLM 复现，GPT-6 Astra vs Claude Opus 5.5 表（Breakout 457 vs 112 等） | `git ls-remote` 要求登录（公开不可达）；网络搜索无结果；维基百科参考文献未列 | **`[未核实]`——表中数字不进任何结论** |
| F8 | “xhigh 下 Claude 多次耗尽输出预算未写出权重” | 只出现在 F7 项下；但 flecart 文被称有同类观察 | `[未核实]`；仍作为风险写入设计（成本很低） |
| F9 | arXiv 2611.05341 Nusrat & Nusrat：Claude Opus 5.5 审查 433–4,801 参数网络，16/18 | 搜索无此论文；未能打开 | `[未核实]`；不引用数字 |
| F10 | arXiv 2611.04268 扩散模型构想文；arXiv 2610.12407 LeWAM 等 | 未打开 | `[未核实]` |
| F11 | vukrosic/llm-weight-editing：Opus 5.5，16 个有规则小任务，14/16 优于训练，均值 91.3% vs 83.3%；CIFAR-10 Opus 71.6% vs 训练 87.8%；部分用 SVD/最小二乘 | 仓库存在（commit c63d6730），README 逐字含上述数字；**自报、未独立验证、作者也声明是程序合成** | `[✔claim]`（仅作为“该项目自报”） |
| F12 | ValerioDolci/frontier-weight-edit：Opus 5.5 改 Gemma 4 12B 的 4–8 个 MLP 神经元（9.2 万参数），选择性行为改变 | 仓库存在（commit 09223ef8）；README 数字与报告一致（26%→5%、32%→16%、182/200 不变） | `[✔claim]`（自报） |
| F13 | Chandan Singh 同日用 Claude 手写小 transformer 解 10 个算法任务，评测有弱点 | 维基百科参考文献第 6 条（csinva.io 报告）；原文未读 | `[✔meta]` |
| F14 | Kinetix / OCBench / Meta-World / Tracr 仓库 | `git ls-remote` 全部可达（FLAIROx/Kinetix、seohongpark/ocbench、Farama-Foundation/Metaworld、google-deepmind/tracr） | `[✔meta]` |
| F15 | RoboDojo、LIBERO、LeWAM（三个同名）的具体细节 | 未核 | `[未核实]`；仅作 Tier-3/4 占位，不影响首期 |
| F16 | “Claude Opus 5.5 自适应思考始终开启，默认 medium；五档 effort” | 本机 CLI 有 `--effort`；允许值与行为待登录后探针验证 | `[未核实]`；探针前不写入预注册 |
| F17 | 价格：Opus 5.5 输入 $4/百万、输出 $20/百万 | 订阅路径不按此计费；仅作成本折算参考 | 不采用 |

## C. 结论
- 首期可推进的硬前提：P3（Tier-0）、P5、P6；**P1/P2 被登录阻塞**。
- 任务书里 F7–F10、F15 无法核实，**设计不依赖它们**；只引用已读一手（F1–F3、F5、F6、F11–F12）。
- 满足数足够继续（不需要放宽任何用户条件）。选项：维持（推荐）/放宽到仅 Tier-0/停止——当前选“维持”，等用户登录。

## D. 决策相关性（K-S0-03）
- 结果 A：Claude 在 L2 修复上稳定有效、L3 增强无效 → 改变实践：把 LLM 当“神经网络调试器”，不当通用优化器；协议与对照可用于评估任意后续模型（含 GPT）。
- 结果 B：在统一、可审计协议下 Claude 连 L2 也不稳 → 改变实践：社区转述的“稳定修复”需要加上协议/预算条件才可信。
- 零成本替代：直接让 LLM 写 PD 控制器或用最小二乘（Analytical 赛道对照）——这正是要排除的替代解释，已内置。

## E. 外部任务书完整性（K-S0-07）
- 用户粘贴的任务书首尾连贯（以“一、研究结论”开头，以“最终研究方案与论文定位”结尾），章节编号 一至十六齐全；两处渲染残留（图片占位、mermaid CSS）无语义；已向用户提出缺失项：是（登录、额度读数、授权确认）。
