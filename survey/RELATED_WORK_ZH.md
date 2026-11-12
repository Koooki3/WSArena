# WSArena 文献与新颖性核验（2026-11-06；检索日志在末尾）

等级：`[✔claim]` 读过一手原文并核对到主张；`[✔meta]` 只核到存在性/元数据；`[未核实]` 只见转述。**只有 `[✔claim]` 可支撑结论。**
一手落盘：`survey/sources/`（REPOS.tsv 登记三份克隆的 commit；MIT 博客由抓取代理读取，原文哈希未落盘 → 见“待办”）。

## 1. 已覆盖 / 我们还剩什么（K-S1-05）

| 工作 | 已覆盖 | 等级 | WSArena 还剩（差异） |
|---|---|---|---|
| MIT CSAIL《Sparks of Weight-Space Programming》（Takehi, Teoh, Zhang, Khattab；2026-11-02） | 8 个实验：读权重识别游戏/形状/语言/注意力头；Kinetix 单次修复（3.5% 权重，1024/1024 成功）；MinAtar 从零合成（Breakout 517.4 对 PPO-500M 25.3）；Go 150 万权重；Qwen3.5-0.8B 改 0.016% 权重 MATH-500 7.9→48.1；Gemma 3 270M 无提升；**日志与 harness 公开**（`codex exec`，工具全关，逐回合审计，effort high/xhigh/max） | `[✔claim]`（原文，经抓取代理逐项核对数字） | **单次/单种子、无对照臂（随机/解析/神谕）、无隐藏 OOD、无选择性变化度量、无失败分母、仅 GPT-6 Astra**。原文自述“正在构建 benchmark 与 harness”，未发布 → 空白 |
| MIT `lm_understand-logs`（RikiyaT） | 完整日志、harness 配置、edits 语言、sandbox（bubblewrap，无网络） | `[✔claim]`（克隆 commit 4d09ba6d；读过 README、harness、exp5 提示词） | WEP-1 兼容其 edits 语言；我们加了服务端预算、检查点哈希、重放 |
| vukrosic/llm-weight-editing | Claude Opus 5.5 在 16 个有规则小任务上不训练写权重：均值 91.3% 对训练最优 83.3%（14/16 更优）；CIFAR-10 Opus 71.6% 对训练 87.8%；承认部分用 SVD/最小二乘（程序合成） | `[✔claim]`（仅作“该项目自报”；README 逐字） | 自报、单次、未独立验证；没有统一的 Native/Analytical 分赛道与对照臂；本项目把“解析求解器”作为强制对照 |
| ValerioDolci/frontier-weight-edit | Opus 5.5 改 Gemma 4 12B 的 4–8 个 MLP 神经元（9.2 万参数），选择性行为改变；DI 句 26%→5%，对照 182/200 逐字不变 | `[✔claim]`（自报） | 语言任务；支持把“选择性行为改变（保留约束）”做成核心指标 |
| Tracr（Lindner et al. 2023，arXiv 2301.05062） | 程序→transformer 权重的编译器；已知结构的“标准答案”权重 | `[✔meta]` | 可作 L1/L4 的确定性对照来源（计划，未用） |
| ROME 等模型编辑（Meng et al. 2022，arXiv 2202.05262） | 固定算法的窄编辑 | `[✔meta]` | 与“LLM 自行决定编辑”不同；不作基线 |
| Text-to-LoRA、Code2LoRA、COMPNET（超网络生成权重） | 学习式权重生成 | `[✔meta]`（搜索摘要） | 需训练的权重生成器，不是“通用 LLM 读写权重”；不在本协议内（会被归为独立权重生成模型，协议明确禁止） |
| Chandan Singh（2026-11-02）Claude 手写小 transformer（10 个算法任务） | 维基百科引用，称部分结果受评测弱点影响 | `[✔meta]`（只见维基转述；原页未读） | 正好说明“评测可信度”是核心；待办：读原页 |
| Honnibal / Goedecke 批评 | Qwen 的改善主要来自移除推理循环；设置并非“只有原始权重”（用了 Python 与最多 50 批练习题） | `[✔meta]`（维基转述；原页未读）；其核心事实已由 MIT 原文印证（exp8 有 Python 沙箱与前向次数） | 在协议里区分 Native/Analytical/Agentic；声明工具权限 |
| Alan Turing Institute 复现报告页（用户 2026-11-07 提供 URL） | — | **`[未核实]`**：页面抓取 HTTP 403，未读到任何内容；清单所述“Claude 高 effort 运行耗尽 token”只被 flecart 的独立现象间接支持（Opus 5.5 max effort 用满 128k 输出上限后不作答） | 其数字不进任何结论；可由用户直接打开页面后贴内容再纳入 |
| arXiv 2611.05341 Nusrat & Nusrat《Weight-space programming for auditing clinical decision networks》（2026-11-03） | 433–4,801 参数网络，4 条规则，经训练数据植入缺陷（混杂、3 Gy 阈值偏移、隐藏触发）；Claude Opus 5.5 无工具盲审，一次通过；约 42 万请求验证 | `[✔abstract]`（仅读摘要，经子代理转述） | 18 个缺陷网络识别 16 个（含全部 15 个混杂/阈值偏移），每个修复平均改 3 个参数；2/5 健康网络未被误报；1/3 触发被发现；**对每类缺陷挑选的简单统计量也能定位同样缺陷**；微调也能修复但几乎改全部参数；代码 github.com/hnusrat1/weightspace-rt。启示：对照臂必须含“非 LLM 故障检测”（我们 `analytic_ridge`/统计类对照的对应物）；其统计量是事后按缺陷类型挑选的 |
| arXiv 2611.08070 Ahmed《Can an LLM actually reason and program about neural network’s weights…》（2026-11-05；单作者） | Claude Opus 5.5，无工具，约 12.5 万参数 CNN × 5 个数据集，权重纯文本，无数据集/类名 | `[✔abstract]` | 能从 BatchNorm 统计推断“约 3 次优化步”，从输出层符号模式还原 CIFAR-10 的交通工具/动物划分；语义判断时好时坏（把 99.57% 的 SAT-6 网络读成 CIFAR-10）；只做了 CIFAR-10 标签置换；作者自述“探索性、多为单次、未做前向测试”。清单所说“数据集置换后出错”摘要里只有标签置换 |
| arXiv 2611.07319 WeightLoop（Li 等；2026-11-05；5 页） | GPT-6 Astra 读任务规格与模型实现，写一个有界、可复用的“参数编辑程序”，之后无 LLM 调用、无梯度；离线重放；3 个顺序任务 | `[✔abstract]` | 8 个“端点-预算”设置里 7 个在平均排序效用上优于对照（摘要中的对照是**随机编辑**，不是强基线）；程序化更新，**不属于 Native**；无代码链接 |
| flecart《I Tried to Break Weight Programming》（Huang；2026-11-03） | gpt-6-astra high，单次回复无隐藏工具；4 个 PPO 策略（Breakout/Pong/Pac-Man/Freeway）在 10×10×5 网格；W1 的 500 行输入顺序用仿射/XOR/SHA-256 重排 | `[✔partial]`（前 39k 字为原文，其余为模型摘要） | 游戏识别 156/156；Breakout 复现 501.2±7.8（对论文 517.4±7.4，1000 局）；有“隐藏单元打乱+符号翻转”对照行（标为 unstated）；非 LLM 基线 8.4±0.6 / 6.9±0.1（摘要）；max effort 的 Opus 5.5 用满 128k 输出预算后不作答（直接相关：我们的 64k 上限与 P2 灵敏度实验）；Kinetix 仅一个场景；代码 github.com/flecart/weight-programming |
| OpenWSP `openwsp/openwsp`（独立复现，MIT 协议） | 编辑者 gpt-6-astra max effort，经 Codex；持久 Python 内核 + 前向服务 + 版本化编辑格式；对 Qwen3.5-0.8B 只改第 23 层 down_proj；上限 50 次前向、300 次 Python | `[✔partial]`（README 前 7.9 万字为原文，后段未读） | 现场编辑 MATH-500 12.9%（258/2000），84.25% 触顶 token 上限；完成样本 79.7%（315 个）；作者自己的 p50c 为 48.1%（不同采样/上限/引擎，页面自称不可比）。启示：结果对 token 上限、引擎、接口高度敏感；README 里“patch store/replay”未在已读部分找到 |
| 维基百科“Weight-space programming”（rev 1383865623，2026-11-05） | 概念定义、历史、批评综述 | `[✔meta]` | 二手综述，仅作定位 |

**结论（对新颖性）**：截至 2026-11-06 未发现“带强制对照臂、检查点哈希、服务端预算、隐藏 OOD、选择性变化、失败入分母、分五级能力的统一权重空间编程基准”。MIT 自述正在构建同类 benchmark/harness，**存在被抢先的风险**——因此首要产出是协议与可审计 harness，而非单一排行榜数字；新颖性主张限定在“协议+对照+分层评分+对 Claude 的系统实验”，并在 MIT 发布后复查。

## 2. 待办（不影响当前推进）
1. 读 csinva.io（Singh）、honnibal.dev、seangoedecke.com 原页，升级为 `[✔claim]`。
2. MIT 博客原文落盘 + sha256 登记（目前由抓取代理读取并核对）。
3. 每周复查：MIT benchmark/harness 是否发布；Turing/arXiv 条目是否出现可核实链接；Claude 项目更新。
4. 前向引用检索（引用 MIT 博客的新论文）— 搜索引擎对 2026-11 的收录有限，需在 arXiv 上直接查。

## 3. 检索日志（K-S1-01 四步）
| 日期 | 步骤 | 查询 / 动作 | 结果 |
|---|---|---|---|
| 2026-11-06 | 本地索引 | `kb_search.sh` 关键词 weight editing / OCBench / Kinetix / memguard / claude -p headless / judge integrity | 无同题；仅 OCBench/memguard 相关卷宗 |
| 2026-11-06 | 联网 | “weight space programming” LLM；programming in weight space compile program into weights benchmark；hypernetwork LoRA from task description；Tracr RASP | 命中 DataCamp、The Register、Xenospectrum、Menon Lab、Tildes（均二手）；Text-to-LoRA/Code2LoRA；Tracr |
| 2026-11-06 | 联网 | weights_editing alan-turing-institute；2611.05341；llm-weight-editing；flecart 博客 | 搜索无命中（其中 llm-weight-editing 经 `git ls-remote` 证实仓库存在） |
| 2026-11-06 | GitHub | `git ls-remote`：RikiyaT/lm_understand-logs ✓、vukrosic/llm-weight-editing ✓、ValerioDolci/frontier-weight-edit ✓、FLAIROx/Kinetix ✓、seohongpark/ocbench ✓、Farama-Foundation/Metaworld ✓、google-deepmind/tracr ✓；alan-turing-institute/weights_editing ✗（要求登录=不可公开访问） | 见上表 |
| 2026-11-06 | 一手阅读 | 抓取代理读 MIT 博客与维基百科；本机克隆三份仓库读 README/harness/提示词 | 见上表 |

## 2026-11-07 核对记录（用户转贴 ChatGPT 清单后）
子代理（只读，web-fetch）读取 6 个 URL：Turing 403；WeightLoop、2611.05341、2611.08070 仅读 arXiv 摘要页；flecart 前 39k 字为原文；OpenWSP README 前 7.9 万字为原文。**以上均为二手转述，论文引用前须由我们自己再读原页/全文**。清单中其余条目（honnibal、frontier-weight-edit、Goedecke、Willison、Program-as-Weights 2607.02512、Compile by Training 2609.04199、2611.04268）本轮未核对，状态不变。
对 WSArena 的直接含义：①token 预算耗尽在多个独立复现里反复出现（Turing[未证实]、flecart、我们的 Pilot）→ 把“输出预算”作为协议变量报告，P2 灵敏度实验正是回应；②“简单统计也能定位缺陷”（2611.05341）支持我们的非 LLM 对照梯度；③权重重排/打乱对照（flecart、2611.08070）与我们的 A3/A4 同类，但我们是在控制任务上做成功率增益，他们是在识别任务上做；④这些工作多为单次运行，我们的重复运行与聚类推断是差异点。

## 2026-11-08 新增检索与核对（用户：“这几天又有很多……放出，搜索、分析并参考”）
检索（4 组标准搜索）覆盖报道、博客、arXiv、GitHub、HF。结果：**绝大多数是对 MIT 11-02 博客的二手转述**（DataCamp、The Register、GOpenAI、LetsDataScience 等），技术增量很少；个别二手文章的模型名互相矛盾（“Qwen 3.8 27B”“GPT-5 Astra”等），一律不引用。真正有新信息的只有下面几条，均已回到一手页面核对（子代理只读转述）：

| 条目 | 一手来源 | 核对结果 | 对 WSArena 的含义 |
|---|---|---|---|
| **MIT 原博客的“正在建 benchmark+harness”** | rikiyat.github.io/blog/weight-space-programming/ | 原文：“We are currently building a benchmark and a harness for LLMs' capability on Weight-Space Programming.” 无日期、无链接 | 同期竞争/互补工作；我们的相关工作与讨论要写明；其发布后做逐项对照；我们的差异（预注册、受控对照梯度、聚类推断、盲读/重参数化诊断、跨规模）要在 README 里讲清 |
| **Coletti 独立复现**（HF：JonathanColetti/Qwen3.5-0.8B-Astra-WeightEdit） | huggingface.co 页面（已读） | MATH-500（500×4）：原模型思考开 7.4%（93% 触 token 上限）→ 编辑后 50.2%（32%）；**思考关**：45.8% → 43.6%（编辑反而略降）。GSM8K（1319×2）13.0→67.6（思考开）、57.1→55.7（关）；IFEval 19.2→62.1（开）、52.9→51.2（关）。设置：思考开 T=0.6、32k token；关 T=0.7、8k；vLLM bf16、seed 0。页面原话：“The edit only helps with thinking on.” “Almost all of the gain comes from answers that no longer loop to the token limit.” | **与我们的 A3 同向**：头条增益主要来自“不再循环”的行为调节，不依赖对数学能力的理解；且结果对生成设置高度敏感。我们已固定协议并报告输出预算（P2）；论文要引用并强调“增益来源要用消融证明”。Magica 的摘要把“思考开的编辑后 vs 思考关的原模型”说成“编辑后仍更好”，措辞有误导，不引用该摘要句 |
| MIT 博客的负面/控制结果 | 同上 | Gemma 3 270M：基线 6.8，编辑后 7.0/7.6/7.7/7.0（5/10/25/50 次前向），“Astra could not improve it”；Seaquest 53.1 对 PPO-500M 88.9；Asterix 53.2 对 55.2；实验 7（谄媚）仅凭权重 4 次都判不出 | 可作为“能力边界”的一手证据（小模型、分布式故障、从权重判行为） |
| arXiv 2611.04268（扩散复原） | arxiv 摘要 | 自称“Technical Conjectures”，无实验验证 | 仅作未来方向，不作证据 |
| Khaliq（HF）重建 51.0% | MIT 博客致谢并链接其 HF 主页/演示 | 博客只致谢，未读到 51.0% 的一手页面 | 未核实；不引用数字 |
| Honnibal 反驳（buildmvpfast 转载） | 403 | 未读到 | 之前已有的 honnibal.dev 条目仍为元数据级；不引用其论点 |

**启示（并入 ROADMAP 的论文讨论项）**：①“增益来自行为调节而非理解”现在有三条独立线索——我们的 WRI≈0、Coletti 的思考开关、MIT 自己的 Gemma 失败——可以作为论文的一个小节而非脚注；②MIT 说会放 benchmark，我们的发布窗口要关注，但门槛仍是 U8/U9（完备性、科学性、泛用性）；③生成设置敏感性是普遍问题，我们的 P2（输出上限）与 effort 对比可作为“设置敏感性”的报告范式。
