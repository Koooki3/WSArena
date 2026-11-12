# GPT Tier-0 主研究预注册 v1

冻结日期 2026-11-12。复用共享预注册 §11 的 16 个 seed2/3 检查点及运行协议。四个原生 GPT 实验臂各跑一次，共 64 个运行、最多 256 轮。执行顺序以 `configs/main-v1-t0.yaml` 的 `class_interleave` 排序实现为准。seed1 冒烟和 Pilot 单独存档。

每个 Native 运行 4 轮，初始报告和每轮 dev 各 16 个 episode，隐藏 ID 和 OOD 各 200 个 episode。实际保存的首轮提示按同一检查点与 Claude Sonnet medium 对照逐字节核对。U14 下 GPT CLI 用默认输出设置，Claude 请求 64,000 token，双方生效上限均未验证。原生 effort 各按本提供方定义。

主终点为最终轮隐藏 ID ΔSR。统计单位为检查点。同一对照臂的重复先按检查点平均。共享分析在 10 个 L2 检查点使用的 160 份基线记录由 `tier0-baselines.json` 固定路径和 SHA256，覆盖随机、特权解析及三个次要对照。每个 L2 检查点含 6 次 random5、2 次 analytic_ridge、6 次 greedy_f02_s01、1 次 gain_greedy、1 次 tensor_gain_greedy。

C1 为 L2 对 random5，C2 为 L2 对特权 analytic_ridge。点估计对检查点等权，95% CI 用按任务分层的 5,000 次配对 bootstrap，双侧 p 用 20,000 次符号翻转。C3 为同任务内 S3−L2 后按任务等权平均，95% CI 用按任务和类别分层的 5,000 次 bootstrap，双侧 p 用 5,000 次任务内置换。C3 同时披露检查点加权点估计。各检验随机种子均为 0，置换 p 用加一修正。四臂共 12 项主要检验做 Holm 校正，三个贪心对照的 12 项次要检验另做 Holm 校正。后续机制、盲读与提供方比较属于探索性分析。

每项主要比较使用预定的完整检查点集合。终态失败运行保留实测 ΔSR。中断保留已保存回合，未启动或未到终态的运行记为缺失。64 个预定运行齐全、证据完整并通过提示核对后才出推断结果。完成率低于共享停止判据时记不可估计。

每次调用串行，单个逻辑批次最多 8 个运行。逐轮核对订阅的 5 小时 90% 与周 80% 停线。读不到额度、触及限额、鉴权或超时则停批，保存证据并保留原 attempt 和 batch，等待后续人工续跑。U14 截断率超过 1% 暂停该臂，截断轮仍入分母。

```sh
.venv/bin/python -B -m gpt.tier0 preflight
.venv/bin/python -B -m gpt.tier0 run --attempt tier0-native-20261112 --batch b01
.venv/bin/python -B -m gpt.tier0 analyze --attempt tier0-native-20261112
```
