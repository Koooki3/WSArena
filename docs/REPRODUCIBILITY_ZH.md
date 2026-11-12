# 运行与复现

## 仓库布局

| 路径 | 用途 |
|---|---|
| `wsarena/` | 检查点存储、WEP-1 补丁、任务、探针、网关、评测、统计和可视化 |
| `configs/` | 任务清单、实验定义、官方环境依赖快照和额度阈值 |
| `scripts/`、`checks/` | 检查点构造、实验执行、独立复算、统计和图表工具 |
| `gpt/` | Codex CLI 适配、请求审计、额度检查、实验计划和 Tier-0 主分析 |
| `tests/`、`gpt/tests/` | 共享与 GPT 回归测试 |
| `runs/SUMMARY/`、`reports/`、`paper/` | 公开汇总数据、结果图表和论文草稿 |

## 检查点与离线试跑

从 [项目发布页](https://github.com/Koooki3/WSArena/releases/tag/v0.1.0) 下载两个归档及各自的 SHA256 文件，放在克隆目录根部。

- `wsarena-checkpoints-v14.tar.gz` 包含 v14 清单 142 个 built 条目对应的 140 份唯一权重。
- `wsarena-evaluations-v14.tar.gz` 包含已终态 Claude 和脚本对照的公开评测记录及其引用的补充权重。仓库中已固定的 160 份 Tier-0 对照原件不在归档中重复打包。归档中的记录保留评测和图表字段，账户及模型原始事件已去除。

```bash
sha256sum -c wsarena-checkpoints-v14.sha256
sha256sum -c wsarena-evaluations-v14.sha256
tar -xzf wsarena-checkpoints-v14.tar.gz
tar -xzf wsarena-evaluations-v14.tar.gz
python -m pytest tests gpt/tests
python scripts/run_experiment.py configs/experiments/baselines-v1.yaml --attempt local-smoke --max-runs 1
```

该命令使用确定性脚本对照，无模型调用。构造新的检查点可使用 `scripts/make_tasks.py`。它会更新清单，研究复现请使用已发布的清单与对应归档。

## 官方环境与模型调用

安装 uv 后可用 `bash scripts/setup_envs.sh metaworld` 或 `bash scripts/setup_envs.sh ocbench` 创建隔离环境。使用对应环境的 Python 运行官方任务。安装脚本会重写该环境的依赖快照，原研究版本见 Git 记录。Kinetix 提供环境安装入口，目前主研究的官方层是 Meta-World。

Claude 订阅网关由 `scripts/run_experiment.py` 调用。GPT 离线计划可用 `python -m gpt.run_gpt gpt/configs/main-v1-t0.yaml --attempt preview --dry-run` 查看。GPT 在线验收需用自己的固定版本 Codex CLI，填入 `gpt/runtime.json` 中的 CLI、隔离认证目录、中性工作目录和本地 runs 绝对路径。用本地的订阅登录完成配置，然后运行 `python -m gpt.online canary --attempt native-recheck-20261112` 和同 attempt 的 `smoke`。Tier-0 入口验证冒烟与出站审计证据，并在每次请求前核对额度。

## 已发布数据及范围

GPT Tier-0 的 160 份脚本对照按预注册哈希原样保留，供 `gpt.tier0 preflight` 校验。公开图表、汇总结果与 [2,548 条运行级数值记录](../reports/data/run-metrics.csv) 用于查阅和复算。16 个检查点各保留一份 Claude 首轮提示和运行规格，用于跨侧逐字节对账。完整历史模型流、账户记录、未登记检查点、中间实验缓存、本机二进制及第三方摘录保留在研究工作区。基于原始模型流的历史回放仍需要原始证据。

项目可公开查阅。源工作区的许可证选择仍待确定，本次未添加开源许可。
