# WSArena 文件管理与清理规范 v1

依据：`~/docs/workspace-rules.txt`（放置与命名）、ops-playbook §7（证据）、K-O-04/05/07、用户 2026-11-06 要求“注重规范文件管理和清理”。
原则：**能重新生成的不存；必须存的只存一份；任何删除都先列清单、留哈希、可追溯。** 不删实验目录、不 `git clean`、不终止身份未知的进程（ops-playbook §7）。

## 1. 目录约定（项目根 `<wsarena-root>/`）

```
wsarena/
├── AGENTS.md  PROGRESS.md  pyproject.toml  uv.lock  .gitignore  .gitattributes
├── docs/                    # 章程、设计、计划、预注册、决策记录（人写，进 git）
│   ├── CHARTER_ZH.md  PREMISE_CHECK_ZH.md  DESIGN_ZH.md  EXPERIMENT_PLAN_ZH.md
│   ├── HYPOTHESES_ZH.md  PREREG_ZH.md  FILE_MANAGEMENT_ZH.md
│   └── decisions/           # 用户授权与关键决策，逐字
├── survey/
│   ├── RELATED_WORK_ZH.md   # 文献核验表 + 检索日志
│   └── sources/             # 一手原文摘录/抓取（不可信数据）+ MANIFEST.sha256
├── configs/                 # YAML：models / tasks / protocols / experiments（进 git）
├── wsarena/                 # Python 包（进 git）
├── tests/                   # pytest（进 git）
├── scripts/                 # 命令行入口与一次性工具，每个脚本顶部写用途（进 git）
├── checks/                  # 独立复算脚本，不导入 wsarena（进 git）
├── artifacts/               # 可再生成的任务产物，按内容哈希寻址（不进 git）
│   ├── checkpoints/         #   sha256/<hash[:2]>/<hash>.npz   S0/S1 可由 seed 重建，S2/S3 基础检查点保留
│   └── tasks/               #   任务定义快照（配置哈希）
├── runs/                    # 实验输出，只追加（不进 git；摘要与 MANIFEST 进 git）
│   ├── QUOTA_LOG.tsv        #   每批前后额度读数
│   ├── <exp-id>/<attempt-YYYYMMDD-HHMMSS>/<run-id>/   run.json、turns/NN.{prompt,reply,cli}.json、patches/、feedback/、audit.json
│   └── SUMMARY/<exp-id>-<attempt>.{json,md}           每阶段一份结果 JSON + 一份人读摘要
├── reports/                 # 分析图表、结题报告（进 git）
├── external/                # 第三方克隆：只读、不可信、不运行；gitignore；见 §3
├── venvs/                   # 官方基准各自的虚拟环境（metaworld/ocbench/kinetix，约 1.8 GB）；不进 git，可删可重建（scripts/setup_envs.sh）；版本清单 configs/envs/*.freeze.txt 进 git
├── vendor/                  # 需要运行的第三方代码副本（注明来源 commit）；目前为空
└── .venv/                   # uv 虚拟环境（不进 git；可删可重建）
```

临时文件一律放会话 scratchpad（`/tmp/claude-1000/.../scratchpad/`），不放 `/tmp`、不放项目根。证据不放 `/tmp`（K-O-04）。

## 2. 命名
- 目录小写英文加连字符；日期 `YYYY-MM-DD`，唯一运行目录 `YYYYMMDD-HHMMSS`。
- run-id：`<task>__<ckpt>__<model>-<effort>__r<rep>`，例 `c01-reach__s2-seed3__opus55-high__r2`。
- 文件名表达用途，不用 `new`、`final2`、`test-copy`；版本靠 git 与文件内版本号，不靠后缀。
- 配置、补丁、反馈为 JSON/YAML；原始模型回复为 JSONL/文本，不改写。

## 3. 外部克隆（`external/`）
- 现有三份浅克隆（lm_understand-logs 367 MB、llm-weight-editing 38 MB、frontier-weight-edit 59 MB，合计约 464 MB）只为读取一手材料。
- 登记于 `survey/sources/MANIFEST.sha256` 与 `survey/sources/REPOS.tsv`（仓库、commit、取得时间、用途）。
- **已执行（2026-11-06）**：三份克隆的关键文件（harness 配置/提示词/README/结果表，共 11 个，164 KB）已摘录到 `survey/sources/extracts/` 并登记 `MANIFEST.sha256`，457.5 MB 克隆已由 `wsa_clean --include-external --apply` 删除（回执 `runs/SUMMARY/clean-*.json`），可凭 REPOS.tsv 重新克隆。
- **S1 关闭时处置**：把确实需要的小文件（harness 配置、提示词、README、结果表）复制进 `survey/sources/extracts/<repo>/` 并登记哈希，然后删除 `external/` 对应克隆（可凭 REPOS.tsv 重新克隆）。需要运行第三方代码时，先复制到 `vendor/<名>/` 并写明来源 commit，不在 `external/` 内运行。
- 禁止在 `external/` 内执行解释器或构建；读取用 `python3 -I` 或纯文本工具。

## 4. 保留等级
| 等级 | 内容 | 处置 |
|---|---|---|
| 永久 | docs、预注册、决策记录、configs、代码、测试、checks、summary JSON/MD、patches、`run.json`、原始模型回复（turns/）、`audit.json`、QUOTA_LOG | 进 git 或随 runs 保留；阶段结束时 turns/ 打包为 `.tar.zst` 并核对哈希后再删明文 |
| 可再生成 | S0/S1 检查点、任务快照、缓存的前向结果 | 只保留生成配置与哈希；磁盘紧张时可删，重建后哈希必须一致 |
| 一次性 | scratchpad、smoke 目录、`__pycache__`、`.pytest_cache`、`.venv`、`external/` 克隆 | 收尾清理 |
| 失败证据 | 失败/中断/触限 run | **永不删除**，留在分母；只可压缩 |

## 5. 运行记录最小字段（身份绑定，K-S4-05）
每个 run 的 `run.json` 必须有：`run_id`、`exp_id`、`attempt`、`task_id`、`ckpt_sha256`、`code_git_sha`（或源码树哈希）、`config_sha256`、`protocol`（WSP-Native/Analytical/Agentic）、`model_id`、`effort`、`claude_cli_version`、`system_prompt_sha256`、各回合 `usage`（输入/输出/思考 token 与 CLI 报告的成本）、`status`（verified/failed/interrupted/unknown/not_run/blocked）、`failure_type`、起止时间（+08）。缺任一字段，汇总脚本拒绝纳入。

## 6. 阶段结束与项目收尾清理清单
每个阶段结束（checkpoint 前）：
- [ ] `wsa_clean --dry-run` 列出将要处理的项（缓存、smoke、可再生成大文件），人工看一眼再 `--apply`。
- [ ] 本阶段 `turns/` 压缩并校验；明文只在校验通过后删除。
- [ ] 更新 `runs/SUMMARY/` 与 `PROGRESS.md` 首段四行；`work-handoff checkpoint`。
- [ ] `external/` 若已完成取材则按 §3 处置。
- [ ] 无未登记的后台进程、tmux、cron（本项目不设 cron）；`work-handoff` 里无陈旧 running。
项目收尾（结题）：
- [ ] RETRO 投 `research-library/inbox/2026-MM-DD-wsarena-<主题>.md`。
- [ ] 删 `.venv`、`venvs/`、`external/`、`__pycache__`（`wsa_clean --include-venv`）；保留 `uv.lock` 与 `configs/envs/*.freeze.txt`。
- [ ] `artifacts/checkpoints` 中可再生成者按 §4 处置；S2/S3 基础检查点与 MANIFEST 保留。
- [ ] 更新 `~/README.txt` 与 `docs/layout.json`（ops-playbook §10）。
- [ ] `work-handoff release --state ready|paused|complete`（不留陈旧 active）。
- [ ] 若在 MLDA/ROSE 跑过：`ws-clean`，确认 home 外无遗留；先确认结果已带回本机。

## 7. 自动清理工具（随代码交付）
`scripts/wsa_clean.py`：
- 默认 `--dry-run`，只处理 §4 “可再生成/一次性”项，永不触碰“永久/失败证据”；
- 输出清单（路径、字节数、等级、可再生成依据），`--apply` 时把清单与哈希写入 `runs/SUMMARY/clean-<时间>.json`；
- 有单元测试：造“失败 run”“永久文件”“缓存”三类夹具，断言前两类任何参数下都不会被删。
