# 输出上限政策（共享；2026-11-08，U14）——把“有效输出上限”作为被记录的协议变量

**背景**：固定版本的 Codex CLI 没有服务端 `max_output_tokens` 通路，Claude CLI 虽接受 64000 但无法从返回值证实一定生效。两侧都不能保证“同一个有效上限”。数据显示上限没有成为约束（Claude：Tier-0 主研究 256 回合 0 截断；P2 32k 下 0/32；Meta-World 34 个主运行无截断；单运行最大输出 45,860 token）。

**政策**（对 Claude 与 GPT 两侧都适用）
1. **记录而非阻断**：每个 run 的 spec 记录 `output_cap_requested`（请求值或 `none`）和 `output_cap_effective`（`unverified` 或 `observed_max=<N>`，N 为该 run 单回合观测到的最大输出 token）；GPT 臂写 `output_cap_requested=none (CLI default)`。
2. **截断计入分母**：触到上限（`output_budget_exhausted`）的回合按既有规则记为“无编辑”，留在分母，并在汇总里单列失败类型；不得用人为截断回复来“伪造上限”，也不得静默改走 API。
3. **预先规定的停止规则**：某一臂在一个批次内的截断回合占比 **>1%** → 标注该臂并暂停该臂的后续运行，由负责的会话评估（可能需提高有效上限或改用别的通道），结果里并报“截断敏感性”。
4. **跨侧声明**：GPT 与 Claude 的比较必须在方法/局限里写明“两侧有效输出上限未对齐，二者均未观察到截断（或观察到 N%）”。Claude 侧 P2（32k 对 64k）只作为 Claude 侧的灵敏度证据，不推广到 GPT。
5. **在线联调的首个检查**：GPT 侧第一批成本冒烟里，必须读取并保存各回合的实际输出 token 数，用以检验 3 的 1% 规则；无法读取输出 token 数则该臂不得进入正式批次。

**状态**：本政策解除“因 CLI 无法设置 64k 而阻断”这一共享层面的阻断；GPT 侧是否开始在线联调、何时开始，仍由用户在 Codex 会话里明确指示（Codex 只写 `gpt/**`，需由它自己把本政策落到 `gpt/` 的运行器与配置中）。更正（v13）：Claude 侧**不需要**改 `RunSpec`——请求值本来就在 `spec.max_output_tokens`，有效值由分析从 `rounds[].usage.output_tokens` 取单回合最大值得到，截断由 `patch_status=turn_failed:output_budget_exhausted` 计数；契约 v13 只为探针赛道新增文件。
