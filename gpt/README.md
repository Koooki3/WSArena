# WSArena GPT 网关

GPT-6.1 Sol 与 GPT-6 Astra 各使用 medium 和 high。Native Pilot 已完成。Tier-0 主研究的实验范围、停止条件和分析见 [预注册](TIER0_PREREG_ZH.md)。后续 Meta-World、盲读、重复和尺度配置仍为候选。

从项目根运行离线规划。下列命令无模型调用。

```sh
python -m gpt.run_gpt gpt/configs/main-v1-t0.yaml --attempt preview --dry-run
python -m gpt.tier0 preflight
python -m pytest tests gpt/tests
```

| 模块 | 功能 |
|---|---|
| `codex_cli.py`、`wire_audit.py` | 无状态 CLI 请求、事件解析、出站工具表与输入审计 |
| `online.py`、`account_limits.py`、`quota.py` | 联调、订阅额度读取和请求前停线 |
| `run_gpt.py`、`identity.py`、`output_caps.py` | 有界批次、输入与回放证据、U14 输出计数 |
| `parity.py`、`tier0.py` | 跨提供方首轮提示对账、主研究执行和统计 |

[本地环境与在线验收](../docs/REPRODUCIBILITY_ZH.md#官方环境与模型调用) · [Pilot 协议](PILOT_PROTOCOL_ZH.md)
