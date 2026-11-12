"""Render the WSArena figure set: light + dark, png/pdf/svg, a CSV table view per figure, manifest.json and FIGURES_INDEX.md.
  .venv/bin/python scripts/make_figures.py [--only FIG-06 FIG-08] [--theme light|dark|both]
Figures read only recorded artifacts (see each figure's provenance footer). LLM-era templates (LLM-xx) are rendered only when
LLM runs exist; otherwise the index lists them as 'pending data' (no placeholder charts)."""
import argparse, hashlib, json, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wsarena.viz import figures, style
try:
    from wsarena.viz import llm_figures  # noqa: F401  (registers LLM-era templates)
except ImportError:
    pass
from wsarena.version import source_hash

ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*"); ap.add_argument("--theme", default="both"); a = ap.parse_args()
out = ROOT / "reports" / "figures"; (out / "data").mkdir(parents=True, exist_ok=True)
modes = ["light", "dark"] if a.theme == "both" else [a.theme]
manifest = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"), "source_hash": source_hash(), "contract_version": style.contract_version(), "figures": []}
fail = 0
for f in figures.REGISTRY:
    if a.only and f.fid not in a.only: continue
    ent = {"id": f.fid, "title": f.title, "caption_zh": f.caption, "group": f.group, "files": {}, "status": "rendered"}
    for mode in modes:
        try:
            fig, tab, note = f.fn(mode)
        except figures.PendingData as e:
            ent["status"] = f"pending: {e}"; break
        except Exception:
            traceback.print_exc(); ent["status"] = "ERROR"; fail += 1; break
        ent["files"][mode] = style.save_all(fig, f.fid, out, mode); ent["data_note"] = note
        if mode == modes[0]:
            tab.to_csv(out / "data" / f"{f.fid}.csv", index=False); ent["table_view"] = f"reports/figures/data/{f.fid}.csv"
            ent["table_sha256"] = hashlib.sha256((out / "data" / f"{f.fid}.csv").read_bytes()).hexdigest()[:16]
    print(f.fid, ent["status"], flush=True); manifest["figures"].append(ent)
old = {}
if a.only and (out / "manifest.json").exists():          # partial re-render: keep the other figures' entries
    old = {e["id"]: e for e in json.loads((out / "manifest.json").read_text())["figures"]}
new = {e["id"]: e for e in manifest["figures"]}
merged = {**old, **new}
manifest["figures"] = [merged[f.fid] for f in figures.REGISTRY if f.fid in merged]
(out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
L = ["# WSArena 图表记录（自动生成，勿手改）", "",
     f"生成：{manifest['generated_at']} · 契约 v{manifest['contract_version']} · 源码哈希 {manifest['source_hash'][:8]}。每张图有浅色/深色版本（png / pdf / svg）、CSV 表格视图和图内溯源脚注。",
     "配色经 `PALETTE_VALIDATION.txt` 验证；颜色跟实体走（见 `wsarena/viz/style.py::ENTITY_SLOT`）。", "",
     "| ID | 标题 | 类别 | 状态 | 图（浅色 png） | 表格视图 |", "|---|---|---|---|---|---|"]
for e in manifest["figures"]:
    png = f"[{e['id']}.png](figures/light/{e['id']}.png)" if "light" in e["files"] else "—"
    L.append(f"| {e['id']} | {e['title']} | {e['group']} | {e['status']} | {png} | {('[csv](figures/data/' + e['id'] + '.csv)') if e.get('table_view') else '—'} |")
L += ["", "## 图注", ""] + [f"- **{e['id']} {e['title']}**：{e['caption_zh']}" for e in manifest["figures"]]
(ROOT / "reports" / "FIGURES_INDEX.md").write_text("\n".join(L) + "\n")
sys.exit(1 if fail else 0)
