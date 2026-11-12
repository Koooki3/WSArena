import importlib.util, json
from pathlib import Path
spec = importlib.util.spec_from_file_location("wsa_clean", Path(__file__).resolve().parents[1] / "scripts" / "wsa_clean.py")
wc = importlib.util.module_from_spec(spec); spec.loader.exec_module(wc)

def mk(root):
    for p in ("runs/e/a/run1/run.json", "runs/e/a/failed/run.json", "runs/e/a/run1/raw/ev.jsonl", "docs/x.md", "configs/c.yaml",
              "wsarena/m.py", "wsarena/__pycache__/m.pyc", "tests/__pycache__/t.pyc", ".pytest_cache/v", "artifacts/checkpoints/c.npz",
              "external/repoA/f", "external/unregistered/f", ".venv/bin/python", "survey/sources/REPOS.tsv", "survey/sources/MANIFEST.sha256"):
        f = root / p; f.parent.mkdir(parents=True, exist_ok=True); f.write_text("x")
    (root / "survey/sources/REPOS.tsv").write_text("repo\turl\ncommit\nrepoA\tu\n")

def test_default_only_ephemeral(tmp_path):
    mk(tmp_path); got = {Path(i["path"]).relative_to(tmp_path).as_posix() for i in wc.plan_clean(tmp_path)}
    assert got == {"wsarena/__pycache__", "tests/__pycache__", ".pytest_cache"}

def test_runs_and_permanent_never_listed_with_any_flags(tmp_path):
    mk(tmp_path)
    got = {Path(i["path"]).relative_to(tmp_path).as_posix() for i in wc.plan_clean(tmp_path, True, True, True)}
    assert not any(g.startswith(("runs", "docs", "configs", "survey", "checks", "reports", "gpt")) for g in got)
    assert "external/repoA" in got and "external/unregistered" not in got and ".venv" in got and "artifacts/checkpoints" in got

def test_apply_writes_receipt_first_and_keeps_failed_runs(tmp_path):
    mk(tmp_path); items = wc.plan_clean(tmp_path, True, True, True); r = wc.apply_clean(tmp_path, items)
    assert json.loads(r.read_text())["items"] and (tmp_path / "runs/e/a/failed/run.json").exists()
    assert (tmp_path / "docs/x.md").exists() and not (tmp_path / ".venv").exists()


def test_venvs_only_listed_with_flag_and_never_runs(tmp_path):
    mk(tmp_path); (tmp_path / "venvs/kinetix/bin").mkdir(parents=True); (tmp_path / "venvs/kinetix/bin/python").write_text("x")
    off = {Path(i["path"]).relative_to(tmp_path).as_posix() for i in wc.plan_clean(tmp_path)}
    on = {Path(i["path"]).relative_to(tmp_path).as_posix() for i in wc.plan_clean(tmp_path, include_venv=True)}
    assert "venvs/kinetix" not in off and "venvs/kinetix" in on and not any(p.startswith("runs") for p in on)
