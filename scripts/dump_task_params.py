"""Dump task parameters (ID + OOD) to configs/tasks/task_params.json so independent checks need not import wsarena."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from wsarena import tasks
out = {tid: {"params_id": t.params_id.to_dict(), "params_ood": t.params_ood.to_dict(), "n_hidden": t.n_hidden}
       for tid, t in tasks.TASKS.items()}
(ROOT / "configs/tasks/task_params.json").write_text(json.dumps(out, indent=1, sort_keys=True)); print("ok", list(out))
