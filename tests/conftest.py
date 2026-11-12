import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsarena import tasks
from wsarena.store import CheckpointStore


@pytest.fixture(scope="session")
def store(tmp_path_factory):
    return CheckpointStore(tmp_path_factory.mktemp("ckpts"))


@pytest.fixture(scope="session")
def reach_s1(store):
    """One S1 (planted) checkpoint on c01-reach, built once per session (~15 s)."""
    task = tasks.TASKS["c01-reach"]
    ck = tasks.build_s1(task, 1, "weak_axis")
    assert ck is not None
    meta = dict(ck["meta"]); meta["class"] = "S1"
    h = store.put(ck["weights"], meta)
    return h, ck
