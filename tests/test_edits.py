import numpy as np
import pytest
from wsarena import edits
from wsarena.mlp import init_weights

def W():
    return init_weights(4, 6, 2, zeros=True)

def run(block, **kw):
    return edits.apply_edits(W(), block, **kw)

def test_single_entry_and_range_and_fill():
    r = run("W1[2, 0] = 2.5\nW1[0:4, 3] = [1, 0, -1, 0]\nb1[0:4] = -1")
    w = r.new_weights
    assert w["W1"][2, 0] == 2.5 and list(w["W1"][:, 3]) == [1, 0, -1, 0] and list(w["b1"][:4]) == [-1] * 4
    assert r.n_applied == 3 and r.n_skipped == 0

def test_2d_block_scale_and_strided():
    r = run("W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]\nW2[0:2, 0:3] *= 2\nW1[0:4:2, 0] = [7, 8]")
    w = r.new_weights
    assert w["W2"][0, 1] == -2 and w["W2"][1, 1] == 4 and list(w["W1"][[0, 2], 0]) == [7, 8]

def test_only_last_block_is_read():
    reply = "x\n```edits\nb1[0] = 1\n```\ntext\n```edits\nb1[1] = 2\n```"
    blk = edits.extract_last_edits_block(reply)
    w = edits.apply_edits(W(), blk).new_weights
    assert w["b1"][0] == 0 and w["b1"][1] == 2

def test_bad_statements_skipped_others_apply():
    r = run("b1[0] = 1\nQ9[0] = 3\nb1[99] = 1\nb1[1] = nan\nb1[2] = 1e9\nW1[0:2, 0] = [1, 2, 3]\nb1[3] = 4")
    assert r.n_applied == 2 and r.n_skipped == 5
    assert r.new_weights["b1"][0] == 1 and r.new_weights["b1"][3] == 4

def test_no_eval_no_code_execution():
    r = run("b1[__import__('os').system('echo x')] = 1\nb1[0] = 1 + 2\nb1[0] = [open('/etc/passwd')]")
    assert r.n_applied == 0

def test_input_never_mutated_and_reset():
    base = W(); r = edits.apply_edits(base, "b1[0] = 5\nRESET\nb1[1] = 3", reset_to=base)
    assert base["b1"].sum() == 0 and r.new_weights["b1"][0] == 0 and r.new_weights["b1"][1] == 3

def test_editable_restriction():
    r = edits.apply_edits(W(), "W1[0, 0] = 1\nb3[0] = 1", editable={"b3"})
    assert r.n_applied == 1 and r.new_weights["W1"][0, 0] == 0

def test_no_block_means_none():
    assert edits.apply_edits(W(), None).new_weights is None

def test_continuation_lines():
    r = run("W1[0:4, 0] = [1, 2,\n 3, 4]")
    assert list(r.new_weights["W1"][:, 0]) == [1, 2, 3, 4]

def test_hash_is_canonical_and_diff_stats():
    a, b = W(), W(); b["b1"][2] = 0.5
    assert edits.weights_sha256(a) == edits.weights_sha256(W())
    assert edits.weights_sha256(a) != edits.weights_sha256(b)
    d = edits.diff_stats(a, b)
    assert d["n_changed"] == 1 and abs(d["l2_delta"] - 0.5) < 1e-12
