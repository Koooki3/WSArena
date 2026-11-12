"""WEP-1 edit language: deterministic parser and applier (no eval, no exec).

Model-facing syntax is a superset of the numpy-slice style 'edits' language used in the
MIT CSAIL logs (RikiyaT/lm_understand-logs, exp5), so those tasks can be replayed:

    W1[12, 0] = 2.5                    # one entry
    W1[0:6, 3] = [1, 0, -1, 0, 2, 0]   # explicit values along a range
    b1[0:4] = -1                       # fill a range (a:b half-open; ':' = whole axis; a:b:s strided)
    W2[0:2, 0:3] = [[1, -1, .5], [0, 2, 0]]
    W3[:, 2] *= 2                      # also +=  -=

Rules (protocol, frozen with the preregistration):
  * only the LAST fenced ```edits block of a reply is read;
  * statements apply in order; a statement that cannot be read/applied is SKIPPED (the others
    still apply) and counted -- the skip count is reported to the model and logged;
  * values must be plain finite decimals with |v| <= VALUE_LIMIT;
  * only tensors that the task marks editable may be touched; shapes never change.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

import numpy as np

VALUE_LIMIT = 1e6
MAX_STATEMENTS = 5000
MAX_STATEMENT_CHARS = 200_000

_BLOCK_RE = re.compile(r"```[ \t]*edits[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)
_STMT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\[(.*?)\]\s*(=|\+=|-=|\*=)\s*(.+?)\s*$", re.DOTALL)
_NUM_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


class EditError(ValueError):
    """A single statement could not be read or applied."""


def extract_last_edits_block(reply: str) -> str | None:
    blocks = _BLOCK_RE.findall(reply)
    return blocks[-1] if blocks else None


def _split_statements(block: str) -> list[str]:
    """Join continuation lines (a list left open with '[' continues until brackets balance)."""
    out: list[str] = []
    cur = ""
    depth = 0
    for raw in block.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip() and depth == 0:
            continue
        cur = (cur + " " + line.strip()) if cur else line.strip()
        depth += line.count("[") - line.count("]")
        if depth <= 0:
            if cur:
                out.append(cur)
            cur, depth = "", 0
    if cur:
        out.append(cur)
    return out


def _parse_number(tok: str) -> float:
    tok = tok.strip()
    if not _NUM_RE.match(tok):
        raise EditError(f"not a plain decimal: {tok[:30]!r}")
    v = float(tok)
    if not np.isfinite(v) or abs(v) > VALUE_LIMIT:
        raise EditError(f"value out of range: {v}")
    return v


def _parse_values(text: str):
    """Parse a scalar or a (nested) list of plain decimals into float / ndarray."""
    text = text.strip()
    if not text.startswith("["):
        return _parse_number(text)
    # tokenise brackets, commas and numbers only
    if re.search(r"[^\s\[\],0-9eE+\-.]", text):
        raise EditError("only numbers and brackets are allowed in value lists")
    pos = 0

    def parse_list():
        nonlocal pos
        assert text[pos] == "["
        pos += 1
        items = []
        while True:
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                raise EditError("unterminated list")
            c = text[pos]
            if c == "]":
                pos += 1
                return items
            if c == ",":
                pos += 1
                continue
            if c == "[":
                items.append(parse_list())
            else:
                m = re.match(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", text[pos:])
                if not m:
                    raise EditError(f"bad token near {text[pos:pos+12]!r}")
                items.append(_parse_number(m.group(0)))
                pos += m.end()

    lst = parse_list()
    if text[pos:].strip():
        raise EditError("trailing characters after list")
    arr = np.array(lst, dtype=np.float64)
    if arr.dtype == object:
        raise EditError("ragged list")
    return arr


def _parse_index(idx_text: str, shape: tuple[int, ...]):
    parts = [p.strip() for p in idx_text.split(",")]
    if len(parts) != len(shape):
        raise EditError(f"index has {len(parts)} axes, tensor has {len(shape)}")
    out = []
    for p, n in zip(parts, shape):
        if ":" in p:
            segs = p.split(":")
            if len(segs) > 3:
                raise EditError(f"bad slice {p!r}")
            vals = []
            for s in segs:
                s = s.strip()
                vals.append(None if s == "" else int(s))
            sl = slice(*vals)
            start, stop, step = sl.indices(n)
            if step <= 0:
                raise EditError("slice step must be positive")
            out.append(slice(start, stop, step))
        else:
            i = int(p)
            if i < 0:
                i += n
            if not (0 <= i < n):
                raise EditError(f"index {p} out of range for axis of size {n}")
            out.append(i)
    return tuple(out)


@dataclass
class StatementResult:
    text: str
    applied: bool
    error: str | None = None
    tensor: str | None = None
    op: str | None = None
    n_entries: int = 0
    kind: str = "element"  # element | range | scale


@dataclass
class EditResult:
    ok_block: bool                 # an edits block was found
    statements: list[StatementResult] = field(default_factory=list)
    new_weights: dict[str, np.ndarray] | None = None

    @property
    def n_total(self) -> int:
        return len(self.statements)

    @property
    def n_applied(self) -> int:
        return sum(s.applied for s in self.statements)

    @property
    def n_skipped(self) -> int:
        return self.n_total - self.n_applied

    def summary(self) -> dict:
        return {
            "block_found": self.ok_block,
            "statements": self.n_total,
            "applied": self.n_applied,
            "skipped": self.n_skipped,
            "skipped_reasons": [s.error for s in self.statements if not s.applied][:10],
        }


def apply_edits(
    weights: dict[str, np.ndarray],
    block: str | None,
    editable: set[str] | None = None,
    reset_to: dict[str, np.ndarray] | None = None,
) -> EditResult:
    """Apply an edits block to a COPY of `weights`. Never mutates the input."""
    if block is None:
        return EditResult(ok_block=False, new_weights=None)
    new = {k: v.copy() for k, v in weights.items()}
    res = EditResult(ok_block=True)
    stmts = _split_statements(block)
    if len(stmts) > MAX_STATEMENTS:
        stmts = stmts[:MAX_STATEMENTS]
    for text in stmts:
        sr = StatementResult(text=text[:200], applied=False)
        try:
            if text.strip().upper() == "RESET":
                if reset_to is None:
                    raise EditError("RESET unavailable")
                new = {k: v.copy() for k, v in reset_to.items()}
                sr.applied, sr.op, sr.kind = True, "RESET", "reset"
                res.statements.append(sr)
                continue
            if len(text) > MAX_STATEMENT_CHARS:
                raise EditError("statement too long")
            m = _STMT_RE.match(text)
            if not m:
                raise EditError("unreadable statement")
            name, idx_text, op, val_text = m.groups()
            if name not in new:
                raise EditError(f"unknown tensor {name!r}")
            if editable is not None and name not in editable:
                raise EditError(f"tensor {name!r} is not editable")
            sr.tensor, sr.op = name, op
            idx = _parse_index(idx_text, new[name].shape)
            target = new[name][idx]
            vals = _parse_values(val_text)
            sr.kind = "scale" if op == "*=" else ("range" if any(isinstance(i, slice) for i in idx) else "element")
            tshape = np.shape(target)
            if np.ndim(vals) == 0:
                v = np.full(tshape, float(vals)) if tshape else float(vals)
            else:
                v = np.asarray(vals, dtype=np.float64)
                if v.shape != tshape:
                    if v.size == int(np.prod(tshape)) and v.ndim == 1:
                        v = v.reshape(tshape)
                    else:
                        raise EditError(f"value shape {v.shape} != target shape {tshape}")
            if op == "=":
                new_val = v
            elif op == "+=":
                new_val = target + v
            elif op == "-=":
                new_val = target - v
            else:
                new_val = target * v
            if not np.all(np.isfinite(new_val)):
                raise EditError("non-finite result")
            new[name][idx] = new_val
            sr.applied = True
            sr.n_entries = int(np.prod(tshape)) if tshape else 1
        except (EditError, ValueError, TypeError) as e:
            sr.error = str(e)[:160]
        res.statements.append(sr)
    res.new_weights = new
    return res


# ---------------------------------------------------------------- diff statistics

def weights_sha256(weights: dict[str, np.ndarray]) -> str:
    """Canonical hash: sorted names, shape, float64 little-endian bytes."""
    h = hashlib.sha256()
    for k in sorted(weights):
        a = np.ascontiguousarray(weights[k], dtype="<f8")
        h.update(k.encode())
        h.update(json.dumps(list(a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def diff_stats(old: dict[str, np.ndarray], new: dict[str, np.ndarray]) -> dict:
    per = {}
    n_changed = 0
    tot_params = 0
    l2sq = 0.0
    base_sq = 0.0
    linf = 0.0
    for k in sorted(old):
        d = new[k] - old[k]
        ch = int(np.count_nonzero(d))
        per[k] = {"changed": ch, "size": int(old[k].size), "l2": float(np.linalg.norm(d))}
        n_changed += ch
        tot_params += old[k].size
        l2sq += float(np.sum(d * d))
        base_sq += float(np.sum(old[k] * old[k]))
        linf = max(linf, float(np.max(np.abs(d))) if d.size else 0.0)
    return {
        "n_changed": n_changed,
        "n_params": int(tot_params),
        "frac_changed": n_changed / max(tot_params, 1),
        "l2_delta": l2sq ** 0.5,
        "rel_l2_delta": (l2sq ** 0.5) / max(base_sq ** 0.5, 1e-12),
        "linf_delta": linf,
        "per_tensor": per,
    }
