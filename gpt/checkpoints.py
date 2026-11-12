"""Read shared checkpoints; write generated weights only inside a GPT run."""
from pathlib import Path
from wsarena.store import CheckpointStore


class OverlayStore:
    def __init__(self, shared: Path, generated: Path):
        self.shared = CheckpointStore(shared)
        self.generated = CheckpointStore(generated)

    def get(self, digest):
        return (self.generated if self.generated.has(digest) else self.shared).get(digest)

    def meta(self, digest):
        return (self.generated if self.generated.has(digest) else self.shared).meta(digest)

    def has(self, digest):
        return self.generated.has(digest) or self.shared.has(digest)

    def put(self, weights, meta=None):
        from wsarena.edits import weights_sha256
        digest = weights_sha256(weights)
        if self.shared.has(digest):
            self.shared.get(digest)  # verify before reusing a shared immutable object
            return digest
        return self.generated.put(weights, meta)
