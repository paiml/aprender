"""Shared serialization and hashing helpers of the Laya back office (fixtures.py, train.py, lifecycle.py).

No torch import at module level: data.py re-exports the hash helpers and its self-test runs with numpy +
pyyaml only. `save_f16` imports torch / safetensors lazily, so only its callers pay for the ML stack.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(root):
    """{relative path: sha256} of every file under `root`, in sorted path order."""
    root = Path(root)
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_json(path, obj, compact=False):
    """UTF-8 JSON plus a trailing newline: indent 2, or compact separators when `compact`."""
    text = (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) if compact
            else json.dumps(obj, ensure_ascii=False, indent=2))
    write_bytes(path, (text + "\n").encode("utf-8"))


def jsonl_bytes(rows):
    """`(text, label)` rows as the train/eval JSONL bytes the pins are taken over."""
    return "".join(json.dumps({"text": t, "label": lab}, ensure_ascii=False) + "\n" for t, lab in rows).encode("utf-8")


def f32_list(t):
    """Exact f32 values as JSON numbers (the f64 repr of each f32)."""
    return [float(x) for x in np.asarray(t, dtype=np.float32).reshape(-1)]


def f32_hex_list(t):
    """One f32 bit pattern per value, 8 hex digits (big-endian), the decide-apr-v1 probe convention."""
    h = np.ascontiguousarray(np.asarray(t, dtype=np.float32).reshape(-1)).astype(">f4").tobytes().hex()
    return [h[i:i + 8] for i in range(0, len(h), 8)]


def save_f16(state_dict, path, keep_f32=("temperature",)):
    """The ONE checkpoint dtype policy: every tensor F16 except the names in `keep_f32` (Laya's
    `temperature` buffer stays F32), saved as safetensors with metadata {"format": "pt"}."""
    import torch
    from safetensors.torch import save_file

    out = {k: v.detach().to(torch.float32 if k in keep_f32 else torch.float16).contiguous().cpu()
           for k, v in state_dict.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(path), metadata={"format": "pt"})
