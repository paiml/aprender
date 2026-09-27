"""Content check for the HF cache that CRUX's source-weight engines load (#3971).

Stdlib only, shared by scripts/crux_hf/engine.py and scripts/crux_vllm/engine.py; `verified_source` alone
needs huggingface_hub (both engines' locked environments carry it).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# The HF cache is content-addressed: a snapshot entry is a symlink to blobs/<name>, where <name> is the
# file's LFS sha256 (64 hex) or its git blob sha1 (40 hex). Nothing that loads by repo + revision checks
# the bytes against that name. On lambda, 2026-09-23, the Qwen2.5-Coder-1.5B-Instruct blob named by
# upstream's sha256 held a rewritten float32 file (written THROUGH the snapshot symlink), and vLLM and hf
# both answered garbage from it while the sidecar said "lfs_sha256_matches: true" — measured via the HF API,
# never on disk. So every file a source-weight engine loads is hashed against its own name first.
STAMPS = Path(os.environ.get("CRUX_VERIFIED_BLOBS", str(Path.home() / ".local/share/crux/verified-blobs.json")))


def blob_digest(path: Path, name: str) -> tuple[str, str]:
    """(kind, hex) of `path`'s content, in the scheme its name uses."""
    if len(name) == 64:
        h = hashlib.sha256()
        kind = "sha256"
    elif len(name) == 40:
        h = hashlib.sha1(usedforsecurity=False)
        h.update(b"blob %d\0" % path.stat().st_size)
        kind = "git-blob-sha1"
    else:
        raise RuntimeError(f"cached blob {name!r} is named by neither an LFS sha256 nor a git blob sha1; cannot verify")
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return kind, h.hexdigest()


def verify_snapshot_dir(d: Path) -> int:
    """Hash every file of a snapshot against the blob name it links to; raise, by name, on the first mismatch.
    A (blob, size, mtime_ns) already verified is not re-hashed — a rewrite changes size or mtime, as the
    #3971 one did. Returns how many files were hashed this call."""
    try:
        stamps = json.loads(STAMPS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stamps = {}
    hashed = 0
    entries = sorted(p for p in d.iterdir() if not p.name.startswith("."))
    if not entries:
        raise RuntimeError(f"snapshot {d} is empty")
    for entry in entries:
        if not entry.is_symlink():
            raise RuntimeError(f"snapshot file {entry.name} is not a link into blobs/, so its content has no name "
                               "to be checked against")
        # The PROMISED name is the snapshot link's own target in blobs/. huggingface_hub 1.x may make that a
        # second link into blobs/<xx>/<xet hash> (measured, 1.32.0), so the name is read from the first hop
        # and the CONTENT from the end of the chain — resolving first would check the xet name instead.
        name = Path(os.readlink(entry)).name
        blob = entry.resolve()
        st = blob.stat()
        key = str(blob)
        if stamps.get(key) == [st.st_size, st.st_mtime_ns, name]:
            continue
        kind, got = blob_digest(blob, name)
        hashed += 1
        if got != name:
            raise RuntimeError(f"cached {entry.name} is not the file its name promises: content {kind} {got}, "
                               f"blob named {name} ({st.st_size} bytes) — the local HF cache entry was "
                               "rewritten (#3971)")
        stamps[key] = [st.st_size, st.st_mtime_ns, name]
    STAMPS.parent.mkdir(parents=True, exist_ok=True)
    STAMPS.write_text(json.dumps(stamps, indent=0), encoding="utf-8")
    return hashed


def verified_source(repo: str, revision: str) -> Path:
    """The local snapshot of repo@revision, fetched if absent, with every file's content checked."""
    from huggingface_hub import snapshot_download

    d = Path(snapshot_download(repo, revision=revision,
                               allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]))
    verify_snapshot_dir(d)
    return d
