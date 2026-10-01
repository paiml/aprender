#!/usr/bin/env python3
"""Materialise the FULL pinned MiniLM checkpoint for the D-10 gated parity suite.

Run:  uv run python fetch_full_weights.py
Developer workflow only. NEVER called from CI -- the ~90 MB checkpoint is deliberately
not vendored in git (D-10), and no test in the default gate depends on this artifact.

WHY THIS PRODUCES AN APR AND NOT JUST A CHECKPOINT
---------------------------------------------------
The gated suite loads through the PUBLIC full-pin import path
(`SetFitMiniLm::from_pretrained_dir`, plan 01-07), which validates config.json, the
tokenizer bytes, and the module graph -- not just weights. Dropping a bare safetensors
into a cache directory would leave that path missing inputs it validates, so the failure
would surface at wave 7 as "the loader is broken" rather than "the fixture step never
produced what the loader reads". So this script emits every input that path consumes.

WHAT full_manifest.json IS FOR
-------------------------------
It records the revision, the SOURCE safetensors sha256, AND the sha256 of the APR this
script produced. A D-10 run can then prove WHICH BYTES it tested instead of asserting it
by intent (CLAUDE.md verification rule 2).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from slice_model import (
    FIXTURE_DIR,
    REPO_ID,
    REVISION,
    UPSTREAM_FILES,
    resolve_apr_bin,
    sha256_file,
)

DEFAULT_DIR = Path.home() / ".cache" / "aprender" / "minilm-l6-v2-1110a243"


def target_dir() -> Path:
    return Path(os.environ.get("APRENDER_MINILM_DIR", str(DEFAULT_DIR))).expanduser()


def main() -> None:
    dest = target_dir()
    dest.mkdir(parents=True, exist_ok=True)

    manifest_path = FIXTURE_DIR / "upstream_manifest.json"
    if not manifest_path.exists():
        sys.exit(
            f"FATAL: {manifest_path} is missing. Run `uv run python slice_model.py` first "
            "so the pinned upstream digests are recorded."
        )
    recorded = json.loads(manifest_path.read_text())
    if recorded.get("revision") != REVISION:
        sys.exit(
            f"FATAL: upstream_manifest.json pins {recorded.get('revision')} but this "
            f"script pins {REVISION}."
        )

    # (1)+(2) fetch at the pinned revision and verify fail-closed against the manifest.
    fetched: dict[str, Path] = {}
    for name in UPSTREAM_FILES:
        src = Path(hf_hub_download(REPO_ID, name, revision=REVISION))
        digest = sha256_file(src)
        want = recorded["files"].get(name)
        if digest != want:
            sys.exit(
                f"FATAL: digest mismatch for {name}\n  recorded: {want}\n  fetched : {digest}\n"
                "A pinned artifact changed underneath the pin; refusing to build a "
                "parity artifact from it (T-1-06)."
            )
        fetched[name] = src
    print(f"upstream digests verified ({len(fetched)} files) @ {REVISION[:12]}")

    # (4) copy every input the public from_pretrained_dir path validates.
    for name, src in fetched.items():
        out = dest / name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)

    # (3) convert to APR with the PINNED binary (never a bare or hardcoded `apr`).
    apr_bin = resolve_apr_bin()
    apr_path = dest / "full_model.apr"
    apr_path.unlink(missing_ok=True)
    proc = subprocess.run(
        [apr_bin, "import", str(dest / "model.safetensors"), "-o", str(apr_path), "--arch", "bert"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"FATAL: `apr import` failed.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

    # (5) prove which bytes were produced, from which source.
    full_manifest = {
        "revision": REVISION,
        "repo_id": REPO_ID,
        "source_safetensors_sha256": recorded["files"]["model.safetensors"],
        "apr_sha256": sha256_file(apr_path),
        "apr_bytes": apr_path.stat().st_size,
        "apr_path": str(apr_path),
        "produced_by": Path(__file__).name,
    }
    (dest / "full_manifest.json").write_text(
        json.dumps(full_manifest, indent=2, sort_keys=True) + "\n"
    )

    print(f"wrote {apr_path} ({apr_path.stat().st_size / 1_048_576:.1f} MB)")
    print(f"wrote {dest / 'full_manifest.json'}")
    print("\n(6) next, run the gated real-weight suite:\n")
    print(f"    APRENDER_MINILM_DIR={dest} \\")
    print("      cargo test -p aprender-core --features setfit,model-tests -- --ignored")


if __name__ == "__main__":
    main()
