#!/usr/bin/env python3
"""Derive the committed deterministic real-weight MiniLM slice (D-09).

Downloads the PINNED revision of sentence-transformers/all-MiniLM-L6-v2, records and
fail-closed verifies the upstream file digests, then index-slices the real tensors into
a ~0.5 MB encoder that every CI job can run against REAL WEIGHT VALUES rather than
synthetic shapes.

Run:  uv run python slice_model.py
This is a deliberate developer workflow. It is never wired into CI (D-12).

HEAD BOUNDARIES ARE PRESERVED (the point of the [0:64] slice)
--------------------------------------------------------------
MiniLM is hidden 384 / 12 heads, so head_dim = 384 / 12 = 32. HF lays the Q/K/V
projection out as [num_heads * head_dim, hidden] with head h occupying output rows
[h*32, (h+1)*32). Taking output rows [0:64] therefore yields COMPLETE original heads 0
and 1 -- not four half-heads. Slicing to 4 x 16 instead would cut each real 32-dim head
in half and manufacture a synthetic attention structure out of real numbers, weakening
exactly the property D-09 exists to protect. ``assert_head_boundaries`` below proves the
offsets rather than trusting this paragraph.

WHY DIGESTS ARE VERIFIED FAIL-CLOSED
-------------------------------------
The revision sha is the primary integrity anchor, but a re-pointed or corrupted artifact
must not silently regenerate fixtures (T-1-06). On the first run the per-file sha256s are
recorded into upstream_manifest.json; on every subsequent run they are re-verified and
any mismatch aborts with a non-zero exit.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

import corpus

# --- pins --------------------------------------------------------------------------
REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Immutable revision sha, never a branch name (T-1-06).
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

UPSTREAM_FILES = [
    "config.json",
    "tokenizer.json",
    "model.safetensors",
    "modules.json",
    "1_Pooling/config.json",
]

# --- slice geometry ----------------------------------------------------------------
SLICE_LAYERS = 2
SLICE_HIDDEN = 64
SLICE_HEADS = 2
SLICE_HEAD_DIM = 32
SLICE_INTERMEDIATE = 256
SLICE_POSITIONS = 64
SOURCE_HEAD_INDICES = [0, 1]

# Special tokens always retained in the slice vocabulary regardless of the corpus.
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "crates" / "aprender-core" / "tests" / "fixtures" / "setfit"
BUILD_DIR = Path(__file__).resolve().parent / "build"


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_pinned() -> dict[str, Path]:
    """Download each pinned file at the exact revision."""
    paths: dict[str, Path] = {}
    for name in UPSTREAM_FILES:
        paths[name] = Path(hf_hub_download(REPO_ID, name, revision=REVISION))
    return paths


def verify_or_write_upstream_manifest(paths: dict[str, Path]) -> dict[str, str]:
    """Record digests on first run; fail closed on any later mismatch."""
    digests = {name: sha256_file(p) for name, p in paths.items()}
    manifest_path = FIXTURE_DIR / "upstream_manifest.json"

    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text())
        if recorded.get("revision") != REVISION:
            sys.exit(
                f"FATAL: upstream_manifest.json pins revision {recorded.get('revision')} "
                f"but this script pins {REVISION}. Refusing to regenerate."
            )
        drift = [
            (n, recorded["files"].get(n), digests[n])
            for n in digests
            if recorded["files"].get(n) != digests[n]
        ]
        if drift:
            for name, want, got in drift:
                print(f"  {name}\n    recorded: {want}\n    fetched : {got}", file=sys.stderr)
            sys.exit(
                "FATAL: upstream digest mismatch. A pinned artifact changed underneath "
                "the pin; refusing to regenerate fixtures from it (T-1-06)."
            )
        print(f"upstream digests verified against {manifest_path.name} ({len(digests)} files)")
    else:
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({"revision": REVISION, "files": digests}, indent=2, sort_keys=True) + "\n"
        )
        print(f"wrote {manifest_path.name} ({len(digests)} files)")
    return digests


def assert_head_boundaries(num_heads: int, hidden: int) -> None:
    """Prove [0:SLICE_HIDDEN] is a whole number of COMPLETE original heads."""
    head_dim = hidden // num_heads
    if head_dim != SLICE_HEAD_DIM:
        sys.exit(f"FATAL: source head_dim {head_dim} != expected {SLICE_HEAD_DIM}")
    if SLICE_HIDDEN % head_dim != 0:
        sys.exit(
            f"FATAL: slice hidden {SLICE_HIDDEN} is not a whole multiple of the source "
            f"head_dim {head_dim} -- the slice would cut real heads in half."
        )
    kept = SLICE_HIDDEN // head_dim
    if kept != SLICE_HEADS or SOURCE_HEAD_INDICES != list(range(kept)):
        sys.exit(f"FATAL: expected to keep heads {SOURCE_HEAD_INDICES}, computed {kept} heads")
    # The concrete offsets, proven not asserted-by-comment: head h occupies output rows
    # [h*head_dim, (h+1)*head_dim). Rows [0:64) == head 0 [0:32) + head 1 [32:64).
    for h in SOURCE_HEAD_INDICES:
        start, stop = h * head_dim, (h + 1) * head_dim
        assert stop <= SLICE_HIDDEN, f"head {h} rows [{start}:{stop}) escape the slice"
    assert SOURCE_HEAD_INDICES[-1] * head_dim + head_dim == SLICE_HIDDEN, (
        "the kept heads must exactly tile [0:SLICE_HIDDEN) with no partial head"
    )
    print(
        f"head boundaries OK: source {num_heads}x{head_dim} -> slice "
        f"{SLICE_HEADS}x{head_dim} = rows [0:{SLICE_HIDDEN}) = complete heads "
        f"{SOURCE_HEAD_INDICES}"
    )


def build_vocab_closure(tokenizer_path: Path) -> tuple[dict[str, int], list[int]]:
    """Closure of every token id the corpus produces, plus specials, densely remapped.

    Ids are sorted ascending so the slice row order is deterministic and reviewable.
    [PAD] is canonical id 0, so it lands on slice row 0 and pad_token_id stays 0.
    """
    tok = Tokenizer.from_file(str(tokenizer_path))
    ids: set[int] = set()
    for tid in (tok.token_to_id(t) for t in SPECIAL_TOKENS):
        if tid is None:
            sys.exit("FATAL: a required special token is missing from the tokenizer")
        ids.add(tid)
    for text in corpus.all_texts():
        ids.update(tok.encode(text).ids)

    slice_to_orig = sorted(ids)
    orig_to_slice = {str(o): i for i, o in enumerate(slice_to_orig)}
    if slice_to_orig[0] != 0:
        sys.exit("FATAL: expected canonical [PAD]=0 to be the lowest retained id")
    return orig_to_slice, slice_to_orig


def slice_state_dict(sd: dict[str, torch.Tensor], keep_rows: list[int]) -> dict[str, torch.Tensor]:
    """Index-slice the real tensors. No re-initialisation anywhere (D-09)."""
    H = SLICE_HIDDEN
    I = SLICE_INTERMEDIATE
    P = SLICE_POSITIONS
    rows = torch.tensor(keep_rows, dtype=torch.long)
    out: dict[str, torch.Tensor] = {}

    # Embeddings. word_embeddings keeps only the retained vocabulary rows.
    out["embeddings.word_embeddings.weight"] = sd["embeddings.word_embeddings.weight"][rows, :H].clone()
    out["embeddings.position_embeddings.weight"] = sd["embeddings.position_embeddings.weight"][:P, :H].clone()
    out["embeddings.token_type_embeddings.weight"] = sd["embeddings.token_type_embeddings.weight"][:, :H].clone()
    out["embeddings.LayerNorm.weight"] = sd["embeddings.LayerNorm.weight"][:H].clone()
    out["embeddings.LayerNorm.bias"] = sd["embeddings.LayerNorm.bias"][:H].clone()

    for layer in range(SLICE_LAYERS):
        p = f"encoder.layer.{layer}"
        # Q/K/V: [out=num_heads*head_dim, in=hidden] -> rows [0:H) are complete heads 0,1.
        for proj in ("query", "key", "value"):
            out[f"{p}.attention.self.{proj}.weight"] = sd[f"{p}.attention.self.{proj}.weight"][:H, :H].clone()
            out[f"{p}.attention.self.{proj}.bias"] = sd[f"{p}.attention.self.{proj}.bias"][:H].clone()
        # Attention output projection: [hidden, num_heads*head_dim].
        out[f"{p}.attention.output.dense.weight"] = sd[f"{p}.attention.output.dense.weight"][:H, :H].clone()
        out[f"{p}.attention.output.dense.bias"] = sd[f"{p}.attention.output.dense.bias"][:H].clone()
        out[f"{p}.attention.output.LayerNorm.weight"] = sd[f"{p}.attention.output.LayerNorm.weight"][:H].clone()
        out[f"{p}.attention.output.LayerNorm.bias"] = sd[f"{p}.attention.output.LayerNorm.bias"][:H].clone()
        # FFN.
        out[f"{p}.intermediate.dense.weight"] = sd[f"{p}.intermediate.dense.weight"][:I, :H].clone()
        out[f"{p}.intermediate.dense.bias"] = sd[f"{p}.intermediate.dense.bias"][:I].clone()
        out[f"{p}.output.dense.weight"] = sd[f"{p}.output.dense.weight"][:H, :I].clone()
        out[f"{p}.output.dense.bias"] = sd[f"{p}.output.dense.bias"][:H].clone()
        out[f"{p}.output.LayerNorm.weight"] = sd[f"{p}.output.LayerNorm.weight"][:H].clone()
        out[f"{p}.output.LayerNorm.bias"] = sd[f"{p}.output.LayerNorm.bias"][:H].clone()

    return out


def resolve_apr_bin() -> str:
    """Resolve the apr binary via scripts/apr_bin.sh.

    Never a bare `apr` and never a hardcoded absolute path: four apr binaries were once
    found coexisting on one dev box and a bare `apr` resolved to a 26-day-old copy.
    apr_bin.sh additionally PROVES the binary was built from HEAD.
    """
    resolved = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "apr_bin.sh")],
        capture_output=True,
        text=True,
    )
    if resolved.returncode != 0:
        sys.exit(f"FATAL: could not resolve a fresh apr binary:\n{resolved.stderr}")
    return resolved.stdout.strip()


def convert_to_apr(safetensors_path: Path, apr_path: Path, slice_config: dict) -> str:
    """Convert the slice to APR F32 using the PINNED apr binary.

    `apr import` is the safetensors -> APR path. `apr convert` was tried FIRST per the
    plan and is not applicable: its own help states the input is a "Path to .apr model
    file" (it is an APR->APR quantize/compress optimizer and rejects the call outright
    with "At least one of --quantize or --compress must be specified"). `apr import
    --arch bert` maps the sliced BERT tensor names natively -- no Python-driven JSON
    dump / dev-only Rust writer fallback was needed.

    The importer reads hyperparameters from a config.json beside the weights; without one
    it errors unless --allow-no-config is passed, and inferring dims from tensor shapes is
    exactly the guessing this step must avoid. So the slice's own HF-shaped config is
    written next to the safetensors first.
    """
    apr_bin = resolve_apr_bin()

    hf_config = {
        "architectures": ["BertModel"],
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": slice_config["hidden_act"],
        "hidden_dropout_prob": 0.1,
        "hidden_size": slice_config["hidden"],
        "initializer_range": 0.02,
        "intermediate_size": slice_config["intermediate"],
        "layer_norm_eps": slice_config["layer_norm_eps"],
        "max_position_embeddings": slice_config["positions"],
        "model_type": "bert",
        "num_attention_heads": slice_config["heads"],
        "num_hidden_layers": slice_config["num_layers"],
        "pad_token_id": slice_config["pad_token_id"],
        "position_embedding_type": "absolute",
        "type_vocab_size": slice_config["type_vocab_size"],
        "vocab_size": slice_config["vocab"],
    }
    (safetensors_path.parent / "config.json").write_text(
        json.dumps(hf_config, indent=2, sort_keys=True) + "\n"
    )

    # `apr import` has no --force; regeneration must start from a clean output path.
    apr_path.unlink(missing_ok=True)
    proc = subprocess.run(
        [apr_bin, "import", str(safetensors_path), "-o", str(apr_path), "--arch", "bert"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.exit(
            "FATAL: `apr import` failed on the sliced checkpoint.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    print(f"converted via `apr import --arch bert` ({apr_bin})")
    return apr_bin


def main() -> None:
    torch.manual_seed(0)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    paths = fetch_pinned()
    digests = verify_or_write_upstream_manifest(paths)

    config = json.loads(paths["config.json"].read_text())
    assert_head_boundaries(config["num_attention_heads"], config["hidden_size"])

    orig_to_slice, slice_to_orig = build_vocab_closure(paths["tokenizer.json"])
    print(f"vocab closure: {len(slice_to_orig)} canonical ids retained")

    sd = load_file(str(paths["model.safetensors"]))
    sliced = slice_state_dict(sd, slice_to_orig)

    slice_st = BUILD_DIR / "slice_model.safetensors"
    save_file(sliced, str(slice_st))
    print(f"wrote {slice_st.relative_to(REPO_ROOT)} ({slice_st.stat().st_size / 1024:.0f} KB)")

    # Tokenizer bytes must live in-repo: offline tokenizer-parity tests and the
    # import tokenizer-hash check both load them.
    shutil.copyfile(paths["tokenizer.json"], FIXTURE_DIR / "tokenizer.json")

    slice_config = {
        "num_layers": SLICE_LAYERS,
        "hidden": SLICE_HIDDEN,
        "heads": SLICE_HEADS,
        "head_dim": SLICE_HEAD_DIM,
        "intermediate": SLICE_INTERMEDIATE,
        "vocab": len(slice_to_orig),
        "positions": SLICE_POSITIONS,
        # Read from the pinned config, never assumed.
        "layer_norm_eps": config["layer_norm_eps"],
        "hidden_act": config["hidden_act"],
        "pad_token_id": config["pad_token_id"],
        "type_vocab_size": config["type_vocab_size"],
        "source_revision": REVISION,
        "source_head_indices": SOURCE_HEAD_INDICES,
        "source_hidden": config["hidden_size"],
        "source_heads": config["num_attention_heads"],
        "source_layers": config["num_hidden_layers"],
        "tokenizer_sha256": digests["tokenizer.json"],
    }
    (FIXTURE_DIR / "slice_config.json").write_text(
        json.dumps(slice_config, indent=2, sort_keys=True) + "\n"
    )

    (FIXTURE_DIR / "vocab_remap.json").write_text(
        json.dumps(
            {"orig_to_slice": orig_to_slice, "slice_to_orig": slice_to_orig},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    apr_path = FIXTURE_DIR / "slice_model.apr"
    convert_to_apr(slice_st, apr_path, slice_config)
    print(f"wrote {apr_path.relative_to(REPO_ROOT)} ({apr_path.stat().st_size / 1024:.0f} KB)")

    print("\nnext: uv run python generate_fixtures.py")


if __name__ == "__main__":
    main()
