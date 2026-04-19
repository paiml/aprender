# Five Whys — SHIP-TWO-001 EX-04 dogfood miss

**Symptom:** Teacher artifact uploaded to `paiml/qwen2.5-coder-7b-apache-q4k-v1` via
`scripts/ship-two-001/ex-04-upload-hf.sh` (Python `huggingface_hub` via
`uv run --with`) instead of our own `apr publish` subcommand.

**Why 1 — Why was the script used, not `apr publish`?**
Because `apr publish` only uploads files matching `.apr / .safetensors / .gguf`
extensions (`crates/apr-cli/src/commands/publish.rs::find_model_files` lines
217-238). The ship needed to upload `.apr` plus `tokenizer.json` plus a
hand-crafted `manifest.yaml` — the last two are structurally outside the
extension allowlist.

**Why 2 — Why does `apr publish` filter by model-weight extensions only?**
Because the allowlist is hardcoded in `find_model_files` with no override flag,
no glob override, and no `--extra-file` input. The original design assumed the
only sidecar would be an auto-generated `README.md`, which `push_to_hub` writes
automatically (`crates/aprender-core/src/hf_hub/client_impl.rs:189-202`).

**Why 3 — Why was the design only auto-README?**
Because when `apr publish` was first implemented (APR-PUB-001), the publish
workflow was "ship a weights file + HF model card". The model-card path was
`ModelCard::to_huggingface_extended(…)` → `README.md`. There was no concept of
a separate publish-manifest schema yet.

**Why 4 — Why does the ship workflow need a separate manifest schema?**
Because SHIP-TWO-001 later introduced `contracts/publish-manifest-v1.yaml`
(F-PUBLISH-MANIFEST-001) — a 12-field-top + 7-field-provenance schema that
provides falsifiable gates: sha256 stream-match, URL liveness, SPDX validity,
recipe_sha256, parent-chain termination. An auto-generated README cannot
discharge those gates; it's documentation, not a provable contract.

**Why 5 — Why is the product decoupled from the contract schema it must ship?**
Because `apr publish` and `publish-manifest-v1.yaml` evolved on **separate
timelines** with no explicit integration step. `apr publish` shipped assuming
generic model-card uploads; the manifest contract landed later for
SHIP-TWO-001 rigor; nobody closed the product-↔-contract loop. The script is
the symptom; the root cause is that our *own product* doesn't speak our *own
publish schema* yet.

## Root Cause

Product development decoupled from contract development. `apr publish`
does not natively consume `publish-manifest-v1.yaml`.

## Full Fix (scope of F-PUBLISH-EXTRA-001)

Extend `apr publish` to:

1. Accept `--manifest <PATH>`: reads manifest, validates it internally (calls
   `validate_manifest::execute`), uploads the manifest file itself, uses
   manifest-declared sha256/size as pre-upload guard.
2. Accept `--extra-file <PATH>` (repeatable): uploads arbitrary sidecars
   (tokenizer.json, vocab files, special configs).
3. When `--manifest` is passed, suppress README auto-generation (the manifest
   IS the source of truth). A lightweight `README.md` may still ship as a
   redirect to `manifest.yaml`, but it must NOT claim provenance.

After that, rewrite `scripts/ship-two-001/ex-04-upload-hf.sh` to shell into
`apr publish` only, no Python, no `uv run`.

## Why Full Fix (not partial)

Per `feedback_full_problems_pmat_contracts.md` (2026-04-18): every ship step
must use our own product when a product-shaped solution exists. A byte-identical
result from a bypass script does NOT discharge a dogfood gate, because the next
ship will hit the same wall. The product gap must be closed in the product.
