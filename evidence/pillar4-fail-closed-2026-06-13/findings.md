# Pillar-4 fail-closed correctness beat — measured head-to-head (2026-06-13)

**Claim (mission headline):** apr provably refuses to ship garbage; the incumbents
(llama.cpp / Ollama) silently accept and run a semantically-broken model.

**Host:** noah-Lambda-Vector (RTX 4090). **Tools:** `apr validate` (cuda build),
`~/src/llama.cpp/build/bin/llama-cli` (CUDA build).

## Method
Took a copy of `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` (qwen2 arch, supported by
both apr and llama.cpp) and zeroed the data region of `blk.0.ffn_down.weight`
(11,289,600 bytes) via `gguf.GGUFReader` offset + binary patch. The file remains
structurally valid (magic/version/tensor metadata/shape intact); only the weight
*values* are dead. This is the discriminating case: a parse-valid, semantically
corrupt artifact.

## Result (same broken file, both tools)

| Tool | Verdict | Evidence |
|------|---------|----------|
| **`apr validate`** | **REJECT** | `blk.0.ffn_down.weight ✗ FAIL` — `[F-DATA-QUALITY-001] All values are zero (uninitialized?)`; `[F-DATA-QUALITY-003] L2 norm ~0`; `[F-DATA-QUALITY-003] All values identical`. Other tensors PASS (no false positive). |
| **`llama-cli`** | **ACCEPT** | 0 load-error lines; model loaded and ran (`Generation: 398 t/s`). No semantic weight-quality gate exists in the GGUF loader. |

Note: with only 1 of 28 layers' down-proj zeroed the model still answered a trivial
prompt — the point is not output quality but that **llama.cpp performs no semantic
validation and will run a corrupt checkpoint silently** (exit 0, no warning), where
apr fails closed.

## CI-gated form
The apr-side invariant is enforced deterministically (no model file / GPU needed) by
`crates/aprender-serve/tests/beat_fail_closed_garbage.rs`:
`validate_weight`/`validate_embedding` reject all 10 broken classes (6 weight + 4
embedding) and accept healthy tensors. Contract:
`contracts/apr-fail-closed-garbage-beat-v1.yaml` (beat-benchmark, Pillar 4).

The incumbent measurement above is the head-to-head evidence; wiring llama.cpp/Ollama
into per-PR CI is out of scope (needs model files + runtimes), so the falsifiable CI
gate is the apr-side fail-closed invariant.
