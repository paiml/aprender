# L0-1 — P0 discovery (2026-09-06, read-only) and P1 plan

## What exists (cited)
- Load-time gate: `crates/aprender-serve/src/gguf/cuda/mod_parity_gate.rs::parity_gate` — ONE token (BOS, position 0), cosine of CPU vs GPU logits against `PARITY_GATE_COSINE_MIN`, a retry on the unfused FFN path when cosine ∈ [0.90, gate). One position: violates I8 (≥ 64 positions) by construction.
- Sequence check: `crates/aprender-serve/src/gguf/parity.rs::check_parity(tokens)` → per-position `ParityResult` (cosine_similarity …) — the engine behind `apr parity` (`crates/apr-cli/src/commands/parity_03.rs`, cuda-only, `--json` emits cosine per position).
- The downgrade: the gate's failure is a `RealizarError` at CUDA model load; the CLI catches it and prints `[CUDA init failed: …, falling back to CPU]` (`chat_generate_session_02.rs:471`; `run_resolve_tokenizer.rs:132` "Fallback to CPU") — a forced `--gpu` silently becomes CPU. This is the "apr then runs cpu under --gpu".
- `SKIP_PARITY_GATE`: read at `gguf/cuda/mod.rs:349`; SET SILENTLY by `commands/comparison.rs:204` and `commands/diff_benchmark_report.rs:82`; `contracts-staging/gpu-multi-backend-parity-v1.yaml:231` already says "SKIP_PARITY_GATE=1 is forbidden in production" — a contract nothing enforces.
- Model names in the tree: README/BEATS/book name qwen2.5-coder-1.5b(-instruct)(-q4_k_m), qwen2.5-coder-0.5b, qwen3.5-9b, qwen3.5-0.8b; perf-matrix.yaml names qwen2.5-coder-1.5b-instruct; the 0.65.2 dogfood receipts name qwen2.5-coder-1.5b-instruct-q4_k_m.gguf. No 7B is named anywhere the manifest would read — the "7B GREEN" leg of the horizon test needs a manifest entry that a doc actually names, or it is not in the manifest.

## P1 phases (A_i as commands; `.pr/L0-1/accept.sh` re-runs them)
| P | deliverable | A_i |
|---|---|---|
| 1 | `scripts/derive_model_manifest.sh` → `evidence/models/supported.yaml` from README, docs/BEATS.md, book/src, evidence/dogfood/*/*.json, scripts/perf-matrix.yaml (names + where cited); `check_readme_claims.sh`: a model named ∉ manifest → RED (case table) | `bash scripts/derive_model_manifest.sh --check` exit 0; `bash scripts/check_readme_claims.sh --self-test` (new rows) |
| 2 | RED first: `scripts/check_model_parity.sh --manifest` (C14) runs `apr parity --json` per manifest model × host over ≥ 64 positions and compares cosine to the threshold file; recorded 1.5B RED / 7B GREEN on lambda and gx10 BEFORE any kernel edit; must-RED twin fixture (`tests/fixtures/parity/defective/`) | `bash scripts/check_model_parity.sh --self-test`; the recorded RED receipts in `evidence/parity/l0-1/` |
| 3 | five whys by `apr parity` SPC down to the op (N-lane root cause: three model families), then the fix | the horizon test flips to GREEN on 1.5B; revert → RED |
| 4 | REG-15 admission in apr-cli: forced backend never downgrades (code from error.rs + reason), unforced prints `selected: cpu (reason: parity FAILED …)`, effective-config `parity: {status, cosine, positions, threshold, basis}`, `SKIP_PARITY_GATE` prints `override:` + receipts INVALID-CORRECTNESS + asserted unset in dogfood and `ci / gate`; the two silent `set_var` sites removed | `cargo test -p apr-cli --test backend_refusal_case_table` (REG-15 rows); `SKIP_PARITY_GATE=1` exported in the gate step → RED |
| 5 | threshold: `evidence/parity/thresholds.yaml` measured n ≥ 5 per known-good pair, `basis=` (0.98 stays [U] until then) | the file's rows carry n ≥ 5 and a command |
| 6 | contract `contracts/apr-gpu-cpu-parity-v1.yaml`; dogfood P6 falsifier; C14 wired in `apr-dogfood --release`, C4 post-publish, R-8 nightly; every (1.5B, cuda) receipt relabelled INVALID-CORRECTNESS citing #3017 | `pv validate`; `check_guards_are_wired.sh` |

Routing: Fable (N-lane root-cause row); lanes: three model families on agy, async. Hosts: lambda + gx10 via `make fleet-verify ROW=L0-1` (G-11b) — until it exists, ad-hoc ssh to lambda is the one excepted host (I10).
K̂ [U].
