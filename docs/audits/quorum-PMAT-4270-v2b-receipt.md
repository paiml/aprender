# Quorum receipt — PMAT-4270 V2b (qa dense gates on DenseSession)

Author: claude-opus-5-5. Base: origin/rc/0.69.3-rc.2 @ 3644c2677. Shape: sonnet-5 + 1 agy gemini-3.1-pro-high + haiku-4-5 (operator 2026-09-24).

## Round 2 — AGREED 3/3 PASS on head 2efa2224a (brief sha256 94795f31…d17, 18395 B, same brief to all lanes)
| lane | model | verdict |
|------|-------|---------|
| agy  | gemini-3.1-pro-high (measured) | PASS — `quorum-PMAT-4270-v2b-r2.json` |
| claude | claude-sonnet-5 | PASS, 0 findings |
| claude | claude-haiku-4-5 | PASS, 0 findings |

## Round 1 — NOT AGREED on 6d556ccb6 (3 FAIL), adjudicated
- sonnet-5: V2b hunks in output_verification.rs / golden_output.rs / speedup.rs not rustfmt-clean; `cargo fmt --all -- --check` does not reach them. **Real** — fixed in 2efa2224a (own hunks only).
- haiku-4-5: "missing comma after match-arm block" (output_verification.rs:843). **False** — block arms need no comma; `cargo check -p apr-cli --lib --features cuda` rc=0.
- gemini-3.1-pro-high: (a) no session reset between gpu_isolation turns / throughput iterations; (b) use-after-move in speedup.rs. **False** — (a) `Session::extends` (session.rs) is true only for a STRICT extension of the held tokens; any other prompt restarts at position 0 (`advance_and_choose` start=0 / `advance_to`), so each turn rewinds; (b) compiles under `--features cuda`.

## Evidence (release mode, rc/0.69.3-rc.2 base)
- `cargo test --release -p apr-cli --lib -- qa_dense_session --test-threads=1`: 2 passed (model present; neither witness returns early), on 6d556ccb6 and 2efa2224a.
- Mutant `if false && require_gpu && !used_gpu` → `a_gpu_gate_refuses_a_turn_served_on_the_cpu` FAILED (rc=101); restored.
- `cargo check -p apr-cli --lib --features cuda` rc=0. Not run on a GPU: the CUDA gate path compiled only.
