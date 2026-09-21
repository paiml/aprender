# PMAT-3748 — implementation receipt

Ticket: #3748 (0.69.1; re-scoped from perf to CORRECTNESS by the cop, 2026-09-21T21:20:57Z; done_when 1 amended 21:27:14Z).
Code commit: `e771c8995`. The branch head (`4baa0cb98`) adds only the fragment and the aggregate.

## The hole, on device (before)
Binary X = `apr 0.69.0 (5615e7afe)`; X2 = X with one byte appended. Same version, DIFFERENT executable. Qwen3.5-9B-Q4_K_M (sha256 `03b74727a860…`), a fresh `APR_F2_RECEIPT_DIR`, `apr run --gpu`, under `gpu-q --prio 1`.

| host | run | F2 source | wall |
|---|---|---|---|
| lambda (RTX 4090) | X cold | fresh (validated in 4.6 s) | 19.0 s |
| lambda | X warm | receipt, **re-hashing the model: 2436 ms** | 14.3 s |
| lambda | **X2** | **receipt: SKIPPED on X's receipt** ("receipt matches … apr 0.69.0") | 14.8 s |
| gx10 (GB10) | X cold | fresh (17.4 s) | 44.6 s |
| gx10 | X warm | receipt, **re-hashing: 10147 ms** | 27.7 s |
| gx10 | **X2** | **receipt: SKIPPED on X's receipt** | 25.4 s |

A different binary served on another build's validation, on both architectures. This is the class aprender-37's vacuous Qwen3.5 cells fell into.

## After, at `e771c8995` (gx10 binary built from `4baa0cb98`, same code)
A = this build; B = A with one byte appended. The same model and prompt, with a fresh receipt dir.

| host | run | F2 source | model / exe hash | wall |
|---|---|---|---|---|
| lambda | A cold | fresh (no receipt for this (model, executable, device)), validated 3.9 s | both hashed, 2503 ms | 18.1 s |
| lambda | A warm | **receipt, exe=afb521a6f11acc79** | **both cached, 0 ms** | **11.9 s** |
| lambda | **B** | **fresh**: no receipt for B's triple, validated 7.4 s | model cached, exe hashed, 63 ms | 20.6 s |
| lambda | A again (A→B→A) | **receipt, exe=afb521a6f11acc79** | cached, 0 ms | 12.0 s |
| gx10 | A cold | fresh, validated 9.0 s | hashed, 10048 ms | 34.2 s |
| gx10 | A warm | **receipt, exe=71c8f297b2719f2c** | **cached, 0 ms** | **14.8 s** |
| gx10 | **B** | **fresh**, validated 9.3 s | model cached, exe hashed, 134 ms | 24.7 s |
| gx10 | A again | **receipt** | cached, 3 ms | 15.4 s |

Afterwards the receipt dir holds `<model>-<exe16>-<device>.json` for A and for B side by side, plus `sha256-cache/`. The before dir holds one `<model>.json`.

- **done_when 1:** a different executable never skips (B validates). Builds A→B→A never overwrite each other (A's third run skips).
- **done_when 2:** a warm run hashes nothing (0 ms, down from 2.4 s on lambda and 10.1 s on gx10).
- **done_when 3:** the warm-run F2 cost is 0 ms of hashing plus 0 ms of validation. Warm wall time fell 14.3 → 11.9 s on lambda and 27.7 → 14.8 s on gx10. The cold run is unchanged in kind: it validates. Its hash is ~2.5 s on lambda and ~10 s on gx10, paid once per file identity.
- **Cop (b):** every F2 line names its source with the executable (`[source=receipt exe=…]` / `[source=fresh exe=…]`) and how each hash was obtained. `F2Outcome.exe_sha256` carries it for cells.

## Mutants (each applied, run, reverted; the restored tree is 22/22 green)
| mutant | tests turned RED |
|---|---|
| key on the version (the exe comparison dropped from `decide`) | `falsifier_a_planted_receipt_from_a_different_executable_revalidates`, `falsifier_the_same_version_from_another_executable_never_skips` |
| key on the model only (one file per model) | `builds_a_then_b_then_a_never_overwrite_each_other`, `a_second_device_is_a_second_receipt_file` |
| stale cache after an in-place rebuild (mtime ignored) | `the_hash_cache_hits_warm_and_misses_on_any_identity_change` |

## Other checks
- `cargo test -p aprender-serve --lib f2_receipt`: 22 passed.
- `cargo test -p apr-cli --lib`: 7300 passed.
- The cuda release build succeeds on both hosts.
- The existing #3604 falsifiers (wrong model, wrong device, absent, unreadable, old schema, `--revalidate`) are kept, re-keyed to the executable.

## Not in this row
Recording `source` inside ladder/derived cells is the judge's side (#3745 S2 / #3712 B2). This row supplies the field and the log line. The qwen3moe F2 (#3714) writes no receipt; it validates every run.
