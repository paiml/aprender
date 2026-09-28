# PMAT-3602 — implementation receipt (gh#3602)

- Branch `PMAT-3602-fp8-act-cache`, base `car/0.70.0` @ `11bcf6c2ba`.
- Heads reviewed: `36fc26df7f` (port) and `fd5961ad8a` (positions_judged fix).
- Author: claude-opus-5-5 (aprender-52).

## Finding

The cosine-0.4153 rejection named in the issue no longer reproduces on `car/0.70.0`: #3727 made PMAT-084 activation reuse opt-in, and every `--gpu` run reported `ran=gpu, fell_back=false`.

What remained of the 2-3x was the guard itself. `validate_gpu_first_token` runs a CPU reference forward of the whole prompt on every dense `--gpu` run, so `--gpu` could not reach its first token before `--no-gpu` had finished its prompt.

## Fix

The #3604 receipt is now used on the dense GGUF path. Its key is (model sha256, apr build, device), per the operator ruling of 2026-09-20 on the hybrid path.

The dense path also adds the prefill precision that passed to the device key. That precision is one of `prefill=fp8`, `prefill=fp16` or `prefill=fp16;fp8-rejected`. The reason is #3807: on the dense path the guard can switch FP8 off, and without the precision in the key a receipt could re-enable an FP8 prefill that the guard rejected.

## Measured

Setup: lambda, RTX 4090, apr 0.70.0 (11bcf6c2ba) pinned, qwen2.5-coder-0.5b Q4_K_M.

| run | setup_ms | note |
|---|---|---|
| fresh | 24 726 | guard 22 771 ms, receipt `prefill=fp8` written |
| receipt | 1 539 | sha256 391 ms, same text |
| receipt | 1 548 | sha256 413 ms, same text |
| `--revalidate` | — | guard re-ran (19 086 ms) |
| planted `fp16;fp8-rejected` | 1 431 | skipped, FP8 off, FP8 weight cache cleared, same text |
| planted plain `fp16`, FP8 on | — | validated, naming the device/precision mismatch |

## Falsifiers

`f2_dense_receipt_tests.rs` holds 11 rows. Three mutants were tried, and each turned RED:
- `force_fp16` always false
- no rejected-tag lookup
- `Fp16` receipt vouches for `Fp8`

After the mutation runs the file was restored byte-identical (`cmp` OK).

## Quorum (sonnet-5 + agy gemini-3.1-pro-high + haiku-4-5, read-only; not degraded)

| lane | head | verdict |
|---|---|---|
| claude-sonnet-5 | fd5961ad8a | APPROVE |
| claude-haiku-4-5 | fd5961ad8a | APPROVE |
| agy gemini-3.1-pro-high (conv `a247edfd-32d2-451f-a1ad-b9c709d91227`) | 36fc26df7f | REQUEST_CHANGES |

Disposition of the agy lane's findings, each checked against the cited lines:

1. **positions_judged was one short on the dense path. FIXED in `fd5961ad8a`.** The CPU reference holds `probe.len()+1` logits, and the guard judges `len-1 = probe.len()`. Sonnet confirmed the fix.
   - The lane's "+1 on the hybrid path" is about pre-existing `forward_qwen35.rs` code, outside this diff and not addressed here.
2. **`fp8_decode` is not in the key. Minor, no change, and Sonnet concurs.**
   - FP8 decode fires only at m ≥ 5 (`batched_ffn.rs:111`, `batched_qkv.rs:119`), and this call site is single-request (m = 1). The guard never judges a batched decode, so no receipt could vouch for one.
   - `force_fp16` turns off both FP8 flags and clears the cache.
   - Revisit this if a batched-decode caller is ever routed through the receipt.
3. **The `DeviceMismatch` short-circuit in `decide_dense` is an equivalent mutant. Nit, kept as explicit intent.**
   - Sonnet agrees. Haiku claimed the opposite, that removing it turns `plain_fp16_receipt_does_not_vouch_for_fp8` RED. That claim is wrong: `decide` checks schema, model and build before device, so both calls return the same reason.

## Out of scope (not done here)

- The APR-format CUDA path (`try_apr_cuda_inference`) still validates on every call.
- done_when items 1–2 (the rejection reason and `validate_ms` in `--json`) need the #3606 StageTimings channel. #3606 closed unmerged, so the roadmap's "Item 2 SATISFIED BY #3606" is stale.
- The #3728 FP16 overflow.

## Gates

- `cargo fmt --all -- --check`: 0.
- `cargo test -p aprender-serve --lib --features cuda f2_`: 37 passed (the pure `f2_dense` rows re-run on `fd5961ad8a`: 11 passed).
- `scripts/check_include_files.sh`: OK.
- `scripts/check_roadmap_sorted.sh`: 0.
- `clippy -D warnings`: 27 errors, all in files this diff does not touch; none in the changed files.
