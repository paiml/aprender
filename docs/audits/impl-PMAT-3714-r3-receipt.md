# PMAT-3714 R3 — port to batch/0.70.0, measured at the port's own head

Author: aprender-1c (claude-opus-5-5), 2026-09-24. Branch `PMAT-3714-moe-on-070`.

## Why a port
R2 (`PMAT-3714-r2-on-v2@6f6fa9a83`, aprender-eb) was measured green but never
landed: `crates/apr-cli/src/commands/parity_moe.rs` exists in none of
`origin/main`, `origin/batch/0.70.0`, `v0.69.1`, `v0.69.3-rc.1/rc.2`, and the
`fold/3714-moe-into-0.69.1` branch never got a PR. This row merges R2 into
batch/0.70.0.

## Port edits
- `golden_output.rs` conflict: batch's `cuda_device_present()` (#3711) kept
  together with R2's 8-byte `model_header::read_prefix` magic (#3750).
- `inference_result.rs`: batch's #3718 context clamp read `canonical_arch`,
  which R2 removed. `is_moe = moe_forward_handles(..)` is now bound once before
  `model` moves and is used for both the dispatch and the clamp.

## Two defects the at-head measurement found (not visible in R2's evidence)
1. **qa throughput counted setup as generation.** At `0979b3481`,
   `apr qa` on Qwen3-30B-A3B-Instruct-2507 returned **rc=5**: `throughput 8.6 tok/s
   < 10`, while `apr run --gpu` decoded the same file at ~80 tok/s. The MoE GPU
   dispatch builds the CUDA model (~1.7–4.9 s) and runs the F2 guard (~2–3.5 s)
   inside the window `inference_result` times, which breaks
   `inference_ms` = "generation only, excludes model load" (batch.rs:51).
   Fix `6d8d5b155`: the dispatch returns its setup ms, and the caller subtracts it.
   After the fix, throughput is **45.1 tok/s** (2507) and **59.9 tok/s** (Coder),
   against 12.1 before.
2. **The throughput label came from the build, not from what ran.** The gate
   printed `hybrid forward, GPU #3090` whenever the build had cuda. That was
   wrong for a MoE file, and wrong for any run that fell back to the CPU. Fix
   `805803f7b`: the label is derived from the timed runs' `used_gpu`. Its case
   table is `runtime_backend_label_follows_used_gpu_not_the_build`, and a mutant
   that swaps the all-GPU arm to CPU turns it red (nextest rc=100).

## done_when, measured at `6d8d5b155` (lambda, RTX 4090)
Evidence: `evidence/3714/r3-port-at-head-lambda.txt`. The binary was pinned
by `scripts/apr_bin.sh`, which reported `apr 0.69.0 (6d8d5b155)`. A stale
build was refused once, and the pin forced a rebuild.

| file | `apr run --gpu` | `apr parity --assert` | `apr qa --json` |
|---|---|---|---|
| Qwen3-Coder-30B-A3B-Instruct-Q4_K_M | rc=0, used_gpu=true, ran=gpu, fell_back=false, "2 + 2 = 4" | rc=0, min cosine 1.000000 | rc=0, 12 gates, 7 executed, 0 failed; throughput 59.9 tok/s, labelled "GPU on every timed run" |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | rc=0, used_gpu=true, ran=gpu, fell_back=false, "2 + 2 = 4." | rc=0, min cosine 1.000000 | rc=0, 12 gates, 7 executed, 0 failed; throughput 45.1 tok/s, labelled "GPU on every timed run" |

The four skipped qa gates (ollama_parity, gpu_speedup, format_parity,
gpu_state_isolation) are dense-loader gates. Each one says so and names the
routed-expert forward. classifier_head is opt-in.

## Checks
- `cargo nextest run --profile ci -p apr-cli -p aprender-serve --lib` at
  `0979b3481`: 23300 of 23304 passed. The 4 failures were `cargo-memcap`
  SIGKILLs at its 16G lambda cap, all in long-context memory tests
  (`test_{super,ultra,mega}_long_context_memory_bound`,
  `qwen35_gqa_4b_matches_llama_cpp_greedy_tokens`), none in a file this branch
  touches.
- Targeted moe|parity|golden|qa|model_header|decode_budget|run_report filter:
  1285/1285. After the fixes, the qa|throughput|runtime_backend_label filter:
  117/117.
- `cargo fmt --all -- --check` passes. `check_roadmap_fragment_required.sh`
  passes. Clippy (cuda) reports nothing new in any file this branch touches;
  the aprender-data build.rs and iq*.rs findings were already on batch.

## Not done here
- gx10 was not re-measured at this head (gx10 builds are forbidden: its disk is
  full). R2's gx10 evidence was taken at `f6c84afd7`.
- The qwen35 hybrid dispatch has the same shape (GPU build inside the timed
  window). I did not touch it here: it is out of scope, and its throughput gate
  passes today.
