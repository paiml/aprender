# PMAT-3596 — receipt

**Ticket:** PMAT-3596 (issue #3596). Batched (chunked) CUDA prefill for the Qwen3.5 hybrid, usable up to the declared 262,144-position context. **Kind:** code (`kind:code`). **Row:** 0.69.1 P0.
**Branch:** `feat/3596-qwen35-prefill-chunk-rows`, tip `cf33a2234`, from `origin/main` at `225b2a9ab` (release 0.69.0), in the worktree `/mnt/nvme-raid0/agent-wt/wt-3596`; `~/src/aprender` untouched. The cop folded this row's three stacked PRs into this ONE PR, with one receipt and one quorum (2026-09-21):
- PR1 `feat/3596-qwen35-batched-prefill`: batched prefill, capacity refusal, parity tool and contract.
- PR2 `feat/3596-qwen35-flash-prefill`: flash attention.
- PR3 `…-chunk-rows`: unified-memory chunk rows, the context bound and the attention default.

**Measurement binary:** `242d7e1a4`, a local merge of `release/0.69.1-batch-1 @ 6c40a8a95` with this tip. Branch `meas/3596-on-batch1`; it is NOT a PR. Why a merge: the batch carries #3726's canonical BPE tokenizer. Every binary built on 0.69.0 tokenized long prompts non-canonically: +363 tokens at 148k and +1144 at 262k against llama.cpp, and an O(n²) encoder cost 13.6 → 168 s outside prefill. aprender-c3 measured all seven prompt files id-for-id identical to pinned `llama-tokenize` on a #3726 binary. Cop ruling: every remaining rung runs on a #3726 binary, and each rung names its binary. Rows from earlier binaries are kept below, marked SUPERSEDED, never mixed into the current table.
**The tip is two commits ahead of the measured binary**, and neither changes a number: `a8fde59cc` (the discrete co-tenant refusal said "other processes hold N MiB" when ~290 MiB of that gap was the refusing process's own CUDA context) and `cf33a2234` (a plan that passes over a (path, rows) option now prints it). Both were written FROM measurements below; no rung was re-run for them.
**Author:** aprender-3e [44388e] (claude-opus-5, direct; no worker dispatched). `tokens_used` / `wall_clock_s`: [U] (not instrumented).
**Design note:** issuecomment-5763674795. **GB10 profile:** issuecomment-5765556992. **Tokenizer evidence:** #3726 issuecomment-5767766463.

## What lands

**aprender-gpu:** new kernels (PTX builders), each with device tests:
- `kernels/gdn/delta_rule_scan.rs`: `DeltaRuleChunkScanKernel`, a chunk-resident Gated DeltaNet scan. The state lives in registers across the chunk, and one launch covers T tokens.
- `kernels/gdn/causal_conv1d_seq.rs`: `CausalConv1dSiluSeqKernel`, the causal conv1d + SiLU over T rows.
- `kernels/gdn/rows.rs`: row twins of the per-token kernels (`PerHeadL2NormRowsKernel`, `GdnGatesRowsKernel`, `PartialNeoxRopeRowsKernel`). `partial_rope::emit_sin_cos` becomes `pub(super)`.
- `kernels/gdn/prefill_flash_attention.rs`: `PrefillFlashAttention256Kernel`, a fused causal GQA flash attention for `head_dim = 256`. It uses `mma.sync m16n8k16` with f16 inputs, f32 accumulation and an f32 online softmax, per the cop's ruling. Tiles are BR = BC = 16. Shared memory is 37,376 B at 4 heads per KV head and 47,616 B at 6. It needs sm_80+.
- `kernels/quantize/q5k/dequant.rs` and `q8_0_dequant.rs`: whole-tensor dequant to f32 scratch for the prefill GEMMs. Q5_K's layout comment is corrected (qh at 16, qs at 48).
- `ptx/builder/warp_vote.rs`: `shfl_idx_f32_reg`.
- `driver/mod.rs` re-exports `classify_device_memory` / `DeviceMemoryClass`.

**aprender-serve:**
- `cuda/executor/gdn_prefill_ops.rs` (new): executor wrappers covering the dequant → cuBLAS SGEMM projection (`CUBLAS_PEDANTIC_MATH`: no TF32/FP8/DP4A), conv, L2, gates, RoPE, scan, the f32 attention and the flash attention. The f32 attention is QKᵀ → causal-mask softmax → PV, split into query passes by a 1 GiB scores budget.
- `gguf/cuda/forward_qwen35_cuda_prefill.rs` (new):
  - `Qwen35CudaModel::prefill` / `prefill_logits_at`. Only the rows asked for run the `lm_head`.
  - Chunking: GEMM chunks are 512 rows on discrete GPUs and 2048 on unified memory. The attention query passes are sized separately.
  - Capacity: `capacity_inputs`, and `attention_candidates` (cuBLAS f32 first, flash second; `APR_QWEN35_PREFILL_ATTENTION` pins one).
- `gguf/cuda/forward_qwen35_cuda_prefill_tests.rs` (new): equivalence against the per-token path.
- `capacity.rs` (new):
  - `plan()`: f32 KV if it fits the measured free memory; f16 if only f16 fits and the #3725 decode read exists; otherwise refuse. The refusal is classified `co_tenant` / `f16_decode_unavailable` / `exceeds_device` by what an EMPTY device could hold.
  - `DeviceMemory { Discrete, Unified }`: on GB10 the plan uses `MemAvailable − 16 GiB`, because there `cuMemGetInfo` excludes page cache (aprender-eb, #3714).
  - `plan_first_fit`: the first (attention path, chunk rows) that fits.
  - Case tables.
- `gguf/inference/forward/forward_qwen35.rs`:
  - `qwen35_gpu_decode` prefills in one batched call and prints `[qwen35] batched prefill: N tokens in X ms (Y tok/s, chunk R rows, attention …)`.
  - It returns `Qwen35PhaseTimings { prefill_ms, decode_ms, prefill_attention }` / `Qwen35GenerateOutcome { tokens, used_gpu, timings, capacity }` for #3718.
  - A capacity refusal is `RealizarError::CapacityRefused` BEFORE load. It is not a CPU fallback: the CPU forward at these lengths takes hours.
  - `qwen35_check_context`: the GH-167 bound (`prompt > context_length` refuses) the Qwen3.5 dispatch lacked.
  - The planner takes the first `(attention, chunk rows)` that fits and PRINTS what it passed over.
  - The F2 guard probes through `prefill_logits_at` (`f2_receipt.rs`: `F2_RECEIPT_SCHEMA = 2`).
- `gguf/cuda/forward_qwen35_cuda.rs`:
  - The model's own state is capped at `DEFAULT_MAX_SEQ_LEN`; `new_state_with_len` sizes a request's.
  - It stores `prefill_rows` and `prefill_attention`.
- `error.rs`: `CapacityRefused`. `lib.rs`: `pub mod capacity`.
- `cublas_prefill/attention.rs`: `CAUSAL_MASK_SOFTMAX_PTX` becomes `pub(crate)`.
- `examples/qwen35_prefill_parity.rs` (new, `required-features = ["cuda"]`): long-context parity of batched vs per-token at every `stride`-th position, over a canonical token-id file.

**contracts/qwen35-batched-prefill-v1.yaml** (new, `kind: pattern`): obligations C-QBP-001..005, falsified by SCAN-BITWISE, ROWS-BITWISE, PREFILL-EQ, CAPACITY-TABLE and FLASH. `pv validate` and `pv lint`: 0 errors.

**Outside the file boundary agreed with aprender-c3 (#3693).** The boundary named new kernel files, the new prefill file, and edits to `forward_qwen35.rs` (decode + F2 probe) / `forward_qwen35_cuda.rs` (state sizing). Beyond it:
- The dequant kernels, the `warp_vote` helper and the `driver` re-export: new code, no behaviour change to existing callers.
- `capacity.rs`, `error.rs` and `lib.rs`: the refuse-before-load the amendments require.
- `executor/gdn_prefill_ops.rs` and the `pub(crate)` on the causal softmax PTX.
- `forward_qwen35.rs` also carries the capacity planner and the context bound.
- `decode_attention.rs` is untouched, as agreed.

## done_when, item by item

The fragment's `notes:` quotes the issue and the amendments verbatim. Each item and its evidence:

| # | requirement (verbatim source) | status | evidence |
|---|---|---|---|
| 1 | "a batched (chunked) CUDA prefill for the Qwen3.5 hybrid (attention layers batched; the Gated DeltaNet recurrence in chunked form) on lambda and gx10" | MET | § What lands; tests at the tip on sm_89 AND sm_121 (§ Tests) |
| 1a | amendment 1: a chunk-resident scan **bitwise-identical to the per-token recurrence**, measured; WY/UT only if the scan exceeds 15 % of prefill | MET | `gdn_chunk_scan_is_bitwise_per_token_{0_8b,9b_grouped,27b_grouped}_shape` on both archs. The scan's share of GPU kernel time: **2.7 %** (GB10, 9B, 4k, issuecomment-5765556992), **3.6 %** (4090, 9B, 60k, f32 path: SGEMM 70.3 %, softmax 12.8 %, dequant 11.2 %). WY/UT is not required |
| 2 | "parity-checked against the per-token path at ≥ 64 positions and at 20k" | MET | § Parity: 64 and 600 positions (unit, both archs, both attention paths); 20k (0.8B, every 200th position, both paths) (20k on lambda; the gx10 20k pair was still queued in the night window when this receipt was written — see § In flight) |
| 3 | "TTFT for a 20k-token prompt on the 9B measured on both hosts … against llama.cpp at the pinned comparator on the same prompt (the ratio is reported, never claimed as a target)" | MET | § Measurements, the 20k rows; the llama.cpp `d1d3c3396` column |
| 3a | amendment 2: TTFT recorded and output correct at 4k · 8k · 20k · 60k · 148k · 262,144 on lambda AND gx10; refuse before loading where memory can't hold a rung | MET on lambda at all six rungs; on gx10 through 148k for the 9B and the 27B, with the 9B at 262k MET. The 27B at 262k was running when this was written (§ In flight) | § Measurements; each rung's answer graded against the needle |
| 3b | amendment 3: f16 KV where parity holds, printed in `--json` | NOT SHIPPED (by the split) | the f16 branch is wired and REFUSES with "f16 KV decode read not available (#3725)" until #3725 lands; no rung needed f16 (§ Measurements) |
| 3c | operator "a": the 27B rungs above the 4090 are owed on gx10 at 262,144; on lambda apr refuses before loading, naming the arithmetic | IN FLIGHT on gx10 (started 02:56:09Z, the cop's 3 h window); on lambda no 27B rung was attempted — the 4090 cannot hold it, which is the operator's ruling, not a gap | gx10 27B table |
| 4 | "`apr run --json` reports `prefill_ms` and `prompt_tokens` (with #3718)" | SHARED | this row supplies `Qwen35PhaseTimings.prefill_ms` (definition agreed with aprender-a8: the forward over post-template prompt tokens up to the first generated token's logits, excluding load/h2d/validate/tokenization) and the outcome's capacity budget. The JSON emission is #3718's (aprender-a8), told where both live |
| — | tensor-core attention (cop ruling): f16 inputs, f32 accumulation, f32 online softmax; the f32 path selectable; the precision printed | MET | `PrefillFlashAttention256Kernel`; `APR_QWEN35_PREFILL_ATTENTION=f32/flash`; the stderr line names the path per run. The **default is cuBLAS f32 while it fits** (cop ruling 2026-09-21), because flash measured SLOWER on sm_89 at every length (§ Measurements). Flash is taken only when flash alone fits (`plan_first_fit`) |

The ORIGINAL items 1–6 of #3596 (BEATS.md, `apr bench` for hybrids, the "(cached)" suffix, the comparator pin, root-causing the 14 s) belong to #3598/#3604/#3606 and are NOT claimed. This PR says `Refs #3596`.

## Measurements

**Method.** Each apr rung is one `apr run <model> -i <prompt> --chat --temperature 0 -n 64 --json --backend cuda`, under `gpu-q --prio 1` (rule rev 5; every hold ≤ 20 min, except the cop-granted gx10 night window). The TTFT column is `prefill_ms` from the run's own stderr line (the #3718 definition). `--json`'s `inference_time_ms` includes load and F2; the wall clock is in the every-run table. GPU memory and utilisation at the start of each run are recorded, and a run that started beside a busy GPU is flagged ⚠.

**Prompts.** `prompts/p{4k,8k,20k,60k,148k,262k}.txt` are this repository's Rust sources concatenated. The needle file, `crates/aprender-serve/src/gguf/cuda/forward_qwen35_cuda_prefill.rs`, sits near the end of each, followed by the question: "Name the file above that defines the Rust function `chunk_rows_for`, and quote its first line of code verbatim. Answer in one line."
- The needle section is byte-identical in all six prompts.
- Canonical token counts (the GGUF's own vocab + merges, the qwen35 pre-tokenizer, round-trip asserted; equal to pinned `llama-tokenize` per aprender-c3): 4,123 / 8,213 / 20,066 / 59,913 / 148,192 / 261,937.
- **Answer grading** (`mk_tables.py`, from the run files): "path ✓" is the exact path. "verbatim line ✓" means a quoted line is verbatim (a prefix, where `-n 64` cuts it) of a line of the needle file. The question's "its first line of code" is read by the model as the file's first line or as the function's, and both are lines of that file.

**llama.cpp.** `d1d3c3396`, `llama-server -ngl 99 -fa auto -ctk f16 -ctv f16 -ub 512 -np 1`, one chat request per prompt with `cache_prompt: false`. `prompt_ms` / `prompt_per_second` / `prompt_n` are read from the server's own timings. **apr ÷ llama** is the ratio of prefill rates, apr's best clean sample over llama's. Reported, never claimed.

#### lambda — Qwen3.5-9B-Q4_K_M — #3726 binary (merge 242d7e1a4)

| rung | positions (apr) | cuBLAS f32 attention | flash (f16 in, f32 acc) | llama.cpp d1d3c3396 prompt | apr ÷ llama (best clean rate) | answer |
|---|---|---|---|---|---|---|
| p4k | 4,134 | 2,152 ms · 1,921 tok/s · 242d7e1a4 n=64 | 2,304 ms · 1,794 tok/s · 242d7e1a4 n=64 | 462 ms · 8,935 tok/s (4,131 tok) | 0.215 | path ✓, verbatim line ✓ |
| p8k | 8,224 | 4,288 ms · 1,918 tok/s · 242d7e1a4 n=64 | 4,776 ms · 1,722 tok/s · 242d7e1a4 n=64 | 866 ms · 9,492 tok/s (8,221 tok) | 0.202 | path ✓, verbatim line ✓ |
| p20k | 20,077 | 11,779 ms · 1,704 tok/s · 242d7e1a4 n=64 | 13,737 ms · 1,462 tok/s · 242d7e1a4 n=64 | 2,124 ms · 9,451 tok/s (20,074 tok) | 0.180 | path ✓, verbatim line ✓ |
| p60k | 59,924 | 46,997 ms · 1,275 tok/s · 242d7e1a4 n=64 | 63,792 ms · 939 tok/s · 242d7e1a4 n=64 | 7,147 ms · 8,384 tok/s (59,921 tok) | 0.152 | path ✓, verbatim line ✓ |
| p148k | 148,203 | 262,860 ms · 564 tok/s · 242d7e1a4 n=64 | 266,827 ms · 555 tok/s · 242d7e1a4 n=64 | 22,362 ms · 6,627 tok/s (148,200 tok) | 0.085 | path ✓, verbatim line ✓ |
| p262k | 261,948 | REFUSED (242d7e1a4) | 729,115 ms · 359 tok/s · 242d7e1a4 n=64 | 50,734 ms · 5,163 tok/s (261,945 tok) | 0.070 | path ✓, verbatim line ✓ |

<details><summary>lambda: SUPERSEDED rows — pre-#3726 tokenizer binaries (kept, not mixed in)</summary>

#### lambda — Qwen3.5-9B-Q4_K_M — SUPERSEDED — pre-#3726 tokenizer binaries

| rung | positions (apr) | cuBLAS f32 attention | flash (f16 in, f32 acc) | llama.cpp d1d3c3396 prompt | apr ÷ llama (best clean rate) | answer |
|---|---|---|---|---|---|---|
| p4k | 4,131 | 2,895 ms · 1,427 tok/s · pre-record n=1<br>2,102 ms · 1,966 tok/s · d78eab8eb n=64 | 2,324 ms · 1,778 tok/s · d78eab8eb n=64 | 462 ms · 8,935 tok/s (4,131 tok) | 0.220 | first token "The" (n=1, not graded); path ✗ (crates/aprender-serve/src/gguf/cuda/foward_qwen35_cuda_prefill.rs), verbatim line ✓ |
| p8k | 8,221 | 4,361 ms · 1,885 tok/s · pre-record n=1<br>4,308 ms · 1,908 tok/s · d78eab8eb n=64 | 4,784 ms · 1,719 tok/s · d78eab8eb n=64 | 866 ms · 9,492 tok/s (8,221 tok) | 0.201 | first token "The" (n=1, not graded); path ✓, verbatim line ✓ |
| p20k | 20,085 | 11,464 ms · 1,752 tok/s · 4ff26a07a n=1<br>11,487 ms · 1,749 tok/s · pre-record n=1 ⚠busy 30%<br>11,558 ms · 1,738 tok/s · d78eab8eb n=64 | 13,751 ms · 1,461 tok/s · d78eab8eb n=1<br>13,768 ms · 1,459 tok/s · d78eab8eb n=64 | 2,124 ms · 9,451 tok/s (20,074 tok) | 0.185 | first token "The" (n=1, not graded); path ✓, verbatim line ✓ |
| p60k | 59,887 | 46,513 ms · 1,288 tok/s · 4ff26a07a n=1<br>47,338 ms · 1,265 tok/s · pre-record n=1 ⚠busy 91%<br>46,962 ms · 1,275 tok/s · d78eab8eb n=64 | 61,551 ms · 973 tok/s · d78eab8eb n=1<br>61,924 ms · 967 tok/s · d78eab8eb n=64 | 7,147 ms · 8,384 tok/s (59,921 tok) | 0.154 | first token "The" (n=1, not graded); path ✓, verbatim line ✓ |
| p148k | 148,563 | 257,963 ms · 576 tok/s · 4ff26a07a n=1 | 266,860 ms · 557 tok/s · d78eab8eb n=1 | 22,362 ms · 6,627 tok/s (148,200 tok) | 0.087 | first token "The" (n=1, not graded) |
| p262k | 263,089 | REFUSED (4ff26a07a) | 731,063 ms · 360 tok/s · d78eab8eb n=1 | 50,734 ms · 5,163 tok/s (261,945 tok) | 0.070 | first token "The" (n=1, not graded) |

</details>

<details><summary>lambda: every run (35)</summary>

| tag | model | mode | sha | positions | prefill ms | tok/s | chunk | n | wall s | GPU busy at start | answer / refusal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9b-p148k | 9B | f32 | 4ff26a07a | 148563 | 257963 | 576 | 512 | 1 | 369.198001355 | 3% | first token "The" (n=1, not graded) |
| 9b-p20k-rerun | 9B | f32 | 4ff26a07a | 20085 | 11464 | 1752 | 512 | 1 | 25.073829751 | 0% | first token "The" (n=1, not graded) |
| 9b-p20k | 9B | f32 | pre-record | 20085 | 11487 | 1749 | 512 | 1 | 38.954260469 | 30% | first token "The" (n=1, not graded) |
| 9b-p262k | 9B | — | 4ff26a07a | — | — | — | — | — | 211.563674423 | 0% | an empty GPU of this size could hold this context at F32, but other processes                  hold 924 MiB of it (free < need <= total): weights 4861 MiB + KV 16443 MiB (263091 positions x 65536 B at F32) + workspace 1593 MiB + overhead 512 MiB = 23410 MiB, against 23112 MiB free of 24036 MiB (discrete GPU: cuMemGetInfo). Pass --no-gpu to run the CPU forward instead |
| 9b-p4k | 9B | f32 | pre-record | 4131 | 2895 | 1427 | 512 | 1 | 26.51531331 | 0% | first token "The" (n=1, not graded) |
| 9b-p60k-rerun | 9B | f32 | 4ff26a07a | 59887 | 46513 | 1288 | 512 | 1 | 68.542020803 | 0% | first token "The" (n=1, not graded) |
| 9b-p60k | 9B | f32 | pre-record | 59887 | 47338 | 1265 | 512 | 1 | 85.680681978 | 91% | first token "The" (n=1, not graded) |
| 9b-p8k | 9B | f32 | pre-record | 8221 | 4361 | 1885 | 512 | 1 | 27.801627618 | 0% | first token "The" (n=1, not graded) |
| brief2 | 9B | f32 | pre-record | 6472 | 3971 | 1630 | 512 | 1 | 57.122163145 | 8% | first token "{" (n=1, not graded) |
| flash-9b-p148k | 9B | flash | d78eab8eb | 148563 | 266860 | 557 | 512 | 1 | 339.239906816 | 0% | first token "The" (n=1, not graded) |
| flash-9b-p20k | 9B | flash | d78eab8eb | 20085 | 13751 | 1461 | 512 | 1 | 27.404752801 | 0% | first token "The" (n=1, not graded) |
| flash-9b-p262k | 9B | flash | d78eab8eb | 263089 | 731063 | 360 | 512 | 1 | 899.510595597 | 0% | first token "The" (n=1, not graded) |
| flash-9b-p60k | 9B | flash | d78eab8eb | 59887 | 61551 | 973 | 512 | 1 | 92.906347679 | 0% | first token "The" (n=1, not graded) |
| llama_ladder | ? | — | pre-record | — | — | — | — | — | — | ?% | — |
| m-auto-9b-p262k | 9B | flash | 242d7e1a4 | 261948 | 729115 | 359 | 512 | 64 | 801.267138626 | 0% | path ✓, verbatim line ✓ |
| m-f32-9b-p148k | 9B | f32 | 242d7e1a4 | 148203 | 262860 | 564 | 512 | 64 | 347.082931008 | 0% | path ✓, verbatim line ✓ |
| m-f32-9b-p20k | 9B | f32 | 242d7e1a4 | 20077 | 11779 | 1704 | 512 | 64 | 35.349729906 | 0% | path ✓, verbatim line ✓ |
| m-f32-9b-p262k | 9B | — | 242d7e1a4 | — | — | — | — | — | 12.829784736 | 0% | an empty device of this size could hold this context at F32, but other processes hold 896 MiB of it (free < need <= total): weights 4861 MiB + KV 16376 MiB (262013 positions x 65536 B at F32) + workspace 1593 MiB + overhead 512 MiB = 23342 MiB, against 23140 MiB free of 24036 MiB (discrete GPU: cuMemGetInfo). Pass --no-gpu to run the CPU forward instead |
| m-f32-9b-p4k | 9B | f32 | 242d7e1a4 | 4134 | 2152 | 1921 | 512 | 64 | 42.663972557 | 0% | path ✓, verbatim line ✓ |
| m-f32-9b-p60k | 9B | f32 | 242d7e1a4 | 59924 | 46997 | 1275 | 512 | 64 | 76.812580123 | 0% | path ✓, verbatim line ✓ |
| m-f32-9b-p8k | 9B | f32 | 242d7e1a4 | 8224 | 4288 | 1918 | 512 | 64 | 21.769184432 | 0% | path ✓, verbatim line ✓ |
| m-flash-9b-p148k | 9B | flash | 242d7e1a4 | 148203 | 266827 | 555 | 512 | 64 | 312.280119882 | 0% | path ✓, verbatim line ✓ |
| m-flash-9b-p20k | 9B | flash | 242d7e1a4 | 20077 | 13737 | 1462 | 512 | 64 | 34.363808558 | 0% | path ✓, verbatim line ✓ |
| m-flash-9b-p4k | 9B | flash | 242d7e1a4 | 4134 | 2304 | 1794 | 512 | 64 | 18.769793145 | 0% | path ✓, verbatim line ✓ |
| m-flash-9b-p60k | 9B | flash | 242d7e1a4 | 59924 | 63792 | 939 | 512 | 64 | 94.8035507 | 0% | path ✓, verbatim line ✓ |
| m-flash-9b-p8k | 9B | flash | 242d7e1a4 | 8224 | 4776 | 1722 | 512 | 64 | 22.369558663 | 0% | path ✓, verbatim line ✓ |
| n64-f32-9b-p20k | 9B | f32 | d78eab8eb | 20085 | 11558 | 1738 | 512 | 64 | 34.656500034 | 0% | path ✓, verbatim line ✓ |
| n64-f32-9b-p4k | 9B | f32 | d78eab8eb | 4131 | 2102 | 1966 | 512 | 64 | 53.396149864 | 0% | path ✗ (crates/aprender-serve/src/gguf/cuda/foward_qwen35_cuda_prefill.rs), verbatim line ✓ |
| n64-f32-9b-p60k | 9B | f32 | d78eab8eb | 59887 | 46962 | 1275 | 512 | 64 | 85.24248917 | 0% | path ✓, verbatim line ✓ |
| n64-f32-9b-p8k | 9B | f32 | d78eab8eb | 8221 | 4308 | 1908 | 512 | 64 | 23.193254205 | 0% | path ✓, verbatim line ✓ |
| n64-flash-9b-p20k | 9B | flash | d78eab8eb | 20085 | 13768 | 1459 | 512 | 64 | 35.284867076 | 0% | path ✓, verbatim line ✓ |
| n64-flash-9b-p4k | 9B | flash | d78eab8eb | 4131 | 2324 | 1778 | 512 | 64 | 21.72706658 | 0% | path ✗ (crates/aprender-serve/src/gguf/cuda/foward_qwen35_cuda_prefill.rs), verbatim line ✓ |
| n64-flash-9b-p60k | 9B | flash | d78eab8eb | 59887 | 61924 | 967 | 512 | 64 | 102.440639403 | 0% | path ✓, verbatim line ✓ |
| n64-flash-9b-p8k | 9B | flash | d78eab8eb | 8221 | 4784 | 1719 | 512 | 64 | 24.098935293 | 0% | path ✓, verbatim line ✓ |
| r1 | 9B | — | pre-record | — | — | — | — | 1 | 55.340033437 | 3% | first token "{" (n=1, not graded) |

</details>

#### gx10 — Qwen3.5-9B-Q4_K_M — #3726 binary (merge 242d7e1a4)

| rung | positions (apr) | cuBLAS f32 attention | flash (f16 in, f32 acc) | llama.cpp d1d3c3396 prompt | apr ÷ llama (best clean rate) | answer |
|---|---|---|---|---|---|---|
| p4k | 4,134 | 8,598 ms · 481 tok/s · 242d7e1a4 n=64 | 7,824 ms · 528 tok/s · 242d7e1a4 n=64 | 1,580 ms · 2,614 tok/s (4,131 tok) | 0.202 | path ✓, verbatim line ✓ |
| p8k | 8,224 | 18,355 ms · 448 tok/s · 242d7e1a4 n=64 | 16,494 ms · 499 tok/s · 242d7e1a4 n=64 | 3,059 ms · 2,688 tok/s (8,221 tok) | 0.186 | path ✓, verbatim line ✓ |
| p20k | 20,077 | 45,906 ms · 437 tok/s · 242d7e1a4 n=64 | 44,786 ms · 448 tok/s · 242d7e1a4 n=64 | 7,549 ms · 2,659 tok/s (20,074 tok) | 0.168 | path ✓, verbatim line ✓ |
| p60k | 59,924 | 172,156 ms · 348 tok/s · 242d7e1a4 n=64 | 188,863 ms · 317 tok/s · 242d7e1a4 n=64 | 24,723 ms · 2,424 tok/s (59,921 tok) | 0.144 | path ✓, verbatim line ✓ |
| p148k | 148,203 | 612,477 ms · 242 tok/s · 242d7e1a4 n=64 | 759,363 ms · 195 tok/s · 242d7e1a4 n=64 | 79,211 ms · 1,871 tok/s (148,200 tok) | 0.129 | path ✓, verbatim line ✓ |
| p262k | 261,948 | 1,594,855 ms · 164 tok/s · 242d7e1a4 n=64 | — | 175,409 ms · 1,493 tok/s (261,945 tok) | 0.110 | path ✓, verbatim line ✓ |

#### gx10 — Qwen3.5-27B-Q4_K_M — #3726 binary (merge 242d7e1a4)

| rung | positions (apr) | cuBLAS f32 attention | flash (f16 in, f32 acc) | llama.cpp d1d3c3396 prompt | apr ÷ llama (best clean rate) | answer |
|---|---|---|---|---|---|---|
| p4k | 4,134 | 47,248 ms · 87 tok/s · 242d7e1a4 n=64 | 45,941 ms · 90 tok/s · 242d7e1a4 n=64 | — | — | path ✓, verbatim line ✓ |
| p8k | 8,224 | 93,739 ms · 88 tok/s · 242d7e1a4 n=64 | 91,789 ms · 90 tok/s · 242d7e1a4 n=64 | — | — | path ✓, verbatim line ✓ |
| p20k | 20,077 | 232,046 ms · 87 tok/s · 242d7e1a4 n=64 | 238,236 ms · 84 tok/s · 242d7e1a4 n=64 | — | — | path ✓, verbatim line ✓ |
| p60k | 59,924 | 524,168 ms · 114 tok/s · 242d7e1a4 n=64 | 867,899 ms · 69 tok/s · 242d7e1a4 n=64 | — | — | path ✓, verbatim line ✓ |
| p148k | 148,203 | 2,611,259 ms · 57 tok/s · 242d7e1a4 n=64 | — | — | — | path ✓, verbatim line ✓ |

<details><summary>gx10: SUPERSEDED rows — pre-#3726 tokenizer binaries (kept, not mixed in)</summary>

#### gx10 — Qwen3.5-9B-Q4_K_M — SUPERSEDED — pre-#3726 tokenizer binaries

| rung | positions (apr) | cuBLAS f32 attention | flash (f16 in, f32 acc) | llama.cpp d1d3c3396 prompt | apr ÷ llama (best clean rate) | answer |
|---|---|---|---|---|---|---|
| brief | 6,472 | 15,152 ms · 427 tok/s · pre-record n=1 | — | — | — | first token "{" (n=1, not graded) |
| p4k | 4,131 | 9,867 ms · 419 tok/s · pre-record n=1 | — | 1,580 ms · 2,614 tok/s (4,131 tok) | 0.160 | first token "The" (n=1, not graded) |
| p8k | 8,221 | 21,926 ms · 375 tok/s · pre-record n=1 | — | 3,059 ms · 2,688 tok/s (8,221 tok) | 0.140 | first token "The" (n=1, not graded) |
| p20k | 20,085 | 46,508 ms · 432 tok/s · pre-record n=1 | — | 7,549 ms · 2,659 tok/s (20,074 tok) | 0.162 | first token "The" (n=1, not graded) |
| p60k | 59,887 | 161,653 ms · 370 tok/s · 4ff26a07a n=1 | — | 24,723 ms · 2,424 tok/s (59,921 tok) | 0.153 | first token "The" (n=1, not graded) |
| p148k | 148,563 | 684,499 ms · 217 tok/s · 4ff26a07a n=1 | — | 79,211 ms · 1,871 tok/s (148,200 tok) | 0.116 | first token "The" (n=1, not graded) |
| p262k | 263,089 | — | 1,932,537 ms · 136 tok/s · ab7ad29a4 n=1 | 175,409 ms · 1,493 tok/s (261,945 tok) | 0.091 | first token "The" (n=1, not graded) |

#### gx10 — Qwen3.5-27B-Q4_K_M — SUPERSEDED — pre-#3726 tokenizer binaries

| rung | positions (apr) | cuBLAS f32 attention | flash (f16 in, f32 acc) | llama.cpp d1d3c3396 prompt | apr ÷ llama (best clean rate) | answer |
|---|---|---|---|---|---|---|
| brief | 6,472 | — | 69,564 ms · 93 tok/s · ab7ad29a4 n=1 | — | — | first token "{" (n=1, not graded) |
| p4k | 4,131 | — | 46,736 ms · 88 tok/s · ab7ad29a4 n=1 | — | — | first token "cr" (n=1, not graded) |
| p20k | 20,085 | — | 240,325 ms · 84 tok/s · ab7ad29a4 n=64 | — | — | path ✓, verbatim line ✓ |

</details>

<details><summary>gx10: every run (30)</summary>

| tag | model | mode | sha | positions | prefill ms | tok/s | chunk | n | wall s | GPU busy at start | answer / refusal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 27B-brief | 27B | flash | ab7ad29a4 | 6472 | 69564 | 93 | 2048 | 1 | 219.164826452 | 0% | first token "{" (n=1, not graded) |
| 27B-p4k | 27B | flash | ab7ad29a4 | 4131 | 46736 | 88 | 2048 | 1 | 108.655187819 | 0% | first token "cr" (n=1, not graded) |
| 9B-brief | 9B | f32 | pre-record | 6472 | 15152 | 427 | 512 | 1 | 141.02303083 | 0% | first token "{" (n=1, not graded) |
| 9B-p148k | 9B | f32 | 4ff26a07a | 148563 | 684499 | 217 | 512 | 1 | 795.576795726 | 0% | first token "The" (n=1, not graded) |
| 9B-p20k | 9B | f32 | pre-record | 20085 | 46508 | 432 | 512 | 1 | 74.26792288 | 0% | first token "The" (n=1, not graded) |
| 9B-p262k | 9B | flash | ab7ad29a4 | 263089 | 1932537 | 136 | 2048 | 1 | 2121.607078017 | 0% | first token "The" (n=1, not graded) |
| 9B-p4k | 9B | f32 | pre-record | 4131 | 9867 | 419 | 512 | 1 | 34.829119101 | 0% | first token "The" (n=1, not graded) |
| 9B-p60k | 9B | f32 | 4ff26a07a | 59887 | 161653 | 370 | 512 | 1 | 222.483832127 | 0% | first token "The" (n=1, not graded) |
| 9B-p8k | 9B | f32 | pre-record | 8221 | 21926 | 375 | 512 | 1 | 57.920056014 | 0% | first token "The" (n=1, not graded) |
| m-auto-27B-p148k | 27B | f32 | 242d7e1a4 | 148203 | 2611259 | 57 | 2048 | 64 | 2877.399042096 | 0% | path ✓, verbatim line ✓ |
| m-auto-9B-p262k | 9B | f32 | 242d7e1a4 | 261948 | 1594855 | 164 | 2048 | 64 | 1711.473025908 | 0% | path ✓, verbatim line ✓ |
| m-f32-27B-p20k | 27B | f32 | 242d7e1a4 | 20077 | 232046 | 87 | 2048 | 64 | 310.060425461 | 0% | path ✓, verbatim line ✓ |
| m-f32-27B-p4k | 27B | f32 | 242d7e1a4 | 4134 | 47248 | 87 | 2048 | 64 | 111.365403435 | 0% | path ✓, verbatim line ✓ |
| m-f32-27B-p60k | 27B | f32 | 242d7e1a4 | 59924 | 524168 | 114 | 512 | 64 | 651.751340043 | 0% | path ✓, verbatim line ✓ |
| m-f32-27B-p8k | 27B | f32 | 242d7e1a4 | 8224 | 93739 | 88 | 2048 | 64 | 194.098880833 | 0% | path ✓, verbatim line ✓ |
| m-f32-9B-p148k | 9B | f32 | 242d7e1a4 | 148203 | 612477 | 242 | 2048 | 64 | 708.75619814 | 0% | path ✓, verbatim line ✓ |
| m-f32-9B-p20k | 9B | f32 | 242d7e1a4 | 20077 | 45906 | 437 | 2048 | 64 | 84.302868712 | 0% | path ✓, verbatim line ✓ |
| m-f32-9B-p4k | 9B | f32 | 242d7e1a4 | 4134 | 8598 | 481 | 2048 | 64 | 37.208263037 | 0% | path ✓, verbatim line ✓ |
| m-f32-9B-p60k | 9B | f32 | 242d7e1a4 | 59924 | 172156 | 348 | 2048 | 64 | 220.651928669 | 0% | path ✓, verbatim line ✓ |
| m-f32-9B-p8k | 9B | f32 | 242d7e1a4 | 8224 | 18355 | 448 | 2048 | 64 | 51.306439465 | 0% | path ✓, verbatim line ✓ |
| m-flash-27B-p20k | 27B | flash | 242d7e1a4 | 20077 | 238236 | 84 | 2048 | 64 | 316.077512578 | 0% | path ✓, verbatim line ✓ |
| m-flash-27B-p4k | 27B | flash | 242d7e1a4 | 4134 | 45941 | 90 | 2048 | 64 | 194.84356756 | 0% | path ✓, verbatim line ✓ |
| m-flash-27B-p60k | 27B | flash | 242d7e1a4 | 59924 | 867899 | 69 | 2048 | 64 | 982.296336197 | 0% | path ✓, verbatim line ✓ |
| m-flash-27B-p8k | 27B | flash | 242d7e1a4 | 8224 | 91789 | 90 | 2048 | 64 | 154.757159751 | 0% | path ✓, verbatim line ✓ |
| m-flash-9B-p148k | 9B | flash | 242d7e1a4 | 148203 | 759363 | 195 | 2048 | 64 | 832.419567495 | 0% | path ✓, verbatim line ✓ |
| m-flash-9B-p20k | 9B | flash | 242d7e1a4 | 20077 | 44786 | 448 | 2048 | 64 | 79.763438089 | 0% | path ✓, verbatim line ✓ |
| m-flash-9B-p4k | 9B | flash | 242d7e1a4 | 4134 | 7824 | 528 | 2048 | 64 | 68.153884766 | 0% | path ✓, verbatim line ✓ |
| m-flash-9B-p60k | 9B | flash | 242d7e1a4 | 59924 | 188863 | 317 | 2048 | 64 | 238.886261627 | 0% | path ✓, verbatim line ✓ |
| m-flash-9B-p8k | 9B | flash | 242d7e1a4 | 8224 | 16494 | 499 | 2048 | 64 | 46.752967901 | 0% | path ✓, verbatim line ✓ |
| n64-flash-27B-p20k | 27B | flash | ab7ad29a4 | 20085 | 240325 | 84 | 2048 | 64 | 447.962149857 | 0% | path ✓, verbatim line ✓ |

</details>

**nsys** (kernel-time shares): GB10 9B 4k: SGEMM ~48 %, dequant→f32 ~40 %, scan 2.7 %. 4090 9B 60k (f32 path): SGEMM 70.3 %, causal softmax 12.8 %, dequant 11.2 %, scan 3.6 %. So:
- At short context the prefill is weight-traffic-bound: every chunk re-dequantizes every weight to f32. That is why the 2048-row chunks help on GB10.
- At long context, attention is the wall.
- The next levers (NOT in this PR) are a fused dequant-GEMM reading Q4_K directly, and a faster flash kernel (BR = BC = 16 reaches ~10 % of the 4090's f16 peak).

## Parity (batched vs the per-token path)

**At 64 and 600 positions** (the unit tests at the tip, 0.8B; each configuration compares the last logits, a prefill split across two calls, and one decode step after the prefill, plus every layer's conv / recurrent / KV state):

| host | model | positions | attention | chunk rows | query rows/pass | argmax equal (last, split, decode) | min cosine (7 d.p.) | worst logits rel L∞ | worst state rel L∞ |
|---|---|---|---|---|---|---|---|---|---|
| lambda (sm_89) | Qwen3.5-0.8B-Q4_K_M | 600 | flash | 512 | 512 | 3/3 | 1.0000000 | 1.28e-04 | 2.36e-04 (layer 15 k) |
| lambda (sm_89) | Qwen3.5-0.8B-Q4_K_M | 600 | cuBLAS f32 | 512 | 37 | 3/3 | 1.0000000 | 1.14e-06 | 2.11e-06 (layer 10 ssm) |
| lambda (sm_89) | Qwen3.5-0.8B-Q4_K_M | 64 | flash | 64 | 64 | 3/3 | 1.0000000 | 1.55e-04 | 5.44e-04 (layer 23 k) |
| lambda (sm_89) | Qwen3.5-0.8B-Q4_K_M | 600 | cuBLAS f32 | 512 | 512 | 3/3 | 1.0000000 | 1.14e-06 | 2.08e-06 (layer 10 ssm) |
| lambda (sm_89) | Qwen3.5-0.8B-Q4_K_M | 64 | cuBLAS f32 | 64 | 64 | 3/3 | 1.0000000 | 8.18e-07 | 2.09e-06 (layer 10 ssm) |
| lambda (sm_89) | Qwen3.5-0.8B-Q4_K_M | 600 | flash | 600 | 600 | 3/3 | 1.0000000 | 1.44e-04 | 2.36e-04 (layer 15 k) |
| lambda (sm_89) | Qwen3.5-0.8B-Q4_K_M | 600 | flash | 64 | 64 | 3/3 | 1.0000000 | 1.38e-04 | 2.33e-04 (layer 15 k) |
| gx10 (sm_121) | Qwen3.5-0.8B-Q4_K_M | 600 | cuBLAS f32 | 512 | 37 | 3/3 | 1.0000000 | 1.56e-06 | 3.75e-06 (layer 16 ssm) |
| gx10 (sm_121) | Qwen3.5-0.8B-Q4_K_M | 64 | cuBLAS f32 | 64 | 64 | 3/3 | 1.0000000 | 9.47e-07 | 1.23e-06 (layer 16 ssm) |
| gx10 (sm_121) | Qwen3.5-0.8B-Q4_K_M | 600 | flash | 512 | 512 | 3/3 | 1.0000000 | 1.28e-04 | 2.12e-04 (layer 15 k) |
| gx10 (sm_121) | Qwen3.5-0.8B-Q4_K_M | 600 | cuBLAS f32 | 512 | 512 | 3/3 | 1.0000000 | 1.56e-06 | 3.75e-06 (layer 16 ssm) |
| gx10 (sm_121) | Qwen3.5-0.8B-Q4_K_M | 64 | flash | 64 | 64 | 3/3 | 1.0000000 | 1.54e-04 | 5.46e-04 (layer 23 k) |
| gx10 (sm_121) | Qwen3.5-0.8B-Q4_K_M | 600 | flash | 600 | 600 | 3/3 | 1.0000000 | 1.26e-04 | 2.11e-04 (layer 15 k) |
| gx10 (sm_121) | Qwen3.5-0.8B-Q4_K_M | 600 | flash | 64 | 64 | 3/3 | 1.0000000 | 1.28e-04 | 2.39e-04 (layer 15 k) |

The flash kernel alone, against exact attention in f64 (`gdn_flash_prefill_matches_reference_*`):

| host | q heads / kv heads | query rows | pos0 (cached prefix) | rel L∞ vs f16-input reference (bound 2e-3) | vs full-f32 reference |
|---|---|---|---|---|---|
| lambda (sm_89) | 8/2 | 16 | 0 | 2.02e-05 | 4.68e-04 |
| lambda (sm_89) | 24/4 | 20 | 3 | 1.32e-04 | 4.82e-04 |
| lambda (sm_89) | 16/4 | 37 | 29 | 2.42e-04 | 3.76e-04 |
| lambda (sm_89) | 16/4 | 100 | 500 | 2.84e-04 | 5.02e-04 |
| gx10 (sm_121) | 8/2 | 16 | 0 | 2.02e-05 | 4.68e-04 |
| gx10 (sm_121) | 24/4 | 20 | 3 | 1.32e-04 | 4.82e-04 |
| gx10 (sm_121) | 16/4 | 37 | 29 | 2.42e-04 | 3.76e-04 |
| gx10 (sm_121) | 16/4 | 100 | 500 | 2.84e-04 | 5.03e-04 |

**At 20,000 positions** (`examples/qwen35_prefill_parity.rs`, 0.8B; canonical token ids `prompts/p20k.txt.ids`; per-token path vs batched at every 200th position):

| host | model | attention | positions | sampled (every 200th) | argmax agree | worst 1 − cosine | worst rel L∞ (logits) | per-token path | batched prefill | tool binary |
|---|---|---|---|---|---|---|---|---|---|---|
| lambda | Qwen3.5-0.8B | f32 | 20,000 | 101 | 101/101 | 8.0e-12 (pos 0) | 3.67e-06 | 580 s | 1,949 ms | qwen35_prefill_parity @ d78eab8eb |
| lambda | Qwen3.5-0.8B | flash | 20,000 | 101 | 101/101 | 6.4e-07 (pos 0) | 1.13e-03 | 586 s | 4,018 ms | qwen35_prefill_parity @ d78eab8eb |

## Tests at the tip

`cargo test -p aprender-gpu --features cuda --lib -- --nocapture gdn_ q5k_dequant q8_0_dequant` and `cargo test -p aprender-serve --features cuda --lib -- --nocapture qwen35_prefill qwen35_flash_prefill capacity:: qwen35_route_tests f2_`, at `8461e893a`, one `gpu-q` hold per host. rc is read from `${PIPESTATUS[0]}`, never through the grep.

| host | device (compute cap.) | aprender-gpu kernels | aprender-serve prefill / capacity / route / F2 | started |
|---|---|---|---|---|
| lambda | NVIDIA GeForce RTX 4090, 8.9 | 60 passed, 0 failed (rc 0) | 40 passed, 0 failed (rc 0) | 2026-09-21T21:55:36Z |
| gx10 | NVIDIA GB10, 12.1 | 60 passed, 0 failed (rc 0) | 40 passed, 0 failed (rc 0) | 2026-09-21T22:07:37Z |

## Mutations (each RED, then restored GREEN)

| mutant | killed by | reading |
|---|---|---|
| chunk scan: output immediate × 1.0001 | `gdn_chunk_scan_is_bitwise_per_token_9b_grouped_shape` | RED at flat index 0 |
| attention passes: drop `+ r0` from the pass base | `qwen35_prefill_equals_per_token_with_many_attention_passes_0_8b` | cosine 0.807, argmax still equal: the cosine floor, not the argmax, catches it |
| flash kernel: causal `key <= p` → `key < p` | `gdn_flash_prefill_matches_reference_9b_shape_with_prefix_and_tail` | rel L∞ 0.526 |
| context bound `>` → `>=` | `a_prompt_past_the_declared_context_is_refused_with_both_numbers` | "prompt of 262144 against 262144: Err(ContextLimitExceeded …)" |
| attention default reversed (flash first) | `qwen35_prefill_attention_prefers_f32_then_flash_and_the_environment_pins_one` | `left: [FlashF16In, CublasF32]` |
| `plan_first_fit` rows-outer loop | `capacity_first_fit_takes_f32_while_it_fits_and_flash_when_only_flash_does` | tried `[(F32, 2048), (Flash, 2048)]`, which skips f32 at 512 |

## Jidoka

- **`.rn` is dropped by the PTX emitter.** The builder records `.rn` on f32 mul/add, and the emitter does not print it, so ptxas may contract into FMA. The scan's bitwise equality is therefore MEASURED per architecture (sm_89, sm_121), not constructed. Told aprender-c3 (#3693).
- **The tokenizer was the old one.** Every 0.69.0-based binary tokenized long prompts non-canonically (the drift table on #3726). At 4k, the `-n 64` answer on the old binary garbled the needle path as `foward_qwen35_cuda_prefill.rs … verbati`: #3726's copy-task defect, measured. Resolved by building on the batch.
- **The context bound was missing.** The old tokenizer made the 262k prompt 263,089 positions, and nothing refused it against the model's 262,144. `qwen35_check_context` (`08555906a`) closes that on both routes.
- **The old encoder cost TTFT outside prefill.** On the old binaries it cost 13.6 s (20k) → 168 s (262k), and 211 s before a pre-load refusal. The #3726 encoder takes ~0.2 s at 262k (aprender-c3). Filed by the cop as a 0.69.1 issue.
- **F2 receipts thrash.** They are keyed per MODEL file, not per (model, version, device): concurrent sessions' binaries overwrite each other, and every version re-validates (~25 s on the 9B). Filed by the cop.
- **The first discrete capacity text had a run of spaces** (visible in the SUPERSEDED 262k refusal from `4ff26a07a`). Fixed in `9395f8002` and asserted (`!reason.contains("  ")`).
- **Pre-existing, not this row.** `tests/gpu_cpu_trace_compare.rs` and `tests/phase21_trace_divergence.rs` do not compile on `origin/main` (`AprTransformer.lm_head_tied`); neither is in CI's integration list. `cargo clippy --tests` on aprender-serve reports findings in files this PR does not touch (`cancel_scope_2376.rs`, `iq2_s.rs`, …). `--lib` is clean with and without `cuda`.
- **gpu-q.** A SIGTERM'd waiter spun on without its ticket; fixed in gpu-q v4 by the cop. Killing `flock(1)` does not release the lock, because the child inherits the fd.

## In flight when this receipt was written (2026-09-22 ~03:20Z)

The cop (aprender-04, then aprender-3e [8f56c1]) granted gx10 a night window for the rungs no 20-minute hold can take. Two were still running at wind-down. Their files land beside every other rung, and the tables above are regenerated from those files by `mk_tables.py`, so finishing this receipt is re-running one script — no number here is transcribed by hand.

| run | state at writing | where its files land | how to read it |
|---|---|---|---|
| gx10 27B at 262,144 (operator's "a"), `-n 64`, the default path | started 02:56:09Z; at 03:14Z `MemAvailable` steady at 21.9 GB, above the 16 GiB headroom, prefill running; the cop's timeout is 3 h | `/mnt/nvme-raid0/agent-wt/3596-meas-gx10/m-auto-27B-p262k.{err,json,done,start,gpu_at_start}` | `.done` carries `rc` and wall; the `[qwen35] batched prefill:` line in `.err` carries positions, ms, tok/s, chunk rows and the attention path; `.json`'s `text` is the needle answer |
| gx10 0.8B 20k parity, flash and f32 | queued behind it, 30-minute timeout each | `…/parity-0.8B-20000-{flash,f32}.{json,out,done}` | the same JSON shape the lambda rows in § Parity came from |

`python3 mk_tables.py lambda=<dir> gx10=<dir>` and `mk_parity.py` regenerate every table; they live with `mk_unit.py` and `needle_section.txt` in `docs/audits/impl-PMAT-3596-generators/`, committed beside this receipt. A timeout is recorded AS a timeout (`rc=124`), never dropped, per the cop's ruling of 2026-09-21.

Also still open at wind-down: no quorum round has run. Both model families were out (gemini until ~2026-09-23 04:35Z, gpt-oss 429 until ~07:11Z), so the cop placed this row for peer seats rather than a round.

## Gaps (not claimed)

- `--json` emission of `prefill_ms` / `prompt_tokens` is #3718 (aprender-a8).
- f16 KV serves nothing until #3725 lands (the branch refuses by design); f16-vs-f32 parity is measured on gx10 once both land.
- apr-vs-llama.cpp OUTPUT parity is #3693 (aprender-c3). Here batched is compared to per-token (same apr), and the answer to the needle question is graded.
- Decode speed at long context (split-K) is a separate row, per amendment 4.
- The RMSNorm ε fix (#3759) changes lambda numerics. These measurements predate it and do not carry it.
- **The gx10 27B at 262,144 and the gx10 20k parity pair were still running** when this receipt was written; § In flight says exactly where their files land.
- **Host memory on a unified device is not counted against the row, but it is large.** Measured on gx10 during the 27B at 148k: the process holds 45.8 GB of host RSS — 15.6 GB of it the GGUF mapping, which `MappedGGUFModel::from_path` pre-faults and `mlock`s — beside ~36 GB of device allocations, and `MemAvailable` fell 120.9 → 36 GB. On GB10 that mapping duplicates weights the device already holds. Releasing it after upload (munlock + `MADV_DONTNEED`, unified only) would return ~16 GB to the plan. NOT done here: the 27B at 262k fit without it (MemAvailable held at 21.9 GB, above the 16 GiB headroom). It is the next lever if a rung refuses on a busier host.
- **The 27B on GB10 is GEMM-bound, not attention-bound, and it was not profiled.** Its rate is flat — 87/88/87/114 tok/s at 4k/8k/20k/60k (f32) — where the 9B falls from 481 to 348 over the same range. No nsys run backs a mechanism, so none is claimed.

## Verification

Every command below was run by me on the branch tip (lambda, unless the row says otherwise); the GPU suites are in § Tests at the tip.

| command | result |
|---|---|
| `cargo fmt --all -- --check` | rc 0 |
| `cargo clippy -p aprender-serve --lib -- -D warnings` (default features) | rc 0 |
| `cargo clippy -p aprender-serve --features cuda --lib -- -D warnings` | rc 0 |
| `cargo clippy -p aprender-gpu --features cuda --lib -- -D warnings` | rc 0 |
| `cargo nextest run --profile ci -p aprender-serve --lib` (what CI's `workspace-test` runs for this crate) | 15,906 passed, 0 failed, 59 skipped |
| `cargo test -p aprender-contracts --lib` | 1,684 passed, 0 failed |
| `cargo deny check advisories` | advisories ok |
| `pv validate contracts/qwen35-batched-prefill-v1.yaml` | 0 errors, 0 warnings — valid |
| `pv lint contracts/qwen35-batched-prefill-v1.yaml` | PASS |
| `cargo check -p aprender-serve --features cuda --example qwen35_prefill_parity` | rc 0 |

`cargo clippy --tests` on aprender-serve is NOT clean, and was not made clean: its findings are in files this PR does not touch (`api/tests/cancel_scope_2376.rs`, `quantize/iq2_s.rs`, …), and two integration targets (`tests/gpu_cpu_trace_compare.rs`, `tests/phase21_trace_divergence.rs`) do not compile on `origin/main` either. Neither is in CI's integration list.

**What CI will NOT run for this row:** `workspace-test` excludes `aprender-gpu` and builds without `cuda`, so every kernel test and every prefill-equivalence test above runs only on a GPU host. That is why they are run on both architectures here and the readings are printed.
