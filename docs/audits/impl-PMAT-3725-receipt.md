# PMAT-3725: receipt

**Ticket:** PMAT-3725 (issue #3725). Qwen3.5 CUDA long-context decode: split-K (flash-decoding) decode attention, f32 and f16 KV read. This is site 4 of the #3596 f16 split. **Kind:** code.
**Branch:** `PMAT-3725-splitk-decode-attention`, from `origin/main` at `52f43da71`, in `/mnt/nvme-raid0/agent-wt/3725-splitk`. The receipts below were measured on commit `e52d34af5` (the kernel commit). Later commits add tests, evidence and this file only.
**Plan of record:**
- design note: #3725 issuecomment-5763965543
- cop rulings: #3725, 2026-09-21T16:35:12Z, quoted in the fragment's `notes`
- scope split with #3596: cop comment, 2026-09-21T16:20:41Z
**Closing keyword:** `Refs #3725`, not `Closes`. The issue's second done_when (e2e decode tok/s vs pinned llama.cpp at 4k…262k) needs #3596's batched prefill. The pinned comparator does not resolve on either host today (#3563's class). The issue stays open for both.

## What landed

- **`aprender-gpu/src/kernels/gdn/decode_attention_splitk.rs`**
  - `DecodeAttentionSplitKKernel` (kernel A):
    - Grid `(num_kv_heads, n_splits)`, block 256. One block serves all `num_heads / num_kv_heads` query heads of a KV head, so K/V are read once per token, not once per query head.
    - Warp `w` takes positions `start+w, +8, …`. Lane `l` holds elements `l, l+32, …, l+224`, so each load is one contiguous 128 B (f32) or 64 B (f16) span.
    - The online softmax runs per (warp, head) in registers. The 8 warps merge in shared memory, and the block writes one partial `(m, l, acc[256])` per (query head, split).
  - `DecodeAttentionSplitKReduceKernel` (kernel B): the log-sum-exp merge.
  - `KvStorage::{F32, F16}`: f16 is loaded as `b16` and widened with `cvt.f32.f16`. All arithmetic is fp32.
  - `SplitKPlan`: `chunk = max(64, ⌈L/256⌉)`, and a split is never empty.
  - `decode_attention_splitk_cpu`: the CPU twin, in the kernel's own split/warp/merge order.
  - `decode_attention_reference_f64`.
- **`aprender-serve`**
  - `KernelType::{GdnDecodeAttentionSplitK, GdnDecodeAttentionSplitKReduce}`, as their own chained name/PTX fns so no match grows past the complexity ceiling.
  - `CudaExecutor::gdn_decode_attention_splitk_into`: raw K/V pointers plus the storage tag. That is #3596's interface for its f16 branch, agreed with aprender-3e [44388e].
  - Refusals, never a silent launch: `seq_len == 0` and scratch smaller than the plan.
  - `gdn_decode_attention_splitk_scratch_splits`. `Qwen35AttnScratch` gains `splitk_acc` / `splitk_ml`.
  - The attention layer step calls the split-K pair with `KvStorage::F32`.
- **Not modified:**
  - `DecodeAttention256Kernel` / `decode_attention.rs` is the parity reference.
  - `gdn_decode_attention_into` is now called only by the executor-level parity test.
  - `Qwen35CudaState` and the KV row writes belong to #3596.
- **`aprender-gpu/examples/gdn_splitk_rungs.rs`:** the rungs receipt. `--emit-ptx DIR` writes the builder PTX for the oxide A/B.
- **`experiments/cuda-oxide/splitk-decode-attention/`:** the same algorithm as cuda-oxide `#[kernel]`s (b9847e95, nightly-2026-08-28), with the twin, an f64 reference, and an in-process A/B against the builder PTX. This is cop ruling (2): production stays builder-PTX until OXIDE-001 O-1.
- **Found and filed, not fixed here: #3746.** `KernelBuilder::shfl_xor_f32` emits `shflbfly`, which ptxas rejects. No kernel had ever loaded it. This kernel uses `shfl.down` + `shfl.idx`.

## Verification

```
cmd=cargo test -p aprender-gpu --features cuda --lib decode_attention_splitk   host=lambda sm_89   exit=0   # 10 passed, 0 SKIPPED
cmd=<same test binary> decode_attention_splitk                                  host=gx10 sm_121    exit=0   # 10 passed, 0 SKIPPED (grep -c SKIPPED = 0), 9.24 s
cmd=cargo test -p aprender-serve --features cuda --lib qwen35_cuda              host=lambda         exit=0   # 16 passed in 139 s on the real 0.8B and 4B files, incl. attention_layers_match_cpu, forward_single_matches_cpu_logits_end_to_end, 4b_forward_single_matches_cpu_argmax_end_to_end; no test skipped
cmd=<realizar lib test binary> gdn_splitk                                       host=lambda         exit=0   # 2 passed: executor path at seq_len 5000 (79 splits) vs the 16-block wrapper AND f64 (1e-4 x max|ref|); seq_len 0 and one-split scratch refused
cmd=MUTATION reduce kernel scale := 1.0 (no split rescale)                      host=lambda         exit=101 # 3 device tests RED
cmd=MUTATION f16 read at the f32 element stride (bytes.max(4))                  host=lambda         exit=101 # splitk_f16_read_matches_widened_f32 RED
cmd=MUTATION warp position stride 9 instead of 8                                host=lambda         exit=101 # 3 device tests RED
cmd=cargo fmt --all -- --check                                                                      exit=0
cmd=cargo clippy -p aprender-gpu -p aprender-serve --features cuda --lib --tests --examples           # zero findings in any file this diff touches; pre-existing errors elsewhere (driver_and_context.rs, falsification_crux_c_34.rs) are untouched
cmd=ptxas -v (sm_89) on the builder PTX                                                              # kernel A 120 regs (16/4) / 155 regs (24/4), 0 spills, 8256 B smem; kernel B 22 regs
```

Every device run held `flock /tmp/apr-gpu.lock choom -n 1000`. The lambda serve run recompiled under the lock (a formatting change landed after the pre-build). The receipt runs below used pre-built binaries.

## Rungs: split-K vs the kernel it replaces (`evidence/section-3725/{lambda,gx10}.json`)

Method:
- Same cache, same q, 16/4 (9B) and 24/4 (27B) geometry, `head_dim` 256.
- µs per call is host wall time over 20 back-to-back launches plus one sync, after 3 warmup launches.
- `rel` is split-K vs the old kernel: max |Δ| / max |old|.
- f16 = split-K reading f16 KV vs the old kernel reading the same values widened to f32.
- Both kernels are also measured against an f64 reference at every rung.
- GPU state at start: lambda 22976/24035 MiB free; gx10 72 GB free of 122 GB.

**9B geometry (16/4), µs per layer per token:**

| L | lambda old | lambda split-K f32 / f16 | × (f32 / f16) | gx10 old | gx10 split-K f32 / f16 | × (f32 / f16) |
|---|---|---|---|---|---|---|
| 4,096 | 1186 | 26.9 / 26.8 | 44 / 44 | 2712 | 264 / 138 | 10.3 / 19.5 |
| 8,192 | 2480 | 50.4 / 49.8 | 49 / 46 | 5429 | 503 / 276 | 10.8 / 19.6 |
| 20,000 | 8699 | 248 / 131 | 35 / 66 | 13349 | 1234 / 698 | 10.8 / 19.0 |
| 60,000 | 26021 | 607 / 352 | 43 / 74 | 39587 | 3231 / 1732 | 12.3 / 22.9 |
| 148,000 | 64146 | 1429 / 740 | 45 / 87 | 97952 | 7978 / 4024 | 12.3 / 24.4 |
| 262,144 | 113675 | 2344 / 1233 | 48 / 92 | 172658 | 13235 / 6977 | 13.1 / 24.8 |

**27B geometry (24/4)** follows the same shape. lambda 262,144: 110225 → 2371 / 1308 µs. gx10: 170246 → 14138 / 6766 µs. All 24 rows per host are in the JSON.

**Parity, all 48 rows (2 hosts × 2 geometries × 2 storages × 6 rungs):**
- split-K vs the old kernel: rel ≤ 1.59e-5, cosine 1.0000000.
- **Against f64, split-K is closer than the kernel it replaces at every rung:** split-K ≤ 1.04e-6 vs old ≤ 1.60e-5.
- The accuracy difference is the old kernel's single running sum over up to 262k positions.

**Two timing instruments disagree on gx10, so read the rungs table as an upper bound on split-K time.**
- The rungs harness times on the host wall: 20 back-to-back launches plus one sync, through aprender-gpu's driver wrapper.
- The oxide A/B uses CUDA events on the same device and the same builder kernels, with the cache sized to 60k rows instead of 262k.
- At 4,096 on gx10 the event timer reads the builder pair at 182 µs (f32) and 47 µs (f16), against the harness's 264 µs and 138 µs. At 60k it reads 2284 µs (f32) and 1235 µs (f16), against 3231 µs and 1732 µs.
- On lambda the two instruments agree within ~2% at 20k–60k (f32 20k: 244.6 vs 248.4 µs; 60k: 606.3 vs 606.7 µs).
- **The cause of the gx10 gap is not measured.** Per-launch host cost on the GB10's CPU and a different L2 footprint are candidates, not findings.
- Relative to the device-event timer, the harness reads split-K HIGHER on gx10, so the gx10 speedups in the table are conservative. The ms-scale old-kernel column uses the same instrument, where a per-launch host cost would be a rounding error.

**Arithmetic (an estimate from the measurements, not an e2e measurement):**
- At 262,144 on lambda, split-K moves the 2 GiB f32 K+V of one layer in 2.34 ms, about 0.92 TB/s. The 1 GiB f16 cache moves in 1.23 ms, about 0.87 TB/s. The 4090's peak is ~1.0 TB/s.
- Over the 9B's 8 attention layers that is ~18.8 ms (f32) or ~9.9 ms (f16) of attention per token, against ~909 ms with the old kernel.
- gx10 reaches about 0.16 TB/s at 262k, of LPDDR5X's ~0.27.
- End-to-end decode tok/s is NOT claimed here. It follows #3596.

## cuda-oxide A/B (`evidence/kernels/gdn_decode_attention_splitk/{lambda,gx10}.json`, schema `apr-kernel-receipt/v1`)

Same buffers, same stream, CUDA events, median of 5 × 20. The builder PTX comes from `gdn_splitk_rungs --emit-ptx` for the same device.

| host | parity (vs twin / vs f64 / vs builder) | oxide ÷ builder, 4k … 60k |
|---|---|---|
| lambda sm_89 | PASS: ≤ 3.7e-7 / ≤ 7.1e-7 / ≤ 2.4e-7, cos 1.0000000 | 3.6 (f32 4k), 2.4 (20k), 2.3 (60k); f16 3.6, 3.5, 3.2. All 10 rows: 2.14–3.61 |
| gx10 sm_121 | PASS: ≤ 3.7e-7 / ≤ 7.1e-7 / ≤ 2.4e-7, cos 1.0000000 | 3.5 (f32 4k), 3.9 (20k), 3.7 (60k); f16 11.4, 5.8, 5.8. All 10 rows: 2.97–11.4 |

**Mechanism, from `ptxas -v` on the oxide PTX (sm_89):**
- The oxide kernel keeps its per-lane `q` and accumulator arrays on a **288-byte local-memory stack** (40 registers). The builder kernel holds them in 120 registers.
- The oxide PTX also carries 10 bounds-check `trap`s.

**Decision recorded for OXIDE-001 O-4:** oxide/builder is well outside the 1.2× KERNEL-001 margin, so the production kernel stays builder-PTX. The stack-resident arrays are the first thing an oxide revision should remove.

## Deviations and open items

- **The e2e decode tok/s table vs pinned llama.cpp is not here.** Cop ruling (1): it follows #3596's batched prefill, the issue stays open, and this PR uses `Refs`.
- **Occupancy headroom:** 120 regs → ~16 warps/SM on sm_89, and 155 regs for the 27B → 8 warps/SM. At 148k–262k the f32 kernel already runs near peak bandwidth on lambda. The gx10 f16 path, at ~55% of LPDDR5X peak, is the place a later tuning pass would pay.
- **Graph replay:** the split count is a runtime scalar. The Qwen3.5 decode path does not use the manual decode graph (neither did the 16-block kernel, whose `seq_len` is also a per-token scalar). The launches are still recorded, like every GDN wrapper.
