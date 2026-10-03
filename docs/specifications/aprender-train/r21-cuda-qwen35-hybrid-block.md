# R21: the Qwen3.5 hybrid block on CUDA (S-R21 desk spike, 0.72, L2)

Status: desk read 2026-10-03 at origin/main `316dee2cd4` and at `la-72/fold-r2r3` @5a837dfa3b (the CPU oracle).
Nothing here was built or run. `[V]` marks facts read in the code, with file:line; `[A]` marks estimates. Paths are
under `crates/`. This sizes spec §3 row 4a, which carried K̂ 360–480 `[A]` from `r15-cuda-lora-cells.md`
§Consequences. The falsifiers are PROPOSED rows: FALSIFY-QTC-001..005 in `contracts/qwen35-train-cuda-v1.yaml`, and
the value-head-order rows FALSIFY-QQE-007/008 in `contracts/qwen35-qlora-e2e-v1.yaml`. Each test is `[U]` until
cargo (and, for the CUDA rows, a GPU) is allowed after LIVE 0.70.1.

## Result

- **R21 = 400 `[A]` (325–465) in f32**, the precision R4's NF4 path computes in. T2's bf16 path adds **25 `[A]`
  (20–30)**: its ten GDN GEMM sites (five projections forward, their five dX backward) on C5's bf16 GEMM. The scan
  state stays f32 on both paths.
- **The row is the whole hybrid block, not GDN alone.** The CUDA trainer also lacks two pieces the 4B's 8
  full-attention layers need: the attention output gate and partial RoPE (§The CUDA trainer). They cost 35–50 of
  the 400.
- **Memory decides the design.** The scan's full state history is 1 GiB per layer per sequence at seq 512. R21
  recomputes each layer during its backward and checkpoints the scan state every 8 steps: 0.5 GiB at batch 4, for
  the one layer in flight (§Memory).
- **Out of R21:** a chunked (WY) scan fast enough to race fla is R14's. The value-head order that differs between
  GGUF and HF is a boundary defect for R4's PEFT export and for any HF-sourced base, not a kernel cost (§Value-head
  order).

## Qwen3.5-4B, the shapes that size this `[V]`

Read from a local Qwen3.5-4B `config.json` (`text_config`). It is an AWQ repack, so R4 re-checks the shapes on the
checkpoint it trains. `contracts/model-families/qwen3_5.yaml` lists only the 9B and 27B.

| | 4B |
|---|---|
| hidden, layers | 2560; 32 = 24 Gated DeltaNet + 8 full attention (`full_attention_interval` 4) |
| GDN | 16 key heads, 32 value heads, d_k = d_v = 128; conv kernel 4 over 2·16·128 + 32·128 = 8192 channels |
| full attention | 16 query heads, 4 KV heads, head_dim 256, `attn_output_gate` true |
| RoPE | theta 1e7, `partial_rotary_factor` 0.25 (64 of 256 dims); interleaved M-RoPE [11, 11, 10], which is plain 1-D RoPE when the three position ids are equal, as they are for text `[A]` |
| FFN, vocab | 9216; 248320, tied |

## What serving lends `[V]`

The CUDA GDN kernels are in `aprender-gpu/src/kernels/gdn/`, as f32 PTX builders (default target sm_70). Serving
composes them for prefill in `aprender-serve/src/gguf/cuda/forward_qwen35_cuda_prefill.rs:802-938`. All of them are
forward only.

| Kernel | Does | Training needs in addition |
|---|---|---|
| `CausalConv1dSiluSeqKernel` (`causal_conv1d_seq.rs:37`) | conv + SiLU over a chunk; stores only the SiLU output (:171) and overwrites the carried window (:178) | the pre-SiLU value (kept or recomputed), a zero window per sequence, a backward for dx |
| `PerHeadL2NormRowsKernel` (`rows.rs:29`) | the q and k L2 norm per head, in place | the norms (kept or recomputed), a backward |
| `GdnGatesRowsKernel` (`rows.rs:148`) | the decay g and β | a backward to the two projection outputs |
| `DeltaRuleChunkScanKernel` (`delta_rule_scan.rs:61`) | grid (H_v, 1, 1), block (d_v) (:46): one thread per state row, held in registers; asserts the tiled head map (:80); sequential in t; writes each o_t (:275) and the final state once (:282) | a batch axis, a zero initial state per sequence, saved states (history or checkpoints), and the reverse scan |
| `GatedRmsNormKernel` (`gated_rmsnorm.rs:26`) | RMSNorm(o)·w·SiLU(z) | a backward to o and z |
| `SplitInterleavedKernel` (`split_interleave.rs:28`), `SigmoidGateKernel` (`sigmoid_gate.rs:15`), `partial_rope.rs` | the full-attention query/gate split, the output gate, partial RoPE | backwards for the gate and the partial RoPE; the split's backward is the inverse scatter |

"Chunk" in `DeltaRuleChunkScanKernel` means one launch per prefill chunk. It is the sequential recurrence, not the
chunkwise-parallel (WY) form. No GDN, delta-rule or conv1d backward kernel exists in the tree.

## The CUDA trainer, for the 8 full-attention layers `[V]`

- RoPE is `BatchedRopeNeoxKernel::new(num_heads, head_dim, batch_size, theta)`
  (`aprender-gpu/src/kernels/elementwise/rope/neox.rs:155`), built in
  `aprender-train/src/autograd/cuda_forward/cache.rs:352` and applied in
  `aprender-train/src/transformer/cuda_block.rs:1001-1022`. It rotates the whole head and takes no rotary width. On
  Qwen3.5 it would rotate 256 dims where the model rotates 64: a wrong answer, not a slow one.
- Neither `cuda_block.rs` nor `finetune/instruct_pipeline` has an attention output gate (no output_gate, attn_gate,
  sigmoid_gate or gated_attn). q_norm and k_norm exist.
- The attention core materialises the scores: batched GEMM, scale, causal mask, softmax, batched GEMM (NF4 path,
  `cuda_block.rs:3636`, `:3743`, `:3801`). Nothing there caps head_dim, so 256 should work, but nothing has run it
  `[A]`.

## The CPU oracle (`la-72/fold-r2r3` @5a837dfa3b) `[V]`

Paths are under `aprender-train/src/transformer/`.

- `gdn_mixer_forward` (`gdn.rs:380`). The recurrence (:291-299) decays the state row by e^g, predicts p = row·k, sets
  δ = β(v − p), adds δ·k to the row and reads o = row·q/√d_k. As matrices, with S as [d_v × d_k]:
  S̃ = e^{g_t}·S_{t−1}, S_t = S̃ + β_t(v_t − S̃k_t)k_tᵀ, o_t = S_t q_t/√d_k.
- The head map is tiled, `kh = h % nk` (:338), as in serving's GGUF path. The loader reads GGUF only
  (`qwen35_model.rs:3-8`), and `ssm_a` is −exp(A_log) (`gdn.rs:73`).
- `GdnScan { out, final_state, history }` (:87-96) keeps a [seq_len × state] history.
- `gated_delta_scan_backward` (`gdn_backward.rs:113`). `gdn_mixer_backward` (:314) re-runs the scan from a zero state
  with history on (:352) and passes no final-state gradient (:369). R21 copies that: zero state in, nothing carried
  out.
- The GDN LoRA targets are `GdnQkv`, `GdnGate` and `GdnOut`: attn_qkv, attn_gate, ssm_out (`qwen35_lora.rs:31-36`).
  ssm_alpha, ssm_beta, ssm_a, dt_bias, conv1d and ssm_norm stay frozen.
- Full attention uses `partial_neox_rope_seq` (`qwen35_layer.rs:87`). A step's loss and gradients come from
  `loss_and_grads` (`qwen35_lm.rs:113`).

## Memory and traffic at T2's shape (batch 4, seq 512) `[A]`

- One GDN state is 32·128·128·4 B = 2 MiB per layer per sequence.
- Full history is 512 states: 1 GiB per layer per sequence, 4 GiB at batch 4, and 96 GiB for all 24 GDN layers. So
  R21 keeps only each layer's input (all 33 boundaries at 4·512·2560·4 B ≈ 0.69 GB) and recomputes a layer's
  forward during its backward.
- Even one layer at a time, full history moves 8–12 GiB per layer per step: written during the recompute, read back
  by the reverse scan, and more if the first forward keeps it too. Over 24 layers that is 0.2–0.3 s per step at the
  4090's 1,008 GB/s. T2's GEMMs are about 52 TFLOP per step, 0.4–0.5 s, so full history would add 40–75%.
- Checkpointing every c steps costs batch·(T/c + c)·2 MiB per layer. At c = 8 that is 64 checkpoints per sequence
  (512 MiB at batch 4) plus a replay scratch of 128 blocks × 8 states × 64 KiB = 64 MiB, which fits in the 4090's
  72 MB L2. The reverse sweep replays each 8-step chunk from its checkpoint into the scratch, then walks it backwards.
- The replay calls the same step code as the forward, so checkpointed gradients should be bit-identical to the
  full-history ones, not merely close (QTC-003). A separately fused replay would lose that.

## Design `[A]`

1. **Precision and GEMMs.** f32 first. The GDN projections use the trainer's existing GEMMs. LoRA on attn_qkv,
   attn_gate and ssm_out goes through C1's `lora_backward` helper (R15a), so C1 lands before this cell.
2. **Training forward.** Grid (H_v, batch), block d_v, one thread per state row, as in serving. A zero initial
   state and a zero conv window for every sequence. It can save every state (history) or every c-th (checkpoints).
3. **Reverse scan.** Same grid. Each thread keeps its row of dS = ∂L/∂S_t in registers across steps; the S rows
   stream in from the history or the replay scratch. For t = T…1, with p = S̃k_t, δ = β_t(v_t − p),
   S_t = S̃ + δk_tᵀ and s = 1/√d_k:
   - dS += s·dO_t q_tᵀ, and dq_t = s·S_tᵀ dO_t
   - dδ = dS·k_t, dv_t = β_t·dδ, dβ_t = Σ_j dδ_j (v_{t,j} − p_j)
   - dk_t = dSᵀδ − β_t S̃ᵀ dδ
   - dS̃ = dS − β_t dδ k_tᵀ, dg_t = ⟨dS̃, S̃⟩, then the carry dS ← e^{g_t} dS̃

   Row terms stay in registers. dq and dk are column sums over the block's d_v rows, done in shared memory; dβ and
   dg are block sums.
4. **No float atomics.** dq and dk are written per value head. A second kernel adds the two value heads that share
   each key head (h and h + 16 under the tiled map) in a fixed order, so two runs give the same bits.
5. **The pointwise backwards.** Gated RMSNorm, the gates, the L2 norm, SiLU, and the depthwise causal conv. The conv
   needs only dx, since conv1d is frozen.
6. **Build order.** Training forward with history; the backward against full history; the checkpointed variant,
   bit-identical to it; gated full attention; the hybrid step.

The main risk is register pressure. The dS row is 128 floats per thread. If the reverse scan spills, use two threads
per row (block 2·d_v), with a pair shuffle for the row dot products.

## Components `[A]`

| Part | K̂ |
|---|---|
| Training forward: batch grid, zero state per sequence, history and checkpoint modes, saved pre-activations | 50–70 |
| GDN projections, with LoRA on attn_qkv, attn_gate, ssm_out through C1's `lora_backward` | 20–30 |
| Reverse scan | 60–90 |
| Checkpointed variant (c = 8), bit-identical to full history | 20–30 |
| Backwards for gated RMSNorm, gates, L2 norm, SiLU and conv | 50–65 |
| Gated full attention: query/gate split, sigmoid output gate, partial NeoX RoPE, forward and backward | 35–50 |
| Hybrid schedule (interval 4) and the GGUF loader into the CUDA block | 40–60 |
| Parity tests and planted mutants (QTC-001..005) | 50–70 |
| **Total (f32)** | **325–465, carried as 400** |
| bf16 on T2's path: the ten GDN GEMM sites on C5's bf16 GEMM | 20–30, carried as 25 |

## Falsifiers in `qwen35-train-cuda-v1` (PROPOSED, `[U]`)

Each runs on a tiny fixture against the CPU oracle above: 2 key heads, 4 value heads, d_k = d_v = 8, conv kernel 4,
T ≥ 9, batch 2, random weights and a fixed seed.

| ID | Claim | Planted mutation that must turn it RED |
|---|---|---|
| QTC-001 | the CUDA GDN mixer's training forward equals `gdn_mixer_forward`, max abs diff ≤ 1e-5·max(scale, 1), from a zero state per sequence | state carried from one sequence to the next; the grouped head map in place of the tiled one |
| QTC-002 | the CUDA mixer backward equals `gdn_mixer_backward`: dX and the LoRA gradients of attn_qkv, attn_gate and ssm_out, relative L2 ≤ 1e-4 in f32 | drop the −β_t S̃ᵀdδ term of dk; reduce dq and dk over the grouped pairs |
| QTC-003 | the checkpointed reverse scan (c ∈ {1, 8, T}, T = 13) is bit-identical to full history | the chunk replay restarts one step late |
| QTC-004 | gated full attention on CUDA (query/gate split, sigmoid output gate, q/k norm, partial NeoX RoPE over head_dim/4) equals the CPU layer, forward and backward | full-width RoPE, which is today's kernel; the gate dropped |
| QTC-005 | two CUDA steps on a 4-layer hybrid model (interval 4) change nothing but LoRA A and B; ssm_a, dt_bias, conv1d, ssm_norm, ssm_alpha, ssm_beta and every base weight stay bit-identical; the step-1 loss equals `loss_and_grads` within 1e-4 relative | conv1d made trainable |

## Value-head order `[V]`

At 4B the 32 value heads share 16 key heads, and the two file formats order them differently.

- **GGUF is tiled:** value head h reads key head h % 16. The GGUF conversion permutes every value-head-indexed tensor
  out of HF's order (`aprender-serve/src/gguf/inference/forward/forward_qwen35.rs:160-187`, which cites llama.cpp's
  `_LinearAttentionVReorderBase`). The oracle, serving's GGUF path and the CUDA scan all use this order.
- **HF is grouped:** value head h reads key head h / 2, because HF expands q and k with `repeat_interleave`. Serving's
  safetensors path keeps that order (`aprender-serve/src/gpu/scheduler/linear_attn.rs:218-233`,
  `vh = kh * heads_ratio + r`) and reads A_log raw.
- **Nothing converts between them.** `apr import` from HF safetensors keeps the grouped order and A_log as they are
  (`aprender-core/src/format/tensor_expectation.rs:12`, `:267`). PEFT export writes lora_A and lora_B byte for byte
  (`aprender-train/src/lora/adapter/peft_export.rs:100-110`).

Two consequences sit outside the kernels:

- **Export.** QQE says the adapter is "standard PEFT layout". An HF-ordered adapter needs the value rows of attn_qkv's
  B, all rows of attn_gate's B and all columns of ssm_out's A permuted from tiled back to grouped. Without that, HF,
  vLLM and serving's own safetensors path apply 30 of the 32 value heads' deltas to the wrong head (heads 0 and 31 are
  the only fixed points). QQE's adapter_serve_identity check cannot see the fault, because the trainer and the GGUF
  server share the tiled order. Proposed as FALSIFY-QQE-007.
- **Load.** The oracle reads GGUF. A Qwen3.5 base imported from HF safetensors carries grouped heads, A_log, and
  possibly HF's zero-centred RMSNorm weights (applied as 1 + w; `[A]`, unverified). QQE's base is "a .apr imported by
  a released apr" without saying from what. Until a loader normalises the conventions, the trainer should refuse a
  GDN layer that carries A_log and no ssm_a, rather than train on it. Proposed as FALSIFY-QQE-008, a cheap refusal in
  the same family as DBH-001 and MOF-002.

## Open items

1. **Register pressure** in the reverse scan; the fallback is above.
2. **Speed is R14's.** R21 is a sequential scan for correctness, and it is latency-bound: 128 blocks of 128 threads
   for the 4090's 128 SMs, 512 dependent steps per layer `[A]`. R5 measures its share of T2's step rather than
   assuming it.
3. **The 4B shapes** come from an AWQ repack's config. R4 confirms them on the checkpoint it trains.
4. **An HF-sourced loader** (permute the value heads, A_log to ssm_a, squeeze conv1d to [C, K], and fold the norm
   offset if HF has one) is sized nowhere. T2 needs it only if apr's side loads the HF checkpoint rather than the
   GGUF.
5. **Contract text on `fold-r2r3`.** `qwen35-train-gdn-v1` (:26-27) still states the old recurrence: α = sigmoid(·)
   and o = (Sᵀq) ⊙ z. The code decays by e^g (`gdn.rs:291`) and applies z only in the gated norm. That contract's
   note at :96 already records the right decay, so this is a text fix in that PR, not a defect in the code.
