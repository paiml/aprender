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
  GGUF and HF is a boundary cost, not a kernel cost (§Value-head order): a load-time conversion on both R4's path and
  T2's (spec §3 row 4b, 25 `[A]`), and a PEFT-export permutation on R4's alone (10–15 `[A]`).

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

Each runs on a tiny fixture against the CPU oracle above: 3 key heads of width d_k = 8, 6 value heads of width
d_v = 12, model width 16, conv kernel 4, T ≥ 9, batch 2, random weights and a fixed seed. Not 2 and 4 heads: there
the key-head count equals the ratio, so a kernel that uses one for the other still passes (§Value-head order). Not
equal widths either: every local Qwen3.5 checkpoint has d_k = d_v = 128, so a kernel that uses one width for both
passes on all of them, and on a fixture with equal widths. Nor equal totals: at d_k = 8 and d_v = 4 the key and
value blocks are both 24 wide, as they are on the 0.8B and the 2B (2048 and 2048), so a kernel that sizes or places
the value block with the key total passes there. The 4B's are 2048 and 4096. The CPU oracle's own tests at
`80723cf206` (`gdn_tests.rs`, and the QTG-003 gradcheck in `gdn_backward_tests.rs`) all use equal widths too, so
they get a case at these widths first.

| ID | Claim | Planted mutation that must turn it RED |
|---|---|---|
| QTC-001 | the CUDA GDN mixer's training forward equals `gdn_mixer_forward`, max abs diff ≤ 1e-5·max(scale, 1), from a zero state per sequence | state carried from one sequence to the next; the grouped head map in place of the tiled one; h mod 2 (the ratio) in place of h mod 3; d_v as a state row's length; the value block read from n_k·d_k + n_v·d_v instead of 2·n_k·d_k |
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
- **Nothing on main converts between them.** `apr import` from HF safetensors keeps the grouped order and A_log as
  they are (`aprender-core/src/format/tensor_expectation.rs:12`, `:267`). PEFT export writes lora_A and lora_B byte
  for byte (`aprender-train/src/lora/adapter/peft_export.rs:100-110`).
- **#4418's branch has the conversion** (`m0694/4418-qwen35-gguf-main` @1af0e3cc11, 0.71's R9, not on main at
  `316dee2cd4`). `transform_qwen35_tensor` (`aprender-core/src/format/converter/qwen35_gguf.rs:212-259`) is
  llama.cpp's HF → GGUF value transform, checked element-wise against llama.cpp `d1d3c3396` on a seeded tiny
  checkpoint (:9-18). It passes every attention and MLP projection through unchanged (:256) and changes four things:
  - +1 on the RMSNorm weights that HF stores zero-centred: input, post-attention, q_norm, k_norm and the final norm,
    never `linear_attn.norm` (:203-208);
  - A_log → −exp(A_log), which is ssm_a;
  - conv1d squeezed from [C, 1, K] to [C, K];
  - every value-head-indexed axis regrouped from key-head-major to value-head-major by `reorder_v_heads` (:149-168):
    A_log, dt_bias, the rows of in_proj_a, in_proj_b and in_proj_z, the value rows of in_proj_qkv and conv1d, and
    the columns of out_proj (:172-185).
- **Which checkpoints and fixtures can see the order.** The GGUF metadata of the local checkpoints
  (`qwen35.ssm.group_count` and `qwen35.ssm.time_step_rank`, read 2026-10-03) gives 16 key heads and 16 value heads
  for the 0.8B and the 2B, 16 and 32 for the 4B and the 9B, and 16 and 48 for the 27B. With equal counts the
  permutation is the identity, so the 4B is the smallest Qwen3.5 on which the order matters, and no 0.8B or 2B cell
  can catch a fault in it. A fixture with 2 key heads and 4 value heads is blind in another way: the key-head count
  equals the ratio, so the permutation is its own inverse and h mod nk = h mod ratio. Code that applies the
  permutation backwards, or uses one count for the other, passes there. That is the fixture of #4418's
  `reorder_v_heads` test (`qwen35_gguf_tests.rs:141-168`). The fixtures in this document, QQE-007/008 and QFR-006
  therefore use 3 key heads and 6 value heads: the ratio is the 4B's, and the fixed points are 0 and 5, as the 4B's
  are 0 and 31. For [0, 1, 2, 3, 4, 5], `reorder_v_heads` returns [0, 2, 4, 1, 3, 5] and the backwards version
  [0, 3, 1, 4, 2, 5] (a Python port that passes #4418's own asserts).

Two consequences sit outside the kernels:

- **Export.** QQE says the adapter is "standard PEFT layout". An HF-ordered adapter needs the value rows of attn_qkv's
  B, all rows of attn_gate's B and all columns of ssm_out's A permuted from tiled back to grouped. Without that, HF,
  vLLM and serving's own safetensors path apply 30 of the 32 value heads' deltas to the wrong head (heads 0 and 31 are
  the only fixed points). QQE's adapter_serve_identity check cannot see the fault, because the trainer and the GGUF
  server share the tiled order. Proposed as FALSIFY-QQE-007. The permutation back is `reorder_v_heads` itself with
  nk and nv/nk exchanged (checked in Python on four shapes, the 4B's among them), so QQE-007 costs 10–15 `[A]` with
  its test. It is R4's alone: T2's canonical cell targets no GDN projection
  (`beat-unsloth-finetune-throughput-v1` :29-32), and its seven targets have one layout in both conventions (:256).
- **Load.** The oracle reads GGUF conventions, and neither R4's base nor T2's can be a GGUF today. `apr finetune -m
  lora|qlora` trains from .apr only (`apr-cli/src/commands/finetune.rs:354-364`), and `apr import` refuses every
  real Qwen3.5 GGUF (spec §2 S-R10, QFR-005). So the only Qwen3.5 base either row can train is a .apr imported from
  HF safetensors, in HF conventions: grouped heads, A_log, zero-centred norms and a [C, 1, K] conv. A qwen35 GGUF
  import path would change that, and no row holds one (S-R10). The trainer therefore needs a load-time conversion:
  `qwen35_gguf_name` for each name and `transform_qwen35_tensor` for each value, in memory, so that no file format
  and no serve path changes. The norm weights stay f32 after the +1, as #4418's GGUF writer keeps them (:316-318).
  bf16's spacing at 1.0 is 2⁻⁷, so a bf16 copy would round every norm of T2's bf16 base in a way the served model
  does not. Until the conversion lands, the trainer must refuse a GDN layer that carries A_log and no ssm_a rather
  than train on it. FALSIFY-QQE-008 covers both outcomes: the conversion, or the refusal, which is a cheap one in the
  same family as DBH-001 and MOF-002.

## Open items

1. **Register pressure** in the reverse scan; the fallback is above.
2. **Speed is R14's.** R21 is a sequential scan for correctness, and it is latency-bound: 128 blocks of 128 threads
   for the 4090's 128 SMs, 512 dependent steps per layer `[A]`. R5 measures its share of T2's step rather than
   assuming it.
3. **The 4B shapes are confirmed.** Every row of the table above matches `~/models/Qwen3.5-4B-Q4_K_M.gguf`, read
   2026-10-03 with python `gguf`: hidden 2560, 32 blocks, interval 4; 16 key and 32 value heads, d_k 128, inner size
   4096, conv 4 × 8192; 16 query and 4 KV heads of 256, 64 rotary dims, theta 1e7; FFN 9216, vocab 248320, and no
   output.weight, so tied. R4 re-checks them on the HF snapshot it imports.
4. **The HF-convention loader is sized** as spec §3 row 4b: 25 `[A]` (20–30) with QQE-008, calling #4418's
   `qwen35_gguf_name` and `transform_qwen35_tensor`. Both are `pub(crate)` in aprender-core, which aprender-train
   already depends on (`aprender-train/Cargo.toml:85`), so the change is a visibility change plus the loader. Add 20
   `[A]` if #4418 has not landed by then and the transforms must first move to a shared module. It is on R4's path
   and T2's (§Value-head order, Load).
   The transforms also need the head counts and widths, and the tensors cannot give them: the rows of `in_proj_qkv`
   give only 2·n_k·d_k + n_v·d_v. #4418's import stores them in the .apr's custom key `linear_attn_hparams`
   (`write.rs:151`). Its GGUF export takes them from there or from a config.json beside the input, and otherwise
   refuses (`gguf_export_config.rs:611-625`). The loader does the same, so a .apr imported at `316dee2cd4`, which has
   no such key, is refused. Every Qwen3.5 load path, `from_gguf` included, also refuses an ssm_a value that is not
   strictly negative. A_log is ≥ 0 on 192 of the 4B's 768 heads, so A_log under the ssm_a name cannot pass. Both
   are in QQE-008 (desk read, 2026-10-04).
5. **Contract text on `fold-r2r3`, fixed at `80723cf206`.** `qwen35-train-gdn-v1` stated the old recurrence:
   α = sigmoid(·) and o = (Sᵀq) ⊙ z. The code decays by e^g (`gdn.rs:291`), reads the decayed state, scales the
   read-out by 1/√d_k and applies z only in the gated norm, and the contract now says so. The shared
   `gated-delta-net-v1` 1.0.0 on main still states the old recurrence, which neither crate computes; R21's kernels
   follow `qwen35-train-cuda-v1` and the CPU oracle, not that file.
