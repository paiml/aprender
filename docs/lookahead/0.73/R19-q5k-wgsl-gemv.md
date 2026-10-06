# Row 19: a Q5_K WGSL GEMV

Read at origin/main 11f844a772. No build, no model file opened.

## Today
- Every wgpu weight upload goes through `dequant_tensor_public`
  (`crates/aprender-serve/src/gpu/adapters/wgpu_adapter.rs:301`). Q5_K is dequantized
  to F32 on the host at `:312` (`dequantize_q5_k`, CPU reference at
  `crates/aprender-serve/src/quantize/dequant_q4k.rs:65`) and uploaded as F32.
- Only Q4_K keeps its raw bytes on the device: the raw list filters on
  `GGUF_TYPE_Q4_K` (`wgpu_adapter.rs:291`), and `q4k_gemv_pipeline`
  (`crates/aprender-compute/src/backends/gpu/device/linalg/wgsl_forward.rs:64-65`,
  built at `:434-439`, C-WGPU-Q4K-001) reads `Q4K_GEMV_SHADER`
  (`crates/aprender-compute/src/backends/gpu/shaders/basic_ops.rs:555`).
- P4 item 4 adds WGSL GEMVs for Q6_K, Q8_0 and Q4_0. Q5_K is in no ticket.

## Cost of the host dequant
A Q5_K super-block is 176 bytes for 256 weights, 5.5 bits a weight. F32 is 32 bits. So
each Q5_K tensor takes 5.8x its file size in device memory, the load pays a host
dequant pass, and every decode GEMV reads 5.8x the bytes. On a memory-bound M=1 decode
that is the whole speed gap for those tensors, and it is an E2 cost on every wgpu cell
(C1-C3). E4 lists it as T3 (host span on load).

## What to port from
- The CUDA fused Q5_K GEMV: `Q5KGemvKernel`
  (`crates/aprender-gpu/src/kernels/quantize/q5k/gemv.rs:1-19`, PAR-003), one block per
  output row, `(k + 255) / 256` super-blocks (`:50-52`), with the super-block layout
  constants `Q5K_SUPER_BLOCK_BYTES` and `Q5K_SUPER_BLOCK_SIZE`.
- The WGSL template: `Q4K_GEMV_SHADER`, its bind group layout (`matmul_bgl`, shared, so
  a Q5_K pipeline needs no new layout), and the raw-buffer map `q4k_weights`
  (`wgsl_forward.rs:82-83`).
- Q5_K is Q4_K's layout plus a 32-byte `qh` array of fifth bits (176 = 144 + 32). The
  WGSL kernel is the Q4K shader plus one bit read per weight: `q | (((qh[l] >> j) & 1) << 4)`.
  Exact byte offsets to be read from `dequant_q4k.rs:65` and the PTX kernel [U].

## Gate on whether it matters (moves the row up or not)
Whether Qwen3.5-4B's GGUF (the 0.73 model) holds Q5_K tensors is [U]. Read it, with no
compute, from the GGUF header tensor table: count tensors by `ggml_type` 13. Common
Q4_K_M quants put some `attn_v` and `ffn_down` tensors at Q5_K or Q6_K, so the default
expectation is "some". If the count is 0, this row stays last; if not, it ranks next to
P4 item 4 and should ride in the same PR as the Q6_K kernel (same template, same tests).

## Draft for the row-19 ticket
1. `Q5K_GEMV_SHADER` in `shaders/basic_ops.rs`, a `q5k_gemv_pipeline` next to the Q4K
   one, and Q5_K added to the raw-weight filter in `wgpu_adapter.rs`.
2. `dequant_tensor_public` keeps its Q5_K arm for the CPU and for any route that still
   needs F32, but the wgpu forward stops calling it for Q5_K.
3. A load-time counter: tensors uploaded raw vs dequantized, by type, in the receipt.

## Falsifiers
| id | claim | how |
|---|---|---|
| F19-1 | the WGSL Q5_K GEMV equals the CPU dequant then F32 GEMV | random Q5_K blocks, all `qh` bits set and all clear, cosine >= 0.9999 and max abs diff per BPM |
| F19-2 | the fifth bit is read | a block whose `qh` is all 1s against one with all 0s must give different outputs (kills a kernel that ignores `qh`) |
| F19-3 | k not a multiple of 256 is refused or padded, never read past the end | k = 256*n + 32 |
| F19-4 | a Q5_K model's wgpu load has no host dequant for Q5_K | the receipt counter: Q5_K dequantized = 0 (closes E4 T3 for Q5_K) |
| F19-5 | E1 parity holds | E1 cosine >= 0.995 on C1-C3 with the kernel |

## Open
- The Q5_K tensor count in the 0.73 GGUF (header read only).
- Byte offsets of `d`, `dmin`, scales, `qh`, `qs` in the 176-byte block.
