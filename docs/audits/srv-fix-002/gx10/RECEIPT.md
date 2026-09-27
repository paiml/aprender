# SRV-FIX-002 gx10 receipt — #4484 flash prefill measured (#4504)

Pre-registration: `../PREREG.md` (8aaa57e108, pushed before any run). Cell gx10 (GB10, sm_121).
Code `origin/79/4484-flash-prefill` @ `3b8fc87675`; `apr 0.70.0 (8aaa57e10)` built on gx10 with
`--features cuda`. Binary and model sha256 values are in `sums.txt` (GGUF `00fe7986…`). GPU use
is proven by the stderr line from each run (`attention flash (f16 inputs, f32 accumulation)` /
`attention cuBLAS f32`) and by the nsys kernel names below.

| Gate | Result | Evidence |
|---|---|---|
| P1 greedy parity, flash vs f32, 64 tokens | **PASS 5/5** byte-identical (~30 → 28,931 prompt tokens) | `parity-verdicts.txt`, `base/rc.txt` |
| P2 planted mutant (softmax scale ×1.25) | **PASS**: parity breaks on **5/5** prompts (first diff at bytes 3–47) | `mutant.diff`, `mutant/rc.txt`; mutant binary sha differs |
| P3 prefill attention vs llama.cpp, 30k prompt | **PASS: 1.15×** (bar ≤ 2.0×) | `p3/*.kern.csv` |

**P3 detail** (nsys `cuda_gpu_kern_sum`, one 28,931-token prompt plus 64 decoded tokens, runs sequential under `/tmp/apr-gpu.lock`):
- apr flash prefill attention is `gdn_prefill_flash_attention_256` (872.1 ms, n=128) plus `gdn_prefill_flash_combine_256` (0.8 ms), **872.9 ms** in all.
- llama.cpp `d1d3c3396` (`llama-server`, 28,920 prompt tokens) runs `flash_attn_ext_f16<256…>` (743.1 ms, n=456) plus `flash_attn_stream_k_fixup_general` (13.1 ms), **756.2 ms** in all.
- apr's f32 path, for comparison, spends 5,874 ms in `causal_mask_softmax` alone.

**Also measured (not gated by the prereg):**
- **End-to-end prefill:** apr flash 27.56 s (1,050 tok/s), apr f32 39.85 s (726 tok/s), llama.cpp 7.45 s (3,881 tok/s). apr is 3.7× slower than llama.cpp. Attention is no longer the reason.
- **Where apr's prefill time goes:** f32 GEMM. `cutlass_80_simt_sgemm_128x256` takes 12.19 s and `magma_sgemmEx` 8.24 s, which is 67% of apr's 30.4 s of kernel time. These are SIMT, not tensor-core. The f16 prefill GEMM (SRV-FIX-001) is not on this branch, or does not engage on gx10.
- **Decode attention at ~29k context:** `gdn_decode_attention_split` takes 723.5 ms over 512 calls (64 tokens × 8 layers), about 1.4 ms per call. llama.cpp's decode flash kernels total under 15 ms.

**Deviation (recorded, not an edit of the prereg):** the first comparator `cmp`'d the whole stdout,
including apr's `Completed in <t>s` timing line, and so reported 0/5. `cmp_out.sh` compares only
the generated text (between `Output:` and `Completed in`), which is what P1 pre-registered.
Both base and mutant verdicts come from `cmp_out.sh`. The mutant was applied at both
`c_exp` sites (`PrefillFlashAttention256Kernel` and `PrefillFlashCombine256Kernel` `build_ptx`).

Completions are not committed; only verdicts, timings and sha's are.
