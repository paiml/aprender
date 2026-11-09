# APR-EMBED-001 EG-4 (#4903): GPU forward against the CPU forward

**Status: sm_89 measured; sm_121 not run.** The second required host is not measured, so this is not the EG-4
gate receipt.

| Item | Value |
|------|-------|
| GPU | GeForce RTX 4060 Laptop, compute capability 8.9, driver 595.99.02 |
| sm_121 | not run: no build on that host in this change |
| Files | unsloth UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, Q8_0, BF16, F16 (6 of 6); ggml-org BF16, Q8_0 (2 of 2, control) |
| Lengths | 8, 128, 1,024, 2,000 tokens on the real files; 2,000 and 8,000 on the long-context fixture (unit test) |
| Measure | cosine and max \|Δ\| of the unit-length 768-wide vectors, GPU against `EmbeddingGemma2::embed` on the same file and tokens |
| Harness | the real-file cells were run by a one-off test that is **not** in the tree: it maps each file, builds both encoders and prints one line per cell |

## Unit tests (`--features cuda`, run with `--nocapture`: no `SKIP` line)

`forward_embedding_gemma2_cuda_tests.rs`, 8 of 8 pass: windowed and global attention against
`parallel_multihead_attention_windowed` (one pass and many passes), the whole forward on the unit and served
fixtures at 1e-5, 2,000 and 8,000 tokens at 1e-4, I-5 refusal at construction and at a projection, and the CPU's
refusals (empty input, over the context length, a token outside the vocabulary). `band_softmax` PTX test passes.

## Real files, GPU against CPU

| File | Min cosine | Max \|Δ\| | Finite |
|------|------------|-----------|--------|
| UD-Q4_K_XL | 0.9974403 | 0.01146 | 4/4 |
| UD-Q5_K_XL | 0.9999064 | 0.00177 | 4/4 |
| UD-Q6_K_XL | 0.9999073 | 0.00175 | 4/4 |
| Q8_0 | 0.9993603 | 0.00491 | 4/4 |
| BF16 | 1.0000002 | 4.5e-7 | 4/4 |
| F16 | 1.0000001 | 3.7e-7 | 4/4 |
| ggml-org BF16 | 1.0000002 | 4.5e-7 | 4/4 |
| ggml-org Q8_0 | 0.9993603 | 0.00491 | 4/4 |

All 32 cells are finite and above G11's fail-closed floor of 0.98. No line reads `falling back to CPU`: the GPU
encoder has no CPU path.

BF16 and F16 agree to float rounding. The quantized files do not, and the gap follows the weight precision. The
likely cause, not yet proven: the CPU's `matmul_any` runs Q8_0 and the K-quants through fused integer kernels that
quantize the activations, while the GPU projection is an f32 GEMM on dequantized weights. A host threshold for GPU
against CPU still needs the thresholds file's own method (known-good and known-bad pairs, n = 5, each host).
