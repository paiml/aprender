# #4313 prefill — RTX 4090 receipt at #4502 head

aprender-79, 2026-09-28 (Madrid ~00:00–00:40). The cell is lambda-vector: RTX 4090, sm_89,
driver 580.119.02. The model is `Qwen3.5-4B-Q4_K_M.gguf`, the same file as the SRV-FIX-002
receipts. The prompts come from `docs/audits/srv-fix-002/gx10/prompts.sh`; sha256s are in
`prompt-sums.txt`. Every run was sequential under `/tmp/apr-gpu.lock` with `--max-tokens 1`.
Each number is apr's own `[qwen35] batched prefill:` stderr line.

## Is the #4313 fix in #4502 (`8d021f61e1`)?

| #4313 piece | In `8d021f61e1`? |
|---|---|
| f16 tensor-core prefill GEMM, fp16 weight prewarm, f16 as the default (`4d9dbea515` … `8ea9b35bef`) | **yes** |
| `ea530bd0b9`: create the f16 cuBLAS handle in the prewarm, not in the first prefill (98, `perf/4313-mmq-prefill`) | **no**. This branch cherry-picks it cleanly as `90d84c86d9` |
| MMQ int8 matmul | as the `APR_QWEN35_PREFILL_GEMM=dp4a` leg only. It is not the default and is the slowest leg (below) |
| LM head on the last token only | this run has one line per prefill chunk and no full-vocab logits over the rows. Not separately measured |

## A: `ea530bd0b9` alone (base `8d021f61e1` vs fix `90d84c86d9`, f16 default, n=3 alternating)

| prompt | tokens | base ms (3 runs) | fix ms (3 runs) | prewarm base / fix |
|---|---|---|---|---|
| p2_2k | 2,154 | 229 / 224 / 230 | 228 / 230 / 226 | ~42 / ~90 ms |
| p3_8k | 8,119 | 1031 / 981 / 978 | 1014 / 984 / 999 | ~42 / ~90 ms |
| p5_30k | 28,924 | 6659 / 6613 / 6613 | 6639 / 6733 / 6644 | ~42 / ~90 ms |

**On this cell, `ea530bd0b9` has no effect on prefill time.** The differences are inside
run-to-run noise. It adds about 48 ms to the load-time prewarm, which is the handle creation
it moves there. It is safe to fold. Measured here, the reason to fold it is correctness of
where the cost lands, not a 4090 speedup.

## B: the three GEMM legs (fix binary, `APR_QWEN35_PREFILL_GEMM`, n=3)

| prompt | f16 (default) | f32 | dp4a (int8) |
|---|---|---|---|
| p2_2k | 362 / 326 / 463 ms | 750 / 796 / 716 ms | 986 / 925 / 1017 ms |
| p5_30k | 6715 / 6728 / 6733 ms | 11951 / 11905 / 12240 ms | 15081 / 14755 / 14691 ms |

- f16 is 1.8× faster than f32 at 29k tokens.
- **The int8 leg is the slowest: 0.8× of f32 and 0.45× of f16.**
- The p2_2k f16 runs in B are slower than the same binary and prompt in A (326–463 ms vs
  226–230 ms). A and B ran about 10 minutes apart, and the GPU was not reserved beyond the
  lock. The ranking of the legs is the same in every run.

## C: against llama.cpp `d1d3c3396` (`llama-bench -fa 1 -ngl 99 -r 3 -n 0`)

| prompt | apr f16 tok/s (A, median) | llama.cpp pp tok/s | apr / llama |
|---|---|---|---|
| 2,154 | 9,397 | 13,609 | 0.69× |
| 8,119 | 8,253 | 14,040 | 0.59× |
| 28,924 | 4,374 | 12,724 | 0.34× |

The gap grows with context. At long context the loss is attention, not the projection GEMM.
This build prints `attention cuBLAS f32`; SRV-FIX-002's flash path is opt-in via
`APR_QWEN35_PREFILL_ATTENTION=flash`. See `docs/audits/srv-fix-002/rtx4090/RECEIPT.md`.

## Verdict

- The #4313 f16 fix **is in #4502**. The one missing commit, `ea530bd0b9`, is on
  `79/4313-on-4502` and is prefill-neutral on the 4090.
- The "MMQ int8 matmul" half of #4313 is **not met by anything in the tree**. The only int8
  leg (`dp4a`) loses to both f32 and f16.
- Blocker: a real MMQ kernel, i.e. int8 activations with an int32 accumulate on tensor-core
  `mma` and not DP4A. That is #4376's slice 1 (aprender-1c). Until it lands, f16 is the
  fastest prefill GEMM measured here.

Evidence: `runs.tsv` (A), `legs.tsv` (B), `llama.csv` (C), `bin-sums.txt`, `prompt-sums.txt`.
