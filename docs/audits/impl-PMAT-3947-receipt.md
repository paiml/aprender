# PMAT-3947 implementation receipt (PR #3958)

**Ticket:** GitHub #3947. `aprender-core` had no IQ dequantizer, so `apr qa`'s `tensor_contract` could not inspect three release models.
**Base:** `release/0.69.1-batch-2` @ `8963f91a3`. **Author:** Claude (claude-opus-5-5).

## What the diff does
1. `crates/aprender-quant/src/iq.rs` (lib `trueno_quant`) adds decoders for IQ4_NL (20), IQ3_S (21) and IQ4_XS (23), transcribed from `aprender-serve/src/quantize/{iq4_nl,iq3_s,iq4_xs,iq_grids}.rs`. It adds `dequantize_iq_to_f32(type, data, n)` with a typed error: `UnsupportedType`, `PartialBlock`, `Truncated`.
2. `crates/aprender-core/src/format/gguf/shape.rs::get_tensor_f32` dispatches 20/21/23 to it. Every other type still reaches #3656's refusal. Nothing falls back to another format (#3850).
3. `reader_tests_ggml_sizes.rs`: `test_3656` is narrowed to the 8 still-undecoded IQ/TQ ids, and asserts that set's size. The new `test_3947` checks that the three decode, with a 0xEE tail after the payload to catch over-reads.
4. `evidence/iq-dequant-3947/`: oracle harnesses, logs, and `apr qa` before/after.

## Claims and where they are backed
- Census and oracle cells (gguf-py 0.17.1, all 271 IQ tensors, 913,735,680 elements bit-exact through core's `get_tensor_f32`, shapes agree): `evidence/iq-dequant-3947/compare_core.txt`, `compare_core_report.json`.
- `apr qa` before rc 5 (117/144/10 violations) → after rc 0 (290/290/426 pass): `evidence/iq-dequant-3947/qa/`.
- Gate positive control (planted NaN scale → FAIL on exactly those 2 tensors, 32 and 256 NaNs): `qa/gate-control.json`.
- Unit tests pin real blocks and are killed by three planted decoder mutants (not committed, run by the author).

## BEHAVIOUR CHANGE THE LANES MUST JUDGE EXPLICITLY (cop's requirement)
`apr import` on a GGUF takes the raw path first. If raw import fails with "cannot represent exactly" / "not yet supported", the GH-375 fallback (`converter/import.rs::try_gguf_raw_import`, ~line 216) moves to dequant→F32, which calls `get_tensor_f32`. Before this PR, that call refused IQ tensors, so the import failed. It now succeeds. Measured on lambda, CPU-only:

| run | before @ 8963f91a3 | after @ 18f10a78e |
|---|---|---|
| `apr import Qwen2.5-0.5B-Instruct-IQ3_M.gguf` | rc 5: GH-375 fallback, then "IQ4_NL … no dequantizer … Refusing" | rc 0: GH-375 fallback, then an APR of **2,524,512,516 bytes** (source 342,752,576, ×7.4: the IQ weights land as F32) |

Output check, 3 prompts, `--max-tokens 20`. The control is `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf`, which also takes the GH-375 fallback (its Q8_0/Q5_0 tensors) and has **no IQ tensors**:
- IQ3_M-imported APR: "2+2" → rambling ("Answers\nMath\nGeometry…"); "capital of France" → "The capital of France is Paris…"; haiku → coherent. The source GGUF is coherent on all three.
- Control APR: "2+2" → "The answer is 4."; "capital of France" → Python code; haiku → repetition loop. The source GGUF is coherent on all three.
Reading: both APRs made by the fallback degrade relative to their GGUF, by about the same amount, so the degradation belongs to the F32-fallback APR path, not to the IQ decode. The decode itself is bit-exact (above). The author does **not** claim IQ-imported APRs are production-quality. The claim is only that the weights are real rather than invented, which is what #3656's refusal existed to prevent.

Question for the lanes: is "import now succeeds and writes large F32 IQ weights, through a fallback path that degrades output for non-IQ models too" acceptable in this PR, or must IQ stay refused at import (i.e. `get_tensor_f32` decodes for inspection but the import fallback keeps refusing)?

## Out of scope (stated on the PR)
The CUDA IQ4 GEMV kernels (`iq4_xs_gemv_warp_reduce`, `iq4_nl_gemv_warp_reduce`) are unmeasured. Moving serve onto `trueno_quant::iq` is #3959.
