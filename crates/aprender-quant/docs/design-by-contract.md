# Design by Contract in Trueno-Quant

## Overview

Trueno-Quant is the single source of truth for K-quantization block sizes and
dequantization routines. It sits at the bottom of the dependency chain:

```
apr-cli -> entrenar -> aprender -> realizar -> trueno -> trueno-quant
```

Every crate that touches quantized weights depends on trueno-quant's constants.

## Block Size Constants as Contracts

The core contract is the mapping from quantization format to block geometry:

| Format | Elements/Block | Bytes/Block | Constants                            |
|--------|---------------|-------------|--------------------------------------|
| Q4_K   | 256           | 144         | `Q4_K_BLOCK_SIZE`, `Q4_K_BLOCK_BYTES` |
| Q5_K   | 256           | 176         | `Q5_K_BLOCK_SIZE`, `Q5_K_BLOCK_BYTES` |
| Q6_K   | 256           | 210         | `Q6_K_BLOCK_SIZE`, `Q6_K_BLOCK_BYTES` |

These constants are consumed by:
- `trueno/src/contracts.rs` -- `QuantFormat` structs re-export these values
- `aprender/src/format/layout_contract.rs` -- import-time buffer validation
- `realizar` -- dequantization buffer allocation during inference

Changing these constants without updating all consumers breaks the chain.

## Quantization Functions

Each quantization function operates on exactly one block of elements:
- `quantize_q4_k(data)` -- Quantizes a `&[f32]` slice into Q4_K bytes
- `quantize_q5_k(data)` -- Quantizes a `&[f32]` slice into Q5_K bytes
- `quantize_q6_k(data)` -- Quantizes a `&[f32]` slice into Q6_K bytes

Matrix variants (`quantize_q4_k_matrix`, etc.) handle multi-row quantization
with proper block alignment.

## Dequantization Invariants

Each `dequantize_*` function has a postcondition:
- Output length = `(input_bytes / block_bytes) * block_elements`
- Input must be a multiple of `block_bytes` in length

The three dequantization functions are:
- `dequantize_q4_k_to_f32(data, num_elements)` -> `Vec<f32>`
- `dequantize_q5_k_to_f32(data, num_elements)` -> `Vec<f32>`
- `dequantize_q6_k_to_f32(data, num_elements)` -> `Vec<f32>`

## Transpose Functions

For GEMV, weight matrices must be in row-major layout. The transpose functions
convert column-major GGUF block data into row-major APR block data:
- `transpose_q4k_for_matmul(data, rows, cols)` -> `Vec<u8>`
- `transpose_q5k_for_matmul(data, rows, cols)` -> `Vec<u8>`
- `transpose_q6k_for_matmul(data, rows, cols)` -> `Vec<u8>`

These are called at the GGUF import boundary in aprender, not at inference time.

## F16 Boundary

`F16_MIN_NORMAL` (approximately 6.1e-5) defines the smallest valid f16 normal
value. Scale factors below this threshold would produce NaN on round-trip through
f16 encoding. This constant guards the quantization path.

## Source of Truth

- **Block constants**: `src/lib.rs`
- **Consumer**: `~/src/trueno/src/contracts.rs`
- **Full DbC spec**: `~/src/aprender/docs/specifications/enforce-provable-DbC.md`
