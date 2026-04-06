# Design by Contract in Trueno

## Overview

Trueno implements kernel-level contracts as the computational foundation of the
Sovereign AI Stack's Design by Contract (DbC) system. These contracts enforce
tensor layout, quantization format, and shape invariants at the boundary between
high-level ML code and SIMD/GPU kernels.

## Contract Hierarchy

```
aprender (import) -- enforce_architecture_completeness()  [tensor names]
                     |
realizar (load)   -- ValidatedModelConfig::validate()     [architecture]
                     |
trueno (kernel)   -- contracts::validate_weight_buffer()   [bytes & layout]
```

Trueno sits at the bottom of the stack: if a buffer passes trueno's contract
validation, the kernel will produce correct output.

## Core Module: `src/contracts.rs`

### TensorLayout

Only `RowMajor` is supported. All trueno kernels assume `weight[row * cols + col]`
memory access. Column-major data produces **garbage** -- there is no runtime flag
to switch layout. The canonical constant is `STACK_LAYOUT = TensorLayout::RowMajor`.

### QuantFormat

Block-based quantization descriptors used by every quantized kernel:

| Format | Block Elements | Block Bytes | Bits/Weight | GGML Type ID |
|--------|---------------|-------------|-------------|--------------|
| Q4_0   | 32            | 18          | 4.5         | 2            |
| Q4_1   | 32            | 20          | 5.0         | 3            |
| Q5_0   | 32            | 22          | 5.5         | 6            |
| Q8_0   | 32            | 34          | 8.5         | 8            |
| Q4_K   | 256           | 144         | 4.5         | 12           |
| Q5_K   | 256           | 176         | 5.5         | 13           |
| Q6_K   | 256           | 210         | 6.5         | 14           |

All seven formats are listed in `ALL_FORMATS` and can be looked up by GGML type
ID via `format_by_ggml_type(type_id)`.

### Validation Functions

- **`validate_weight_buffer(name, ggml_type, actual_bytes, rows, cols)`** --
  Primary entry point. Resolves the GGML type ID to a `QuantFormat`, then checks
  that `actual_bytes == rows * ceil(cols / block_size) * block_bytes`.
- **`QuantFormat::validate_buffer(name, actual_bytes, rows, cols)`** -- Same
  check on a known format (skips the type lookup).
- **`QuantFormat::expected_bytes(rows, cols)`** -- Pure computation, no error.
- **`validate_f32_buffer(name, actual_elements, rows, cols)`** -- Unquantized:
  `actual_elements == rows * cols`.
- **`validate_gemv_shapes(name, weight_rows, weight_cols, input_len, output_len)`** --
  GEMV invariant: `weight_cols == input_len` and `weight_rows == output_len`.

### LAYOUT-001: Row-Major Only

APR and realizar are exclusively row-major. GGUF column-major data is transposed
at the import boundary (in aprender). By the time data reaches trueno, it is
**always** row-major. The colmajor kernel variants are deprecated and must not be
called with APR/GGUF data.

### expect() Unreachability Proofs

Trueno uses `expect("message")` instead of `unwrap()`. Each `expect()` message
documents **why** the branch is unreachable given the precondition established by
contract validation. This serves as an inline proof obligation.

## Source of Truth

- **Contract YAML**: `~/src/aprender/contracts/tensor-layout-v1.yaml`
- **Rust implementation**: `src/contracts.rs`
- **Quantization constants**: `crates/trueno-quant/src/lib.rs`
- **Full DbC spec**: `~/src/aprender/docs/specifications/enforce-provable-DbC.md`

## Running Falsification Tests

```bash
cargo test --lib -- contracts
cargo test --lib -- validate_weight
cargo test --lib -- validate_gemv
cargo test --lib -- validate_f32
```

## Cross-References

- [aprender DbC docs](../../aprender/docs/design-by-contract.md)
- [realizar DbC docs](../../realizar/docs/design-by-contract.md)
- [trueno-gpu DbC docs](../trueno-gpu/docs/design-by-contract.md)
- [trueno-quant DbC docs](../crates/trueno-quant/docs/design-by-contract.md)
