# PTX Best Practices

This document covers PTX assembly generation best practices learned from
development and debugging of trueno-gpu CUDA kernels.

## Register Types

### U8 Registers Are Not Supported

**Issue**: PTX does not support 8-bit register types (`.u8`, `.s8`).

**Incorrect**:
```ptx
.reg .u8 %rs<1>;  // ERROR: Invalid register type
ld.global.u8 %rs0, [%rd0];
```

**Correct**:
```ptx
.reg .u16 %rh<1>;  // Minimum register size is 16-bit
ld.global.u8 %rh0, [%rd0];  // Load zero-extends to 16-bit
```

The `ld.global.u8` instruction is valid, but it must store into a 16-bit
or larger register. The loaded byte is zero-extended.

## Half-Precision (F16) Operations

### Loading F16 Values

**Issue**: PTX uses `.b16` (binary 16-bit) for half-precision loads, not `.f16`.

**Incorrect**:
```ptx
ld.global.f16 %h0, [%rd0];  // ERROR: Invalid type for load
```

**Correct**:
```ptx
ld.global.b16 %h0, [%rd0];  // Load 16-bit binary value
```

### F16 to F32 Conversion

**Issue**: Converting from f16 to f32 is exact and does NOT require a rounding modifier.

**Incorrect**:
```ptx
cvt.rn.f32.f16 %f0, %h0;  // ERROR: Illegal rounding modifier
```

**Correct**:
```ptx
cvt.f32.f16 %f0, %h0;  // No rounding needed (exact conversion)
```

Note: The reverse conversion (f32 → f16) DOES require a rounding modifier:
```ptx
cvt.rn.f16.f32 %h0, %f0;  // Correct: rounding needed for narrowing
```

## Bitwise Operations

### AND, OR, XOR Types

**Issue**: PTX requires `.b32` (binary) type for bitwise operations, not `.u32`.

**Incorrect**:
```ptx
and.u32 %r2, %r0, %r1;  // ERROR: Invalid type
or.u32 %r2, %r0, %r1;   // ERROR: Invalid type
```

**Correct**:
```ptx
and.b32 %r2, %r0, %r1;  // Use .b32 for bitwise ops
or.b32 %r2, %r0, %r1;
xor.b32 %r2, %r0, %r1;
```

## Warp Shuffle Operations

### Shuffle Width Parameter

**Issue**: The `width` parameter in `shfl.sync.idx` must be a power of 2 (1, 2, 4, 8, 16, or 32).

**Incorrect**:
```ptx
shfl.sync.idx.b32 %f0, %f1, 0, 31, 0xFFFFFFFF;  // ERROR: 31 is not power of 2
```

**Correct**:
```ptx
shfl.sync.idx.b32 %f0, %f1, 0, 32, 0xFFFFFFFF;  // 32 is valid
```

### Warp Participation

**Issue**: `shfl.sync` with mask `0xFFFFFFFF` requires ALL 32 threads in the warp
to execute the instruction simultaneously.

If some threads exit early (e.g., via `@%p bra exit`), the remaining threads
cannot perform shuffles.

**Solution**: Use address clamping to ensure all threads access valid memory,
then skip only the final store for out-of-bounds threads:

```ptx
// Clamp addresses for all threads
min.u32 %r_clamped_row, %r_global_row, %r_m_minus_1;
min.u32 %r_clamped_col, %r_global_col, %r_n_minus_1;

// All threads participate in computation and shuffles
// ...shuffle reduction code...

// Only in-bounds threads store
@%p_row_oob bra exit;
@%p_col_oob bra exit;
st.global.f32 [%rd_out], %f_result;
exit:
    ret;
```

## Memory Alignment

### 4-Byte Alignment for U32 Loads

**Issue**: `ld.global.u32` requires the address to be 4-byte aligned.

**Incorrect**:
```ptx
// If header has 2-byte f16 scale at offset 0, and we try to read
// another u32 at offset 2, it will be misaligned
add.u64 %rd1, %rd0, 2;
ld.global.u32 %r0, [%rd1];  // ERROR: Misaligned access
```

**Correct**: Use smaller loads for misaligned data:
```ptx
ld.global.b16 %rh0, [%rd0];  // Load 2-byte aligned data
```

## Testing PTX

Always validate generated PTX with `ptxas`:

```bash
ptxas -arch=sm_89 -v kernel.ptx -o kernel.cubin
```

Use `compute-sanitizer` for runtime memory access checking:

```bash
compute-sanitizer --tool memcheck ./your_program
```

## References

- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [GitHub Issue #67](https://github.com/paiml/trueno/issues/67) - U8 register bug
- [GitHub Issue #68](https://github.com/paiml/trueno/issues/68) - F16 load/convert bug
