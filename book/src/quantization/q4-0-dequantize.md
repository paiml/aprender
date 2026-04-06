# Q4_0 Dequantization

Q4_0 is a 4-bit quantization format used by llama.cpp and GGUF models. Each block stores 32 values using 18 bytes (2 bytes for scale + 16 bytes for quantized values).

## Block Structure

```
┌──────────────────────────────────────────────────┐
│  Q4_0 Block (18 bytes, 32 values)                │
├──────────────────────────────────────────────────┤
│  Bytes 0-1:  Scale (f16)                         │
│  Bytes 2-17: Quantized values (16 bytes)         │
│              Each byte holds 2 x 4-bit values    │
└──────────────────────────────────────────────────┘
```

## Nibble Layout (Critical)

The nibble ordering follows llama.cpp/candle convention:

- **Positions 0-15**: Low nibbles of bytes 0-15
- **Positions 16-31**: High nibbles of bytes 0-15

```rust
// Correct layout (matches llama.cpp/candle)
for (j, &byte) in quants.iter().enumerate() {
    // Low 4 bits go to position j (0-15)
    let low = (byte & 0x0F) as i16 - 8;
    result[out_start + j] = scale * (low as f32);

    // High 4 bits go to position j + 16 (16-31)
    let high = (byte >> 4) as i16 - 8;
    result[out_start + j + 16] = scale * (high as f32);
}
```

### Common Mistake: Interleaved Output

An incorrect implementation might interleave low/high nibbles:

```rust
// WRONG: Interleaved layout
for &byte in quants {
    let low = (byte & 0x0F) as i16 - 8;
    result.push(scale * low as f32);  // Position 0, 2, 4, ...

    let high = (byte >> 4) as i16 - 8;
    result.push(scale * high as f32); // Position 1, 3, 5, ...
}
```

This produces:
```
Interleaved: [low0, high0, low1, high1, low2, high2, ...]
Correct:     [low0, low1, low2, ..., low15, high0, high1, ..., high15]
```

## Dequantization Formula

For each 4-bit value `q` in range [0, 15]:

```
dequantized = scale * (q - 8)
```

The subtraction of 8 centers the values around zero: [-8, 7].

## Implementation

```rust
pub fn dequantize_q4_0(data: &[u8]) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18; // 2 (scale) + 16 (quants)

    let num_blocks = data.len() / BLOCK_BYTES;
    let mut result = vec![0.0f32; num_blocks * BLOCK_SIZE];

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let out_start = block_idx * BLOCK_SIZE;

        // Read f16 scale
        let scale_bytes = &data[block_start..block_start + 2];
        let scale = f16::from_le_bytes([scale_bytes[0], scale_bytes[1]]).to_f32();

        // Read quantized values
        let quants = &data[block_start + 2..block_start + 18];

        // Dequantize: low nibbles first, then high nibbles
        for (j, &byte) in quants.iter().enumerate() {
            let low = (byte & 0x0F) as i16 - 8;
            result[out_start + j] = scale * (low as f32);

            let high = (byte >> 4) as i16 - 8;
            result[out_start + j + 16] = scale * (high as f32);
        }
    }

    Ok(result)
}
```

## Performance

- **Memory**: 4.5 bits per weight (18 bytes / 32 values)
- **Compression**: ~7x vs FP32
- **Speed**: Dequantization is memory-bound; SIMD helps with the arithmetic

## Testing

```rust
#[test]
fn test_q4_0_dequantize_layout() {
    // Create a test block with known values
    let mut block = vec![0u8; 18];

    // Scale = 1.0 (f16)
    block[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());

    // Quant byte 0: low=0, high=15 -> values -8, 7
    block[2] = 0xF0;

    let result = dequantize_q4_0(&block).unwrap();

    // Low nibble at position 0
    assert_eq!(result[0], -8.0);  // 0 - 8 = -8

    // High nibble at position 16
    assert_eq!(result[16], 7.0);  // 15 - 8 = 7
}
```

## Critical Fix

The nibble ordering was fixed to match llama.cpp/candle:

**Before (incorrect):** Interleaved output caused model predictions to be wrong
- "The capital of France is" -> "a country that..." (Paris rank: 24,573)

**After (correct):** Sequential low/high nibbles
- "The capital of France is" -> "Paris" (rank: 470, further improved by tokenizer fix to rank 1)

## See Also

- [Q4_0 Format](./q4-0.md)
- [Q4_K/Q5_K/Q6_K](./k-quants.md)
- [What is Quantization?](./what-is-quantization.md)
