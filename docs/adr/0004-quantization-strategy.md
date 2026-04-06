# ADR-0004: Quantization Strategy

## Status

Accepted

## Date

2024-11-22

## Context

Realizar needs to support quantized models for efficient inference on memory-constrained hardware. Options:

1. **Q4_0/Q8_0** - llama.cpp legacy formats (4-bit, 8-bit per weight)
2. **K-quants** - Q4_K, Q5_K, Q6_K with variable bit allocation
3. **GPTQ** - Post-training quantization with calibration data
4. **AWQ** - Activation-aware weight quantization

## Decision

Implement Q4_0, Q8_0, Q4_K, Q5_K, Q6_K dequantization from scratch.

## Rationale

1. **Compatibility** - Q4_0/Q8_0 are ubiquitous in llama.cpp ecosystem
2. **Quality vs Size** - K-quants offer better quality/size tradeoffs
3. **No calibration needed** - Dequantization doesn't require training data
4. **Inference focus** - We dequantize at inference, don't quantize

## Implementation

```rust
pub enum QuantType {
    Q4_0,  // 4-bit, block size 32, single scale
    Q8_0,  // 8-bit, block size 32, single scale
    Q4_K,  // 4-bit, super-blocks with min/scale
    Q5_K,  // 5-bit, super-blocks
    Q6_K,  // 6-bit, super-blocks
}

pub fn dequantize(data: &[u8], qtype: QuantType) -> Vec<f32> {
    match qtype {
        QuantType::Q4_0 => dequantize_q4_0(data),
        QuantType::Q8_0 => dequantize_q8_0(data),
        QuantType::Q4_K => dequantize_q4_k(data),
        // ...
    }
}
```

## Consequences

### Positive
- Support all common llama.cpp quantization formats
- Memory-efficient loading for large models
- No external quantization libraries needed

### Negative
- Dequantization adds CPU overhead per inference
- Quality loss vs FP16/FP32
- Must track format changes in llama.cpp

## Performance Analysis

| Format | Bits/Weight | Memory (7B model) | Quality Loss |
|--------|-------------|-------------------|--------------|
| FP16 | 16 | 14 GB | Baseline |
| Q8_0 | 8 | 7 GB | <0.1% |
| Q6_K | 6.5 | 5.7 GB | <0.5% |
| Q5_K | 5.5 | 4.8 GB | <1% |
| Q4_K | 4.5 | 3.9 GB | <2% |
| Q4_0 | 4 | 3.5 GB | <3% |

## Validation

**Falsifiable claim**: Q4_K quantization provides <1% perplexity degradation vs FP16.

**Test**: Measure perplexity on WikiText-2 for both formats with same model.

## References

- [K-quants paper](https://github.com/ggerganov/llama.cpp/pull/1684)
- [llama.cpp quantization](https://github.com/ggerganov/llama.cpp)
