mod dequantize;
mod dequantize_02;
mod q8k_superblock;
mod dequantize_q8ksuperblock_quantize;
mod dequantize_03;
mod interleaved_q4k;
mod quantize_activations;
mod tests_07;
mod tests_08; // Phase 36: Fused K-quantization math kernel tests (proptest)
mod f16; // Phase 36: SIMD helper function tests (proptest)
mod tests_10; // Phase 37: Additional SIMD coverage tests
mod tests_11; // Phase 38: Extended coverage for quantize/types.rs
mod tests_12; // Phase 38: Deep coverage tests for activation.rs
mod tests_13; // Phase 39: Additional coverage for quantize/mod.rs uncovered paths
mod tests_14; // Phase 39: Parallel K-quantization coverage tests for parallel_k.rs
mod tests_15; // Activation function edge cases: special floats, boundaries, SIMD paths
mod tests_16; // F16 LUT, Q8K dequantize, SimdBackend Display, DequantStats, extract_scale_min blocks 4-7
mod tests_17; // Additional activation.rs coverage: AVX2 remainder loops, fused function edge cases
mod tests_18; // Phase 40: SIMD coverage enhancement (simd.rs edge cases, alignment, fallback paths)
mod tests_19; // Phase 45: Parallel dequantization coverage (parallel_dequant.rs)
mod tests_21; // Phase 46: Comprehensive activation.rs coverage
mod f16_02; // Additional SIMD coverage: f16 subnormal, extract_scale_min odd idx, horizontal sums, AVX2 RoPE
mod tests_24; // Comprehensive coverage for quantize/mod.rs functions (f16 LUT, Q8K into, InterleavedQ4K, fused matvec)
mod tests_25; // Popperian SIMD Falsification: performance tests, SIMD/scalar parity, path verification
mod tests_26; // T-COV-001: Error path and edge case coverage tests
mod tests_27; // T-COV-95 Directive 3: Scalar exhaustion for fused_k.rs
mod tests_28; // T-COV-95 Directive 4: Performance Falsification Gate (SIMD vs scalar)
mod dequantize_04; // T-COV-95 Coverage Bridge B7 (Q4_1, Q5_0, Q5_1, Q2_K dequantization)
mod q8k; // T-COV-95 Deep Coverage Bridge (q8k_into, q8_blocks, InterleavedQ4K, f16 LUT)
mod tests_31; // T-COV-95 Coverage Bridge (fused Q4_0/Q8_0 matvec, extract_scale_min blocks 4-7)
mod q8k_02; // T-COV-95 Coverage Bridge (q8k_into, dequant q4_0/q8_0/q4_1/q5_0/q5_1)
mod blocks; // T-COV-95 Extended Coverage (Q8 blocks, dequant edge cases, block boundaries)
mod tests_34; // T-COV-95 Phase 50: Deep coverage for quantize/mod.rs and activation.rs
mod fused_q6k; // T-COV-95 Phase 50: Deep coverage for fused_q5k_q6k.rs and fused_k.rs
mod valid; // T-COV-95: Deep inner-loop coverage for fused_k.rs (scalar + SIMD parity)
