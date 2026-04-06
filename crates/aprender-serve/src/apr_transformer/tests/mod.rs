//! APR Transformer Tests (PMAT-803)
//!
//! Split into parts for PMAT compliance (<2000 lines per file).
//!
//! Part organization:
//! - apr_transformer: Core AprTransformer tests (initialization, forward, error handling)
//! - q4_simd: Quantized Q4_0 SIMD transformer tests
//! - coverage: Comprehensive coverage tests for loader, dequant, config, helpers
//! - part_02: Additional coverage for forward pass, attention, KV cache edge cases

mod apr_transformer;
mod convert_gguf;
mod coverage;
mod tests_02;
mod benchmark_runner;
mod create; // T-COV-95 Deep Coverage Bridge (generate, forward_with_cache, embed, layer params)
mod create_02; // T-COV-95 Coverage Bridge (num_parameters, memory_size, embed edge cases, generate)
mod apr; // T-COV-95 Synthetic Falsification (AprTransformer via Pygmy Models)
mod tests_07; // T-COV-95 Phase 50: ActivationStats, dequant_perrow, block dequant, ForwardTrace
mod tests_08; // T-COV-95 Phase 51: AprTransformer new/embed/forward/traced/generate/from_apr_bytes
mod tests_09; // T-COV-95 Phase 52: dequant blocks, perrow edge cases, ActivationStats, from_apr_bytes errors
mod tests_10; // T-COV-95 Phase 53: from_apr_bytes dtype dispatch (Q4K, Q5K, Q6K, Q8_0, F16)
mod q4k_bytes_q6k;
mod q4_simd;
mod q4k_forward; // T-COV-95 Phase 54: Q4K fused kernel dispatch, Q6K variants, force-F32
