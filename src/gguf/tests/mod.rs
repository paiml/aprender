//! GGUF module tests - shattered from gguf_monolith.rs (PMAT-802)
//!
//! This module organizes tests from the monolith into focused part files
//! for better maintainability. Monolith reduced from 54,792 to 29,894 lines.
//!
//! ## Part Files
//!
//! - `part_01.rs`: Config + IMP-101 (RoPE, Causal Attention) + IMP-105 (GQA)
//! - `part_02.rs`: IMP-106 + IMP-107 + IMP-108 (Batch operations)
//! - `part_03.rs`: IMP-109 + IMP-110 (Fused kernels, parallel attention)
//! - `part_04.rs`: IMP-112, IMP-111, IMP-113, IMP-114, PARITY-114
//! - `part_05.rs`: IMP-115 - IMP-119
//! - `part_06.rs`: IMP-121-125, IMP-123, IMP-129, PARITY-002
//! - `part_07.rs`: PARITY-005 - PARITY-010 (Contiguous KV Cache, Batch Processing, Benchmarks)
//! - `part_08.rs`: PARITY-011 - PARITY-012 (Integration QA, GPU Optimization)
//! - `part_09.rs`: PARITY-013 - PARITY-017 (GPU Optimization, Multi-Request Batching)
//! - `part_10.rs`: PARITY-050 - PARITY-054 (Batch Inference Analysis & API Integration)
//! - `part_11.rs`: PARITY-055 - PARITY-063 (Batch Throughput Benchmarking & Speculative Phase 2)
//! - `part_12.rs`: PARITY-070 - PARITY-077 (Phase 3: Quantized Attention)
//! - `part_13.rs`: PARITY-078 - PARITY-086 (Phase 4/5: FlashAttention-2 & Stream-K)
//! - `part_14.rs`: PARITY-018 - PARITY-025 (GPU Batch FFN & Request Infrastructure)
//! - `part_15.rs`: PARITY-026 - PARITY-034 (FlashAttention & Infrastructure)
//! - `part_16.rs`: PARITY-035 (Chunked Prefill for Long Contexts)
//! - `part_17.rs`: Phase 33 - Forward Pass Coverage (forward/core.rs)
//! - `part_18.rs`: Phase 33 - GGUF Loader Coverage (loader.rs)
//! - `part_19.rs`: Phase 34 - Thread-safe Cache (cached/sync.rs)
//! - `part_20.rs`: Phase 34 - Matmul Coverage (matmul.rs)
//! - `part_21.rs`: Phase 34 - Transformer Structure (transformer.rs)
//! - `part_22.rs`: Phase 35 - GGUF Edge Cases (loader.rs, types.rs)
//! - `part_23.rs`: Phase 36 - GGUF Parsing and Error Handling (mod.rs coverage)
//! - `part_24.rs`: Phase 52 - GGUF Loader Metadata Types (state-space exhaustion)
//! - `part_25.rs`: Phase 54 - Fixture-Based Loader Tests (ModelFixture pattern)

mod bytes; // T-COV-95 Coverage Bridge B6 (loader.rs metadata, parsing, transformer)
mod bytes_02; // T-COV-95 Synthetic Falsification (loader.rs via Pygmy GGUF models)
mod config;
mod imp_115a;
mod imp_121a;
mod loader_tests_build_minimal;
mod parity005a_contiguous;
mod parity011a_bench;
mod parity035a_chunked;
mod phase_fixture_based_loader;
mod phase_forward_pass_coverage;
mod phase_gguf_edge_cases;
mod phase_gguf_loader_coverage;
mod phase_gguf_loader_metadata;
mod phase_gguf_parsing_error;
mod phase_matmul_coverage;
mod phase_thread_safe_cache;
mod phase_transformer_structure;
mod tests_02;
mod tests_03;
mod tests_04;
mod tests_09;
mod tests_10;
mod tests_11;
mod tests_12;
mod tests_13;
mod tests_14;
mod tests_15;
mod tests_27; // T-COV-95 Deep Coverage Bridge (get_tensor_f32 qtypes, rope_type, tied embeddings)
mod tests_28; // T-COV-95 Deep Coverage Bridge (Q2_K/F16/Q4_1/Q5_0/Q5_1, metadata types, decode/encode)
mod tests_30; // T-COV-95 Active Pygmy: Dynamic Falsification (forward_cached via F32 Pygmies)
mod tests_31; // T-COV-95 Data Storm: Multi-Tensor Pygmies for loader.rs
mod tests_32; // T-COV-95 Menagerie of Pygmies: Complex Structural GGUF Generators
mod tests_33; // T-COV-95 Ancestral Pygmies: GGUF Legacy Version & Alignment Assault
mod tests_34; // T-COV-95 Generative Falsification: Proptest GGUF Header Assault
mod tests_35; // Coverage: loader.rs decode/encode edge cases, rope_type paths, metadata accessors // Coverage: OwnedQuantizedModel::to_apr_bytes + from_apr roundtrip
