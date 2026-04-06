//! Falsification Tests for Batched Multi-Head Attention (WAPR-PERF-008)
//!
//! Karl Popper's Mandate: Test each stage of the batched attention pipeline.
//!
//! ## The "Catley" Bug
//!
//! The batched attention pipeline produces garbage output, causing transcription
//! to hallucinate "[Catley]" instead of "The birds can use".
//!
//! ## Pipeline Stages to Test
//!
//! 1. InterleavedToBatched: [seq_len, d_model] -> [n_heads, seq_len, head_dim]
//! 2. BatchedTranspose: K from [n_heads, seq_len, head_dim] -> [n_heads, head_dim, seq_len]
//! 3. BatchedGemm: Q @ K^T -> scores [n_heads, seq_len, seq_len]
//! 4. BatchedScale: scores / sqrt(head_dim)
//! 5. BatchedSoftmax: attention weights (VERIFIED WORKING)
//! 6. BatchedGemm: attn @ V -> output
//! 7. BatchedToInterleaved: [n_heads, seq_len, head_dim] -> [seq_len, d_model]

#[allow(dead_code)]
mod cpu_references;
mod small_cuda_tests;
mod whisper_scale_tests;
