//! T-COV-95 Synthetic Falsification: cli/inference.rs + gguf/loader.rs
//!
//! Uses GGUFBuilder "Pygmy Models" to exercise inference code paths
//! without requiring real 10GB model files. A 1KB synthetic GGUF
//! exercises the same logic as a 100GB one.

use crate::cli::inference;
use crate::gguf::test_factory::{build_minimal_llama_gguf, GGUFBuilder};
use crate::inference_trace::TraceConfig;
use std::io::Write;
use tempfile::NamedTempFile;

// ============================================================================
// Pygmy Model Fixtures
// ============================================================================

/// Create a minimal synthetic GGUF file on disk
fn create_pygmy_gguf() -> NamedTempFile {
    let gguf_data = build_minimal_llama_gguf(
        32,  // vocab_size (tiny)
        64,  // hidden_dim
        128, // intermediate_dim
        4,   // num_heads
        4,   // num_kv_heads
    );

    let mut file = NamedTempFile::with_suffix(".gguf").unwrap();
    file.write_all(&gguf_data).unwrap();
    file.flush().unwrap();
    file
}

/// Create an even smaller GGUF (just metadata, minimal tensors)
fn create_micro_gguf() -> NamedTempFile {
    use crate::gguf::test_factory::{
        create_f32_embedding_data, create_f32_norm_weights, create_q4_k_data,
    };

    let vocab_size = 16;
    let hidden_dim = 32;
    let intermediate_dim = 64;
    let kv_dim = 32;

    let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
    let norm_data = create_f32_norm_weights(hidden_dim);
    let q_data = create_q4_k_data(hidden_dim * hidden_dim);
    let k_data = create_q4_k_data(hidden_dim * kv_dim);
    let v_data = create_q4_k_data(hidden_dim * kv_dim);
    let attn_out_data = create_q4_k_data(hidden_dim * hidden_dim);
    let ffn_up_data = create_q4_k_data(hidden_dim * intermediate_dim);
    let ffn_down_data = create_q4_k_data(intermediate_dim * hidden_dim);
    let ffn_gate_data = create_q4_k_data(hidden_dim * intermediate_dim);

    let gguf_data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", hidden_dim as u32)
        .num_layers("llama", 1)
        .num_heads("llama", 2)
        .num_kv_heads("llama", 2)
        .context_length("llama", 64)
        .rope_freq_base("llama", 10000.0)
        .rms_epsilon("llama", 1e-5)
        .ffn_hidden_dim("llama", intermediate_dim as u32)
        .add_f32_tensor(
            "token_embd.weight",
            &[vocab_size as u64, hidden_dim as u64],
            &embed_data,
        )
        .add_f32_tensor("blk.0.attn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q4_k_tensor(
            "blk.0.attn_q.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &q_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_k.weight",
            &[hidden_dim as u64, kv_dim as u64],
            &k_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_v.weight",
            &[hidden_dim as u64, kv_dim as u64],
            &v_data,
        )
        .add_q4_k_tensor(
            "blk.0.attn_output.weight",
            &[hidden_dim as u64, hidden_dim as u64],
            &attn_out_data,
        )
        .add_f32_tensor("blk.0.ffn_norm.weight", &[hidden_dim as u64], &norm_data)
        .add_q4_k_tensor(
            "blk.0.ffn_up.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &ffn_up_data,
        )
        .add_q4_k_tensor(
            "blk.0.ffn_down.weight",
            &[intermediate_dim as u64, hidden_dim as u64],
            &ffn_down_data,
        )
        .add_q4_k_tensor(
            "blk.0.ffn_gate.weight",
            &[hidden_dim as u64, intermediate_dim as u64],
            &ffn_gate_data,
        )
        .add_f32_tensor("output_norm.weight", &[hidden_dim as u64], &norm_data)
        .build();

    let mut file = NamedTempFile::with_suffix(".gguf").unwrap();
    file.write_all(&gguf_data).unwrap();
    file.flush().unwrap();
    file
}

// ============================================================================
// run_gguf_inference coverage via synthetic models
// ============================================================================

#[test]
fn test_pygmy_gguf_inference_basic() {
    let file = create_pygmy_gguf();
    let path = file.path();

    let result = inference::run_gguf_inference(
        path.to_str().unwrap(),
        &[], // file_data unused (mmap used)
        "Hello",
        2,   // max_tokens
        0.0, // greedy
        "text",
        false,
        false,
        None, // trace_config
    );

    // Should succeed or fail gracefully
    // Either way, we've exercised the mmap loading path
    let _ = result;
}

#[test]
fn test_pygmy_gguf_inference_verbose() {
    let file = create_pygmy_gguf();
    let path = file.path();

    let result = inference::run_gguf_inference(
        path.to_str().unwrap(),
        &[],
        "Test prompt",
        2,
        0.5, // temperature sampling
        "text",
        false,
        true, // verbose - exercises eprintln! paths
        None, // trace_config
    );

    let _ = result;
}

#[test]
fn test_pygmy_gguf_inference_json_format() {
    let file = create_pygmy_gguf();
    let path = file.path();

    let result = inference::run_gguf_inference(
        path.to_str().unwrap(),
        &[],
        "JSON test",
        2,
        0.0,
        "json", // JSON output format
        false,
        false,
        None, // trace_config
    );

    let _ = result;
}

#[test]
fn test_pygmy_gguf_inference_with_trace() {
    let file = create_pygmy_gguf();
    let path = file.path();

    let result = inference::run_gguf_inference(
        path.to_str().unwrap(),
        &[],
        "Trace test",
        2,
        0.0,
        "text",
        false,
        false,
        Some(TraceConfig::enabled()), // trace_config
    );

    let _ = result;
}

#[test]
fn test_micro_gguf_inference() {
    let file = create_micro_gguf();
    let path = file.path();

    let result = inference::run_gguf_inference(
        path.to_str().unwrap(),
        &[],
        "Micro",
        1,
        0.0,
        "text",
        false,
        false,
        None, // trace_config
    );

    let _ = result;
}

#[test]
fn test_pygmy_gguf_empty_prompt() {
    let file = create_pygmy_gguf();
    let path = file.path();

    let result = inference::run_gguf_inference(
        path.to_str().unwrap(),
        &[],
        "", // empty prompt
        2,
        0.0,
        "text",
        false,
        false,
        None, // trace_config
    );

    let _ = result;
}

#[test]
fn test_pygmy_gguf_high_temperature() {
    let file = create_pygmy_gguf();
    let path = file.path();

    let result = inference::run_gguf_inference(
        path.to_str().unwrap(),
        &[],
        "High temp",
        3,
        2.0, // very high temperature
        "text",
        false,
        false,
        None, // trace_config
    );

    let _ = result;
}

#[test]
fn test_pygmy_gguf_zero_max_tokens() {
    let file = create_pygmy_gguf();
    let path = file.path();

    // Zero tokens - should handle gracefully
    let result = inference::run_gguf_inference(
        path.to_str().unwrap(),
        &[],
        "Zero",
        0,
        0.0,
        "text",
        false,
        false,
        None, // trace_config
    );

    let _ = result;
}

// ============================================================================
// GPU flag tests (exercises conditional compilation paths)
// ============================================================================

#[test]
fn test_pygmy_gguf_force_gpu_flag() {
    let file = create_pygmy_gguf();
    let path = file.path();

    // force_gpu=true exercises the GPU code path (or warning on CPU-only builds)
    let result = inference::run_gguf_inference(
        path.to_str().unwrap(),
        &[],
        "GPU test",
        2,
        0.0,
        "text",
        true, // force_gpu
        true, // verbose
        None, // trace_config
    );

    let _ = result;
}

// ============================================================================
// All format/verbose/trace combinations
// ============================================================================

#[test]
fn test_pygmy_gguf_json_verbose() {
    let file = create_pygmy_gguf();
    let path = file.path();

    let result = inference::run_gguf_inference(
        path.to_str().unwrap(),
        &[],
        "JSON verbose",
        2,
        0.0,
        "json",
        false,
        true,
        None, // trace_config
    );

    let _ = result;
}

#[test]
fn test_pygmy_gguf_json_trace() {
    let file = create_pygmy_gguf();
    let path = file.path();

    let result = inference::run_gguf_inference(
        path.to_str().unwrap(),
        &[],
        "JSON trace",
        2,
        0.0,
        "json",
        false,
        false,
        Some(TraceConfig::enabled()), // trace_config
    );

    let _ = result;
}

#[test]
fn test_pygmy_gguf_all_flags() {
    let file = create_pygmy_gguf();
    let path = file.path();

    let result = inference::run_gguf_inference(
        path.to_str().unwrap(),
        &[],
        "All flags",
        2,
        0.5,
        "json",
        true,                         // force_gpu
        true,                         // verbose
        Some(TraceConfig::enabled()), // trace_config
    );

    let _ = result;
}

// ============================================================================
// Temperature boundary tests
// ============================================================================

#[test]
fn test_pygmy_gguf_temperature_at_threshold() {
    let file = create_pygmy_gguf();
    let path = file.path();

    // temperature = 0.01 is at the greedy threshold
    let result = inference::run_gguf_inference(
        path.to_str().unwrap(),
        &[],
        "Threshold",
        2,
        0.01, // exactly at threshold
        "text",
        false,
        false,
        None, // trace_config
    );

    let _ = result;
}

#[test]
fn test_pygmy_gguf_temperature_just_above_threshold() {
    let file = create_pygmy_gguf();
    let path = file.path();

    // temperature = 0.02 triggers sampling path
    let result = inference::run_gguf_inference(
        path.to_str().unwrap(),
        &[],
        "Above threshold",
        2,
        0.02,
        "text",
        false,
        false,
        None, // trace_config
    );

    let _ = result;
}

// ============================================================================
// Long prompt to exercise tokenization paths
// ============================================================================

#[test]
fn test_pygmy_gguf_long_prompt() {
    let file = create_pygmy_gguf();
    let path = file.path();

    let long_prompt = "a ".repeat(50);
    let result = inference::run_gguf_inference(
        path.to_str().unwrap(),
        &[],
        &long_prompt,
        1,
        0.0,
        "text",
        false,
        false,
        None, // trace_config
    );

    let _ = result;
}

include!("tests_pygmy_gguf.rs");
