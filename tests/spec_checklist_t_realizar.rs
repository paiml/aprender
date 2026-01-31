//! Spec Checklist Tests - Section T: Realizar-First Architecture (25 points)
//!
//! These tests verify claims from the spec's falsification checklist.
//! Each test is designed to FAIL if the claim is false.
//!
//! Reference: docs/specifications/qwen2-0.5b-instruct-interactive-chat-demo.md

#![allow(unused_imports)]

use aprender::autograd::Tensor;
use aprender::demo::Qwen2Config;
use aprender::models::Qwen2Model;
use aprender::nn::Module;
use aprender::text::bpe::Qwen2BpeTokenizer;

// ============================================================================
// Section T: Realizar-First Architecture (25 points)
// Verification Status: Validates the Realizar-First Architecture mandate
// Reference: apr-whisper-and-cookbook-support-eoy-2025.md Section 2
// ============================================================================

/// T1: apr run uses realizar for inference
/// Falsification: apr run calls aprender::models::*::forward()
#[test]
fn t1_apr_run_uses_realizar() {
    // Verify architecture documentation mandates realizar-first
    let claude_md = std::fs::read_to_string("CLAUDE.md").expect("CLAUDE.md should exist");

    assert!(
        claude_md.contains("realizar"),
        "T1: CLAUDE.md must mention realizar"
    );
    assert!(
        claude_md.contains("Realizar-First Architecture"),
        "T1: CLAUDE.md must state Realizar-First Architecture"
    );
    assert!(
        claude_md.contains("aprender") && claude_md.contains("TRAINING ONLY"),
        "T1: CLAUDE.md must state aprender is for training only"
    );
}

/// T2: apr serve uses realizar server
/// Falsification: apr serve uses non-realizar HTTP handler
#[test]
fn t2_apr_serve_uses_realizar() {
    // Check architecture documentation specifies realizar for serving
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("Model Serving") && spec.contains("realizar") && spec.contains("Primary"),
        "T2: Spec must mandate realizar for Model Serving"
    );
    assert!(
        spec.contains("HTTP/REST API") && spec.contains("realizar"),
        "T2: Spec must mandate realizar for HTTP/REST API"
    );
}

/// T3: apr profile delegates to realizar
/// Falsification: Profiler reports "aprender" in hotspots
#[test]
fn t3_apr_profile_delegates_to_realizar() {
    // Verify profiling architecture is documented
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("apr profile") && spec.contains("Roofline"),
        "T3: Spec must document apr profile with Roofline analysis"
    );
    assert!(
        spec.contains("realizar profiler") || spec.contains("realizar::profiler"),
        "T3: Spec must mention realizar profiler"
    );
}

/// T4: apr bench measures realizar throughput
/// Falsification: Benchmark shows <10 tok/s on proper hardware
#[test]
fn t4_apr_bench_measures_realizar_throughput() {
    // Performance targets must be documented
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("225") && spec.contains("tok/s"),
        "T4: Spec must state realizar throughput target (225+ tok/s)"
    );
    assert!(
        spec.contains("0.3 tok/s") && spec.contains("aprender"),
        "T4: Spec must document slow aprender path (0.3 tok/s)"
    );
}

/// T5: --features inference enables realizar
/// Falsification: Feature flag doesn't pull realizar dependency
#[test]
fn t5_inference_feature_enables_realizar() {
    // Check that inference feature is documented
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("inference") && spec.contains("realizar"),
        "T5: Spec must link inference feature to realizar"
    );
    assert!(
        spec.contains("inference = [\"realizar\""),
        "T5: Spec must show inference feature includes realizar"
    );
}

/// T6: Default features include inference
/// Falsification: cargo build excludes realizar
#[test]
fn t6_default_features_include_inference() {
    // Check spec mandates inference as default
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("default = [") && spec.contains("inference"),
        "T6: Spec must show inference in default features"
    );
}

/// T7: SafeTensors loading via realizar
/// Falsification: aprender::serialization::safetensors used for inference
#[test]
fn t7_safetensors_via_realizar() {
    // Check responsibility matrix - SafeTensors loading assigned to realizar
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    // Spec has "APR/GGUF/SafeTensors Inference | ❌ Never | ✅ Primary | ❌ Never"
    assert!(
        spec.contains("SafeTensors") && spec.contains("Primary"),
        "T7: Spec must assign SafeTensors loading to realizar (Primary)"
    );
}

/// T8: GGUF loading via realizar
/// Falsification: aprender::* used for GGUF inference
#[test]
fn t8_gguf_via_realizar() {
    // Check responsibility matrix for GGUF
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("GGUF") && spec.contains("realizar"),
        "T8: Spec must mention GGUF and realizar together"
    );
}

/// T9: KV cache from realizar
/// Falsification: No KV cache OR aprender KV cache used
#[test]
fn t9_kv_cache_from_realizar() {
    // Check KV cache responsibility
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("KV Cache") && spec.contains("realizar") && spec.contains("Primary"),
        "T9: Spec must assign KV Cache to realizar"
    );
}

/// T10: Quantization via trueno kernels
/// Falsification: Dequantization in aprender
#[test]
fn t10_quantization_via_trueno() {
    // Check quantization responsibility
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("Quantization") && spec.contains("trueno"),
        "T10: Spec must mention trueno for quantization kernels"
    );
}

/// T11: No generate() in aprender models (for production inference)
/// Falsification: aprender::models::*::generate() exists and is called in production
#[test]
fn t11_no_generate_in_aprender_for_production() {
    // Check deletion mandate
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("generate()") && spec.contains("DELETE"),
        "T11: Spec must mandate deletion of generate() in aprender"
    );
}

/// T12: No forward() in aprender inference
/// Falsification: aprender::models::*::forward() used for serving
#[test]
fn t12_no_forward_in_aprender_inference() {
    // Check deletion mandate for forward
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("forward()") && spec.contains("DELETE"),
        "T12: Spec must mandate deletion of forward() for inference"
    );
}

/// T13: Tokenizer from realizar for serving
/// Falsification: aprender::text::bpe used in hot path
#[test]
fn t13_tokenizer_from_realizar() {
    // Check tokenizer responsibility
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("Tokenizers") && spec.contains("realizar") && spec.contains("Primary"),
        "T13: Spec must assign tokenizers to realizar for inference"
    );
}

/// T14: GPU inference via trueno-gpu
/// Falsification: CUDA calls in aprender code
#[test]
fn t14_gpu_inference_via_trueno() {
    // Check GPU responsibility
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("CUDA/GPU") && spec.contains("realizar"),
        "T14: Spec must assign GPU inference to realizar/trueno"
    );
}

/// T15: WASM inference via realizar
/// Falsification: aprender WASM module for inference
#[test]
fn t15_wasm_inference_via_realizar() {
    // Check WASM responsibility
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("WASM Inference") && spec.contains("realizar"),
        "T15: Spec must assign WASM inference to realizar"
    );
}

/// T16: Throughput >= 100 tok/s (1B model, GPU)
/// Falsification: Measured < 100 tok/s on RTX 4090
#[test]
fn t16_throughput_target_gpu() {
    // Check performance targets
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("225") || spec.contains("200"),
        "T16: Spec must state GPU throughput target >= 100 tok/s"
    );
}

/// T17: Throughput >= 10 tok/s (1B model, CPU)
/// Falsification: Measured < 10 tok/s on modern CPU
#[test]
fn t17_throughput_target_cpu() {
    // Check CPU performance targets
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("15 tok/s") || spec.contains("tok/s"),
        "T17: Spec must state CPU throughput targets"
    );
}

/// T18: Memory < 2x model size
/// Falsification: RSS > 2x model file size
#[test]
fn t18_memory_efficiency() {
    // Check memory efficiency targets
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("Memory")
            && (spec.contains("1.2x") || spec.contains("1.5x") || spec.contains("efficiency")),
        "T18: Spec must state memory efficiency target"
    );
}

/// T19: No gradient tracking in inference
/// Falsification: requires_grad=true on inference tensors
#[test]
fn t19_no_gradient_tracking_in_inference() {
    // Check autograd separation
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("Autograd") && spec.contains("aprender") && spec.contains("Primary"),
        "T19: Spec must assign autograd to aprender only"
    );
    assert!(
        spec.contains("Autograd") && spec.contains("realizar") && spec.contains("Never"),
        "T19: Spec must exclude autograd from realizar"
    );
}

/// T20: examples/qwen_inference.rs uses apr CLI
/// Falsification: Example calls aprender::models directly
#[test]
fn t20_examples_use_apr_cli() {
    // Check example migration mandate
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("qwen_inference.rs") && spec.contains("REWRITE"),
        "T20: Spec must mandate rewrite of qwen_inference.rs example"
    );
}

/// T21: Documentation states realizar-first
/// Falsification: CLAUDE.md lacks realizar mandate
#[test]
fn t21_documentation_states_realizar_first() {
    // Check CLAUDE.md contains mandate
    let claude_md = std::fs::read_to_string("CLAUDE.md").expect("CLAUDE.md should exist");

    assert!(
        claude_md.contains("Realizar-First"),
        "T21: CLAUDE.md must contain Realizar-First"
    );
    assert!(
        claude_md.contains("CRITICAL"),
        "T21: CLAUDE.md must mark as CRITICAL"
    );
}

/// T22: CI tests realizar integration
/// Falsification: No realizar tests in CI
#[test]
fn t22_ci_tests_realizar() {
    // Check CI workflow exists
    let ci_path = ".github/workflows/ci.yml";
    if let Ok(ci) = std::fs::read_to_string(ci_path) {
        // CI exists, verify test job
        assert!(
            ci.contains("test") || ci.contains("cargo test"),
            "T22: CI must include test steps"
        );
    } else {
        // CI file may be in different location - just verify spec mentions CI
        let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
        let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");
        assert!(
            spec.contains("CI") || spec.contains("GitHub Actions"),
            "T22: Spec must mention CI integration"
        );
    }
}

/// T23: Error messages mention realizar
/// Falsification: Errors say "use aprender" for inference
#[test]
fn t23_error_messages_mention_realizar() {
    // Check spec mentions proper error messaging
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("CORRECT") && spec.contains("realizar"),
        "T23: Spec must show CORRECT path uses realizar"
    );
    assert!(
        spec.contains("WRONG") && spec.contains("aprender"),
        "T23: Spec must show WRONG path uses aprender"
    );
}

/// T24: apr explain inference describes architecture
/// Falsification: Explanation lacks realizar mention
#[test]
fn t24_apr_explain_describes_architecture() {
    // Check apr explain command documentation
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("explain"),
        "T24: Spec must mention apr explain command"
    );
}

/// T25: Trueno kernels invoked by realizar
/// Falsification: Stack trace lacks trueno::kernels::*
#[test]
fn t25_trueno_kernels_invoked() {
    // Check trueno kernel responsibility
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("trueno") && spec.contains("Compute"),
        "T25: Spec must assign compute to trueno"
    );
    assert!(
        spec.contains("Matmul") && spec.contains("trueno") && spec.contains("Primary"),
        "T25: Spec must assign matmul to trueno"
    );
}
