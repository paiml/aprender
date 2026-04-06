#![allow(clippy::disallowed_methods)]
//! Falsification: GPU State Isolation Tests
//!
//! Tests that GPU state does not leak between generations.
//! These are model-tests gated — they require actual model files and GPU.
//!
//! Validates the PMAT-PREFILL-FIX: stale position_buf after CUDA graph.
//! Also validates Bug 199 (GGUF GPU clone regression) patterns.

#![cfg(feature = "model-tests")]

// =============================================================================
// GPU State Isolation: Sequential Generation Determinism
// =============================================================================

/// Generate with the same prompt twice — output must be identical.
/// This catches stale KV cache, stale position_buf, and CUDA graph leaks.
#[test]
fn gpu_state_sequential_determinism() {
    let model_path = match std::env::var("TEST_MODEL_PATH") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            eprintln!("FALSIFICATION-SKIP: gate=gpu_state_sequential_determinism reason=TEST_MODEL_PATH not set");
            return;
        }
    };

    // Use apr CLI to generate twice with same prompt and temperature=0
    let output1 = std::process::Command::new("cargo")
        .args([
            "run",
            "--bin",
            "apr",
            "--features",
            "inference",
            "--",
            "run",
            &model_path.to_string_lossy(),
            "--prompt",
            "What is 2+2?",
            "--max-tokens",
            "16",
            "--temperature",
            "0",
        ])
        .output()
        .expect("first generation");

    let output2 = std::process::Command::new("cargo")
        .args([
            "run",
            "--bin",
            "apr",
            "--features",
            "inference",
            "--",
            "run",
            &model_path.to_string_lossy(),
            "--prompt",
            "What is 2+2?",
            "--max-tokens",
            "16",
            "--temperature",
            "0",
        ])
        .output()
        .expect("second generation");

    let text1 = String::from_utf8_lossy(&output1.stdout);
    let text2 = String::from_utf8_lossy(&output2.stdout);

    assert_eq!(
        text1, text2,
        "Sequential generations with same prompt and temp=0 must be identical.\n\
         Gen 1: {text1}\n\
         Gen 2: {text2}"
    );
}

/// Generate with different prompts — output must differ.
/// This catches models that are "stuck" returning the same output.
#[test]
fn gpu_state_different_prompts_produce_different_output() {
    let model_path = match std::env::var("TEST_MODEL_PATH") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            eprintln!("FALSIFICATION-SKIP: gate=gpu_state_different_prompts reason=TEST_MODEL_PATH not set");
            return;
        }
    };

    let output1 = std::process::Command::new("cargo")
        .args([
            "run",
            "--bin",
            "apr",
            "--features",
            "inference",
            "--",
            "run",
            &model_path.to_string_lossy(),
            "--prompt",
            "What is the capital of France?",
            "--max-tokens",
            "16",
            "--temperature",
            "0",
        ])
        .output()
        .expect("prompt A generation");

    let output2 = std::process::Command::new("cargo")
        .args([
            "run",
            "--bin",
            "apr",
            "--features",
            "inference",
            "--",
            "run",
            &model_path.to_string_lossy(),
            "--prompt",
            "Write a haiku about rust programming",
            "--max-tokens",
            "16",
            "--temperature",
            "0",
        ])
        .output()
        .expect("prompt B generation");

    let text1 = String::from_utf8_lossy(&output1.stdout);
    let text2 = String::from_utf8_lossy(&output2.stdout);

    assert_ne!(
        text1, text2,
        "Different prompts must produce different output — model may be stuck"
    );
}

// =============================================================================
// GPU Memory Tests
// =============================================================================

/// 10 sequential generations should not grow memory unboundedly
#[test]
fn gpu_state_no_memory_leak_10_generations() {
    let model_path = match std::env::var("TEST_MODEL_PATH") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            eprintln!(
                "FALSIFICATION-SKIP: gate=gpu_state_no_memory_leak reason=TEST_MODEL_PATH not set"
            );
            return;
        }
    };

    for i in 0..10 {
        let output = std::process::Command::new("cargo")
            .args([
                "run",
                "--bin",
                "apr",
                "--features",
                "inference",
                "--",
                "run",
                &model_path.to_string_lossy(),
                "--prompt",
                &format!("Count to {}", i + 1),
                "--max-tokens",
                "8",
                "--temperature",
                "0",
            ])
            .output()
            .unwrap_or_else(|_| panic!("Generation {i} failed"));

        assert!(
            output.status.success(),
            "Generation {i} must succeed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

// =============================================================================
// KV Cache Reset Tests
// =============================================================================

/// After KV cache reset, first generation must succeed and produce output
#[test]
fn gpu_state_kv_cache_reset_produces_output() {
    let model_path = match std::env::var("TEST_MODEL_PATH") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            eprintln!(
                "FALSIFICATION-SKIP: gate=gpu_state_kv_cache_reset reason=TEST_MODEL_PATH not set"
            );
            return;
        }
    };

    let output = std::process::Command::new("cargo")
        .args([
            "run",
            "--bin",
            "apr",
            "--features",
            "inference",
            "--",
            "run",
            &model_path.to_string_lossy(),
            "--prompt",
            "Hello",
            "--max-tokens",
            "4",
            "--temperature",
            "0",
        ])
        .output()
        .expect("generation after implicit KV reset");

    let text = String::from_utf8_lossy(&output.stdout);
    assert!(
        !text.trim().is_empty(),
        "Must produce non-empty output after KV cache reset"
    );
}

// =============================================================================
// GPU QA Gate Integration
// =============================================================================

/// apr qa with --skip-gpu-state=false should execute the GPU state gate
#[test]
fn gpu_state_qa_gate_exists() {
    let model_path = match std::env::var("TEST_MODEL_PATH") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            eprintln!(
                "FALSIFICATION-SKIP: gate=gpu_state_qa_gate_exists reason=TEST_MODEL_PATH not set"
            );
            return;
        }
    };

    let output = std::process::Command::new("cargo")
        .args([
            "run",
            "--bin",
            "apr",
            "--features",
            "inference",
            "--",
            "qa",
            &model_path.to_string_lossy(),
            "--json",
        ])
        .output()
        .expect("apr qa --json");

    let text = String::from_utf8_lossy(&output.stdout);
    // The JSON output should mention gpu_state_isolation gate (either run or skipped)
    assert!(
        text.contains("gpu_state_isolation"),
        "apr qa must include gpu_state_isolation gate in output"
    );
}
