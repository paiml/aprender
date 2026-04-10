#[test]
fn test_load_golden_from_corpus() {
    // Path to the Qwen golden corpus
    let corpus_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("../hf-ground-truth-corpus/oracle/qwen2.5-coder-1.5b/v1"));

    let Some(corpus_path) = corpus_path else {
        eprintln!("Skipping integration test: corpus path not found");
        return;
    };

    if !corpus_path.exists() {
        eprintln!("Skipping integration test: corpus not generated yet");
        return;
    }

    // Load a known golden file
    let prompt = "def fibonacci(n):";
    let hash = hash_prompt(prompt);
    let safetensors_path = corpus_path.join(format!("{hash}.safetensors"));
    let json_path = corpus_path.join(format!("{hash}.json"));

    assert!(
        safetensors_path.exists(),
        "SafeTensors file not found: {safetensors_path:?}"
    );
    assert!(json_path.exists(), "JSON metadata not found: {json_path:?}");

    // Load and verify the golden output
    let result = HfParityOracle::load_golden_from_path(&safetensors_path, prompt, &hash);
    assert!(result.is_ok(), "Failed to load golden: {result:?}");

    let golden = result.expect("already checked");
    assert_eq!(golden.input_hash, hash);
    assert_eq!(golden.prompt, prompt);
    assert!(!golden.logits.is_empty(), "Logits should not be empty");
    assert_eq!(
        golden.shape.len(),
        2,
        "Logits should be 2D [seq_len, vocab]"
    );
    assert_eq!(
        golden.shape[1], 151936,
        "Qwen2.5 vocab size should be 151936"
    );
}

#[test]
fn test_oracle_verify_with_golden_corpus() {
    // Path to the Qwen golden corpus
    let corpus_base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("../hf-ground-truth-corpus/oracle"));

    let Some(corpus_base) = corpus_base else {
        eprintln!("Skipping integration test: corpus path not found");
        return;
    };

    let corpus_path = corpus_base.join("qwen2.5-coder-1.5b/v1");
    if !corpus_path.exists() {
        eprintln!("Skipping integration test: corpus not generated yet");
        return;
    }

    // Create oracle pointing to the corpus
    let oracle = HfParityOracle::new(&corpus_base, "qwen2.5-coder-1.5b/v1");

    // Load a golden output
    let prompt = "def fibonacci(n):";
    let golden_result = oracle.load_golden(prompt);
    assert!(
        golden_result.is_ok(),
        "Failed to load golden: {golden_result:?}"
    );

    let golden = golden_result.expect("already checked");

    // Verify that comparing golden with itself passes
    let verify_result = oracle.tensors_close(&golden.logits, &golden.logits);
    assert!(
        verify_result.is_ok(),
        "Golden should match itself: {verify_result:?}"
    );
}
