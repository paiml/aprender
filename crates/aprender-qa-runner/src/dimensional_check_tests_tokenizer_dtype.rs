use super::*;
use crate::playbook::Playbook;
use std::io::Write;
use tempfile::TempDir;

fn make_minimal_playbook(hf_repo: &str) -> Playbook {
    let yaml = format!(
        r#"
name: test-playbook
version: "1.0"
model:
  hf_repo: "{hf_repo}"
  expected_hidden_dim: 896
  expected_num_layers: 24
  expected_num_heads: 14
  expected_num_kv_heads: 2
  expected_vocab_size: 151936
test_matrix:
  modalities: [run]
  backends: [cpu]
  formats: [safetensors]
  prompts:
    - "hello"
"#
    );
    Playbook::from_yaml(&yaml).expect("valid test playbook")
}

fn write_config_json(dir: &Path, config: &serde_json::Value) {
    let path = dir.join("config.json");
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(serde_json::to_string(config).unwrap().as_bytes())
        .unwrap();
}

fn write_file(dir: &Path, name: &str, content: &str) {
    let path = dir.join(name);
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(content.as_bytes()).unwrap();
}

fn write_safetensors_with_dtype(dir: &Path, tensors: &[(&str, &[usize], &str)]) {
    let path = dir.join("model.safetensors");

    let mut header_map: std::collections::HashMap<&str, serde_json::Value> =
        std::collections::HashMap::new();
    let mut offset: u64 = 0;
    for &(name, shape, dtype) in tensors {
        let num_elements: usize = shape.iter().product();
        let byte_size = num_elements * 4;
        let tensor_info = serde_json::json!({
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + byte_size as u64]
        });
        header_map.insert(name, tensor_info);
        offset += byte_size as u64;
    }

    let header_json = serde_json::to_string(&header_map).unwrap();
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&header_len.to_le_bytes()).unwrap();
    f.write_all(header_bytes).unwrap();
    f.write_all(&vec![0u8; offset as usize]).unwrap();
}

fn write_minimal_safetensors(dir: &Path, tensors: &[(&str, &[usize])]) {
    let with_dtype: Vec<(&str, &[usize], &str)> =
        tensors.iter().map(|&(n, s)| (n, s, "F32")).collect();
    write_safetensors_with_dtype(dir, &with_dtype);
}

// ── Tokenizer checks ──────────────────────────────────────────────

#[test]
fn test_tokenizer_exists_json() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(
        dir.path(),
        &[
            ("model.embed_tokens.weight", &[151_936, 896]),
            ("lm_head.weight", &[151_936, 896]),
        ],
    );
    write_file(dir.path(), "tokenizer.json", r#"{"version":"1.0"}"#);

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "tokenizer_exists")
        .unwrap();
    assert!(
        check.passed,
        "tokenizer.json should satisfy tokenizer_exists"
    );
    assert!(check.actual.contains("tokenizer.json"));
}

#[test]
fn test_tokenizer_exists_model() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(
        dir.path(),
        &[
            ("model.embed_tokens.weight", &[151_936, 896]),
            ("lm_head.weight", &[151_936, 896]),
        ],
    );
    // SentencePiece tokenizer.model (binary, but we just need existence)
    write_file(dir.path(), "tokenizer.model", "sentencepiece binary");

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "tokenizer_exists")
        .unwrap();
    assert!(
        check.passed,
        "tokenizer.model should satisfy tokenizer_exists"
    );
    assert!(check.actual.contains("tokenizer.model"));
}

#[test]
fn test_tokenizer_missing() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(
        dir.path(),
        &[
            ("model.embed_tokens.weight", &[151_936, 896]),
            ("lm_head.weight", &[151_936, 896]),
        ],
    );
    // No tokenizer files at all

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "tokenizer_exists")
        .unwrap();
    assert!(!check.passed, "missing tokenizer should fail");
}

#[test]
fn test_tokenizer_config_valid() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(
        dir.path(),
        &[("model.embed_tokens.weight", &[151_936, 896])],
    );
    write_file(dir.path(), "tokenizer.json", "{}");
    write_file(
        dir.path(),
        "tokenizer_config.json",
        r#"{"eos_token":"<|endoftext|>"}"#,
    );

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "tokenizer_config_valid")
        .unwrap();
    assert!(check.passed, "valid JSON tokenizer_config should pass");
}

#[test]
fn test_tokenizer_config_invalid_json() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(
        dir.path(),
        &[("model.embed_tokens.weight", &[151_936, 896])],
    );
    write_file(dir.path(), "tokenizer.json", "{}");
    write_file(dir.path(), "tokenizer_config.json", "{broken json");

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "tokenizer_config_valid")
        .unwrap();
    assert!(
        !check.passed,
        "broken JSON should fail tokenizer_config_valid"
    );
}

#[test]
fn test_eos_token_string() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(
        dir.path(),
        &[("model.embed_tokens.weight", &[151_936, 896])],
    );
    write_file(dir.path(), "tokenizer.json", "{}");
    write_file(
        dir.path(),
        "tokenizer_config.json",
        r#"{"eos_token":"<|endoftext|>"}"#,
    );

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "eos_token_valid")
        .unwrap();
    assert!(check.passed, "string eos_token should pass");
    assert!(check.actual.contains("<|endoftext|>"));
}

#[test]
fn test_eos_token_object() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(
        dir.path(),
        &[("model.embed_tokens.weight", &[151_936, 896])],
    );
    write_file(dir.path(), "tokenizer.json", "{}");
    write_file(
        dir.path(),
        "tokenizer_config.json",
        r#"{"eos_token":{"content":"<|im_end|>","lstrip":false,"single_word":false}}"#,
    );

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "eos_token_valid")
        .unwrap();
    assert!(check.passed, "object-format eos_token (Qwen) should pass");
}

#[test]
fn test_eos_token_missing() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(
        dir.path(),
        &[("model.embed_tokens.weight", &[151_936, 896])],
    );
    write_file(dir.path(), "tokenizer.json", "{}");
    write_file(
        dir.path(),
        "tokenizer_config.json",
        r#"{"bos_token":"<s>"}"#,
    );

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "eos_token_valid")
        .unwrap();
    assert!(!check.passed, "missing eos_token should fail");
}

#[test]
fn test_bos_token_absent_ok() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(
        dir.path(),
        &[("model.embed_tokens.weight", &[151_936, 896])],
    );
    write_file(dir.path(), "tokenizer.json", "{}");
    write_file(
        dir.path(),
        "tokenizer_config.json",
        r#"{"eos_token":"<|endoftext|>"}"#,
    );

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    // bos_token absent → no bos_token_valid check emitted
    let bos_check = result.checks.iter().find(|c| c.name == "bos_token_valid");
    assert!(
        bos_check.is_none(),
        "absent bos_token should not emit check"
    );
}

// ── Dtype checks ──────────────────────────────────────────────────

#[test]
fn test_dtype_supported_f32() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    write_safetensors_with_dtype(
        dir.path(),
        &[
            ("model.embed_tokens.weight", &[151_936, 896], "F32"),
            ("lm_head.weight", &[151_936, 896], "F32"),
        ],
    );

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "dtype_supported")
        .unwrap();
    assert!(check.passed, "F32 should be supported");
}

#[test]
fn test_dtype_supported_bf16() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    write_safetensors_with_dtype(
        dir.path(),
        &[
            ("model.embed_tokens.weight", &[151_936, 896], "BF16"),
            ("lm_head.weight", &[151_936, 896], "BF16"),
        ],
    );

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "dtype_supported")
        .unwrap();
    assert!(check.passed, "BF16 should be supported");
}

#[test]
fn test_dtype_unsupported() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    write_safetensors_with_dtype(
        dir.path(),
        &[("model.embed_tokens.weight", &[151_936, 896], "Q4_0")],
    );

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "dtype_supported")
        .unwrap();
    assert!(!check.passed, "Q4_0 is not a valid SafeTensors dtype");
    assert!(check.actual.contains("Q4_0"));
}

#[test]
fn test_dtype_consistent_all_same() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    write_safetensors_with_dtype(
        dir.path(),
        &[
            ("model.embed_tokens.weight", &[151_936, 896], "BF16"),
            ("lm_head.weight", &[151_936, 896], "BF16"),
            (
                "model.layers.0.self_attn.q_proj.weight",
                &[896, 896],
                "BF16",
            ),
            // 1D bias — different dtype is OK (1D excluded from consistency check)
            ("model.layers.0.input_layernorm.weight", &[896], "F32"),
        ],
    );

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "dtype_consistent")
        .unwrap();
    assert!(
        check.passed,
        "all 2D tensors BF16 → consistent (1D F32 excluded)"
    );
}

#[test]
fn test_dtype_consistent_mixed_interior() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    // Mix F32 and BF16 in interior weight tensors (not embed/lm_head)
    write_safetensors_with_dtype(
        dir.path(),
        &[
            ("model.layers.0.self_attn.q_proj.weight", &[896, 896], "F32"),
            (
                "model.layers.0.self_attn.k_proj.weight",
                &[128, 896],
                "BF16",
            ),
        ],
    );

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "dtype_consistent")
        .unwrap();
    assert!(
        !check.passed,
        "mixed F32+BF16 in interior weight tensors should fail"
    );
}

#[test]
fn test_dtype_config_match_bfloat16() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936,
        "torch_dtype": "bfloat16"
    });
    write_config_json(dir.path(), &config);
    write_safetensors_with_dtype(
        dir.path(),
        &[
            ("model.embed_tokens.weight", &[151_936, 896], "BF16"),
            ("lm_head.weight", &[151_936, 896], "BF16"),
        ],
    );

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "dtype_config_match")
        .unwrap();
    assert!(check.passed, "bfloat16 config + BF16 tensors should match");
}

#[test]
fn test_dtype_config_match_mismatch() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936,
        "torch_dtype": "float16"
    });
    write_config_json(dir.path(), &config);
    write_safetensors_with_dtype(
        dir.path(),
        &[
            ("model.embed_tokens.weight", &[151_936, 896], "BF16"),
            ("lm_head.weight", &[151_936, 896], "BF16"),
        ],
    );

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "dtype_config_match")
        .unwrap();
    assert!(
        !check.passed,
        "float16 config + BF16 tensors should mismatch"
    );
}

#[test]
fn test_dtype_config_no_torch_dtype() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    write_safetensors_with_dtype(
        dir.path(),
        &[("model.embed_tokens.weight", &[151_936, 896], "BF16")],
    );

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "dtype_config_match");
    assert!(
        check.is_none(),
        "no torch_dtype → no dtype_config_match check"
    );
}

// ── End-to-end integration ────────────────────────────────────────

#[test]
fn test_full_dim_check_with_tokenizer_and_dtype() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "vocab_size": 151_936,
        "torch_dtype": "bfloat16"
    });
    write_config_json(dir.path(), &config);
    write_safetensors_with_dtype(
        dir.path(),
        &[
            ("model.embed_tokens.weight", &[151_936, 896], "BF16"),
            (
                "model.layers.0.self_attn.q_proj.weight",
                &[896, 896],
                "BF16",
            ),
            ("lm_head.weight", &[151_936, 896], "BF16"),
        ],
    );
    write_file(dir.path(), "tokenizer.json", r#"{"version":"1.0"}"#);
    write_file(
        dir.path(),
        "tokenizer_config.json",
        r#"{"eos_token":"<|endoftext|>","bos_token":"<|startoftext|>"}"#,
    );

    let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
    let result = run_dimensional_check(dir.path(), &playbook);

    assert!(
        result.passed,
        "full dim check with tokenizer + dtype should pass: {:#?}",
        result.checks
    );

    // Verify all new checks are present
    let check_names: Vec<&str> = result.checks.iter().map(|c| c.name.as_str()).collect();
    assert!(
        check_names.contains(&"tokenizer_exists"),
        "missing tokenizer_exists"
    );
    assert!(
        check_names.contains(&"tokenizer_config_valid"),
        "missing tokenizer_config_valid"
    );
    assert!(
        check_names.contains(&"eos_token_valid"),
        "missing eos_token_valid"
    );
    assert!(
        check_names.contains(&"bos_token_valid"),
        "missing bos_token_valid"
    );
    assert!(
        check_names.contains(&"dtype_supported"),
        "missing dtype_supported"
    );
    assert!(
        check_names.contains(&"dtype_consistent"),
        "missing dtype_consistent"
    );
    assert!(
        check_names.contains(&"dtype_config_match"),
        "missing dtype_config_match"
    );
}

// ── Audit fix B1: Corrupt tokenizer file content ──────────────────

#[test]
fn test_tokenizer_json_corrupt() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(
        dir.path(),
        &[("model.embed_tokens.weight", &[151_936, 896])],
    );
    // tokenizer.json exists but is not valid JSON
    write_file(dir.path(), "tokenizer.json", "{corrupt garbage");

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "tokenizer_exists")
        .unwrap();
    assert!(
        !check.passed,
        "corrupt tokenizer.json should fail (invalid JSON)"
    );
    assert!(check.actual.contains("invalid JSON"));
}

#[test]
fn test_tokenizer_model_empty() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(
        dir.path(),
        &[("model.embed_tokens.weight", &[151_936, 896])],
    );
    // tokenizer.model exists but is empty (0 bytes)
    write_file(dir.path(), "tokenizer.model", "");

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "tokenizer_exists")
        .unwrap();
    assert!(!check.passed, "empty tokenizer.model should fail");
}

// ── Audit fix B3: eos_token_id fallback from config.json ──────────

#[test]
fn test_eos_token_id_fallback_from_config() {
    let dir = TempDir::new().unwrap();
    // config.json has eos_token_id but no eos_token string
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936,
        "eos_token_id": 50256
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(
        dir.path(),
        &[("model.embed_tokens.weight", &[151_936, 896])],
    );
    write_file(dir.path(), "tokenizer.json", "{}");
    // tokenizer_config.json has no eos_token
    write_file(
        dir.path(),
        "tokenizer_config.json",
        r#"{"bos_token":"<s>"}"#,
    );

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "eos_token_valid")
        .unwrap();
    assert!(
        check.passed,
        "eos_token_id=50256 in config.json should satisfy fallback"
    );
    assert!(check.actual.contains("50256"));
}

#[test]
fn test_eos_token_id_fallback_no_tokenizer_config() {
    let dir = TempDir::new().unwrap();
    // config.json has eos_token_id, no tokenizer_config.json at all
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936,
        "eos_token_id": 2
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(
        dir.path(),
        &[("model.embed_tokens.weight", &[151_936, 896])],
    );
    write_file(dir.path(), "tokenizer.json", "{}");
    // No tokenizer_config.json

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "eos_token_valid")
        .unwrap();
    assert!(
        check.passed,
        "eos_token_id in config.json should work without tokenizer_config"
    );
}

#[test]
fn test_eos_token_id_fallback_missing_everywhere() {
    let dir = TempDir::new().unwrap();
    // No eos_token in tokenizer_config, no eos_token_id in config.json
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(
        dir.path(),
        &[("model.embed_tokens.weight", &[151_936, 896])],
    );
    write_file(dir.path(), "tokenizer.json", "{}");
    write_file(
        dir.path(),
        "tokenizer_config.json",
        r#"{"bos_token":"<s>"}"#,
    );

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "eos_token_valid")
        .unwrap();
    assert!(
        !check.passed,
        "no eos_token or eos_token_id anywhere should fail"
    );
}

// ── Audit fix A3: Embedding layers exempt from dtype consistency ──

#[test]
fn test_dtype_consistent_embed_f32_weights_bf16() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    // F32 embeddings + BF16 interior weights — legitimate pattern (Llama-70B)
    write_safetensors_with_dtype(
        dir.path(),
        &[
            ("model.embed_tokens.weight", &[151_936, 896], "F32"),
            ("lm_head.weight", &[151_936, 896], "F32"),
            (
                "model.layers.0.self_attn.q_proj.weight",
                &[896, 896],
                "BF16",
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                &[128, 896],
                "BF16",
            ),
            ("model.layers.0.mlp.gate_proj.weight", &[4864, 896], "BF16"),
        ],
    );

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "dtype_consistent")
        .unwrap();
    assert!(
        check.passed,
        "F32 embed + BF16 interior weights should pass (embed layers excluded): {}",
        check.actual
    );
}

#[test]
fn test_dtype_consistent_only_embed_tensors() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896, "num_hidden_layers": 24, "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    // Only embedding tensors, no interior weights — consistency check still emitted (passes vacuously)
    write_safetensors_with_dtype(
        dir.path(),
        &[
            ("model.embed_tokens.weight", &[151_936, 896], "F32"),
            ("lm_head.weight", &[151_936, 896], "BF16"),
        ],
    );

    let playbook = make_minimal_playbook("test/model");
    let result = run_dimensional_check(dir.path(), &playbook);
    let check = result
        .checks
        .iter()
        .find(|c| c.name == "dtype_consistent")
        .expect("dtype_consistent check should always be emitted");
    assert!(
        check.passed,
        "no interior weights → passes (all embeddings)"
    );
    assert!(
        check.actual.contains("no interior weight tensors"),
        "actual should note absence: {}",
        check.actual
    );
}
