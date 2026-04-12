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

fn write_minimal_tokenizer(dir: &Path) {
    let path = dir.join("tokenizer.json");
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(b"{}").unwrap();
    let tc_path = dir.join("tokenizer_config.json");
    let mut f2 = std::fs::File::create(tc_path).unwrap();
    f2.write_all(br#"{"eos_token":"<|endoftext|>"}"#).unwrap();
}

fn write_minimal_safetensors(dir: &Path, tensors: &[(&str, &[usize])]) {
    use std::collections::HashMap;
    let path = dir.join("model.safetensors");

    let mut header_map: HashMap<&str, serde_json::Value> = HashMap::new();
    let mut offset: u64 = 0;
    for &(name, shape) in tensors {
        let num_elements: usize = shape.iter().product();
        let byte_size = num_elements * 4;
        let tensor_info = serde_json::json!({
            "dtype": "F32",
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

#[test]
fn test_check_valid_config() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(
        dir.path(),
        &[
            ("model.embed_tokens.weight", &[151_936, 896]),
            ("lm_head.weight", &[151_936, 896]),
        ],
    );
    write_minimal_tokenizer(dir.path());

    let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
    let result = run_dimensional_check(dir.path(), &playbook);

    assert!(
        result.passed,
        "all checks should pass: {:#?}",
        result.checks
    );
    assert!(result.duration_ms < 5000, "should complete quickly");
    assert!(
        result.checks.len() >= 12,
        "expected at least 12 checks, got {}",
        result.checks.len()
    );
}

#[test]
fn test_check_mismatched_hidden_size() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 512,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);

    let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
    let result = run_dimensional_check(dir.path(), &playbook);

    assert!(!result.passed);
    let hidden_check = result
        .checks
        .iter()
        .find(|c| c.name == "hidden_size")
        .unwrap();
    assert!(!hidden_check.passed);
    assert_eq!(hidden_check.expected, "896");
    assert_eq!(hidden_check.actual, "512");
}

#[test]
fn test_check_missing_config() {
    let dir = TempDir::new().unwrap();

    let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
    let result = run_dimensional_check(dir.path(), &playbook);

    assert!(!result.passed);
    let config_check = result
        .checks
        .iter()
        .find(|c| c.name == "config_parse")
        .unwrap();
    assert!(!config_check.passed);
}

#[test]
fn test_check_safetensors_header() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(
        dir.path(),
        &[
            ("model.embed_tokens.weight", &[151_936, 896]),
            ("lm_head.weight", &[151_936, 896]),
            ("model.layers.0.self_attn.q_proj.weight", &[896, 896]),
        ],
    );
    write_minimal_tokenizer(dir.path());

    let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
    let result = run_dimensional_check(dir.path(), &playbook);

    assert!(
        result.passed,
        "all checks should pass: {:#?}",
        result.checks
    );
    let header_check = result
        .checks
        .iter()
        .find(|c| c.name == "safetensors_header")
        .unwrap();
    assert!(header_check.passed);
    assert_eq!(header_check.actual, "3 tensor(s)");
}

#[test]
fn test_check_wrong_tensor_shape() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(
        dir.path(),
        &[("model.embed_tokens.weight", &[151_936, 512])],
    );

    let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
    let result = run_dimensional_check(dir.path(), &playbook);

    assert!(!result.passed);
    let tensor_check = result
        .checks
        .iter()
        .find(|c| c.name == "tensor_embed_tokens")
        .unwrap();
    assert!(!tensor_check.passed);
}

#[test]
fn test_check_no_safetensors_files() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896,
        "num_hidden_layers": 24
    });
    write_config_json(dir.path(), &config);

    let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
    let result = run_dimensional_check(dir.path(), &playbook);

    assert!(!result.passed);
    let st_check = result
        .checks
        .iter()
        .find(|c| c.name == "safetensors_found")
        .unwrap();
    assert!(!st_check.passed);
}

#[test]
fn test_check_no_expected_params() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896,
        "num_hidden_layers": 24
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(dir.path(), &[("some.tensor", &[10, 20])]);
    write_minimal_tokenizer(dir.path());

    let yaml = r#"
name: test-playbook
version: "1.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  formats: [safetensors]
  prompts:
    - "hello"
"#;
    let playbook = Playbook::from_yaml(yaml).expect("valid test playbook");
    let result = run_dimensional_check(dir.path(), &playbook);

    assert!(
        result.passed,
        "should pass with no expected params: {:#?}",
        result.checks
    );
}

#[test]
fn test_result_model_id() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({"hidden_size": 896});
    write_config_json(dir.path(), &config);

    let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
    let result = run_dimensional_check(dir.path(), &playbook);
    assert_eq!(result.model_id, "Qwen/Qwen2.5-Coder-0.5B-Instruct");
}

/// GH-266: Mamba SSM has no attention heads — dim-smoke should skip head checks
#[test]
fn test_mamba_ssm_no_attention_heads() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "model_type": "mamba",
        "hidden_size": 1024,
        "num_hidden_layers": 48,
        "vocab_size": 50280,
        "state_size": 16,
        "conv_kernel": 4,
        "expand": 2
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(
        dir.path(),
        &[
            ("backbone.embedding.weight", &[50280, 1024]),
            ("lm_head.weight", &[50280, 1024]),
        ],
    );
    write_minimal_tokenizer(dir.path());

    let yaml = r#"
name: mamba-370m-dim-smoke
version: "1.0"
model:
  hf_repo: "state-spaces/mamba-370m-hf"
  expected_hidden_dim: 1024
  expected_num_layers: 48
  expected_vocab_size: 50280
test_matrix:
  modalities: [run]
  backends: [cpu]
  formats: [safetensors]
  prompts:
    - "hello"
"#;
    let playbook = Playbook::from_yaml(yaml).expect("valid mamba playbook");
    let result = run_dimensional_check(dir.path(), &playbook);

    // Should pass — no num_heads expected, none present
    assert!(
        result.passed,
        "Mamba SSM should pass without attention head checks: {:#?}",
        result.checks
    );
    // Verify no num_heads check was emitted
    let head_check = result.checks.iter().find(|c| c.name == "num_heads");
    assert!(
        head_check.is_none(),
        "Mamba should not have a num_heads check"
    );
}

/// GH-266: OpenELM has array-valued num_query_heads — should return None, skip check
#[test]
fn test_openelm_array_heads_skipped() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "model_type": "openelm",
        "model_dim": 1280,
        "num_transformer_layers": 16,
        "vocab_size": 32000,
        "num_query_heads": [12, 12, 12, 12, 12, 16, 16, 16, 16, 16, 16, 16, 20, 20, 20, 20],
        "num_kv_heads": [3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5]
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(
        dir.path(),
        &[
            ("transformer.token_embeddings.weight", &[32000, 1280]),
            ("lm_head.weight", &[32000, 1280]),
        ],
    );
    write_minimal_tokenizer(dir.path());

    let yaml = r#"
name: openelm-270m-dim-smoke
version: "1.0"
model:
  hf_repo: "apple/OpenELM-270M-Instruct"
  expected_hidden_dim: 1280
  expected_num_layers: 16
  expected_vocab_size: 32000
test_matrix:
  modalities: [run]
  backends: [cpu]
  formats: [safetensors]
  prompts:
    - "hello"
"#;
    let playbook = Playbook::from_yaml(yaml).expect("valid openelm playbook");
    let result = run_dimensional_check(dir.path(), &playbook);

    // Should pass — array-valued heads return None from get_usize, no expected set
    assert!(
        result.passed,
        "OpenELM should pass without scalar head checks: {:#?}",
        result.checks
    );
}

/// Verify non-2D tensor is flagged as failure
#[test]
fn test_check_non_2d_tensor() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    // Write a 3D embed_tokens tensor — should fail check
    write_minimal_safetensors(
        dir.path(),
        &[("model.embed_tokens.weight", &[2, 151_936, 896])],
    );

    let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
    let result = run_dimensional_check(dir.path(), &playbook);

    assert!(!result.passed);
    let tensor_check = result
        .checks
        .iter()
        .find(|c| c.name == "tensor_embed_tokens")
        .unwrap();
    assert!(!tensor_check.passed);
    assert!(tensor_check.actual.contains("3D"));
}

/// Verify vocab_size (dim0) mismatch is caught
#[test]
fn test_check_vocab_dim0_mismatch() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    // Wrong dim0 (vocab_size 32000 instead of 151936) but correct dim1
    write_minimal_safetensors(dir.path(), &[("model.embed_tokens.weight", &[32000, 896])]);

    let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
    let result = run_dimensional_check(dir.path(), &playbook);

    assert!(!result.passed);
    let tensor_check = result
        .checks
        .iter()
        .find(|c| c.name == "tensor_embed_tokens")
        .unwrap();
    assert!(!tensor_check.passed);
}

/// Verify corrupted safetensors header results in parse error check
#[test]
fn test_check_corrupted_safetensors_header() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);

    // Write a corrupt safetensors file (invalid header)
    let path = dir.path().join("model.safetensors");
    let mut f = std::fs::File::create(path).unwrap();
    // Write invalid header length pointing to garbage
    f.write_all(&999_999_u64.to_le_bytes()).unwrap();
    f.write_all(b"not valid json").unwrap();

    let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
    let result = run_dimensional_check(dir.path(), &playbook);

    assert!(!result.passed);
    let header_check = result
        .checks
        .iter()
        .find(|c| c.name == "safetensors_header")
        .unwrap();
    assert!(!header_check.passed);
    assert_eq!(header_check.actual, "parse error");
}

/// Verify dim1 (hidden_size) mismatch is caught even when vocab_size matches
#[test]
fn test_check_hidden_dim1_mismatch() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    // Correct dim0 (vocab_size) but wrong dim1 (hidden_size 512 instead of 896)
    write_minimal_safetensors(
        dir.path(),
        &[("model.embed_tokens.weight", &[151_936, 512])],
    );

    let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
    let result = run_dimensional_check(dir.path(), &playbook);

    assert!(!result.passed);
    let tensor_check = result
        .checks
        .iter()
        .find(|c| c.name == "tensor_embed_tokens")
        .unwrap();
    assert!(!tensor_check.passed);
}

/// Verify tensor check emits NO evidence when no vocab/hidden expectations are set.
/// Popperian: untested hypotheses must not be marked as corroborated.
#[test]
fn test_check_tensor_no_expected_dims() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "num_hidden_layers": 24
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(dir.path(), &[("model.embed_tokens.weight", &[1000, 500])]);
    write_minimal_tokenizer(dir.path());

    // Use a playbook with NO expected dim/vocab/heads
    let yaml = r#"
name: test-playbook
version: "1.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  formats: [safetensors]
  prompts:
    - "hello"
"#;
    let playbook = Playbook::from_yaml(yaml).expect("valid test playbook");
    let result = run_dimensional_check(dir.path(), &playbook);

    // Should pass — no hypothesis to test
    assert!(
        result.passed,
        "should pass with no expected dims: {:#?}",
        result.checks
    );
    // Popperian: no tensor_embed_tokens check should exist — no dims to validate
    let tensor_check = result
        .checks
        .iter()
        .find(|c| c.name == "tensor_embed_tokens");
    assert!(
        tensor_check.is_none(),
        "no tensor check should be emitted when config has no vocab/hidden dims"
    );
}

/// GH-270: RWKV7 has explicit null num_heads — dim-smoke should skip head checks
#[test]
fn test_rwkv7_null_num_heads() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "model_type": "rwkv7",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": null,
        "vocab_size": 65536
    });
    write_config_json(dir.path(), &config);
    write_minimal_safetensors(
        dir.path(),
        &[
            ("rwkv.embeddings.weight", &[65536, 768]),
            ("head.weight", &[65536, 768]),
        ],
    );
    write_minimal_tokenizer(dir.path());

    let yaml = r#"
name: rwkv7-dim-smoke
version: "1.0"
model:
  hf_repo: "RWKV/rwkv-7-world-0.1b"
  expected_hidden_dim: 768
  expected_num_layers: 12
  expected_vocab_size: 65536
test_matrix:
  modalities: [run]
  backends: [cpu]
  formats: [safetensors]
  prompts:
    - "hello"
"#;
    let playbook = Playbook::from_yaml(yaml).expect("valid rwkv7 playbook");
    let result = run_dimensional_check(dir.path(), &playbook);

    assert!(
        result.passed,
        "RWKV7 should pass with null num_heads: {:#?}",
        result.checks
    );
}

/// Verify lm_head.weight check triggers when present but with wrong dimensions
#[test]
fn test_check_lm_head_dim_mismatch() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    // embed_tokens matches, but lm_head has wrong dim0 (vocab_size)
    write_minimal_safetensors(
        dir.path(),
        &[
            ("model.embed_tokens.weight", &[151_936, 896]),
            ("lm_head.weight", &[50_000, 896]),
        ],
    );

    let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
    let result = run_dimensional_check(dir.path(), &playbook);
    assert!(!result.passed);
    let lm_check = result
        .checks
        .iter()
        .find(|c| c.name == "tensor_lm_head")
        .unwrap();
    assert!(!lm_check.passed);
}

/// Verify check_safetensors reports when no safetensors files exist
#[test]
fn test_check_safetensors_zero_files() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    // No .safetensors files at all

    let playbook = make_minimal_playbook("Qwen/Qwen2.5-Coder-0.5B-Instruct");
    let result = run_dimensional_check(dir.path(), &playbook);
    assert!(!result.passed);
    let st_found = result
        .checks
        .iter()
        .find(|c| c.name == "safetensors_found")
        .unwrap();
    assert!(!st_found.passed);
    assert!(st_found.actual.contains("0 file"));
}

/// Verify only vocab_size mismatch on dim0 when hidden_size is None
#[test]
fn test_check_tensor_only_vocab_expected() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "num_hidden_layers": 24,
        "vocab_size": 151_936
    });
    write_config_json(dir.path(), &config);
    // dim0 doesn't match vocab_size, but no hidden_size to check
    write_minimal_safetensors(dir.path(), &[("model.embed_tokens.weight", &[50_000, 512])]);

    let yaml = r#"
name: test-playbook
version: "1.0"
model:
  hf_repo: "test/model"
  expected_vocab_size: 151936
test_matrix:
  modalities: [run]
  backends: [cpu]
  formats: [safetensors]
  prompts:
    - "hello"
"#;
    let playbook = Playbook::from_yaml(yaml).expect("valid playbook");
    let result = run_dimensional_check(dir.path(), &playbook);
    assert!(
        !result.passed,
        "dim0 mismatch should fail: {:#?}",
        result.checks
    );
}
