use super::*;

fn test_contract() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../contracts/softmax-kernel-v1.yaml")
}

fn contracts_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../contracts")
}

#[test]
fn dispatch_validate() {
    let result = dispatch(Commands::Validate {
        contract: test_contract(),
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_scaffold() {
    let result = dispatch(Commands::Scaffold {
        contract: test_contract(),
        r#trait: false,
        output: None,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_kani() {
    let result = dispatch(Commands::Kani {
        contract: test_contract(),
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_status() {
    let result = dispatch(Commands::Status {
        contract: test_contract(),
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_diff() {
    let c = test_contract();
    let result = dispatch(Commands::Diff {
        old: c.clone(),
        new: c,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_lean() {
    let result = dispatch(Commands::Lean {
        contract: test_contract(),
        output_dir: None,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_lean_status() {
    let result = dispatch(Commands::LeanStatus {
        path: test_contract(),
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_lean_status_directory() {
    let result = dispatch(Commands::LeanStatus {
        path: contracts_dir(),
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_proof_status() {
    let result = dispatch(Commands::ProofStatus {
        path: test_contract(),
        binding: None,
        format: "text".to_string(),
        table: false,
        kind: None,
        verify_bindings: None,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_proof_status_json() {
    let result = dispatch(Commands::ProofStatus {
        path: test_contract(),
        binding: None,
        format: "json".to_string(),
        table: false,
        kind: None,
        verify_bindings: None,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_proof_status_directory() {
    let result = dispatch(Commands::ProofStatus {
        path: contracts_dir(),
        binding: None,
        format: "text".to_string(),
        table: false,
        kind: None,
        verify_bindings: None,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_proof_status_with_binding() {
    let binding =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../contracts/aprender/binding.yaml");
    let result = dispatch(Commands::ProofStatus {
        path: contracts_dir(),
        binding: Some(binding),
        format: "json".to_string(),
        table: false,
        kind: None,
        verify_bindings: None,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_score_single() {
    let result = dispatch(Commands::Score {
        path: test_contract(),
        binding: None,
        format: "text".to_string(),
        min_score: None,
        summary: false,
        top_gaps: 5,
        weights: None,
        exit_code: false,
        pvscore: false,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_score_directory() {
    let result = dispatch(Commands::Score {
        path: contracts_dir(),
        binding: None,
        format: "json".to_string(),
        min_score: None,
        summary: false,
        top_gaps: 5,
        weights: None,
        exit_code: false,
        pvscore: false,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_score_min_threshold_fails() {
    let result = dispatch(Commands::Score {
        path: test_contract(),
        binding: None,
        format: "text".to_string(),
        min_score: Some(0.99),
        summary: false,
        top_gaps: 5,
        weights: None,
        exit_code: false,
        pvscore: false,
    });
    assert!(result.is_err());
}

#[test]
fn dispatch_score_custom_weights() {
    let result = dispatch(Commands::Score {
        path: test_contract(),
        binding: None,
        format: "text".to_string(),
        min_score: None,
        summary: false,
        top_gaps: 5,
        weights: Some(
            r#"{"spec_depth":0.1,"falsification":0.3,"kani":0.3,"lean":0.1,"binding":0.2}"#
                .to_string(),
        ),
        exit_code: false,
        pvscore: false,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_score_markdown() {
    let result = dispatch(Commands::Score {
        path: test_contract(),
        binding: None,
        format: "markdown".to_string(),
        min_score: None,
        summary: false,
        top_gaps: 0,
        weights: None,
        exit_code: false,
        pvscore: false,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_score_directory_markdown() {
    let result = dispatch(Commands::Score {
        path: contracts_dir(),
        binding: None,
        format: "markdown".to_string(),
        min_score: None,
        summary: false,
        top_gaps: 3,
        weights: None,
        exit_code: false,
        pvscore: false,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_score_summary() {
    let result = dispatch(Commands::Score {
        path: contracts_dir(),
        binding: None,
        format: "text".to_string(),
        min_score: None,
        summary: true,
        top_gaps: 0,
        weights: None,
        exit_code: false,
        pvscore: false,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_score_summary_json() {
    let result = dispatch(Commands::Score {
        path: contracts_dir(),
        binding: None,
        format: "json".to_string(),
        min_score: None,
        summary: true,
        top_gaps: 0,
        weights: None,
        exit_code: false,
        pvscore: false,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_score_summary_markdown() {
    let result = dispatch(Commands::Score {
        path: contracts_dir(),
        binding: None,
        format: "markdown".to_string(),
        min_score: None,
        summary: true,
        top_gaps: 0,
        weights: None,
        exit_code: false,
        pvscore: false,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_score_with_binding() {
    let binding =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../contracts/aprender/binding.yaml");
    let result = dispatch(Commands::Score {
        path: contracts_dir(),
        binding: Some(binding),
        format: "text".to_string(),
        min_score: None,
        summary: false,
        top_gaps: 3,
        weights: None,
        exit_code: false,
        pvscore: false,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_score_with_binding_json() {
    let binding =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../contracts/aprender/binding.yaml");
    let result = dispatch(Commands::Score {
        path: contracts_dir(),
        binding: Some(binding),
        format: "json".to_string(),
        min_score: None,
        summary: false,
        top_gaps: 0,
        weights: None,
        exit_code: false,
        pvscore: false,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_score_with_binding_markdown() {
    let binding =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../contracts/aprender/binding.yaml");
    let result = dispatch(Commands::Score {
        path: contracts_dir(),
        binding: Some(binding),
        format: "markdown".to_string(),
        min_score: None,
        summary: false,
        top_gaps: 0,
        weights: None,
        exit_code: false,
        pvscore: false,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_score_directory_threshold_fails() {
    let result = dispatch(Commands::Score {
        path: contracts_dir(),
        binding: None,
        format: "text".to_string(),
        min_score: Some(0.99),
        summary: false,
        top_gaps: 0,
        weights: None,
        exit_code: false,
        pvscore: false,
    });
    assert!(result.is_err());
}

#[test]
fn dispatch_book() {
    let dir = tempfile::tempdir().expect("tempdir");
    let result = dispatch(Commands::Book {
        contract_dir: contracts_dir(),
        output: dir.path().to_path_buf(),
        update_summary: false,
        summary_path: None,
    });
    assert!(result.is_ok());
    assert!(dir.path().join("softmax-kernel-v1.md").exists());
}

#[test]
fn dispatch_lean_with_output_dir() {
    let dir = tempfile::tempdir().expect("tempdir");
    let result = dispatch(Commands::Lean {
        contract: test_contract(),
        output_dir: Some(dir.path().to_path_buf()),
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_proof_status_markdown() {
    let result = dispatch(Commands::ProofStatus {
        path: contracts_dir(),
        binding: None,
        format: "markdown".to_string(),
        table: false,
        kind: None,
        verify_bindings: None,
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_roofline_text() {
    let result = dispatch(Commands::Roofline {
        contract_dir: contracts_dir(),
        params: 7_000_000_000,
        bits: 4,
        hardware: "apple-m".to_string(),
        format: "text".to_string(),
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_roofline_json() {
    let result = dispatch(Commands::Roofline {
        contract_dir: contracts_dir(),
        params: 7_000_000_000,
        bits: 4,
        hardware: "a100".to_string(),
        format: "json".to_string(),
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_roofline_invalid_hw() {
    let result = dispatch(Commands::Roofline {
        contract_dir: contracts_dir(),
        params: 7_000_000_000,
        bits: 4,
        hardware: "invalid".to_string(),
        format: "text".to_string(),
    });
    assert!(result.is_err());
}

#[test]
fn dispatch_pipeline_text() {
    let pipeline_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../aprender-contracts-staging/contracts/pipelines/inference-forward-v1.yaml");
    let result = dispatch(Commands::Pipeline {
        pipeline: pipeline_path,
        format: "text".to_string(),
    });
    assert!(result.is_ok());
}

#[test]
fn dispatch_pipeline_json() {
    let pipeline_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../aprender-contracts-staging/contracts/pipelines/inference-forward-v1.yaml");
    let result = dispatch(Commands::Pipeline {
        pipeline: pipeline_path,
        format: "json".to_string(),
    });
    assert!(result.is_ok());
}
