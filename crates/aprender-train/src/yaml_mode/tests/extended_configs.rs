//! Extended configuration tests - YAML Mode QA Epic
//!
//! Tests for extended configuration options like CITL, RAG, graph, distillation,
//! privacy, audit, session, stress, benchmark, debug, signing, and verification.

use crate::yaml_mode::*;

#[test]
fn test_parse_citl_config() {
    let yaml = r#"
entrenar: "1.0"
name: "citl-test"
version: "1.0.0"

citl:
  mode: "error_suggest"
  error_code: "E0308"
  top_k: 5
  workspace: true
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let citl = manifest.citl.expect("operation should succeed");
    assert_eq!(citl.mode, "error_suggest");
    assert_eq!(citl.error_code, Some("E0308".to_string()));
    assert_eq!(citl.top_k, Some(5));
    assert_eq!(citl.workspace, Some(true));
}

#[test]
fn test_parse_rag_config() {
    let yaml = r#"
entrenar: "1.0"
name: "rag-test"
version: "1.0.0"

rag:
  store: "vectordb://localhost:6333"
  similarity_threshold: 0.85
  max_results: 10
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let rag = manifest.rag.expect("operation should succeed");
    assert_eq!(rag.store, "vectordb://localhost:6333");
    assert_eq!(rag.similarity_threshold, Some(0.85));
    assert_eq!(rag.max_results, Some(10));
}

#[test]
fn test_parse_graph_config() {
    let yaml = r#"
entrenar: "1.0"
name: "graph-test"
version: "1.0.0"

graph:
  output: "./graphs/model.dot"
  format: "dot"
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let graph = manifest.graph.expect("operation should succeed");
    assert_eq!(graph.output, "./graphs/model.dot");
    assert_eq!(graph.format, Some("dot".to_string()));
}

#[test]
fn test_parse_distillation_config() {
    let yaml = r#"
entrenar: "1.0"
name: "distill-test"
version: "1.0.0"

distillation:
  teacher:
    source: "hf://teacher-model"
  student:
    source: "hf://student-model"
  temperature: 4.0
  alpha: 0.5
  loss: "kl_div"
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let distill = manifest.distillation.expect("operation should succeed");
    assert_eq!(distill.teacher.source, "hf://teacher-model");
    assert_eq!(distill.student.source, "hf://student-model");
    assert_eq!(distill.temperature, 4.0);
    assert_eq!(distill.alpha, 0.5);
    assert_eq!(distill.loss, Some("kl_div".to_string()));
}

#[test]
fn test_parse_inspect_config() {
    let yaml = r#"
entrenar: "1.0"
name: "inspect-test"
version: "1.0.0"

inspect:
  mode: "detect"
  z_threshold: 3.0
  columns:
    - column_1
    - column_2
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let inspect = manifest.inspect.expect("operation should succeed");
    assert_eq!(inspect.mode, "detect");
    assert_eq!(inspect.z_threshold, Some(3.0));
    assert_eq!(inspect.columns, Some(vec!["column_1".to_string(), "column_2".to_string()]));
}

#[test]
fn test_parse_privacy_config() {
    let yaml = r#"
entrenar: "1.0"
name: "privacy-test"
version: "1.0.0"

privacy:
  differential: true
  epsilon: 1.0
  delta: 1e-5
  max_grad_norm: 1.0
  noise_multiplier: 1.1
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let privacy = manifest.privacy.expect("operation should succeed");
    assert!(privacy.differential);
    assert_eq!(privacy.epsilon, 1.0);
    assert_eq!(privacy.delta, Some(1e-5));
    assert_eq!(privacy.max_grad_norm, Some(1.0));
    assert_eq!(privacy.noise_multiplier, Some(1.1));
}

#[test]
fn test_parse_audit_config() {
    let yaml = r#"
entrenar: "1.0"
name: "audit-test"
version: "1.0.0"

audit:
  type: "fairness"
  protected_attr: "gender"
  threshold: 0.1
  metrics:
    - demographic_parity
    - equalized_odds
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let audit = manifest.audit.expect("operation should succeed");
    assert_eq!(audit.audit_type, "fairness");
    assert_eq!(audit.protected_attr, Some("gender".to_string()));
    assert_eq!(audit.threshold, Some(0.1));
    assert_eq!(
        audit.metrics,
        Some(vec!["demographic_parity".to_string(), "equalized_odds".to_string()])
    );
}

#[test]
fn test_parse_session_config() {
    let yaml = r#"
entrenar: "1.0"
name: "session-test"
version: "1.0.0"

session:
  id: "session-001"
  auto_save: true
  state_dir: "./checkpoints/session-001"
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let session = manifest.session.expect("operation should succeed");
    assert_eq!(session.id, "session-001");
    assert_eq!(session.auto_save, Some(true));
    assert_eq!(session.state_dir, Some("./checkpoints/session-001".to_string()));
}

#[test]
fn test_parse_stress_config() {
    let yaml = r#"
entrenar: "1.0"
name: "stress-test"
version: "1.0.0"

stress:
  parallel_jobs: 8
  duration: "4h"
  memory_limit: 0.9
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let stress = manifest.stress.expect("operation should succeed");
    assert_eq!(stress.parallel_jobs, 8);
    assert_eq!(stress.duration, Some("4h".to_string()));
    assert_eq!(stress.memory_limit, Some(0.9));
}

#[test]
fn test_parse_benchmark_config() {
    let yaml = r#"
entrenar: "1.0"
name: "benchmark-test"
version: "1.0.0"

benchmark:
  mode: "latency"
  warmup: 10
  iterations: 100
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let benchmark = manifest.benchmark.expect("operation should succeed");
    assert_eq!(benchmark.mode, "latency");
    assert_eq!(benchmark.warmup, Some(10));
    assert_eq!(benchmark.iterations, Some(100));
}

#[test]
fn test_parse_debug_config() {
    let yaml = r#"
entrenar: "1.0"
name: "debug-test"
version: "1.0.0"

debug:
  memory_profile: true
  log_interval: 100
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let debug = manifest.debug.expect("operation should succeed");
    assert_eq!(debug.memory_profile, Some(true));
    assert_eq!(debug.log_interval, Some(100));
}

#[test]
fn test_parse_signing_config() {
    let yaml = r#"
entrenar: "1.0"
name: "signing-test"
version: "1.0.0"

signing:
  enabled: true
  algorithm: "ed25519"
  key: "${SIGNING_KEY}"
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let signing = manifest.signing.expect("operation should succeed");
    assert!(signing.enabled);
    assert_eq!(signing.algorithm, Some("ed25519".to_string()));
    assert_eq!(signing.key, Some("${SIGNING_KEY}".to_string()));
}

#[test]
fn test_parse_verification_config() {
    let yaml = r#"
entrenar: "1.0"
name: "verification-test"
version: "1.0.0"

verification:
  all_25_checks: true
  qa_lead_sign_off: "required"
  eng_lead_sign_off: "required"
  safety_officer_sign_off: "required"
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let verification = manifest.verification.expect("operation should succeed");
    assert_eq!(verification.all_25_checks, Some(true));
    assert_eq!(verification.qa_lead_sign_off, Some("required".to_string()));
}

#[test]
fn test_parse_strict_mode() {
    let yaml = r#"
entrenar: "1.0"
name: "strict-test"
version: "1.0.0"

strict_validation: true
require_peer_review: true
lockfile: "./train.lock"
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    assert_eq!(manifest.strict_validation, Some(true));
    assert_eq!(manifest.require_peer_review, Some(true));
    assert_eq!(manifest.lockfile, Some("./train.lock".to_string()));
}
