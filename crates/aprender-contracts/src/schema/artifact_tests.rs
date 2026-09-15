use super::*;
use crate::binding::{parse_binding_str, validate_binding_registry};
use crate::error::{Severity, Violation};

/// Repo-root-relative path to a real artifact, resolved from this crate's
/// manifest dir so the tests read the SAME bytes `pv validate` reads.
fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn error_rules(violations: &[Violation]) -> Vec<String> {
    violations
        .iter()
        .filter(|v| v.severity == Severity::Error)
        .map(|v| format!("{}: {}", v.rule, v.message))
        .collect()
}

/// The discriminator: `metadata:` present ⇒ contract, whatever else is there.
#[test]
fn classification_case_table() {
    let cases: &[(&str, ArtifactKind)] = &[
        (
            "metadata:\n  version: \"1.0.0\"\n  description: d\nbindings: []\ntarget_crate: x\n",
            ArtifactKind::Contract,
        ),
        (
            "version: 1.0.0\ntarget_crate: aprender\nbindings: []\n",
            ArtifactKind::Binding,
        ),
        (
            "model_id: paiml/x\nprovenance:\n  pipeline: finetune\n",
            ArtifactKind::PublishManifest,
        ),
        // Unrecognised shape stays a contract, so it still fails with the
        // contract parse error rather than passing unchecked.
        ("some_other_key: 1\n", ArtifactKind::Contract),
        ("- a\n- b\n", ArtifactKind::Contract),
        ("not: [valid: yaml: {{", ArtifactKind::Contract),
    ];
    for (yaml, expected) in cases {
        assert_eq!(
            classify_artifact(yaml),
            *expected,
            "misclassified: {yaml:?}"
        );
    }
}

/// pv can now validate its OWN artifact — both binding registries in the tree.
#[test]
fn real_binding_registries_validate() {
    for relative in ["contracts/binding.yaml", "contracts/aprender/binding.yaml"] {
        let path = repo_path(relative);
        let (kind, violations) = validate_artifact(&path).expect("binding registry validates");
        assert_eq!(kind, ArtifactKind::Binding, "{relative}");
        assert!(
            error_rules(&violations).is_empty(),
            "{relative} rejected: {:?}",
            error_rules(&violations)
        );
    }
}

/// A malformed entry must be REJECTED, one arm per rule. Without the
/// must-not-flag counterpart above, a validator that refuses every registry
/// would look identical to this.
#[test]
fn malformed_binding_entries_are_rejected() {
    let no_target_crate = "version: 1.0.0\ntarget_crate: \"\"\nbindings:\n  - contract: c-v1.yaml\n    equation: e\n    function: f\n    module_path: m\n    status: implemented\n";
    let registry = parse_binding_str(no_target_crate).expect("parses");
    let rules = error_rules(&validate_binding_registry(&registry));
    assert!(
        rules.iter().any(|r| r.starts_with("BINDING-002")),
        "{rules:?}"
    );

    let empty_equation = "version: 1.0.0\ntarget_crate: aprender\nbindings:\n  - contract: c-v1.yaml\n    equation: \"\"\n    function: f\n    module_path: m\n    status: implemented\n";
    let registry = parse_binding_str(empty_equation).expect("parses");
    let rules = error_rules(&validate_binding_registry(&registry));
    assert!(
        rules.iter().any(|r| r.starts_with("BINDING-004")),
        "{rules:?}"
    );

    // The one that matters: `implemented`, pointing at nothing.
    let phantom = "version: 1.0.0\ntarget_crate: aprender\nbindings:\n  - contract: c-v1.yaml\n    equation: e\n    status: implemented\n";
    let registry = parse_binding_str(phantom).expect("parses");
    let rules = error_rules(&validate_binding_registry(&registry));
    assert!(
        rules.iter().any(|r| r.starts_with("BINDING-005")),
        "{rules:?}"
    );

    let duplicate = "version: 1.0.0\ntarget_crate: aprender\nbindings:\n  - contract: c-v1.yaml\n    equation: e\n    function: f\n    module_path: m\n    status: implemented\n  - contract: c-v1\n    equation: e\n    function: g\n    module_path: m\n    status: implemented\n";
    let registry = parse_binding_str(duplicate).expect("parses");
    let rules = error_rules(&validate_binding_registry(&registry));
    assert!(
        rules.iter().any(|r| r.starts_with("BINDING-006")),
        "{rules:?}"
    );

    let empty = "version: 1.0.0\ntarget_crate: aprender\nbindings: []\n";
    let registry = parse_binding_str(empty).expect("parses");
    let rules = error_rules(&validate_binding_registry(&registry));
    assert!(
        rules.iter().any(|r| r.starts_with("BINDING-003")),
        "{rules:?}"
    );
}

/// A shell-discharged binding — no Rust path, but `notes:` saying exactly what
/// runs — is accepted. Five such entries live in `contracts/binding.yaml`;
/// rejecting them would have made BINDING-005 false of the corpus.
#[test]
fn a_shell_discharged_binding_is_accepted() {
    let yaml = "version: 1.0.0\ntarget_crate: aprender\nbindings:\n  - contract: pr-review-skill-v2.yaml\n    equation: grounding_marks_are_closed\n    status: implemented\n    notes: Discharged by scripts/check_pr_review_receipt.sh rows 3, 11, 12, 14.\n";
    let registry = parse_binding_str(yaml).expect("parses");
    assert!(
        error_rules(&validate_binding_registry(&registry)).is_empty(),
        "shell-discharged binding rejected"
    );
}

/// The three real publish manifests validate against the field list their own
/// schema contract declares.
#[test]
fn real_publish_manifests_validate() {
    for relative in [
        "contracts/publish-manifests/paiml-qwen2.5-coder-7b-apache-q4k-v1-apr.yaml",
        "contracts/publish-manifests/paiml-qwen2.5-coder-7b-apache-q4k-v1-gguf.yaml",
        "contracts/publish-manifests/paiml-qwen2.5-coder-7b-apache-q4k-v1-safetensors.yaml",
    ] {
        let path = repo_path(relative);
        let (kind, violations) = validate_artifact(&path).expect("manifest validates");
        assert_eq!(kind, ArtifactKind::PublishManifest, "{relative}");
        assert!(
            error_rules(&violations).is_empty(),
            "{relative} rejected: {:?}",
            error_rules(&violations)
        );
    }
}

/// Mutation arm: drop a required field from a real manifest and the gate must
/// turn RED. The field list comes from `publish-manifest-v1.yaml` §schema, so
/// this also proves that list is actually being read.
#[test]
fn a_manifest_missing_a_required_field_is_rejected() {
    let path = repo_path("contracts/publish-manifests/paiml-qwen2.5-coder-7b-apache-q4k-v1-apr.yaml");
    let text = std::fs::read_to_string(&path).expect("manifest readable");
    let mut doc: Value = serde_yaml::from_str(&text).expect("manifest parses");
    let map = doc.as_mapping_mut().expect("manifest is a mapping");
    map.remove("license");
    let rules = error_rules(&validate_publish_manifest(&doc, &path));
    assert!(
        rules.iter().any(|r| r.starts_with("PM-SHAPE-001")),
        "removing `license` must turn the gate RED, got: {rules:?}"
    );
}

/// The other three manifest rules, each with a mutation that must turn RED.
#[test]
fn manifest_field_shape_case_table() {
    let path =
        repo_path("contracts/publish-manifests/paiml-qwen2.5-coder-7b-apache-q4k-v1-apr.yaml");
    let text = std::fs::read_to_string(&path).expect("manifest readable");

    let mutations: &[(&str, Value, &str)] = &[
        (
            "sha256",
            Value::String("0A854098D05B15921C173B7C8DEB87C1CBECDFFC66E918825C11A02775C73666".into()),
            "PM-SHAPE-003",
        ),
        ("size_bytes", Value::Number(0u64.into()), "PM-SHAPE-004"),
        (
            "artifact_url",
            Value::String("http://huggingface.co/paiml/x/resolve/main/x.apr".into()),
            "PM-SHAPE-005",
        ),
        (
            "license",
            Value::String("   ".into()),
            "PM-SHAPE-001",
        ),
    ];
    for (field, bad, rule) in mutations {
        let mut doc: Value = serde_yaml::from_str(&text).expect("manifest parses");
        let map = doc.as_mapping_mut().expect("mapping");
        map.insert(Value::String((*field).to_string()), bad.clone());
        let rules = error_rules(&validate_publish_manifest(&doc, &path));
        assert!(
            rules.iter().any(|r| r.starts_with(rule)),
            "mutating {field} must raise {rule}, got: {rules:?}"
        );
    }
}

/// Without the schema contract there is nothing to check against, and that
/// must be a loud failure rather than a silent pass.
#[test]
fn a_manifest_with_no_locatable_schema_is_rejected() {
    let doc: Value =
        serde_yaml::from_str("model_id: paiml/x\nprovenance:\n  pipeline: finetune\n")
            .expect("parses");
    let orphan = Path::new("/nonexistent-root-for-this-test/manifest.yaml");
    let rules = error_rules(&validate_publish_manifest(&doc, orphan));
    assert!(
        rules.iter().any(|r| r.starts_with("PM-SHAPE-000")),
        "{rules:?}"
    );
}

/// A contract still goes down the contract path and still gets contract rules.
#[test]
fn a_real_contract_still_validates_as_a_contract() {
    let path = repo_path("contracts/beat-sklearn-iris-v1.yaml");
    let (kind, violations) = validate_artifact(&path).expect("contract validates");
    assert_eq!(kind, ArtifactKind::Contract);
    assert!(error_rules(&violations).is_empty());
}
