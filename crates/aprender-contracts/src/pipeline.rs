//! Pipeline contract support — cross-repo compositional verification.
//!
//! Parses pipeline YAML files that declare multi-stage inference pipelines
//! spanning multiple repos, with cross-boundary obligations verifying that
//! output invariants of one stage satisfy input preconditions of the next.
//!
//! Section 26: Two-Tier Architecture and Compositional Contracts.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::ContractError;

/// A pipeline contract declaring cross-repo data flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineContract {
    pub metadata: PipelineMetadata,
    #[serde(default)]
    pub stages: Vec<PipelineStage>,
    #[serde(default)]
    pub cross_boundary_obligations: Vec<CrossBoundaryObligation>,
    #[serde(default)]
    pub performance_contract: Option<PerformanceContract>,
}

/// Pipeline metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetadata {
    pub version: String,
    #[serde(default)]
    pub created: Option<String>,
    #[serde(default)]
    pub author: Option<String>,
    pub description: String,
    #[serde(default)]
    pub pipeline: bool,
    #[serde(default)]
    pub references: Vec<String>,
    #[serde(default)]
    pub depends_on: Vec<String>,
}

/// A single stage in the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStage {
    pub name: String,
    #[serde(default)]
    pub repo: Option<String>,
    #[serde(default)]
    pub contract: Option<String>,
    #[serde(default)]
    pub equation: Option<String>,
    #[serde(default)]
    pub input_requires: Option<String>,
    #[serde(default)]
    pub output_invariant: Option<String>,
    #[serde(default)]
    pub output_shape: Option<String>,
    #[serde(default)]
    pub repeat: Option<String>,
    #[serde(default)]
    pub substages: Vec<PipelineStage>,
    #[serde(default)]
    pub depends_on_contracts: Vec<String>,
}

/// Cross-boundary obligation between pipeline stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossBoundaryObligation {
    pub id: String,
    pub property: String,
    pub from_stage: String,
    pub to_stage: String,
    pub formal: String,
    #[serde(default)]
    pub verification: Option<String>,
}

/// Performance contract tied to roofline model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceContract {
    #[serde(default)]
    pub roofline: Option<String>,
    #[serde(default)]
    pub prefill_bound: Option<String>,
    #[serde(default)]
    pub decode_bound: Option<String>,
    #[serde(default)]
    pub throughput_ceiling: Option<String>,
}

/// Parse a pipeline contract YAML file.
pub fn parse_pipeline(path: &Path) -> Result<PipelineContract, ContractError> {
    let content = std::fs::read_to_string(path)?;
    parse_pipeline_str(&content)
}

/// Parse a pipeline contract from a YAML string.
pub fn parse_pipeline_str(yaml: &str) -> Result<PipelineContract, ContractError> {
    let pipeline: PipelineContract = serde_yaml::from_str(yaml)?;
    Ok(pipeline)
}

/// Validate a pipeline contract for internal consistency.
///
/// Checks:
/// 1. All stages have names
/// 2. Cross-boundary obligations reference valid stage names
/// 3. Output invariants chain to input requirements
pub fn validate_pipeline(pipeline: &PipelineContract) -> Vec<PipelineIssue> {
    let mut issues = Vec::new();

    // Collect all stage names (including substages)
    let stage_names: Vec<String> = collect_stage_names(&pipeline.stages);

    // Check for duplicate stage names
    let mut seen = std::collections::HashSet::new();
    for name in &stage_names {
        if !seen.insert(name.as_str()) {
            issues.push(PipelineIssue {
                severity: IssueSeverity::Warning,
                message: format!("Duplicate stage name: {name}"),
            });
        }
    }

    // Check cross-boundary obligations reference valid stages
    for ob in &pipeline.cross_boundary_obligations {
        if !stage_names.contains(&ob.from_stage) {
            issues.push(PipelineIssue {
                severity: IssueSeverity::Error,
                message: format!(
                    "Obligation {} references unknown from_stage: {}",
                    ob.id, ob.from_stage
                ),
            });
        }
        if !stage_names.contains(&ob.to_stage) {
            issues.push(PipelineIssue {
                severity: IssueSeverity::Error,
                message: format!(
                    "Obligation {} references unknown to_stage: {}",
                    ob.id, ob.to_stage
                ),
            });
        }
    }

    // Check invariant chaining: output of stage N should relate to input of N+1
    for window in pipeline.stages.windows(2) {
        let prev = &window[0];
        let next = &window[1];
        if prev.output_invariant.is_none() {
            issues.push(PipelineIssue {
                severity: IssueSeverity::Warning,
                message: format!("Stage '{}' has no output_invariant", prev.name),
            });
        }
        if next.input_requires.is_none() {
            issues.push(PipelineIssue {
                severity: IssueSeverity::Warning,
                message: format!("Stage '{}' has no input_requires", next.name),
            });
        }
    }

    issues
}

/// Collect all stage names recursively (including substages).
fn collect_stage_names(stages: &[PipelineStage]) -> Vec<String> {
    let mut names = Vec::new();
    for stage in stages {
        names.push(stage.name.clone());
        if !stage.substages.is_empty() {
            names.extend(collect_stage_names(&stage.substages));
        }
    }
    names
}

/// A validation issue found in a pipeline contract.
#[derive(Debug, Clone)]
pub struct PipelineIssue {
    pub severity: IssueSeverity,
    pub message: String,
}

/// Issue severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity {
    Error,
    Warning,
}

impl std::fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Error => write!(f, "ERROR"),
            Self::Warning => write!(f, "WARN"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_pipeline() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test pipeline"
  pipeline: true
stages:
  - name: stage_a
    repo: repo_a
    contract: contract-v1
    equation: eq_a
    output_invariant: "x > 0"
  - name: stage_b
    repo: repo_b
    contract: contract-v1
    equation: eq_b
    input_requires: "x > 0"
cross_boundary_obligations:
  - id: PIPE-001
    property: "A feeds B"
    from_stage: stage_a
    to_stage: stage_b
    formal: "output(a) satisfies input(b)"
"#;
        let pipeline = parse_pipeline_str(yaml).unwrap();
        assert_eq!(pipeline.stages.len(), 2);
        assert_eq!(pipeline.cross_boundary_obligations.len(), 1);
        assert!(pipeline.metadata.pipeline);
    }

    #[test]
    fn validate_valid_pipeline() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
  pipeline: true
stages:
  - name: a
    output_invariant: "x > 0"
  - name: b
    input_requires: "x > 0"
    output_invariant: "y > 0"
cross_boundary_obligations:
  - id: P1
    property: "a→b"
    from_stage: a
    to_stage: b
    formal: "ok"
"#;
        let pipeline = parse_pipeline_str(yaml).unwrap();
        let issues = validate_pipeline(&pipeline);
        let errors: Vec<_> = issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .collect();
        assert!(errors.is_empty());
    }

    #[test]
    fn validate_bad_stage_ref() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
  pipeline: true
stages:
  - name: a
cross_boundary_obligations:
  - id: P1
    property: "bad ref"
    from_stage: a
    to_stage: nonexistent
    formal: "fail"
"#;
        let pipeline = parse_pipeline_str(yaml).unwrap();
        let issues = validate_pipeline(&pipeline);
        let errors: Vec<_> = issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .collect();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("nonexistent"));
    }

    #[test]
    fn parse_inference_forward() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/pipelines/inference-forward-v1.yaml");
        if path.exists() {
            let pipeline = parse_pipeline(&path).unwrap();
            assert!(pipeline.metadata.pipeline);
            assert!(!pipeline.stages.is_empty());
            assert!(!pipeline.cross_boundary_obligations.is_empty());
            let issues = validate_pipeline(&pipeline);
            let errors: Vec<_> = issues
                .iter()
                .filter(|i| i.severity == IssueSeverity::Error)
                .collect();
            assert!(errors.is_empty(), "Errors: {errors:?}");
        }
    }

    #[test]
    fn substage_names_collected() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
  pipeline: true
stages:
  - name: outer
    substages:
      - name: inner_a
      - name: inner_b
cross_boundary_obligations:
  - id: P1
    property: "inner ref"
    from_stage: inner_a
    to_stage: inner_b
    formal: "ok"
"#;
        let pipeline = parse_pipeline_str(yaml).unwrap();
        let issues = validate_pipeline(&pipeline);
        let errors: Vec<_> = issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .collect();
        assert!(errors.is_empty());
    }
}
