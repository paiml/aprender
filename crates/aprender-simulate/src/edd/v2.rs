//! EDD v2: YAML-Only, probar-First Simulation Framework
//!
//! Key changes from EDD v1:
//! - YAML-ONLY: No JavaScript/HTML/custom code
//! - probar-FIRST: Foundation of testing pyramid
//! - 95% mutation coverage (hard requirement)
//! - Replayable simulations (WASM/TUI/.mp4)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ============================================================================
// Core Types
// ============================================================================

/// EDD v2 Experiment loaded from YAML only
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YamlExperiment {
    /// Unique experiment identifier
    pub id: String,
    /// Random seed (MANDATORY)
    pub seed: u64,
    /// Reference to EMC (MANDATORY)
    pub emc_ref: String,
    /// Simulation configuration
    pub simulation: SimulationConfig,
    /// Falsification criteria (MANDATORY)
    pub falsification: FalsificationConfig,
}

/// Simulation configuration - all from YAML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Simulation type (must be supported by core engine)
    #[serde(rename = "type")]
    pub sim_type: String,
    /// Parameters as key-value pairs
    #[serde(default)]
    pub parameters: HashMap<String, serde_yaml::Value>,
}

/// Falsification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationConfig {
    /// List of falsification criteria
    pub criteria: Vec<FalsificationCriterionV2>,
}

/// Single falsification criterion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationCriterionV2 {
    /// Criterion identifier
    pub id: String,
    /// Metric to evaluate
    #[serde(default)]
    pub metric: Option<String>,
    /// Threshold value
    pub threshold: f64,
    /// Condition expression
    pub condition: String,
    /// Severity level
    #[serde(default = "default_severity")]
    pub severity: String,
}

fn default_severity() -> String {
    "major".to_string()
}

/// Replay file format for shareable simulations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayFile {
    /// Format version
    pub version: String,
    /// Original seed
    pub seed: u64,
    /// Reference to experiment YAML
    pub experiment_ref: String,
    /// Timeline of simulation steps
    pub timeline: Vec<ReplayStep>,
    /// Available export outputs
    #[serde(default)]
    pub outputs: ReplayOutputs,
}

/// Single step in replay timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayStep {
    /// Step number
    pub step: u64,
    /// State snapshot
    pub state: HashMap<String, serde_yaml::Value>,
    /// Equation evaluations at this step
    #[serde(default)]
    pub equations: Vec<EquationEvaluation>,
}

/// Equation evaluation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquationEvaluation {
    /// Equation identifier
    pub id: String,
    /// Computed value
    pub value: f64,
}

/// Available replay outputs
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReplayOutputs {
    /// WASM bundle path
    #[serde(default)]
    pub wasm: Option<String>,
    /// MP4 video path
    #[serde(default)]
    pub mp4: Option<String>,
    /// TUI session path
    #[serde(default)]
    pub tui_session: Option<String>,
}

// ============================================================================
// YAML-Only Loader
// ============================================================================

/// Errors for YAML loading
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum YamlLoadError {
    /// File not found
    FileNotFound(String),
    /// Parse error
    ParseError(String),
    /// Missing required field
    MissingField(String),
    /// Invalid configuration
    InvalidConfig(String),
    /// Custom code detected (PROHIBITED)
    CustomCodeDetected(String),
}

impl std::fmt::Display for YamlLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound(p) => write!(f, "File not found: {p}"),
            Self::ParseError(e) => write!(f, "YAML parse error: {e}"),
            Self::MissingField(field) => write!(f, "Missing required field: {field}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid configuration: {msg}"),
            Self::CustomCodeDetected(msg) => {
                write!(f, "PROHIBITED: Custom code detected - {msg}")
            }
        }
    }
}

impl std::error::Error for YamlLoadError {}

/// Load experiment from YAML file
///
/// # Errors
/// Returns error if file doesn't exist, has parse errors, or contains custom code
pub fn load_yaml_experiment(path: &Path) -> Result<YamlExperiment, YamlLoadError> {
    // Check file exists
    if !path.exists() {
        return Err(YamlLoadError::FileNotFound(path.display().to_string()));
    }

    // Read file contents
    let contents =
        std::fs::read_to_string(path).map_err(|e| YamlLoadError::ParseError(e.to_string()))?;

    // Check for prohibited patterns (custom code)
    check_for_custom_code(&contents)?;

    // Parse YAML
    let experiment: YamlExperiment =
        serde_yaml::from_str(&contents).map_err(|e| YamlLoadError::ParseError(e.to_string()))?;

    // Validate required fields
    validate_experiment(&experiment)?;

    Ok(experiment)
}

/// Check for prohibited custom code patterns
fn check_for_custom_code(contents: &str) -> Result<(), YamlLoadError> {
    let prohibited_patterns = [
        ("javascript:", "JavaScript code"),
        ("script:", "Script code"),
        ("<script", "HTML script tag"),
        ("function(", "JavaScript function"),
        ("() =>", "Arrow function"),
        ("eval(", "Eval expression"),
        ("new Function", "Function constructor"),
    ];

    for (pattern, description) in prohibited_patterns {
        if contents.to_lowercase().contains(&pattern.to_lowercase()) {
            return Err(YamlLoadError::CustomCodeDetected(description.to_string()));
        }
    }

    Ok(())
}

/// Validate experiment has all required fields
fn validate_experiment(exp: &YamlExperiment) -> Result<(), YamlLoadError> {
    if exp.id.is_empty() {
        return Err(YamlLoadError::MissingField("id".to_string()));
    }
    if exp.emc_ref.is_empty() {
        return Err(YamlLoadError::MissingField("emc_ref".to_string()));
    }
    if exp.falsification.criteria.is_empty() {
        return Err(YamlLoadError::MissingField(
            "falsification.criteria".to_string(),
        ));
    }

    Ok(())
}

// ============================================================================
// Replay System
// ============================================================================

/// Replay recorder for capturing simulation state
#[derive(Debug)]
pub struct ReplayRecorder {
    seed: u64,
    experiment_ref: String,
    timeline: Vec<ReplayStep>,
}

impl ReplayRecorder {
    /// Create new replay recorder
    #[must_use]
    pub fn new(seed: u64, experiment_ref: &str) -> Self {
        Self {
            seed,
            experiment_ref: experiment_ref.to_string(),
            timeline: Vec::new(),
        }
    }

    /// Record a step
    pub fn record_step(&mut self, step: u64, state: HashMap<String, serde_yaml::Value>) {
        contract_pre_iterator!();
        self.timeline.push(ReplayStep {
            step,
            state,
            equations: Vec::new(),
        });
    }

    /// Record step with equation evaluations
    pub fn record_step_with_equations(
        &mut self,
        step: u64,
        state: HashMap<String, serde_yaml::Value>,
        equations: Vec<EquationEvaluation>,
    ) {
        contract_pre_iterator!();
        self.timeline.push(ReplayStep {
            step,
            state,
            equations,
        });
    }

    /// Finalize and create replay file
    #[must_use]
    pub fn finalize(self) -> ReplayFile {
        ReplayFile {
            version: "1.0".to_string(),
            seed: self.seed,
            experiment_ref: self.experiment_ref,
            timeline: self.timeline,
            outputs: ReplayOutputs::default(),
        }
    }

    /// Get number of recorded steps
    #[must_use]
    pub fn step_count(&self) -> usize {
        contract_pre_iterator!();
        self.timeline.len()
    }
}

/// Export replay to different formats
pub struct ReplayExporter;

impl ReplayExporter {
    /// Export replay to YAML file
    ///
    /// # Errors
    /// Returns error if serialization or file write fails
    pub fn export_yaml(replay: &ReplayFile, path: &Path) -> Result<(), std::io::Error> {
        let yaml = serde_yaml::to_string(replay)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, yaml)
    }

    /// Export replay to JSON (for WASM consumption)
    ///
    /// # Errors
    /// Returns error if serialization fails
    pub fn export_json(replay: &ReplayFile) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(replay)
    }
}

// ============================================================================
// Falsification Evaluator
// ============================================================================

/// Result of falsification evaluation
#[derive(Debug, Clone)]
pub struct FalsificationEvalResult {
    /// Criterion ID
    pub criterion_id: String,
    /// Whether criterion passed
    pub passed: bool,
    /// Actual value measured
    pub actual_value: f64,
    /// Threshold from criterion
    pub threshold: f64,
    /// Message explaining result
    pub message: String,
}

/// Evaluate falsification criteria against simulation results
pub struct FalsificationEvaluator;

impl FalsificationEvaluator {
    /// Evaluate a single criterion
    #[must_use]
    pub fn evaluate_criterion(
        criterion: &FalsificationCriterionV2,
        actual_value: f64,
    ) -> FalsificationEvalResult {
        let passed =
            Self::evaluate_condition(&criterion.condition, actual_value, criterion.threshold);

        let message = if passed {
            format!(
                "PASSED: {} = {:.6} satisfies '{}'",
                criterion.id, actual_value, criterion.condition
            )
        } else {
            format!(
                "FAILED: {} = {:.6} violates '{}' (threshold: {})",
                criterion.id, actual_value, criterion.condition, criterion.threshold
            )
        };

        FalsificationEvalResult {
            criterion_id: criterion.id.clone(),
            passed,
            actual_value,
            threshold: criterion.threshold,
            message,
        }
    }

    /// Evaluate condition expression
    fn evaluate_condition(condition: &str, value: f64, threshold: f64) -> bool {
        // Simple condition evaluator
        // Supports: "< threshold", "> threshold", "<= threshold", ">= threshold",
        // "value < threshold", "gap < threshold", etc.
        let condition_lower = condition.to_lowercase();

        if condition_lower.contains("< threshold") || condition_lower.contains("<= threshold") {
            value <= threshold
        } else if condition_lower.contains("> threshold")
            || condition_lower.contains(">= threshold")
        {
            value >= threshold
        } else if condition_lower.contains('<') {
            value < threshold
        } else if condition_lower.contains('>') {
            value > threshold
        } else {
            // Default: check if value is within threshold of expected
            (value - threshold).abs() < threshold * 0.01
        }
    }

    /// Evaluate all criteria
    #[must_use]
    pub fn evaluate_all(
        criteria: &[FalsificationCriterionV2],
        values: &HashMap<String, f64>,
    ) -> Vec<FalsificationEvalResult> {
        criteria
            .iter()
            .filter_map(|c| {
                let metric_key = c.metric.as_ref().unwrap_or(&c.id);
                values
                    .get(metric_key)
                    .map(|&v| Self::evaluate_criterion(c, v))
            })
            .collect()
    }

    /// Check if all criteria passed
    #[must_use]
    pub fn all_passed(results: &[FalsificationEvalResult]) -> bool {
        results.iter().all(|r| r.passed)
    }
}

// ============================================================================
// JSON Schema Validation
// ============================================================================

/// Schema validation errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaValidationError {
    /// Schema file not found
    SchemaNotFound(String),
    /// Schema parse error
    SchemaParseError(String),
    /// Validation failed
    ValidationFailed(Vec<String>),
    /// YAML parse error
    YamlParseError(String),
}

impl std::fmt::Display for SchemaValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SchemaNotFound(p) => write!(f, "Schema not found: {p}"),
            Self::SchemaParseError(e) => write!(f, "Schema parse error: {e}"),
            Self::ValidationFailed(errors) => {
                write!(f, "Validation failed: {}", errors.join("; "))
            }
            Self::YamlParseError(e) => write!(f, "YAML parse error: {e}"),
        }
    }
}

impl std::error::Error for SchemaValidationError {}

/// Schema validator for experiments and EMCs
pub struct SchemaValidator {
    experiment_schema: serde_json::Value,
    emc_schema: serde_json::Value,
}

impl SchemaValidator {
    /// Create new validator from schema files
    ///
    /// # Errors
    /// Returns error if schema files cannot be read or parsed
    pub fn from_files(
        experiment_schema_path: &Path,
        emc_schema_path: &Path,
    ) -> Result<Self, SchemaValidationError> {
        let experiment_schema = Self::load_schema(experiment_schema_path)?;
        let emc_schema = Self::load_schema(emc_schema_path)?;

        Ok(Self {
            experiment_schema,
            emc_schema,
        })
    }

    /// Create validator from embedded schemas (for use without file access).
    ///
    /// # Panics
    ///
    /// Panics if embedded schema files are invalid JSON. This is a compile-time
    /// guarantee and indicates a build system error, not a runtime condition.
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn from_embedded() -> Self {
        let experiment_schema: serde_json::Value =
            serde_json::from_str(include_str!("../../schemas/experiment.schema.json"))
                .expect("Embedded experiment schema must be valid JSON");
        let emc_schema: serde_json::Value =
            serde_json::from_str(include_str!("../../schemas/emc.schema.json"))
                .expect("Embedded EMC schema must be valid JSON");

        Self {
            experiment_schema,
            emc_schema,
        }
    }

    fn load_schema(path: &Path) -> Result<serde_json::Value, SchemaValidationError> {
        if !path.exists() {
            return Err(SchemaValidationError::SchemaNotFound(
                path.display().to_string(),
            ));
        }

        let contents = std::fs::read_to_string(path)
            .map_err(|e| SchemaValidationError::SchemaParseError(e.to_string()))?;

        serde_json::from_str(&contents)
            .map_err(|e| SchemaValidationError::SchemaParseError(e.to_string()))
    }

    /// Validate experiment YAML against schema
    ///
    /// # Errors
    /// Returns error if YAML is invalid or doesn't conform to schema
    pub fn validate_experiment(&self, yaml_content: &str) -> Result<(), SchemaValidationError> {
        // Parse YAML to JSON Value
        let yaml_value: serde_json::Value = serde_yaml::from_str(yaml_content)
            .map_err(|e| SchemaValidationError::YamlParseError(e.to_string()))?;

        self.validate_against_schema(&yaml_value, &self.experiment_schema)
    }

    /// Validate EMC YAML against schema
    ///
    /// # Errors
    /// Returns error if YAML is invalid or doesn't conform to schema
    pub fn validate_emc(&self, yaml_content: &str) -> Result<(), SchemaValidationError> {
        let yaml_value: serde_json::Value = serde_yaml::from_str(yaml_content)
            .map_err(|e| SchemaValidationError::YamlParseError(e.to_string()))?;

        self.validate_against_schema(&yaml_value, &self.emc_schema)
    }

    #[cfg(feature = "schema-validation")]
    #[allow(clippy::unused_self)]
    fn validate_against_schema(
        &self,
        instance: &serde_json::Value,
        schema: &serde_json::Value,
    ) -> Result<(), SchemaValidationError> {
        let compiled = jsonschema::validator_for(schema)
            .map_err(|e| SchemaValidationError::SchemaParseError(e.to_string()))?;

        let result = compiled.validate(instance);

        if let Err(error) = result {
            // Single error case - collect into vec
            return Err(SchemaValidationError::ValidationFailed(vec![
                error.to_string()
            ]));
        }

        // Also check iter_errors for additional validation errors
        let errors: Vec<String> = compiled
            .iter_errors(instance)
            .map(|e| e.to_string())
            .collect();
        if !errors.is_empty() {
            return Err(SchemaValidationError::ValidationFailed(errors));
        }

        Ok(())
    }

    #[cfg(not(feature = "schema-validation"))]
    #[allow(clippy::unused_self, clippy::unnecessary_wraps)]
    fn validate_against_schema(
        &self,
        _instance: &serde_json::Value,
        _schema: &serde_json::Value,
    ) -> Result<(), SchemaValidationError> {
        // Schema validation disabled (WASM build)
        Ok(())
    }

    /// Validate experiment file
    ///
    /// # Errors
    /// Returns error if file cannot be read or validation fails
    pub fn validate_experiment_file(&self, path: &Path) -> Result<(), SchemaValidationError> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| SchemaValidationError::YamlParseError(e.to_string()))?;
        self.validate_experiment(&contents)
    }

    /// Validate EMC file
    ///
    /// # Errors
    /// Returns error if file cannot be read or validation fails
    pub fn validate_emc_file(&self, path: &Path) -> Result<(), SchemaValidationError> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| SchemaValidationError::YamlParseError(e.to_string()))?;
        self.validate_emc(&contents)
    }
}

/// Validate experiment YAML with embedded schema (convenience function)
///
/// # Errors
/// Returns error if validation fails
pub fn validate_experiment_yaml(yaml_content: &str) -> Result<(), SchemaValidationError> {
    let validator = SchemaValidator::from_embedded();
    validator.validate_experiment(yaml_content)
}

/// Validate EMC YAML with embedded schema (convenience function)
///
/// # Errors
/// Returns error if validation fails
pub fn validate_emc_yaml(yaml_content: &str) -> Result<(), SchemaValidationError> {
    let validator = SchemaValidator::from_embedded();
    validator.validate_emc(yaml_content)
}

// ============================================================================
// Unit tests — red-green-refactor cycle
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // =========================================================================
    // RED PHASE: YAML-Only Loading Tests
    // =========================================================================

    #[test]
    fn test_yaml_loader_rejects_javascript() {
        let yaml_with_js = r#"
experiment:
  id: "BAD-001"
  seed: 42
  emc_ref: "test/emc"
  simulation:
    type: "custom"
    javascript: "function() { alert('bad'); }"
  falsification:
    criteria:
      - id: "test"
        threshold: 0.1
        condition: "value < threshold"
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(yaml_with_js.as_bytes()).unwrap();

        let result = load_yaml_experiment(file.path());
        assert!(
            matches!(result, Err(YamlLoadError::CustomCodeDetected(_))),
            "Should reject YAML with JavaScript"
        );
    }

    #[test]
    fn test_yaml_loader_rejects_script_tags() {
        let yaml_with_html = r#"
experiment:
  id: "BAD-002"
  seed: 42
  emc_ref: "test/emc"
  simulation:
    type: "custom"
    html: "<script>alert('bad')</script>"
  falsification:
    criteria:
      - id: "test"
        threshold: 0.1
        condition: "value < threshold"
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(yaml_with_html.as_bytes()).unwrap();

        let result = load_yaml_experiment(file.path());
        assert!(
            matches!(result, Err(YamlLoadError::CustomCodeDetected(_))),
            "Should reject YAML with HTML script tags"
        );
    }

    #[test]
    fn test_yaml_loader_rejects_arrow_functions() {
        let yaml_with_arrow = r#"
experiment:
  id: "BAD-003"
  seed: 42
  emc_ref: "test/emc"
  simulation:
    type: "custom"
    callback: "() => console.log('bad')"
  falsification:
    criteria:
      - id: "test"
        threshold: 0.1
        condition: "value < threshold"
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(yaml_with_arrow.as_bytes()).unwrap();

        let result = load_yaml_experiment(file.path());
        assert!(
            matches!(result, Err(YamlLoadError::CustomCodeDetected(_))),
            "Should reject YAML with arrow functions"
        );
    }

    #[test]
    fn test_yaml_loader_accepts_valid_yaml() {
        let valid_yaml = r#"
id: "GOOD-001"
seed: 42
emc_ref: "optimization/tsp_grasp"
simulation:
  type: "tsp_grasp"
  parameters:
    n_cities: 25
    rcl_size: 5
falsification:
  criteria:
    - id: "optimality_gap"
      threshold: 0.25
      condition: "gap < threshold"
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(valid_yaml.as_bytes()).unwrap();

        let result = load_yaml_experiment(file.path());
        assert!(result.is_ok(), "Should accept valid YAML-only experiment");

        let exp = result.unwrap();
        assert_eq!(exp.id, "GOOD-001");
        assert_eq!(exp.seed, 42);
        assert_eq!(exp.emc_ref, "optimization/tsp_grasp");
    }

    #[test]
    fn test_yaml_loader_requires_seed() {
        let yaml_no_seed = r#"
id: "NO-SEED"
emc_ref: "test/emc"
simulation:
  type: "test"
falsification:
  criteria:
    - id: "test"
      threshold: 0.1
      condition: "value < threshold"
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(yaml_no_seed.as_bytes()).unwrap();

        let result = load_yaml_experiment(file.path());
        // Should fail parsing because seed is required
        assert!(result.is_err(), "Should reject YAML without seed");
    }

    #[test]
    fn test_yaml_loader_requires_emc_ref() {
        let yaml_no_emc = r#"
id: "NO-EMC"
seed: 42
emc_ref: ""
simulation:
  type: "test"
falsification:
  criteria:
    - id: "test"
      threshold: 0.1
      condition: "value < threshold"
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(yaml_no_emc.as_bytes()).unwrap();

        let result = load_yaml_experiment(file.path());
        assert!(
            matches!(result, Err(YamlLoadError::MissingField(_))),
            "Should reject YAML without emc_ref"
        );
    }

    #[test]
    fn test_yaml_loader_requires_falsification_criteria() {
        let yaml_no_falsification = r#"
id: "NO-FALSIFICATION"
seed: 42
emc_ref: "test/emc"
simulation:
  type: "test"
falsification:
  criteria: []
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(yaml_no_falsification.as_bytes()).unwrap();

        let result = load_yaml_experiment(file.path());
        assert!(
            matches!(result, Err(YamlLoadError::MissingField(_))),
            "Should reject YAML without falsification criteria"
        );
    }

    // =========================================================================
    // RED PHASE: Replay System Tests
    // =========================================================================

    #[test]
    fn test_replay_recorder_captures_steps() {
        let mut recorder = ReplayRecorder::new(42, "experiments/test.yaml");

        let mut state1 = HashMap::new();
        state1.insert(
            "tour_length".to_string(),
            serde_yaml::Value::Number(1234.into()),
        );
        recorder.record_step(0, state1);

        let mut state2 = HashMap::new();
        state2.insert(
            "tour_length".to_string(),
            serde_yaml::Value::Number(1198.into()),
        );
        recorder.record_step(1, state2);

        assert_eq!(recorder.step_count(), 2, "Should record 2 steps");
    }

    #[test]
    fn test_replay_recorder_captures_equations() {
        let mut recorder = ReplayRecorder::new(42, "experiments/test.yaml");

        let state = HashMap::new();
        let equations = vec![
            EquationEvaluation {
                id: "tour_length".to_string(),
                value: 1234.5,
            },
            EquationEvaluation {
                id: "two_opt_delta".to_string(),
                value: 36.3,
            },
        ];

        recorder.record_step_with_equations(0, state, equations);

        let replay = recorder.finalize();
        assert_eq!(replay.timeline[0].equations.len(), 2);
    }

    #[test]
    fn test_replay_finalize_creates_valid_file() {
        let mut recorder = ReplayRecorder::new(42, "experiments/tsp.yaml");
        recorder.record_step(0, HashMap::new());

        let replay = recorder.finalize();

        assert_eq!(replay.version, "1.0");
        assert_eq!(replay.seed, 42);
        assert_eq!(replay.experiment_ref, "experiments/tsp.yaml");
        assert_eq!(replay.timeline.len(), 1);
    }

    #[test]
    fn test_replay_export_yaml() {
        let replay = ReplayFile {
            version: "1.0".to_string(),
            seed: 42,
            experiment_ref: "test.yaml".to_string(),
            timeline: vec![ReplayStep {
                step: 0,
                state: HashMap::new(),
                equations: vec![],
            }],
            outputs: ReplayOutputs::default(),
        };

        let file = NamedTempFile::new().unwrap();
        let result = ReplayExporter::export_yaml(&replay, file.path());
        assert!(result.is_ok(), "Should export to YAML");

        // Verify file exists and has content
        let contents = std::fs::read_to_string(file.path()).unwrap();
        assert!(contents.contains("version: '1.0'") || contents.contains("version: \"1.0\""));
        assert!(contents.contains("seed: 42"));
    }

    #[test]
    fn test_replay_export_json() {
        let replay = ReplayFile {
            version: "1.0".to_string(),
            seed: 42,
            experiment_ref: "test.yaml".to_string(),
            timeline: vec![],
            outputs: ReplayOutputs::default(),
        };

        let result = ReplayExporter::export_json(&replay);
        assert!(result.is_ok(), "Should export to JSON");

        let json = result.unwrap();
        assert!(json.contains("\"version\": \"1.0\""));
        assert!(json.contains("\"seed\": 42"));
    }

    // =========================================================================
    // RED PHASE: Falsification Evaluator Tests
    // =========================================================================

    #[test]
    fn test_falsification_evaluator_passes_when_under_threshold() {
        let criterion = FalsificationCriterionV2 {
            id: "optimality_gap".to_string(),
            metric: None,
            threshold: 0.25,
            condition: "gap < threshold".to_string(),
            severity: "critical".to_string(),
        };

        let result = FalsificationEvaluator::evaluate_criterion(&criterion, 0.18);
        assert!(
            result.passed,
            "Should pass when gap (0.18) < threshold (0.25)"
        );
    }

    #[test]
    fn test_falsification_evaluator_fails_when_over_threshold() {
        let criterion = FalsificationCriterionV2 {
            id: "optimality_gap".to_string(),
            metric: None,
            threshold: 0.25,
            condition: "gap < threshold".to_string(),
            severity: "critical".to_string(),
        };

        let result = FalsificationEvaluator::evaluate_criterion(&criterion, 0.30);
        assert!(
            !result.passed,
            "Should fail when gap (0.30) > threshold (0.25)"
        );
    }

    #[test]
    fn test_falsification_evaluator_handles_greater_than() {
        let criterion = FalsificationCriterionV2 {
            id: "accuracy".to_string(),
            metric: None,
            threshold: 0.95,
            condition: "accuracy > threshold".to_string(),
            severity: "major".to_string(),
        };

        let result = FalsificationEvaluator::evaluate_criterion(&criterion, 0.97);
        assert!(
            result.passed,
            "Should pass when accuracy (0.97) > threshold (0.95)"
        );
    }

    #[test]
    fn test_falsification_evaluator_all_criteria() {
        let criteria = vec![
            FalsificationCriterionV2 {
                id: "gap".to_string(),
                metric: Some("gap".to_string()),
                threshold: 0.25,
                condition: "gap < threshold".to_string(),
                severity: "critical".to_string(),
            },
            FalsificationCriterionV2 {
                id: "energy".to_string(),
                metric: Some("energy_drift".to_string()),
                threshold: 1e-9,
                condition: "drift < threshold".to_string(),
                severity: "critical".to_string(),
            },
        ];

        let mut values = HashMap::new();
        values.insert("gap".to_string(), 0.18);
        values.insert("energy_drift".to_string(), 1e-10);

        let results = FalsificationEvaluator::evaluate_all(&criteria, &values);
        assert_eq!(results.len(), 2);
        assert!(FalsificationEvaluator::all_passed(&results));
    }

    #[test]
    fn test_falsification_evaluator_detects_failure() {
        let criteria = vec![FalsificationCriterionV2 {
            id: "gap".to_string(),
            metric: Some("gap".to_string()),
            threshold: 0.25,
            condition: "gap < threshold".to_string(),
            severity: "critical".to_string(),
        }];

        let mut values = HashMap::new();
        values.insert("gap".to_string(), 0.30); // Over threshold!

        let results = FalsificationEvaluator::evaluate_all(&criteria, &values);
        assert!(!FalsificationEvaluator::all_passed(&results));
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_full_edd_v2_workflow() {
        // 1. Create valid YAML experiment
        let yaml = r#"
id: "TSP-GRASP-001"
seed: 42
emc_ref: "optimization/tsp_grasp"
simulation:
  type: "tsp_grasp"
  parameters:
    n_cities: 25
    rcl_size: 5
    max_iterations: 100
falsification:
  criteria:
    - id: "optimality_gap"
      metric: "gap"
      threshold: 0.25
      condition: "gap < threshold"
      severity: "critical"
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(yaml.as_bytes()).unwrap();

        // 2. Load experiment
        let experiment = load_yaml_experiment(file.path()).expect("Should load valid experiment");
        assert_eq!(experiment.id, "TSP-GRASP-001");

        // 3. Create replay recorder
        let mut recorder = ReplayRecorder::new(experiment.seed, &experiment.id);

        // 4. Simulate steps (mock)
        for step in 0..5 {
            let mut state = HashMap::new();
            state.insert(
                "tour_length".to_string(),
                serde_yaml::Value::Number((1000 - step * 50).into()),
            );
            recorder.record_step(step, state);
        }

        // 5. Finalize replay
        let replay = recorder.finalize();
        assert_eq!(replay.timeline.len(), 5);

        // 6. Evaluate falsification
        let mut values = HashMap::new();
        values.insert("gap".to_string(), 0.18);

        let results =
            FalsificationEvaluator::evaluate_all(&experiment.falsification.criteria, &values);
        assert!(FalsificationEvaluator::all_passed(&results));
    }

    // =========================================================================
    // RED PHASE: Schema Validation Tests
    // =========================================================================

    #[test]
    fn test_schema_validator_from_embedded() {
        // Should load embedded schemas without error
        let validator = SchemaValidator::from_embedded();
        // If we get here without panic, schemas are valid JSON
        assert!(!validator.experiment_schema.is_null());
        assert!(!validator.emc_schema.is_null());
    }

    #[test]
    fn test_schema_validates_valid_experiment() {
        let valid_yaml = r#"
id: "TSP-GRASP-001"
seed: 42
emc_ref: "optimization/tsp_grasp"
simulation:
  type: "tsp_grasp"
  parameters:
    n_cities: 25
    rcl_size: 5
falsification:
  criteria:
    - id: "optimality_gap"
      threshold: 0.25
      condition: "gap < threshold"
"#;

        let result = validate_experiment_yaml(valid_yaml);
        assert!(
            result.is_ok(),
            "Valid experiment YAML should pass: {:?}",
            result
        );
    }

    #[test]
    #[cfg(feature = "schema-validation")]
    fn test_schema_rejects_missing_seed() {
        let invalid_yaml = r#"
id: "TSP-001"
emc_ref: "optimization/tsp"
simulation:
  type: "tsp"
falsification:
  criteria:
    - id: "gap"
      threshold: 0.25
      condition: "gap < threshold"
"#;

        let result = validate_experiment_yaml(invalid_yaml);
        assert!(
            matches!(result, Err(SchemaValidationError::ValidationFailed(_))),
            "Should reject YAML without seed"
        );
    }

    #[test]
    #[cfg(feature = "schema-validation")]
    fn test_schema_rejects_missing_falsification() {
        let invalid_yaml = r#"
id: "TSP-001"
seed: 42
emc_ref: "optimization/tsp"
simulation:
  type: "tsp"
"#;

        let result = validate_experiment_yaml(invalid_yaml);
        assert!(
            matches!(result, Err(SchemaValidationError::ValidationFailed(_))),
            "Should reject YAML without falsification"
        );
    }

    #[test]
    #[cfg(feature = "schema-validation")]
    fn test_schema_rejects_empty_falsification_criteria() {
        let invalid_yaml = r#"
id: "TSP-001"
seed: 42
emc_ref: "optimization/tsp"
simulation:
  type: "tsp"
falsification:
  criteria: []
"#;

        let result = validate_experiment_yaml(invalid_yaml);
        assert!(
            matches!(result, Err(SchemaValidationError::ValidationFailed(_))),
            "Should reject YAML with empty falsification criteria"
        );
    }

    #[test]
    #[cfg(feature = "schema-validation")]
    fn test_schema_rejects_javascript_field() {
        let invalid_yaml = r#"
id: "TSP-001"
seed: 42
emc_ref: "optimization/tsp"
simulation:
  type: "tsp"
  javascript: "alert('bad')"
falsification:
  criteria:
    - id: "gap"
      threshold: 0.25
      condition: "gap < threshold"
"#;

        let result = validate_experiment_yaml(invalid_yaml);
        assert!(
            matches!(result, Err(SchemaValidationError::ValidationFailed(_))),
            "Should reject YAML with javascript field"
        );
    }

    #[test]
    fn test_schema_validates_valid_emc() {
        let valid_emc = r#"
emc_version: "1.0"
emc_id: "optimization/tsp_grasp"
identity:
  name: "TSP GRASP"
  version: "1.0.0"
governing_equation:
  latex: "L(\\pi) = \\sum d(\\pi_i, \\pi_{i+1})"
  plain_text: "L(π) = Σ d(πᵢ, πᵢ₊₁)"
  description: "Tour length is sum of edge distances"
domain_of_validity:
  parameters:
    n_cities:
      min: 3
      max: 1000
  assumptions:
    - "Euclidean distance"
falsification:
  criteria:
    - id: "optimality_gap"
      condition: "gap <= 0.25"
"#;

        let result = validate_emc_yaml(valid_emc);
        assert!(result.is_ok(), "Valid EMC YAML should pass: {:?}", result);
    }

    #[test]
    #[cfg(feature = "schema-validation")]
    fn test_schema_rejects_emc_without_governing_equation() {
        let invalid_emc = r#"
emc_version: "1.0"
emc_id: "optimization/tsp"
identity:
  name: "TSP"
  version: "1.0.0"
domain_of_validity:
  assumptions:
    - "Test"
falsification:
  criteria:
    - id: "gap"
      condition: "gap < 0.25"
"#;

        let result = validate_emc_yaml(invalid_emc);
        assert!(
            matches!(result, Err(SchemaValidationError::ValidationFailed(_))),
            "Should reject EMC without governing_equation"
        );
    }

    #[test]
    #[cfg(feature = "schema-validation")]
    fn test_schema_rejects_emc_invalid_version() {
        let invalid_emc = r#"
emc_version: "invalid"
emc_id: "optimization/tsp"
identity:
  name: "TSP"
  version: "1.0.0"
governing_equation:
  latex: "L = sum"
  plain_text: "L = sum"
  description: "Tour length"
domain_of_validity:
  assumptions:
    - "Test"
falsification:
  criteria:
    - id: "gap"
      condition: "gap < 0.25"
"#;

        let result = validate_emc_yaml(invalid_emc);
        assert!(
            matches!(result, Err(SchemaValidationError::ValidationFailed(_))),
            "Should reject EMC with invalid version format"
        );
    }

    #[test]
    fn test_schema_validation_error_display() {
        let errors = vec!["error1".to_string(), "error2".to_string()];
        let err = SchemaValidationError::ValidationFailed(errors);
        let display = format!("{err}");
        assert!(display.contains("error1"));
        assert!(display.contains("error2"));

        let not_found = SchemaValidationError::SchemaNotFound("test.json".to_string());
        assert!(format!("{not_found}").contains("test.json"));
    }

    #[test]
    fn test_validate_experiment_file() {
        let valid_yaml = r#"
id: "TSP-001"
seed: 42
emc_ref: "optimization/tsp"
simulation:
  type: "tsp"
falsification:
  criteria:
    - id: "gap"
      threshold: 0.25
      condition: "gap < threshold"
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(valid_yaml.as_bytes()).unwrap();

        let validator = SchemaValidator::from_embedded();
        let result = validator.validate_experiment_file(file.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_emc_file() {
        let valid_emc = r#"
emc_version: "1.0"
emc_id: "test/emc"
identity:
  name: "Test EMC"
  version: "1.0.0"
governing_equation:
  latex: "x = y"
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(valid_emc.as_bytes()).unwrap();

        let validator = SchemaValidator::from_embedded();
        let result = validator.validate_emc_file(file.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_experiment_file_not_found() {
        let validator = SchemaValidator::from_embedded();
        let result = validator.validate_experiment_file(Path::new("/nonexistent/file.yaml"));
        assert!(matches!(
            result,
            Err(SchemaValidationError::YamlParseError(_))
        ));
    }

    #[test]
    fn test_validate_emc_file_not_found() {
        let validator = SchemaValidator::from_embedded();
        let result = validator.validate_emc_file(Path::new("/nonexistent/emc.yaml"));
        assert!(matches!(
            result,
            Err(SchemaValidationError::YamlParseError(_))
        ));
    }

    #[test]
    fn test_schema_validation_error_yaml_parse() {
        let err = SchemaValidationError::YamlParseError("invalid yaml".to_string());
        let display = format!("{err}");
        assert!(display.contains("YAML parse error"));
        assert!(display.contains("invalid yaml"));
    }

    #[test]
    fn test_schema_validation_error_schema_parse() {
        let err = SchemaValidationError::SchemaParseError("bad schema".to_string());
        let display = format!("{err}");
        assert!(display.contains("Schema parse error"));
        assert!(display.contains("bad schema"));
    }

    #[test]
    fn test_yaml_load_error_display() {
        let file_not_found = YamlLoadError::FileNotFound("/path/to/file".to_string());
        assert!(format!("{file_not_found}").contains("File not found"));

        let parse_error = YamlLoadError::ParseError("bad syntax".to_string());
        assert!(format!("{parse_error}").contains("YAML parse error"));

        let missing_field = YamlLoadError::MissingField("seed".to_string());
        assert!(format!("{missing_field}").contains("Missing required field"));

        let invalid_config = YamlLoadError::InvalidConfig("bad config".to_string());
        assert!(format!("{invalid_config}").contains("Invalid configuration"));

        let custom_code = YamlLoadError::CustomCodeDetected("javascript".to_string());
        assert!(format!("{custom_code}").contains("PROHIBITED"));
    }

    #[test]
    fn test_falsification_evaluator_default_condition() {
        // Test condition that doesn't contain < or > - uses default path
        let criterion = FalsificationCriterionV2 {
            id: "equality".to_string(),
            metric: None,
            threshold: 100.0,
            condition: "value equals threshold".to_string(),
            severity: "major".to_string(),
        };

        // Default behavior: check if value is within 1% of threshold
        let result = FalsificationEvaluator::evaluate_criterion(&criterion, 100.5);
        assert!(
            result.passed,
            "Should pass when value is within 1% of threshold"
        );

        let result_fail = FalsificationEvaluator::evaluate_criterion(&criterion, 110.0);
        assert!(
            !result_fail.passed,
            "Should fail when value is not within 1% of threshold"
        );
    }

    #[test]
    fn test_falsification_evaluator_gte_condition() {
        let criterion = FalsificationCriterionV2 {
            id: "min_coverage".to_string(),
            metric: None,
            threshold: 0.95,
            condition: "coverage >= threshold".to_string(),
            severity: "critical".to_string(),
        };

        let result = FalsificationEvaluator::evaluate_criterion(&criterion, 0.95);
        assert!(result.passed, "Should pass when coverage equals threshold");

        let result_above = FalsificationEvaluator::evaluate_criterion(&criterion, 0.98);
        assert!(result_above.passed, "Should pass when coverage > threshold");
    }

    #[test]
    fn test_falsification_evaluator_lte_condition() {
        let criterion = FalsificationCriterionV2 {
            id: "max_error".to_string(),
            metric: None,
            threshold: 0.01,
            condition: "error <= threshold".to_string(),
            severity: "critical".to_string(),
        };

        let result = FalsificationEvaluator::evaluate_criterion(&criterion, 0.01);
        assert!(result.passed, "Should pass when error equals threshold");

        let result_below = FalsificationEvaluator::evaluate_criterion(&criterion, 0.005);
        assert!(result_below.passed, "Should pass when error < threshold");
    }

    #[test]
    fn test_falsification_evaluator_missing_metric() {
        let criteria = vec![FalsificationCriterionV2 {
            id: "gap".to_string(),
            metric: Some("missing_metric".to_string()),
            threshold: 0.25,
            condition: "gap < threshold".to_string(),
            severity: "critical".to_string(),
        }];

        let values = HashMap::new(); // No values at all
        let results = FalsificationEvaluator::evaluate_all(&criteria, &values);
        assert_eq!(
            results.len(),
            0,
            "Should skip criteria with missing metrics"
        );
    }

    #[test]
    fn test_replay_outputs_default() {
        let outputs = ReplayOutputs::default();
        assert!(outputs.wasm.is_none());
        assert!(outputs.mp4.is_none());
        assert!(outputs.tui_session.is_none());
    }

    #[test]
    fn test_default_severity() {
        assert_eq!(default_severity(), "major");
    }

    #[test]
    fn test_schema_validates_invalid_yaml_syntax() {
        let invalid_yaml = "{ this is: [ not valid yaml:";
        let result = validate_experiment_yaml(invalid_yaml);
        assert!(matches!(
            result,
            Err(SchemaValidationError::YamlParseError(_))
        ));
    }

    #[test]
    fn test_schema_validates_emc_invalid_yaml_syntax() {
        let invalid_yaml = "{ this is: [ not valid yaml:";
        let result = validate_emc_yaml(invalid_yaml);
        assert!(matches!(
            result,
            Err(SchemaValidationError::YamlParseError(_))
        ));
    }

    #[test]
    fn test_load_yaml_file_not_found() {
        let result = load_yaml_experiment(Path::new("/nonexistent/experiment.yaml"));
        assert!(matches!(result, Err(YamlLoadError::FileNotFound(_))));
    }

    #[test]
    fn test_load_yaml_invalid_syntax() {
        let invalid_yaml = "{ this is: [ not valid yaml:";
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(invalid_yaml.as_bytes()).unwrap();

        let result = load_yaml_experiment(file.path());
        assert!(matches!(result, Err(YamlLoadError::ParseError(_))));
    }
}
