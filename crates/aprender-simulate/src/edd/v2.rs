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

    /// Without `jsonschema` compiled in, check the structural core of the schema
    /// (`type`, `required`, nested `properties`). This used to return `Ok(())`
    /// unconditionally, so `simular validate` printed "Schema validation PASSED"
    /// for an empty file (#4079).
    #[cfg(not(feature = "schema-validation"))]
    #[allow(clippy::unused_self)]
    fn validate_against_schema(
        &self,
        instance: &serde_json::Value,
        schema: &serde_json::Value,
    ) -> Result<(), SchemaValidationError> {
        let mut errors = Vec::new();
        check_structure(instance, schema, schema, "", &mut errors);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(SchemaValidationError::ValidationFailed(errors))
        }
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

/// The JSON Schema keywords the embedded schemas use, checked without `jsonschema`:
/// `type`, `required`, `properties`, `additionalProperties`, `items`, `minItems`,
/// `enum`, `pattern`, `minLength`, `minimum`, `maximum`, `anyOf`, `oneOf`, `not`
/// and local `$ref` (`#/$defs/...`). Annotations (`title`, `default`, ...) are ignored.
/// Resolves a local `$ref` against `root` and recurses into it, or records an
/// unresolvable-ref error. Returns `true` if the caller should stop (either
/// case: a `$ref` present means no other keyword on this schema applies).
#[cfg(not(feature = "schema-validation"))]
fn check_ref(
    instance: &serde_json::Value,
    schema: &serde_json::Value,
    root: &serde_json::Value,
    at: &str,
    here: &str,
    errors: &mut Vec<String>,
) -> bool {
    use serde_json::Value;
    let Some(r) = schema.get("$ref").and_then(Value::as_str) else {
        return false;
    };
    match r.strip_prefix('#').and_then(|p| root.pointer(p)) {
        Some(target) => check_structure(instance, target, root, at, errors),
        None => errors.push(format!("{here}: unresolvable $ref {r}")),
    }
    true
}

/// Checks `anyOf`/`oneOf`/`not` combinators, each re-running `check_structure`
/// on `instance` against the candidate sub-schemas.
#[cfg(not(feature = "schema-validation"))]
fn check_combinators(
    instance: &serde_json::Value,
    schema: &serde_json::Value,
    root: &serde_json::Value,
    at: &str,
    here: &str,
    errors: &mut Vec<String>,
) {
    use serde_json::Value;
    let passes = |sub: &Value| {
        let mut e = Vec::new();
        check_structure(instance, sub, root, at, &mut e);
        e.is_empty()
    };
    if let Some(branches) = schema.get("anyOf").and_then(Value::as_array) {
        if !branches.iter().any(passes) {
            errors.push(format!("{here}: matches none of anyOf"));
        }
    }
    if let Some(branches) = schema.get("oneOf").and_then(Value::as_array) {
        let n = branches.iter().filter(|b| passes(b)).count();
        if n != 1 {
            errors.push(format!("{here}: matches {n} of oneOf, expected exactly 1"));
        }
    }
    if let Some(forbidden) = schema.get("not") {
        if passes(forbidden) {
            errors.push(format!("{here}: matches a forbidden (`not`) schema"));
        }
    }
}

/// Checks `enum` membership.
#[cfg(not(feature = "schema-validation"))]
fn check_enum(
    instance: &serde_json::Value,
    schema: &serde_json::Value,
    here: &str,
    errors: &mut Vec<String>,
) {
    use serde_json::Value;
    if let Some(allowed) = schema.get("enum").and_then(Value::as_array) {
        if !allowed.contains(instance) {
            errors.push(format!("{here}: {instance} is not one of {allowed:?}"));
        }
    }
}

/// Checks `minimum`/`maximum` for a numeric instance.
#[cfg(not(feature = "schema-validation"))]
fn check_numeric_bounds(x: f64, schema: &serde_json::Value, here: &str, errors: &mut Vec<String>) {
    use serde_json::Value;
    if let Some(min) = schema.get("minimum").and_then(Value::as_f64) {
        if x < min {
            errors.push(format!("{here}: {x} < minimum {min}"));
        }
    }
    if let Some(max) = schema.get("maximum").and_then(Value::as_f64) {
        if x > max {
            errors.push(format!("{here}: {x} > maximum {max}"));
        }
    }
}

/// Checks `enum` membership and, for numeric instances, `minimum`/`maximum`.
#[cfg(not(feature = "schema-validation"))]
fn check_enum_and_bounds(
    instance: &serde_json::Value,
    schema: &serde_json::Value,
    here: &str,
    errors: &mut Vec<String>,
) {
    check_enum(instance, schema, here, errors);
    if let Some(x) = instance.as_f64() {
        check_numeric_bounds(x, schema, here, errors);
    }
}

/// Checks `minItems` and recurses `check_structure` over an `items` schema.
#[cfg(not(feature = "schema-validation"))]
fn check_array(
    items: &[serde_json::Value],
    schema: &serde_json::Value,
    root: &serde_json::Value,
    at: &str,
    here: &str,
    errors: &mut Vec<String>,
) {
    use serde_json::Value;
    if let Some(min) = schema.get("minItems").and_then(Value::as_u64) {
        if (items.len() as u64) < min {
            errors.push(format!(
                "{here}: needs at least {min} items, has {}",
                items.len()
            ));
        }
    }
    if let Some(item_schema) = schema.get("items") {
        for (i, item) in items.iter().enumerate() {
            check_structure(item, item_schema, root, &format!("{at}/{i}"), errors);
        }
    }
}

#[cfg(not(feature = "schema-validation"))]
/// Checks the by-kind keywords (`minLength`/`pattern`, `minItems`/`items`,
/// object properties) matching whichever kind `instance` actually is.
#[cfg(not(feature = "schema-validation"))]
fn check_by_kind(
    instance: &serde_json::Value,
    schema: &serde_json::Value,
    root: &serde_json::Value,
    at: &str,
    here: &str,
    errors: &mut Vec<String>,
) {
    if let Some(text) = instance.as_str() {
        check_string(text, schema, here, errors);
    }
    if let Some(items) = instance.as_array() {
        check_array(items, schema, root, at, here, errors);
    }
    if let Some(obj) = instance.as_object() {
        check_object(obj, schema, root, at, errors);
    }
}

#[cfg(not(feature = "schema-validation"))]
fn check_structure(
    instance: &serde_json::Value,
    schema: &serde_json::Value,
    root: &serde_json::Value,
    at: &str,
    errors: &mut Vec<String>,
) {
    let here = if at.is_empty() { "/" } else { at };
    if check_ref(instance, schema, root, at, here, errors) {
        return;
    }
    if !type_matches(instance, schema.get("type")) {
        errors.push(format!("{here}: expected type {}", schema["type"]));
        return;
    }
    check_combinators(instance, schema, root, at, here, errors);
    check_enum_and_bounds(instance, schema, here, errors);
    check_by_kind(instance, schema, root, at, here, errors);
}

#[cfg(not(feature = "schema-validation"))]
fn type_matches(instance: &serde_json::Value, ty: Option<&serde_json::Value>) -> bool {
    use serde_json::Value;
    let one = |t: &str| match t {
        "object" => instance.is_object(),
        "array" => instance.is_array(),
        "string" => instance.is_string(),
        "integer" => instance.is_i64() || instance.is_u64(),
        "number" => instance.is_number(),
        "boolean" => instance.is_boolean(),
        "null" => instance.is_null(),
        _ => true,
    };
    match ty {
        Some(Value::String(t)) => one(t),
        Some(Value::Array(ts)) => ts.iter().filter_map(Value::as_str).any(one),
        _ => true,
    }
}

#[cfg(not(feature = "schema-validation"))]
fn check_string(text: &str, schema: &serde_json::Value, here: &str, errors: &mut Vec<String>) {
    use serde_json::Value;
    if let Some(min) = schema.get("minLength").and_then(Value::as_u64) {
        if (text.chars().count() as u64) < min {
            errors.push(format!("{here}: shorter than minLength {min}"));
        }
    }
    if let Some(pat) = schema.get("pattern").and_then(Value::as_str) {
        match regex::Regex::new(pat) {
            Ok(re) if re.is_match(text) => {}
            Ok(_) => errors.push(format!("{here}: \"{text}\" does not match {pat}")),
            Err(e) => errors.push(format!("{here}: bad pattern {pat}: {e}")),
        }
    }
}

#[cfg(not(feature = "schema-validation"))]
fn check_object(
    obj: &serde_json::Map<String, serde_json::Value>,
    schema: &serde_json::Value,
    root: &serde_json::Value,
    at: &str,
    errors: &mut Vec<String>,
) {
    use serde_json::Value;
    let here = if at.is_empty() { "/" } else { at };
    if let Some(req) = schema.get("required").and_then(Value::as_array) {
        for key in req.iter().filter_map(Value::as_str) {
            if !obj.contains_key(key) {
                errors.push(format!("{here}: missing required property \"{key}\""));
            }
        }
    }
    let props = schema.get("properties").and_then(Value::as_object);
    let extra = schema.get("additionalProperties");
    for (key, v) in obj {
        let path = format!("{at}/{key}");
        match (props.and_then(|p| p.get(key)), extra) {
            (Some(sub), _) | (None, Some(sub @ Value::Object(_))) => {
                check_structure(v, sub, root, &path, errors);
            }
            (None, Some(Value::Bool(false))) => {
                errors.push(format!("{here}: unknown property \"{key}\""));
            }
            (None, _) => {}
        }
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
    /// Free fns in this file's non-test part as (name, gated `not(schema-validation)`, body).
    fn free_fns(src: &str) -> Vec<(String, bool, String)> {
        let head = src.split("\n#[cfg(test)]\nmod tests").next().unwrap_or(src);
        let lines: Vec<&str> = head.lines().collect();
        let starts: Vec<usize> = (0..lines.len())
            .filter(|&i| lines[i].starts_with("fn ") || lines[i].starts_with("pub fn "))
            .collect();
        let mut out = Vec::new();
        for (k, &i) in starts.iter().enumerate() {
            let sig = lines[i]
                .trim_start_matches("pub ")
                .trim_start_matches("fn ");
            let name: String = sig
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            let gated = i > 0 && lines[i - 1] == "#[cfg(not(feature = \"schema-validation\"))]";
            // the body ends at its own closing brace in column 0, never in a later impl block
            let stop = starts.get(k + 1).copied().unwrap_or(lines.len());
            let end = (i..stop).find(|&j| lines[j] == "}").map_or(stop, |j| j + 1);
            out.push((name, gated, lines[i..end].join("\n")));
        }
        out
    }

    /// FALSIFY: an ungated free fn that calls a `not(schema-validation)` fn does not compile
    /// under `--features schema-validation` (check_structure lost its gate in the CB-200 split).
    #[test]
    fn falsify_every_caller_of_a_non_schema_validation_fn_is_gated_too() {
        let fns = free_fns(include_str!("v2.rs"));
        let gated: Vec<&str> = fns.iter().filter(|f| f.1).map(|f| f.0.as_str()).collect();
        assert!(
            gated.contains(&"check_ref"),
            "scan found no gated fns: {gated:?}"
        );
        let bad: Vec<String> = fns
            .iter()
            .filter(|f| !f.1)
            .flat_map(|f| {
                gated
                    .iter()
                    .filter(|g| f.2.contains(&format!("{g}(")))
                    .map(move |g| format!("{} calls {g}", f.0))
            })
            .collect();
        assert!(
            bad.is_empty(),
            "ungated fn calls a not(schema-validation) fn: {bad:?}"
        );
    }
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
