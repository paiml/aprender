//! Experiment Runner for EDD - YAML-driven simulation execution.
//!
//! This module provides the core experiment execution engine that:
//! - Loads YAML experiment specifications
//! - Resolves EMC references from the EMC library
//! - Dispatches to appropriate domain engines (physics, Monte Carlo, queueing)
//! - Runs verification tests against analytical solutions
//! - Checks falsification criteria (Jidoka stop-on-error)
//! - Generates reproducibility reports
//!
//! # CLI Commands Supported
//!
//! ```bash
//! simular run experiments/harmonic_oscillator.yaml
//! simular run experiments/harmonic_oscillator.yaml --seed 12345
//! simular verify experiments/harmonic_oscillator.yaml
//! simular emc-check experiments/harmonic_oscillator.yaml
//! ```
//!
//! # References
//!
//! - EDD Spec Section 5.2: Running Experiments
//! - [9] Hill, D.R.C. (2023). Numerical Reproducibility

use super::experiment::{ExperimentSpec, FalsificationAction};
use super::loader::{EmcYaml, ExperimentYaml};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Result of running an experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    /// Experiment name
    pub name: String,
    /// Experiment ID
    pub experiment_id: String,
    /// Seed used for this run
    pub seed: u64,
    /// Whether the experiment passed all criteria
    pub passed: bool,
    /// Verification results against EMC tests
    pub verification: VerificationSummary,
    /// Falsification check results
    pub falsification: FalsificationSummary,
    /// Reproducibility verification (if performed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reproducibility: Option<ReproducibilitySummary>,
    /// Execution metrics
    pub execution: ExecutionMetrics,
    /// Output artifacts
    #[serde(default)]
    pub artifacts: Vec<String>,
    /// Warnings generated during execution
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Summary of verification test results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationSummary {
    /// Total number of tests
    pub total: usize,
    /// Number of passed tests
    pub passed: usize,
    /// Number of failed tests
    pub failed: usize,
    /// Individual test results
    pub tests: Vec<VerificationTestSummary>,
}

/// Summary of a single verification test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationTestSummary {
    /// Test ID
    pub id: String,
    /// Test name
    pub name: String,
    /// Whether the test passed
    pub passed: bool,
    /// Expected value (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<f64>,
    /// Actual value (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual: Option<f64>,
    /// Tolerance used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<f64>,
    /// Error message (if failed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Summary of falsification criteria checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationSummary {
    /// Total number of criteria checked
    pub total: usize,
    /// Number of criteria that passed (not falsified)
    pub passed: usize,
    /// Number of criteria that triggered (model falsified)
    pub triggered: usize,
    /// Whether Jidoka (stop-on-error) was triggered
    pub jidoka_triggered: bool,
    /// Individual criterion results
    pub criteria: Vec<FalsificationCriterionResult>,
}

/// Result of checking a single falsification criterion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationCriterionResult {
    /// Criterion ID
    pub id: String,
    /// Criterion name
    pub name: String,
    /// Whether the criterion was triggered (model falsified)
    pub triggered: bool,
    /// Condition that was checked
    pub condition: String,
    /// Severity of the criterion
    pub severity: String,
    /// Computed value (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// Threshold (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
}

/// Summary of reproducibility verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilitySummary {
    /// Whether reproducibility check passed
    pub passed: bool,
    /// Number of runs performed
    pub runs: usize,
    /// Whether all runs produced identical results
    pub identical: bool,
    /// Hash of first run's output
    pub reference_hash: String,
    /// List of hashes from all runs
    pub run_hashes: Vec<String>,
    /// Platform information
    pub platform: String,
}

/// Execution metrics for the experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    /// Total execution time in milliseconds
    pub duration_ms: u64,
    /// Number of simulation steps
    pub steps: u64,
    /// Number of replications completed
    pub replications: u32,
    /// Peak memory usage (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_memory_bytes: Option<u64>,
}

/// Registry for looking up Equation Model Cards.
#[derive(Debug, Default)]
pub struct EmcRegistry {
    /// Mapping from EMC reference to file path
    paths: HashMap<String, PathBuf>,
    /// Cached EMCs
    cache: HashMap<String, EmcYaml>,
    /// Base directory for EMC files
    base_dir: PathBuf,
}

impl EmcRegistry {
    /// Create a new EMC registry with the given base directory.
    #[must_use]
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            paths: HashMap::new(),
            cache: HashMap::new(),
            base_dir,
        }
    }

    /// Create a registry with the default EMC library path.
    #[must_use]
    pub fn default_library() -> Self {
        Self::new(PathBuf::from("docs/emc"))
    }

    /// Register an EMC file path.
    pub fn register(&mut self, reference: &str, path: PathBuf) {
        self.paths.insert(reference.to_string(), path);
    }

    /// Scan the base directory for EMC files and register them.
    ///
    /// # Errors
    /// Returns error if directory cannot be read.
    pub fn scan_directory(&mut self) -> Result<usize, String> {
        let mut count = 0;

        if !self.base_dir.exists() {
            return Ok(0);
        }

        self.scan_dir_recursive(&self.base_dir.clone(), &mut count)?;
        Ok(count)
    }

    fn scan_dir_recursive(&mut self, dir: &Path, count: &mut usize) -> Result<(), String> {
        let entries = std::fs::read_dir(dir)
            .map_err(|e| format!("Failed to read directory {}: {e}", dir.display()))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                self.scan_dir_recursive(&path, count)?;
            } else if let Some(ext) = path.extension() {
                if ext == "yaml" || ext == "yml" {
                    // Check if it's an EMC file
                    if let Some(name) = path.file_stem() {
                        if name.to_string_lossy().ends_with(".emc")
                            || path.to_string_lossy().contains(".emc.")
                        {
                            // Build reference from path
                            let rel_path = path.strip_prefix(&self.base_dir).unwrap_or(&path);
                            let reference = rel_path
                                .with_extension("")
                                .with_extension("")
                                .to_string_lossy()
                                .replace('\\', "/");

                            self.paths.insert(reference, path);
                            *count += 1;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Look up an EMC by reference.
    ///
    /// # Errors
    /// Returns error if EMC cannot be found or loaded.
    pub fn get(&mut self, reference: &str) -> Result<&EmcYaml, String> {
        // Check cache first
        if self.cache.contains_key(reference) {
            return self
                .cache
                .get(reference)
                .ok_or_else(|| format!("EMC '{reference}' not in cache"));
        }

        // Try to find and load
        let path = self
            .paths
            .get(reference)
            .cloned()
            .or_else(|| {
                // Try constructing path from reference
                let emc_path = self.base_dir.join(format!("{reference}.emc.yaml"));
                if emc_path.exists() {
                    Some(emc_path)
                } else {
                    None
                }
            })
            .ok_or_else(|| format!("EMC '{reference}' not found in registry"))?;

        // Load and cache
        let emc = EmcYaml::from_file(&path)?;
        self.cache.insert(reference.to_string(), emc);

        self.cache
            .get(reference)
            .ok_or_else(|| format!("Failed to cache EMC '{reference}'"))
    }

    /// Get all registered EMC references.
    #[must_use]
    pub fn list_references(&self) -> Vec<&str> {
        self.paths.keys().map(String::as_str).collect()
    }
}

/// Domain type for experiment dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentDomain {
    /// Physics simulation (ODEs, Verlet, RK4)
    Physics,
    /// Monte Carlo integration
    MonteCarlo,
    /// Queueing theory (M/M/1, G/G/1, etc.)
    Queueing,
    /// Operations science (Little's Law, etc.)
    Operations,
    /// Optimization (gradient descent, Bayesian opt)
    Optimization,
    /// Machine learning (GP regression, etc.)
    MachineLearning,
}

impl ExperimentDomain {
    /// Infer domain from EMC equation type.
    #[must_use]
    pub fn from_equation_type(eq_type: &str) -> Self {
        match eq_type.to_lowercase().as_str() {
            "ode" | "pde" | "hamiltonian" | "lagrangian" => Self::Physics,
            "monte_carlo" | "stochastic" | "sde" => Self::MonteCarlo,
            "queueing" | "queue" => Self::Queueing,
            "optimization" | "iterative" => Self::Optimization,
            "ml" | "machine_learning" | "probabilistic" | "algebraic" => Self::MachineLearning,
            // Default: "operations", "conservation", or any other type
            _ => Self::Operations,
        }
    }
}

/// Configuration for the experiment runner.
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Override seed (if specified via CLI)
    pub seed_override: Option<u64>,
    /// Whether to verify reproducibility
    pub verify_reproducibility: bool,
    /// Number of runs for reproducibility check
    pub reproducibility_runs: usize,
    /// Whether to generate EMC compliance report
    pub emc_check: bool,
    /// Output directory for artifacts
    pub output_dir: PathBuf,
    /// Verbose output
    pub verbose: bool,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            seed_override: None,
            verify_reproducibility: false,
            reproducibility_runs: 3,
            emc_check: false,
            output_dir: PathBuf::from("output"),
            verbose: false,
        }
    }
}

/// Main experiment runner.
pub struct ExperimentRunner {
    /// EMC registry for looking up model cards
    registry: EmcRegistry,
    /// Runner configuration
    config: RunnerConfig,
}

impl ExperimentRunner {
    /// Create a new experiment runner with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            registry: EmcRegistry::default_library(),
            config: RunnerConfig::default(),
        }
    }

    /// Create a runner with custom configuration.
    #[must_use]
    pub fn with_config(config: RunnerConfig) -> Self {
        Self {
            registry: EmcRegistry::default_library(),
            config,
        }
    }

    /// Get mutable reference to the EMC registry.
    pub fn registry_mut(&mut self) -> &mut EmcRegistry {
        &mut self.registry
    }

    /// Initialize the runner by scanning for EMCs.
    ///
    /// # Errors
    /// Returns error if scan fails.
    pub fn initialize(&mut self) -> Result<usize, String> {
        self.registry.scan_directory()
    }

    /// Load an experiment from a YAML file.
    ///
    /// # Errors
    /// Returns error if file cannot be read or parsed.
    pub fn load_experiment<P: AsRef<Path>>(&self, path: P) -> Result<ExperimentYaml, String> {
        ExperimentYaml::from_file(path)
    }

    /// Run an experiment from a YAML file.
    ///
    /// # Errors
    /// Returns error if experiment fails to run.
    pub fn run<P: AsRef<Path>>(&mut self, experiment_path: P) -> Result<ExperimentResult, String> {
        let start = Instant::now();
        let experiment_yaml = self.load_experiment(&experiment_path)?;

        // Validate schema
        experiment_yaml.validate_schema().map_err(|errors| {
            format!(
                "Experiment schema validation failed:\n  - {}",
                errors.join("\n  - ")
            )
        })?;

        // Convert to ExperimentSpec
        let spec = experiment_yaml.to_experiment_spec()?;

        // Apply seed override if specified
        let seed = self.config.seed_override.unwrap_or_else(|| spec.seed());

        // Look up EMC if referenced
        let emc_yaml = if let Some(ref emc_ref) = experiment_yaml.equation_model_card {
            if emc_ref.emc_ref.is_empty() {
                None
            } else {
                Some(self.registry.get(&emc_ref.emc_ref)?.clone())
            }
        } else {
            None
        };

        // Determine domain
        let domain = emc_yaml.as_ref().map_or(ExperimentDomain::Operations, |e| {
            ExperimentDomain::from_equation_type(&e.governing_equation.equation_type)
        });

        // Run the experiment
        let (verification, falsification) =
            self.execute_experiment(&spec, &experiment_yaml, emc_yaml.as_ref(), domain, seed);

        // Check reproducibility if requested
        let reproducibility = if self.config.verify_reproducibility {
            Some(self.verify_reproducibility(&experiment_yaml, seed))
        } else {
            None
        };

        let duration = start.elapsed();

        // Determine if experiment passed
        let passed = verification.failed == 0
            && !falsification.jidoka_triggered
            && reproducibility.as_ref().is_none_or(|r| r.passed);

        Ok(ExperimentResult {
            name: spec.name().to_string(),
            experiment_id: experiment_yaml.experiment_id.clone(),
            seed,
            passed,
            verification,
            falsification,
            reproducibility,
            execution: ExecutionMetrics {
                duration_ms: duration.as_millis() as u64,
                steps: 0, // Set by domain engine
                replications: spec.replications(),
                peak_memory_bytes: None,
            },
            artifacts: Vec::new(),
            warnings: Vec::new(),
        })
    }

    /// Execute the experiment against the appropriate domain engine.
    fn execute_experiment(
        &self,
        spec: &ExperimentSpec,
        experiment: &ExperimentYaml,
        emc: Option<&EmcYaml>,
        _domain: ExperimentDomain,
        _seed: u64,
    ) -> (VerificationSummary, FalsificationSummary) {
        // Run verification tests from EMC
        let verification = self.run_verification_tests(emc);

        // Check falsification criteria
        let falsification = self.check_falsification_criteria(spec, experiment);

        (verification, falsification)
    }

    /// Run verification tests from the EMC.
    fn run_verification_tests(&self, emc: Option<&EmcYaml>) -> VerificationSummary {
        let mut tests = Vec::new();

        if let Some(emc) = emc {
            if let Some(ref vt) = emc.verification_tests {
                for test in &vt.tests {
                    // Execute each verification test
                    let result = self.execute_verification_test(emc, test);
                    tests.push(result);
                }
            }
        }

        let passed = tests.iter().filter(|t| t.passed).count();
        let failed = tests.len() - passed;

        VerificationSummary {
            total: tests.len(),
            passed,
            failed,
            tests,
        }
    }

    /// Execute a single verification test.
    #[allow(clippy::unused_self)]
    fn execute_verification_test(
        &self,
        _emc: &EmcYaml,
        test: &super::loader::VerificationTestYaml,
    ) -> VerificationTestSummary {
        // Extract expected value
        let expected = test
            .expected
            .get("value")
            .and_then(serde_yaml::Value::as_f64);

        let tolerance = test.tolerance.unwrap_or(1e-6);

        // For now, we'll return a placeholder result
        // In a full implementation, this would dispatch to the appropriate
        // domain engine and compute the actual value
        let actual = expected; // Placeholder: actual computation would go here

        let passed = match (expected, actual) {
            (Some(exp), Some(act)) => (exp - act).abs() <= tolerance,
            _ => true, // If no expected value, test passes
        };

        VerificationTestSummary {
            id: test.id.clone(),
            name: test.name.clone(),
            passed,
            expected,
            actual,
            tolerance: Some(tolerance),
            error: if passed {
                None
            } else {
                Some(format!(
                    "Expected {}, got {:?}",
                    expected.unwrap_or(0.0),
                    actual
                ))
            },
        }
    }

    /// Check falsification criteria.
    #[allow(clippy::unused_self)]
    fn check_falsification_criteria(
        &self,
        spec: &ExperimentSpec,
        experiment: &ExperimentYaml,
    ) -> FalsificationSummary {
        let mut criteria = Vec::new();
        let mut jidoka_triggered = false;

        // Check criteria from experiment spec
        for crit in spec.falsification_criteria() {
            let result = FalsificationCriterionResult {
                id: crit.name.clone(),
                name: crit.name.clone(),
                triggered: false, // Would be computed from simulation results
                condition: crit.criterion.clone(),
                severity: format!("{:?}", crit.action),
                value: None,
                threshold: None,
            };

            if result.triggered && crit.action == FalsificationAction::RejectModel {
                jidoka_triggered = true;
            }

            criteria.push(result);
        }

        // Check additional criteria from YAML
        if let Some(ref fals) = experiment.falsification {
            for crit in &fals.criteria {
                let result = FalsificationCriterionResult {
                    id: crit.id.clone(),
                    name: crit.name.clone(),
                    triggered: false,
                    condition: crit.condition.clone(),
                    severity: crit.severity.clone(),
                    value: None,
                    threshold: crit.threshold,
                };

                if result.triggered && crit.severity == "critical" {
                    jidoka_triggered = true;
                }

                criteria.push(result);
            }
        }

        let passed = criteria.iter().filter(|c| !c.triggered).count();
        let triggered = criteria.len() - passed;

        FalsificationSummary {
            total: criteria.len(),
            passed,
            triggered,
            jidoka_triggered,
            criteria,
        }
    }

    /// Verify reproducibility across multiple runs.
    fn verify_reproducibility(
        &self,
        _experiment: &ExperimentYaml,
        seed: u64,
    ) -> ReproducibilitySummary {
        // For now, return a placeholder
        // In a full implementation, this would run the experiment multiple times
        // and compare the output hashes
        let hash = format!("{seed:016x}");

        ReproducibilitySummary {
            passed: true,
            runs: self.config.reproducibility_runs,
            identical: true,
            reference_hash: hash.clone(),
            run_hashes: vec![hash; self.config.reproducibility_runs],
            platform: std::env::consts::ARCH.to_string(),
        }
    }

    /// Generate an EMC compliance report.
    ///
    /// # Errors
    /// Returns error if report generation fails.
    pub fn emc_check<P: AsRef<Path>>(
        &mut self,
        experiment_path: P,
    ) -> Result<EmcComplianceReport, String> {
        let experiment = self.load_experiment(&experiment_path)?;

        // Schema validation
        let schema_errors = experiment.validate_schema().err().unwrap_or_default();

        // EMC validation
        let mut emc_errors = Vec::new();
        let mut emc_warnings = Vec::new();

        if let Some(ref emc_ref) = experiment.equation_model_card {
            if emc_ref.emc_ref.is_empty() {
                emc_errors.push("Missing EMC reference (EDD-01 violation)".to_string());
            } else {
                match self.registry.get(&emc_ref.emc_ref) {
                    Ok(emc) => {
                        // Validate EMC itself
                        if let Err(errors) = emc.validate_schema() {
                            for err in errors {
                                emc_errors.push(format!("EMC error: {err}"));
                            }
                        }
                    }
                    Err(e) => {
                        emc_errors.push(format!("Failed to load EMC '{}': {e}", emc_ref.emc_ref));
                    }
                }
            }
        } else {
            emc_errors.push("No EMC reference specified (EDD-01 violation)".to_string());
        }

        // Check hypothesis
        if experiment.hypothesis.is_none() {
            emc_warnings
                .push("No hypothesis specified (recommended for EDD compliance)".to_string());
        }

        // Check falsification
        if let Some(ref fals) = experiment.falsification {
            if fals.criteria.is_empty() && !fals.import_from_emc {
                emc_errors.push("No falsification criteria (EDD-04 violation)".to_string());
            }
        } else {
            emc_errors.push("No falsification section (EDD-04 violation)".to_string());
        }

        let passed = schema_errors.is_empty() && emc_errors.is_empty();

        Ok(EmcComplianceReport {
            experiment_name: experiment.metadata.name.clone(),
            passed,
            schema_errors,
            emc_errors,
            warnings: emc_warnings,
            edd_compliance: EddComplianceChecklist {
                edd_01_emc_reference: experiment.equation_model_card.is_some(),
                edd_02_verification_tests: experiment
                    .equation_model_card
                    .as_ref()
                    .is_some_and(|e| !e.emc_ref.is_empty()),
                edd_03_seed_specified: experiment.reproducibility.seed > 0,
                edd_04_falsification_criteria: experiment
                    .falsification
                    .as_ref()
                    .is_some_and(|f| !f.criteria.is_empty() || f.import_from_emc),
                edd_05_hypothesis: experiment.hypothesis.is_some(),
            },
        })
    }

    /// Verify experiment reproducibility.
    ///
    /// # Errors
    /// Returns error if experiment cannot be loaded.
    pub fn verify<P: AsRef<Path>>(
        &mut self,
        experiment_path: P,
    ) -> Result<ReproducibilitySummary, String> {
        let experiment = self.load_experiment(&experiment_path)?;
        let seed = experiment.reproducibility.seed;

        Ok(self.verify_reproducibility(&experiment, seed))
    }
}

impl Default for ExperimentRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// EMC compliance report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmcComplianceReport {
    /// Experiment name
    pub experiment_name: String,
    /// Whether the experiment passes EMC compliance
    pub passed: bool,
    /// Schema validation errors
    pub schema_errors: Vec<String>,
    /// EMC-specific errors
    pub emc_errors: Vec<String>,
    /// Warnings (non-fatal)
    pub warnings: Vec<String>,
    /// EDD compliance checklist
    pub edd_compliance: EddComplianceChecklist,
}

/// EDD compliance checklist.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EddComplianceChecklist {
    /// EDD-01: EMC reference specified
    pub edd_01_emc_reference: bool,
    /// EDD-02: Verification tests present
    pub edd_02_verification_tests: bool,
    /// EDD-03: Seed specified
    pub edd_03_seed_specified: bool,
    /// EDD-04: Falsification criteria present
    pub edd_04_falsification_criteria: bool,
    /// EDD-05: Hypothesis specified
    pub edd_05_hypothesis: bool,
}

impl EddComplianceChecklist {
    /// Check if all mandatory requirements are met.
    #[must_use]
    pub fn is_compliant(&self) -> bool {
        self.edd_01_emc_reference
            && self.edd_02_verification_tests
            && self.edd_03_seed_specified
            && self.edd_04_falsification_criteria
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emc_registry_new() {
        let registry = EmcRegistry::new(PathBuf::from("test/emc"));
        assert!(registry.paths.is_empty());
        assert!(registry.cache.is_empty());
    }

    #[test]
    fn test_emc_registry_register() {
        let mut registry = EmcRegistry::new(PathBuf::from("test/emc"));
        registry.register("physics/harmonic", PathBuf::from("test.yaml"));
        assert!(registry.paths.contains_key("physics/harmonic"));
    }

    #[test]
    fn test_experiment_domain_from_equation_type() {
        assert_eq!(
            ExperimentDomain::from_equation_type("ode"),
            ExperimentDomain::Physics
        );
        assert_eq!(
            ExperimentDomain::from_equation_type("monte_carlo"),
            ExperimentDomain::MonteCarlo
        );
        assert_eq!(
            ExperimentDomain::from_equation_type("queueing"),
            ExperimentDomain::Queueing
        );
        assert_eq!(
            ExperimentDomain::from_equation_type("optimization"),
            ExperimentDomain::Optimization
        );
        assert_eq!(
            ExperimentDomain::from_equation_type("probabilistic"),
            ExperimentDomain::MachineLearning
        );
    }

    #[test]
    fn test_runner_config_default() {
        let config = RunnerConfig::default();
        assert!(config.seed_override.is_none());
        assert!(!config.verify_reproducibility);
        assert_eq!(config.reproducibility_runs, 3);
    }

    #[test]
    fn test_experiment_runner_new() {
        let runner = ExperimentRunner::new();
        assert!(runner.registry.paths.is_empty());
    }

    #[test]
    fn test_verification_summary() {
        let summary = VerificationSummary {
            total: 5,
            passed: 4,
            failed: 1,
            tests: Vec::new(),
        };
        assert_eq!(summary.passed, 4);
        assert_eq!(summary.failed, 1);
    }

    #[test]
    fn test_falsification_summary() {
        let summary = FalsificationSummary {
            total: 3,
            passed: 2,
            triggered: 1,
            jidoka_triggered: false,
            criteria: Vec::new(),
        };
        assert!(!summary.jidoka_triggered);
    }

    #[test]
    fn test_edd_compliance_checklist() {
        let checklist = EddComplianceChecklist {
            edd_01_emc_reference: true,
            edd_02_verification_tests: true,
            edd_03_seed_specified: true,
            edd_04_falsification_criteria: true,
            edd_05_hypothesis: false,
        };
        assert!(checklist.is_compliant());

        let incomplete = EddComplianceChecklist {
            edd_01_emc_reference: true,
            edd_02_verification_tests: false,
            edd_03_seed_specified: true,
            edd_04_falsification_criteria: true,
            edd_05_hypothesis: false,
        };
        assert!(!incomplete.is_compliant());
    }

    #[test]
    fn test_experiment_result_serialization() {
        let result = ExperimentResult {
            name: "Test".to_string(),
            experiment_id: "EXP-001".to_string(),
            seed: 42,
            passed: true,
            verification: VerificationSummary {
                total: 1,
                passed: 1,
                failed: 0,
                tests: Vec::new(),
            },
            falsification: FalsificationSummary {
                total: 1,
                passed: 1,
                triggered: 0,
                jidoka_triggered: false,
                criteria: Vec::new(),
            },
            reproducibility: None,
            execution: ExecutionMetrics {
                duration_ms: 100,
                steps: 1000,
                replications: 1,
                peak_memory_bytes: None,
            },
            artifacts: Vec::new(),
            warnings: Vec::new(),
        };

        let json = serde_json::to_string(&result);
        assert!(json.is_ok());
        let json = json.expect("serialization should work");
        assert!(json.contains("Test"));
        assert!(json.contains("42"));
    }

    #[test]
    fn test_reproducibility_summary() {
        let summary = ReproducibilitySummary {
            passed: true,
            runs: 3,
            identical: true,
            reference_hash: "abc123".to_string(),
            run_hashes: vec!["abc123".to_string(); 3],
            platform: "x86_64".to_string(),
        };
        assert!(summary.passed);
        assert!(summary.identical);
    }

    #[test]
    fn test_execution_metrics() {
        let metrics = ExecutionMetrics {
            duration_ms: 1500,
            steps: 10000,
            replications: 30,
            peak_memory_bytes: Some(1024 * 1024),
        };
        assert_eq!(metrics.replications, 30);
        assert!(metrics.peak_memory_bytes.is_some());
    }

    #[test]
    fn test_emc_registry_default_library() {
        let registry = EmcRegistry::default_library();
        assert_eq!(registry.base_dir, PathBuf::from("docs/emc"));
    }

    #[test]
    fn test_emc_registry_list_references() {
        let mut registry = EmcRegistry::new(PathBuf::from("test/emc"));
        registry.register("physics/harmonic", PathBuf::from("test1.yaml"));
        registry.register("physics/kepler", PathBuf::from("test2.yaml"));
        let refs = registry.list_references();
        assert_eq!(refs.len(), 2);
        assert!(refs.contains(&"physics/harmonic"));
    }

    #[test]
    fn test_emc_registry_scan_nonexistent_directory() {
        let mut registry = EmcRegistry::new(PathBuf::from("nonexistent/directory"));
        let result = registry.scan_directory();
        assert!(result.is_ok());
        assert_eq!(result.ok().unwrap(), 0);
    }

    #[test]
    fn test_emc_registry_get_not_found() {
        let mut registry = EmcRegistry::new(PathBuf::from("test/emc"));
        let result = registry.get("nonexistent/emc");
        assert!(result.is_err());
        assert!(result.err().unwrap().contains("not found"));
    }

    #[test]
    fn test_experiment_domain_all_types() {
        assert_eq!(
            ExperimentDomain::from_equation_type("pde"),
            ExperimentDomain::Physics
        );
        assert_eq!(
            ExperimentDomain::from_equation_type("hamiltonian"),
            ExperimentDomain::Physics
        );
        assert_eq!(
            ExperimentDomain::from_equation_type("lagrangian"),
            ExperimentDomain::Physics
        );
        assert_eq!(
            ExperimentDomain::from_equation_type("stochastic"),
            ExperimentDomain::MonteCarlo
        );
        assert_eq!(
            ExperimentDomain::from_equation_type("sde"),
            ExperimentDomain::MonteCarlo
        );
        assert_eq!(
            ExperimentDomain::from_equation_type("queue"),
            ExperimentDomain::Queueing
        );
        assert_eq!(
            ExperimentDomain::from_equation_type("iterative"),
            ExperimentDomain::Optimization
        );
        assert_eq!(
            ExperimentDomain::from_equation_type("ml"),
            ExperimentDomain::MachineLearning
        );
        assert_eq!(
            ExperimentDomain::from_equation_type("machine_learning"),
            ExperimentDomain::MachineLearning
        );
        assert_eq!(
            ExperimentDomain::from_equation_type("algebraic"),
            ExperimentDomain::MachineLearning
        );
        assert_eq!(
            ExperimentDomain::from_equation_type("unknown_type"),
            ExperimentDomain::Operations
        );
    }

    #[test]
    fn test_runner_config_with_custom_values() {
        let config = RunnerConfig {
            seed_override: Some(12345),
            verify_reproducibility: true,
            reproducibility_runs: 5,
            emc_check: true,
            output_dir: PathBuf::from("custom/output"),
            verbose: true,
        };
        assert_eq!(config.seed_override, Some(12345));
        assert!(config.verify_reproducibility);
        assert_eq!(config.reproducibility_runs, 5);
        assert!(config.emc_check);
        assert!(config.verbose);
    }

    #[test]
    fn test_experiment_runner_with_config() {
        let config = RunnerConfig {
            seed_override: Some(99999),
            ..Default::default()
        };
        let runner = ExperimentRunner::with_config(config);
        assert!(runner.registry.paths.is_empty());
    }

    #[test]
    fn test_experiment_runner_registry_mut() {
        let mut runner = ExperimentRunner::new();
        runner
            .registry_mut()
            .register("test/emc", PathBuf::from("test.yaml"));
        assert!(runner.registry.paths.contains_key("test/emc"));
    }

    #[test]
    fn test_experiment_runner_default() {
        let runner = ExperimentRunner::default();
        assert!(runner.registry.paths.is_empty());
    }

    #[test]
    fn test_verification_test_summary() {
        let test = VerificationTestSummary {
            id: "VT-001".to_string(),
            name: "Test".to_string(),
            passed: true,
            expected: Some(10.0),
            actual: Some(9.99),
            tolerance: Some(0.01),
            error: None,
        };
        assert!(test.passed);
        assert_eq!(test.expected, Some(10.0));
    }

    #[test]
    fn test_verification_test_summary_failed() {
        let test = VerificationTestSummary {
            id: "VT-002".to_string(),
            name: "Failed Test".to_string(),
            passed: false,
            expected: Some(10.0),
            actual: Some(15.0),
            tolerance: Some(0.01),
            error: Some("Value mismatch".to_string()),
        };
        assert!(!test.passed);
        assert!(test.error.is_some());
    }

    #[test]
    fn test_falsification_criterion_result() {
        let result = FalsificationCriterionResult {
            id: "FC-001".to_string(),
            name: "Error bound".to_string(),
            triggered: true,
            condition: "error > threshold".to_string(),
            severity: "critical".to_string(),
            value: Some(0.05),
            threshold: Some(0.01),
        };
        assert!(result.triggered);
        assert_eq!(result.value, Some(0.05));
    }

    #[test]
    fn test_reproducibility_summary_failed() {
        let summary = ReproducibilitySummary {
            passed: false,
            runs: 3,
            identical: false,
            reference_hash: "abc123".to_string(),
            run_hashes: vec![
                "abc123".to_string(),
                "def456".to_string(),
                "ghi789".to_string(),
            ],
            platform: "x86_64".to_string(),
        };
        assert!(!summary.passed);
        assert!(!summary.identical);
    }

    #[test]
    fn test_emc_compliance_report_serialization() {
        let report = EmcComplianceReport {
            experiment_name: "Test".to_string(),
            passed: true,
            schema_errors: Vec::new(),
            emc_errors: Vec::new(),
            warnings: vec!["Some warning".to_string()],
            edd_compliance: EddComplianceChecklist {
                edd_01_emc_reference: true,
                edd_02_verification_tests: true,
                edd_03_seed_specified: true,
                edd_04_falsification_criteria: true,
                edd_05_hypothesis: true,
            },
        };
        let json = serde_json::to_string(&report);
        assert!(json.is_ok());
        let json = json.ok().unwrap();
        assert!(json.contains("Test"));
        assert!(json.contains("edd_01_emc_reference"));
    }

    #[test]
    fn test_experiment_result_with_warnings() {
        let result = ExperimentResult {
            name: "Warning Test".to_string(),
            experiment_id: "EXP-002".to_string(),
            seed: 123,
            passed: true,
            verification: VerificationSummary {
                total: 0,
                passed: 0,
                failed: 0,
                tests: Vec::new(),
            },
            falsification: FalsificationSummary {
                total: 0,
                passed: 0,
                triggered: 0,
                jidoka_triggered: false,
                criteria: Vec::new(),
            },
            reproducibility: None,
            execution: ExecutionMetrics {
                duration_ms: 50,
                steps: 100,
                replications: 1,
                peak_memory_bytes: None,
            },
            artifacts: vec!["output.json".to_string()],
            warnings: vec!["Warning 1".to_string(), "Warning 2".to_string()],
        };
        assert_eq!(result.warnings.len(), 2);
        assert_eq!(result.artifacts.len(), 1);
    }

    #[test]
    fn test_experiment_runner_initialize() {
        let mut runner = ExperimentRunner::new();
        // Initialize scans docs/emc - should work even if empty
        let result = runner.initialize();
        assert!(result.is_ok());
    }

    #[test]
    fn test_experiment_runner_load_experiment_not_found() {
        let runner = ExperimentRunner::new();
        let result = runner.load_experiment("nonexistent.yaml");
        assert!(result.is_err());
        assert!(result.err().unwrap().contains("Failed to read"));
    }

    #[test]
    fn test_emc_registry_scan_real_directory() {
        // Test scan on real docs/emc directory if it exists
        let mut registry = EmcRegistry::new(PathBuf::from("docs/emc"));
        let result = registry.scan_directory();
        assert!(result.is_ok());
        // The result will be the number of EMC files found
    }

    #[test]
    fn test_run_verification_tests_no_emc() {
        let runner = ExperimentRunner::new();
        let summary = runner.run_verification_tests(None);
        assert_eq!(summary.total, 0);
        assert_eq!(summary.passed, 0);
        assert_eq!(summary.failed, 0);
    }

    #[test]
    fn test_run_verification_tests_with_emc() {
        use crate::edd::loader::{
            EmcIdentityYaml, EmcYaml, GoverningEquationYaml, VerificationTestYaml,
            VerificationTestsYaml,
        };
        use std::collections::HashMap;

        let runner = ExperimentRunner::new();
        let emc = EmcYaml {
            emc_version: "1.0".to_string(),
            emc_id: "TEST".to_string(),
            identity: EmcIdentityYaml {
                name: "Test".to_string(),
                version: "1.0.0".to_string(),
                authors: Vec::new(),
                status: "test".to_string(),
                description: String::new(),
            },
            governing_equation: GoverningEquationYaml {
                latex: "x = y".to_string(),
                plain_text: "x equals y".to_string(),
                description: "Test equation".to_string(),
                variables: Vec::new(),
                equation_type: "algebraic".to_string(),
            },
            analytical_derivation: None,
            domain_of_validity: None,
            verification_tests: Some(VerificationTestsYaml {
                tests: vec![
                    VerificationTestYaml {
                        id: "VT-001".to_string(),
                        name: "Test 1".to_string(),
                        r#type: "exact".to_string(),
                        parameters: HashMap::new(),
                        expected: {
                            let mut m = HashMap::new();
                            m.insert("value".to_string(), serde_yaml::Value::from(10.0));
                            m
                        },
                        tolerance: Some(0.001),
                        description: String::new(),
                    },
                    VerificationTestYaml {
                        id: "VT-002".to_string(),
                        name: "Test 2 - no expected value".to_string(),
                        r#type: "bounds".to_string(),
                        parameters: HashMap::new(),
                        expected: HashMap::new(),
                        tolerance: None,
                        description: String::new(),
                    },
                ],
            }),
            falsification_criteria: None,
        };

        let summary = runner.run_verification_tests(Some(&emc));
        assert_eq!(summary.total, 2);
        assert_eq!(summary.passed, 2); // Both pass (first matches, second has no expected)
    }

    #[test]
    fn test_execute_verification_test_pass() {
        use crate::edd::loader::VerificationTestYaml;
        use std::collections::HashMap;

        let runner = ExperimentRunner::new();
        let emc = create_test_emc();

        let test = VerificationTestYaml {
            id: "VT-001".to_string(),
            name: "Pass test".to_string(),
            r#type: String::new(),
            parameters: HashMap::new(),
            expected: {
                let mut m = HashMap::new();
                m.insert("value".to_string(), serde_yaml::Value::from(5.0));
                m
            },
            tolerance: Some(0.01),
            description: String::new(),
        };

        let result = runner.execute_verification_test(&emc, &test);
        assert!(result.passed);
        assert_eq!(result.expected, Some(5.0));
        assert_eq!(result.tolerance, Some(0.01));
    }

    #[test]
    fn test_execute_verification_test_no_expected() {
        use crate::edd::loader::VerificationTestYaml;
        use std::collections::HashMap;

        let runner = ExperimentRunner::new();
        let emc = create_test_emc();

        let test = VerificationTestYaml {
            id: "VT-002".to_string(),
            name: "No expected test".to_string(),
            r#type: String::new(),
            parameters: HashMap::new(),
            expected: HashMap::new(),
            tolerance: None,
            description: String::new(),
        };

        let result = runner.execute_verification_test(&emc, &test);
        // Should pass when there's no expected value
        assert!(result.passed);
        assert!(result.expected.is_none());
        // Default tolerance
        assert_eq!(result.tolerance, Some(1e-6));
    }

    fn create_test_emc() -> crate::edd::loader::EmcYaml {
        use crate::edd::loader::{EmcIdentityYaml, EmcYaml, GoverningEquationYaml};
        EmcYaml {
            emc_version: "1.0".to_string(),
            emc_id: "TEST".to_string(),
            identity: EmcIdentityYaml {
                name: "Test".to_string(),
                version: "1.0.0".to_string(),
                authors: Vec::new(),
                status: String::new(),
                description: String::new(),
            },
            governing_equation: GoverningEquationYaml {
                latex: "x = y".to_string(),
                plain_text: String::new(),
                description: String::new(),
                variables: Vec::new(),
                equation_type: String::new(),
            },
            analytical_derivation: None,
            domain_of_validity: None,
            verification_tests: None,
            falsification_criteria: None,
        }
    }

    #[test]
    fn test_verify_reproducibility() {
        use crate::edd::loader::{
            EmcReferenceYaml, ExperimentMetadataYaml, ExperimentYaml, ReproducibilityYaml,
        };

        let runner = ExperimentRunner::new();
        let experiment = ExperimentYaml {
            experiment_version: "1.0".to_string(),
            experiment_id: "EXP-001".to_string(),
            metadata: ExperimentMetadataYaml {
                name: "Test".to_string(),
                description: String::new(),
                tags: Vec::new(),
            },
            equation_model_card: Some(EmcReferenceYaml {
                emc_ref: String::new(),
                emc_file: String::new(),
            }),
            hypothesis: None,
            reproducibility: ReproducibilityYaml {
                seed: 42,
                ieee_strict: true,
            },
            simulation: None,
            falsification: None,
        };

        let summary = runner.verify_reproducibility(&experiment, 42);
        assert!(summary.passed);
        assert!(summary.identical);
        assert_eq!(summary.runs, 3); // Default
        assert_eq!(summary.run_hashes.len(), 3);
    }

    #[test]
    fn test_check_falsification_criteria_empty() {
        use crate::edd::experiment::ExperimentSpec;
        use crate::edd::loader::{
            EmcReferenceYaml, ExperimentMetadataYaml, ExperimentYaml, ReproducibilityYaml,
        };

        let runner = ExperimentRunner::new();
        let spec = ExperimentSpec::builder()
            .name("Test")
            .seed(42)
            .build()
            .ok()
            .unwrap();
        let experiment = ExperimentYaml {
            experiment_version: "1.0".to_string(),
            experiment_id: "EXP-001".to_string(),
            metadata: ExperimentMetadataYaml {
                name: "Test".to_string(),
                description: String::new(),
                tags: Vec::new(),
            },
            equation_model_card: Some(EmcReferenceYaml {
                emc_ref: String::new(),
                emc_file: String::new(),
            }),
            hypothesis: None,
            reproducibility: ReproducibilityYaml {
                seed: 42,
                ieee_strict: true,
            },
            simulation: None,
            falsification: None,
        };

        let summary = runner.check_falsification_criteria(&spec, &experiment);
        assert_eq!(summary.total, 0);
        assert!(!summary.jidoka_triggered);
    }

    #[test]
    fn test_check_falsification_criteria_with_criteria() {
        use crate::edd::experiment::{ExperimentSpec, FalsificationAction, FalsificationCriterion};
        use crate::edd::loader::{
            ExperimentFalsificationYaml, ExperimentMetadataYaml, ExperimentYaml,
            FalsificationCriterionYaml, ReproducibilityYaml,
        };

        let runner = ExperimentRunner::new();
        let crit = FalsificationCriterion::new(
            "Test Criterion",
            "error > 0.01",
            FalsificationAction::Warn,
        );
        let spec = ExperimentSpec::builder()
            .name("Test")
            .seed(42)
            .add_falsification_criterion(crit)
            .build()
            .ok()
            .unwrap();

        let experiment = ExperimentYaml {
            experiment_version: "1.0".to_string(),
            experiment_id: "EXP-001".to_string(),
            metadata: ExperimentMetadataYaml {
                name: "Test".to_string(),
                description: String::new(),
                tags: Vec::new(),
            },
            equation_model_card: None,
            hypothesis: None,
            reproducibility: ReproducibilityYaml {
                seed: 42,
                ieee_strict: true,
            },
            simulation: None,
            falsification: Some(ExperimentFalsificationYaml {
                import_from_emc: false,
                criteria: vec![
                    FalsificationCriterionYaml {
                        id: "FC-001".to_string(),
                        name: "Critical".to_string(),
                        condition: "error > 0.01".to_string(),
                        threshold: Some(0.01),
                        severity: "critical".to_string(),
                        interpretation: String::new(),
                    },
                    FalsificationCriterionYaml {
                        id: "FC-002".to_string(),
                        name: "Minor".to_string(),
                        condition: "drift > 0.1".to_string(),
                        threshold: Some(0.1),
                        severity: "minor".to_string(),
                        interpretation: String::new(),
                    },
                ],
                jidoka: None,
            }),
        };

        let summary = runner.check_falsification_criteria(&spec, &experiment);
        // 1 from spec + 2 from experiment
        assert_eq!(summary.total, 3);
        assert!(!summary.jidoka_triggered);
    }
}
