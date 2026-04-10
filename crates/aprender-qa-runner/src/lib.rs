//! APR QA Runner
//!
//! Playbook executor for model qualification testing.
//! Implements parallel execution with Jidoka (stop-on-failure) support.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
// Allow common patterns
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::unused_self)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::single_char_pattern)]
// Allow common patterns in test code
#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::panic,
        clippy::assertions_on_constants
    )
)]
#![cfg_attr(test, allow(clippy::redundant_closure_for_method_calls))]
#![cfg_attr(test, allow(clippy::redundant_clone))]
#![cfg_attr(test, allow(clippy::uninlined_format_args))]
#![cfg_attr(test, allow(clippy::cast_sign_loss))]

pub mod command;
pub mod contract;
pub mod conversion;
pub mod diagnostics;
pub mod differential;
pub mod dimensional_check;
pub mod error;
pub mod evidence;
pub mod executor;
pub mod family_contract;
pub mod integrity;
pub mod layout_contract;
pub mod oracle;
pub mod parallel;
pub mod patterns;
pub mod playbook;
pub mod process;
pub mod provenance;
pub use provenance::{
    add_derived, compute_sha256, create_source_provenance, get_apr_cli_version, load_provenance,
    save_provenance, validate_comparison, validate_provenance, verify_files_exist,
    verify_provenance_integrity, DerivedProvenance, Provenance, ProvenanceError, SourceProvenance,
};

#[cfg(test)]
pub mod test_fixtures;

pub use command::{CommandOutput, CommandRunner, MockCommandRunner, RealCommandRunner};
pub use contract::{
    load_format_contract, lookup_tolerance, run_contract_tests, validate_dtype_bytes,
    validate_tensor_name, ContractTestConfig, DtypeByteEntry, DtypeByteSection, FormatContract,
    InvariantDef, InvariantId, NamingExample, TensorNamingContract, ToleranceEntry,
};
pub use conversion::{
    all_backends, all_conversion_pairs, check_cardinality, check_tensor_names, classify_failure,
    generate_conversion_tests, get_hf_cache_dir, resolve_hf_repo_to_cache, resolve_model_path,
    split_hf_repo, tolerance_for, ByteLevelRoundTripTest, CommutativityTest, ConversionBugType,
    ConversionConfig, ConversionEvidence, ConversionExecutionResult, ConversionExecutor,
    ConversionFailureType, ConversionOutputDir, ConversionResult, ConversionTest,
    ConversionTolerance, IdempotencyTest, QuantType, RoundTripTest, SemanticConversionTest,
    SemanticTestResult, TensorNaming, DEFAULT_TOLERANCES, EPSILON,
};
pub use diagnostics::{
    DiagnosticResult, DiagnosticsBundle, EnvironmentContext, FailFastReport, FailFastReporter,
    FailureDetails, ReproductionInfo,
};
pub use differential::{
    convert_format_cached, prepare_model_with_provenance, run_bench_throughput, run_diff_benchmark,
    run_inspect, run_profile_ci, run_six_column_profile, verify_comparison_provenance, BenchResult,
    BenchmarkMetrics, CiAssertion, CiProfileResult, DiffBenchmarkResult, DiffConfig,
    DifferentialExecutor, FormatConversionResult, InferenceComparisonResult, InspectResult,
    ModelPreparationResult, ProfileAssertion, SixColumnProfile, TensorDiffResult, TensorMismatch,
    TensorMismatchType, TokenComparison,
};
pub use dimensional_check::{run_dimensional_check, DimensionalCheck, DimensionalCheckResult};
pub use error::{Error, Result};
pub use evidence::{Evidence, EvidenceCollector, Outcome, PerformanceMetrics};
pub use executor::{
    ExecutionConfig, ExecutionResult, Executor, FailurePolicy, ToolExecutor, ToolTestResult,
};
pub use integrity::{
    check_safetensors_integrity, gate_ids as integrity_gate_ids, ConfigValues, IntegrityResult,
    TensorDerivedValues,
};
pub use layout_contract::{
    find_and_load_config, find_safetensors_files, get_critical_tensors, get_validation_rules,
    load_contract, load_contract_from, read_safetensors_metadata, validate_model,
    LayoutModelConfig, ModelValidationResult, TensorLayoutContract, TensorSpec,
    TensorValidationResult, ValidationRule,
};
pub use oracle::{
    generate_checklist_markdown, CheckStatus, Confidence, CrossReference, FalsificationCheckItem,
    OracleContext, OracleEnhancer, OracleError, RankedHypothesis,
};
pub use parallel::{ParallelConfig, ParallelExecutor, ParallelResult};
pub use patterns::{
    ApiComplianceChecker, ApiComplianceResult, BugPattern, CompanionCheckResult, DosCheckResult,
    DosProtectionConfig, DosViolation, IntegrityCheckResult, IntegrityChecker,
    NumericalStabilityResult, ParityCheckResult, ParityChecker, PathSafetyResult, PathViolation,
    PatternDetector, PerformanceCheckResult, PerformanceThresholds, PerformanceValidator,
    PromptPattern, PromptSafetyResult, SpecGate, TensorValidityResult,
};
pub use playbook::{
    compute_playbook_hash, detect_implicit_skips, find_skip_files, generate_lock_entry,
    load_lock_file, save_lock_file, verify_playbook_integrity, DifferentialTestConfig,
    DistillConfig, FingerprintConfig, FormatValidationConfig, ImportConfig, InferenceCompareConfig,
    OllamaParityConfig, Playbook, PlaybookLockEntry, PlaybookLockFile, PlaybookStep,
    ProfileCiAssertions, ProfileCiConfig, PruneConfig, QuantizeConfig, SkipReason, SkipType,
    StatsToleranceConfig, TensorDiffConfig, TracePayloadConfig, TransformationConfig,
    ValidateStatsConfig,
};
