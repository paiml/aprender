//! Playbook types — all serde types from spec §7.2
//!
//! These types represent the YAML schema for deterministic pipeline orchestration.
//! Types for Phase 2+ features (parallel, retry, resources, compliance) are defined
//! here so YAML with those features parses correctly, but they are not executed
//! until their respective phases.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

// ============================================================================
// Playbook root
// ============================================================================

/// Top-level playbook definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Playbook {
    /// Schema version (must be "1.0")
    pub version: String,

    /// Human-readable pipeline name
    pub name: String,

    /// Optional description
    #[serde(default)]
    pub description: Option<String>,

    /// Global parameters (key-value pairs, supports strings and numbers)
    #[serde(default)]
    pub params: HashMap<String, serde_yaml_ng::Value>,

    /// Named execution targets (machines)
    #[serde(default)]
    pub targets: HashMap<String, Target>,

    /// Pipeline stages (order-preserving)
    pub stages: IndexMap<String, Stage>,

    /// Compliance gates (parsed, not executed in Phase 1)
    #[serde(default)]
    pub compliance: Option<Compliance>,

    /// Execution policy
    #[serde(default)]
    pub policy: Policy,
}

// ============================================================================
// Stages
// ============================================================================

/// A single pipeline stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage {
    /// Human-readable description
    #[serde(default)]
    pub description: Option<String>,

    /// Shell command (supports template substitution)
    pub cmd: String,

    /// Input dependencies
    #[serde(default)]
    pub deps: Vec<Dependency>,

    /// Output artifacts
    #[serde(default)]
    pub outs: Vec<Output>,

    /// Explicit ordering constraints (stage names)
    #[serde(default)]
    pub after: Vec<String>,

    /// Execution target name (from playbook.targets)
    #[serde(default)]
    pub target: Option<String>,

    /// Param keys this stage depends on (for granular invalidation)
    #[serde(default)]
    pub params: Option<Vec<String>>,

    /// Parallel fan-out configuration (Phase 2)
    #[serde(default)]
    pub parallel: Option<ParallelConfig>,

    /// Retry configuration (Phase 2)
    #[serde(default)]
    pub retry: Option<RetryConfig>,

    /// Resource requirements (Phase 4)
    #[serde(default)]
    pub resources: Option<ResourceConfig>,

    /// Frozen stage — never re-execute (Phase 4)
    #[serde(default)]
    pub frozen: bool,

    /// Shell mode override
    #[serde(default)]
    pub shell: Option<ShellMode>,
}

/// Input dependency reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    /// File or directory path
    pub path: String,

    /// Dependency type (e.g., "file", "directory")
    #[serde(rename = "type", default)]
    pub dep_type: Option<String>,
}

/// Output artifact reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Output {
    /// File or directory path
    pub path: String,

    /// Output type
    #[serde(rename = "type", default)]
    pub out_type: Option<String>,

    /// Remote target name (if output is on a remote machine)
    #[serde(default)]
    pub remote: Option<String>,
}

// ============================================================================
// Targets
// ============================================================================

/// Named execution target (machine)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Target {
    /// Hostname (e.g., "localhost", "gpu-box.local")
    #[serde(default)]
    pub host: Option<String>,

    /// SSH user for remote targets
    #[serde(default)]
    pub ssh_user: Option<String>,

    /// CPU cores available
    #[serde(default)]
    pub cores: Option<u32>,

    /// Memory in GB
    #[serde(default)]
    pub memory_gb: Option<u32>,

    /// Working directory on target
    #[serde(default)]
    pub workdir: Option<String>,

    /// Environment variables
    #[serde(default)]
    pub env: HashMap<String, String>,
}

// ============================================================================
// Policy
// ============================================================================

/// Failure handling policy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum FailurePolicy {
    /// Stop pipeline on first stage failure (Jidoka)
    #[default]
    StopOnFirst,
    /// Continue running independent stages
    ContinueIndependent,
}

/// Validation policy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum ValidationPolicy {
    /// BLAKE3 checksum validation
    #[default]
    Checksum,
    /// No validation
    None,
}

/// Concurrency policy for lock file access
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum ConcurrencyPolicy {
    /// Wait for lock
    #[default]
    Wait,
    /// Fail if locked
    Fail,
}

/// Execution policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// Failure handling strategy
    #[serde(default)]
    pub failure: FailurePolicy,

    /// Validation strategy
    #[serde(default)]
    pub validation: ValidationPolicy,

    /// Whether to maintain a lock file
    #[serde(default = "Policy::default_lock_file")]
    pub lock_file: bool,

    /// Concurrency policy (Phase 2)
    #[serde(default)]
    pub concurrency: Option<ConcurrencyPolicy>,

    /// Working directory isolation mode (Phase 2)
    #[serde(default)]
    pub work_dir: Option<PathBuf>,

    /// Clean work directory on success (Phase 2)
    #[serde(default)]
    pub clean_on_success: Option<bool>,
}

impl Policy {
    fn default_lock_file() -> bool {
        true
    }
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            failure: FailurePolicy::default(),
            validation: ValidationPolicy::default(),
            lock_file: Self::default_lock_file(),
            concurrency: None,
            work_dir: None,
            clean_on_success: None,
        }
    }
}

// ============================================================================
// Phase 2+ types (parsed, not executed)
// ============================================================================

/// Parallel fan-out configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Strategy (e.g., "per_file")
    pub strategy: String,

    /// Glob pattern for file discovery
    #[serde(default)]
    pub glob: Option<String>,

    /// Maximum concurrent workers
    #[serde(default)]
    pub max_workers: Option<u32>,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    #[serde(default = "RetryConfig::default_limit")]
    pub limit: u32,

    /// Retry policy
    #[serde(default = "RetryConfig::default_policy")]
    pub policy: String,

    /// Backoff configuration
    #[serde(default)]
    pub backoff: Option<BackoffConfig>,
}

impl RetryConfig {
    fn default_limit() -> u32 {
        3
    }
    fn default_policy() -> String {
        "on_failure".to_string()
    }
}

/// Exponential backoff configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackoffConfig {
    /// Initial delay in seconds
    #[serde(default = "BackoffConfig::default_initial")]
    pub initial_seconds: f64,

    /// Backoff multiplier
    #[serde(default = "BackoffConfig::default_multiplier")]
    pub multiplier: f64,

    /// Maximum delay in seconds
    #[serde(default = "BackoffConfig::default_max")]
    pub max_seconds: f64,
}

impl BackoffConfig {
    fn default_initial() -> f64 {
        1.0
    }
    fn default_multiplier() -> f64 {
        2.0
    }
    fn default_max() -> f64 {
        60.0
    }
}

/// Resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// CPU cores
    #[serde(default)]
    pub cores: Option<u32>,

    /// Memory in GB
    #[serde(default)]
    pub memory_gb: Option<f64>,

    /// GPU devices (0 = CPU only)
    #[serde(default)]
    pub gpu: Option<u32>,

    /// Timeout in seconds
    #[serde(default)]
    pub timeout: Option<u64>,
}

/// Shell execution mode
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShellMode {
    /// Purified via bashrs/rash (Phase 2)
    Rash,
    /// Raw sh -c execution
    Raw,
}

// ============================================================================
// Compliance (parsed, not executed in Phase 1)
// ============================================================================

/// PMAT compliance gate configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Compliance {
    /// Gates to run before pipeline stages
    #[serde(default)]
    pub pre_flight: Vec<ComplianceCheck>,

    /// Gates to run after pipeline stages
    #[serde(default)]
    pub post_flight: Vec<ComplianceCheck>,
}

/// A single compliance check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCheck {
    /// Check type (e.g., "tdg", "quality_gate", "coverage")
    #[serde(rename = "type")]
    pub check_type: String,

    /// Minimum acceptable grade
    #[serde(default)]
    pub min_grade: Option<String>,

    /// Target path
    #[serde(default)]
    pub path: Option<String>,

    /// Minimum threshold value
    #[serde(default)]
    pub min: Option<f64>,
}

// ============================================================================
// Lock file types
// ============================================================================

/// Lock file representing cached pipeline state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockFile {
    /// Schema version
    pub schema: String,

    /// Playbook name
    pub playbook: String,

    /// When the lock was generated
    pub generated_at: String,

    /// Generator version string
    pub generator: String,

    /// BLAKE3 version used
    pub blake3_version: String,

    /// Hash of global params
    #[serde(default)]
    pub params_hash: Option<String>,

    /// Per-stage lock data (order-preserving for deterministic YAML output)
    pub stages: IndexMap<String, StageLock>,
}

/// Per-stage lock data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageLock {
    /// Completion status
    pub status: StageStatus,

    /// When the stage started
    #[serde(default)]
    pub started_at: Option<String>,

    /// When the stage completed
    #[serde(default)]
    pub completed_at: Option<String>,

    /// Duration in seconds
    #[serde(default)]
    pub duration_seconds: Option<f64>,

    /// Target name
    #[serde(default)]
    pub target: Option<String>,

    /// Dependency hashes
    #[serde(default)]
    pub deps: Vec<DepLock>,

    /// Parameters hash for this stage
    #[serde(default)]
    pub params_hash: Option<String>,

    /// Output hashes
    #[serde(default)]
    pub outs: Vec<OutLock>,

    /// Hash of the resolved command
    #[serde(default)]
    pub cmd_hash: Option<String>,

    /// Composite cache key
    #[serde(default)]
    pub cache_key: Option<String>,
}

/// Stage completion status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    Completed,
    Failed,
    Cached,
    Running,
    Pending,
    Hashing,
    Validating,
}

/// Dependency hash entry in lock file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepLock {
    pub path: String,
    pub hash: String,
    #[serde(default)]
    pub file_count: Option<u64>,
    #[serde(default)]
    pub total_bytes: Option<u64>,
}

/// Output hash entry in lock file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutLock {
    pub path: String,
    pub hash: String,
    #[serde(default)]
    pub file_count: Option<u64>,
    #[serde(default)]
    pub total_bytes: Option<u64>,
    /// Remote target name (if output is on remote machine)
    #[serde(default)]
    pub remote: Option<String>,
}

// ============================================================================
// Pipeline events (JSONL event log)
// ============================================================================

/// Pipeline execution event
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum PipelineEvent {
    RunStarted {
        playbook: String,
        run_id: String,
        batuta_version: String,
    },
    RunCompleted {
        playbook: String,
        run_id: String,
        stages_run: u32,
        stages_cached: u32,
        stages_failed: u32,
        total_seconds: f64,
    },
    RunFailed {
        playbook: String,
        run_id: String,
        error: String,
    },
    StageCached {
        stage: String,
        cache_key: String,
        reason: String,
    },
    StageStarted {
        stage: String,
        target: String,
        cache_miss_reason: String,
    },
    StageCompleted {
        stage: String,
        duration_seconds: f64,
        #[serde(default)]
        outs_hash: Option<String>,
    },
    StageFailed {
        stage: String,
        exit_code: Option<i32>,
        error: String,
        #[serde(default)]
        retry_attempt: Option<u32>,
    },
}

/// Timestamped event wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedEvent {
    /// ISO 8601 timestamp
    pub ts: String,

    /// The event payload (flattened)
    #[serde(flatten)]
    pub event: PipelineEvent,
}

// ============================================================================
// Cache invalidation
// ============================================================================

/// Reason why a stage cache was invalidated
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidationReason {
    /// No lock file exists
    NoLockFile,
    /// Stage not found in lock file
    StageNotInLock,
    /// Previous run did not complete successfully
    PreviousRunIncomplete { status: String },
    /// Command changed
    CmdChanged { old: String, new: String },
    /// Dependency hash changed
    DepChanged { path: String, old_hash: String, new_hash: String },
    /// Parameters hash changed
    ParamsChanged { old: String, new: String },
    /// Cache key mismatch
    CacheKeyMismatch { old: String, new: String },
    /// Output file missing
    OutputMissing { path: String },
    /// Forced re-run
    Forced,
    /// Upstream stage was re-run
    UpstreamRerun { stage: String },
}

impl fmt::Display for InvalidationReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoLockFile => write!(f, "no lock file found"),
            Self::StageNotInLock => write!(f, "stage not found in lock file"),
            Self::PreviousRunIncomplete { status } => {
                write!(f, "previous run status: {}", status)
            }
            Self::CmdChanged { old, new } => {
                write!(f, "cmd_hash changed: {} → {}", old, new)
            }
            Self::DepChanged { path, old_hash, new_hash } => {
                write!(f, "dep '{}' hash changed: {} → {}", path, old_hash, new_hash)
            }
            Self::ParamsChanged { old, new } => {
                write!(f, "params_hash changed: {} → {}", old, new)
            }
            Self::CacheKeyMismatch { old, new } => {
                write!(f, "cache_key mismatch: {} → {}", old, new)
            }
            Self::OutputMissing { path } => {
                write!(f, "output '{}' is missing", path)
            }
            Self::Forced => write!(f, "forced re-run (--force)"),
            Self::UpstreamRerun { stage } => {
                write!(f, "upstream stage '{}' was re-run", stage)
            }
        }
    }
}

// ============================================================================
// Validation
// ============================================================================

/// Validation warning (non-fatal)
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub message: String,
}

impl fmt::Display for ValidationWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Convert a serde_yaml_ng::Value to a string for template resolution
pub fn yaml_value_to_string(val: &serde_yaml_ng::Value) -> String {
    match val {
        serde_yaml_ng::Value::String(s) => s.clone(),
        serde_yaml_ng::Value::Number(n) => n.to_string(),
        serde_yaml_ng::Value::Bool(b) => b.to_string(),
        serde_yaml_ng::Value::Null => String::new(),
        other => format!("{:?}", other),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn test_PB001_playbook_serde_roundtrip() {
        let yaml = r#"
version: "1.0"
name: test-pipeline
params:
  model: "base"
  chunk_size: 512
targets: {}
stages:
  hello:
    cmd: "echo hello"
    deps: []
    outs:
      - path: /tmp/out.txt
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#;
        let pb: Playbook = serde_yaml_ng::from_str(yaml).expect("yaml deserialize failed");
        assert_eq!(pb.version, "1.0");
        assert_eq!(pb.name, "test-pipeline");
        assert_eq!(yaml_value_to_string(pb.params.get("model").expect("key not found")), "base");
        // Numeric params now work
        assert_eq!(
            yaml_value_to_string(pb.params.get("chunk_size").expect("key not found")),
            "512"
        );
        assert_eq!(pb.stages.len(), 1);
        assert!(pb.stages.contains_key("hello"));
    }

    #[test]
    fn test_PB001_numeric_params() {
        let yaml = r#"
version: "1.0"
name: numeric
params:
  chunk_size: 512
  bm25_weight: 0.3
  enabled: true
targets: {}
stages:
  test:
    cmd: "echo test"
    deps: []
    outs: []
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#;
        let pb: Playbook = serde_yaml_ng::from_str(yaml).expect("yaml deserialize failed");
        assert_eq!(
            yaml_value_to_string(pb.params.get("chunk_size").expect("key not found")),
            "512"
        );
        assert_eq!(
            yaml_value_to_string(pb.params.get("bm25_weight").expect("key not found")),
            "0.3"
        );
        assert_eq!(yaml_value_to_string(pb.params.get("enabled").expect("key not found")), "true");
    }

    #[test]
    fn test_PB001_stage_defaults() {
        let yaml = r#"
cmd: "echo test"
deps: []
outs: []
"#;
        let stage: Stage = serde_yaml_ng::from_str(yaml).expect("yaml deserialize failed");
        assert!(stage.description.is_none());
        assert!(stage.target.is_none());
        assert!(stage.after.is_empty());
        assert!(stage.params.is_none());
        assert!(stage.parallel.is_none());
        assert!(stage.retry.is_none());
        assert!(stage.resources.is_none());
        assert!(!stage.frozen);
        assert!(stage.shell.is_none());
    }

    #[test]
    fn test_PB001_stage_params_list() {
        let yaml = r#"
cmd: "echo {{params.model}}"
deps: []
outs: []
params:
  - model
  - chunk_size
"#;
        let stage: Stage = serde_yaml_ng::from_str(yaml).expect("yaml deserialize failed");
        let params = stage.params.expect("unexpected failure");
        assert_eq!(params, vec!["model", "chunk_size"]);
    }

    #[test]
    fn test_PB001_policy_defaults() {
        let policy = Policy::default();
        assert_eq!(policy.failure, FailurePolicy::StopOnFirst);
        assert_eq!(policy.validation, ValidationPolicy::Checksum);
        assert!(policy.lock_file);
        assert!(policy.concurrency.is_none());
    }

    #[test]
    fn test_PB001_policy_enum_serde() {
        let yaml = r#"
failure: stop_on_first
validation: checksum
lock_file: true
"#;
        let policy: Policy = serde_yaml_ng::from_str(yaml).expect("yaml deserialize failed");
        assert_eq!(policy.failure, FailurePolicy::StopOnFirst);

        let yaml2 = r#"
failure: continue_independent
validation: none
lock_file: false
"#;
        let policy2: Policy = serde_yaml_ng::from_str(yaml2).expect("yaml deserialize failed");
        assert_eq!(policy2.failure, FailurePolicy::ContinueIndependent);
        assert_eq!(policy2.validation, ValidationPolicy::None);
        assert!(!policy2.lock_file);
    }

    #[test]
    fn test_PB001_stage_with_phase2_fields() {
        let yaml = r#"
cmd: "echo test"
deps: []
outs: []
parallel:
  strategy: per_file
  glob: "*.txt"
  max_workers: 4
retry:
  limit: 3
  policy: on_failure
resources:
  cores: 4
  memory_gb: 8.0
  gpu: 2
  timeout: 3600
"#;
        let stage: Stage = serde_yaml_ng::from_str(yaml).expect("yaml deserialize failed");
        let par = stage.parallel.expect("unexpected failure");
        assert_eq!(par.strategy, "per_file");
        assert_eq!(par.glob.expect("unexpected failure"), "*.txt");
        assert_eq!(par.max_workers.expect("unexpected failure"), 4);

        let retry = stage.retry.expect("unexpected failure");
        assert_eq!(retry.limit, 3);

        let res = stage.resources.expect("unexpected failure");
        assert_eq!(res.cores.expect("unexpected failure"), 4);
        assert_eq!(res.memory_gb.expect("unexpected failure"), 8.0);
        assert_eq!(res.gpu.expect("unexpected failure"), 2);
        assert_eq!(res.timeout.expect("unexpected failure"), 3600);
    }

    #[test]
    fn test_PB001_lock_file_serde_roundtrip() {
        let lock = LockFile {
            schema: "1.0".to_string(),
            playbook: "test".to_string(),
            generated_at: "2026-02-16T14:00:00Z".to_string(),
            generator: "batuta 0.6.5".to_string(),
            blake3_version: "1.8".to_string(),
            params_hash: Some("blake3:abc123".to_string()),
            stages: IndexMap::from([(
                "hello".to_string(),
                StageLock {
                    status: StageStatus::Completed,
                    started_at: Some("2026-02-16T14:00:00Z".to_string()),
                    completed_at: Some("2026-02-16T14:00:01Z".to_string()),
                    duration_seconds: Some(1.0),
                    target: None,
                    deps: vec![DepLock {
                        path: "/tmp/in.txt".to_string(),
                        hash: "blake3:def456".to_string(),
                        file_count: Some(1),
                        total_bytes: Some(100),
                    }],
                    params_hash: Some("blake3:aaa".to_string()),
                    outs: vec![OutLock {
                        path: "/tmp/out.txt".to_string(),
                        hash: "blake3:ghi789".to_string(),
                        file_count: Some(1),
                        total_bytes: Some(200),
                        remote: None,
                    }],
                    cmd_hash: Some("blake3:cmd111".to_string()),
                    cache_key: Some("blake3:key222".to_string()),
                },
            )]),
        };

        let yaml = serde_yaml_ng::to_string(&lock).expect("yaml serialize failed");
        let lock2: LockFile = serde_yaml_ng::from_str(&yaml).expect("yaml deserialize failed");
        assert_eq!(lock2.playbook, "test");
        assert_eq!(lock2.stages["hello"].status, StageStatus::Completed);
    }

    #[test]
    fn test_PB001_stage_status_serde() {
        let statuses = vec![
            (StageStatus::Completed, "\"completed\""),
            (StageStatus::Failed, "\"failed\""),
            (StageStatus::Cached, "\"cached\""),
            (StageStatus::Running, "\"running\""),
            (StageStatus::Pending, "\"pending\""),
            (StageStatus::Hashing, "\"hashing\""),
            (StageStatus::Validating, "\"validating\""),
        ];
        for (status, expected) in statuses {
            let json = serde_json::to_string(&status).expect("json serialize failed");
            assert_eq!(json, expected);
            let parsed: StageStatus = serde_json::from_str(&json).expect("json deserialize failed");
            assert_eq!(parsed, status);
        }
    }

    #[test]
    fn test_PB001_invalidation_reason_display() {
        assert_eq!(InvalidationReason::NoLockFile.to_string(), "no lock file found");
        assert_eq!(InvalidationReason::Forced.to_string(), "forced re-run (--force)");
        assert_eq!(
            InvalidationReason::PreviousRunIncomplete { status: "failed".to_string() }.to_string(),
            "previous run status: failed"
        );
    }

    #[test]
    fn test_PB001_pipeline_event_serde() {
        let event = PipelineEvent::RunStarted {
            playbook: "test".to_string(),
            run_id: "r-abc123".to_string(),
            batuta_version: "0.6.5".to_string(),
        };
        let json = serde_json::to_string(&event).expect("json serialize failed");
        assert!(json.contains("\"event\":\"run_started\""));
        assert!(json.contains("\"run_id\":\"r-abc123\""));
    }

    #[test]
    fn test_PB001_run_completed_has_stages_failed() {
        let event = PipelineEvent::RunCompleted {
            playbook: "test".to_string(),
            run_id: "r-abc".to_string(),
            stages_run: 3,
            stages_cached: 1,
            stages_failed: 1,
            total_seconds: 5.0,
        };
        let json = serde_json::to_string(&event).expect("json serialize failed");
        assert!(json.contains("\"stages_failed\":1"));
        assert!(json.contains("\"total_seconds\":5.0"));
    }

    #[test]
    fn test_PB001_timestamped_event_serde() {
        let te = TimestampedEvent {
            ts: "2026-02-16T14:00:00Z".to_string(),
            event: PipelineEvent::StageCached {
                stage: "hello".to_string(),
                cache_key: "blake3:abc".to_string(),
                reason: "cache_key matches lock".to_string(),
            },
        };
        let json = serde_json::to_string(&te).expect("json serialize failed");
        assert!(json.contains("\"ts\":"));
        assert!(json.contains("\"event\":\"stage_cached\""));
    }

    #[test]
    fn test_PB001_compliance_parse() {
        let yaml = r#"
pre_flight:
  - type: tdg
    min_grade: B
    path: src/
post_flight:
  - type: coverage
    min: 85.0
"#;
        let compliance: Compliance =
            serde_yaml_ng::from_str(yaml).expect("yaml deserialize failed");
        assert_eq!(compliance.pre_flight.len(), 1);
        assert_eq!(compliance.pre_flight[0].check_type, "tdg");
        assert_eq!(compliance.post_flight.len(), 1);
        assert_eq!(compliance.post_flight[0].min.expect("unexpected failure"), 85.0);
    }

    #[test]
    fn test_PB001_indexmap_preserves_stage_order() {
        let yaml = r#"
version: "1.0"
name: ordered
params: {}
targets: {}
stages:
  alpha:
    cmd: "echo alpha"
    deps: []
    outs: []
  beta:
    cmd: "echo beta"
    deps: []
    outs: []
  gamma:
    cmd: "echo gamma"
    deps: []
    outs: []
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#;
        let pb: Playbook = serde_yaml_ng::from_str(yaml).expect("yaml deserialize failed");
        let keys: Vec<&String> = pb.stages.keys().collect();
        assert_eq!(keys, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn test_PB001_target_with_spec_fields() {
        let yaml = r#"
host: "gpu-box.local"
ssh_user: noah
cores: 32
memory_gb: 288
workdir: "/data/pipeline"
env:
  CUDA_VISIBLE_DEVICES: "0,1"
"#;
        let target: Target = serde_yaml_ng::from_str(yaml).expect("yaml deserialize failed");
        assert_eq!(target.host.as_deref(), Some("gpu-box.local"));
        assert_eq!(target.ssh_user.as_deref(), Some("noah"));
        assert_eq!(target.cores, Some(32));
        assert_eq!(target.memory_gb, Some(288));
    }

    #[test]
    fn test_PB001_dep_and_output_with_type() {
        let yaml = r#"
path: /data/input.wav
type: file
"#;
        let dep: Dependency = serde_yaml_ng::from_str(yaml).expect("yaml deserialize failed");
        assert_eq!(dep.path, "/data/input.wav");
        assert_eq!(dep.dep_type.as_deref(), Some("file"));

        let yaml2 = r#"
path: /data/output/
type: directory
remote: intel
"#;
        let out: Output = serde_yaml_ng::from_str(yaml2).expect("yaml deserialize failed");
        assert_eq!(out.path, "/data/output/");
        assert_eq!(out.remote.as_deref(), Some("intel"));
    }
}
