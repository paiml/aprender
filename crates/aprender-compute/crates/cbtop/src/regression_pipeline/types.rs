//! Core types for the regression pipeline.

/// Result type for pipeline operations
pub type PipelineResult<T> = Result<T, PipelineError>;

/// Errors in pipeline operations
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineError {
    /// Git operation failed
    GitError { reason: String },
    /// Benchmark execution failed
    BenchmarkFailed { reason: String },
    /// Baseline not found
    BaselineNotFound { commit: String },
    /// Invalid configuration
    InvalidConfig { reason: String },
    /// Timeout waiting for results
    Timeout { timeout_sec: u64 },
    /// PR status update failed
    StatusUpdateFailed { reason: String },
    /// Artifact storage failed
    ArtifactError { reason: String },
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GitError { reason } => write!(f, "Git error: {}", reason),
            Self::BenchmarkFailed { reason } => write!(f, "Benchmark failed: {}", reason),
            Self::BaselineNotFound { commit } => write!(f, "Baseline not found for {}", commit),
            Self::InvalidConfig { reason } => write!(f, "Invalid config: {}", reason),
            Self::Timeout { timeout_sec } => write!(f, "Timeout after {}s", timeout_sec),
            Self::StatusUpdateFailed { reason } => write!(f, "Status update failed: {}", reason),
            Self::ArtifactError { reason } => write!(f, "Artifact error: {}", reason),
        }
    }
}

impl std::error::Error for PipelineError {}

/// Pipeline execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStatus {
    /// Pipeline not started
    Pending,
    /// Pipeline is running
    Running,
    /// Pipeline completed successfully (no regressions)
    Passed,
    /// Pipeline completed with warnings (minor regressions)
    Warning,
    /// Pipeline failed (significant regressions)
    Failed,
    /// Pipeline was cancelled
    Cancelled,
    /// Pipeline encountered an error
    Error,
}

impl PipelineStatus {
    /// Check if status is terminal
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Passed | Self::Warning | Self::Failed | Self::Cancelled | Self::Error)
    }

    /// Get status name for GitHub
    pub fn github_state(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "pending",
            Self::Passed => "success",
            Self::Warning => "success",
            Self::Failed => "failure",
            Self::Cancelled => "error",
            Self::Error => "error",
        }
    }
}

/// Git reference type
#[derive(Debug, Clone)]
pub enum GitRef {
    /// Branch name
    Branch(String),
    /// Commit SHA
    Commit(String),
    /// Tag name
    Tag(String),
    /// Pull request number
    PullRequest(u64),
}

impl GitRef {
    /// Get ref string for git commands
    pub fn as_ref_str(&self) -> String {
        match self {
            Self::Branch(name) => name.clone(),
            Self::Commit(sha) => sha.clone(),
            Self::Tag(name) => format!("refs/tags/{}", name),
            Self::PullRequest(num) => format!("refs/pull/{}/head", num),
        }
    }
}

/// Configuration for the regression pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Base branch for comparison (default: main)
    pub base_branch: String,
    /// Benchmark command to run
    pub benchmark_command: String,
    /// Working directory
    pub work_dir: String,
    /// Maximum execution time in seconds
    pub timeout_sec: u64,
    /// Regression threshold (percent)
    pub regression_threshold_percent: f64,
    /// Warning threshold (percent)
    pub warning_threshold_percent: f64,
    /// GitHub token for status updates (optional)
    pub github_token: Option<String>,
    /// Repository name (owner/repo)
    pub repository: Option<String>,
    /// Artifact storage path
    pub artifact_path: String,
    /// Number of benchmark iterations
    pub iterations: u32,
    /// Warmup iterations
    pub warmup_iterations: u32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            base_branch: "main".to_string(),
            benchmark_command: "cargo bench --no-fail-fast".to_string(),
            work_dir: ".".to_string(),
            timeout_sec: 600,
            regression_threshold_percent: 5.0,
            warning_threshold_percent: 2.0,
            github_token: None,
            repository: None,
            artifact_path: "./benchmark-artifacts".to_string(),
            iterations: 10,
            warmup_iterations: 3,
        }
    }
}

/// PR status check result
#[derive(Debug, Clone)]
pub struct StatusCheck {
    /// Check name
    pub name: String,
    /// Status state
    pub state: PipelineStatus,
    /// Description
    pub description: String,
    /// Target URL for details
    pub target_url: Option<String>,
    /// Context (e.g., "cbtop/regression")
    pub context: String,
}
