//! Bug Hunter Types
//!
//! Types for representing bug hunting results, findings, and hunt configurations.
//!
//! # Popperian Philosophy
//!
//! Bug hunting operationalizes falsification: we systematically attempt to break
//! code, not merely verify it works. Each finding represents a successful
//! falsification of the implicit claim "this code is correct."

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Mode of bug hunting operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HuntMode {
    /// Mutation-based invariant falsification (FDV pattern)
    Falsify,
    /// SBFL without failing tests (SBEST pattern)
    Hunt,
    /// LLM-augmented static analysis (LLIFT pattern)
    Analyze,
    /// Targeted unsafe Rust fuzzing (FourFuzz pattern)
    Fuzz,
    /// Hybrid concolic + SBFL (COTTONTAIL pattern)
    DeepHunt,
    /// Quick pattern-only scan (no clippy, no coverage)
    Quick,
}

impl std::fmt::Display for HuntMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HuntMode::Falsify => write!(f, "Falsify"),
            HuntMode::Hunt => write!(f, "Hunt"),
            HuntMode::Analyze => write!(f, "Analyze"),
            HuntMode::Fuzz => write!(f, "Fuzz"),
            HuntMode::DeepHunt => write!(f, "DeepHunt"),
            HuntMode::Quick => write!(f, "Quick"),
        }
    }
}

/// Severity of a bug finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FindingSeverity {
    /// Informational finding
    Info,
    /// Low severity - style or minor issue
    Low,
    /// Medium severity - potential bug
    Medium,
    /// High severity - likely bug
    High,
    /// Critical - security vulnerability or crash
    Critical,
}

impl std::fmt::Display for FindingSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FindingSeverity::Info => write!(f, "INFO"),
            FindingSeverity::Low => write!(f, "LOW"),
            FindingSeverity::Medium => write!(f, "MEDIUM"),
            FindingSeverity::High => write!(f, "HIGH"),
            FindingSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Category of defect (aligned with OIP categories).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DefectCategory {
    /// Missing or incorrect trait bounds
    TraitBounds,
    /// Syntax/structure issues, macro expansion bugs
    AstTransform,
    /// Ownership/lifetime errors
    OwnershipBorrow,
    /// Config/environment issues
    ConfigurationErrors,
    /// Race conditions, data races
    ConcurrencyBugs,
    /// Security issues
    SecurityVulnerabilities,
    /// Type mismatches
    TypeErrors,
    /// Memory bugs (use-after-free, etc.)
    MemorySafety,
    /// Logic errors
    LogicErrors,
    /// Performance issues
    PerformanceIssues,
    /// GPU/CUDA kernel bugs (PTX, memory access, dimension limits)
    GpuKernelBugs,
    /// Silent degradation (fallbacks that hide failures)
    SilentDegradation,
    /// Test debt (skipped/ignored tests indicating known bugs)
    TestDebt,
    /// Hidden debt (euphemisms like 'placeholder', 'stub', 'demo')
    HiddenDebt,
    /// Contract verification gap (BH-26: missing proof, partial binding)
    ContractGap,
    /// Model parity gap (BH-27: oracle mismatch, quantization drift)
    ModelParityGap,
    /// Unknown/uncategorized
    Unknown,
}

impl std::fmt::Display for DefectCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DefectCategory::TraitBounds => write!(f, "TraitBounds"),
            DefectCategory::AstTransform => write!(f, "ASTTransform"),
            DefectCategory::OwnershipBorrow => write!(f, "OwnershipBorrow"),
            DefectCategory::ConfigurationErrors => write!(f, "ConfigurationErrors"),
            DefectCategory::ConcurrencyBugs => write!(f, "ConcurrencyBugs"),
            DefectCategory::SecurityVulnerabilities => write!(f, "SecurityVulnerabilities"),
            DefectCategory::TypeErrors => write!(f, "TypeErrors"),
            DefectCategory::MemorySafety => write!(f, "MemorySafety"),
            DefectCategory::LogicErrors => write!(f, "LogicErrors"),
            DefectCategory::PerformanceIssues => write!(f, "PerformanceIssues"),
            DefectCategory::GpuKernelBugs => write!(f, "GpuKernelBugs"),
            DefectCategory::SilentDegradation => write!(f, "SilentDegradation"),
            DefectCategory::TestDebt => write!(f, "TestDebt"),
            DefectCategory::HiddenDebt => write!(f, "HiddenDebt"),
            DefectCategory::ContractGap => write!(f, "ContractGap"),
            DefectCategory::ModelParityGap => write!(f, "ModelParityGap"),
            DefectCategory::Unknown => write!(f, "Unknown"),
        }
    }
}

/// A single bug finding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    /// Unique identifier
    pub id: String,

    /// File path where the finding was located
    pub file: PathBuf,

    /// Line number (1-indexed)
    pub line: usize,

    /// Column number (1-indexed, optional)
    pub column: Option<usize>,

    /// Finding title
    pub title: String,

    /// Detailed description
    pub description: String,

    /// Severity level
    pub severity: FindingSeverity,

    /// Defect category
    pub category: DefectCategory,

    /// Suspiciousness score (0.0 - 1.0)
    pub suspiciousness: f64,

    /// Hunt mode that discovered this finding
    pub discovered_by: HuntMode,

    /// Evidence supporting the finding
    pub evidence: Vec<FindingEvidence>,

    /// Suggested fix (if available)
    pub suggested_fix: Option<String>,

    /// Related findings (by ID)
    pub related: Vec<String>,

    /// Regression risk score (0.0 - 1.0) from PMAT quality data (BH-24)
    #[serde(default)]
    pub regression_risk: Option<f64>,

    /// Git blame information: author name
    #[serde(default)]
    pub blame_author: Option<String>,

    /// Git blame information: commit hash (short)
    #[serde(default)]
    pub blame_commit: Option<String>,

    /// Git blame information: date of last change
    #[serde(default)]
    pub blame_date: Option<String>,
}

impl Finding {
    /// Create a new finding.
    pub fn new(
        id: impl Into<String>,
        file: impl Into<PathBuf>,
        line: usize,
        title: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            file: file.into(),
            line,
            column: None,
            title: title.into(),
            description: String::new(),
            severity: FindingSeverity::Medium,
            category: DefectCategory::Unknown,
            suspiciousness: 0.5,
            discovered_by: HuntMode::Analyze,
            evidence: Vec::new(),
            suggested_fix: None,
            related: Vec::new(),
            regression_risk: None,
            blame_author: None,
            blame_commit: None,
            blame_date: None,
        }
    }

    /// Set column.
    pub fn with_column(mut self, column: usize) -> Self {
        self.column = Some(column);
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set severity.
    pub fn with_severity(mut self, severity: FindingSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Set category.
    pub fn with_category(mut self, category: DefectCategory) -> Self {
        self.category = category;
        self
    }

    /// Set suspiciousness score.
    pub fn with_suspiciousness(mut self, score: f64) -> Self {
        self.suspiciousness = score.clamp(0.0, 1.0);
        self
    }

    /// Set discovery mode.
    pub fn with_discovered_by(mut self, mode: HuntMode) -> Self {
        self.discovered_by = mode;
        self
    }

    /// Add evidence.
    pub fn with_evidence(mut self, evidence: FindingEvidence) -> Self {
        self.evidence.push(evidence);
        self
    }

    /// Set suggested fix.
    pub fn with_fix(mut self, fix: impl Into<String>) -> Self {
        self.suggested_fix = Some(fix.into());
        self
    }

    /// Set regression risk score (BH-24).
    pub fn with_regression_risk(mut self, risk: f64) -> Self {
        self.regression_risk = Some(risk.clamp(0.0, 1.0));
        self
    }

    /// Set git blame information.
    pub fn with_blame(
        mut self,
        author: impl Into<String>,
        commit: impl Into<String>,
        date: impl Into<String>,
    ) -> Self {
        self.blame_author = Some(author.into());
        self.blame_commit = Some(commit.into());
        self.blame_date = Some(date.into());
        self
    }

    /// Get location string.
    pub fn location(&self) -> String {
        match self.column {
            Some(col) => format!("{}:{}:{}", self.file.display(), self.line, col),
            None => format!("{}:{}", self.file.display(), self.line),
        }
    }
}

/// Evidence supporting a finding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingEvidence {
    /// Evidence type
    pub evidence_type: EvidenceKind,

    /// Description
    pub description: String,

    /// Raw data
    pub data: Option<String>,
}

impl FindingEvidence {
    /// Create mutation evidence.
    pub fn mutation(mutant_id: impl Into<String>, survived: bool) -> Self {
        Self {
            evidence_type: EvidenceKind::MutationSurvival,
            description: format!(
                "Mutant {} {}",
                mutant_id.into(),
                if survived { "SURVIVED" } else { "KILLED" }
            ),
            data: Some(if survived { "SURVIVED".into() } else { "KILLED".into() }),
        }
    }

    /// Create SBFL evidence.
    pub fn sbfl(formula: impl Into<String>, score: f64) -> Self {
        Self {
            evidence_type: EvidenceKind::SbflScore,
            description: format!("{} suspiciousness: {:.3}", formula.into(), score),
            data: Some(format!("{:.6}", score)),
        }
    }

    /// Create static analysis evidence.
    pub fn static_analysis(tool: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            evidence_type: EvidenceKind::StaticAnalysis,
            description: format!("[{}] {}", tool.into(), message.into()),
            data: None,
        }
    }

    /// Create fuzzing evidence.
    pub fn fuzzing(input: impl Into<String>, crash_type: impl Into<String>) -> Self {
        Self {
            evidence_type: EvidenceKind::FuzzingCrash,
            description: crash_type.into(),
            data: Some(input.into()),
        }
    }

    /// Create quality metrics evidence (BH-21).
    pub fn quality_metrics(grade: impl Into<String>, tdg_score: f64, complexity: u32) -> Self {
        Self {
            evidence_type: EvidenceKind::QualityMetrics,
            description: format!(
                "PMAT grade {} (TDG: {:.1}, complexity: {})",
                grade.into(),
                tdg_score,
                complexity
            ),
            data: Some(format!("{:.1}", tdg_score)),
        }
    }

    /// Create contract binding evidence (BH-26).
    pub fn contract_binding(
        contract: impl Into<String>,
        equation: impl Into<String>,
        status: impl Into<String>,
    ) -> Self {
        Self {
            evidence_type: EvidenceKind::ContractBinding,
            description: format!(
                "Contract {} eq {} — {}",
                contract.into(),
                equation.into(),
                status.into()
            ),
            data: None,
        }
    }

    /// Create model parity evidence (BH-27).
    pub fn model_parity(
        model: impl Into<String>,
        check: impl Into<String>,
        result: impl Into<String>,
    ) -> Self {
        Self {
            evidence_type: EvidenceKind::ModelParity,
            description: format!("Model {} — {} — {}", model.into(), check.into(), result.into()),
            data: None,
        }
    }

    /// Create concolic evidence.
    pub fn concolic(path_constraint: impl Into<String>) -> Self {
        Self {
            evidence_type: EvidenceKind::ConcolicPath,
            description: "Path constraint solved".into(),
            data: Some(path_constraint.into()),
        }
    }
}

/// Kind of evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceKind {
    /// Mutation survived (test weakness)
    MutationSurvival,
    /// SBFL suspiciousness score
    SbflScore,
    /// Static analysis warning
    StaticAnalysis,
    /// Fuzzing crash/violation
    FuzzingCrash,
    /// Concolic execution path
    ConcolicPath,
    /// LLM classification
    LlmClassification,
    /// Git history correlation
    GitHistory,
    /// PMAT quality metrics (BH-21)
    QualityMetrics,
    /// Contract binding status (BH-26)
    ContractBinding,
    /// Model parity result (BH-27)
    ModelParity,
}

/// Configuration for a bug hunt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuntConfig {
    /// Hunt mode
    pub mode: HuntMode,

    /// Target paths to analyze
    pub targets: Vec<PathBuf>,

    /// Minimum suspiciousness threshold
    pub min_suspiciousness: f64,

    /// Maximum findings to report
    pub max_findings: usize,

    /// SBFL formula (for Hunt/DeepHunt modes)
    pub sbfl_formula: SbflFormula,

    /// Enable LLM filtering
    pub llm_filter: bool,

    /// Fuzzing duration in seconds (for Fuzz mode)
    pub fuzz_duration_secs: u64,

    /// Mutation timeout in seconds (for Falsify mode)
    pub mutation_timeout_secs: u64,

    /// Categories to include (empty = all)
    pub include_categories: Vec<DefectCategory>,

    /// Categories to exclude
    pub exclude_categories: Vec<DefectCategory>,

    // =========================================================================
    // BH-11 to BH-15: Advanced Features
    // =========================================================================
    /// Spec file path (BH-11: Spec-Driven Bug Hunting)
    pub spec_path: Option<PathBuf>,

    /// Spec section filter (e.g., "Authentication")
    pub spec_section: Option<String>,

    /// PMAT ticket reference (BH-12: Ticket Integration)
    pub ticket_ref: Option<String>,

    /// Update spec with findings (BH-14: Bidirectional Linking)
    pub update_spec: bool,

    /// Analyze library only (BH-13: Scoped Analysis)
    pub lib_only: bool,

    /// Analyze specific binary (BH-13: Scoped Analysis)
    pub bin_target: Option<String>,

    /// Exclude test code (BH-13: Scoped Analysis)
    pub exclude_tests: bool,

    /// Suppress known false positive patterns (BH-15)
    pub suppress_false_positives: bool,

    /// Custom coverage data path (lcov.info)
    pub coverage_path: Option<PathBuf>,

    /// Coverage weight factor for hotpath weighting (default 0.5)
    pub coverage_weight: f64,

    // =========================================================================
    // BH-16 to BH-20: Research-Based Fault Localization
    // =========================================================================
    /// Fault localization strategy (BH-16 to BH-19)
    pub localization_strategy: LocalizationStrategy,

    /// Channel weights for multi-channel localization (BH-19)
    pub channel_weights: ChannelWeights,

    /// Enable predictive mutation testing (BH-18)
    pub predictive_mutation: bool,

    /// Enable semantic crash bucketing (BH-20)
    pub crash_bucketing: CrashBucketingMode,

    // =========================================================================
    // BH-21 to BH-25: PMAT Quality Integration
    // =========================================================================
    /// Enable PMAT quality-weighted suspiciousness (BH-21)
    pub use_pmat_quality: bool,

    /// Quality weight factor for suspiciousness adjustment (BH-21, default 0.5)
    pub quality_weight: f64,

    /// Use PMAT to scope targets by quality (BH-22)
    pub pmat_scope: bool,

    /// Enable SATD-enriched findings from PMAT (BH-23, default true)
    pub pmat_satd: bool,

    /// PMAT query string for scoping (BH-22)
    pub pmat_query: Option<String>,

    // =========================================================================
    // BH-26 to BH-27: Contract & Model Parity Analysis
    // =========================================================================
    /// Explicit path to provable-contracts directory (BH-26)
    pub contracts_path: Option<PathBuf>,

    /// Auto-discover provable-contracts in sibling directories (BH-26)
    pub contracts_auto: bool,

    /// Explicit path to tiny-model-ground-truth directory (BH-27)
    pub model_parity_path: Option<PathBuf>,

    /// Auto-discover tiny-model-ground-truth in sibling directories (BH-27)
    pub model_parity_auto: bool,
}

/// Crash bucketing mode (BH-20).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CrashBucketingMode {
    /// No bucketing
    #[default]
    None,
    /// Stack trace similarity only
    StackTrace,
    /// Semantic root cause analysis
    Semantic,
}

impl Default for HuntConfig {
    fn default() -> Self {
        Self {
            mode: HuntMode::Analyze,
            targets: vec![PathBuf::from("src")],
            min_suspiciousness: 0.5,
            max_findings: 50,
            sbfl_formula: SbflFormula::Ochiai,
            llm_filter: false,
            fuzz_duration_secs: 60,
            mutation_timeout_secs: 30,
            include_categories: Vec::new(),
            exclude_categories: Vec::new(),
            // BH-11 to BH-15 defaults
            spec_path: None,
            spec_section: None,
            ticket_ref: None,
            update_spec: false,
            lib_only: false,
            bin_target: None,
            exclude_tests: true,            // Default to excluding tests
            suppress_false_positives: true, // Default to suppressing
            coverage_path: None,
            coverage_weight: 0.5, // Default hotpath weight
            // BH-16 to BH-20 defaults
            localization_strategy: LocalizationStrategy::default(),
            channel_weights: ChannelWeights::default(),
            predictive_mutation: false,
            crash_bucketing: CrashBucketingMode::default(),
            // BH-21 to BH-25 defaults
            use_pmat_quality: false,
            quality_weight: 0.5,
            pmat_scope: false,
            pmat_satd: true,
            pmat_query: None,
            // BH-26 to BH-27 defaults
            contracts_path: None,
            contracts_auto: false,
            model_parity_path: None,
            model_parity_auto: false,
        }
    }
}

/// SBFL formula for fault localization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SbflFormula {
    /// Tarantula formula (classic)
    Tarantula,
    /// Ochiai formula (cosine similarity)
    #[default]
    Ochiai,
    /// DStar with power 2
    DStar2,
    /// DStar with power 3
    DStar3,
}

/// Fault localization strategy (BH-16 to BH-19).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LocalizationStrategy {
    /// Spectrum-Based Fault Localization only
    #[default]
    Sbfl,
    /// Mutation-Based Fault Localization (BH-16)
    Mbfl,
    /// Causal inference with interventions (BH-17)
    Causal,
    /// Multi-channel combination (BH-19)
    MultiChannel,
    /// Hybrid SBFL + MBFL with configurable weights
    Hybrid,
}

impl std::fmt::Display for LocalizationStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sbfl => write!(f, "SBFL"),
            Self::Mbfl => write!(f, "MBFL"),
            Self::Causal => write!(f, "Causal"),
            Self::MultiChannel => write!(f, "MultiChannel"),
            Self::Hybrid => write!(f, "Hybrid"),
        }
    }
}

/// Multi-channel weights for fault localization (BH-19, BH-21).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelWeights {
    /// SBFL spectrum-based weight
    pub spectrum: f64,
    /// MBFL mutation-based weight
    pub mutation: f64,
    /// Static analysis weight (clippy/patterns)
    pub static_analysis: f64,
    /// Semantic similarity weight (error message matching)
    pub semantic: f64,
    /// PMAT quality weight (BH-21)
    #[serde(default)]
    pub quality: f64,
}

impl Default for ChannelWeights {
    fn default() -> Self {
        Self {
            spectrum: 0.30,
            mutation: 0.25,
            static_analysis: 0.20,
            semantic: 0.15,
            quality: 0.10,
        }
    }
}

impl ChannelWeights {
    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let sum =
            self.spectrum + self.mutation + self.static_analysis + self.semantic + self.quality;
        if sum > 0.0 {
            self.spectrum /= sum;
            self.mutation /= sum;
            self.static_analysis /= sum;
            self.semantic /= sum;
            self.quality /= sum;
        }
    }

    /// Compute weighted score from channel scores (5 channels)
    pub fn combine(
        &self,
        spectrum: f64,
        mutation: f64,
        static_score: f64,
        semantic: f64,
        quality: f64,
    ) -> f64 {
        self.spectrum * spectrum
            + self.mutation * mutation
            + self.static_analysis * static_score
            + self.semantic * semantic
            + self.quality * quality
    }
}

impl std::fmt::Display for SbflFormula {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SbflFormula::Tarantula => write!(f, "Tarantula"),
            SbflFormula::Ochiai => write!(f, "Ochiai"),
            SbflFormula::DStar2 => write!(f, "DStar2"),
            SbflFormula::DStar3 => write!(f, "DStar3"),
        }
    }
}

/// Result of a bug hunt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuntResult {
    /// Project path analyzed
    pub project_path: PathBuf,

    /// Hunt mode used
    pub mode: HuntMode,

    /// Configuration used
    pub config: HuntConfig,

    /// Findings discovered
    pub findings: Vec<Finding>,

    /// Statistics
    pub stats: HuntStats,

    /// Timestamp
    pub timestamp: String,

    /// Duration in milliseconds
    pub duration_ms: u64,

    /// Phase timing breakdown
    #[serde(default)]
    pub phase_timings: PhaseTimings,
}

impl HuntResult {
    /// Create a new hunt result.
    pub fn new(project_path: impl Into<PathBuf>, mode: HuntMode, config: HuntConfig) -> Self {
        Self {
            project_path: project_path.into(),
            mode,
            config,
            findings: Vec::new(),
            stats: HuntStats::default(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            duration_ms: 0,
            phase_timings: PhaseTimings::default(),
        }
    }

    /// Add a finding.
    pub fn add_finding(&mut self, finding: Finding) {
        self.findings.push(finding);
    }

    /// Set duration.
    pub fn with_duration(mut self, ms: u64) -> Self {
        self.duration_ms = ms;
        self
    }

    /// Finalize statistics.
    pub fn finalize(&mut self) {
        self.stats = HuntStats::from_findings(&self.findings);
    }

    /// Get findings sorted by suspiciousness (descending).
    pub fn top_findings(&self, n: usize) -> Vec<&Finding> {
        let mut sorted: Vec<_> = self.findings.iter().collect();
        sorted.sort_by(|a, b| {
            b.suspiciousness.partial_cmp(&a.suspiciousness).unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(n).collect()
    }

    /// Get summary string.
    pub fn summary(&self) -> String {
        let c = self.stats.by_severity.get(&FindingSeverity::Critical).unwrap_or(&0);
        let h = self.stats.by_severity.get(&FindingSeverity::High).unwrap_or(&0);
        format!(
            "{} mode: {} findings in {} files ({}C {}H) -- {}ms",
            self.mode,
            self.findings.len(),
            self.stats.files_analyzed,
            c,
            h,
            self.duration_ms
        )
    }
}

impl Default for HuntResult {
    fn default() -> Self {
        Self {
            project_path: PathBuf::new(),
            mode: HuntMode::Quick,
            config: HuntConfig::default(),
            findings: Vec::new(),
            stats: HuntStats::default(),
            timestamp: String::new(),
            duration_ms: 0,
            phase_timings: PhaseTimings::default(),
        }
    }
}

/// Statistics from a bug hunt.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HuntStats {
    /// Total findings
    pub total_findings: usize,

    /// Findings by severity
    pub by_severity: HashMap<FindingSeverity, usize>,

    /// Findings by category
    pub by_category: HashMap<DefectCategory, usize>,

    /// Files analyzed
    pub files_analyzed: usize,

    /// Lines analyzed
    pub lines_analyzed: usize,

    /// Average suspiciousness
    pub avg_suspiciousness: f64,

    /// Max suspiciousness
    pub max_suspiciousness: f64,

    /// Mode-specific stats
    pub mode_stats: ModeStats,
}

impl HuntStats {
    /// Compute statistics from findings.
    pub fn from_findings(findings: &[Finding]) -> Self {
        let mut by_severity: HashMap<FindingSeverity, usize> = HashMap::new();
        let mut by_category: HashMap<DefectCategory, usize> = HashMap::new();
        let mut total_suspiciousness = 0.0;
        let mut max_suspiciousness = 0.0;
        let mut unique_files: std::collections::HashSet<&std::path::Path> =
            std::collections::HashSet::new();

        for finding in findings {
            *by_severity.entry(finding.severity).or_default() += 1;
            *by_category.entry(finding.category).or_default() += 1;
            total_suspiciousness += finding.suspiciousness;
            if finding.suspiciousness > max_suspiciousness {
                max_suspiciousness = finding.suspiciousness;
            }
            unique_files.insert(&finding.file);
        }

        Self {
            total_findings: findings.len(),
            by_severity,
            by_category,
            files_analyzed: unique_files.len(),
            avg_suspiciousness: if findings.is_empty() {
                0.0
            } else {
                total_suspiciousness / findings.len() as f64
            },
            max_suspiciousness,
            ..Default::default()
        }
    }
}

/// Phase timing breakdown for bug-hunter pipeline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhaseTimings {
    /// Mode dispatch (scanning) phase duration in ms
    pub mode_dispatch_ms: u64,
    /// PMAT index construction duration in ms
    pub pmat_index_ms: u64,
    /// PMAT weight application duration in ms
    pub pmat_weights_ms: u64,
    /// Finalization phase duration in ms
    pub finalize_ms: u64,
    /// Contract gap analysis duration in ms (BH-26)
    pub contract_gap_ms: u64,
    /// Model parity analysis duration in ms (BH-27)
    pub model_parity_ms: u64,
}

/// Mode-specific statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModeStats {
    /// Mutation testing: total mutants
    pub mutants_total: usize,
    /// Mutation testing: killed mutants
    pub mutants_killed: usize,
    /// Mutation testing: survived mutants
    pub mutants_survived: usize,

    /// SBFL: passing tests
    pub sbfl_passing_tests: usize,
    /// SBFL: failing tests
    pub sbfl_failing_tests: usize,

    /// Fuzzing: total executions
    pub fuzz_executions: usize,
    /// Fuzzing: crashes found
    pub fuzz_crashes: usize,
    /// Fuzzing: coverage percentage
    pub fuzz_coverage: f64,

    /// Concolic: paths explored
    pub concolic_paths: usize,
    /// Concolic: constraints solved
    pub concolic_constraints_solved: usize,
    /// Concolic: timeouts
    pub concolic_timeouts: usize,

    /// LLM: warnings filtered
    pub llm_filtered: usize,
    /// LLM: true positives retained
    pub llm_retained: usize,
}

#[cfg(test)]
#[path = "types_tests.rs"]
mod tests;
