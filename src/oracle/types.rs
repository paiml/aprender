//! Oracle Mode type definitions
//!
//! Based on Oracle Mode Specification v1.0 (BATUTA-ORACLE-001)

use serde::{Deserialize, Serialize};

// =============================================================================
// Stack Layer Definitions
// =============================================================================

/// Layer in the Sovereign AI Stack hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StackLayer {
    /// Layer 0: Compute primitives (trueno, trueno-db, trueno-graph)
    Primitives,
    /// Layer 1: ML algorithms (aprender)
    MlAlgorithms,
    /// Layer 2: Training & inference (entrenar, realizar, simular)
    MlPipeline,
    /// Layer 3: Transpilers (depyler, decy, bashrs)
    Transpilers,
    /// Layer 4: Orchestration (batuta, repartir)
    Orchestration,
    /// Layer 5: Quality & profiling (certeza, pmat, renacer, probar)
    Quality,
    /// Layer 6: Data loading (alimentar)
    Data,
    /// Layer 7: Media production (rmedia)
    Media,
}

impl StackLayer {
    /// Get numeric layer index
    pub fn index(&self) -> u8 {
        match self {
            StackLayer::Primitives => 0,
            StackLayer::MlAlgorithms => 1,
            StackLayer::MlPipeline => 2,
            StackLayer::Transpilers => 3,
            StackLayer::Orchestration => 4,
            StackLayer::Quality => 5,
            StackLayer::Data => 6,
            StackLayer::Media => 7,
        }
    }

    /// Get all layers in order
    pub fn all() -> Vec<StackLayer> {
        vec![
            StackLayer::Primitives,
            StackLayer::MlAlgorithms,
            StackLayer::MlPipeline,
            StackLayer::Transpilers,
            StackLayer::Orchestration,
            StackLayer::Quality,
            StackLayer::Data,
            StackLayer::Media,
        ]
    }
}

impl std::fmt::Display for StackLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StackLayer::Primitives => write!(f, "Compute Primitives"),
            StackLayer::MlAlgorithms => write!(f, "ML Algorithms"),
            StackLayer::MlPipeline => write!(f, "Training & Inference"),
            StackLayer::Transpilers => write!(f, "Transpilers"),
            StackLayer::Orchestration => write!(f, "Orchestration"),
            StackLayer::Quality => write!(f, "Quality & Profiling"),
            StackLayer::Data => write!(f, "Data Loading"),
            StackLayer::Media => write!(f, "Media Production"),
        }
    }
}

// =============================================================================
// Capability Definitions
// =============================================================================

/// Category of capability
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapabilityCategory {
    Compute,
    Storage,
    MachineLearning,
    Transpilation,
    Validation,
    Profiling,
    Distribution,
    Media,
    ContentGeneration,
}

/// A capability provided by a component
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Capability {
    pub name: String,
    pub category: CapabilityCategory,
    pub description: Option<String>,
}

impl Capability {
    pub fn new(name: impl Into<String>, category: CapabilityCategory) -> Self {
        Self { name: name.into(), category, description: None }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

// =============================================================================
// Component Definitions
// =============================================================================

/// A component in the Sovereign AI Stack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackComponent {
    /// Component name (e.g., "trueno", "aprender")
    pub name: String,
    /// Component version
    pub version: String,
    /// Stack layer
    pub layer: StackLayer,
    /// Description of component purpose
    pub description: String,
    /// Capabilities provided
    pub capabilities: Vec<Capability>,
    /// Crates.io package name (if different from name)
    pub crate_name: Option<String>,
    /// Academic references
    pub references: Vec<Citation>,
}

impl StackComponent {
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        layer: StackLayer,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            layer,
            description: description.into(),
            capabilities: Vec::new(),
            crate_name: None,
            references: Vec::new(),
        }
    }

    pub fn with_capability(mut self, cap: Capability) -> Self {
        self.capabilities.push(cap);
        self
    }

    pub fn with_capabilities(mut self, caps: Vec<Capability>) -> Self {
        self.capabilities.extend(caps);
        self
    }

    pub fn has_capability(&self, name: &str) -> bool {
        self.capabilities.iter().any(|c| c.name == name)
    }
}

/// Academic citation/reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub id: u32,
    pub authors: String,
    pub year: u16,
    pub title: String,
    pub venue: Option<String>,
}

// =============================================================================
// Query Types
// =============================================================================

/// Hardware specification for queries
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HardwareSpec {
    pub has_gpu: bool,
    pub gpu_memory_gb: Option<f32>,
    pub cpu_cores: Option<u32>,
    pub ram_gb: Option<f32>,
    pub is_distributed: bool,
    pub node_count: Option<u32>,
}

impl HardwareSpec {
    pub fn cpu_only() -> Self {
        Self { has_gpu: false, ..Default::default() }
    }

    pub fn with_gpu(memory_gb: f32) -> Self {
        Self { has_gpu: true, gpu_memory_gb: Some(memory_gb), ..Default::default() }
    }

    pub fn has_gpu(&self) -> bool {
        self.has_gpu
    }
}

/// Data size specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataSize {
    /// Number of samples/rows
    Samples(u64),
    /// Size in bytes
    Bytes(u64),
    /// Unknown/unspecified
    Unknown,
}

impl DataSize {
    pub fn samples(n: u64) -> Self {
        DataSize::Samples(n)
    }

    pub fn bytes(n: u64) -> Self {
        DataSize::Bytes(n)
    }

    /// Get sample count if available
    pub fn as_samples(&self) -> Option<u64> {
        match self {
            DataSize::Samples(n) => Some(*n),
            _ => None,
        }
    }

    /// Estimate if this is "large" data (>100K samples or >1GB)
    pub fn is_large(&self) -> bool {
        match self {
            DataSize::Samples(n) => *n > 100_000,
            DataSize::Bytes(n) => *n > 1_000_000_000,
            DataSize::Unknown => false,
        }
    }
}

/// Optimization target for queries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum OptimizationTarget {
    /// Optimize for execution speed
    #[default]
    Speed,
    /// Optimize for memory efficiency
    Memory,
    /// Optimize for power efficiency
    Power,
    /// Balance all factors
    Balanced,
}

/// Query constraints
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryConstraints {
    /// Maximum latency requirement (ms)
    pub max_latency_ms: Option<u64>,
    /// Data size
    pub data_size: Option<DataSize>,
    /// Must run locally (no cloud)
    pub sovereign_only: bool,
    /// EU AI Act compliance required
    pub eu_compliant: bool,
    /// Available hardware
    pub hardware: HardwareSpec,
}

/// Query preferences
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryPreferences {
    /// Optimization target
    pub optimize_for: OptimizationTarget,
    /// Preference for simpler solutions (0.0-1.0)
    pub simplicity_weight: f32,
    /// Existing stack components to integrate with
    pub existing_components: Vec<String>,
}

/// Oracle query structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleQuery {
    /// Problem description in natural language
    pub description: String,
    /// Constraints
    pub constraints: QueryConstraints,
    /// Preferences
    pub preferences: QueryPreferences,
}

impl OracleQuery {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            constraints: QueryConstraints::default(),
            preferences: QueryPreferences::default(),
        }
    }

    pub fn with_constraints(mut self, constraints: QueryConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    pub fn with_preferences(mut self, preferences: QueryPreferences) -> Self {
        self.preferences = preferences;
        self
    }

    pub fn with_data_size(mut self, size: DataSize) -> Self {
        self.constraints.data_size = Some(size);
        self
    }

    pub fn with_hardware(mut self, hardware: HardwareSpec) -> Self {
        self.constraints.hardware = hardware;
        self
    }

    pub fn sovereign_only(mut self) -> Self {
        self.constraints.sovereign_only = true;
        self
    }

    pub fn eu_compliant(mut self) -> Self {
        self.constraints.eu_compliant = true;
        self
    }
}

// =============================================================================
// Response Types
// =============================================================================

/// Compute backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub enum Backend {
    Scalar,
    SIMD,
    GPU,
    Distributed,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Scalar => write!(f, "Scalar"),
            Backend::SIMD => write!(f, "SIMD"),
            Backend::GPU => write!(f, "GPU"),
            Backend::Distributed => write!(f, "Distributed"),
        }
    }
}

/// Operation complexity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpComplexity {
    /// O(n) - element-wise operations
    Low,
    /// O(n log n) to O(n²) - reductions, sorts
    Medium,
    /// O(n²) to O(n³) - matrix operations
    High,
}

/// Compute recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeRecommendation {
    pub backend: Backend,
    pub rationale: String,
}

/// Distribution recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionRecommendation {
    pub tool: Option<String>,
    pub needed: bool,
    pub rationale: String,
    pub node_count: Option<u32>,
}

impl DistributionRecommendation {
    pub fn not_needed(rationale: impl Into<String>) -> Self {
        Self { tool: None, needed: false, rationale: rationale.into(), node_count: None }
    }
}

/// Component recommendation with confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentRecommendation {
    /// Component name
    pub component: String,
    /// Specific module/function path (e.g., "aprender::tree::RandomForest")
    pub path: Option<String>,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Reason for recommendation
    pub rationale: String,
}

impl ComponentRecommendation {
    pub fn new(
        component: impl Into<String>,
        confidence: f32,
        rationale: impl Into<String>,
    ) -> Self {
        Self { component: component.into(), confidence, rationale: rationale.into(), path: None }
    }

    pub fn with_path(
        component: impl Into<String>,
        confidence: f32,
        rationale: impl Into<String>,
        path: String,
    ) -> Self {
        Self {
            component: component.into(),
            confidence,
            rationale: rationale.into(),
            path: Some(path),
        }
    }
}

/// Complete Oracle response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleResponse {
    /// Problem classification
    pub problem_class: String,
    /// Detected algorithm/approach
    pub algorithm: Option<String>,
    /// Primary recommendation
    pub primary: ComponentRecommendation,
    /// Supporting components
    pub supporting: Vec<ComponentRecommendation>,
    /// Compute backend recommendation
    pub compute: ComputeRecommendation,
    /// Distribution recommendation
    pub distribution: DistributionRecommendation,
    /// Example code snippet
    pub code_example: Option<String>,
    /// Related queries for follow-up
    pub related_queries: Vec<String>,
}

impl OracleResponse {
    pub fn new(problem_class: impl Into<String>, primary: ComponentRecommendation) -> Self {
        Self {
            problem_class: problem_class.into(),
            algorithm: None,
            primary,
            supporting: Vec::new(),
            compute: ComputeRecommendation {
                backend: Backend::SIMD,
                rationale: "Default SIMD backend".into(),
            },
            distribution: DistributionRecommendation {
                tool: None,
                needed: false,
                rationale: "Single-node sufficient".into(),
                node_count: None,
            },
            code_example: None,
            related_queries: Vec::new(),
        }
    }

    pub fn with_algorithm(mut self, algo: impl Into<String>) -> Self {
        self.algorithm = Some(algo.into());
        self
    }

    pub fn with_supporting(mut self, rec: ComponentRecommendation) -> Self {
        self.supporting.push(rec);
        self
    }

    pub fn with_compute(mut self, compute: ComputeRecommendation) -> Self {
        self.compute = compute;
        self
    }

    pub fn with_distribution(mut self, dist: DistributionRecommendation) -> Self {
        self.distribution = dist;
        self
    }

    pub fn with_code_example(mut self, code: impl Into<String>) -> Self {
        self.code_example = Some(code.into());
        self
    }
}

// =============================================================================
// Problem Domain Classification
// =============================================================================

/// Problem domain for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProblemDomain {
    // ML domains
    SupervisedLearning,
    UnsupervisedLearning,
    DeepLearning,
    Inference,
    SpeechRecognition,
    // Compute domains
    LinearAlgebra,
    VectorSearch,
    GraphAnalytics,
    // Transpilation domains
    PythonMigration,
    CMigration,
    ShellMigration,
    // Infrastructure domains
    DistributedCompute,
    DataPipeline,
    ModelServing,
    // Quality domains
    Testing,
    Profiling,
    Validation,
    // Media domains
    MediaProduction,
}

impl std::fmt::Display for ProblemDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProblemDomain::SupervisedLearning => write!(f, "Supervised Learning"),
            ProblemDomain::UnsupervisedLearning => write!(f, "Unsupervised Learning"),
            ProblemDomain::DeepLearning => write!(f, "Deep Learning"),
            ProblemDomain::Inference => write!(f, "Model Inference"),
            ProblemDomain::SpeechRecognition => write!(f, "Speech Recognition"),
            ProblemDomain::LinearAlgebra => write!(f, "Linear Algebra"),
            ProblemDomain::VectorSearch => write!(f, "Vector Search"),
            ProblemDomain::GraphAnalytics => write!(f, "Graph Analytics"),
            ProblemDomain::PythonMigration => write!(f, "Python Migration"),
            ProblemDomain::CMigration => write!(f, "C/C++ Migration"),
            ProblemDomain::ShellMigration => write!(f, "Shell Migration"),
            ProblemDomain::DistributedCompute => write!(f, "Distributed Computing"),
            ProblemDomain::DataPipeline => write!(f, "Data Pipeline"),
            ProblemDomain::ModelServing => write!(f, "Model Serving"),
            ProblemDomain::Testing => write!(f, "Testing"),
            ProblemDomain::Profiling => write!(f, "Profiling"),
            ProblemDomain::Validation => write!(f, "Validation"),
            ProblemDomain::MediaProduction => write!(f, "Media Production"),
        }
    }
}

// =============================================================================
// Integration Patterns
// =============================================================================

/// Integration pattern between components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationPattern {
    pub from: String,
    pub to: String,
    pub pattern_name: String,
    pub description: String,
    pub code_template: Option<String>,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[path = "types_tests.rs"]
mod tests;
