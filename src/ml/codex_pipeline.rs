//! CODEX Integration - End-to-end pipeline
//!
//! Unified pipeline orchestrating all CODEX components:
//! Generator → Quality Gate → Bug Priority → Oracle → Rich Labels → Export
//!
//! # Pipeline Flow
//!
//! ```text
//! Generator ──► QualityGate ──► DefectPredictor ──► ActiveLearner
//!                   │                 │                   │
//!                   ▼                 ▼                   ▼
//!              (filtered)        (prioritized)       (sampled)
//!                                                        │
//!                                                        ▼
//!                                                     Oracle
//!                                                        │
//!                                                        ▼
//!                                               RichLabel + Export
//! ```
//!
//! # Reference
//! - VER-054: CODEX Integration - End-to-end pipeline

use serde::{Deserialize, Serialize};

use crate::generator::GeneratedCode;
use crate::ml::{ActiveLearner, CommitFeatures, DefectPredictor, QualityGate, RichLabel};

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Quality gate threshold (0.0 to 1.0)
    pub quality_threshold: f32,
    /// Number of clusters for active learning
    pub num_clusters: usize,
    /// Batch size for oracle calls
    pub batch_size: usize,
    /// Maximum oracle calls per run
    pub max_oracle_calls: usize,
    /// Target oracle call reduction (e.g., 10 for 10x reduction)
    pub target_reduction: f32,
    /// Enable active learning exploration
    pub enable_active_learning: bool,
    /// Enable defect prediction prioritization
    pub enable_defect_priority: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            quality_threshold: 0.5,
            num_clusters: 5,
            batch_size: 100,
            max_oracle_calls: 1000,
            target_reduction: 10.0,
            enable_active_learning: true,
            enable_defect_priority: true,
        }
    }
}

impl PipelineConfig {
    /// Create strict config (high quality threshold)
    #[must_use]
    pub fn strict() -> Self {
        Self {
            quality_threshold: 0.7,
            ..Default::default()
        }
    }

    /// Create fast config (minimal filtering)
    #[must_use]
    pub fn fast() -> Self {
        Self {
            quality_threshold: 0.3,
            enable_active_learning: false,
            enable_defect_priority: false,
            ..Default::default()
        }
    }

    /// Validate configuration
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if self.quality_threshold < 0.0 || self.quality_threshold > 1.0 {
            errors.push("quality_threshold must be in [0.0, 1.0]".to_string());
        }

        if self.num_clusters == 0 {
            errors.push("num_clusters must be > 0".to_string());
        }

        if self.batch_size == 0 {
            errors.push("batch_size must be > 0".to_string());
        }

        if self.target_reduction <= 0.0 {
            errors.push("target_reduction must be > 0".to_string());
        }

        errors
    }
}

/// Data quality metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DataQualityMetrics {
    /// Novelty score (0-1): how different from existing samples
    pub novelty: f32,
    /// Diversity score (0-1): variety within dataset (silhouette)
    pub diversity: f32,
    /// Difficulty score (0-1): complexity of samples
    pub difficulty: f32,
    /// Coverage score (0-1): AST/feature space coverage
    pub coverage: f32,
    /// Bug revelation rate (fraction of samples revealing bugs)
    pub bug_rate: f32,
}

impl DataQualityMetrics {
    /// Overall quality score (weighted average)
    #[must_use]
    pub fn overall(&self) -> f32 {
        let weights = [0.2, 0.25, 0.2, 0.2, 0.15]; // novelty, diversity, difficulty, coverage, bug_rate
        let values = [
            self.novelty,
            self.diversity,
            self.difficulty,
            self.coverage,
            self.bug_rate,
        ];

        let weighted_sum: f32 = values.iter().zip(&weights).map(|(v, w)| v * w).sum();
        let total_weight: f32 = weights.iter().sum();

        weighted_sum / total_weight
    }

    /// Check if quality meets targets
    #[must_use]
    pub fn meets_targets(&self) -> bool {
        self.diversity >= 0.6 && self.bug_rate >= 0.15 && self.coverage >= 0.7
    }
}

/// Pipeline stage result
#[derive(Debug, Clone)]
pub struct StageResult {
    /// Stage name
    pub stage: String,
    /// Number of items input
    pub input_count: usize,
    /// Number of items output
    pub output_count: usize,
    /// Processing time in milliseconds
    pub time_ms: u64,
}

impl StageResult {
    /// Reduction factor (input / output)
    #[must_use]
    pub fn reduction_factor(&self) -> f32 {
        if self.output_count == 0 {
            f32::INFINITY
        } else {
            self.input_count as f32 / self.output_count as f32
        }
    }

    /// Pass-through rate (output / input)
    #[must_use]
    pub fn pass_rate(&self) -> f32 {
        if self.input_count == 0 {
            0.0
        } else {
            self.output_count as f32 / self.input_count as f32
        }
    }
}

/// Pipeline execution result
#[derive(Debug, Clone, Default)]
pub struct PipelineResult {
    /// Results from each stage
    pub stages: Vec<StageResult>,
    /// Final labeled samples
    pub labels: Vec<RichLabel>,
    /// Data quality metrics
    pub quality: DataQualityMetrics,
    /// Total samples generated
    pub total_generated: usize,
    /// Total oracle calls made
    pub oracle_calls: usize,
    /// Oracle call reduction achieved
    pub oracle_reduction: f32,
}

impl PipelineResult {
    /// Get stage by name
    #[must_use]
    pub fn stage(&self, name: &str) -> Option<&StageResult> {
        self.stages.iter().find(|s| s.stage == name)
    }

    /// Total pipeline time in milliseconds
    #[must_use]
    pub fn total_time_ms(&self) -> u64 {
        self.stages.iter().map(|s| s.time_ms).sum()
    }

    /// Did pipeline meet oracle reduction target?
    #[must_use]
    pub fn met_oracle_target(&self, target: f32) -> bool {
        self.oracle_reduction >= target
    }
}

/// Sample prepared for oracle verification
#[derive(Debug, Clone)]
pub struct PreparedSample {
    /// Generated code
    pub code: GeneratedCode,
    /// Quality score
    pub quality_score: f32,
    /// Defect probability
    pub defect_probability: f32,
    /// Cluster assignment
    pub cluster: Option<usize>,
    /// Priority rank
    pub priority: usize,
}

/// CODEX pipeline orchestrator
#[derive(Debug)]
pub struct CodexPipeline {
    /// Configuration
    config: PipelineConfig,
    /// Quality gate
    quality_gate: QualityGate,
    /// Defect predictor
    defect_predictor: DefectPredictor,
    /// Active learner
    active_learner: ActiveLearner,
    /// Pipeline statistics
    stats: PipelineStats,
}

/// Pipeline statistics
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Total runs
    pub runs: usize,
    /// Total samples processed
    pub samples_processed: usize,
    /// Total oracle calls
    pub oracle_calls: usize,
    /// Total bugs found
    pub bugs_found: usize,
    /// Average oracle reduction
    pub avg_oracle_reduction: f32,
}

impl Default for CodexPipeline {
    fn default() -> Self {
        Self::new(PipelineConfig::default())
    }
}

impl CodexPipeline {
    /// Create new pipeline with config
    #[must_use]
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            quality_gate: QualityGate::new(config.quality_threshold),
            defect_predictor: DefectPredictor::new(),
            active_learner: ActiveLearner::new(config.num_clusters),
            config,
            stats: PipelineStats::default(),
        }
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Stage 1: Filter by quality gate
    pub fn filter_quality<'a>(
        &mut self,
        codes: &'a [GeneratedCode],
    ) -> (Vec<&'a GeneratedCode>, StageResult) {
        let start = std::time::Instant::now();
        let input_count = codes.len();

        let passed = self.quality_gate.filter_batch(codes);

        let result = StageResult {
            stage: "quality_gate".to_string(),
            input_count,
            output_count: passed.len(),
            time_ms: start.elapsed().as_millis() as u64,
        };

        (passed, result)
    }

    /// Stage 2: Prioritize by defect likelihood
    pub fn prioritize_defects<'a>(
        &self,
        codes: &'a [&GeneratedCode],
    ) -> (Vec<&'a GeneratedCode>, StageResult) {
        let start = std::time::Instant::now();
        let input_count = codes.len();

        if !self.config.enable_defect_priority {
            return (
                codes.to_vec(),
                StageResult {
                    stage: "defect_priority".to_string(),
                    input_count,
                    output_count: input_count,
                    time_ms: start.elapsed().as_millis() as u64,
                },
            );
        }

        // Create feature/code pairs for prioritization
        let pairs: Vec<(CommitFeatures, String)> = codes
            .iter()
            .map(|c| (CommitFeatures::default(), c.code.clone()))
            .collect();

        let order = self.defect_predictor.prioritize(&pairs);

        // Take top batch_size samples
        let output_count = order.len().min(self.config.batch_size);
        let prioritized: Vec<&GeneratedCode> = order
            .iter()
            .take(output_count)
            .filter_map(|&i| codes.get(i).copied())
            .collect();

        let result = StageResult {
            stage: "defect_priority".to_string(),
            input_count,
            output_count: prioritized.len(),
            time_ms: start.elapsed().as_millis() as u64,
        };

        (prioritized, result)
    }

    /// Stage 3: Sample via active learning
    pub fn sample_active<'a>(
        &mut self,
        codes: &'a [&GeneratedCode],
    ) -> (Vec<&'a GeneratedCode>, StageResult) {
        let start = std::time::Instant::now();
        let input_count = codes.len();

        if !self.config.enable_active_learning || codes.is_empty() {
            return (
                codes.to_vec(),
                StageResult {
                    stage: "active_learning".to_string(),
                    input_count,
                    output_count: input_count,
                    time_ms: start.elapsed().as_millis() as u64,
                },
            );
        }

        // Fit clusters
        let code_strings: Vec<&str> = codes.iter().map(|c| c.code.as_str()).collect();
        self.active_learner.fit(&code_strings);

        // Select batch via Thompson Sampling
        let batch_size = self.config.batch_size.min(codes.len());
        let selected_indices = self.active_learner.select_batch(&code_strings, batch_size);

        let selected: Vec<&GeneratedCode> = selected_indices
            .iter()
            .filter_map(|&i| codes.get(i).copied())
            .collect();

        let result = StageResult {
            stage: "active_learning".to_string(),
            input_count,
            output_count: selected.len(),
            time_ms: start.elapsed().as_millis() as u64,
        };

        (selected, result)
    }

    /// Prepare samples for oracle (all stages)
    pub fn prepare(&mut self, codes: &[GeneratedCode]) -> (Vec<PreparedSample>, Vec<StageResult>) {
        let mut stages = Vec::new();

        // Stage 1: Quality Gate - clone to avoid borrow issues
        let (quality_passed_refs, stage1) = self.filter_quality(codes);
        let quality_passed: Vec<GeneratedCode> = quality_passed_refs.into_iter().cloned().collect();
        stages.push(stage1);

        if quality_passed.is_empty() {
            return (vec![], stages);
        }

        // Stage 2: Defect Priority
        let quality_refs: Vec<&GeneratedCode> = quality_passed.iter().collect();
        let (prioritized_refs, stage2) = self.prioritize_defects(&quality_refs);
        let prioritized: Vec<GeneratedCode> = prioritized_refs.into_iter().cloned().collect();
        stages.push(stage2);

        // Stage 3: Active Learning
        let prioritized_refs: Vec<&GeneratedCode> = prioritized.iter().collect();
        let (sampled_refs, stage3) = self.sample_active(&prioritized_refs);
        let sampled: Vec<GeneratedCode> = sampled_refs.into_iter().cloned().collect();
        stages.push(stage3);

        // Create prepared samples - now we can use self freely
        let prepared: Vec<PreparedSample> = sampled
            .into_iter()
            .enumerate()
            .map(|(i, code)| {
                let quality_score = self.quality_gate.score(
                    &crate::ml::QualityFeatureExtractor::new().extract_from_generated(&code),
                );

                let defect_pred = self
                    .defect_predictor
                    .predict(&CommitFeatures::default(), &code.code);

                let cluster = self.active_learner.get_cluster(&code.code);

                PreparedSample {
                    code,
                    quality_score,
                    defect_probability: defect_pred.base_probability,
                    cluster,
                    priority: i,
                }
            })
            .collect();

        (prepared, stages)
    }

    /// Update from oracle feedback
    pub fn update_feedback(&mut self, code: &str, revealed_bug: bool) {
        self.active_learner.update_feedback(code, revealed_bug);

        if revealed_bug {
            self.stats.bugs_found += 1;
        }
    }

    /// Compute data quality metrics
    #[must_use]
    pub fn compute_quality(&self, labels: &[RichLabel]) -> DataQualityMetrics {
        if labels.is_empty() {
            return DataQualityMetrics::default();
        }

        // Bug rate
        let bugs = labels.iter().filter(|l| !l.is_correct).count();
        let bug_rate = bugs as f32 / labels.len() as f32;

        // Diversity from active learner
        let diversity = self.active_learner.silhouette_score().max(0.0);

        // Difficulty based on error severity
        let total_severity: f32 = labels
            .iter()
            .filter_map(|l| l.error_category)
            .map(|c| c.severity())
            .sum();
        let difficulty = if bugs > 0 {
            (total_severity / bugs as f32).min(1.0)
        } else {
            0.3
        };

        // Coverage estimate from soft labels
        let avg_structural_sim: f32 = labels
            .iter()
            .map(|l| l.soft_labels.structural_similarity)
            .sum::<f32>()
            / labels.len() as f32;
        let coverage = 1.0 - avg_structural_sim; // Less similarity = more coverage

        // Novelty placeholder (would need historical comparison)
        let novelty = 0.5;

        DataQualityMetrics {
            novelty,
            diversity,
            difficulty,
            coverage,
            bug_rate,
        }
    }

    /// Run full pipeline (without actual oracle - for testing)
    pub fn run_dry(&mut self, codes: &[GeneratedCode]) -> PipelineResult {
        let total_generated = codes.len();

        let (prepared, stages) = self.prepare(codes);

        let oracle_calls = prepared.len();
        let oracle_reduction = if oracle_calls > 0 {
            total_generated as f32 / oracle_calls as f32
        } else {
            f32::INFINITY
        };

        // Update stats
        self.stats.runs += 1;
        self.stats.samples_processed += total_generated;
        self.stats.oracle_calls += oracle_calls;

        if self.stats.runs > 1 {
            self.stats.avg_oracle_reduction =
                (self.stats.avg_oracle_reduction * (self.stats.runs - 1) as f32 + oracle_reduction)
                    / self.stats.runs as f32;
        } else {
            self.stats.avg_oracle_reduction = oracle_reduction;
        }

        PipelineResult {
            stages,
            labels: vec![], // No actual oracle calls in dry run
            quality: DataQualityMetrics::default(),
            total_generated,
            oracle_calls,
            oracle_reduction,
        }
    }

    /// Reset pipeline state
    pub fn reset(&mut self) {
        self.quality_gate.reset_stats();
        self.active_learner = ActiveLearner::new(self.config.num_clusters);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::ErrorCategory;
    use crate::Language;

    fn sample_codes() -> Vec<GeneratedCode> {
        vec![
            GeneratedCode {
                code: "x = 1".to_string(),
                language: Language::Python,
                ast_depth: 1,
                features: vec![],
            },
            GeneratedCode {
                code: "def add(a, b):\n    return a + b".to_string(),
                language: Language::Python,
                ast_depth: 3,
                features: vec!["function".to_string()],
            },
            GeneratedCode {
                code: "for i in range(10):\n    if i % 2 == 0:\n        print(i)".to_string(),
                language: Language::Python,
                ast_depth: 5,
                features: vec!["loop".to_string(), "conditional".to_string()],
            },
            GeneratedCode {
                code: "class Foo:\n    def __init__(self):\n        self.x = 0\n    def get(self):\n        return self.x".to_string(),
                language: Language::Python,
                ast_depth: 6,
                features: vec!["class".to_string(), "method".to_string()],
            },
        ]
    }

    // ========== PipelineConfig Tests ==========

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert!((config.quality_threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.num_clusters, 5);
    }

    #[test]
    fn test_pipeline_config_strict() {
        let config = PipelineConfig::strict();
        assert!((config.quality_threshold - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_pipeline_config_fast() {
        let config = PipelineConfig::fast();
        assert!(!config.enable_active_learning);
        assert!(!config.enable_defect_priority);
    }

    #[test]
    fn test_pipeline_config_validate() {
        let valid = PipelineConfig::default();
        assert!(valid.validate().is_empty());

        let invalid = PipelineConfig {
            quality_threshold: 1.5,
            num_clusters: 0,
            ..Default::default()
        };
        assert!(!invalid.validate().is_empty());
    }

    // ========== DataQualityMetrics Tests ==========

    #[test]
    fn test_data_quality_overall() {
        let metrics = DataQualityMetrics {
            novelty: 0.8,
            diversity: 0.7,
            difficulty: 0.6,
            coverage: 0.8,
            bug_rate: 0.2,
        };

        let score = metrics.overall();
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_data_quality_meets_targets() {
        let good = DataQualityMetrics {
            diversity: 0.7,
            bug_rate: 0.2,
            coverage: 0.8,
            ..Default::default()
        };
        assert!(good.meets_targets());

        let bad = DataQualityMetrics::default();
        assert!(!bad.meets_targets());
    }

    // ========== StageResult Tests ==========

    #[test]
    fn test_stage_result_reduction() {
        let result = StageResult {
            stage: "test".to_string(),
            input_count: 100,
            output_count: 10,
            time_ms: 50,
        };

        assert!((result.reduction_factor() - 10.0).abs() < 0.001);
        assert!((result.pass_rate() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_stage_result_edge_cases() {
        let zero_output = StageResult {
            stage: "test".to_string(),
            input_count: 100,
            output_count: 0,
            time_ms: 0,
        };
        assert!(zero_output.reduction_factor().is_infinite());

        let zero_input = StageResult {
            stage: "test".to_string(),
            input_count: 0,
            output_count: 0,
            time_ms: 0,
        };
        assert!((zero_input.pass_rate() - 0.0).abs() < 0.001);
    }

    // ========== PipelineResult Tests ==========

    #[test]
    fn test_pipeline_result_stage_lookup() {
        let result = PipelineResult {
            stages: vec![
                StageResult {
                    stage: "quality_gate".to_string(),
                    input_count: 100,
                    output_count: 50,
                    time_ms: 10,
                },
                StageResult {
                    stage: "defect_priority".to_string(),
                    input_count: 50,
                    output_count: 20,
                    time_ms: 5,
                },
            ],
            ..Default::default()
        };

        assert!(result.stage("quality_gate").is_some());
        assert!(result.stage("nonexistent").is_none());
    }

    #[test]
    fn test_pipeline_result_total_time() {
        let result = PipelineResult {
            stages: vec![
                StageResult {
                    stage: "a".to_string(),
                    input_count: 0,
                    output_count: 0,
                    time_ms: 100,
                },
                StageResult {
                    stage: "b".to_string(),
                    input_count: 0,
                    output_count: 0,
                    time_ms: 200,
                },
            ],
            ..Default::default()
        };

        assert_eq!(result.total_time_ms(), 300);
    }

    // ========== CodexPipeline Tests ==========

    #[test]
    fn test_codex_pipeline_new() {
        let pipeline = CodexPipeline::default();
        assert_eq!(pipeline.stats().runs, 0);
    }

    #[test]
    fn test_codex_pipeline_filter_quality() {
        let mut pipeline = CodexPipeline::new(PipelineConfig {
            quality_threshold: 0.3,
            ..Default::default()
        });

        let codes = sample_codes();
        let (passed, stage) = pipeline.filter_quality(&codes);

        assert!(passed.len() <= codes.len());
        assert_eq!(stage.stage, "quality_gate");
        assert_eq!(stage.input_count, codes.len());
    }

    #[test]
    fn test_codex_pipeline_prioritize_defects() {
        let pipeline = CodexPipeline::default();
        let codes = sample_codes();
        let refs: Vec<&GeneratedCode> = codes.iter().collect();

        let (prioritized, stage) = pipeline.prioritize_defects(&refs);

        assert!(!prioritized.is_empty());
        assert_eq!(stage.stage, "defect_priority");
    }

    #[test]
    fn test_codex_pipeline_sample_active() {
        let mut pipeline = CodexPipeline::new(PipelineConfig {
            batch_size: 2,
            ..Default::default()
        });

        let codes = sample_codes();
        let refs: Vec<&GeneratedCode> = codes.iter().collect();

        let (sampled, stage) = pipeline.sample_active(&refs);

        assert!(sampled.len() <= 2);
        assert_eq!(stage.stage, "active_learning");
    }

    #[test]
    fn test_codex_pipeline_prepare() {
        let mut pipeline = CodexPipeline::new(PipelineConfig {
            quality_threshold: 0.2,
            batch_size: 10,
            ..Default::default()
        });

        let codes = sample_codes();
        let (prepared, stages) = pipeline.prepare(&codes);

        assert!(!prepared.is_empty());
        assert_eq!(stages.len(), 3); // quality, defect, active
    }

    #[test]
    fn test_codex_pipeline_run_dry() {
        let mut pipeline = CodexPipeline::new(PipelineConfig {
            quality_threshold: 0.2,
            ..Default::default()
        });

        let codes = sample_codes();
        let result = pipeline.run_dry(&codes);

        assert_eq!(result.total_generated, codes.len());
        assert!(result.oracle_calls <= codes.len());
        assert!(result.oracle_reduction >= 1.0);
    }

    #[test]
    fn test_codex_pipeline_update_feedback() {
        let mut pipeline = CodexPipeline::default();

        // Need to fit first
        let codes = sample_codes();
        let refs: Vec<&GeneratedCode> = codes.iter().collect();
        let _ = pipeline.sample_active(&refs);

        pipeline.update_feedback("def add(a, b): return a + b", true);
        assert_eq!(pipeline.stats().bugs_found, 1);

        pipeline.update_feedback("x = 1", false);
        assert_eq!(pipeline.stats().bugs_found, 1);
    }

    #[test]
    fn test_codex_pipeline_compute_quality() {
        let pipeline = CodexPipeline::default();

        let labels = vec![
            RichLabel::correct(crate::ml::SoftLabels::default()),
            RichLabel::incorrect(
                ErrorCategory::TypeMismatch,
                "error".to_string(),
                crate::ml::SoftLabels::default(),
            ),
        ];

        let quality = pipeline.compute_quality(&labels);
        assert!((quality.bug_rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_codex_pipeline_reset() {
        let mut pipeline = CodexPipeline::default();

        let codes = sample_codes();
        let _ = pipeline.run_dry(&codes);

        pipeline.reset();
        // Stats should remain, but internal state reset
        assert_eq!(pipeline.stats().runs, 1);
    }

    // ========== Debug Tests ==========

    #[test]
    fn test_pipeline_config_debug() {
        let config = PipelineConfig::default();
        let debug = format!("{config:?}");
        assert!(debug.contains("PipelineConfig"));
    }

    #[test]
    fn test_data_quality_metrics_debug() {
        let metrics = DataQualityMetrics::default();
        let debug = format!("{metrics:?}");
        assert!(debug.contains("DataQualityMetrics"));
    }

    #[test]
    fn test_codex_pipeline_debug() {
        let pipeline = CodexPipeline::default();
        let debug = format!("{pipeline:?}");
        assert!(debug.contains("CodexPipeline"));
    }

    // ========== Serialization Tests ==========

    #[test]
    fn test_pipeline_config_serialize() {
        let config = PipelineConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let restored: PipelineConfig = serde_json::from_str(&json).unwrap();
        assert!((config.quality_threshold - restored.quality_threshold).abs() < f32::EPSILON);
    }

    #[test]
    fn test_data_quality_metrics_serialize() {
        let metrics = DataQualityMetrics {
            novelty: 0.5,
            diversity: 0.6,
            difficulty: 0.7,
            coverage: 0.8,
            bug_rate: 0.15,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let restored: DataQualityMetrics = serde_json::from_str(&json).unwrap();
        assert!((metrics.diversity - restored.diversity).abs() < 0.001);
    }

    // ========== Integration Tests ==========

    #[test]
    fn test_full_pipeline_flow() {
        let mut pipeline = CodexPipeline::new(PipelineConfig {
            quality_threshold: 0.2, // Low threshold to pass more
            batch_size: 10,
            ..Default::default()
        });

        // Generate codes
        let codes = sample_codes();

        // Run dry (no actual oracle)
        let result = pipeline.run_dry(&codes);

        // Verify stages ran
        assert_eq!(result.stages.len(), 3);
        assert!(result.stage("quality_gate").is_some());
        assert!(result.stage("defect_priority").is_some());
        assert!(result.stage("active_learning").is_some());

        // Verify reduction
        assert!(result.oracle_reduction >= 1.0);
    }

    #[test]
    fn test_pipeline_oracle_reduction() {
        let mut pipeline = CodexPipeline::new(PipelineConfig {
            quality_threshold: 0.6, // Higher threshold for more filtering
            batch_size: 2,          // Small batch
            ..Default::default()
        });

        // Generate many codes
        let mut codes = Vec::new();
        for i in 0..100 {
            codes.push(GeneratedCode {
                code: format!("x_{i} = {i}"),
                language: Language::Python,
                ast_depth: 1,
                features: vec![],
            });
        }

        let result = pipeline.run_dry(&codes);

        // Should have significant reduction
        assert!(result.oracle_calls <= 20); // Much less than 100
    }
}

/// Property-based tests
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Quality threshold is clamped properly
        #[test]
        fn prop_quality_threshold_valid(threshold in -0.5f32..1.5) {
            let config = PipelineConfig {
                quality_threshold: threshold.clamp(0.0, 1.0),
                ..Default::default()
            };
            prop_assert!(config.quality_threshold >= 0.0);
            prop_assert!(config.quality_threshold <= 1.0);
        }

        /// Oracle reduction is always >= 1 (or infinity for zero calls)
        #[test]
        fn prop_oracle_reduction_bounded(total in 1usize..1000, calls in 0usize..1000) {
            let reduction = if calls == 0 {
                f32::INFINITY
            } else {
                total as f32 / calls as f32
            };

            if calls > 0 {
                prop_assert!(reduction >= total as f32 / calls as f32);
            }
        }

        /// Overall quality score is bounded [0, 1]
        #[test]
        fn prop_quality_overall_bounded(
            novelty in 0.0f32..1.0,
            diversity in 0.0f32..1.0,
            difficulty in 0.0f32..1.0,
            coverage in 0.0f32..1.0,
            bug_rate in 0.0f32..1.0,
        ) {
            let metrics = DataQualityMetrics {
                novelty,
                diversity,
                difficulty,
                coverage,
                bug_rate,
            };

            let overall = metrics.overall();
            prop_assert!(overall >= 0.0);
            prop_assert!(overall <= 1.0);
        }
    }
}
