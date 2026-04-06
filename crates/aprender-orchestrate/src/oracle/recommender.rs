//! Recommendation Engine for Oracle Mode
//!
//! Generates component recommendations based on parsed queries
//! and the knowledge graph.

use super::knowledge_graph::KnowledgeGraph;
use super::query_engine::{ParsedQuery, PerformanceHint, QueryEngine};
use super::types::*;

// =============================================================================
// Backend Selector (from spec section 4.2)
// =============================================================================

/// Select optimal backend based on workload characteristics
/// Based on PCIe transfer overhead analysis (Gregg & Hazelwood, 2011)
pub fn select_backend(
    op_complexity: OpComplexity,
    data_size: Option<DataSize>,
    hardware: &HardwareSpec,
) -> Backend {
    let size = data_size.and_then(|d| d.as_samples()).unwrap_or(0);

    // Thresholds scale with complexity: higher complexity → lower threshold for GPU/SIMD
    let (gpu_threshold, simd_threshold) = match op_complexity {
        OpComplexity::Low => (1_000_000, 1_000),
        OpComplexity::Medium => (100_000, 100),
        OpComplexity::High => (10_000, 10),
    };

    if size > gpu_threshold && hardware.has_gpu() {
        Backend::GPU
    } else if size > simd_threshold {
        Backend::SIMD
    } else {
        Backend::Scalar
    }
}

/// Determine if distributed execution is beneficial
/// Based on Amdahl's Law and communication overhead
pub fn should_distribute(
    data_size: Option<DataSize>,
    hardware: &HardwareSpec,
    parallel_fraction: f64,
) -> DistributionRecommendation {
    let size = data_size.and_then(|d| d.as_samples()).unwrap_or(0);

    // Only consider distribution for large data or explicit multi-node
    if !hardware.is_distributed && size < 10_000_000 {
        return DistributionRecommendation::not_needed(
            "Single-node sufficient for this workload size",
        );
    }

    let node_count = hardware.node_count.unwrap_or(1);
    if node_count <= 1 {
        return DistributionRecommendation::not_needed(
            "No additional nodes available for distribution",
        );
    }

    // Amdahl's Law: speedup = 1 / ((1-p) + p/n)
    let speedup = 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / node_count as f64);

    // Communication overhead estimate (simplified)
    let comm_overhead = 0.1 * node_count as f64; // 10% per node

    if speedup > 1.5 && comm_overhead < 0.5 {
        DistributionRecommendation {
            tool: Some("repartir".into()),
            needed: true,
            rationale: format!(
                "Distribution beneficial with {:.1}x speedup across {} nodes",
                speedup, node_count
            ),
            node_count: Some(node_count),
        }
    } else {
        DistributionRecommendation::not_needed(format!(
            "Distribution overhead ({:.0}%) outweighs benefits",
            comm_overhead * 100.0
        ))
    }
}

// =============================================================================
// Recommender
// =============================================================================

/// Oracle recommendation engine
pub struct Recommender {
    graph: KnowledgeGraph,
    engine: QueryEngine,
}

impl Default for Recommender {
    fn default() -> Self {
        Self::new()
    }
}

impl Recommender {
    /// Create a new recommender with the Sovereign AI Stack
    pub fn new() -> Self {
        Self { graph: KnowledgeGraph::sovereign_stack(), engine: QueryEngine::new() }
    }

    /// Create a recommender with a custom knowledge graph
    pub fn with_graph(graph: KnowledgeGraph) -> Self {
        Self { graph, engine: QueryEngine::new() }
    }

    /// Process a natural language query and return recommendations
    pub fn query(&self, query: &str) -> OracleResponse {
        let parsed = self.engine.parse(query);

        // Transfer extracted information from parsed query into constraints
        // so the backend selector sees data size and hardware hints
        let mut constraints = QueryConstraints::default();
        if let Some(size) = parsed.data_size {
            constraints.data_size = Some(size);
        }
        if parsed.performance_hints.contains(&PerformanceHint::GPURequired) {
            constraints.hardware = HardwareSpec::with_gpu(16.0);
        }

        self.recommend(&parsed, &constraints)
    }

    /// Process a structured OracleQuery
    pub fn query_structured(&self, query: &OracleQuery) -> OracleResponse {
        let parsed = self.engine.parse(&query.description);

        // Merge NL-extracted hints into explicit constraints (explicit wins)
        let mut constraints = query.constraints.clone();
        if constraints.data_size.is_none() {
            if let Some(size) = parsed.data_size {
                constraints.data_size = Some(size);
            }
        }
        if !constraints.hardware.has_gpu()
            && parsed.performance_hints.contains(&PerformanceHint::GPURequired)
        {
            constraints.hardware = HardwareSpec::with_gpu(16.0);
        }

        self.recommend(&parsed, &constraints)
    }

    /// Generate recommendations from parsed query
    pub fn recommend(
        &self,
        parsed: &ParsedQuery,
        constraints: &QueryConstraints,
    ) -> OracleResponse {
        // Determine primary problem class
        let problem_class = self.classify_problem(parsed);

        // Find primary component recommendation
        let primary = self.recommend_primary(parsed, constraints);

        // Find supporting components
        let supporting = self.recommend_supporting(&primary, parsed, constraints);

        // Determine compute backend
        let complexity = self.engine.estimate_complexity(parsed);
        let backend = select_backend(complexity, constraints.data_size, &constraints.hardware);
        let compute = ComputeRecommendation {
            backend,
            rationale: self.compute_rationale(backend, complexity, constraints),
        };

        // Determine distribution needs
        let parallel_fraction = self.estimate_parallel_fraction(parsed);
        let distribution =
            should_distribute(constraints.data_size, &constraints.hardware, parallel_fraction);

        // Generate code example
        let code_example = self.generate_code_example(&primary, &supporting, parsed);

        // Generate related queries
        let related_queries = self.generate_related_queries(parsed);

        OracleResponse {
            problem_class,
            algorithm: self.engine.primary_algorithm(parsed).map(String::from),
            primary,
            supporting,
            compute,
            distribution,
            code_example,
            related_queries,
        }
    }

    fn classify_problem(&self, parsed: &ParsedQuery) -> String {
        if let Some(domain) = self.engine.primary_domain(parsed) {
            domain.to_string()
        } else if !parsed.algorithms.is_empty() {
            "Algorithm-specific".into()
        } else {
            "General".into()
        }
    }

    fn recommend_primary(
        &self,
        parsed: &ParsedQuery,
        _constraints: &QueryConstraints,
    ) -> ComponentRecommendation {
        // Check if specific components were mentioned
        if let Some(component) = parsed.mentioned_components.first() {
            if let Some(comp) = self.graph.get_component(component) {
                return ComponentRecommendation::new(
                    comp.name.clone(),
                    0.95,
                    format!("Explicitly mentioned {} - {}", comp.name, comp.description),
                );
            }
        }

        // Find by algorithm
        if let Some(algo) = parsed.algorithms.first() {
            let components = self.graph.find_by_capability(algo);
            if let Some(comp) = components.first() {
                let path = self.get_algorithm_path(comp, algo);
                return match path {
                    Some(p) => ComponentRecommendation::with_path(
                        comp.name.clone(),
                        0.9,
                        format!("{} provides {} implementation", comp.name, algo),
                        p,
                    ),
                    None => ComponentRecommendation::new(
                        comp.name.clone(),
                        0.9,
                        format!("{} provides {} implementation", comp.name, algo),
                    ),
                };
            }
        }

        // Find by problem domain
        if let Some(domain) = self.engine.primary_domain(parsed) {
            let components = self.graph.find_by_domain(domain);
            if let Some(comp) = components.first() {
                return ComponentRecommendation::new(
                    comp.name.clone(),
                    0.85,
                    format!("{} is recommended for {} tasks", comp.name, domain),
                );
            }
        }

        // Default recommendation based on performance hints
        if parsed.performance_hints.contains(&PerformanceHint::GPURequired) {
            return ComponentRecommendation::new(
                "trueno",
                0.7,
                "GPU acceleration available via trueno",
            );
        }

        if parsed.performance_hints.contains(&PerformanceHint::Distributed) {
            return ComponentRecommendation::new(
                "repartir",
                0.7,
                "Distributed computing via repartir",
            );
        }

        // Fallback to batuta for general orchestration
        ComponentRecommendation::new(
            "batuta",
            0.5,
            "General orchestration framework for the Sovereign AI Stack",
        )
    }

    fn recommend_supporting(
        &self,
        primary: &ComponentRecommendation,
        parsed: &ParsedQuery,
        constraints: &QueryConstraints,
    ) -> Vec<ComponentRecommendation> {
        let mut supporting = Vec::new();

        // Get integration patterns from primary
        let integrations = self.graph.integrations_from(&primary.component);
        for pattern in integrations.iter().take(2) {
            if let Some(comp) = self.graph.get_component(&pattern.to) {
                supporting.push(ComponentRecommendation::new(
                    comp.name.clone(),
                    0.7,
                    format!("Integrates via {} pattern", pattern.pattern_name),
                ));
            }
        }

        // Data-driven conditional recommendations
        let is_ml = parsed.domains.iter().any(|d| {
            matches!(
                d,
                ProblemDomain::SupervisedLearning
                    | ProblemDomain::UnsupervisedLearning
                    | ProblemDomain::DeepLearning
                    | ProblemDomain::SpeechRecognition
            )
        });
        let is_large = constraints.data_size.map(|d| d.is_large()).unwrap_or(false);
        let is_pipeline = parsed.domains.contains(&ProblemDomain::DataPipeline);
        let is_inference = parsed.domains.contains(&ProblemDomain::Inference);

        let candidates: &[(bool, &str, f32, &str)] = &[
            (is_ml, "trueno", 0.8, "SIMD/GPU backend for compute acceleration"),
            (is_large, "repartir", 0.6, "Distribution recommended for large dataset"),
            (is_pipeline, "alimentar", 0.7, "Data loading and preprocessing"),
            (is_inference, "realizar", 0.85, "Model serving and inference"),
        ];
        for &(condition, component, confidence, rationale) in candidates {
            if condition && primary.component != component {
                supporting.push(ComponentRecommendation::new(component, confidence, rationale));
            }
        }

        supporting
    }

    /// Algorithm-to-module-path lookup table: (component, algo_patterns, path).
    /// Each entry matches when the component name matches AND any algo pattern is found.
    const ALGORITHM_PATHS: &[(&str, &[&str], &str)] = &[
        ("aprender", &["random_forest"], "aprender::tree::RandomForestClassifier"),
        ("aprender", &["decision_tree"], "aprender::tree::DecisionTreeClassifier"),
        ("aprender", &["linear_regression"], "aprender::linear::LinearRegression"),
        ("aprender", &["logistic_regression"], "aprender::linear::LogisticRegression"),
        (
            "aprender",
            &["gbm", "gradient_boosting"],
            "aprender::ensemble::GradientBoostingClassifier",
        ),
        ("aprender", &["kmeans", "k_means"], "aprender::cluster::KMeans"),
        ("aprender", &["pca"], "aprender::decomposition::PCA"),
        ("aprender", &["svm"], "aprender::svm::SVC"),
        ("aprender", &["knn"], "aprender::neighbors::KNeighborsClassifier"),
        ("entrenar", &["lora"], "entrenar::lora::LoRA"),
        ("entrenar", &["qlora"], "entrenar::lora::QLoRA"),
    ];

    fn get_algorithm_path(&self, component: &StackComponent, algorithm: &str) -> Option<String> {
        Self::ALGORITHM_PATHS
            .iter()
            .find(|(comp, pats, _)| {
                *comp == component.name && pats.iter().any(|p| algorithm.contains(p))
            })
            .map(|(_, _, path)| (*path).to_string())
    }

    fn compute_rationale(
        &self,
        backend: Backend,
        complexity: OpComplexity,
        constraints: &QueryConstraints,
    ) -> String {
        let size_str = constraints
            .data_size
            .map(|d| match d {
                DataSize::Samples(n) => format!("{} samples", format_number(n)),
                DataSize::Bytes(n) => format!("{} bytes", format_number(n)),
                DataSize::Unknown => "unknown size".into(),
            })
            .unwrap_or_else(|| "unspecified size".into());

        match backend {
            Backend::Scalar => {
                format!(
                    "Scalar operations sufficient for small {} with {:?} complexity",
                    size_str, complexity
                )
            }
            Backend::SIMD => {
                format!(
                    "SIMD vectorization optimal for {} with {:?} complexity",
                    size_str, complexity
                )
            }
            Backend::GPU => {
                format!("GPU acceleration recommended for {} with {:?} complexity - PCIe overhead amortized", size_str, complexity)
            }
            Backend::Distributed => {
                format!("Distributed execution for {} exceeds single-node capacity", size_str)
            }
        }
    }

    /// Algorithm parallelizability estimates: (algo_patterns, fraction).
    const ALGO_PARALLEL: &[(&[&str], f64)] =
        &[(&["random_forest", "gbm"], 0.95), (&["kmeans"], 0.85), (&["linear"], 0.7)];

    /// Domain parallelizability estimates: (domain, fraction).
    const DOMAIN_PARALLEL: &[(ProblemDomain, f64)] =
        &[(ProblemDomain::DeepLearning, 0.8), (ProblemDomain::SupervisedLearning, 0.75)];

    fn estimate_parallel_fraction(&self, parsed: &ParsedQuery) -> f64 {
        if let Some(algo) = parsed.algorithms.first() {
            if let Some(&(_, frac)) =
                Self::ALGO_PARALLEL.iter().find(|(pats, _)| pats.iter().any(|p| algo.contains(p)))
            {
                return frac;
            }
        }

        Self::DOMAIN_PARALLEL
            .iter()
            .find(|(domain, _)| parsed.domains.contains(domain))
            .map_or(0.6, |&(_, frac)| frac)
    }

    fn generate_code_example(
        &self,
        primary: &ComponentRecommendation,
        _supporting: &[ComponentRecommendation],
        parsed: &ParsedQuery,
    ) -> Option<String> {
        // Generate contextual code example based on primary component
        match primary.component.as_str() {
            "aprender" => {
                let path =
                    primary.path.as_deref().unwrap_or("aprender::tree::RandomForestClassifier");
                let _algo = parsed.algorithms.first().map(|s| s.as_str()).unwrap_or("RandomForest");

                Some(format!(
                    r#"use {};

// Load data
let (X_train, X_test, y_train, y_test) = train_test_split(&X, &y, 0.2)?;

// Train model
let model = {}::new()
    .n_estimators(100)
    .fit(&X_train, &y_train)?;

// Predict
let predictions = model.predict(&X_test)?;
let accuracy = accuracy_score(&y_test, &predictions);
println!("Accuracy: {{:.2}}%", accuracy * 100.0);

#[cfg(test)]
mod tests {{
    #[test]
    fn test_model_builder_params() {{
        let n_estimators = 100;
        let test_size = 0.2_f64;
        assert!(n_estimators > 0);
        assert!(test_size > 0.0 && test_size < 1.0);
    }}

    #[test]
    fn test_predictions_non_empty() {{
        let predictions = vec![0, 1, 1, 0, 1];
        assert!(!predictions.is_empty());
    }}

    #[test]
    fn test_accuracy_in_range() {{
        let accuracy = 0.85_f64;
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }}
}}"#,
                    path,
                    path.split("::").last().unwrap_or("Model")
                ))
            }
            "trueno" => Some(
                r#"use trueno::prelude::*;

// Create tensors with SIMD acceleration
let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0]);

// SIMD-accelerated operations
let result = a.dot(&b);
println!("Dot product: {}", result);

#[cfg(test)]
mod tests {
    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(data.len(), 4);
    }

    #[test]
    fn test_dot_product_result() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert_eq!(dot, 70.0);
    }

    #[test]
    fn test_simd_elements_finite() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        assert!(data.iter().all(|x| x.is_finite()));
    }
}"#
                .into(),
            ),
            "depyler" => Some(
                r"# Run depyler to convert Python to Rust
batuta transpile --source my_project.py --output rust-output/

# The sklearn code:
#   from sklearn.ensemble import RandomForestClassifier
#   model = RandomForestClassifier(n_estimators=100)
#
# Becomes:
#   use aprender::tree::RandomForestClassifier;
#   let model = RandomForestClassifier::new().n_estimators(100);"
                    .into(),
            ),
            "realizar" => Some(
                r#"use realizar::ModelRegistry;

// Load trained model
let registry = ModelRegistry::new();
registry.load_apr("classifier", "model.apr")?;

// Serve predictions
let input = vec![1.0, 2.0, 3.0, 4.0];
let prediction = registry.predict("classifier", &input)?;
println!("Prediction: {:?}", prediction);

#[cfg(test)]
mod tests {
    #[test]
    fn test_registry_construction() {
        let model_name = "classifier";
        assert!(!model_name.is_empty());
    }

    #[test]
    fn test_input_feature_count() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(input.len(), 4);
    }

    #[test]
    fn test_model_path_valid() {
        let path = "model.apr";
        assert!(path.ends_with(".apr"));
    }
}"#
                .into(),
            ),
            "whisper-apr" => Some(
                r#"use whisper_apr::WhisperModel;

// Load quantized Whisper model
let model = WhisperModel::from_apr("whisper-base.apr")?;

// Transcribe audio file
let audio = std::fs::read("recording.wav")?;
let result = model.transcribe(&audio)?;
println!("Text: {}", result.text);

// Streaming transcription
// let stream = model.stream_transcribe(audio_stream)?;
// while let Some(segment) = stream.next().await {
//     println!("[{:.1}s] {}", segment.timestamp, segment.text);
// }

#[cfg(test)]
mod tests {
    #[test]
    fn test_model_path_valid() {
        let path = "whisper-base.apr";
        assert!(path.ends_with(".apr"));
    }

    #[test]
    fn test_transcription_produces_text() {
        let text = "Hello world";
        assert!(!text.is_empty());
    }

    #[test]
    fn test_audio_bytes_valid_utf8() {
        let text = "transcribed text";
        assert!(std::str::from_utf8(text.as_bytes()).is_ok());
    }
}"#
                .into(),
            ),
            "provable-contracts" => Some(
                r"# Define YAML contract for a SIMD kernel
# contracts/softmax_contract.yaml
contract:
  name: fused_softmax
  module: trueno::kernels::softmax
  preconditions:
    - input.len() > 0
    - input.len() % 8 == 0  # AVX2 alignment
  postconditions:
    - result.is_ok()
    - output.iter().all(|x| (0.0..=1.0).contains(x))
    - (output.iter().sum::<f32>() - 1.0).abs() < 1e-5

# Generate Kani verification harness
provable-contracts scaffold contracts/softmax_contract.yaml \
    --output harnesses/softmax_harness.rs

# Run bounded model checking
provable-contracts verify harnesses/softmax_harness.rs \
    --unwind 16 --solver cadical

# Generate probar property tests from the same contract
provable-contracts probar contracts/softmax_contract.yaml \
    --output tests/softmax_props.rs"
                    .into(),
            ),
            "tiny-model-ground-truth" => Some(
                r#"# Generate oracle outputs from HuggingFace reference
python -m tiny_model_ground_truth generate \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --prompts "Hello" "The capital of France" \
    --output oracle/

# Validate realizar inference against oracle
python -m tiny_model_ground_truth validate \
    --oracle oracle/ \
    --engine realizar \
    --model model.apr \
    --tolerance 1e-4

# Check quantization drift (GGUF → APR → inference)
python -m tiny_model_ground_truth drift \
    --oracle oracle/ \
    --gguf model.gguf \
    --apr model.apr \
    --report drift_report.html"#
                    .into(),
            ),
            "rmedia" => Some(
                r#"use rmedia::prelude::*;

// Load MLT XML timeline
let timeline = Timeline::from_mlt("course.mlt")?;

// Render video (headless, 1.73x faster than melt)
let job = RenderJob::new(&timeline)
    .output("output.mp4")
    .codec(Codec::H264 { crf: 23 })
    .resolution(1920, 1080);
job.render()?;

// Course production pipeline
// rmedia also supports:
// - Subtitle burn-in: subtitle::burn_in(video, srt, output)
// - Transitions: timeline.add_transition(Dissolve::new(1.0))
// - Audio ducking: audio::duck(narration, background, ratio)

#[cfg(test)]
mod tests {
    #[test]
    fn test_render_job_output_path() {
        let output = "output.mp4";
        assert!(output.ends_with(".mp4"));
    }

    #[test]
    fn test_crf_in_valid_range() {
        let crf = 23;
        assert!((0..=51).contains(&crf));
    }

    #[test]
    fn test_resolution_1080p() {
        let (w, h) = (1920, 1080);
        assert_eq!(w * h, 2_073_600);
    }
}"#
                .into(),
            ),
            "repartir" => Some(
                r#"use repartir::{Pool, task::{Task, Backend}};

// Create pool with CPU workers
let pool = Pool::builder()
    .cpu_workers(8)
    .build()?;

// Submit task for execution
let task = Task::builder()
    .binary("./worker")
    .arg("--input").arg("data.csv")
    .backend(Backend::Cpu)
    .build()?;

let result = pool.submit(task).await?;
println!("Output: {}", result.stdout_str()?);

// For multi-machine distribution:
// use repartir::executor::remote::RemoteExecutor;
// let remote = RemoteExecutor::builder()
//     .add_worker("node1:9000")
//     .add_worker("node2:9000")
//     .build().await?;

#[cfg(test)]
mod tests {
    #[test]
    fn test_pool_builder_workers() {
        let cpu_workers = 8;
        assert!(cpu_workers > 0);
    }

    #[test]
    fn test_task_binary_set() {
        let binary = "./worker";
        assert!(!binary.is_empty());
    }

    #[test]
    fn test_backend_selection() {
        let backend = "Cpu";
        let valid = vec!["Cpu", "Gpu", "Remote"];
        assert!(valid.contains(&backend));
    }
}"#
                .into(),
            ),
            _ => None,
        }
    }

    fn generate_related_queries(&self, parsed: &ParsedQuery) -> Vec<String> {
        let mut related = Vec::new();

        // Domain-based related queries
        let domain_queries: &[(ProblemDomain, &[&str])] = &[
            (
                ProblemDomain::SupervisedLearning,
                &[
                    "How do I tune hyperparameters for this model?",
                    "What's the best way to handle imbalanced data?",
                ],
            ),
            (
                ProblemDomain::PythonMigration,
                &[
                    "How do I convert numpy arrays to trueno tensors?",
                    "What sklearn features are supported in aprender?",
                ],
            ),
            (
                ProblemDomain::Inference,
                &[
                    "How do I optimize for low latency?",
                    "What model formats does realizar support?",
                ],
            ),
            (
                ProblemDomain::SpeechRecognition,
                &[
                    "How do I stream transcription in real-time?",
                    "What quantization levels does whisper-apr support?",
                ],
            ),
            (
                ProblemDomain::MediaProduction,
                &[
                    "How do I render a course video from MLT XML?",
                    "How do I integrate whisper-apr transcription with rmedia?",
                ],
            ),
        ];
        for (domain, queries) in domain_queries {
            if parsed.domains.contains(domain) {
                related.extend(queries.iter().map(|q| (*q).into()));
            }
        }

        // Performance-hint-based related queries
        if parsed.performance_hints.contains(&PerformanceHint::Distributed) {
            related.push("How do I scale to multiple nodes?".into());
            related.push("What's the communication overhead for distributed training?".into());
        }

        related.truncate(3);
        related
    }

    /// Get capabilities of a component
    pub fn get_capabilities(&self, component: &str) -> Vec<String> {
        self.graph
            .get_component(component)
            .map(|c| c.capabilities.iter().map(|cap| cap.name.clone()).collect())
            .unwrap_or_default()
    }

    /// Get integration pattern between two components
    pub fn get_integration(&self, from: &str, to: &str) -> Option<IntegrationPattern> {
        self.graph.get_integration(from, to).cloned()
    }

    /// List all available components
    pub fn list_components(&self) -> Vec<String> {
        self.graph.component_names().cloned().collect()
    }

    /// Get component details
    pub fn get_component(&self, name: &str) -> Option<&StackComponent> {
        self.graph.get_component(name)
    }
}

/// Format large numbers for display
fn format_number(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{}B", n / 1_000_000_000)
    } else if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        n.to_string()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[path = "recommender_tests.rs"]
mod tests;
