//! Query Engine for Oracle Mode
//!
//! Parses natural language queries and extracts structured information
//! for the recommendation engine.

use super::types::*;
use std::collections::HashSet;

// =============================================================================
// Table-Driven Constants
// =============================================================================

/// Keyword-to-PerformanceHint mapping for simple OR-match patterns.
/// Each entry: a list of keywords (any match triggers the hint) and the hint variant.
///
/// Note: LowLatency also has a compound check ("<" AND "ms") handled separately.
/// Note: LowMemory uses AND logic ("memory" AND ("low" OR "efficient")) handled separately.
const HINT_PATTERNS: &[(&[&str], PerformanceHint)] = &[
    (&["fast", "low latency", "real-time", "realtime"], PerformanceHint::LowLatency),
    (&["throughput", "high volume"], PerformanceHint::HighThroughput),
    (&["gpu"], PerformanceHint::GPURequired),
    (&["distributed", "multi-node", "cluster"], PerformanceHint::Distributed),
    (&["edge", "embedded", "iot"], PerformanceHint::EdgeDeployment),
    (&["sovereign", "gdpr", "local only", "eu ai act", "on-premise"], PerformanceHint::Sovereign),
];

/// Domain-to-OpComplexity mapping for estimate_complexity().
/// Checked in order; first match wins. Domains not listed fall through to Low.
const DOMAIN_COMPLEXITY: &[(ProblemDomain, OpComplexity)] = &[
    (ProblemDomain::DeepLearning, OpComplexity::High),
    (ProblemDomain::SpeechRecognition, OpComplexity::High),
    (ProblemDomain::GraphAnalytics, OpComplexity::Medium),
    (ProblemDomain::SupervisedLearning, OpComplexity::Medium),
    (ProblemDomain::UnsupervisedLearning, OpComplexity::Medium),
    (ProblemDomain::MediaProduction, OpComplexity::Medium),
];

// =============================================================================
// Query Parser
// =============================================================================

/// Parsed query with extracted information
#[derive(Debug, Clone)]
pub struct ParsedQuery {
    /// Original query text
    pub original: String,
    /// Detected problem domains
    pub domains: Vec<ProblemDomain>,
    /// Detected algorithms/techniques
    pub algorithms: Vec<String>,
    /// Extracted keywords
    pub keywords: Vec<String>,
    /// Detected data size indicators
    pub data_size: Option<DataSize>,
    /// Performance requirements detected
    pub performance_hints: Vec<PerformanceHint>,
    /// Component mentions
    pub mentioned_components: Vec<String>,
}

/// Performance hint extracted from query
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceHint {
    LowLatency,
    HighThroughput,
    LowMemory,
    GPURequired,
    Distributed,
    EdgeDeployment,
    Sovereign,
}

/// Query parser for natural language queries
#[derive(Debug, Default)]
pub(crate) struct QueryParser {
    /// Known algorithm keywords
    algorithm_keywords: HashSet<String>,
    /// Problem domain keywords
    domain_keywords: Vec<(String, ProblemDomain)>,
    /// Component names
    component_names: HashSet<String>,
}

impl QueryParser {
    /// Create a new query parser
    pub(crate) fn new() -> Self {
        let mut parser = Self::default();
        parser.initialize_keywords();
        parser
    }

    fn initialize_keywords(&mut self) {
        // Algorithm keywords
        self.algorithm_keywords.extend(
            [
                "random_forest",
                "random forest",
                "randomforest",
                "linear_regression",
                "linear regression",
                "linearregression",
                "logistic_regression",
                "logistic regression",
                "logisticregression",
                "decision_tree",
                "decision tree",
                "decisiontree",
                "gradient_boosting",
                "gradient boosting",
                "gbm",
                "xgboost",
                "lightgbm",
                "naive_bayes",
                "naive bayes",
                "naivebayes",
                "knn",
                "k-nearest",
                "nearest neighbor",
                "svm",
                "support vector",
                "supportvector",
                "kmeans",
                "k-means",
                "clustering",
                "pca",
                "principal component",
                "dimensionality reduction",
                "dbscan",
                "density clustering",
                "neural network",
                "deep learning",
                "transformer",
                "llm",
                "lora",
                "qlora",
                "fine-tuning",
                "fine tuning",
                "finetuning",
                "whisper",
                "speech recognition",
                "speech-to-text",
                "transcription",
                "asr",
            ]
            .map(String::from),
        );

        // Domain keywords
        self.domain_keywords = vec![
            // Supervised Learning
            ("classify".into(), ProblemDomain::SupervisedLearning),
            ("classification".into(), ProblemDomain::SupervisedLearning),
            ("predict".into(), ProblemDomain::SupervisedLearning),
            ("regression".into(), ProblemDomain::SupervisedLearning),
            ("train".into(), ProblemDomain::SupervisedLearning),
            ("supervised".into(), ProblemDomain::SupervisedLearning),
            // Unsupervised Learning
            ("cluster".into(), ProblemDomain::UnsupervisedLearning),
            ("clustering".into(), ProblemDomain::UnsupervisedLearning),
            ("unsupervised".into(), ProblemDomain::UnsupervisedLearning),
            ("anomaly".into(), ProblemDomain::UnsupervisedLearning),
            ("outlier".into(), ProblemDomain::UnsupervisedLearning),
            // Deep Learning
            ("neural".into(), ProblemDomain::DeepLearning),
            ("deep learning".into(), ProblemDomain::DeepLearning),
            ("transformer".into(), ProblemDomain::DeepLearning),
            ("llm".into(), ProblemDomain::DeepLearning),
            ("fine-tune".into(), ProblemDomain::DeepLearning),
            ("lora".into(), ProblemDomain::DeepLearning),
            // Inference
            ("serve".into(), ProblemDomain::Inference),
            ("serving".into(), ProblemDomain::Inference),
            ("inference".into(), ProblemDomain::Inference),
            ("deploy".into(), ProblemDomain::Inference),
            ("production".into(), ProblemDomain::Inference),
            // Speech Recognition
            ("speech".into(), ProblemDomain::SpeechRecognition),
            ("whisper".into(), ProblemDomain::SpeechRecognition),
            ("asr".into(), ProblemDomain::SpeechRecognition),
            ("transcription".into(), ProblemDomain::SpeechRecognition),
            ("speech-to-text".into(), ProblemDomain::SpeechRecognition),
            ("speech recognition".into(), ProblemDomain::SpeechRecognition),
            // Linear Algebra
            ("matrix".into(), ProblemDomain::LinearAlgebra),
            ("tensor".into(), ProblemDomain::LinearAlgebra),
            ("vector".into(), ProblemDomain::LinearAlgebra),
            ("linear algebra".into(), ProblemDomain::LinearAlgebra),
            ("simd".into(), ProblemDomain::LinearAlgebra),
            // Vector Search
            ("similarity".into(), ProblemDomain::VectorSearch),
            ("embedding".into(), ProblemDomain::VectorSearch),
            ("vector search".into(), ProblemDomain::VectorSearch),
            ("nearest neighbor".into(), ProblemDomain::VectorSearch),
            // Graph Analytics
            ("graph".into(), ProblemDomain::GraphAnalytics),
            ("pagerank".into(), ProblemDomain::GraphAnalytics),
            ("pathfinding".into(), ProblemDomain::GraphAnalytics),
            ("community".into(), ProblemDomain::GraphAnalytics),
            // Python Migration
            ("python".into(), ProblemDomain::PythonMigration),
            ("sklearn".into(), ProblemDomain::PythonMigration),
            ("scikit".into(), ProblemDomain::PythonMigration),
            ("numpy".into(), ProblemDomain::PythonMigration),
            ("pandas".into(), ProblemDomain::PythonMigration),
            ("pytorch".into(), ProblemDomain::PythonMigration),
            // C Migration
            ("c code".into(), ProblemDomain::CMigration),
            ("c++".into(), ProblemDomain::CMigration),
            ("cpp".into(), ProblemDomain::CMigration),
            // Shell Migration
            ("bash".into(), ProblemDomain::ShellMigration),
            ("shell".into(), ProblemDomain::ShellMigration),
            ("script".into(), ProblemDomain::ShellMigration),
            // Distribution
            ("distributed".into(), ProblemDomain::DistributedCompute),
            ("parallel".into(), ProblemDomain::DistributedCompute),
            ("multi-node".into(), ProblemDomain::DistributedCompute),
            ("cluster".into(), ProblemDomain::DistributedCompute),
            // Data Pipeline
            ("data loading".into(), ProblemDomain::DataPipeline),
            ("csv".into(), ProblemDomain::DataPipeline),
            ("parquet".into(), ProblemDomain::DataPipeline),
            ("etl".into(), ProblemDomain::DataPipeline),
            // Model Serving
            ("lambda".into(), ProblemDomain::ModelServing),
            ("serverless".into(), ProblemDomain::ModelServing),
            ("container".into(), ProblemDomain::ModelServing),
            ("edge".into(), ProblemDomain::ModelServing),
            // Quality
            ("test".into(), ProblemDomain::Testing),
            ("coverage".into(), ProblemDomain::Testing),
            ("mutation".into(), ProblemDomain::Testing),
            ("profile".into(), ProblemDomain::Profiling),
            ("trace".into(), ProblemDomain::Profiling),
            ("syscall".into(), ProblemDomain::Profiling),
            ("validate".into(), ProblemDomain::Validation),
            ("quality".into(), ProblemDomain::Validation),
            // Formal Verification
            ("formal verification".into(), ProblemDomain::Validation),
            ("kani".into(), ProblemDomain::Validation),
            ("contract".into(), ProblemDomain::Validation),
            ("provable".into(), ProblemDomain::Validation),
            ("proof".into(), ProblemDomain::Validation),
            ("harness".into(), ProblemDomain::Validation),
            ("bounded model checking".into(), ProblemDomain::Validation),
            ("verification".into(), ProblemDomain::Validation),
            // Model Parity / Ground Truth
            ("parity".into(), ProblemDomain::Testing),
            ("falsification".into(), ProblemDomain::Testing),
            ("ground truth".into(), ProblemDomain::Testing),
            ("oracle test".into(), ProblemDomain::Testing),
            ("conversion parity".into(), ProblemDomain::Testing),
            ("quantization drift".into(), ProblemDomain::Testing),
            // Media Production
            ("video".into(), ProblemDomain::MediaProduction),
            ("render".into(), ProblemDomain::MediaProduction),
            ("mlt".into(), ProblemDomain::MediaProduction),
            ("encode".into(), ProblemDomain::MediaProduction),
            ("decode".into(), ProblemDomain::MediaProduction),
            ("transition".into(), ProblemDomain::MediaProduction),
            ("media".into(), ProblemDomain::MediaProduction),
            ("editing".into(), ProblemDomain::MediaProduction),
            ("ffmpeg".into(), ProblemDomain::MediaProduction),
            ("course".into(), ProblemDomain::MediaProduction),
            ("screencast".into(), ProblemDomain::MediaProduction),
            ("dissolve".into(), ProblemDomain::MediaProduction),
            ("fade".into(), ProblemDomain::MediaProduction),
            ("title card".into(), ProblemDomain::MediaProduction),
            ("audio".into(), ProblemDomain::MediaProduction),
            ("transcribe".into(), ProblemDomain::MediaProduction),
            ("subtitle".into(), ProblemDomain::MediaProduction),
            ("caption".into(), ProblemDomain::MediaProduction),
            ("vocabulary".into(), ProblemDomain::MediaProduction),
            ("key terms".into(), ProblemDomain::MediaProduction),
            ("reflection".into(), ProblemDomain::MediaProduction),
            ("landing page".into(), ProblemDomain::MediaProduction),
            ("outline".into(), ProblemDomain::MediaProduction),
            ("syllabus".into(), ProblemDomain::MediaProduction),
            ("svg".into(), ProblemDomain::MediaProduction),
            ("banner".into(), ProblemDomain::MediaProduction),
            ("thumbnail".into(), ProblemDomain::MediaProduction),
            ("grid protocol".into(), ProblemDomain::MediaProduction),
            ("content quality".into(), ProblemDomain::MediaProduction),
            ("completeness".into(), ProblemDomain::MediaProduction),
            ("conformance".into(), ProblemDomain::MediaProduction),
            ("freshness".into(), ProblemDomain::MediaProduction),
            ("transcript".into(), ProblemDomain::MediaProduction),
            ("tts".into(), ProblemDomain::MediaProduction),
            ("narration".into(), ProblemDomain::MediaProduction),
            ("text-to-speech".into(), ProblemDomain::MediaProduction),
            ("av sync".into(), ProblemDomain::MediaProduction),
            ("av-sync".into(), ProblemDomain::MediaProduction),
            ("publishing".into(), ProblemDomain::MediaProduction),
            ("coursepage".into(), ProblemDomain::MediaProduction),
        ];

        // Component names
        self.component_names.extend(
            [
                "trueno",
                "trueno-db",
                "trueno-graph",
                "trueno-viz",
                "trueno-rag",
                "aprender",
                "entrenar",
                "realizar",
                "depyler",
                "decy",
                "bashrs",
                "ruchy",
                "batuta",
                "repartir",
                "pforge",
                "certeza",
                "pmat",
                "renacer",
                "alimentar",
                "pacha",
                "whisper-apr",
                "simular",
                "probar",
                "pepita",
                "provable-contracts",
                "tiny-model-ground-truth",
                "forjar",
                "rmedia",
            ]
            .map(String::from),
        );
    }

    /// Parse a natural language query
    pub(crate) fn parse(&self, query: &str) -> ParsedQuery {
        let lower = query.to_lowercase();

        ParsedQuery {
            original: query.to_string(),
            domains: self.extract_domains(&lower),
            algorithms: self.extract_algorithms(&lower),
            keywords: self.extract_keywords(&lower),
            data_size: self.extract_data_size(&lower),
            performance_hints: self.extract_performance_hints(&lower),
            mentioned_components: self.extract_components(&lower),
        }
    }

    fn extract_domains(&self, query: &str) -> Vec<ProblemDomain> {
        let mut domains = Vec::new();
        let mut seen = HashSet::new();

        for (keyword, domain) in &self.domain_keywords {
            if query.contains(keyword) && !seen.contains(domain) {
                domains.push(*domain);
                seen.insert(*domain);
            }
        }

        domains
    }

    fn extract_algorithms(&self, query: &str) -> Vec<String> {
        let mut algorithms = Vec::new();

        for algo in &self.algorithm_keywords {
            if query.contains(algo) {
                // Normalize algorithm name
                let normalized = algo.replace([' ', '-'], "_").to_lowercase();
                if !algorithms.contains(&normalized) {
                    algorithms.push(normalized);
                }
            }
        }

        algorithms
    }

    fn extract_keywords(&self, query: &str) -> Vec<String> {
        // Extract significant keywords (words > 3 chars, not stopwords)
        let stopwords: HashSet<_> = [
            "the", "and", "for", "with", "how", "what", "can", "does", "want", "need", "use",
            "using", "have", "this", "that", "from", "into", "about", "which", "when", "where",
            "should",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();

        query
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| w.len() > 3 && !stopwords.contains(*w))
            .map(String::from)
            .collect()
    }

    fn extract_data_size(&self, query: &str) -> Option<DataSize> {
        // Look for patterns like "1M samples", "100K rows", "1GB data"
        // Note: regex patterns defined for reference but simple string matching used below
        // for patterns like: r"(\d+)\s*[mM]\s*(samples?|rows?|records?|items?)"

        for (suffix, multiplier) in [
            ("m samples", 1_000_000),
            ("m rows", 1_000_000),
            ("k samples", 1_000),
            ("k rows", 1_000),
            ("million", 1_000_000),
            ("thousand", 1_000),
            ("billion", 1_000_000_000),
        ] {
            if let Some(idx) = query.find(suffix) {
                // Look for number before suffix
                let before = &query[..idx];
                if let Some(num_str) = before.split_whitespace().last() {
                    if let Ok(num) = num_str.parse::<u64>() {
                        return Some(DataSize::samples(num * multiplier));
                    }
                }
            }
        }

        // Look for "large" / "small" / "huge" indicators
        if query.contains("large") || query.contains("huge") || query.contains("big") {
            return Some(DataSize::samples(1_000_000));
        }
        if query.contains("small") || query.contains("tiny") {
            return Some(DataSize::samples(1_000));
        }

        None
    }

    fn extract_performance_hints(&self, query: &str) -> Vec<PerformanceHint> {
        let mut hints = Vec::new();

        // Table-driven: simple OR-match patterns
        for &(keywords, hint) in HINT_PATTERNS {
            if keywords.iter().any(|kw| query.contains(kw)) {
                hints.push(hint);
            }
        }

        // Compound check for LowLatency: "<" AND "ms" (e.g. "<10ms")
        // Only add if not already matched by the table-driven keywords above
        if !hints.contains(&PerformanceHint::LowLatency)
            && query.contains('<')
            && query.contains("ms")
        {
            hints.push(PerformanceHint::LowLatency);
        }

        // Compound AND logic: "memory" AND ("low" OR "efficient")
        if query.contains("memory") && (query.contains("low") || query.contains("efficient")) {
            hints.push(PerformanceHint::LowMemory);
        }

        hints
    }

    fn extract_components(&self, query: &str) -> Vec<String> {
        self.component_names
            .iter()
            .filter(|name| query.contains(name.as_str()) || query.contains(&name.replace('-', " ")))
            .cloned()
            .collect()
    }
}

// =============================================================================
// Query Engine
// =============================================================================

/// Query engine that processes queries and generates responses
pub struct QueryEngine {
    parser: QueryParser,
}

impl Default for QueryEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryEngine {
    /// Create a new query engine
    pub fn new() -> Self {
        Self { parser: QueryParser::new() }
    }

    /// Parse a query string
    pub fn parse(&self, query: &str) -> ParsedQuery {
        self.parser.parse(query)
    }

    /// Determine the primary problem domain from a parsed query
    pub fn primary_domain(&self, parsed: &ParsedQuery) -> Option<ProblemDomain> {
        parsed.domains.first().copied()
    }

    /// Determine the primary algorithm mentioned
    pub fn primary_algorithm<'a>(&self, parsed: &'a ParsedQuery) -> Option<&'a str> {
        parsed.algorithms.first().map(|s| s.as_str())
    }

    /// Check if the query requires GPU
    pub fn requires_gpu(&self, parsed: &ParsedQuery) -> bool {
        parsed.performance_hints.contains(&PerformanceHint::GPURequired)
    }

    /// Check if the query requires distributed computing
    pub fn requires_distribution(&self, parsed: &ParsedQuery) -> bool {
        parsed.performance_hints.contains(&PerformanceHint::Distributed)
            || parsed.data_size.map(|s| s.is_large()).unwrap_or(false)
    }

    /// Check if the query requires sovereign/local execution
    pub fn requires_sovereign(&self, parsed: &ParsedQuery) -> bool {
        parsed.performance_hints.contains(&PerformanceHint::Sovereign)
    }

    /// Estimate operation complexity from parsed query
    pub fn estimate_complexity(&self, parsed: &ParsedQuery) -> OpComplexity {
        // Matrix operations are high complexity (keyword-based, checked first)
        if parsed.keywords.iter().any(|k| k.contains("matrix") || k.contains("matmul")) {
            return OpComplexity::High;
        }

        // Table-driven domain-to-complexity lookup (first match wins)
        for &(domain, complexity) in DOMAIN_COMPLEXITY {
            if parsed.domains.contains(&domain) {
                return complexity;
            }
        }

        // Default to low
        OpComplexity::Low
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Test Fixtures
    // =========================================================================

    /// Create a fresh QueryParser for tests
    fn parser() -> QueryParser {
        QueryParser::new()
    }

    /// Create a fresh QueryEngine for tests
    fn engine() -> QueryEngine {
        QueryEngine::new()
    }

    // =========================================================================
    // QueryParser Tests
    // =========================================================================

    #[test]
    fn test_parser_new() {
        let p = parser();
        assert!(!p.algorithm_keywords.is_empty());
        assert!(!p.domain_keywords.is_empty());
        assert!(!p.component_names.is_empty());
    }

    #[test]
    fn test_parse_basic() {
        let parsed = parser().parse("Train a random forest classifier");

        assert_eq!(parsed.original, "Train a random forest classifier");
        assert!(!parsed.domains.is_empty());
        assert!(!parsed.algorithms.is_empty());
    }

    // =========================================================================
    // Domain Extraction Tests
    // =========================================================================

    #[test]
    fn test_extract_supervised_learning() {
        let parsed = parser().parse("I want to train a classification model");

        assert!(parsed.domains.contains(&ProblemDomain::SupervisedLearning));
    }

    #[test]
    fn test_extract_unsupervised_learning() {
        let parsed = parser().parse("Help me cluster my data for anomaly detection");

        assert!(parsed.domains.contains(&ProblemDomain::UnsupervisedLearning));
    }

    #[test]
    fn test_extract_deep_learning() {
        let parsed = parser().parse("Fine-tune a transformer with LoRA");

        assert!(parsed.domains.contains(&ProblemDomain::DeepLearning));
    }

    #[test]
    fn test_extract_inference() {
        let parsed = parser().parse("Deploy model for production inference");

        assert!(parsed.domains.contains(&ProblemDomain::Inference));
    }

    #[test]
    fn test_extract_python_migration() {
        let parsed = parser().parse("Convert my sklearn pipeline to Rust");

        assert!(parsed.domains.contains(&ProblemDomain::PythonMigration));
    }

    #[test]
    fn test_extract_linear_algebra() {
        let parsed = parser().parse("Fast matrix multiplication with SIMD");

        assert!(parsed.domains.contains(&ProblemDomain::LinearAlgebra));
    }

    #[test]
    fn test_extract_graph_analytics() {
        let parsed = parser().parse("Run pagerank on my graph");

        assert!(parsed.domains.contains(&ProblemDomain::GraphAnalytics));
    }

    #[test]
    fn test_extract_multiple_domains() {
        let parsed = parser().parse("Train a classifier on python sklearn data");

        assert!(parsed.domains.len() >= 2);
        assert!(parsed.domains.contains(&ProblemDomain::SupervisedLearning));
        assert!(parsed.domains.contains(&ProblemDomain::PythonMigration));
    }

    // =========================================================================
    // Algorithm Extraction Tests
    // =========================================================================

    #[test]
    fn test_extract_random_forest() {
        let parsed = parser().parse("Train a random forest on my data");

        assert!(parsed.algorithms.iter().any(|a| a.contains("random_forest")));
    }

    #[test]
    fn test_extract_gradient_boosting() {
        let parsed = parser().parse("Use gradient boosting for regression");

        assert!(parsed.algorithms.iter().any(|a| a.contains("gradient_boosting") || a == "gbm"));
    }

    #[test]
    fn test_extract_kmeans() {
        let parsed = parser().parse("Cluster with k-means algorithm");

        assert!(parsed.algorithms.iter().any(|a| a.contains("kmeans") || a.contains("k_means")));
    }

    #[test]
    fn test_extract_lora() {
        let parsed = parser().parse("Fine-tune with LoRA");

        assert!(parsed.algorithms.iter().any(|a| a.contains("lora")));
    }

    // =========================================================================
    // Data Size Extraction Tests
    // =========================================================================

    #[test]
    fn test_extract_data_size_million() {
        let parsed = parser().parse("Train on 1 million samples");

        assert!(parsed.data_size.is_some());
        let size = parsed.data_size.expect("unexpected failure");
        assert!(size.is_large());
    }

    #[test]
    fn test_extract_data_size_1m() {
        let parsed = parser().parse("Process 5m rows of data");

        assert!(parsed.data_size.is_some());
    }

    #[test]
    fn test_extract_data_size_thousand() {
        let parsed = parser().parse("Test on 10 thousand samples");

        assert!(parsed.data_size.is_some());
        let size = parsed.data_size.expect("unexpected failure");
        assert!(!size.is_large());
    }

    #[test]
    fn test_extract_data_size_large_indicator() {
        let parsed = parser().parse("Handle large dataset");

        assert!(parsed.data_size.is_some());
        assert!(parsed.data_size.expect("unexpected failure").is_large());
    }

    #[test]
    fn test_extract_data_size_small_indicator() {
        let parsed = parser().parse("Small dataset for testing");

        assert!(parsed.data_size.is_some());
        assert!(!parsed.data_size.expect("unexpected failure").is_large());
    }

    // =========================================================================
    // Performance Hints Tests
    // =========================================================================

    #[test]
    fn test_extract_low_latency() {
        let parsed = parser().parse("Need fast inference with low latency");

        assert!(parsed.performance_hints.contains(&PerformanceHint::LowLatency));
    }

    #[test]
    fn test_extract_gpu_required() {
        let parsed = parser().parse("Train model on GPU");

        assert!(parsed.performance_hints.contains(&PerformanceHint::GPURequired));
    }

    #[test]
    fn test_extract_distributed() {
        let parsed = parser().parse("Distributed training on multi-node cluster");

        assert!(parsed.performance_hints.contains(&PerformanceHint::Distributed));
    }

    #[test]
    fn test_extract_edge_deployment() {
        let parsed = parser().parse("Deploy model to edge devices");

        assert!(parsed.performance_hints.contains(&PerformanceHint::EdgeDeployment));
    }

    #[test]
    fn test_extract_sovereign() {
        let parsed = parser().parse("GDPR compliant, sovereign execution");

        assert!(parsed.performance_hints.contains(&PerformanceHint::Sovereign));
    }

    #[test]
    fn test_extract_eu_ai_act() {
        let parsed = parser().parse("Must comply with EU AI Act");

        assert!(parsed.performance_hints.contains(&PerformanceHint::Sovereign));
    }

    // =========================================================================
    // Component Extraction Tests
    // =========================================================================

    #[test]
    fn test_extract_component_trueno() {
        let parsed = parser().parse("Use trueno for tensor operations");

        assert!(parsed.mentioned_components.contains(&"trueno".to_string()));
    }

    #[test]
    fn test_extract_component_aprender() {
        let parsed = parser().parse("Train with aprender random forest");

        assert!(parsed.mentioned_components.contains(&"aprender".to_string()));
    }

    #[test]
    fn test_extract_multiple_components() {
        let parsed = parser().parse("Use depyler to convert sklearn to aprender");

        assert!(parsed.mentioned_components.contains(&"depyler".to_string()));
        assert!(parsed.mentioned_components.contains(&"aprender".to_string()));
    }

    // =========================================================================
    // QueryEngine Tests
    // =========================================================================

    #[test]
    fn test_query_engine_new() {
        let e = engine();
        let parsed = e.parse("Test query");
        assert!(!parsed.original.is_empty());
    }

    #[test]
    fn test_query_engine_default() {
        let e = QueryEngine::default();
        let parsed = e.parse("Test");
        assert_eq!(parsed.original, "Test");
    }

    #[test]
    fn test_primary_domain() {
        let e = engine();
        let parsed = e.parse("Train a classifier");

        let domain = e.primary_domain(&parsed);
        assert!(domain.is_some());
        assert_eq!(domain.expect("unexpected failure"), ProblemDomain::SupervisedLearning);
    }

    #[test]
    fn test_primary_algorithm() {
        let e = engine();
        let parsed = e.parse("Use random forest");

        let algo = e.primary_algorithm(&parsed);
        assert!(algo.is_some());
        assert!(algo.expect("unexpected failure").contains("random_forest"));
    }

    #[test]
    fn test_requires_gpu() {
        let e = engine();

        let parsed = e.parse("Train on GPU");
        assert!(e.requires_gpu(&parsed));

        let parsed = e.parse("Simple CPU training");
        assert!(!e.requires_gpu(&parsed));
    }

    #[test]
    fn test_requires_distribution() {
        let e = engine();

        let parsed = e.parse("Distributed training");
        assert!(e.requires_distribution(&parsed));

        let parsed = e.parse("Train on 1 billion samples");
        assert!(e.requires_distribution(&parsed));

        let parsed = e.parse("Small local training");
        assert!(!e.requires_distribution(&parsed));
    }

    #[test]
    fn test_requires_sovereign() {
        let e = engine();

        let parsed = e.parse("GDPR compliant local execution");
        assert!(e.requires_sovereign(&parsed));

        let parsed = e.parse("Cloud training");
        assert!(!e.requires_sovereign(&parsed));
    }

    #[test]
    fn test_estimate_complexity_high() {
        let e = engine();

        let parsed = e.parse("Matrix multiplication");
        assert_eq!(e.estimate_complexity(&parsed), OpComplexity::High);

        let parsed = e.parse("Deep learning training");
        assert_eq!(e.estimate_complexity(&parsed), OpComplexity::High);
    }

    #[test]
    fn test_estimate_complexity_medium() {
        let e = engine();

        let parsed = e.parse("Train a classifier");
        assert_eq!(e.estimate_complexity(&parsed), OpComplexity::Medium);

        let parsed = e.parse("Graph pagerank");
        assert_eq!(e.estimate_complexity(&parsed), OpComplexity::Medium);
    }

    #[test]
    fn test_estimate_complexity_low() {
        let e = engine();

        let parsed = e.parse("Simple data loading");
        assert_eq!(e.estimate_complexity(&parsed), OpComplexity::Low);
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_full_query_parsing() {
        let e = engine();
        let parsed =
            e.parse("I need to train a random forest on 1 million samples with GPU acceleration");

        // Should detect supervised learning
        assert!(parsed.domains.contains(&ProblemDomain::SupervisedLearning));

        // Should detect random forest
        assert!(parsed.algorithms.iter().any(|a| a.contains("random_forest")));

        // Should detect large data
        assert!(parsed.data_size.is_some());
        assert!(parsed.data_size.expect("unexpected failure").is_large());

        // Should detect GPU requirement
        assert!(parsed.performance_hints.contains(&PerformanceHint::GPURequired));
    }

    #[test]
    fn test_sklearn_migration_query() {
        let e = engine();
        let parsed = e.parse("Convert my sklearn pipeline with RandomForest to Rust aprender");

        assert!(parsed.domains.contains(&ProblemDomain::PythonMigration));
        assert!(parsed.algorithms.iter().any(|a| a.contains("random")));
        assert!(parsed.mentioned_components.contains(&"aprender".to_string()));
    }

    #[test]
    fn test_inference_query() {
        let e = engine();
        let parsed = e.parse("Deploy model to AWS Lambda with <10ms latency");

        assert!(parsed.domains.contains(&ProblemDomain::Inference));
        assert!(parsed.domains.contains(&ProblemDomain::ModelServing));
        assert!(parsed.performance_hints.contains(&PerformanceHint::LowLatency));
    }

    // =========================================================================
    // Falsification: Course Creation Workflow Coverage
    // =========================================================================

    #[test]
    fn test_media_production_course_workflows() {
        let p = parser();

        // Every concept from resolve-pipeline make demo* MUST route to MediaProduction
        let must_route: &[(&str, &str)] = &[
            ("render a course video", "demo: rmedia course"),
            ("transcribe audio from course", "demo: --transcribe"),
            ("check transcript quality", "demo-course: whisper-apr score"),
            ("generate course outline", "demo-outline: rmedia outline"),
            ("extract key terms from transcripts", "demo-key-terms"),
            ("generate reflection prompts", "demo-reflection"),
            ("convert svg banner to png", "demo-banners: svg2png"),
            ("generate landing page for course", "demo-coursera-page"),
            ("tts narration for course video", "demo-tts: espeak-ng"),
            ("check av sync on rendered video", "demo-av-check: probador"),
            ("coursera publishing pipeline", "demo-coursera"),
            ("subtitle burn in", "existing capability"),
            ("generate thumbnail for video", "thumbnail generation"),
            ("vocabulary enrichment from transcripts", "vocab-enrich skill"),
            ("course quality scoring", "coursera-score target"),
            ("content completeness check", "quality dimension"),
            ("syllabus generation", "outline variant"),
        ];

        for (query, context) in must_route {
            let parsed = p.parse(query);
            assert!(
                parsed.domains.contains(&ProblemDomain::MediaProduction),
                "FAIL: '{}' ({}) did not route to MediaProduction. Got: {:?}",
                query,
                context,
                parsed.domains,
            );
        }
    }

    #[test]
    fn test_media_production_complexity_medium() {
        let e = engine();
        let parsed = e.parse("render course video with encoding");
        assert_eq!(e.estimate_complexity(&parsed), OpComplexity::Medium);
    }

    /// Harder falsification: bare concepts without "video"/"course" crutch words
    #[test]
    fn test_media_production_bare_concepts() {
        let p = parser();

        let must_route: &[(&str, &str)] = &[
            ("tts narration pipeline", "demo-tts"),
            ("text-to-speech for lectures", "demo-tts variant"),
            ("av sync verification", "demo-av-check"),
            ("publishing workflow", "coursera pipeline"),
            ("generate coursepage prompt", "demo-coursera-page"),
        ];

        for (query, context) in must_route {
            let parsed = p.parse(query);
            assert!(
                parsed.domains.contains(&ProblemDomain::MediaProduction),
                "FAIL: '{}' ({}) did not route to MediaProduction. Got: {:?}",
                query,
                context,
                parsed.domains,
            );
        }
    }
}
