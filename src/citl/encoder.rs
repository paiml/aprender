//! Error encoder for program-feedback graph construction.
//!
//! Per Yasunaga & Liang (2020), the program-feedback graph connects:
//! - Symbols in source code (variables, types, functions)
//! - Diagnostic feedback (error codes, messages, spans)
//! - AST structure (parent-child, sibling relationships)

use super::diagnostic::{CompilerDiagnostic, SourceSpan};
use super::ErrorCode;
use std::collections::HashMap;
use trueno::Vector;

/// Error embedding vector.
///
/// A fixed-size vector representation of an error pattern
/// suitable for similarity search and ML training.
#[derive(Debug, Clone)]
pub struct ErrorEmbedding {
    /// The embedding vector (256 dimensions by default)
    pub vector: Vec<f32>,
    /// Error code for reference
    pub error_code: ErrorCode,
    /// Hash of surrounding context
    pub context_hash: u64,
}

impl ErrorEmbedding {
    /// Create a new error embedding.
    #[must_use]
    pub fn new(vector: Vec<f32>, error_code: ErrorCode, context_hash: u64) -> Self {
        Self {
            vector,
            error_code,
            context_hash,
        }
    }

    /// Get the embedding dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.vector.len()
    }

    /// Compute cosine similarity with another embedding using trueno SIMD.
    #[must_use]
    pub fn cosine_similarity(&self, other: &ErrorEmbedding) -> f32 {
        if self.vector.len() != other.vector.len() || self.vector.is_empty() {
            return 0.0;
        }

        let va = Vector::from_slice(&self.vector);
        let vb = Vector::from_slice(&other.vector);

        let dot = va.dot(&vb).unwrap_or(0.0);
        let norm_a = va.norm_l2().unwrap_or(0.0);
        let norm_b = vb.norm_l2().unwrap_or(0.0);

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Compute L2 distance to another embedding using trueno SIMD.
    #[must_use]
    pub fn l2_distance(&self, other: &ErrorEmbedding) -> f32 {
        if self.vector.len() != other.vector.len() || self.vector.is_empty() {
            return f32::MAX;
        }

        let va = Vector::from_slice(&self.vector);
        let vb = Vector::from_slice(&other.vector);

        va.sub(&vb)
            .and_then(|diff| diff.norm_l2())
            .unwrap_or(f32::MAX)
    }
}

/// Error encoder using simplified feature extraction.
///
/// In production, this would use a GNN per Yasunaga & Liang (2020).
/// For now, we use bag-of-features encoding suitable for pattern matching.
#[derive(Debug)]
pub struct ErrorEncoder {
    /// Embedding dimension
    dim: usize,
    /// Error code embeddings (learned or hashed)
    error_code_embeddings: HashMap<String, Vec<f32>>,
    /// Vocabulary for source tokens (reserved for future token-level encoding)
    #[allow(dead_code)]
    vocab: HashMap<String, usize>,
}

impl ErrorEncoder {
    /// Create a new error encoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            dim: 256,
            error_code_embeddings: Self::init_error_code_embeddings(),
            vocab: HashMap::new(),
        }
    }

    /// Create encoder with custom dimension.
    #[must_use]
    pub fn with_dim(dim: usize) -> Self {
        Self {
            dim,
            error_code_embeddings: Self::init_error_code_embeddings(),
            vocab: HashMap::new(),
        }
    }

    /// Initialize error code embeddings with deterministic hashing.
    fn init_error_code_embeddings() -> HashMap<String, Vec<f32>> {
        let codes = [
            "E0308", "E0382", "E0597", "E0599", "E0433", "E0432", "E0277", "E0425", "E0282",
            "E0412", "E0502", "E0499", "E0596", "E0507", "E0621", "E0106", "E0373", "E0495",
            "E0623",
        ];

        let mut embeddings = HashMap::new();
        for (i, code) in codes.iter().enumerate() {
            let mut vec = vec![0.0f32; 64];
            // Simple one-hot-ish encoding with some spread
            let base_idx = i % 32;
            vec[base_idx] = 1.0;
            vec[(base_idx + 16) % 64] = 0.5;
            vec[(base_idx + 32) % 64] = 0.25;
            embeddings.insert((*code).to_string(), vec);
        }
        embeddings
    }

    /// Encode a diagnostic into an embedding.
    ///
    /// # Algorithm
    /// 1. Extract error code embedding
    /// 2. Extract source context features
    /// 3. Extract type information (if available)
    /// 4. Concatenate and normalize
    #[must_use]
    pub fn encode(&self, diagnostic: &CompilerDiagnostic, source: &str) -> ErrorEmbedding {
        let mut vector = vec![0.0f32; self.dim];

        // 1. Error code embedding (first 64 dims)
        let code_embedding = self
            .error_code_embeddings
            .get(&diagnostic.code.code)
            .cloned()
            .unwrap_or_else(|| self.hash_code(&diagnostic.code.code));

        for (i, &v) in code_embedding.iter().enumerate().take(64.min(self.dim)) {
            vector[i] = v;
        }

        // 2. Source context features (next 64 dims)
        let context_features = self.extract_context_features(source, &diagnostic.span);
        for (i, &v) in context_features.iter().enumerate().take(64) {
            if i + 64 < self.dim {
                vector[i + 64] = v;
            }
        }

        // 3. Type information features (next 64 dims)
        let type_features = self.extract_type_features(diagnostic);
        for (i, &v) in type_features.iter().enumerate().take(64) {
            if i + 128 < self.dim {
                vector[i + 128] = v;
            }
        }

        // 4. Message features (last 64 dims)
        let message_features = self.extract_message_features(&diagnostic.message);
        for (i, &v) in message_features.iter().enumerate().take(64) {
            if i + 192 < self.dim {
                vector[i + 192] = v;
            }
        }

        // Normalize using trueno SIMD
        let tv = Vector::from_slice(&vector);
        if let Ok(normalized) = tv.normalize() {
            vector.copy_from_slice(normalized.as_slice());
        }

        let context_hash = self.hash_context(source, &diagnostic.span);

        ErrorEmbedding::new(vector, diagnostic.code.clone(), context_hash)
    }

    /// Hash an unknown error code to an embedding.
    #[allow(clippy::unused_self)]
    fn hash_code(&self, code: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; 64];
        let hash = Self::simple_hash(code);

        for (i, v) in vec.iter_mut().enumerate() {
            *v = ((hash >> (i % 64)) & 1) as f32 * 0.5;
        }
        vec
    }

    /// Extract features from source context around error.
    #[allow(clippy::unused_self)]
    fn extract_context_features(&self, source: &str, span: &SourceSpan) -> Vec<f32> {
        let mut features = vec![0.0f32; 64];

        // Extract lines around error
        let lines: Vec<&str> = source.lines().collect();
        let start_line = span.line_start.saturating_sub(1);
        let end_line = span.line_end.min(lines.len());

        // Feature 0-15: Character distribution in error region
        let mut char_counts = [0u32; 16];
        for line in lines.iter().take(end_line).skip(start_line) {
            for c in line.chars() {
                let bucket = (c as usize) % 16;
                char_counts[bucket] += 1;
            }
        }
        let total: f32 = char_counts.iter().sum::<u32>() as f32 + 1.0;
        for (i, &count) in char_counts.iter().enumerate() {
            features[i] = count as f32 / total;
        }

        // Feature 16-31: Keyword presence
        let keywords = [
            "let", "mut", "fn", "struct", "impl", "trait", "use", "mod", "pub", "self", "Self",
            "return", "if", "else", "match", "for",
        ];
        let context: String = lines
            .iter()
            .take(end_line)
            .skip(start_line)
            .copied()
            .collect::<Vec<_>>()
            .join(" ");

        for (i, keyword) in keywords.iter().enumerate() {
            features[16 + i] = if context.contains(keyword) { 1.0 } else { 0.0 };
        }

        // Feature 32-47: Syntax patterns
        let patterns = [
            ("->", 32),
            ("=>", 33),
            ("::", 34),
            ("&mut", 35),
            ("&", 36),
            ("'", 37),
            ("<", 38),
            (">", 39),
            ("()", 40),
            ("[]", 41),
            ("{}", 42),
            (";", 43),
            ("=", 44),
            (".", 45),
            ("?", 46),
            ("!", 47),
        ];
        for (pattern, idx) in &patterns {
            features[*idx] = if context.contains(pattern) { 1.0 } else { 0.0 };
        }

        // Feature 48-63: Line characteristics
        features[48] = end_line.saturating_sub(start_line) as f32 / 10.0; // Span size
        features[49] = span.column_start as f32 / 80.0; // Indentation hint
        features[50] = if context.contains("fn ") { 1.0 } else { 0.0 }; // In function
        features[51] = if context.contains("impl ") { 1.0 } else { 0.0 }; // In impl
        features[52] = if context.contains("struct ") {
            1.0
        } else {
            0.0
        }; // In struct

        features
    }

    /// Extract features from type information.
    fn extract_type_features(&self, diagnostic: &CompilerDiagnostic) -> Vec<f32> {
        let mut features = vec![0.0f32; 64];

        // Features for expected type
        if let Some(expected) = &diagnostic.expected {
            features[0] = 1.0; // Has expected type
            features[1] = if expected.is_reference { 1.0 } else { 0.0 };
            features[2] = if expected.is_mutable { 1.0 } else { 0.0 };
            features[3] = expected.generics.len() as f32 / 4.0;

            // Type category features
            let type_features = self.type_to_features(&expected.base);
            for (i, &v) in type_features.iter().enumerate() {
                if i + 4 < 32 {
                    features[i + 4] = v;
                }
            }
        }

        // Features for found type
        if let Some(found) = &diagnostic.found {
            features[32] = 1.0; // Has found type
            features[33] = if found.is_reference { 1.0 } else { 0.0 };
            features[34] = if found.is_mutable { 1.0 } else { 0.0 };
            features[35] = found.generics.len() as f32 / 4.0;

            let type_features = self.type_to_features(&found.base);
            for (i, &v) in type_features.iter().enumerate() {
                if i + 36 < 64 {
                    features[i + 36] = v;
                }
            }
        }

        features
    }

    /// Convert type name to feature vector.
    #[allow(clippy::unused_self)]
    fn type_to_features(&self, type_name: &str) -> Vec<f32> {
        let mut features = vec![0.0f32; 16];

        // Common type patterns
        let type_patterns = [
            ("String", 0),
            ("str", 1),
            ("Vec", 2),
            ("Option", 3),
            ("Result", 4),
            ("Box", 5),
            ("i32", 6),
            ("i64", 7),
            ("u32", 8),
            ("u64", 9),
            ("f32", 10),
            ("f64", 11),
            ("bool", 12),
            ("char", 13),
            ("usize", 14),
            ("isize", 15),
        ];

        for (pattern, idx) in &type_patterns {
            if type_name.contains(pattern) {
                features[*idx] = 1.0;
            }
        }

        features
    }

    /// Extract features from error message.
    #[allow(clippy::unused_self)]
    fn extract_message_features(&self, message: &str) -> Vec<f32> {
        let mut features = vec![0.0f32; 64];
        let message_lower = message.to_lowercase();

        // Key phrases in error messages
        let phrases = [
            ("mismatched types", 0),
            ("expected", 1),
            ("found", 2),
            ("borrow", 3),
            ("move", 4),
            ("lifetime", 5),
            ("cannot", 6),
            ("trait", 7),
            ("implement", 8),
            ("method", 9),
            ("function", 10),
            ("argument", 11),
            ("return", 12),
            ("value", 13),
            ("type", 14),
            ("reference", 15),
            ("mutable", 16),
            ("immutable", 17),
            ("borrowed", 18),
            ("owned", 19),
            ("copy", 20),
            ("clone", 21),
            ("bound", 22),
            ("satisfy", 23),
            ("require", 24),
            ("missing", 25),
            ("unknown", 26),
            ("unresolved", 27),
            ("import", 28),
            ("module", 29),
            ("crate", 30),
            ("use", 31),
        ];

        for (phrase, idx) in &phrases {
            features[*idx] = if message_lower.contains(phrase) {
                1.0
            } else {
                0.0
            };
        }

        // Message length feature
        features[32] = (message.len() as f32 / 200.0).min(1.0);

        // Word count feature
        features[33] = (message.split_whitespace().count() as f32 / 30.0).min(1.0);

        features
    }

    /// Compute context hash for deduplication.
    #[allow(clippy::unused_self)]
    fn hash_context(&self, source: &str, span: &SourceSpan) -> u64 {
        let lines: Vec<&str> = source.lines().collect();
        let start = span.line_start.saturating_sub(1);
        let end = span.line_end.min(lines.len());

        let context: String = lines
            .iter()
            .take(end)
            .skip(start)
            .copied()
            .collect::<Vec<_>>()
            .join("\n");

        Self::simple_hash(&context)
    }

    /// Simple hash function for strings.
    fn simple_hash(s: &str) -> u64 {
        let mut hash: u64 = 5381;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(u64::from(byte));
        }
        hash
    }
}

impl Default for ErrorEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Program-feedback graph structure.
///
/// Per Yasunaga & Liang (2020), this graph connects symbols
/// in source code with diagnostic feedback for GNN reasoning.
#[derive(Debug, Clone)]
pub struct ProgramFeedbackGraph {
    /// Node features
    pub node_features: Vec<Vec<f32>>,
    /// Node types
    pub node_types: Vec<NodeType>,
    /// Edges (source, target)
    pub edges: Vec<(usize, usize)>,
    /// Edge types
    pub edge_types: Vec<EdgeType>,
}

impl ProgramFeedbackGraph {
    /// Create a new empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            node_features: Vec::new(),
            node_types: Vec::new(),
            edges: Vec::new(),
            edge_types: Vec::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node_type: NodeType, features: Vec<f32>) -> usize {
        let idx = self.node_features.len();
        self.node_features.push(features);
        self.node_types.push(node_type);
        idx
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, source: usize, target: usize, edge_type: EdgeType) {
        self.edges.push((source, target));
        self.edge_types.push(edge_type);
    }

    /// Get the number of nodes.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.node_features.len()
    }

    /// Get the number of edges.
    #[must_use]
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
}

impl Default for ProgramFeedbackGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Node type in program-feedback graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    /// AST node (variable, type, expression, etc.)
    Ast,
    /// Compiler diagnostic
    Diagnostic,
    /// Expected type in type error
    ExpectedType,
    /// Found type in type error
    FoundType,
    /// Compiler suggestion
    Suggestion,
}

/// Edge type in program-feedback graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    /// AST parent-child relationship
    AstChild,
    /// Data flow edge
    DataFlow,
    /// Control flow edge
    ControlFlow,
    /// Diagnostic refers to code location
    DiagnosticRefers,
    /// Type expectation
    Expects,
    /// Type found
    Found,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::citl::diagnostic::DiagnosticSeverity;
    use crate::citl::{Difficulty, ErrorCategory};

    // ==================== ErrorEmbedding Tests ====================

    #[test]
    fn test_error_embedding_new() {
        let vector = vec![0.5; 256];
        let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let embedding = ErrorEmbedding::new(vector.clone(), code, 12345);

        assert_eq!(embedding.dim(), 256);
        assert_eq!(embedding.context_hash, 12345);
    }

    #[test]
    fn test_error_embedding_cosine_similarity() {
        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0, 0.0];
        let v3 = vec![0.0, 1.0, 0.0, 0.0];

        let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);

        let e1 = ErrorEmbedding::new(v1, code.clone(), 0);
        let e2 = ErrorEmbedding::new(v2, code.clone(), 0);
        let e3 = ErrorEmbedding::new(v3, code, 0);

        // Same vectors should have similarity 1.0
        assert!((e1.cosine_similarity(&e2) - 1.0).abs() < 0.001);

        // Orthogonal vectors should have similarity 0.0
        assert!(e1.cosine_similarity(&e3).abs() < 0.001);
    }

    #[test]
    fn test_error_embedding_l2_distance() {
        let v1 = vec![0.0, 0.0];
        let v2 = vec![3.0, 4.0];

        let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);

        let e1 = ErrorEmbedding::new(v1, code.clone(), 0);
        let e2 = ErrorEmbedding::new(v2, code, 0);

        // Distance should be 5.0 (3-4-5 triangle)
        assert!((e1.l2_distance(&e2) - 5.0).abs() < 0.001);
    }

    // ==================== ErrorEncoder Tests ====================

    #[test]
    fn test_error_encoder_new() {
        let encoder = ErrorEncoder::new();
        assert_eq!(encoder.dim, 256);
    }

    #[test]
    fn test_error_encoder_with_dim() {
        let encoder = ErrorEncoder::with_dim(128);
        assert_eq!(encoder.dim, 128);
    }

    #[test]
    fn test_error_encoder_encode() {
        let encoder = ErrorEncoder::new();
        let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let span = SourceSpan::single_line("test.rs", 10, 5, 20);
        let diagnostic =
            CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span);

        let source = "fn main() { let x: i32 = \"hello\"; }";
        let embedding = encoder.encode(&diagnostic, source);

        assert_eq!(embedding.dim(), 256);
        assert_eq!(embedding.error_code.code, "E0308");
    }

    #[test]
    fn test_error_encoder_encode_with_types() {
        let encoder = ErrorEncoder::new();
        let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let span = SourceSpan::single_line("test.rs", 10, 5, 20);
        let diagnostic =
            CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span)
                .with_expected(crate::citl::diagnostic::TypeInfo::new("i32"))
                .with_found(crate::citl::diagnostic::TypeInfo::new("&str"));

        let source = "fn main() { let x: i32 = \"hello\"; }";
        let embedding = encoder.encode(&diagnostic, source);

        assert_eq!(embedding.dim(), 256);
        // Type features should be populated
        let type_region: f32 = embedding.vector[128..192].iter().sum();
        assert!(type_region > 0.0);
    }

    #[test]
    fn test_error_encoder_similar_errors_similar_embeddings() {
        let encoder = ErrorEncoder::new();

        // Two similar type mismatch errors
        let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let span1 = SourceSpan::single_line("test.rs", 10, 5, 20);
        let span2 = SourceSpan::single_line("test.rs", 20, 5, 20);

        let diag1 = CompilerDiagnostic::new(
            code.clone(),
            DiagnosticSeverity::Error,
            "mismatched types",
            span1,
        );
        let diag2 =
            CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span2);

        let source1 = "fn main() { let x: i32 = \"hello\"; }";
        let source2 = "fn foo() { let y: i32 = \"world\"; }";

        let e1 = encoder.encode(&diag1, source1);
        let e2 = encoder.encode(&diag2, source2);

        // Same error type should have high similarity
        let similarity = e1.cosine_similarity(&e2);
        assert!(
            similarity > 0.5,
            "Similar errors should have similarity > 0.5, got {}",
            similarity
        );
    }

    #[test]
    fn test_error_encoder_different_errors_different_embeddings() {
        let encoder = ErrorEncoder::new();

        let code1 = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let code2 = ErrorCode::new("E0382", ErrorCategory::Ownership, Difficulty::Medium);
        let span = SourceSpan::single_line("test.rs", 10, 5, 20);

        let diag1 = CompilerDiagnostic::new(
            code1,
            DiagnosticSeverity::Error,
            "mismatched types",
            span.clone(),
        );
        let diag2 = CompilerDiagnostic::new(
            code2,
            DiagnosticSeverity::Error,
            "borrow of moved value",
            span,
        );

        let source = "fn main() {}";

        let e1 = encoder.encode(&diag1, source);
        let e2 = encoder.encode(&diag2, source);

        // Different error types should have lower similarity
        let similarity = e1.cosine_similarity(&e2);
        assert!(
            similarity < 0.9,
            "Different errors should have similarity < 0.9, got {}",
            similarity
        );
    }

    // ==================== ProgramFeedbackGraph Tests ====================

    #[test]
    fn test_program_feedback_graph_new() {
        let graph = ProgramFeedbackGraph::new();
        assert_eq!(graph.num_nodes(), 0);
        assert_eq!(graph.num_edges(), 0);
    }

    #[test]
    fn test_program_feedback_graph_add_node() {
        let mut graph = ProgramFeedbackGraph::new();
        let features = vec![1.0, 2.0, 3.0];
        let idx = graph.add_node(NodeType::Ast, features);

        assert_eq!(idx, 0);
        assert_eq!(graph.num_nodes(), 1);
    }

    #[test]
    fn test_program_feedback_graph_add_edge() {
        let mut graph = ProgramFeedbackGraph::new();
        graph.add_node(NodeType::Ast, vec![1.0]);
        graph.add_node(NodeType::Diagnostic, vec![2.0]);
        graph.add_edge(0, 1, EdgeType::DiagnosticRefers);

        assert_eq!(graph.num_edges(), 1);
        assert_eq!(graph.edges[0], (0, 1));
        assert_eq!(graph.edge_types[0], EdgeType::DiagnosticRefers);
    }

    #[test]
    fn test_program_feedback_graph_complex() {
        let mut graph = ProgramFeedbackGraph::new();

        // Add AST nodes
        let ast1 = graph.add_node(NodeType::Ast, vec![1.0, 0.0]);
        let ast2 = graph.add_node(NodeType::Ast, vec![0.0, 1.0]);

        // Add diagnostic node
        let diag = graph.add_node(NodeType::Diagnostic, vec![1.0, 1.0]);

        // Add type nodes
        let expected = graph.add_node(NodeType::ExpectedType, vec![1.0, 0.0]);
        let found = graph.add_node(NodeType::FoundType, vec![0.0, 1.0]);

        // Add edges
        graph.add_edge(ast1, ast2, EdgeType::AstChild);
        graph.add_edge(diag, ast2, EdgeType::DiagnosticRefers);
        graph.add_edge(diag, expected, EdgeType::Expects);
        graph.add_edge(diag, found, EdgeType::Found);

        assert_eq!(graph.num_nodes(), 5);
        assert_eq!(graph.num_edges(), 4);
    }

    // ==================== NodeType and EdgeType Tests ====================

    #[test]
    fn test_node_types() {
        let types = [
            NodeType::Ast,
            NodeType::Diagnostic,
            NodeType::ExpectedType,
            NodeType::FoundType,
            NodeType::Suggestion,
        ];
        assert_eq!(types.len(), 5);
    }

    #[test]
    fn test_edge_types() {
        let types = [
            EdgeType::AstChild,
            EdgeType::DataFlow,
            EdgeType::ControlFlow,
            EdgeType::DiagnosticRefers,
            EdgeType::Expects,
            EdgeType::Found,
        ];
        assert_eq!(types.len(), 6);
    }
}
