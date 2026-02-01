//! Error encoder for program-feedback graph construction.
//!
//! Per Yasunaga & Liang (2020), the program-feedback graph connects:
//! - Symbols in source code (variables, types, functions)
//! - Diagnostic feedback (error codes, messages, spans)
//! - AST structure (parent-child, sibling relationships)
//!
//! # GNN-Based Encoding
//!
//! This module provides two encoder types:
//! - [`ErrorEncoder`]: Simple bag-of-features encoding (fast, CPU-only)
//! - [`GNNErrorEncoder`]: Graph neural network encoding (higher quality)
//!
//! The GNN encoder builds a program-feedback graph and uses message passing
//! to produce context-aware embeddings that capture code structure.

use super::diagnostic::{CompilerDiagnostic, SourceSpan};
use super::ErrorCode;
use crate::autograd::Tensor;
use crate::nn::gnn::{AdjacencyMatrix, GCNConv, SAGEAggregation, SAGEConv};
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

/// GNN-based error encoder using program-feedback graphs.
///
/// Per Yasunaga & Liang (2020), this encoder:
/// 1. Builds a heterogeneous graph from source code and diagnostics
/// 2. Applies GNN message passing to learn context-aware representations
/// 3. Pools node embeddings to produce a fixed-size error embedding
///
/// # Architecture
///
/// ```text
/// Source + Diagnostic → ProgramFeedbackGraph → GCN/SAGE layers → Mean Pool → Embedding
/// ```
///
/// # Example
///
/// ```ignore
/// use aprender::citl::encoder::GNNErrorEncoder;
///
/// let encoder = GNNErrorEncoder::new(64, 256);
/// let graph = encoder.build_graph(&diagnostic, source);
/// let embedding = encoder.encode_graph(&graph);
/// ```
#[derive(Debug)]
pub struct GNNErrorEncoder {
    /// Hidden dimension for GNN layers
    #[allow(dead_code)]
    hidden_dim: usize,
    /// Output embedding dimension
    output_dim: usize,
    /// First GCN layer (node features → hidden)
    gcn1: GCNConv,
    /// Second SAGE layer (hidden → hidden)
    sage: SAGEConv,
    /// Final GCN layer (hidden → output)
    gcn2: GCNConv,
    /// Node type embedding dimension
    node_type_dim: usize,
    /// Base feature extractor for node features
    #[allow(dead_code)]
    base_encoder: ErrorEncoder,
}

impl GNNErrorEncoder {
    /// Create a new GNN error encoder.
    ///
    /// # Arguments
    /// * `hidden_dim` - Hidden layer dimension (default: 64)
    /// * `output_dim` - Output embedding dimension (default: 256)
    #[must_use]
    pub fn new(hidden_dim: usize, output_dim: usize) -> Self {
        // Node feature dimension: base features (64) + node type embedding (8)
        let node_feature_dim = 72;

        Self {
            hidden_dim,
            output_dim,
            gcn1: GCNConv::new(node_feature_dim, hidden_dim),
            sage: SAGEConv::new(hidden_dim, hidden_dim).with_aggregation(SAGEAggregation::Mean),
            gcn2: GCNConv::new(hidden_dim, output_dim),
            node_type_dim: 8,
            base_encoder: ErrorEncoder::with_dim(64),
        }
    }

    /// Create encoder with default dimensions (64 hidden, 256 output).
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(64, 256)
    }

    /// Get the output embedding dimension.
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Build a program-feedback graph from a diagnostic and source code.
    ///
    /// The graph includes:
    /// - AST nodes for key source elements (variables, types, keywords)
    /// - Diagnostic node for the error itself
    /// - Type nodes for expected/found types (if present)
    /// - Edges connecting related nodes
    #[must_use]
    pub fn build_graph(
        &self,
        diagnostic: &CompilerDiagnostic,
        source: &str,
    ) -> ProgramFeedbackGraph {
        let mut graph = ProgramFeedbackGraph::new();

        // 1. Add diagnostic node (central node)
        let diag_features = self.extract_diagnostic_features(diagnostic);
        let diag_idx = graph.add_node(NodeType::Diagnostic, diag_features);

        // 2. Add expected type node if present
        let expected_idx = diagnostic.expected.as_ref().map(|expected| {
            let features = self.extract_type_node_features(&expected.base, true);
            graph.add_node(NodeType::ExpectedType, features)
        });

        // 3. Add found type node if present
        let found_idx = diagnostic.found.as_ref().map(|found| {
            let features = self.extract_type_node_features(&found.base, false);
            graph.add_node(NodeType::FoundType, features)
        });

        // 4. Extract AST nodes from source context
        let ast_indices = self.extract_ast_nodes(&mut graph, source, &diagnostic.span);

        // 5. Add edges
        // Diagnostic → Type nodes
        if let Some(exp_idx) = expected_idx {
            graph.add_edge(diag_idx, exp_idx, EdgeType::Expects);
        }
        if let Some(fnd_idx) = found_idx {
            graph.add_edge(diag_idx, fnd_idx, EdgeType::Found);
        }

        // Diagnostic → AST nodes at error location
        for &ast_idx in &ast_indices {
            graph.add_edge(diag_idx, ast_idx, EdgeType::DiagnosticRefers);
        }

        // AST → AST edges (sequential context)
        for window in ast_indices.windows(2) {
            graph.add_edge(window[0], window[1], EdgeType::AstChild);
        }

        // Add suggestion node if present
        if !diagnostic.suggestions.is_empty() {
            let suggestion = &diagnostic.suggestions[0];
            let sugg_features = self.extract_suggestion_features(&suggestion.message);
            let sugg_idx = graph.add_node(NodeType::Suggestion, sugg_features);
            graph.add_edge(diag_idx, sugg_idx, EdgeType::DiagnosticRefers);
        }

        graph
    }

    /// Encode a program-feedback graph into an embedding vector.
    ///
    /// # Algorithm
    /// 1. Convert graph to tensor format
    /// 2. Apply GCN → SAGE → GCN message passing
    /// 3. Mean pool node embeddings
    /// 4. Return normalized embedding
    #[must_use]
    pub fn encode_graph(&self, graph: &ProgramFeedbackGraph) -> ErrorEmbedding {
        if graph.num_nodes() == 0 {
            // Return zero embedding for empty graphs
            return ErrorEmbedding::new(
                vec![0.0; self.output_dim],
                ErrorCode::new(
                    "E0000",
                    super::ErrorCategory::Unknown,
                    super::Difficulty::Easy,
                ),
                0,
            );
        }

        // 1. Convert node features to tensor
        let node_tensor = self.graph_to_tensor(graph);
        let adj = self.graph_to_adjacency(graph);

        // 2. Apply GNN layers
        let h1 = self.gcn1.forward(&node_tensor, &adj);
        let h1_relu = Self::relu(&h1);
        let h2 = self.sage.forward(&h1_relu, &adj);
        let h2_relu = Self::relu(&h2);
        let h3 = self.gcn2.forward(&h2_relu, &adj);

        // 3. Mean pool over all nodes
        let embedding = self.mean_pool(&h3, graph.num_nodes());

        // 4. Normalize
        let normalized = self.normalize_embedding(&embedding);

        // Extract error code from graph (diagnostic node)
        let error_code = self.extract_error_code_from_graph(graph);
        let context_hash = self.compute_graph_hash(graph);

        ErrorEmbedding::new(normalized, error_code, context_hash)
    }

    /// Encode a diagnostic directly (convenience method).
    ///
    /// This builds the graph and encodes it in one step.
    #[must_use]
    pub fn encode(&self, diagnostic: &CompilerDiagnostic, source: &str) -> ErrorEmbedding {
        let graph = self.build_graph(diagnostic, source);
        self.encode_graph(&graph)
    }

    /// Extract features for the diagnostic node.
    fn extract_diagnostic_features(&self, diagnostic: &CompilerDiagnostic) -> Vec<f32> {
        let mut features = vec![0.0f32; 64 + self.node_type_dim];

        // Error code features (first 32 dims)
        let code_hash = Self::simple_hash(&diagnostic.code.code);
        for (i, feature) in features.iter_mut().take(32).enumerate() {
            *feature = ((code_hash >> (i % 64)) & 1) as f32;
        }

        // Message features (next 32 dims)
        let msg_lower = diagnostic.message.to_lowercase();
        let keywords = [
            "type",
            "borrow",
            "move",
            "lifetime",
            "trait",
            "impl",
            "expected",
            "found",
            "cannot",
            "missing",
            "unknown",
            "value",
            "reference",
            "mutable",
            "method",
            "function",
            "argument",
            "return",
            "copy",
            "clone",
            "bound",
            "satisfy",
            "require",
            "import",
            "module",
            "crate",
            "use",
            "struct",
            "enum",
            "unsafe",
            "async",
            "await",
        ];
        for (i, kw) in keywords.iter().enumerate().take(32) {
            features[32 + i] = if msg_lower.contains(kw) { 1.0 } else { 0.0 };
        }

        // Node type embedding (last 8 dims) - Diagnostic type
        features[64] = 1.0; // is_diagnostic
        features[65] = match diagnostic.code.category {
            super::ErrorCategory::TypeMismatch => 1.0,
            _ => 0.0,
        };
        features[66] = match diagnostic.code.category {
            super::ErrorCategory::Ownership => 1.0,
            _ => 0.0,
        };
        features[67] = match diagnostic.code.category {
            super::ErrorCategory::Lifetime => 1.0,
            _ => 0.0,
        };
        features[68] = match diagnostic.code.category {
            super::ErrorCategory::TraitBound => 1.0,
            _ => 0.0,
        };
        features[69] = match diagnostic.code.category {
            super::ErrorCategory::Import => 1.0,
            _ => 0.0,
        };
        features[70] = match diagnostic.code.difficulty {
            super::Difficulty::Easy => 0.25,
            super::Difficulty::Medium => 0.5,
            super::Difficulty::Hard => 0.75,
            super::Difficulty::Expert => 1.0,
        };

        features
    }

    /// Extract features for a type node.
    fn extract_type_node_features(&self, type_name: &str, is_expected: bool) -> Vec<f32> {
        let mut features = vec![0.0f32; 64 + self.node_type_dim];

        // Type name features
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
            ("&", 16),
            ("mut", 17),
            ("'", 18),
            ("<", 19),
            ("impl", 20),
            ("dyn", 21),
            ("Rc", 22),
            ("Arc", 23),
            ("Cell", 24),
            ("RefCell", 25),
            ("Pin", 26),
            ("Future", 27),
            ("Iterator", 28),
            ("IntoIterator", 29),
            ("Clone", 30),
            ("Copy", 31),
        ];

        for (pattern, idx) in &type_patterns {
            if type_name.contains(pattern) {
                features[*idx] = 1.0;
            }
        }

        // Type complexity (32-47)
        features[32] = type_name.len() as f32 / 50.0;
        features[33] = type_name.matches('<').count() as f32 / 3.0;
        features[34] = type_name.matches('&').count() as f32 / 2.0;
        features[35] = type_name.matches('\'').count() as f32 / 2.0;

        // Node type embedding
        features[64] = 0.0; // not diagnostic
        features[65] = if is_expected { 1.0 } else { 0.0 };
        features[66] = if is_expected { 0.0 } else { 1.0 };
        features[67] = 1.0; // is_type_node

        features
    }

    /// Extract AST nodes from source context.
    fn extract_ast_nodes(
        &self,
        graph: &mut ProgramFeedbackGraph,
        source: &str,
        span: &SourceSpan,
    ) -> Vec<usize> {
        let mut indices = Vec::new();
        let lines: Vec<&str> = source.lines().collect();

        let start_line = span.line_start.saturating_sub(1);
        let end_line = span.line_end.min(lines.len());

        // Extract tokens from the error region
        for line in lines.iter().take(end_line).skip(start_line) {
            for token in Self::tokenize_rust(line) {
                let features = self.extract_token_features(&token);
                let idx = graph.add_node(NodeType::Ast, features);
                indices.push(idx);

                // Limit to avoid huge graphs
                if indices.len() >= 20 {
                    return indices;
                }
            }
        }

        indices
    }

    /// Simple Rust tokenizer for AST extraction.
    fn tokenize_rust(line: &str) -> Vec<String> {
        let keywords = [
            "fn", "let", "mut", "struct", "impl", "trait", "use", "mod", "pub", "self", "Self",
            "return", "if", "else", "match", "for", "while", "loop", "break", "continue", "async",
            "await", "move", "ref", "where", "type", "const", "static", "enum", "union",
        ];

        let mut tokens = Vec::new();
        let mut current = String::new();

        for c in line.chars() {
            if c.is_alphanumeric() || c == '_' {
                current.push(c);
            } else {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                // Include significant punctuation as tokens
                if matches!(
                    c,
                    ':' | ';'
                        | ','
                        | '.'
                        | '&'
                        | '*'
                        | '<'
                        | '>'
                        | '('
                        | ')'
                        | '{'
                        | '}'
                        | '['
                        | ']'
                        | '='
                        | '-'
                        | '+'
                        | '!'
                        | '?'
                ) {
                    tokens.push(c.to_string());
                }
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }

        // Prioritize keywords
        let keyword_set: std::collections::HashSet<&str> = keywords.iter().copied().collect();
        tokens.sort_by(|a, b| {
            let a_is_kw = keyword_set.contains(a.as_str());
            let b_is_kw = keyword_set.contains(b.as_str());
            b_is_kw.cmp(&a_is_kw)
        });

        tokens.into_iter().take(10).collect()
    }

    /// Extract features for a token node.
    fn extract_token_features(&self, token: &str) -> Vec<f32> {
        let mut features = vec![0.0f32; 64 + self.node_type_dim];

        // Token hash features
        let hash = Self::simple_hash(token);
        for (i, feature) in features.iter_mut().take(32).enumerate() {
            *feature = ((hash >> (i % 64)) & 1) as f32;
        }

        // Token type features
        let keywords = [
            "fn", "let", "mut", "struct", "impl", "trait", "use", "mod", "pub", "self", "Self",
            "return", "if", "else", "match", "for",
        ];
        for (i, kw) in keywords.iter().enumerate().take(16) {
            features[32 + i] = if token == *kw { 1.0 } else { 0.0 };
        }

        // Token characteristics
        features[48] = token.len() as f32 / 20.0;
        features[49] = if token.chars().all(|c| c.is_uppercase() || c == '_') {
            1.0
        } else {
            0.0
        };
        features[50] = if token.starts_with(char::is_uppercase) {
            1.0
        } else {
            0.0
        };
        features[51] = if token.chars().all(char::is_numeric) {
            1.0
        } else {
            0.0
        };

        // Node type embedding - AST node
        features[64] = 0.0; // not diagnostic
        features[71] = 1.0; // is_ast_node

        features
    }

    /// Extract features for a suggestion node.
    fn extract_suggestion_features(&self, suggestion: &str) -> Vec<f32> {
        let mut features = vec![0.0f32; 64 + self.node_type_dim];

        // Suggestion content features
        let patterns = [
            ("add", 0),
            ("remove", 1),
            ("change", 2),
            ("use", 3),
            ("import", 4),
            ("borrow", 5),
            ("clone", 6),
            ("into", 7),
            ("as", 8),
            ("try", 9),
            ("unwrap", 10),
            ("expect", 11),
            ("lifetime", 12),
            ("'static", 13),
            ("move", 14),
            ("ref", 15),
        ];
        let sugg_lower = suggestion.to_lowercase();
        for (pattern, idx) in &patterns {
            features[*idx] = if sugg_lower.contains(pattern) {
                1.0
            } else {
                0.0
            };
        }

        // Node type embedding - Suggestion node
        features[64] = 0.0;
        features[68] = 1.0; // is_suggestion

        features
    }

    /// Convert graph node features to tensor.
    fn graph_to_tensor(&self, graph: &ProgramFeedbackGraph) -> Tensor {
        let num_nodes = graph.num_nodes();
        let feature_dim = 64 + self.node_type_dim;

        let mut data = vec![0.0f32; num_nodes * feature_dim];
        for (i, features) in graph.node_features.iter().enumerate() {
            for (j, &val) in features.iter().enumerate().take(feature_dim) {
                data[i * feature_dim + j] = val;
            }
        }

        Tensor::new(&data, &[num_nodes, feature_dim])
    }

    /// Convert graph edges to adjacency matrix.
    #[allow(clippy::unused_self)]
    fn graph_to_adjacency(&self, graph: &ProgramFeedbackGraph) -> AdjacencyMatrix {
        let edges: Vec<[usize; 2]> = graph.edges.iter().map(|&(s, t)| [s, t]).collect();

        // Make edges bidirectional for message passing
        let mut all_edges = edges.clone();
        for &[s, t] in &edges {
            all_edges.push([t, s]);
        }

        AdjacencyMatrix::from_edge_index(&all_edges, graph.num_nodes()).add_self_loops()
    }

    /// Apply `ReLU` activation.
    fn relu(tensor: &Tensor) -> Tensor {
        let data: Vec<f32> = tensor.data().iter().map(|&x| x.max(0.0)).collect();
        Tensor::new(&data, tensor.shape())
    }

    /// Mean pool over nodes.
    fn mean_pool(&self, tensor: &Tensor, num_nodes: usize) -> Vec<f32> {
        let data = tensor.data();
        let feature_dim = if num_nodes > 0 {
            data.len() / num_nodes
        } else {
            self.output_dim
        };

        let mut pooled = vec![0.0f32; feature_dim];
        if num_nodes == 0 {
            return pooled;
        }

        for node in 0..num_nodes {
            for f in 0..feature_dim {
                pooled[f] += data[node * feature_dim + f];
            }
        }

        for val in &mut pooled {
            *val /= num_nodes as f32;
        }

        pooled
    }

    /// Normalize embedding to unit length.
    #[allow(clippy::unused_self)]
    fn normalize_embedding(&self, embedding: &[f32]) -> Vec<f32> {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-8 {
            return embedding.to_vec();
        }
        embedding.iter().map(|&x| x / norm).collect()
    }

    /// Extract error code from graph (from diagnostic node).
    #[allow(clippy::unused_self)]
    fn extract_error_code_from_graph(&self, graph: &ProgramFeedbackGraph) -> ErrorCode {
        // Find diagnostic node and infer error category from features
        for (i, node_type) in graph.node_types.iter().enumerate() {
            if *node_type == NodeType::Diagnostic {
                let features = &graph.node_features[i];
                // Decode category from node type embedding (indices 65-69)
                let category = if features.get(65).copied().unwrap_or(0.0) > 0.5 {
                    super::ErrorCategory::TypeMismatch
                } else if features.get(66).copied().unwrap_or(0.0) > 0.5 {
                    super::ErrorCategory::Ownership
                } else if features.get(67).copied().unwrap_or(0.0) > 0.5 {
                    super::ErrorCategory::Lifetime
                } else if features.get(68).copied().unwrap_or(0.0) > 0.5 {
                    super::ErrorCategory::TraitBound
                } else if features.get(69).copied().unwrap_or(0.0) > 0.5 {
                    super::ErrorCategory::Import
                } else {
                    super::ErrorCategory::Unknown
                };

                let difficulty = if features.get(70).copied().unwrap_or(0.0) > 0.8 {
                    super::Difficulty::Expert
                } else if features.get(70).copied().unwrap_or(0.0) > 0.6 {
                    super::Difficulty::Hard
                } else if features.get(70).copied().unwrap_or(0.0) > 0.4 {
                    super::Difficulty::Medium
                } else {
                    super::Difficulty::Easy
                };

                return ErrorCode::new("E0000", category, difficulty);
            }
        }

        ErrorCode::new(
            "E0000",
            super::ErrorCategory::Unknown,
            super::Difficulty::Easy,
        )
    }

    /// Compute hash of graph structure for deduplication.
    #[allow(clippy::unused_self)]
    fn compute_graph_hash(&self, graph: &ProgramFeedbackGraph) -> u64 {
        let mut hash: u64 = 5381;
        hash = hash.wrapping_mul(33).wrapping_add(graph.num_nodes() as u64);
        hash = hash.wrapping_mul(33).wrapping_add(graph.num_edges() as u64);

        for node_type in &graph.node_types {
            hash = hash.wrapping_mul(33).wrapping_add(*node_type as u64);
        }

        hash
    }

    /// Simple hash function.
    fn simple_hash(s: &str) -> u64 {
        let mut hash: u64 = 5381;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(u64::from(byte));
        }
        hash
    }
}

impl Default for GNNErrorEncoder {
    fn default() -> Self {
        Self::default_config()
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
mod tests;
