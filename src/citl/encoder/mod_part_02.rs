
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
