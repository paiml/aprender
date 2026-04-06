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
}
