impl GNNErrorEncoder {

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
    ///
    /// ONE PATH: Delegates to `nn::functional::relu` (UCBD §4).
    fn relu(tensor: &Tensor) -> Tensor {
        crate::nn::functional::relu(tensor)
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
