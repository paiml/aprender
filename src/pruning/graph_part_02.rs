
impl PruningPlan {
    /// Create a new empty pruning plan.
    #[must_use]
    pub fn new() -> Self {
        Self {
            channel_removals: HashMap::new(),
            layer_removals: Vec::new(),
            validated: false,
        }
    }

    /// Add channel removals for a layer.
    pub fn remove_channels(&mut self, layer_id: impl Into<String>, channels: Vec<usize>) {
        self.channel_removals.insert(layer_id.into(), channels);
        self.validated = false;
    }

    /// Add a layer to be removed.
    pub fn remove_layer(&mut self, layer_id: impl Into<String>) {
        self.layer_removals.push(layer_id.into());
        self.validated = false;
    }

    /// Get channels to remove from a layer.
    #[must_use]
    pub fn channels_to_remove(&self, layer_id: &str) -> Option<&Vec<usize>> {
        self.channel_removals.get(layer_id)
    }

    /// Check if a layer should be removed.
    #[must_use]
    pub fn is_layer_removed(&self, layer_id: &str) -> bool {
        self.layer_removals.contains(&layer_id.to_string())
    }

    /// Validate the plan against a dependency graph.
    ///
    /// Ensures:
    /// - All referenced layers exist
    /// - Channel indices are valid
    /// - Dependencies are satisfied
    pub fn validate(&mut self, graph: &DependencyGraph) -> Result<(), PruningError> {
        // Check all referenced layers exist
        for layer_id in self.channel_removals.keys() {
            if graph.get_node(layer_id).is_none() {
                return Err(PruningError::InvalidPattern {
                    message: format!("Layer '{layer_id}' not found in graph"),
                });
            }
        }

        for layer_id in &self.layer_removals {
            if graph.get_node(layer_id).is_none() {
                return Err(PruningError::InvalidPattern {
                    message: format!("Layer '{layer_id}' not found in graph"),
                });
            }
        }

        // Check channel indices are valid
        for (layer_id, channels) in &self.channel_removals {
            if let Some(node) = graph.get_node(layer_id) {
                for &ch in channels {
                    if ch >= node.output_dim && node.output_dim > 0 {
                        return Err(PruningError::InvalidSparsity {
                            value: ch as f32,
                            constraint: format!(
                                "Channel {} >= output_dim {} for layer {}",
                                ch, node.output_dim, layer_id
                            ),
                        });
                    }
                }
            }
        }

        self.validated = true;
        Ok(())
    }

    /// Check if plan is validated.
    #[must_use]
    pub fn is_validated(&self) -> bool {
        self.validated
    }

    /// Get total number of channels being removed.
    pub fn total_channels_removed(&self) -> usize {
        self.channel_removals.values().map(Vec::len).sum()
    }

    /// Get number of layers being removed.
    #[must_use]
    pub fn total_layers_removed(&self) -> usize {
        self.layer_removals.len()
    }
}

impl Default for PruningPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Propagate channel pruning through a dependency graph.
///
/// When output channels are pruned from a layer, this function
/// determines which input channels of downstream layers must also be pruned.
///
/// # Arguments
/// * `graph` - The dependency graph
/// * `source_layer` - Layer where channels are being pruned
/// * `pruned_channels` - Indices of channels being removed
///
/// # Returns
/// Map of `layer_id` -> input channels to remove
#[must_use]
pub fn propagate_channel_pruning(
    graph: &DependencyGraph,
    source_layer: &str,
    pruned_channels: &[usize],
) -> HashMap<String, Vec<usize>> {
    let mut result = HashMap::new();

    // Find all downstream layers connected via sequential edges
    for edge in graph.edges_from(source_layer) {
        if edge.dep_type == DependencyType::Sequential {
            // Downstream layer needs same input channels removed
            result.insert(edge.to.clone(), pruned_channels.to_vec());
        }
    }

    result
}
