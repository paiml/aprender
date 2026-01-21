//! Neural Architecture Search (NAS) Primitives
//!
//! Building blocks for automated neural network architecture design.
//! Supports encoding architectures as searchable representations.
//!
//! # Example
//!
//! ```
//! use aprender::metaheuristics::nas::{NasSearchSpace, LayerType, NasGenome};
//!
//! // Define search space for a small network
//! let space = NasSearchSpace::new()
//!     .with_max_layers(5)
//!     .with_layer_types(vec![LayerType::Dense, LayerType::Conv2d, LayerType::Dropout])
//!     .with_units_range(16, 512)
//!     .with_activation_choices(&["relu", "tanh", "sigmoid"]);
//!
//! // Create a random architecture
//! let genome = NasGenome::random(&space, 42);
//! println!("Architecture: {:?}", genome.layers());
//! ```

use rand::prelude::*;

/// Types of neural network layers supported in NAS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerType {
    /// Fully connected layer
    Dense,
    /// 2D Convolution
    Conv2d,
    /// Max pooling
    MaxPool2d,
    /// Average pooling
    AvgPool2d,
    /// Batch normalization
    BatchNorm,
    /// Dropout regularization
    Dropout,
    /// Skip/residual connection
    Skip,
    /// LSTM recurrent layer
    Lstm,
    /// Attention mechanism
    Attention,
}

impl LayerType {
    /// Get all standard layer types.
    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::Dense,
            Self::Conv2d,
            Self::MaxPool2d,
            Self::AvgPool2d,
            Self::BatchNorm,
            Self::Dropout,
            Self::Skip,
            Self::Lstm,
            Self::Attention,
        ]
    }
}

/// Configuration for a single layer in the architecture.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerConfig {
    /// Type of layer
    pub layer_type: LayerType,
    /// Number of units/filters (for Dense/Conv2d)
    pub units: Option<usize>,
    /// Kernel size (for Conv2d)
    pub kernel_size: Option<usize>,
    /// Activation function name
    pub activation: Option<String>,
    /// Dropout rate (for Dropout layer)
    pub dropout_rate: Option<f64>,
    /// Whether this layer is active (for pruning)
    pub active: bool,
}

impl LayerConfig {
    /// Create a new layer configuration.
    #[must_use]
    pub fn new(layer_type: LayerType) -> Self {
        Self {
            layer_type,
            units: None,
            kernel_size: None,
            activation: None,
            dropout_rate: None,
            active: true,
        }
    }

    /// Set number of units/filters.
    #[must_use]
    pub fn with_units(mut self, units: usize) -> Self {
        self.units = Some(units);
        self
    }

    /// Set kernel size.
    #[must_use]
    pub fn with_kernel_size(mut self, size: usize) -> Self {
        self.kernel_size = Some(size);
        self
    }

    /// Set activation function.
    #[must_use]
    pub fn with_activation(mut self, activation: &str) -> Self {
        self.activation = Some(activation.to_string());
        self
    }

    /// Set dropout rate.
    #[must_use]
    pub fn with_dropout_rate(mut self, rate: f64) -> Self {
        self.dropout_rate = Some(rate.clamp(0.0, 1.0));
        self
    }
}

/// Search space definition for NAS.
#[derive(Debug, Clone)]
pub struct NasSearchSpace {
    /// Maximum number of layers
    pub max_layers: usize,
    /// Minimum number of layers
    pub min_layers: usize,
    /// Allowed layer types
    pub layer_types: Vec<LayerType>,
    /// Range of units for Dense layers
    pub units_range: (usize, usize),
    /// Range of filters for Conv layers
    pub filters_range: (usize, usize),
    /// Allowed kernel sizes
    pub kernel_sizes: Vec<usize>,
    /// Allowed activation functions
    pub activations: Vec<String>,
    /// Dropout rate range
    pub dropout_range: (f64, f64),
}

impl Default for NasSearchSpace {
    fn default() -> Self {
        Self {
            max_layers: 10,
            min_layers: 1,
            layer_types: vec![LayerType::Dense, LayerType::Dropout],
            units_range: (32, 256),
            filters_range: (16, 128),
            kernel_sizes: vec![3, 5, 7],
            activations: vec!["relu".to_string(), "tanh".to_string()],
            dropout_range: (0.1, 0.5),
        }
    }
}

impl NasSearchSpace {
    /// Create a new NAS search space with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum layers.
    #[must_use]
    pub fn with_max_layers(mut self, max: usize) -> Self {
        self.max_layers = max.max(1);
        self
    }

    /// Set minimum layers.
    #[must_use]
    pub fn with_min_layers(mut self, min: usize) -> Self {
        self.min_layers = min.max(1);
        self
    }

    /// Set allowed layer types.
    #[must_use]
    pub fn with_layer_types(mut self, types: Vec<LayerType>) -> Self {
        if !types.is_empty() {
            self.layer_types = types;
        }
        self
    }

    /// Set units range for Dense layers.
    #[must_use]
    pub fn with_units_range(mut self, min: usize, max: usize) -> Self {
        self.units_range = (min.max(1), max.max(min + 1));
        self
    }

    /// Set filters range for Conv layers.
    #[must_use]
    pub fn with_filters_range(mut self, min: usize, max: usize) -> Self {
        self.filters_range = (min.max(1), max.max(min + 1));
        self
    }

    /// Set allowed kernel sizes.
    #[must_use]
    pub fn with_kernel_sizes(mut self, sizes: Vec<usize>) -> Self {
        if !sizes.is_empty() {
            self.kernel_sizes = sizes;
        }
        self
    }

    /// Set allowed activation functions.
    #[must_use]
    pub fn with_activation_choices(mut self, activations: &[&str]) -> Self {
        if !activations.is_empty() {
            self.activations = activations.iter().map(|s| (*s).to_string()).collect();
        }
        self
    }

    /// Set dropout rate range.
    #[must_use]
    pub fn with_dropout_range(mut self, min: f64, max: f64) -> Self {
        self.dropout_range = (min.clamp(0.0, 1.0), max.clamp(min, 1.0));
        self
    }
}

/// A genome representing a neural network architecture.
#[derive(Debug, Clone)]
pub struct NasGenome {
    /// Layers in the architecture
    layers: Vec<LayerConfig>,
    /// Fitness score (lower is better for minimization)
    fitness: Option<f64>,
}

impl NasGenome {
    /// Create empty genome.
    #[must_use]
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            fitness: None,
        }
    }

    /// Create genome from layers.
    #[must_use]
    pub fn from_layers(layers: Vec<LayerConfig>) -> Self {
        Self {
            layers,
            fitness: None,
        }
    }

    /// Generate a random architecture within the search space.
    #[must_use]
    pub fn random(space: &NasSearchSpace, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let n_layers = rng.gen_range(space.min_layers..=space.max_layers);
        let mut layers = Vec::with_capacity(n_layers);

        for _ in 0..n_layers {
            let layer_type = space.layer_types[rng.gen_range(0..space.layer_types.len())];
            let mut config = LayerConfig::new(layer_type);

            match layer_type {
                LayerType::Dense => {
                    config.units = Some(rng.gen_range(space.units_range.0..=space.units_range.1));
                    config.activation =
                        Some(space.activations[rng.gen_range(0..space.activations.len())].clone());
                }
                LayerType::Conv2d => {
                    config.units =
                        Some(rng.gen_range(space.filters_range.0..=space.filters_range.1));
                    config.kernel_size =
                        Some(space.kernel_sizes[rng.gen_range(0..space.kernel_sizes.len())]);
                    config.activation =
                        Some(space.activations[rng.gen_range(0..space.activations.len())].clone());
                }
                LayerType::Dropout => {
                    config.dropout_rate =
                        Some(rng.gen_range(space.dropout_range.0..=space.dropout_range.1));
                }
                LayerType::Lstm => {
                    config.units = Some(rng.gen_range(space.units_range.0..=space.units_range.1));
                }
                _ => {}
            }

            layers.push(config);
        }

        Self {
            layers,
            fitness: None,
        }
    }

    /// Get the layers.
    #[must_use]
    pub fn layers(&self) -> &[LayerConfig] {
        &self.layers
    }

    /// Get mutable layers.
    pub fn layers_mut(&mut self) -> &mut Vec<LayerConfig> {
        &mut self.layers
    }

    /// Get number of layers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if genome is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Get fitness score.
    #[must_use]
    pub fn fitness(&self) -> Option<f64> {
        self.fitness
    }

    /// Set fitness score.
    pub fn set_fitness(&mut self, fitness: f64) {
        self.fitness = Some(fitness);
    }

    /// Count active layers.
    #[must_use]
    pub fn active_layers(&self) -> usize {
        self.layers.iter().filter(|l| l.active).count()
    }

    /// Encode genome to continuous vector for optimization.
    #[must_use]
    pub fn encode(&self, space: &NasSearchSpace) -> Vec<f64> {
        let mut encoding = Vec::new();

        for layer in &self.layers {
            // Encode layer type as index
            let type_idx = space
                .layer_types
                .iter()
                .position(|t| *t == layer.layer_type)
                .unwrap_or(0);
            encoding.push(type_idx as f64 / space.layer_types.len().max(1) as f64);

            // Encode units (normalized)
            if let Some(units) = layer.units {
                let range = space.units_range.1 - space.units_range.0;
                encoding.push((units - space.units_range.0) as f64 / range.max(1) as f64);
            } else {
                encoding.push(0.5);
            }

            // Encode activation as index
            if let Some(ref activation) = layer.activation {
                let act_idx = space
                    .activations
                    .iter()
                    .position(|a| a == activation)
                    .unwrap_or(0);
                encoding.push(act_idx as f64 / space.activations.len().max(1) as f64);
            } else {
                encoding.push(0.0);
            }

            // Encode dropout rate
            encoding.push(layer.dropout_rate.unwrap_or(0.0));

            // Encode active flag
            encoding.push(if layer.active { 1.0 } else { 0.0 });
        }

        encoding
    }

    /// Decode continuous vector back to genome.
    #[must_use]
    pub fn decode(encoding: &[f64], space: &NasSearchSpace, n_layers: usize) -> Self {
        let mut layers = Vec::with_capacity(n_layers);
        let stride = 5; // 5 values per layer

        for i in 0..n_layers {
            let base = i * stride;
            if base + stride > encoding.len() {
                break;
            }

            // Decode layer type
            let type_idx = (encoding[base] * space.layer_types.len() as f64).floor() as usize
                % space.layer_types.len().max(1);
            let layer_type = space
                .layer_types
                .get(type_idx)
                .copied()
                .unwrap_or(LayerType::Dense);

            let mut config = LayerConfig::new(layer_type);

            // Decode units
            let range = space.units_range.1 - space.units_range.0;
            config.units =
                Some((encoding[base + 1] * range as f64).round() as usize + space.units_range.0);

            // Decode activation
            let act_idx = (encoding[base + 2] * space.activations.len() as f64).floor() as usize
                % space.activations.len().max(1);
            config.activation = space.activations.get(act_idx).cloned();

            // Decode dropout
            config.dropout_rate = Some(encoding[base + 3].clamp(0.0, 1.0));

            // Decode active flag
            config.active = encoding[base + 4] > 0.5;

            layers.push(config);
        }

        Self {
            layers,
            fitness: None,
        }
    }
}

impl Default for NasGenome {
    fn default() -> Self {
        Self::new()
    }
}

/// Mutation operators for NAS genomes.
#[derive(Debug, Clone, Copy)]
pub enum NasMutation {
    /// Add a random layer
    AddLayer,
    /// Remove a random layer
    RemoveLayer,
    /// Change layer type
    ChangeType,
    /// Modify layer parameters
    ModifyParams,
    /// Toggle layer active state
    ToggleActive,
}

/// Mutate a genome using the specified operator.
pub fn mutate_genome(
    genome: &mut NasGenome,
    mutation: NasMutation,
    space: &NasSearchSpace,
    rng: &mut impl Rng,
) {
    match mutation {
        NasMutation::AddLayer => mutate_add_layer(genome, space, rng),
        NasMutation::RemoveLayer => mutate_remove_layer(genome, space, rng),
        NasMutation::ChangeType => mutate_change_type(genome, space, rng),
        NasMutation::ModifyParams => mutate_modify_params(genome, space, rng),
        NasMutation::ToggleActive => mutate_toggle_active(genome, rng),
    }
}

fn mutate_add_layer(genome: &mut NasGenome, space: &NasSearchSpace, rng: &mut impl Rng) {
    if genome.len() >= space.max_layers {
        return;
    }
    let layer_type = space.layer_types[rng.gen_range(0..space.layer_types.len())];
    let mut config = LayerConfig::new(layer_type);

    if matches!(layer_type, LayerType::Dense | LayerType::Lstm) {
        config.units = Some(rng.gen_range(space.units_range.0..=space.units_range.1));
    }
    if !space.activations.is_empty() {
        config.activation =
            Some(space.activations[rng.gen_range(0..space.activations.len())].clone());
    }

    let pos = rng.gen_range(0..=genome.len());
    genome.layers_mut().insert(pos, config);
}

fn mutate_remove_layer(genome: &mut NasGenome, space: &NasSearchSpace, rng: &mut impl Rng) {
    if genome.len() > space.min_layers {
        let idx = rng.gen_range(0..genome.len());
        genome.layers_mut().remove(idx);
    }
}

fn mutate_change_type(genome: &mut NasGenome, space: &NasSearchSpace, rng: &mut impl Rng) {
    if !genome.is_empty() && !space.layer_types.is_empty() {
        let idx = rng.gen_range(0..genome.len());
        let new_type = space.layer_types[rng.gen_range(0..space.layer_types.len())];
        genome.layers_mut()[idx].layer_type = new_type;
    }
}

fn mutate_modify_params(genome: &mut NasGenome, space: &NasSearchSpace, rng: &mut impl Rng) {
    if genome.is_empty() {
        return;
    }
    let idx = rng.gen_range(0..genome.len());
    let layer = &mut genome.layers_mut()[idx];

    if let Some(units) = layer.units {
        let delta = rng.gen_range(-32i64..=32);
        let new_units =
            (units as i64 + delta).clamp(space.units_range.0 as i64, space.units_range.1 as i64);
        layer.units = Some(new_units as usize);
    }

    if rng.gen_bool(0.3) && !space.activations.is_empty() {
        layer.activation =
            Some(space.activations[rng.gen_range(0..space.activations.len())].clone());
    }
}

fn mutate_toggle_active(genome: &mut NasGenome, rng: &mut impl Rng) {
    if !genome.is_empty() {
        let idx = rng.gen_range(0..genome.len());
        let layer = &mut genome.layers_mut()[idx];
        layer.active = !layer.active;
    }
}

/// Crossover two parent genomes to produce offspring.
pub fn crossover_genomes(
    parent1: &NasGenome,
    parent2: &NasGenome,
    rng: &mut impl Rng,
) -> NasGenome {
    // Single-point crossover
    if parent1.is_empty() {
        return parent2.clone();
    }
    if parent2.is_empty() {
        return parent1.clone();
    }

    let cut1 = rng.gen_range(0..=parent1.len());
    let cut2 = rng.gen_range(0..=parent2.len());

    let mut child_layers = Vec::new();

    // Take first part from parent1
    child_layers.extend(parent1.layers()[..cut1].iter().cloned());

    // Take second part from parent2
    child_layers.extend(parent2.layers()[cut2..].iter().cloned());

    NasGenome::from_layers(child_layers)
}

/// Evaluate architecture complexity (proxy for compute cost).
#[must_use]
pub fn architecture_complexity(genome: &NasGenome) -> f64 {
    let mut complexity = 0.0;

    for layer in genome.layers() {
        if !layer.active {
            continue;
        }

        let base = match layer.layer_type {
            LayerType::Dense => 1.0,
            LayerType::Conv2d => 2.0,
            LayerType::Lstm => 4.0,
            LayerType::Attention => 3.0,
            LayerType::BatchNorm | LayerType::MaxPool2d | LayerType::AvgPool2d => 0.1,
            LayerType::Dropout | LayerType::Skip => 0.0,
        };

        let units_factor = layer.units.unwrap_or(64) as f64 / 64.0;
        complexity += base * units_factor;
    }

    complexity
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================
    // EXTREME TDD: Tests written first
    // ==========================================================

    #[test]
    fn test_layer_type_all() {
        let types = LayerType::all();
        assert!(types.len() >= 5, "Should have multiple layer types");
        assert!(types.contains(&LayerType::Dense));
        assert!(types.contains(&LayerType::Conv2d));
    }

    #[test]
    fn test_layer_config_builder() {
        let config = LayerConfig::new(LayerType::Dense)
            .with_units(128)
            .with_activation("relu")
            .with_dropout_rate(0.5);

        assert_eq!(config.layer_type, LayerType::Dense);
        assert_eq!(config.units, Some(128));
        assert_eq!(config.activation, Some("relu".to_string()));
        assert_eq!(config.dropout_rate, Some(0.5));
        assert!(config.active);
    }

    #[test]
    fn test_nas_search_space_builder() {
        let space = NasSearchSpace::new()
            .with_max_layers(8)
            .with_min_layers(2)
            .with_units_range(64, 512)
            .with_activation_choices(&["relu", "gelu"]);

        assert_eq!(space.max_layers, 8);
        assert_eq!(space.min_layers, 2);
        assert_eq!(space.units_range, (64, 512));
        assert_eq!(space.activations.len(), 2);
    }

    #[test]
    fn test_nas_genome_random() {
        let space = NasSearchSpace::new().with_max_layers(5).with_min_layers(2);

        let genome = NasGenome::random(&space, 42);

        assert!(genome.len() >= 2);
        assert!(genome.len() <= 5);
        assert!(!genome.is_empty());
    }

    #[test]
    fn test_nas_genome_deterministic() {
        let space = NasSearchSpace::new();

        let genome1 = NasGenome::random(&space, 42);
        let genome2 = NasGenome::random(&space, 42);

        assert_eq!(genome1.len(), genome2.len());
    }

    #[test]
    fn test_nas_genome_encode_decode() {
        let space = NasSearchSpace::new().with_max_layers(3).with_min_layers(3);

        let original = NasGenome::random(&space, 42);
        let encoded = original.encode(&space);
        let decoded = NasGenome::decode(&encoded, &space, original.len());

        assert_eq!(decoded.len(), original.len());
    }

    #[test]
    fn test_mutation_add_layer() {
        let space = NasSearchSpace::new().with_max_layers(10);
        let mut genome = NasGenome::from_layers(vec![LayerConfig::new(LayerType::Dense)]);
        let original_len = genome.len();

        let mut rng = StdRng::seed_from_u64(42);
        mutate_genome(&mut genome, NasMutation::AddLayer, &space, &mut rng);

        assert_eq!(genome.len(), original_len + 1);
    }

    #[test]
    fn test_mutation_remove_layer() {
        let space = NasSearchSpace::new().with_min_layers(1);
        let mut genome = NasGenome::from_layers(vec![
            LayerConfig::new(LayerType::Dense),
            LayerConfig::new(LayerType::Dropout),
        ]);

        let mut rng = StdRng::seed_from_u64(42);
        mutate_genome(&mut genome, NasMutation::RemoveLayer, &space, &mut rng);

        assert_eq!(genome.len(), 1);
    }

    #[test]
    fn test_mutation_respects_bounds() {
        let space = NasSearchSpace::new().with_max_layers(2).with_min_layers(2);

        let mut genome = NasGenome::from_layers(vec![
            LayerConfig::new(LayerType::Dense),
            LayerConfig::new(LayerType::Dense),
        ]);

        let mut rng = StdRng::seed_from_u64(42);

        // Try to add - should not exceed max
        mutate_genome(&mut genome, NasMutation::AddLayer, &space, &mut rng);
        assert!(genome.len() <= 2);

        // Try to remove - should not go below min
        mutate_genome(&mut genome, NasMutation::RemoveLayer, &space, &mut rng);
        assert!(genome.len() >= 2);
    }

    #[test]
    fn test_crossover_genomes() {
        let parent1 = NasGenome::from_layers(vec![
            LayerConfig::new(LayerType::Dense).with_units(64),
            LayerConfig::new(LayerType::Dense).with_units(128),
        ]);
        let parent2 = NasGenome::from_layers(vec![
            LayerConfig::new(LayerType::Conv2d).with_units(32),
            LayerConfig::new(LayerType::Dropout),
        ]);

        let mut rng = StdRng::seed_from_u64(42);
        let child = crossover_genomes(&parent1, &parent2, &mut rng);

        assert!(!child.is_empty());
    }

    #[test]
    fn test_architecture_complexity() {
        let simple =
            NasGenome::from_layers(vec![LayerConfig::new(LayerType::Dense).with_units(64)]);

        let complex = NasGenome::from_layers(vec![
            LayerConfig::new(LayerType::Conv2d).with_units(128),
            LayerConfig::new(LayerType::Lstm).with_units(256),
        ]);

        let simple_complexity = architecture_complexity(&simple);
        let complex_complexity = architecture_complexity(&complex);

        assert!(complex_complexity > simple_complexity);
    }

    #[test]
    fn test_inactive_layers_not_counted() {
        let mut layer = LayerConfig::new(LayerType::Dense).with_units(128);
        layer.active = false;

        let genome = NasGenome::from_layers(vec![layer]);

        assert_eq!(genome.active_layers(), 0);
        assert!(architecture_complexity(&genome) < 0.01);
    }

    #[test]
    fn test_fitness_tracking() {
        let mut genome = NasGenome::new();
        assert!(genome.fitness().is_none());

        genome.set_fitness(0.95);
        assert_eq!(genome.fitness(), Some(0.95));
    }

    #[test]
    fn test_toggle_active_mutation() {
        let space = NasSearchSpace::new();
        let mut genome = NasGenome::from_layers(vec![LayerConfig::new(LayerType::Dense)]);

        assert!(genome.layers()[0].active);

        let mut rng = StdRng::seed_from_u64(42);
        mutate_genome(&mut genome, NasMutation::ToggleActive, &space, &mut rng);

        // Should have toggled
        assert!(!genome.layers()[0].active);
    }

    #[test]
    fn test_empty_genome_handling() {
        let genome = NasGenome::new();
        assert!(genome.is_empty());
        assert_eq!(genome.len(), 0);
        assert_eq!(genome.active_layers(), 0);
    }

    // ========================================================================
    // Additional Coverage Tests
    // ========================================================================

    #[test]
    fn test_layer_config_with_kernel_size() {
        let config = LayerConfig::new(LayerType::Conv2d).with_kernel_size(5);
        assert_eq!(config.kernel_size, Some(5));
    }

    #[test]
    fn test_layer_config_with_dropout_rate_clamping() {
        let config = LayerConfig::new(LayerType::Dropout).with_dropout_rate(1.5);
        assert_eq!(config.dropout_rate, Some(1.0)); // Clamped to max

        let config2 = LayerConfig::new(LayerType::Dropout).with_dropout_rate(-0.5);
        assert_eq!(config2.dropout_rate, Some(0.0)); // Clamped to min
    }

    #[test]
    fn test_search_space_with_filters_range() {
        let space = NasSearchSpace::new().with_filters_range(32, 256);
        assert_eq!(space.filters_range, (32, 256));
    }

    #[test]
    fn test_search_space_with_kernel_sizes() {
        let space = NasSearchSpace::new().with_kernel_sizes(vec![3, 5, 7, 9]);
        assert_eq!(space.kernel_sizes, vec![3, 5, 7, 9]);
    }

    #[test]
    fn test_search_space_with_dropout_range() {
        let space = NasSearchSpace::new().with_dropout_range(0.2, 0.6);
        assert_eq!(space.dropout_range, (0.2, 0.6));
    }

    #[test]
    fn test_search_space_dropout_range_clamping() {
        let space = NasSearchSpace::new().with_dropout_range(-0.5, 1.5);
        assert_eq!(space.dropout_range.0, 0.0);
        assert_eq!(space.dropout_range.1, 1.0);
    }

    #[test]
    fn test_search_space_with_empty_layer_types() {
        let space = NasSearchSpace::new().with_layer_types(vec![]);
        // Should not change when passed empty
        assert!(!space.layer_types.is_empty());
    }

    #[test]
    fn test_search_space_with_empty_kernel_sizes() {
        let space = NasSearchSpace::new().with_kernel_sizes(vec![]);
        // Should not change when passed empty
        assert!(!space.kernel_sizes.is_empty());
    }

    #[test]
    fn test_search_space_with_empty_activations() {
        let space = NasSearchSpace::new().with_activation_choices(&[]);
        // Should not change when passed empty
        assert!(!space.activations.is_empty());
    }

    #[test]
    fn test_nas_genome_random_with_conv2d() {
        let space = NasSearchSpace::new()
            .with_layer_types(vec![LayerType::Conv2d])
            .with_filters_range(16, 64)
            .with_kernel_sizes(vec![3, 5])
            .with_min_layers(3)
            .with_max_layers(3);

        let genome = NasGenome::random(&space, 123);
        assert_eq!(genome.len(), 3);

        for layer in genome.layers() {
            assert_eq!(layer.layer_type, LayerType::Conv2d);
            assert!(layer.units.is_some());
            assert!(layer.kernel_size.is_some());
        }
    }

    #[test]
    fn test_nas_genome_random_with_lstm() {
        let space = NasSearchSpace::new()
            .with_layer_types(vec![LayerType::Lstm])
            .with_min_layers(2)
            .with_max_layers(2);

        let genome = NasGenome::random(&space, 456);

        for layer in genome.layers() {
            assert_eq!(layer.layer_type, LayerType::Lstm);
            assert!(layer.units.is_some());
        }
    }

    #[test]
    fn test_nas_genome_random_with_batchnorm() {
        let space = NasSearchSpace::new()
            .with_layer_types(vec![LayerType::BatchNorm])
            .with_min_layers(1)
            .with_max_layers(1);

        let genome = NasGenome::random(&space, 789);
        assert_eq!(genome.layers()[0].layer_type, LayerType::BatchNorm);
    }

    #[test]
    fn test_nas_genome_encode_without_activation() {
        let space = NasSearchSpace::new();
        let genome = NasGenome::from_layers(vec![
            LayerConfig::new(LayerType::Dropout), // No activation
        ]);

        let encoded = genome.encode(&space);
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_nas_genome_decode_short_encoding() {
        let space = NasSearchSpace::new();
        // Very short encoding - should handle gracefully
        let encoded = vec![0.5, 0.5, 0.5, 0.5, 1.0];

        let decoded = NasGenome::decode(&encoded, &space, 2);
        // Only one layer can be decoded from 5 values (stride=5)
        assert_eq!(decoded.len(), 1);
    }

    #[test]
    fn test_nas_genome_default() {
        let genome = NasGenome::default();
        assert!(genome.is_empty());
    }

    #[test]
    fn test_mutation_change_type() {
        let space = NasSearchSpace::new().with_layer_types(vec![
            LayerType::Dense,
            LayerType::Conv2d,
            LayerType::Dropout,
        ]);
        let mut genome = NasGenome::from_layers(vec![LayerConfig::new(LayerType::Dense)]);

        let mut rng = StdRng::seed_from_u64(42);
        mutate_genome(&mut genome, NasMutation::ChangeType, &space, &mut rng);

        // Type might have changed
        assert!(!genome.is_empty());
    }

    #[test]
    fn test_mutation_modify_params() {
        let space = NasSearchSpace::new().with_units_range(32, 256);
        let mut genome =
            NasGenome::from_layers(vec![LayerConfig::new(LayerType::Dense).with_units(128)]);

        let mut rng = StdRng::seed_from_u64(42);
        mutate_genome(&mut genome, NasMutation::ModifyParams, &space, &mut rng);

        // Units might have changed
        let units = genome.layers()[0].units.unwrap();
        assert!(units >= 32 && units <= 256);
    }

    #[test]
    fn test_mutation_modify_params_without_units() {
        let space = NasSearchSpace::new();
        let mut genome = NasGenome::from_layers(vec![
            LayerConfig::new(LayerType::Dropout), // No units
        ]);

        let mut rng = StdRng::seed_from_u64(42);
        // Should not panic
        mutate_genome(&mut genome, NasMutation::ModifyParams, &space, &mut rng);
    }

    #[test]
    fn test_mutation_change_type_empty_genome() {
        let space = NasSearchSpace::new();
        let mut genome = NasGenome::new();

        let mut rng = StdRng::seed_from_u64(42);
        // Should not panic on empty genome
        mutate_genome(&mut genome, NasMutation::ChangeType, &space, &mut rng);
        assert!(genome.is_empty());
    }

    #[test]
    fn test_mutation_toggle_empty_genome() {
        let space = NasSearchSpace::new();
        let mut genome = NasGenome::new();

        let mut rng = StdRng::seed_from_u64(42);
        mutate_genome(&mut genome, NasMutation::ToggleActive, &space, &mut rng);
        assert!(genome.is_empty());
    }

    #[test]
    fn test_mutation_modify_params_empty_genome() {
        let space = NasSearchSpace::new();
        let mut genome = NasGenome::new();

        let mut rng = StdRng::seed_from_u64(42);
        mutate_genome(&mut genome, NasMutation::ModifyParams, &space, &mut rng);
        assert!(genome.is_empty());
    }

    #[test]
    fn test_crossover_with_empty_parent1() {
        let parent1 = NasGenome::new();
        let parent2 =
            NasGenome::from_layers(vec![LayerConfig::new(LayerType::Dense).with_units(64)]);

        let mut rng = StdRng::seed_from_u64(42);
        let child = crossover_genomes(&parent1, &parent2, &mut rng);

        assert!(!child.is_empty());
    }

    #[test]
    fn test_crossover_with_empty_parent2() {
        let parent1 =
            NasGenome::from_layers(vec![LayerConfig::new(LayerType::Dense).with_units(64)]);
        let parent2 = NasGenome::new();

        let mut rng = StdRng::seed_from_u64(42);
        let child = crossover_genomes(&parent1, &parent2, &mut rng);

        assert!(!child.is_empty());
    }

    #[test]
    fn test_layer_type_debug_clone_hash() {
        use std::collections::HashSet;

        let lt = LayerType::Dense;
        let cloned = lt;
        assert_eq!(lt, cloned);

        let debug = format!("{:?}", lt);
        assert!(debug.contains("Dense"));

        let mut set = HashSet::new();
        set.insert(LayerType::Dense);
        set.insert(LayerType::Conv2d);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_layer_config_debug_clone_eq() {
        let config = LayerConfig::new(LayerType::Dense).with_units(64);
        let cloned = config.clone();
        assert_eq!(config, cloned);

        let debug = format!("{:?}", config);
        assert!(debug.contains("Dense"));
    }

    #[test]
    fn test_search_space_debug_clone() {
        let space = NasSearchSpace::new();
        let cloned = space.clone();
        assert_eq!(space.max_layers, cloned.max_layers);

        let debug = format!("{:?}", space);
        assert!(debug.contains("max_layers"));
    }

    #[test]
    fn test_nas_genome_debug_clone() {
        let genome = NasGenome::from_layers(vec![LayerConfig::new(LayerType::Dense)]);
        let cloned = genome.clone();
        assert_eq!(genome.len(), cloned.len());

        let debug = format!("{:?}", genome);
        assert!(debug.contains("layers"));
    }

    #[test]
    fn test_nas_mutation_debug_clone() {
        let mutation = NasMutation::AddLayer;
        let cloned = mutation;
        let debug = format!("{:?}", cloned);
        assert!(debug.contains("AddLayer"));

        // Test all variants
        let mutations = [
            NasMutation::AddLayer,
            NasMutation::RemoveLayer,
            NasMutation::ChangeType,
            NasMutation::ModifyParams,
            NasMutation::ToggleActive,
        ];
        for m in &mutations {
            let d = format!("{:?}", m);
            assert!(!d.is_empty());
        }
    }

    #[test]
    fn test_architecture_complexity_all_layer_types() {
        let layers = vec![
            LayerConfig::new(LayerType::Dense).with_units(64),
            LayerConfig::new(LayerType::Conv2d).with_units(64),
            LayerConfig::new(LayerType::MaxPool2d),
            LayerConfig::new(LayerType::AvgPool2d),
            LayerConfig::new(LayerType::BatchNorm),
            LayerConfig::new(LayerType::Dropout),
            LayerConfig::new(LayerType::Skip),
            LayerConfig::new(LayerType::Lstm).with_units(64),
            LayerConfig::new(LayerType::Attention).with_units(64),
        ];
        let genome = NasGenome::from_layers(layers);
        let complexity = architecture_complexity(&genome);
        assert!(complexity > 0.0);
    }

    #[test]
    fn test_layer_config_active_default() {
        let config = LayerConfig::new(LayerType::Dense);
        assert!(config.active);
    }

    #[test]
    fn test_encode_layer_type_not_found() {
        // Create space with only one layer type
        let space = NasSearchSpace::new().with_layer_types(vec![LayerType::Dense]);

        // Create genome with a different layer type
        let genome = NasGenome::from_layers(vec![LayerConfig::new(LayerType::Conv2d)]);

        // Should not panic, returns 0 for unknown type
        let encoded = genome.encode(&space);
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_encode_activation_not_found() {
        let space = NasSearchSpace::new().with_activation_choices(&["relu", "tanh"]);

        let genome = NasGenome::from_layers(vec![
            LayerConfig::new(LayerType::Dense).with_activation("unknown_activation")
        ]);

        // Should not panic, returns 0 for unknown activation
        let encoded = genome.encode(&space);
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_units_range_with_same_min_max() {
        let space = NasSearchSpace::new().with_units_range(64, 64);
        // min+1 should make max at least 65
        assert!(space.units_range.1 >= space.units_range.0);
    }

    #[test]
    fn test_filters_range_with_same_min_max() {
        let space = NasSearchSpace::new().with_filters_range(64, 64);
        // min+1 should make max at least 65
        assert!(space.filters_range.1 >= space.filters_range.0);
    }
}
