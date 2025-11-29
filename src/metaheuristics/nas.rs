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
//!     .with_activation_choices(vec!["relu", "tanh", "sigmoid"]);
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
    pub fn with_units(mut self, units: usize) -> Self {
        self.units = Some(units);
        self
    }

    /// Set kernel size.
    pub fn with_kernel_size(mut self, size: usize) -> Self {
        self.kernel_size = Some(size);
        self
    }

    /// Set activation function.
    pub fn with_activation(mut self, activation: &str) -> Self {
        self.activation = Some(activation.to_string());
        self
    }

    /// Set dropout rate.
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
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum layers.
    pub fn with_max_layers(mut self, max: usize) -> Self {
        self.max_layers = max.max(1);
        self
    }

    /// Set minimum layers.
    pub fn with_min_layers(mut self, min: usize) -> Self {
        self.min_layers = min.max(1);
        self
    }

    /// Set allowed layer types.
    pub fn with_layer_types(mut self, types: Vec<LayerType>) -> Self {
        if !types.is_empty() {
            self.layer_types = types;
        }
        self
    }

    /// Set units range for Dense layers.
    pub fn with_units_range(mut self, min: usize, max: usize) -> Self {
        self.units_range = (min.max(1), max.max(min + 1));
        self
    }

    /// Set filters range for Conv layers.
    pub fn with_filters_range(mut self, min: usize, max: usize) -> Self {
        self.filters_range = (min.max(1), max.max(min + 1));
        self
    }

    /// Set allowed kernel sizes.
    pub fn with_kernel_sizes(mut self, sizes: Vec<usize>) -> Self {
        if !sizes.is_empty() {
            self.kernel_sizes = sizes;
        }
        self
    }

    /// Set allowed activation functions.
    pub fn with_activation_choices(mut self, activations: &[&str]) -> Self {
        if !activations.is_empty() {
            self.activations = activations.iter().map(|s| (*s).to_string()).collect();
        }
        self
    }

    /// Set dropout rate range.
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
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            fitness: None,
        }
    }

    /// Create genome from layers.
    pub fn from_layers(layers: Vec<LayerConfig>) -> Self {
        Self {
            layers,
            fitness: None,
        }
    }

    /// Generate a random architecture within the search space.
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
    pub fn layers(&self) -> &[LayerConfig] {
        &self.layers
    }

    /// Get mutable layers.
    pub fn layers_mut(&mut self) -> &mut Vec<LayerConfig> {
        &mut self.layers
    }

    /// Get number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if genome is empty.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Get fitness score.
    pub fn fitness(&self) -> Option<f64> {
        self.fitness
    }

    /// Set fitness score.
    pub fn set_fitness(&mut self, fitness: f64) {
        self.fitness = Some(fitness);
    }

    /// Count active layers.
    pub fn active_layers(&self) -> usize {
        self.layers.iter().filter(|l| l.active).count()
    }

    /// Encode genome to continuous vector for optimization.
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
        NasMutation::AddLayer => {
            if genome.len() < space.max_layers {
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
        }
        NasMutation::RemoveLayer => {
            if genome.len() > space.min_layers {
                let idx = rng.gen_range(0..genome.len());
                genome.layers_mut().remove(idx);
            }
        }
        NasMutation::ChangeType => {
            if !genome.is_empty() && !space.layer_types.is_empty() {
                let idx = rng.gen_range(0..genome.len());
                let new_type = space.layer_types[rng.gen_range(0..space.layer_types.len())];
                genome.layers_mut()[idx].layer_type = new_type;
            }
        }
        NasMutation::ModifyParams => {
            if !genome.is_empty() {
                let idx = rng.gen_range(0..genome.len());
                let layer = &mut genome.layers_mut()[idx];

                // Modify units with some probability
                if let Some(units) = layer.units {
                    let delta = rng.gen_range(-32i64..=32);
                    let new_units = (units as i64 + delta)
                        .clamp(space.units_range.0 as i64, space.units_range.1 as i64);
                    layer.units = Some(new_units as usize);
                }

                // Maybe change activation
                if rng.gen_bool(0.3) && !space.activations.is_empty() {
                    layer.activation =
                        Some(space.activations[rng.gen_range(0..space.activations.len())].clone());
                }
            }
        }
        NasMutation::ToggleActive => {
            if !genome.is_empty() {
                let idx = rng.gen_range(0..genome.len());
                let layer = &mut genome.layers_mut()[idx];
                layer.active = !layer.active;
            }
        }
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
}
