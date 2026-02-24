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

        let n_layers = rng.random_range(space.min_layers..=space.max_layers);
        let mut layers = Vec::with_capacity(n_layers);

        for _ in 0..n_layers {
            let layer_type = space.layer_types[rng.random_range(0..space.layer_types.len())];
            let mut config = LayerConfig::new(layer_type);

            match layer_type {
                LayerType::Dense => {
                    config.units =
                        Some(rng.random_range(space.units_range.0..=space.units_range.1));
                    config.activation = Some(
                        space.activations[rng.random_range(0..space.activations.len())].clone(),
                    );
                }
                LayerType::Conv2d => {
                    config.units =
                        Some(rng.random_range(space.filters_range.0..=space.filters_range.1));
                    config.kernel_size =
                        Some(space.kernel_sizes[rng.random_range(0..space.kernel_sizes.len())]);
                    config.activation = Some(
                        space.activations[rng.random_range(0..space.activations.len())].clone(),
                    );
                }
                LayerType::Dropout => {
                    config.dropout_rate =
                        Some(rng.random_range(space.dropout_range.0..=space.dropout_range.1));
                }
                LayerType::Lstm => {
                    config.units =
                        Some(rng.random_range(space.units_range.0..=space.units_range.1));
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

include!("nas_mutation.rs");
include!("nas_tests.rs");
