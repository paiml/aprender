//! Dequantized weight cache for GPU GEMM operations
//!
//! Stores pre-dequantized f32 weights for GPU GEMM to avoid
//! repeated dequantization on every forward pass.

/// Dequantized FFN weights for a single transformer layer
///
/// Stores pre-dequantized f32 weights for GPU GEMM operations.
/// Cache these to avoid repeated dequantization on every forward pass.
#[derive(Clone)]
pub struct DequantizedFFNWeights {
    /// Up projection weights [hidden_dim, intermediate_dim]
    pub up: Vec<f32>,
    /// Down projection weights [intermediate_dim, hidden_dim]
    pub down: Vec<f32>,
    /// Optional up bias [intermediate_dim]
    pub up_bias: Option<Vec<f32>>,
    /// Optional down bias [hidden_dim]
    pub down_bias: Option<Vec<f32>>,
}

/// Cache for dequantized FFN weights (PARITY-019)
///
/// Uses RwLock for concurrent read access during batch inference.
/// Weights are dequantized once during warmup and reused for GPU GEMM.
///
/// # Performance Impact
/// - Eliminates per-forward dequantization overhead
/// - Enables GPU GEMM with f32 weights
/// - Memory tradeoff: ~6.4 GB for phi-2 32 layers
///
/// # Thread Safety
/// - RwLock allows multiple concurrent readers during inference
/// - Single writer during warmup phase
#[cfg(feature = "gpu")]
pub struct DequantizedWeightCache {
    /// Per-layer dequantized weights
    layers: std::sync::RwLock<std::collections::HashMap<usize, DequantizedFFNWeights>>,
    /// Hidden dimension for validation
    hidden_dim: usize,
    /// Intermediate FFN dimension
    intermediate_dim: usize,
    /// Number of layers to cache
    num_layers: usize,
}

#[cfg(feature = "gpu")]
impl DequantizedWeightCache {
    /// Create a new weight cache with specified dimensions
    ///
    /// # Arguments
    /// * `hidden_dim` - Model hidden dimension (e.g., 2560 for phi-2)
    /// * `intermediate_dim` - FFN intermediate dimension (e.g., 10240 for phi-2)
    /// * `num_layers` - Number of transformer layers to cache
    #[must_use]
    pub fn new(hidden_dim: usize, intermediate_dim: usize, num_layers: usize) -> Self {
        Self {
            layers: std::sync::RwLock::new(std::collections::HashMap::with_capacity(num_layers)),
            hidden_dim,
            intermediate_dim,
            num_layers,
        }
    }

    /// Pre-warmup all layers with dequantized weights
    ///
    /// Call this once at startup to avoid dequantization during inference.
    /// The closure receives layer index and returns (up_weights, down_weights).
    ///
    /// # Arguments
    /// * `dequant_fn` - Closure that dequantizes weights for a given layer index
    ///
    /// # Panics
    /// Panics if the RwLock is poisoned
    pub fn warmup<F>(&self, dequant_fn: F)
    where
        F: Fn(usize) -> (Vec<f32>, Vec<f32>),
    {
        let mut cache = self.layers.write().expect("Cache lock poisoned");
        for layer_idx in 0..self.num_layers {
            cache.entry(layer_idx).or_insert_with(|| {
                let (up, down) = dequant_fn(layer_idx);
                DequantizedFFNWeights {
                    up,
                    down,
                    up_bias: None,
                    down_bias: None,
                }
            });
        }
    }

    /// Warmup with biases
    ///
    /// Same as `warmup` but also caches bias vectors.
    pub fn warmup_with_bias<F>(&self, dequant_fn: F)
    where
        F: Fn(usize) -> (Vec<f32>, Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>),
    {
        let mut cache = self.layers.write().expect("Cache lock poisoned");
        for layer_idx in 0..self.num_layers {
            cache.entry(layer_idx).or_insert_with(|| {
                let (up, down, up_bias, down_bias) = dequant_fn(layer_idx);
                DequantizedFFNWeights {
                    up,
                    down,
                    up_bias,
                    down_bias,
                }
            });
        }
    }

    /// Get cached weights for a layer (read-only access)
    ///
    /// Returns None if the layer hasn't been warmed up.
    /// Uses read lock for concurrent access during batch inference.
    pub fn get(&self, layer_idx: usize) -> Option<DequantizedFFNWeights> {
        let cache = self.layers.read().expect("Cache lock poisoned");
        cache.get(&layer_idx).cloned()
    }

    /// Check if a layer is cached
    pub fn is_cached(&self, layer_idx: usize) -> bool {
        let cache = self.layers.read().expect("Cache lock poisoned");
        cache.contains_key(&layer_idx)
    }

    /// Get number of cached layers
    pub fn cached_count(&self) -> usize {
        let cache = self.layers.read().expect("Cache lock poisoned");
        cache.len()
    }

    /// Get total memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        // Each layer: up + down weights
        // up: hidden_dim × intermediate_dim × 4 bytes
        // down: intermediate_dim × hidden_dim × 4 bytes
        let per_layer = 2 * self.hidden_dim * self.intermediate_dim * 4;
        self.cached_count() * per_layer
    }

    /// Get model dimensions
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.hidden_dim, self.intermediate_dim, self.num_layers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // DequantizedFFNWeights tests
    // ============================================================================

    #[test]
    fn test_dequantized_ffn_weights_basic() {
        let weights = DequantizedFFNWeights {
            up: vec![1.0, 2.0, 3.0, 4.0],
            down: vec![5.0, 6.0, 7.0, 8.0],
            up_bias: None,
            down_bias: None,
        };
        assert_eq!(weights.up.len(), 4);
        assert_eq!(weights.down.len(), 4);
        assert!(weights.up_bias.is_none());
        assert!(weights.down_bias.is_none());
    }

    #[test]
    fn test_dequantized_ffn_weights_with_bias() {
        let weights = DequantizedFFNWeights {
            up: vec![1.0, 2.0],
            down: vec![3.0, 4.0],
            up_bias: Some(vec![0.1, 0.2]),
            down_bias: Some(vec![0.3, 0.4]),
        };
        assert!(weights.up_bias.is_some());
        assert!(weights.down_bias.is_some());
        assert_eq!(weights.up_bias.as_ref().expect("as_ref").len(), 2);
    }

    #[test]
    fn test_dequantized_ffn_weights_clone() {
        let original = DequantizedFFNWeights {
            up: vec![1.0, 2.0],
            down: vec![3.0, 4.0],
            up_bias: Some(vec![0.5]),
            down_bias: None,
        };
        let cloned = original.clone();
        assert_eq!(cloned.up, original.up);
        assert_eq!(cloned.down, original.down);
        assert_eq!(cloned.up_bias, original.up_bias);
    }

    // ============================================================================
    // DequantizedWeightCache tests (require gpu feature)
    // ============================================================================

    #[test]
    #[cfg(feature = "gpu")]
    fn test_weight_cache_new() {
        let cache = DequantizedWeightCache::new(256, 1024, 4);
        let (h, i, n) = cache.dimensions();
        assert_eq!(h, 256);
        assert_eq!(i, 1024);
        assert_eq!(n, 4);
        assert_eq!(cache.cached_count(), 0);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_weight_cache_warmup() {
        let cache = DequantizedWeightCache::new(64, 256, 2);

        cache.warmup(|layer_idx| {
            let up = vec![(layer_idx as f32) * 0.1; 64 * 256];
            let down = vec![(layer_idx as f32) * 0.2; 256 * 64];
            (up, down)
        });

        assert_eq!(cache.cached_count(), 2);
        assert!(cache.is_cached(0));
        assert!(cache.is_cached(1));
        assert!(!cache.is_cached(2));
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_weight_cache_warmup_with_bias() {
        let cache = DequantizedWeightCache::new(32, 128, 1);

        cache.warmup_with_bias(|_| {
            let up = vec![1.0; 32 * 128];
            let down = vec![2.0; 128 * 32];
            let up_bias = Some(vec![0.1; 128]);
            let down_bias = Some(vec![0.2; 32]);
            (up, down, up_bias, down_bias)
        });

        let weights = cache.get(0).expect("weights");
        assert!(weights.up_bias.is_some());
        assert!(weights.down_bias.is_some());
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_weight_cache_get() {
        let cache = DequantizedWeightCache::new(16, 64, 2);

        cache.warmup(|idx| {
            let up = vec![idx as f32; 16 * 64];
            let down = vec![(idx + 10) as f32; 64 * 16];
            (up, down)
        });

        let w0 = cache.get(0).expect("w0");
        assert!((w0.up[0] - 0.0).abs() < f32::EPSILON);
        assert!((w0.down[0] - 10.0).abs() < f32::EPSILON);

        let w1 = cache.get(1).expect("w1");
        assert!((w1.up[0] - 1.0).abs() < f32::EPSILON);

        // Non-existent layer
        assert!(cache.get(99).is_none());
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_weight_cache_is_cached() {
        let cache = DequantizedWeightCache::new(8, 32, 3);

        assert!(!cache.is_cached(0));
        assert!(!cache.is_cached(1));

        cache.warmup(|idx| (vec![idx as f32; 256], vec![idx as f32; 256]));

        assert!(cache.is_cached(0));
        assert!(cache.is_cached(1));
        assert!(cache.is_cached(2));
        assert!(!cache.is_cached(3));
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_weight_cache_memory_bytes() {
        let cache = DequantizedWeightCache::new(64, 256, 2);
        assert_eq!(cache.memory_bytes(), 0); // Nothing cached yet

        cache.warmup(|_| (vec![0.0; 64 * 256], vec![0.0; 256 * 64]));

        // 2 layers × 2 weights × 64×256 × 4 bytes
        let expected = 2 * 2 * 64 * 256 * 4;
        assert_eq!(cache.memory_bytes(), expected);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_weight_cache_dimensions() {
        let cache = DequantizedWeightCache::new(512, 2048, 12);
        let (h, i, n) = cache.dimensions();
        assert_eq!(h, 512);
        assert_eq!(i, 2048);
        assert_eq!(n, 12);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_weight_cache_idempotent_warmup() {
        let cache = DequantizedWeightCache::new(8, 16, 1);

        // Warmup with initial values
        cache.warmup(|_| (vec![1.0; 128], vec![2.0; 128]));

        let initial_up = cache.get(0).expect("initial_up").up[0];
        assert!((initial_up - 1.0).abs() < f32::EPSILON);

        // Warmup again should not overwrite existing entries (or_insert_with)
        cache.warmup(|_| (vec![999.0; 128], vec![888.0; 128]));

        let after_up = cache.get(0).expect("after_up").up[0];
        // Should still be 1.0, not 999.0
        assert!((after_up - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_weight_cache_empty() {
        let cache = DequantizedWeightCache::new(4, 8, 0);
        assert_eq!(cache.cached_count(), 0);
        assert!(cache.get(0).is_none());
        assert!(!cache.is_cached(0));
    }
}
