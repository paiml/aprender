//! Teacher cache for distillation

use ndarray::Array2;
use std::collections::HashMap;

/// Cached teacher outputs for distillation
#[derive(Debug, Clone)]
pub struct TeacherCache {
    /// Cached logits by example index
    logits: HashMap<usize, Array2<f32>>,
    /// Cached hidden states by example index
    hidden_states: HashMap<usize, Vec<Array2<f32>>>,
    /// Cache hit count
    hits: usize,
    /// Cache miss count
    misses: usize,
}

impl Default for TeacherCache {
    fn default() -> Self {
        Self::new()
    }
}

impl TeacherCache {
    /// Create new empty cache
    #[must_use]
    pub fn new() -> Self {
        Self { logits: HashMap::new(), hidden_states: HashMap::new(), hits: 0, misses: 0 }
    }

    /// Get cached logits
    pub fn get_logits(&mut self, index: usize) -> Option<&Array2<f32>> {
        if self.logits.contains_key(&index) {
            self.hits += 1;
            self.logits.get(&index)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Cache logits
    pub fn cache_logits(&mut self, index: usize, logits: Array2<f32>) {
        self.logits.insert(index, logits);
    }

    /// Get cached hidden states
    pub fn get_hidden_states(&mut self, index: usize) -> Option<&Vec<Array2<f32>>> {
        if self.hidden_states.contains_key(&index) {
            self.hits += 1;
            self.hidden_states.get(&index)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Cache hidden states
    pub fn cache_hidden_states(&mut self, index: usize, states: Vec<Array2<f32>>) {
        self.hidden_states.insert(index, states);
    }

    /// Get cache statistics
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            logits_cached: self.logits.len(),
            hidden_states_cached: self.hidden_states.len(),
        }
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.logits.clear();
        self.hidden_states.clear();
        self.hits = 0;
        self.misses = 0;
    }
}

/// Cache statistics
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Number of cached logits entries
    pub logits_cached: usize,
    /// Number of cached hidden state entries
    pub hidden_states_cached: usize,
}

impl CacheStats {
    /// Get hit rate
    #[must_use]
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total > 0 {
            self.hits as f32 / total as f32
        } else {
            0.0
        }
    }
}
