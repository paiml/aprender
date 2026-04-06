//! Layout caching for memoization.

use presentar_core::Size;
use std::collections::HashMap;

/// Cache key combining constraints hash and widget identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Widget identity hash
    pub widget_id: u64,
    /// Constraints hash
    pub constraints_hash: u64,
}

/// Cached layout result.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CacheEntry {
    /// Computed size
    pub size: Size,
    /// Frame when this entry was last used
    pub last_used_frame: u64,
}

/// Layout cache for memoizing measure results.
#[derive(Debug, Default)]
pub struct LayoutCache {
    entries: HashMap<CacheKey, CacheEntry>,
    current_frame: u64,
    hits: usize,
    misses: usize,
}

impl LayoutCache {
    /// Create a new empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a cached size.
    #[must_use]
    pub fn get(&mut self, key: CacheKey) -> Option<Size> {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used_frame = self.current_frame;
            self.hits += 1;
            Some(entry.size)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a computed size into the cache.
    pub fn insert(&mut self, key: CacheKey, size: Size) {
        self.entries.insert(
            key,
            CacheEntry {
                size,
                last_used_frame: self.current_frame,
            },
        );
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Get the number of cache hits.
    #[must_use]
    pub const fn hits(&self) -> usize {
        self.hits
    }

    /// Get the number of cache misses.
    #[must_use]
    pub const fn misses(&self) -> usize {
        self.misses
    }

    /// Advance to the next frame and evict stale entries.
    pub fn advance_frame(&mut self) {
        self.current_frame += 1;

        // Evict entries not used in the last 2 frames
        let threshold = self.current_frame.saturating_sub(2);
        self.entries
            .retain(|_, entry| entry.last_used_frame >= threshold);
    }

    /// Get the number of cached entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_new() {
        let cache = LayoutCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_insert_get() {
        let mut cache = LayoutCache::new();
        let key = CacheKey {
            widget_id: 1,
            constraints_hash: 100,
        };
        let size = Size::new(50.0, 50.0);

        cache.insert(key, size);
        assert_eq!(cache.get(key), Some(size));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = LayoutCache::new();
        let key = CacheKey {
            widget_id: 1,
            constraints_hash: 100,
        };

        assert_eq!(cache.get(key), None);
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = LayoutCache::new();
        let key = CacheKey {
            widget_id: 1,
            constraints_hash: 100,
        };

        cache.insert(key, Size::new(10.0, 10.0));
        assert!(!cache.is_empty());

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = LayoutCache::new();
        let key = CacheKey {
            widget_id: 1,
            constraints_hash: 100,
        };

        cache.insert(key, Size::new(10.0, 10.0));

        // Advance 3 frames without using the entry
        cache.advance_frame();
        cache.advance_frame();
        cache.advance_frame();

        // Entry should be evicted
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_not_evicted_when_used() {
        let mut cache = LayoutCache::new();
        let key = CacheKey {
            widget_id: 1,
            constraints_hash: 100,
        };

        cache.insert(key, Size::new(10.0, 10.0));

        // Advance frames but keep using the entry
        for _ in 0..5 {
            cache.advance_frame();
            let _ = cache.get(key); // Touch the entry
        }

        assert!(!cache.is_empty());
    }

    #[test]
    fn test_cache_hits_and_misses() {
        let mut cache = LayoutCache::new();
        let key = CacheKey {
            widget_id: 1,
            constraints_hash: 100,
        };

        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);

        // Miss
        let _ = cache.get(key);
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 1);

        // Insert and hit
        cache.insert(key, Size::new(10.0, 10.0));
        let _ = cache.get(key);
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);

        // Another hit
        let _ = cache.get(key);
        assert_eq!(cache.hits(), 2);
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn test_cache_clear_resets_stats() {
        let mut cache = LayoutCache::new();
        let key = CacheKey {
            widget_id: 1,
            constraints_hash: 100,
        };

        cache.insert(key, Size::new(10.0, 10.0));
        let _ = cache.get(key);
        let _ = cache.get(CacheKey {
            widget_id: 2,
            constraints_hash: 200,
        });

        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);

        cache.clear();
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }

    // =========================================================================
    // CacheKey Tests
    // =========================================================================

    #[test]
    fn test_cache_key_equality() {
        let key1 = CacheKey {
            widget_id: 1,
            constraints_hash: 100,
        };
        let key2 = CacheKey {
            widget_id: 1,
            constraints_hash: 100,
        };
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_key_inequality_widget_id() {
        let key1 = CacheKey {
            widget_id: 1,
            constraints_hash: 100,
        };
        let key2 = CacheKey {
            widget_id: 2,
            constraints_hash: 100,
        };
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_cache_key_inequality_constraints() {
        let key1 = CacheKey {
            widget_id: 1,
            constraints_hash: 100,
        };
        let key2 = CacheKey {
            widget_id: 1,
            constraints_hash: 200,
        };
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_cache_key_clone() {
        let key = CacheKey {
            widget_id: 42,
            constraints_hash: 999,
        };
        let cloned = key;
        assert_eq!(key, cloned);
    }

    #[test]
    fn test_cache_key_debug() {
        let key = CacheKey {
            widget_id: 1,
            constraints_hash: 100,
        };
        let debug = format!("{:?}", key);
        assert!(debug.contains("widget_id"));
        assert!(debug.contains("constraints_hash"));
    }

    // =========================================================================
    // Multiple Entries Tests
    // =========================================================================

    #[test]
    fn test_cache_multiple_entries() {
        let mut cache = LayoutCache::new();

        for i in 0..10 {
            let key = CacheKey {
                widget_id: i,
                constraints_hash: i * 100,
            };
            cache.insert(key, Size::new(i as f32, i as f32));
        }

        assert_eq!(cache.len(), 10);

        // Verify each entry
        for i in 0..10 {
            let key = CacheKey {
                widget_id: i,
                constraints_hash: i * 100,
            };
            assert_eq!(cache.get(key), Some(Size::new(i as f32, i as f32)));
        }
    }

    #[test]
    fn test_cache_overwrite_entry() {
        let mut cache = LayoutCache::new();
        let key = CacheKey {
            widget_id: 1,
            constraints_hash: 100,
        };

        cache.insert(key, Size::new(10.0, 10.0));
        assert_eq!(cache.get(key), Some(Size::new(10.0, 10.0)));

        // Overwrite with new size
        cache.insert(key, Size::new(20.0, 20.0));
        assert_eq!(cache.get(key), Some(Size::new(20.0, 20.0)));
        assert_eq!(cache.len(), 1);
    }

    // =========================================================================
    // Eviction Edge Cases
    // =========================================================================

    #[test]
    fn test_cache_eviction_threshold() {
        let mut cache = LayoutCache::new();

        let key1 = CacheKey {
            widget_id: 1,
            constraints_hash: 100,
        };
        let key2 = CacheKey {
            widget_id: 2,
            constraints_hash: 200,
        };

        cache.insert(key1, Size::new(10.0, 10.0));
        cache.advance_frame();
        cache.insert(key2, Size::new(20.0, 20.0));
        cache.advance_frame();

        // Both should still be present (threshold is 2 frames)
        assert_eq!(cache.len(), 2);

        // One more frame without touching key1
        let _ = cache.get(key2);
        cache.advance_frame();

        // key1 should be evicted (not used in last 2 frames)
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(key2), Some(Size::new(20.0, 20.0)));
    }

    #[test]
    fn test_cache_eviction_empty_cache() {
        let mut cache = LayoutCache::new();

        // Advancing frames on empty cache should not panic
        for _ in 0..10 {
            cache.advance_frame();
        }

        assert!(cache.is_empty());
    }

    // =========================================================================
    // Default Implementation
    // =========================================================================

    #[test]
    fn test_cache_default() {
        let cache = LayoutCache::default();
        assert!(cache.is_empty());
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }

    // =========================================================================
    // Debug Format
    // =========================================================================

    #[test]
    fn test_cache_debug() {
        let cache = LayoutCache::new();
        let debug = format!("{:?}", cache);
        assert!(debug.contains("LayoutCache"));
    }

    // =========================================================================
    // Size Values
    // =========================================================================

    #[test]
    fn test_cache_with_zero_size() {
        let mut cache = LayoutCache::new();
        let key = CacheKey {
            widget_id: 1,
            constraints_hash: 100,
        };

        cache.insert(key, Size::new(0.0, 0.0));
        assert_eq!(cache.get(key), Some(Size::new(0.0, 0.0)));
    }

    #[test]
    fn test_cache_with_large_size() {
        let mut cache = LayoutCache::new();
        let key = CacheKey {
            widget_id: 1,
            constraints_hash: 100,
        };

        cache.insert(key, Size::new(10000.0, 10000.0));
        assert_eq!(cache.get(key), Some(Size::new(10000.0, 10000.0)));
    }

    #[test]
    fn test_cache_with_fractional_size() {
        let mut cache = LayoutCache::new();
        let key = CacheKey {
            widget_id: 1,
            constraints_hash: 100,
        };

        cache.insert(key, Size::new(10.5, 20.75));
        assert_eq!(cache.get(key), Some(Size::new(10.5, 20.75)));
    }

    // =========================================================================
    // Hash Collision Resistance
    // =========================================================================

    #[test]
    fn test_cache_different_widget_same_constraints() {
        let mut cache = LayoutCache::new();

        let key1 = CacheKey {
            widget_id: 1,
            constraints_hash: 100,
        };
        let key2 = CacheKey {
            widget_id: 2,
            constraints_hash: 100,
        };

        cache.insert(key1, Size::new(10.0, 10.0));
        cache.insert(key2, Size::new(20.0, 20.0));

        assert_eq!(cache.get(key1), Some(Size::new(10.0, 10.0)));
        assert_eq!(cache.get(key2), Some(Size::new(20.0, 20.0)));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_same_widget_different_constraints() {
        let mut cache = LayoutCache::new();

        let key1 = CacheKey {
            widget_id: 1,
            constraints_hash: 100,
        };
        let key2 = CacheKey {
            widget_id: 1,
            constraints_hash: 200,
        };

        cache.insert(key1, Size::new(10.0, 10.0));
        cache.insert(key2, Size::new(20.0, 20.0));

        assert_eq!(cache.get(key1), Some(Size::new(10.0, 10.0)));
        assert_eq!(cache.get(key2), Some(Size::new(20.0, 20.0)));
        assert_eq!(cache.len(), 2);
    }

    // =========================================================================
    // Frame Counter
    // =========================================================================

    #[test]
    fn test_cache_frame_counter_overflow() {
        let mut cache = LayoutCache::new();
        let key = CacheKey {
            widget_id: 1,
            constraints_hash: 100,
        };

        cache.insert(key, Size::new(10.0, 10.0));

        // Advance many frames while keeping entry fresh
        for _ in 0..100 {
            let _ = cache.get(key);
            cache.advance_frame();
        }

        // Entry should still exist
        assert_eq!(cache.get(key), Some(Size::new(10.0, 10.0)));
    }
}
