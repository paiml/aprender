//! Query Plan Cache
//!
//! LRU cache for query plans to speed up repeated queries.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Query plan cache
pub struct QueryPlanCache {
    /// Cache mapping query hash to plan
    cache: HashMap<u64, CachedPlan>,
    /// LRU order tracking (front = most recently used)
    order: VecDeque<u64>,
    /// Maximum capacity
    capacity: usize,
    /// Cache hit counter
    hits: u64,
    /// Cache miss counter
    misses: u64,
    /// TTL for cache entries
    ttl: Duration,
}

/// A cached query plan
#[derive(Debug, Clone)]
pub struct CachedPlan {
    /// Tokenized query terms
    pub terms: Vec<String>,
    /// Term weights
    pub term_weights: Vec<f32>,
    /// Candidate document IDs (pre-filtered)
    pub candidate_docs: Vec<u32>,
    /// Component boosts detected
    pub component_boosts: Vec<(String, f32)>,
    /// When this plan was created
    pub created_at: Instant,
}

impl QueryPlanCache {
    /// Create a new cache with given capacity
    pub fn new(capacity: usize) -> Self {
        let cap = if capacity == 0 { 1000 } else { capacity };
        Self {
            cache: HashMap::with_capacity(cap),
            order: VecDeque::with_capacity(cap),
            capacity: cap,
            hits: 0,
            misses: 0,
            ttl: Duration::from_secs(300), // 5 minutes default
        }
    }

    /// Create with custom TTL
    pub fn with_ttl(capacity: usize, ttl: Duration) -> Self {
        let mut cache = Self::new(capacity);
        cache.ttl = ttl;
        cache
    }

    /// Hash a query string
    fn hash_query(&self, query: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        query.to_lowercase().hash(&mut hasher);
        hasher.finish()
    }

    /// Move a key to the front of the LRU order
    fn touch(&mut self, hash: u64) {
        // Remove from current position if exists
        self.order.retain(|&h| h != hash);
        // Add to front
        self.order.push_front(hash);
    }

    /// Evict oldest entries if over capacity
    fn evict_if_needed(&mut self) {
        while self.order.len() > self.capacity {
            if let Some(old_hash) = self.order.pop_back() {
                self.cache.remove(&old_hash);
            }
        }
    }

    /// Get a cached plan
    pub fn get(&mut self, query: &str) -> Option<&CachedPlan> {
        let hash = self.hash_query(query);

        if let Some(plan) = self.cache.get(&hash) {
            // Check TTL
            if plan.created_at.elapsed() < self.ttl {
                self.hits += 1;
                self.touch(hash);
                // Re-borrow after touch
                return self.cache.get(&hash);
            }
            // Expired - will be replaced on next put
            self.misses += 1;
            return None;
        }

        self.misses += 1;
        None
    }

    /// Get a cloned plan (for modification)
    pub fn get_clone(&mut self, query: &str) -> Option<CachedPlan> {
        let hash = self.hash_query(query);

        if let Some(plan) = self.cache.get(&hash) {
            // Check TTL
            if plan.created_at.elapsed() < self.ttl {
                self.hits += 1;
                self.touch(hash);
                return self.cache.get(&hash).cloned();
            }
            self.misses += 1;
            return None;
        }

        self.misses += 1;
        None
    }

    /// Insert a plan
    pub fn put(&mut self, query: &str, plan: CachedPlan) {
        let hash = self.hash_query(query);
        self.cache.insert(hash, plan);
        self.touch(hash);
        self.evict_if_needed();
    }

    /// Create and insert a new plan
    pub fn create_plan(
        &mut self,
        query: &str,
        terms: Vec<String>,
        term_weights: Vec<f32>,
        candidate_docs: Vec<u32>,
        component_boosts: Vec<(String, f32)>,
    ) -> &CachedPlan {
        let plan = CachedPlan {
            terms,
            term_weights,
            candidate_docs,
            component_boosts,
            created_at: crate::timing::start_timer(),
        };

        let hash = self.hash_query(query);
        self.cache.insert(hash, plan);
        self.touch(hash);
        self.evict_if_needed();
        // SAFETY: touch() moves key to front, evict removes from back
        // so freshly inserted entry is never evicted
        self.cache.get(&hash).expect("freshly inserted cache entry must exist after LRU eviction")
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.order.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let total = self.hits + self.misses;
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            hit_rate: if total > 0 { self.hits as f64 / total as f64 } else { 0.0 },
            size: self.cache.len(),
            capacity: self.capacity,
        }
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }
}

impl Default for QueryPlanCache {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Cache statistics
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub size: usize,
    pub capacity: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a test plan without directly calling Instant::now in test code
    fn test_plan(
        terms: Vec<&str>,
        weights: Vec<f32>,
        docs: Vec<u32>,
        boosts: Vec<(&str, f32)>,
    ) -> CachedPlan {
        CachedPlan {
            terms: terms.into_iter().map(String::from).collect(),
            term_weights: weights,
            candidate_docs: docs,
            component_boosts: boosts.into_iter().map(|(s, v)| (s.to_string(), v)).collect(),
            created_at: crate::timing::start_timer(),
        }
    }

    #[test]
    fn test_cache_creation() {
        let cache = QueryPlanCache::new(100);
        assert_eq!(cache.stats().capacity, 100);
    }

    #[test]
    fn test_cache_put_get() {
        let mut cache = QueryPlanCache::new(100);

        let plan =
            test_plan(vec!["hello", "world"], vec![1.0, 1.0], vec![1, 2, 3], vec![("trueno", 1.5)]);

        cache.put("hello world", plan);

        let retrieved = cache.get("hello world");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.expect("unexpected failure").terms.len(), 2);
    }

    #[test]
    fn test_cache_hit_miss() {
        let mut cache = QueryPlanCache::new(100);

        // Miss
        let _ = cache.get("query1");
        assert_eq!(cache.stats().misses, 1);

        // Put and hit
        cache.create_plan("query1", vec![], vec![], vec![], vec![]);
        let _ = cache.get("query1");
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_cache_case_insensitive() {
        let mut cache = QueryPlanCache::new(100);

        cache.create_plan("Hello World", vec![], vec![], vec![], vec![]);

        assert!(cache.get("hello world").is_some());
        assert!(cache.get("HELLO WORLD").is_some());
    }

    #[test]
    fn test_cache_ttl() {
        let mut cache = QueryPlanCache::with_ttl(100, Duration::from_millis(1));

        cache.create_plan("query", vec![], vec![], vec![], vec![]);

        // Immediately should be valid
        assert!(cache.get("query").is_some());

        // After TTL should be invalid
        std::thread::sleep(Duration::from_millis(10));
        assert!(cache.get("query").is_none());
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = QueryPlanCache::new(3);

        // Fill cache
        cache.create_plan("query1", vec![], vec![], vec![], vec![]);
        cache.create_plan("query2", vec![], vec![], vec![], vec![]);
        cache.create_plan("query3", vec![], vec![], vec![], vec![]);

        assert_eq!(cache.stats().size, 3);

        // Add one more, should evict oldest (query1)
        cache.create_plan("query4", vec![], vec![], vec![], vec![]);

        assert_eq!(cache.stats().size, 3);
        assert!(cache.get_clone("query1").is_none()); // Should be evicted
        assert!(cache.get_clone("query2").is_some());
        assert!(cache.get_clone("query3").is_some());
        assert!(cache.get_clone("query4").is_some());
    }

    #[test]
    fn test_cache_lru_touch() {
        let mut cache = QueryPlanCache::new(3);

        // Fill cache
        cache.create_plan("query1", vec![], vec![], vec![], vec![]);
        cache.create_plan("query2", vec![], vec![], vec![], vec![]);
        cache.create_plan("query3", vec![], vec![], vec![], vec![]);

        // Touch query1, making it most recently used
        let _ = cache.get("query1");

        // Add new item, should evict query2 (now oldest)
        cache.create_plan("query4", vec![], vec![], vec![], vec![]);

        assert!(cache.get_clone("query1").is_some()); // Should still exist
        assert!(cache.get_clone("query2").is_none()); // Should be evicted
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = QueryPlanCache::new(100);
        cache.create_plan("query1", vec![], vec![], vec![], vec![]);
        cache.create_plan("query2", vec![], vec![], vec![], vec![]);

        assert_eq!(cache.stats().size, 2);

        cache.clear();

        assert_eq!(cache.stats().size, 0);
    }

    #[test]
    fn test_cache_stats_reset() {
        let mut cache = QueryPlanCache::new(100);
        cache.create_plan("query", vec![], vec![], vec![], vec![]);
        let _ = cache.get("query"); // hit
        let _ = cache.get("nonexistent"); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);

        cache.reset_stats();

        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut cache = QueryPlanCache::new(100);
        cache.create_plan("query", vec![], vec![], vec![], vec![]);

        // 2 hits
        let _ = cache.get("query");
        let _ = cache.get("query");
        // 1 miss
        let _ = cache.get("nonexistent");

        let stats = cache.stats();
        assert!((stats.hit_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_cache_default() {
        let cache = QueryPlanCache::default();
        assert_eq!(cache.stats().capacity, 1000);
    }

    #[test]
    fn test_cache_zero_capacity() {
        // Zero capacity should default to 1000
        let cache = QueryPlanCache::new(0);
        assert_eq!(cache.stats().capacity, 1000);
    }

    #[test]
    fn test_cached_plan_fields() {
        let plan = test_plan(vec!["test"], vec![0.5], vec![1, 2, 3], vec![("boost", 1.2)]);
        assert_eq!(plan.terms.len(), 1);
        assert_eq!(plan.term_weights.len(), 1);
        assert_eq!(plan.candidate_docs.len(), 3);
        assert_eq!(plan.component_boosts.len(), 1);
    }

    #[test]
    fn test_cache_stats_fields() {
        let stats = CacheStats { hits: 10, misses: 5, hit_rate: 0.666, size: 100, capacity: 1000 };
        assert_eq!(stats.hits, 10);
        assert_eq!(stats.misses, 5);
        assert_eq!(stats.size, 100);
        assert_eq!(stats.capacity, 1000);
    }

    #[test]
    fn test_get_clone_returns_owned() {
        let mut cache = QueryPlanCache::new(100);
        cache.create_plan("query", vec!["term".to_string()], vec![1.0], vec![1], vec![]);

        let cloned = cache.get_clone("query");
        assert!(cloned.is_some());
        let plan = cloned.expect("unexpected failure");
        assert_eq!(plan.terms, vec!["term".to_string()]);
    }

    #[test]
    fn test_get_clone_miss() {
        let mut cache = QueryPlanCache::new(100);
        let result = cache.get_clone("nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_get_clone_expired() {
        let mut cache = QueryPlanCache::with_ttl(100, Duration::from_millis(1));
        cache.create_plan("query", vec![], vec![], vec![], vec![]);

        std::thread::sleep(Duration::from_millis(10));
        let result = cache.get_clone("query");
        assert!(result.is_none());
    }

    #[test]
    fn test_put_replaces_existing() {
        let mut cache = QueryPlanCache::new(100);

        cache.put("query", test_plan(vec!["old"], vec![], vec![], vec![]));
        cache.put("query", test_plan(vec!["new"], vec![], vec![], vec![]));

        let retrieved = cache.get("query").expect("key not found");
        assert_eq!(retrieved.terms, vec!["new".to_string()]);
    }

    #[test]
    fn test_hit_rate_no_accesses() {
        let cache = QueryPlanCache::new(100);
        let stats = cache.stats();
        assert_eq!(stats.hit_rate, 0.0);
    }

    #[test]
    fn test_create_plan_returns_reference() {
        let mut cache = QueryPlanCache::new(100);
        let plan = cache.create_plan(
            "query",
            vec!["term".to_string()],
            vec![1.0, 2.0],
            vec![1, 2, 3],
            vec![("boost".to_string(), 1.5)],
        );
        assert_eq!(plan.terms.len(), 1);
        assert_eq!(plan.term_weights.len(), 2);
        assert_eq!(plan.candidate_docs.len(), 3);
    }

    #[test]
    fn test_with_ttl_custom_duration() {
        let cache = QueryPlanCache::with_ttl(50, Duration::from_secs(60));
        assert_eq!(cache.stats().capacity, 50);
    }

    #[test]
    fn test_cached_plan_clone() {
        let plan = test_plan(vec!["a", "b"], vec![1.0, 2.0], vec![10, 20], vec![]);
        let cloned = plan.clone();
        assert_eq!(cloned.terms, plan.terms);
        assert_eq!(cloned.candidate_docs, plan.candidate_docs);
    }
}
