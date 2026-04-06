//! Heijunka Reindexer - Load-Leveled Incremental Indexing
//!
//! Implements Toyota Way Heijunka (平準化) principle for smooth workload distribution.
//! Prevents thundering herd during bulk updates through batched processing.

use super::fingerprint::DocumentFingerprint;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Heijunka reindexer for load-leveled updates
///
/// Following queueing theory principles from Harchol-Balter (2013)
/// and tail latency management from Dean & Barroso (2013).
#[derive(Debug)]
pub struct HeijunkaReindexer {
    /// Maximum documents per batch (load leveling)
    batch_size: usize,
    /// Inter-batch delay for backpressure (milliseconds)
    batch_delay_ms: u64,
    /// Priority queue ordered by staleness
    queue: BinaryHeap<StalenessEntry>,
    /// Document fingerprints for change detection
    fingerprints: HashMap<String, DocumentFingerprint>,
    /// Query counts for popularity-weighted staleness
    query_counts: HashMap<String, u64>,
    /// Configuration
    config: HeijunkaConfig,
}

/// Heijunka configuration
#[derive(Debug, Clone)]
pub struct HeijunkaConfig {
    /// Maximum batch size
    pub batch_size: usize,
    /// Delay between batches (ms)
    pub batch_delay_ms: u64,
    /// Maximum staleness before forced reindex (seconds)
    pub max_staleness_seconds: u64,
    /// Query count decay factor (for aging popularity)
    pub popularity_decay: f64,
}

impl Default for HeijunkaConfig {
    fn default() -> Self {
        Self {
            batch_size: 50,
            batch_delay_ms: 100,
            max_staleness_seconds: 86400, // 24 hours
            popularity_decay: 0.95,
        }
    }
}

/// Entry in the staleness priority queue
#[derive(Debug, Clone)]
struct StalenessEntry {
    /// Document ID
    doc_id: String,
    /// Staleness score (higher = more stale, process first)
    staleness_score: f64,
    /// Document path
    path: PathBuf,
}

impl From<StalenessEntry> for ReindexTask {
    fn from(entry: StalenessEntry) -> Self {
        Self { doc_id: entry.doc_id, path: entry.path, staleness_score: entry.staleness_score }
    }
}

impl PartialEq for StalenessEntry {
    fn eq(&self, other: &Self) -> bool {
        self.doc_id == other.doc_id
    }
}

impl Eq for StalenessEntry {}

impl PartialOrd for StalenessEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for StalenessEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher staleness score = higher priority
        self.staleness_score
            .partial_cmp(&other.staleness_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl HeijunkaReindexer {
    /// Create a new Heijunka reindexer
    pub fn new() -> Self {
        Self::with_config(HeijunkaConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: HeijunkaConfig) -> Self {
        Self {
            batch_size: config.batch_size,
            batch_delay_ms: config.batch_delay_ms,
            queue: BinaryHeap::new(),
            fingerprints: HashMap::new(),
            query_counts: HashMap::new(),
            config,
        }
    }

    /// Calculate staleness score for a document
    ///
    /// Score formula: recency_weight * popularity_weight
    /// - recency_weight: exponential decay based on age
    /// - popularity_weight: log-scaled query count
    pub fn staleness_score(age_seconds: u64, query_count: u64) -> f64 {
        let recency_weight = 1.0 - (-(age_seconds as f64) / 86400.0).exp();
        let popularity_weight = (query_count as f64 + 1.0).ln();
        recency_weight * popularity_weight
    }

    /// Add a document to the reindex queue
    pub fn enqueue(&mut self, doc_id: &str, path: PathBuf, age_seconds: u64) {
        let query_count = self.query_counts.get(doc_id).copied().unwrap_or(0);
        let staleness_score = Self::staleness_score(age_seconds, query_count);

        self.queue.push(StalenessEntry { doc_id: doc_id.to_string(), staleness_score, path });
    }

    /// Record a query for a document (affects staleness priority)
    pub fn record_query(&mut self, doc_id: &str) {
        *self.query_counts.entry(doc_id.to_string()).or_insert(0) += 1;
    }

    /// Apply popularity decay to all query counts
    pub fn decay_popularity(&mut self) {
        for count in self.query_counts.values_mut() {
            *count = (*count as f64 * self.config.popularity_decay) as u64;
        }
    }

    /// Get the next batch of documents to reindex
    pub fn next_batch(&mut self) -> Vec<ReindexTask> {
        let mut batch = Vec::with_capacity(self.batch_size);

        while batch.len() < self.batch_size {
            if let Some(entry) = self.queue.pop() {
                batch.push(entry.into());
            } else {
                break;
            }
        }

        batch
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Get queue size
    pub fn queue_size(&self) -> usize {
        self.queue.len()
    }

    /// Store fingerprint for a document
    pub fn store_fingerprint(&mut self, doc_id: &str, fingerprint: DocumentFingerprint) {
        self.fingerprints.insert(doc_id.to_string(), fingerprint);
    }

    /// Get fingerprint for a document
    pub fn get_fingerprint(&self, doc_id: &str) -> Option<&DocumentFingerprint> {
        self.fingerprints.get(doc_id)
    }

    /// Calculate delta between old and new chunks (Muda elimination)
    pub fn calculate_delta<'a>(
        old_hashes: &HashSet<[u8; 32]>,
        new_chunks: &'a [(String, [u8; 32])],
    ) -> DeltaSet<'a> {
        let new_hashes: HashSet<[u8; 32]> = new_chunks.iter().map(|(_, h)| *h).collect();

        DeltaSet {
            to_add: new_chunks.iter().filter(|(_, h)| !old_hashes.contains(h)).collect(),
            to_remove: old_hashes.iter().filter(|h| !new_hashes.contains(*h)).copied().collect(),
        }
    }

    /// Get batch delay as Duration
    pub fn batch_delay(&self) -> Duration {
        Duration::from_millis(self.batch_delay_ms)
    }

    /// Get reindexing statistics
    pub fn stats(&self) -> ReindexerStats {
        ReindexerStats {
            queue_size: self.queue.len(),
            tracked_documents: self.fingerprints.len(),
            total_queries: self.query_counts.values().sum(),
        }
    }
}

impl Default for HeijunkaReindexer {
    fn default() -> Self {
        Self::new()
    }
}

/// A task to reindex a document
#[derive(Debug, Clone)]
pub struct ReindexTask {
    /// Document ID
    pub doc_id: String,
    /// Document path
    pub path: PathBuf,
    /// Staleness score (for logging/metrics)
    pub staleness_score: f64,
}

/// Delta set for incremental updates (Muda elimination)
#[derive(Debug)]
pub struct DeltaSet<'a> {
    /// Chunks to add (new or modified)
    pub to_add: Vec<&'a (String, [u8; 32])>,
    /// Chunk hashes to remove
    pub to_remove: Vec<[u8; 32]>,
}

impl DeltaSet<'_> {
    /// Calculate efficiency (percentage of chunks unchanged)
    pub fn efficiency(&self, _total_old: usize, total_new: usize) -> f64 {
        if total_new == 0 {
            return 100.0;
        }
        let unchanged = total_new - self.to_add.len();
        unchanged as f64 / total_new as f64 * 100.0
    }
}

/// Reindexer statistics
#[derive(Debug, Clone)]
pub struct ReindexerStats {
    /// Documents in queue
    pub queue_size: usize,
    /// Documents with stored fingerprints
    pub tracked_documents: usize,
    /// Total queries recorded
    pub total_queries: u64,
}

/// Progress tracker for reindexing
#[derive(Debug)]
pub struct ReindexProgress {
    /// Total documents to process
    pub total: usize,
    /// Documents processed
    pub processed: usize,
    /// Documents modified
    pub modified: usize,
    /// Documents added
    pub added: usize,
    /// Documents removed
    pub removed: usize,
    /// Start time
    start_time: Instant,
}

impl ReindexProgress {
    /// Create a new progress tracker
    pub fn new(total: usize) -> Self {
        Self {
            total,
            processed: 0,
            modified: 0,
            added: 0,
            removed: 0,
            start_time: crate::timing::start_timer(),
        }
    }

    /// Record a processed document
    pub fn record_processed(&mut self, was_modified: bool) {
        self.processed += 1;
        if was_modified {
            self.modified += 1;
        }
    }

    /// Get completion percentage
    pub fn percent_complete(&self) -> f64 {
        if self.total == 0 {
            return 100.0;
        }
        self.processed as f64 / self.total as f64 * 100.0
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get processing rate (docs/second)
    pub fn rate(&self) -> f64 {
        let elapsed = self.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.processed as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Estimate time remaining
    pub fn eta(&self) -> Duration {
        let rate = self.rate();
        if rate > 0.0 {
            let remaining = self.total - self.processed;
            Duration::from_secs_f64(remaining as f64 / rate)
        } else {
            Duration::from_secs(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a synthetic doc ID for test use.
    fn test_doc_id(i: usize) -> String {
        format!("doc{i}")
    }

    /// Generate a synthetic path for test use.
    fn test_doc_path(i: usize) -> PathBuf {
        PathBuf::from(format!("/doc{i}"))
    }

    /// Enqueue `count` synthetic documents with linearly increasing age.
    fn enqueue_synthetic(reindexer: &mut HeijunkaReindexer, count: usize, age_step: u64) {
        for i in 0..count {
            reindexer.enqueue(&test_doc_id(i), test_doc_path(i), i as u64 * age_step);
        }
    }

    /// Build a `HashSet<[u8; 32]>` from an iterator of single-byte fill values.
    fn hash_set_from_fills(fills: impl IntoIterator<Item = u8>) -> HashSet<[u8; 32]> {
        fills.into_iter().map(|b| [b; 32]).collect()
    }

    /// Build a `Vec<(String, [u8; 32])>` of chunks from `(label, fill_byte)` pairs.
    fn chunks_from_fills(pairs: &[(&str, u8)]) -> Vec<(String, [u8; 32])> {
        pairs.iter().map(|(label, b)| (label.to_string(), [*b; 32])).collect()
    }

    #[test]
    fn test_heijunka_creation() {
        let reindexer = HeijunkaReindexer::new();
        assert!(reindexer.is_empty());
        assert_eq!(reindexer.queue_size(), 0);
    }

    #[test]
    fn test_staleness_score_new_document() {
        // Brand new document (0 age) should have low staleness
        let score = HeijunkaReindexer::staleness_score(0, 0);
        assert!(score < 0.1);
    }

    #[test]
    fn test_staleness_score_old_document() {
        // Old document (1 day) should have higher staleness
        let score = HeijunkaReindexer::staleness_score(86400, 1);
        // With 1 day age and query_count=1: recency ~0.63, popularity ~0.69
        assert!(score > 0.3);
    }

    #[test]
    fn test_staleness_score_popular_document() {
        // Popular document should have higher staleness (more important to update)
        let score_low = HeijunkaReindexer::staleness_score(3600, 1);
        let score_high = HeijunkaReindexer::staleness_score(3600, 100);
        assert!(score_high > score_low);
    }

    #[test]
    fn test_enqueue_and_batch() {
        let mut reindexer = HeijunkaReindexer::new();

        // Add query counts so popularity factor is non-zero
        for id in ["doc1", "doc2", "doc3"] {
            reindexer.record_query(id);
        }

        reindexer.enqueue("doc1", test_doc_path(1), 1000);
        reindexer.enqueue("doc2", test_doc_path(2), 5000);
        reindexer.enqueue("doc3", test_doc_path(3), 100);

        assert_eq!(reindexer.queue_size(), 3);

        let batch = reindexer.next_batch();

        // Higher staleness (older) should come first
        assert!(!batch.is_empty());
        assert_eq!(batch.len(), 3);
        // doc2 has highest age (5000s) so should be first
        assert_eq!(batch[0].doc_id, "doc2");
    }

    #[test]
    fn test_batch_size_limit() {
        let config = HeijunkaConfig { batch_size: 2, ..Default::default() };
        let mut reindexer = HeijunkaReindexer::with_config(config);

        enqueue_synthetic(&mut reindexer, 10, 100);

        let batch = reindexer.next_batch();
        assert_eq!(batch.len(), 2); // Limited to batch_size
    }

    #[test]
    fn test_record_query() {
        let mut reindexer = HeijunkaReindexer::new();

        for _ in 0..3 {
            reindexer.record_query("doc1");
        }
        reindexer.record_query("doc2");

        // doc1 should have higher query count
        assert_eq!(*reindexer.query_counts.get("doc1").expect("key not found"), 3);
        assert_eq!(*reindexer.query_counts.get("doc2").expect("key not found"), 1);
    }

    #[test]
    fn test_popularity_decay() {
        let mut reindexer = HeijunkaReindexer::new();

        for _ in 0..4 {
            reindexer.record_query("doc1");
        }

        let before = *reindexer.query_counts.get("doc1").expect("key not found");
        reindexer.decay_popularity();
        let after = *reindexer.query_counts.get("doc1").expect("key not found");

        assert!(after < before);
    }

    #[test]
    fn test_delta_calculation() {
        let old_hashes = hash_set_from_fills(1..=3);
        let new_chunks = chunks_from_fills(&[("chunk1", 2), ("chunk2", 3), ("chunk3", 4)]);

        let delta = HeijunkaReindexer::calculate_delta(&old_hashes, &new_chunks);

        // One chunk to add (hash 4)
        assert_eq!(delta.to_add.len(), 1);
        assert_eq!(delta.to_add[0].1, [4u8; 32]);

        // One chunk to remove (hash 1)
        assert_eq!(delta.to_remove.len(), 1);
        assert!(delta.to_remove.contains(&[1u8; 32]));
    }

    #[test]
    fn test_delta_efficiency() {
        let old_hashes = hash_set_from_fills(1..=4);
        let new_chunks =
            chunks_from_fills(&[("c1", 1), ("c2", 2), ("c3", 3), ("c4", 5) /* one changed */]);

        let delta = HeijunkaReindexer::calculate_delta(&old_hashes, &new_chunks);
        let efficiency = delta.efficiency(4, 4);

        // 3/4 unchanged = 75% efficiency
        assert!((efficiency - 75.0).abs() < 0.1);
    }

    #[test]
    fn test_progress_tracking() {
        let mut progress = ReindexProgress::new(100);

        progress.record_processed(false);
        progress.record_processed(true);
        progress.record_processed(false);

        assert_eq!(progress.processed, 3);
        assert_eq!(progress.modified, 1);
        assert!((progress.percent_complete() - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_progress_rate() {
        let progress = ReindexProgress::new(100);
        // Just created, rate should be 0 or very low
        assert!(progress.rate() >= 0.0);
    }

    #[test]
    fn test_fingerprint_storage() {
        let mut reindexer = HeijunkaReindexer::new();
        let fp = DocumentFingerprint {
            content_hash: [1u8; 32],
            chunker_config_hash: [2u8; 32],
            embedding_model_hash: [3u8; 32],
            indexed_at: 12345,
        };

        reindexer.store_fingerprint("doc1", fp.clone());

        let retrieved = reindexer.get_fingerprint("doc1");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.expect("unexpected failure").content_hash, [1u8; 32]);
    }

    #[test]
    fn test_heijunka_default() {
        let reindexer = HeijunkaReindexer::default();
        assert!(reindexer.is_empty());
    }

    #[test]
    fn test_heijunka_config_default() {
        let config = HeijunkaConfig::default();
        assert_eq!(config.batch_size, 50);
        assert_eq!(config.batch_delay_ms, 100);
        assert_eq!(config.max_staleness_seconds, 86400);
        assert!((config.popularity_decay - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_batch_delay() {
        let reindexer = HeijunkaReindexer::new();
        let delay = reindexer.batch_delay();
        assert_eq!(delay, Duration::from_millis(100));
    }

    #[test]
    fn test_stats() {
        let mut reindexer = HeijunkaReindexer::new();
        reindexer.record_query("doc1");
        reindexer.record_query("doc2");

        let stats = reindexer.stats();
        assert_eq!(stats.queue_size, 0);
        assert_eq!(stats.tracked_documents, 0);
        assert_eq!(stats.total_queries, 2);
    }

    #[test]
    fn test_progress_empty() {
        let progress = ReindexProgress::new(0);
        assert!((progress.percent_complete() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_delta_efficiency_empty() {
        let old_hashes = hash_set_from_fills(std::iter::empty());
        let new_chunks = chunks_from_fills(&[]);
        let delta = HeijunkaReindexer::calculate_delta(&old_hashes, &new_chunks);
        let efficiency = delta.efficiency(0, 0);
        assert!((efficiency - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_progress_eta() {
        let mut progress = ReindexProgress::new(100);
        progress.processed = 50;
        // ETA depends on elapsed time, which is instant here
        let _ = progress.eta();
    }

    #[test]
    fn test_get_fingerprint_not_found() {
        let reindexer = HeijunkaReindexer::new();
        assert!(reindexer.get_fingerprint("nonexistent").is_none());
    }

    // Property-based tests for Heijunka reindexer
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]

            /// Property: Staleness score is non-negative
            #[test]
            fn prop_staleness_score_non_negative(age_seconds in 0u64..1000000, query_count in 0u64..10000) {
                let score = HeijunkaReindexer::staleness_score(age_seconds, query_count);
                prop_assert!(score >= 0.0, "Staleness score {} should be >= 0", score);
            }

            /// Property: Higher age produces higher staleness
            #[test]
            fn prop_higher_age_higher_staleness(
                low_age in 0u64..10000,
                high_age in 50000u64..100000,
                query_count in 1u64..100
            ) {
                let low_score = HeijunkaReindexer::staleness_score(low_age, query_count);
                let high_score = HeijunkaReindexer::staleness_score(high_age, query_count);
                prop_assert!(high_score >= low_score, "Age {} score {} < age {} score {}", high_age, high_score, low_age, low_score);
            }

            /// Property: Higher query count produces higher staleness (for same age)
            #[test]
            fn prop_higher_popularity_higher_staleness(
                age_seconds in 1000u64..50000,
                low_count in 0u64..10,
                high_count in 100u64..1000
            ) {
                let low_score = HeijunkaReindexer::staleness_score(age_seconds, low_count);
                let high_score = HeijunkaReindexer::staleness_score(age_seconds, high_count);
                prop_assert!(high_score >= low_score);
            }

            /// Property: Batch size is respected
            #[test]
            fn prop_batch_size_respected(batch_size in 1usize..20, num_docs in 1usize..100) {
                let config = HeijunkaConfig {
                    batch_size,
                    ..Default::default()
                };
                let mut reindexer = HeijunkaReindexer::with_config(config);

                enqueue_synthetic(&mut reindexer, num_docs, 100);

                let batch = reindexer.next_batch();
                prop_assert!(batch.len() <= batch_size);
            }

            /// Property: Enqueue increases queue size
            #[test]
            fn prop_enqueue_increases_size(num_docs in 1usize..50) {
                let mut reindexer = HeijunkaReindexer::new();

                enqueue_synthetic(&mut reindexer, num_docs, 0);

                prop_assert_eq!(reindexer.queue_size(), num_docs);
            }

            /// Property: Progress percentage is in [0, 100]
            #[test]
            fn prop_progress_percentage_valid(total in 0usize..1000, processed in 0usize..500) {
                let mut progress = ReindexProgress::new(total);
                for _ in 0..processed.min(total) {
                    progress.record_processed(false);
                }
                let pct = progress.percent_complete();
                prop_assert!((0.0..=100.0).contains(&pct), "Progress {} not in [0, 100]", pct);
            }

            /// Property: Delta efficiency is in [0, 100]
            #[test]
            fn prop_delta_efficiency_valid(
                old_count in 0usize..10,
                new_count in 0usize..10,
                overlap in 0usize..10
            ) {
                let overlap = overlap.min(old_count).min(new_count);

                let old_hashes = hash_set_from_fills((0..old_count).map(|i| i as u8));
                let new_chunks: Vec<(String, [u8; 32])> = (0..new_count)
                    .map(|i| {
                        let fill = if i < overlap { i as u8 } else { (old_count + i) as u8 };
                        (format!("c{i}"), [fill; 32])
                    })
                    .collect();

                let delta = HeijunkaReindexer::calculate_delta(&old_hashes, &new_chunks);
                let efficiency = delta.efficiency(old_count, new_count);
                prop_assert!((0.0..=100.0).contains(&efficiency), "Efficiency {} not in [0, 100]", efficiency);
            }
        }
    }
}
