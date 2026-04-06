//! Time-series storage with automatic tiering and SIMD-accelerated queries.
//!
//! ## Architecture
//!
//! Three-tier storage hierarchy:
//! - **Hot tier**: SimdRingBuffer (< 5 minutes, in-memory, O(1) stats)
//! - **Warm tier**: CompressedMetricStore (5 min - 1 hour, delta-encoded)
//! - **Cold tier**: Disk persistence (> 1 hour, fsync batched)
//!
//! ## Performance Targets (Falsifiable - H₁₁)
//!
//! - Query performance: ≥10x vs SQLite for time-range aggregations
//! - Tier migration: < 1ms for hot→warm transition
//! - Disk writes: Batched with configurable fsync interval
//!
//! ## SIMD Query Acceleration
//!
//! Columnar queries use SIMD for:
//! - Range scans (parallel timestamp comparison)
//! - Aggregations (sum, min, max, mean via AVX2/NEON)
//! - Filtering (predicate evaluation on value columns)

use super::compressed::{now_micros, CompressedMetricStore, Timestamp};
use super::kernels;
use super::ring_buffer::SimdRingBuffer;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::RwLock;

/// Configuration for tier boundaries.
#[derive(Debug, Clone)]
pub struct TierConfig {
    /// Hot tier duration in microseconds (default: 5 minutes).
    pub hot_duration_us: u64,
    /// Warm tier duration in microseconds (default: 1 hour).
    pub warm_duration_us: u64,
    /// Fsync batch interval in microseconds (default: 1 second).
    pub fsync_interval_us: u64,
    /// Block size for compression (default: 64 samples).
    pub block_size: usize,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            hot_duration_us: 5 * 60 * 1_000_000,   // 5 minutes
            warm_duration_us: 60 * 60 * 1_000_000, // 1 hour
            fsync_interval_us: 1_000_000,          // 1 second
            block_size: 64,
        }
    }
}

/// Query result with SIMD-computed aggregations.
#[derive(Debug, Clone, Default)]
pub struct QueryResult {
    /// Matching samples.
    pub samples: Vec<(Timestamp, f64)>,
    /// Aggregations computed via SIMD.
    pub aggregations: Aggregations,
    /// Query execution time in microseconds.
    pub query_time_us: u64,
    /// Number of tiers scanned.
    pub tiers_scanned: u8,
}

/// SIMD-computed aggregations.
#[derive(Debug, Clone, Default)]
pub struct Aggregations {
    /// Count of samples.
    pub count: usize,
    /// Sum of values.
    pub sum: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
}

impl Aggregations {
    /// Computes aggregations from samples using SIMD.
    #[must_use]
    pub fn from_samples(samples: &[(Timestamp, f64)]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }

        let values: Vec<f64> = samples.iter().map(|(_, v)| *v).collect();
        let stats = kernels::simd_statistics(&values);

        Self {
            count: values.len(),
            sum: stats.sum,
            min: stats.min,
            max: stats.max,
            mean: stats.mean(),
            std_dev: stats.std_dev(),
        }
    }

    /// Merges two aggregations (for parallel reduction).
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }

        let total_count = self.count + other.count;
        let new_sum = self.sum + other.sum;
        let new_mean = new_sum / total_count as f64;

        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.sum = new_sum;
        self.mean = new_mean;
        self.count = total_count;
        // Note: std_dev merge requires sum of squares - simplified here
    }
}

/// Time-series table with automatic tiering.
#[derive(Debug)]
pub struct TimeSeriesTable {
    /// Table name.
    name: String,
    /// Configuration.
    config: TierConfig,
    /// Hot tier: recent data with O(1) statistics.
    hot: SimdRingBuffer,
    /// Hot tier timestamps (parallel to values).
    hot_timestamps: Vec<Timestamp>,
    /// Warm tier: compressed historical data.
    warm: CompressedMetricStore,
    /// Cold tier: disk persistence path.
    cold_path: Option<PathBuf>,
    /// Pending cold writes (fsync batching).
    cold_pending: Vec<(Timestamp, f64)>,
    /// Last fsync timestamp.
    last_fsync: AtomicU64,
    /// Total samples written to cold tier.
    cold_samples: AtomicU64,
    /// Whether cold tier is enabled.
    cold_enabled: AtomicBool,
}

impl TimeSeriesTable {
    /// Creates a new time-series table.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self::with_config(name, TierConfig::default())
    }

    /// Creates a new time-series table with custom configuration.
    #[must_use]
    pub fn with_config(name: &str, config: TierConfig) -> Self {
        // Hot tier capacity: 5 minutes at 1Hz = 300 samples
        let hot_capacity = (config.hot_duration_us / 1_000_000) as usize;
        let hot_capacity = hot_capacity.max(64); // Minimum 64

        Self {
            name: name.to_string(),
            hot: SimdRingBuffer::new(hot_capacity),
            hot_timestamps: Vec::with_capacity(hot_capacity),
            warm: CompressedMetricStore::new(name, config.block_size),
            cold_path: None,
            cold_pending: Vec::with_capacity(1024),
            last_fsync: AtomicU64::new(0),
            cold_samples: AtomicU64::new(0),
            cold_enabled: AtomicBool::new(false),
            config,
        }
    }

    /// Enables cold tier persistence to the specified directory.
    pub fn enable_persistence(&mut self, dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("{}.tsdb", self.name));
        self.cold_path = Some(path);
        self.cold_enabled.store(true, Ordering::Release);
        Ok(())
    }

    /// Inserts a sample into the table.
    pub fn insert(&mut self, timestamp: Timestamp, value: f64) {
        let now = now_micros();

        // Check for tier migration
        self.maybe_migrate(now);

        // Insert into hot tier
        self.hot.push(value);

        // Track timestamps separately (ring buffer only stores values)
        if self.hot_timestamps.len() >= self.hot.capacity() {
            self.hot_timestamps.remove(0);
        }
        self.hot_timestamps.push(timestamp);
    }

    /// Inserts a sample with current timestamp.
    pub fn insert_now(&mut self, value: f64) {
        self.insert(now_micros(), value);
    }

    /// Performs tier migration if needed.
    fn maybe_migrate(&mut self, now: Timestamp) {
        if self.hot_timestamps.is_empty() {
            return;
        }

        let oldest_hot = self.hot_timestamps[0];
        let hot_age = now.saturating_sub(oldest_hot);

        // Migrate hot → warm if too old
        if hot_age > self.config.hot_duration_us && self.hot_timestamps.len() > 1 {
            // Find samples to migrate
            let migrate_until = now.saturating_sub(self.config.hot_duration_us);
            let mut migrate_count = 0;

            for (i, &ts) in self.hot_timestamps.iter().enumerate() {
                if ts < migrate_until {
                    migrate_count = i + 1;
                } else {
                    break;
                }
            }

            if migrate_count > 0 {
                // Get values from hot tier (this is approximate since ring buffer doesn't track indices)
                let hot_values = self.hot.to_vec();
                let start_idx = hot_values.len().saturating_sub(self.hot_timestamps.len());

                for i in 0..migrate_count.min(hot_values.len()) {
                    let idx = start_idx + i;
                    if idx < hot_values.len() && i < self.hot_timestamps.len() {
                        self.warm.push(self.hot_timestamps[i], hot_values[idx]);
                    }
                }

                // Remove migrated timestamps
                self.hot_timestamps.drain(0..migrate_count);
            }
        }

        // Check warm tier for cold migration
        self.maybe_migrate_to_cold(now);
    }

    /// Migrates old warm data to cold tier.
    fn maybe_migrate_to_cold(&mut self, now: Timestamp) {
        if !self.cold_enabled.load(Ordering::Acquire) {
            return;
        }

        let warm_cutoff = now.saturating_sub(self.config.warm_duration_us);

        // Query warm tier for old samples
        let old_samples = self.warm.query(0, warm_cutoff);

        if !old_samples.is_empty() {
            self.cold_pending.extend(old_samples);

            // Check if we should fsync
            let last = self.last_fsync.load(Ordering::Relaxed);
            if now.saturating_sub(last) > self.config.fsync_interval_us {
                self.flush_cold();
            }
        }
    }

    /// Flushes pending cold writes to disk.
    pub fn flush_cold(&mut self) {
        if self.cold_pending.is_empty() {
            return;
        }

        if let Some(path) = &self.cold_path {
            if let Ok(file) = OpenOptions::new().create(true).append(true).open(path) {
                let mut writer = BufWriter::new(file);

                for (ts, val) in &self.cold_pending {
                    // Simple binary format: u64 timestamp + f64 value
                    let _ = writer.write_all(&ts.to_le_bytes());
                    let _ = writer.write_all(&val.to_le_bytes());
                }

                if writer.flush().is_ok() {
                    // Try to fsync
                    if let Ok(file) = writer.into_inner() {
                        let _ = file.sync_all();
                    }
                }

                self.cold_samples.fetch_add(self.cold_pending.len() as u64, Ordering::Relaxed);
            }

            self.cold_pending.clear();
            self.last_fsync.store(now_micros(), Ordering::Release);
        }
    }

    /// Queries samples in a time range with SIMD-accelerated aggregations.
    pub fn query(&self, start: Timestamp, end: Timestamp) -> QueryResult {
        let query_start = now_micros();
        let mut samples = Vec::new();
        let mut tiers_scanned = 0u8;

        // Query hot tier
        if !self.hot_timestamps.is_empty() {
            let values = self.hot.to_vec();
            let offset = values.len().saturating_sub(self.hot_timestamps.len());

            for (i, &ts) in self.hot_timestamps.iter().enumerate() {
                if ts >= start && ts <= end {
                    let value_idx = offset + i;
                    if value_idx < values.len() {
                        samples.push((ts, values[value_idx]));
                    }
                }
            }
            tiers_scanned += 1;
        }

        // Query warm tier
        let warm_samples = self.warm.query(start, end);
        if !warm_samples.is_empty() {
            samples.extend(warm_samples);
            tiers_scanned += 1;
        }

        // Query cold tier if needed and enabled
        if self.cold_enabled.load(Ordering::Acquire) {
            if let Some(cold_samples) = self.query_cold(start, end) {
                if !cold_samples.is_empty() {
                    samples.extend(cold_samples);
                    tiers_scanned += 1;
                }
            }
        }

        // Sort by timestamp
        samples.sort_by_key(|(ts, _)| *ts);

        // Compute aggregations using SIMD
        let aggregations = Aggregations::from_samples(&samples);

        let query_time_us = now_micros().saturating_sub(query_start);

        QueryResult { samples, aggregations, query_time_us, tiers_scanned }
    }

    /// Queries the cold tier from disk.
    fn query_cold(&self, start: Timestamp, end: Timestamp) -> Option<Vec<(Timestamp, f64)>> {
        let path = self.cold_path.as_ref()?;
        let file = File::open(path).ok()?;
        let mut reader = BufReader::new(file);

        let mut samples = Vec::new();
        let mut ts_buf = [0u8; 8];
        let mut val_buf = [0u8; 8];

        while reader.read_exact(&mut ts_buf).is_ok() && reader.read_exact(&mut val_buf).is_ok() {
            let ts = u64::from_le_bytes(ts_buf);
            let val = f64::from_le_bytes(val_buf);

            if ts >= start && ts <= end {
                samples.push((ts, val));
            }
        }

        Some(samples)
    }

    /// Queries with a predicate filter (SIMD-accelerated).
    pub fn query_filtered<F>(&self, start: Timestamp, end: Timestamp, predicate: F) -> QueryResult
    where
        F: Fn(f64) -> bool,
    {
        let mut result = self.query(start, end);

        // Apply predicate filter
        result.samples.retain(|(_, v)| predicate(*v));

        // Recompute aggregations for filtered data
        result.aggregations = Aggregations::from_samples(&result.samples);

        result
    }

    /// Computes a time-windowed aggregation using SIMD.
    pub fn aggregate_windows(
        &self,
        start: Timestamp,
        end: Timestamp,
        window_size_us: u64,
    ) -> Vec<(Timestamp, Aggregations)> {
        let result = self.query(start, end);
        let mut windows = Vec::new();

        if result.samples.is_empty() || window_size_us == 0 {
            return windows;
        }

        let mut window_start = start;
        while window_start < end {
            let window_end = window_start + window_size_us;

            // Extract samples in this window
            let window_samples: Vec<(Timestamp, f64)> = result
                .samples
                .iter()
                .filter(|(ts, _)| *ts >= window_start && *ts < window_end)
                .copied()
                .collect();

            if !window_samples.is_empty() {
                let agg = Aggregations::from_samples(&window_samples);
                windows.push((window_start, agg));
            }

            window_start = window_end;
        }

        windows
    }

    /// Returns statistics about the table.
    #[must_use]
    pub fn stats(&self) -> TableStats {
        TableStats {
            name: self.name.clone(),
            hot_samples: self.hot.len(),
            warm_samples: self.warm.len(),
            cold_samples: self.cold_samples.load(Ordering::Relaxed),
            warm_compression_ratio: self.warm.avg_compression_ratio(),
            cold_enabled: self.cold_enabled.load(Ordering::Acquire),
        }
    }

    /// Returns the table name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics about a time-series table.
#[derive(Debug, Clone)]
pub struct TableStats {
    /// Table name.
    pub name: String,
    /// Samples in hot tier.
    pub hot_samples: usize,
    /// Samples in warm tier.
    pub warm_samples: usize,
    /// Samples in cold tier.
    pub cold_samples: u64,
    /// Warm tier compression ratio.
    pub warm_compression_ratio: f64,
    /// Whether cold tier is enabled.
    pub cold_enabled: bool,
}

impl TableStats {
    /// Total samples across all tiers.
    #[must_use]
    pub fn total_samples(&self) -> u64 {
        self.hot_samples as u64 + self.warm_samples as u64 + self.cold_samples
    }
}

/// Multi-table time-series database.
#[derive(Debug)]
pub struct TimeSeriesDb {
    /// Tables indexed by name.
    tables: RwLock<HashMap<String, TimeSeriesTable>>,
    /// Default configuration for new tables.
    config: TierConfig,
    /// Persistence directory.
    persist_dir: Option<PathBuf>,
}

impl TimeSeriesDb {
    /// Creates a new in-memory time-series database.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tables: RwLock::new(HashMap::new()),
            config: TierConfig::default(),
            persist_dir: None,
        }
    }

    /// Creates a new time-series database with persistence.
    pub fn with_persistence(dir: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        Ok(Self {
            tables: RwLock::new(HashMap::new()),
            config: TierConfig::default(),
            persist_dir: Some(dir.to_path_buf()),
        })
    }

    /// Gets or creates a table.
    pub fn table(&self, name: &str) -> Option<()> {
        let mut tables = self.tables.write().ok()?;
        if !tables.contains_key(name) {
            let mut table = TimeSeriesTable::with_config(name, self.config.clone());
            if let Some(dir) = &self.persist_dir {
                let _ = table.enable_persistence(dir);
            }
            tables.insert(name.to_string(), table);
        }
        Some(())
    }

    /// Inserts a sample into a table.
    pub fn insert(&self, table: &str, timestamp: Timestamp, value: f64) -> bool {
        if let Ok(mut tables) = self.tables.write() {
            if let Some(t) = tables.get_mut(table) {
                t.insert(timestamp, value);
                return true;
            }
            // Auto-create table
            let mut t = TimeSeriesTable::with_config(table, self.config.clone());
            if let Some(dir) = &self.persist_dir {
                let _ = t.enable_persistence(dir);
            }
            t.insert(timestamp, value);
            tables.insert(table.to_string(), t);
            return true;
        }
        false
    }

    /// Queries a table.
    pub fn query(&self, table: &str, start: Timestamp, end: Timestamp) -> Option<QueryResult> {
        let tables = self.tables.read().ok()?;
        tables.get(table).map(|t| t.query(start, end))
    }

    /// Returns all table names.
    pub fn table_names(&self) -> Vec<String> {
        self.tables.read().map(|t| t.keys().cloned().collect()).unwrap_or_default()
    }

    /// Flushes all tables to disk.
    pub fn flush(&self) {
        if let Ok(mut tables) = self.tables.write() {
            for table in tables.values_mut() {
                table.warm.flush();
                table.flush_cold();
            }
        }
    }
}

impl Default for TimeSeriesDb {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeseries_table_insert_query() {
        let mut table = TimeSeriesTable::new("test_metric");

        // Insert samples
        for i in 0..100 {
            table.insert(i as u64 * 1000, f64::from(i));
        }

        // Query all
        let result = table.query(0, 100_000);
        assert!(!result.samples.is_empty());
        assert!(result.aggregations.count > 0);
    }

    #[test]
    fn test_timeseries_table_aggregations() {
        let mut table = TimeSeriesTable::new("agg_test");

        // Insert known values
        for i in 1..=10 {
            table.insert(i as u64 * 1000, f64::from(i));
        }

        let result = table.query(0, 20_000);
        assert_eq!(result.aggregations.count, 10);
        assert!((result.aggregations.sum - 55.0).abs() < 0.01);
        assert!((result.aggregations.mean - 5.5).abs() < 0.01);
        assert!((result.aggregations.min - 1.0).abs() < 0.01);
        assert!((result.aggregations.max - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_timeseries_table_filtered_query() {
        let mut table = TimeSeriesTable::new("filter_test");

        for i in 1..=20 {
            table.insert(i as u64 * 1000, f64::from(i));
        }

        // Filter to values > 10
        let result = table.query_filtered(0, 30_000, |v| v > 10.0);
        assert!(result.aggregations.count <= 10);
        assert!(result.aggregations.min > 10.0);
    }

    #[test]
    fn test_timeseries_table_windows() {
        let mut table = TimeSeriesTable::new("window_test");

        // Insert 100 samples over 100 seconds
        for i in 0..100 {
            table.insert(i as u64 * 1_000_000, f64::from(i)); // 1 second intervals
        }

        // 10-second windows
        let windows = table.aggregate_windows(0, 100_000_000, 10_000_000);
        assert!(!windows.is_empty());
    }

    #[test]
    fn test_timeseries_table_stats() {
        let mut table = TimeSeriesTable::new("stats_test");

        for i in 0..50 {
            table.insert(i as u64 * 1000, f64::from(i));
        }

        let stats = table.stats();
        assert_eq!(stats.name, "stats_test");
        assert!(stats.hot_samples > 0);
    }

    #[test]
    fn test_timeseries_db_multi_table() {
        let db = TimeSeriesDb::new();

        // Insert into multiple tables
        db.insert("cpu", 1000, 45.0);
        db.insert("cpu", 2000, 50.0);
        db.insert("memory", 1000, 1024.0);
        db.insert("memory", 2000, 2048.0);

        let cpu_result = db.query("cpu", 0, 10000);
        assert!(cpu_result.is_some());

        let mem_result = db.query("memory", 0, 10000);
        assert!(mem_result.is_some());

        let names = db.table_names();
        assert!(names.contains(&"cpu".to_string()));
        assert!(names.contains(&"memory".to_string()));
    }

    #[test]
    fn test_tier_config_default() {
        let config = TierConfig::default();
        assert_eq!(config.hot_duration_us, 5 * 60 * 1_000_000);
        assert_eq!(config.warm_duration_us, 60 * 60 * 1_000_000);
        assert_eq!(config.block_size, 64);
    }

    #[test]
    fn test_aggregations_merge() {
        let mut agg1 =
            Aggregations { count: 5, sum: 15.0, min: 1.0, max: 5.0, mean: 3.0, std_dev: 0.0 };

        let agg2 =
            Aggregations { count: 5, sum: 40.0, min: 6.0, max: 10.0, mean: 8.0, std_dev: 0.0 };

        agg1.merge(&agg2);
        assert_eq!(agg1.count, 10);
        assert!((agg1.sum - 55.0).abs() < 0.01);
        assert!((agg1.min - 1.0).abs() < 0.01);
        assert!((agg1.max - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_query_performance_target() {
        // H₁₁: Query should complete in reasonable time
        let mut table = TimeSeriesTable::new("perf_test");

        // Insert 10,000 samples
        for i in 0..10_000 {
            table.insert(i as u64 * 1000, f64::from(i % 100));
        }

        let start = std::time::Instant::now();
        let result = table.query(0, 10_000_000);
        let elapsed = start.elapsed();

        // Should complete well under 10ms for 10K samples
        assert!(elapsed.as_millis() < 10);
        assert!(result.aggregations.count > 0);
    }

    #[test]
    fn test_persistence_enabled() {
        let temp_dir = std::env::temp_dir().join("tsdb_test");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let mut table = TimeSeriesTable::new("persist_test");
        assert!(table.enable_persistence(&temp_dir).is_ok());

        let stats = table.stats();
        assert!(stats.cold_enabled);

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_timeseries_table_insert_now() {
        let mut table = TimeSeriesTable::new("insert_now_test");

        table.insert_now(42.0);
        table.insert_now(43.0);
        table.insert_now(44.0);

        let stats = table.stats();
        assert!(stats.hot_samples > 0);
    }

    #[test]
    fn test_timeseries_table_name() {
        let table = TimeSeriesTable::new("my_metric");
        assert_eq!(table.name(), "my_metric");
    }

    #[test]
    fn test_table_stats_total_samples() {
        let mut table = TimeSeriesTable::new("total_test");

        for i in 0..10 {
            table.insert(i as u64 * 1000, f64::from(i));
        }

        let stats = table.stats();
        assert!(stats.total_samples() > 0);
    }

    #[test]
    fn test_aggregations_merge_empty() {
        let mut agg1 = Aggregations::default();
        let agg2 =
            Aggregations { count: 5, sum: 25.0, min: 1.0, max: 9.0, mean: 5.0, std_dev: 2.0 };

        agg1.merge(&agg2);
        assert_eq!(agg1.count, 5);
        assert!((agg1.sum - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_aggregations_merge_into_empty() {
        let mut agg1 =
            Aggregations { count: 3, sum: 9.0, min: 1.0, max: 5.0, mean: 3.0, std_dev: 1.5 };
        let agg2 = Aggregations::default();

        agg1.merge(&agg2);
        // Should remain unchanged
        assert_eq!(agg1.count, 3);
    }

    #[test]
    fn test_aggregations_from_empty_samples() {
        let samples: Vec<(Timestamp, f64)> = vec![];
        let agg = Aggregations::from_samples(&samples);
        assert_eq!(agg.count, 0);
    }

    #[test]
    fn test_timeseries_db_default() {
        let db = TimeSeriesDb::default();
        assert!(db.table_names().is_empty());
    }

    #[test]
    fn test_timeseries_db_table_method() {
        let db = TimeSeriesDb::new();

        // Create table
        assert!(db.table("cpu").is_some());

        // Table should exist
        db.insert("cpu", 1000, 50.0);
        let result = db.query("cpu", 0, 2000);
        assert!(result.is_some());
    }

    #[test]
    fn test_timeseries_db_flush() {
        let db = TimeSeriesDb::new();
        db.insert("metric1", 1000, 10.0);
        db.insert("metric2", 2000, 20.0);

        // Flush should complete without panic
        db.flush();
    }

    #[test]
    fn test_timeseries_db_with_persistence() {
        let temp_dir = std::env::temp_dir().join("tsdb_persist_test");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let db = TimeSeriesDb::with_persistence(&temp_dir);
        assert!(db.is_ok());

        let db = db.expect("operation should succeed");
        db.insert("persisted_metric", 1000, 100.0);
        db.flush();

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_custom_tier_config() {
        let config = TierConfig {
            hot_duration_us: 60_000_000,   // 1 minute
            warm_duration_us: 300_000_000, // 5 minutes
            fsync_interval_us: 500_000,    // 0.5 seconds
            block_size: 128,
        };

        let table = TimeSeriesTable::with_config("custom_config", config);
        assert_eq!(table.name(), "custom_config");
    }

    #[test]
    fn test_aggregate_windows_empty() {
        let table = TimeSeriesTable::new("empty_windows");
        let windows = table.aggregate_windows(0, 100_000, 10_000);
        assert!(windows.is_empty());
    }

    #[test]
    fn test_aggregate_windows_zero_size() {
        let mut table = TimeSeriesTable::new("zero_window");
        table.insert(1000, 10.0);

        let windows = table.aggregate_windows(0, 100_000, 0);
        assert!(windows.is_empty());
    }

    #[test]
    fn test_query_result_default() {
        let result = QueryResult::default();
        assert!(result.samples.is_empty());
        assert_eq!(result.aggregations.count, 0);
        assert_eq!(result.tiers_scanned, 0);
    }

    #[test]
    fn test_tier_config_clone() {
        let config = TierConfig::default();
        let cloned = config.clone();
        assert_eq!(config.hot_duration_us, cloned.hot_duration_us);
        assert_eq!(config.warm_duration_us, cloned.warm_duration_us);
    }

    #[test]
    fn test_aggregations_clone() {
        let agg =
            Aggregations { count: 10, sum: 100.0, min: 5.0, max: 15.0, mean: 10.0, std_dev: 2.5 };
        let cloned = agg.clone();
        assert_eq!(agg.count, cloned.count);
        assert!((agg.sum - cloned.sum).abs() < f64::EPSILON);
    }

    #[test]
    fn test_table_stats_clone() {
        let stats = TableStats {
            name: "test".to_string(),
            hot_samples: 100,
            warm_samples: 500,
            cold_samples: 1000,
            warm_compression_ratio: 3.5,
            cold_enabled: true,
        };

        let cloned = stats.clone();
        assert_eq!(stats.name, cloned.name);
        assert_eq!(stats.total_samples(), cloned.total_samples());
    }

    #[test]
    fn test_query_nonexistent_table() {
        let db = TimeSeriesDb::new();
        let result = db.query("nonexistent", 0, 1000);
        assert!(result.is_none());
    }

    #[test]
    fn test_query_range_outside_data() {
        let mut table = TimeSeriesTable::new("range_test");

        table.insert(5000, 10.0);
        table.insert(6000, 20.0);

        // Query outside the data range
        let result = table.query(1000, 2000);
        assert!(result.samples.is_empty() || result.aggregations.count == 0);
    }
}
