//! Core recording methods for BrickProfiler.
//!
//! Extracted from mod.rs to keep file sizes manageable.
//! Contains: start/stop (legacy string API), record_elapsed, record_elapsed_with_bytes,
//! set_brick_bottleneck, stats lookup, all_stats, all_brick_stats, brick_names, reset.

use super::BrickProfiler;
use crate::brick::exec_graph::{BrickBottleneck, BrickId, BrickStats};

impl BrickProfiler {
    /// Start timing a brick. Returns timer handle.
    ///
    /// IMPORTANT: For GPU operations, call sync AFTER the operation
    /// completes but BEFORE calling stop().
    #[must_use]
    pub fn start(&self, name: &str) -> super::BrickTimer {
        super::BrickTimer { name: name.to_string(), start: std::time::Instant::now() }
    }

    /// Stop timing and record the sample.
    ///
    /// # Arguments
    /// - `timer`: Timer handle from `start()`
    /// - `elements`: Number of elements (tokens) processed
    pub fn stop(&mut self, timer: super::BrickTimer, elements: u64) {
        if !self.enabled {
            return;
        }

        let elapsed = timer.start.elapsed();
        let elapsed_ns = elapsed.as_nanos() as u64;

        // PAR-200: Try fast path first if name matches a known BrickId
        if let Some(brick_id) = BrickId::from_str(&timer.name) {
            let stats = &mut self.brick_stats[brick_id as usize];
            stats.add_sample(elapsed_ns, elements);
        } else {
            // Fall back to dynamic stats
            let name = timer.name;
            let stats =
                self.dynamic_stats.entry(name.clone()).or_insert_with(|| BrickStats::new(&name));
            stats.add_sample(elapsed_ns, elements);
        }

        // Update totals
        self.total_tokens += elements;
        self.total_ns += elapsed_ns;
    }

    /// Record a pre-measured duration for a brick.
    ///
    /// PAR-073: This method allows timing with raw `Instant` calls, avoiding
    /// borrow conflicts when profiling CUDA operations that also need `&mut self`.
    ///
    /// # Arguments
    /// - `name`: Brick name
    /// - `elapsed`: Duration of the operation (from `Instant::elapsed()`)
    /// - `elements`: Number of elements (tokens) processed
    ///
    /// # Example
    /// ```rust,ignore
    /// let start = std::time::Instant::now();
    /// cuda_stream.synchronize()?;
    /// self.some_cuda_operation()?;
    /// cuda_stream.synchronize()?;
    /// let elapsed = start.elapsed();
    /// self.profiler.record_elapsed("SomeBrick", elapsed, 1);
    /// ```
    pub fn record_elapsed(&mut self, name: &str, elapsed: std::time::Duration, elements: u64) {
        if !self.enabled {
            return;
        }

        let elapsed_ns = elapsed.as_nanos() as u64;

        // PAR-200: Try fast path first if name matches a known BrickId
        if let Some(brick_id) = BrickId::from_str(name) {
            let stats = &mut self.brick_stats[brick_id as usize];
            stats.add_sample(elapsed_ns, elements);
        } else {
            // Fall back to dynamic stats
            let stats =
                self.dynamic_stats.entry(name.to_string()).or_insert_with(|| BrickStats::new(name));
            stats.add_sample(elapsed_ns, elements);
        }

        // Update totals
        self.total_tokens += elements;
        self.total_ns += elapsed_ns;
    }

    /// PMAT-451: Record elapsed time with byte metrics for compression workloads.
    ///
    /// # Arguments
    /// - `name`: Brick name
    /// - `elapsed`: Duration of the operation
    /// - `elements`: Number of elements (pages) processed
    /// - `input_bytes`: Original uncompressed size
    /// - `output_bytes`: Compressed output size
    ///
    /// # Example
    /// ```rust,ignore
    /// let start = std::time::Instant::now();
    /// let compressed = zstd_compress(&page_data);
    /// let elapsed = start.elapsed();
    /// profiler.record_elapsed_with_bytes(
    ///     "ZstdCompress",
    ///     elapsed,
    ///     1,
    ///     page_data.len() as u64,
    ///     compressed.len() as u64,
    /// );
    /// ```
    pub fn record_elapsed_with_bytes(
        &mut self,
        name: &str,
        elapsed: std::time::Duration,
        elements: u64,
        input_bytes: u64,
        output_bytes: u64,
    ) {
        if !self.enabled {
            return;
        }

        let elapsed_ns = elapsed.as_nanos() as u64;

        // PAR-200: Try fast path first if name matches a known BrickId
        if let Some(brick_id) = BrickId::from_str(name) {
            let stats = &mut self.brick_stats[brick_id as usize];
            stats.add_sample_with_bytes(elapsed_ns, elements, input_bytes, output_bytes);
        } else {
            // Fall back to dynamic stats
            let stats =
                self.dynamic_stats.entry(name.to_string()).or_insert_with(|| BrickStats::new(name));
            stats.add_sample_with_bytes(elapsed_ns, elements, input_bytes, output_bytes);
        }

        // Update totals
        self.total_tokens += elements;
        self.total_ns += elapsed_ns;
    }

    /// PMAT-451: Set bottleneck classification for a brick.
    pub fn set_brick_bottleneck(&mut self, name: &str, bottleneck: BrickBottleneck) {
        // PAR-200: Try fast path first
        if let Some(brick_id) = BrickId::from_str(name) {
            self.brick_stats[brick_id as usize].set_bottleneck(bottleneck);
        } else if let Some(stats) = self.dynamic_stats.get_mut(name) {
            stats.set_bottleneck(bottleneck);
        }
    }

    /// Get statistics for a specific brick by name.
    ///
    /// First checks known BrickId types (O(1)), then falls back to dynamic stats.
    #[must_use]
    pub fn stats(&self, name: &str) -> Option<&BrickStats> {
        // Try fast path first
        if let Some(brick_id) = BrickId::from_str(name) {
            let stats = &self.brick_stats[brick_id as usize];
            if stats.count > 0 {
                return Some(stats);
            }
        }
        // Fall back to dynamic stats
        self.dynamic_stats.get(name)
    }

    /// Get all brick statistics (legacy API, returns dynamic stats only).
    ///
    /// For full statistics including known bricks, use `all_brick_stats()` instead.
    #[must_use]
    #[deprecated(since = "0.12.0", note = "Use all_brick_stats() for complete statistics")]
    pub fn all_stats(&self) -> &std::collections::HashMap<String, BrickStats> {
        &self.dynamic_stats
    }

    /// Get all brick statistics including both known and dynamic bricks.
    pub fn all_brick_stats(&self) -> impl Iterator<Item = &BrickStats> {
        self.brick_stats.iter().filter(|s| s.count > 0).chain(self.dynamic_stats.values())
    }

    /// Get all brick names.
    #[must_use]
    pub fn brick_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .brick_stats
            .iter()
            .enumerate()
            .filter(|(_, s)| s.count > 0)
            .map(|(i, _)| {
                let brick_id = BrickId::ALL[i];
                brick_id.name().to_string()
            })
            .collect();
        names.extend(self.dynamic_stats.keys().cloned());
        names
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        for stats in &mut self.brick_stats {
            stats.count = 0;
            stats.total_ns = 0;
            stats.min_ns = u64::MAX;
            stats.max_ns = 0;
            stats.total_elements = 0;
            stats.total_bytes = 0;
            stats.total_compressed_bytes = 0;
        }
        self.dynamic_stats.clear();
        self.pending.clear();
        self.total_tokens = 0;
        self.total_ns = 0;
    }
}
