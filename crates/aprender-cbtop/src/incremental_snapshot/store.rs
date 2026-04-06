//! Incremental snapshot store with delta compression and retention policies.

use std::collections::HashMap;
use std::time::Instant;

use super::query::{SnapshotConfig, SnapshotQuery};
use super::snapshot::{DeltaSnapshot, ProfileSnapshot, SnapshotIndex};
use super::types::{RetentionTier, SnapshotError, SnapshotResult};

/// Incremental snapshot store
#[derive(Debug)]
pub struct IncrementalSnapshotStore {
    /// Configuration
    config: SnapshotConfig,
    /// Snapshot index
    index: Vec<SnapshotIndex>,
    /// Keyframe snapshots (full snapshots)
    pub(super) keyframes: HashMap<usize, ProfileSnapshot>,
    /// Delta snapshots
    pub(super) deltas: HashMap<usize, DeltaSnapshot>,
    /// Total raw size (uncompressed)
    total_raw_size: usize,
    /// Total compressed size
    total_compressed_size: usize,
    /// Next snapshot index
    next_index: usize,
}

impl IncrementalSnapshotStore {
    /// Create a new snapshot store
    pub fn new(config: SnapshotConfig) -> Self {
        Self {
            config,
            index: Vec::new(),
            keyframes: HashMap::new(),
            deltas: HashMap::new(),
            total_raw_size: 0,
            total_compressed_size: 0,
            next_index: 0,
        }
    }

    /// Append a new snapshot
    pub fn append(&mut self, mut snapshot: ProfileSnapshot) -> SnapshotResult<usize> {
        snapshot.index = self.next_index;
        snapshot.compute_checksum();

        let raw_size = snapshot.size_bytes();
        self.total_raw_size += raw_size;

        // Decide if this should be a keyframe
        let is_keyframe = self.next_index.is_multiple_of(self.config.keyframe_interval)
            || self.keyframes.is_empty();

        let (compressed_size, is_delta, base_index) = if is_keyframe {
            // Store as keyframe
            let size = snapshot.size_bytes();
            self.keyframes.insert(self.next_index, snapshot.clone());
            (size, false, None)
        } else {
            // Find nearest keyframe
            let keyframe_idx =
                (self.next_index / self.config.keyframe_interval) * self.config.keyframe_interval;

            if let Some(base) = self.keyframes.get(&keyframe_idx) {
                // Create delta from keyframe
                let delta = DeltaSnapshot::from_diff(base, &snapshot);
                let size = delta.size_bytes();
                self.deltas.insert(self.next_index, delta);
                (size, true, Some(keyframe_idx))
            } else {
                // No keyframe found, store as keyframe
                let size = snapshot.size_bytes();
                self.keyframes.insert(self.next_index, snapshot.clone());
                (size, false, None)
            }
        };

        self.total_compressed_size += compressed_size;

        // Add index entry
        self.index.push(SnapshotIndex {
            index: self.next_index,
            timestamp_ns: snapshot.timestamp_ns,
            fingerprint: snapshot.workload_fingerprint.clone(),
            tier: RetentionTier::Raw,
            offset: 0, // In-memory, no offset
            size_bytes: compressed_size,
            is_delta,
            base_index,
        });

        let idx = self.next_index;
        self.next_index += 1;

        Ok(idx)
    }

    /// Get snapshot by index
    pub fn get(&self, index: usize) -> SnapshotResult<ProfileSnapshot> {
        if index >= self.next_index {
            return Err(SnapshotError::IndexOutOfBounds {
                index,
                max: self.next_index.saturating_sub(1),
            });
        }

        // Check if it's a keyframe
        if let Some(snapshot) = self.keyframes.get(&index) {
            return Ok(snapshot.clone());
        }

        // It's a delta, need to reconstruct
        if let Some(delta) = self.deltas.get(&index) {
            if let Some(base) = self.keyframes.get(&delta.base_index) {
                let reconstructed = delta.apply_to(base);

                if self.config.verify_checksums && !reconstructed.verify_checksum() {
                    return Err(SnapshotError::Corrupt {
                        reason: "Checksum verification failed".to_string(),
                    });
                }

                return Ok(reconstructed);
            }
        }

        Err(SnapshotError::NotFound { index })
    }

    /// Query snapshots
    pub fn query(&self, query: &SnapshotQuery) -> SnapshotResult<Vec<ProfileSnapshot>> {
        let mut results = Vec::new();
        let mut memory_used = 0;

        for idx_entry in &self.index {
            if !query.matches(idx_entry) {
                continue;
            }

            // Check memory limit
            if memory_used + idx_entry.size_bytes > self.config.max_query_memory_bytes {
                break;
            }

            let snapshot = self.get(idx_entry.index)?;

            // Check metric filter
            if let Some(ref metric_name) = query.metric_name {
                if !snapshot.metrics.contains_key(metric_name) {
                    continue;
                }
            }

            memory_used += idx_entry.size_bytes;
            results.push(snapshot);

            // Check limit
            if let Some(limit) = query.limit {
                if results.len() >= limit {
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.total_raw_size == 0 {
            1.0
        } else {
            self.total_compressed_size as f64 / self.total_raw_size as f64
        }
    }

    /// Get snapshot count
    pub fn count(&self) -> usize {
        self.next_index
    }

    /// Get total raw size
    pub fn total_raw_size(&self) -> usize {
        self.total_raw_size
    }

    /// Get total compressed size
    pub fn total_compressed_size(&self) -> usize {
        self.total_compressed_size
    }

    /// Clean up old snapshots based on retention policy
    pub fn cleanup(&mut self, _now: Instant, reference_time: u64) {
        let _raw_cutoff = reference_time.saturating_sub(self.config.raw_max_age.as_nanos() as u64);
        let compressed_cutoff =
            reference_time.saturating_sub(self.config.compressed_max_age.as_nanos() as u64);

        // Update tiers and remove expired
        self.index.retain(|idx| {
            if idx.timestamp_ns < compressed_cutoff {
                // Remove from storage
                self.keyframes.remove(&idx.index);
                self.deltas.remove(&idx.index);
                return false;
            }
            true
        });
    }

    /// Get configuration
    pub fn config(&self) -> &SnapshotConfig {
        &self.config
    }

    /// Get index entries
    pub fn index(&self) -> &[SnapshotIndex] {
        &self.index
    }
}
