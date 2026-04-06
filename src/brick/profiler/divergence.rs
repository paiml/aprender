//! Checksum recording and divergence detection for BrickProfiler.
//!
//! CORRECTNESS-011: Per-kernel checksum capture for CPU/GPU divergence detection.
//! Extracted from mod.rs to keep file sizes manageable.

use super::BrickProfiler;
use crate::brick::profiler::checksum::{fnv1a_f32_checksum, DivergenceInfo, KernelChecksum};

impl BrickProfiler {
    // =======================================================================
    // CORRECTNESS-011: Per-kernel checksum capture for divergence detection
    // =======================================================================

    /// Record a kernel trace with output checksum for divergence detection.
    ///
    /// This enables automated CPU/GPU divergence detection by capturing
    /// output checksums alongside timing data. When GPU produces wrong output,
    /// this identifies WHICH kernel diverged without hours of manual debugging.
    ///
    /// Five-Whys Root Cause: Hours of manual "let me check X in Y" debugging
    /// -> No automated tool identified which kernel diverged
    /// -> BrickProfiler only captured timing, not checksums
    /// -> Missing feature: per-kernel checksum capture
    ///
    /// # Arguments
    /// - `name`: Brick/kernel name
    /// - `layer_idx`: Layer index (0-N for transformer layers)
    /// - `position`: Position in sequence
    /// - `output`: Output tensor data (first 64 floats checksummed)
    ///
    /// # Example
    /// ```rust,ignore
    /// // After RoPE kernel
    /// profiler.record_checksum("RopeNeox", layer_idx, position, &q_rotated);
    /// ```
    pub fn record_checksum(&mut self, name: &str, layer_idx: usize, position: u32, output: &[f32]) {
        if !self.enabled {
            return;
        }
        let checksum = fnv1a_f32_checksum(output);
        let trace = KernelChecksum { name: name.to_string(), layer_idx, position, checksum };
        self.kernel_checksums.push(trace);
    }

    /// Get all kernel checksums for divergence comparison.
    #[must_use]
    pub fn get_checksums(&self) -> &[KernelChecksum] {
        &self.kernel_checksums
    }

    /// Compare checksums with a reference profiler (e.g., CPU baseline).
    ///
    /// Returns None if all checksums match, or the first divergent kernel.
    #[must_use]
    pub fn find_divergence(&self, reference: &BrickProfiler) -> Option<DivergenceInfo> {
        use std::collections::HashMap;

        // Index reference checksums by (name, layer_idx, position)
        let ref_index: HashMap<(&str, usize, u32), u64> = reference
            .kernel_checksums
            .iter()
            .map(|t| ((t.name.as_str(), t.layer_idx, t.position), t.checksum))
            .collect();

        // Check each of our checksums against reference
        for trace in &self.kernel_checksums {
            let key = (trace.name.as_str(), trace.layer_idx, trace.position);
            if let Some(&expected) = ref_index.get(&key) {
                if trace.checksum != expected {
                    return Some(DivergenceInfo {
                        kernel_name: trace.name.clone(),
                        layer_idx: trace.layer_idx,
                        position: trace.position,
                        expected_checksum: expected,
                        actual_checksum: trace.checksum,
                    });
                }
            }
        }
        None
    }

    /// Reset checksum tracking (call before new forward pass).
    pub fn reset_checksums(&mut self) {
        self.kernel_checksums.clear();
    }
}
