//! Muda (Waste) Detection Engine — Seven categories of GPU compute waste.
//! Mapped from Toyota Production System (Ohno, 1988) [7].

use serde::{Deserialize, Serialize};

/// Seven Muda of GPU Compute, plus CPU-specific waste categories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuMuda {
    /// Muda of Transport: Data moved unnecessarily.
    /// Examples: register spills, redundant L2 traffic, unnecessary H2D copies.
    Transport { register_spills: u64, unnecessary_global_loads: u64, redundant_shared_stores: u64 },

    /// Muda of Waiting: Hardware resources idle.
    /// Examples: barrier stalls, memory latency not hidden, pipeline bubbles.
    Waiting {
        barrier_stall_cycles: u64,
        memory_stall_cycles: u64,
        pipeline_bubbles: u64,
        warp_scheduler_idle_pct: f64,
    },

    /// Muda of Overprocessing: More work than necessary.
    /// Examples: FP32 when FP16 suffices, unneeded boundary checks, redundant instructions.
    Overprocessing {
        precision_waste_pct: f64,
        redundant_instructions: u64,
        unnecessary_bounds_checks: u64,
    },

    /// Muda of Inventory: Resources allocated but unused.
    /// Examples: shared memory allocated but not used, registers reserved but unused.
    Inventory {
        unused_shared_memory_bytes: u64,
        unused_registers_per_thread: u32,
        occupancy_loss_pct: f64,
    },

    /// Muda of Motion: Excessive control flow.
    /// Examples: warp divergence, branch overhead, loop overhead.
    Motion { divergent_branches: u64, branch_efficiency_pct: f64, loop_overhead_cycles: u64 },

    /// Muda of Defects: Incorrect results requiring rework.
    /// Examples: NaN propagation, precision loss, numerical instability.
    Defects { nan_count: u64, inf_count: u64, precision_loss_bits: f64 },

    /// Muda of Overproduction: Computing results that aren't needed.
    /// Examples: padding waste, inactive threads in partial tiles.
    Overproduction { padding_waste_pct: f64, inactive_thread_pct: f64, unused_output_elements: u64 },
}

/// Detected waste with severity and recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MudaDetection {
    pub muda: GpuMuda,
    /// Impact: estimated percentage of execution time wasted.
    pub impact_pct: f64,
    /// Human-readable description of the waste.
    pub description: String,
    /// Actionable recommendation.
    pub recommendation: String,
}

/// Analyze a kernel profile for all seven Muda categories.
#[derive(Default)]
pub struct MudaDetector {
    /// Thresholds for each waste category.
    pub thresholds: MudaThresholds,
}

/// Configurable thresholds for Muda detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MudaThresholds {
    /// Register spills > this count triggers Transport muda.
    pub max_register_spills: u64,
    /// Warp scheduler idle > this percentage triggers Waiting muda.
    pub max_scheduler_idle_pct: f64,
    /// Branch efficiency < this percentage triggers Motion muda.
    pub min_branch_efficiency_pct: f64,
    /// Occupancy loss > this percentage triggers Inventory muda.
    pub max_occupancy_loss_pct: f64,
    /// Padding waste > this percentage triggers Overproduction muda.
    pub max_padding_waste_pct: f64,
    /// Any NaN/Inf triggers Defects muda.
    pub max_nan_inf_count: u64,
    /// Precision waste > this percentage triggers Overprocessing muda.
    pub max_precision_waste_pct: f64,
}

impl Default for MudaThresholds {
    fn default() -> Self {
        Self {
            max_register_spills: 0,
            max_scheduler_idle_pct: 20.0,
            min_branch_efficiency_pct: 90.0,
            max_occupancy_loss_pct: 50.0,
            max_padding_waste_pct: 10.0,
            max_nan_inf_count: 0,
            max_precision_waste_pct: 25.0,
        }
    }
}

impl MudaDetector {
    pub fn new() -> Self {
        Self { thresholds: MudaThresholds::default() }
    }

    pub fn with_thresholds(thresholds: MudaThresholds) -> Self {
        Self { thresholds }
    }

    /// Detect Transport muda from register spill and memory traffic data.
    pub fn detect_transport(
        &self,
        register_spills: u64,
        unnecessary_global_loads: u64,
        redundant_shared_stores: u64,
    ) -> Option<MudaDetection> {
        if register_spills > self.thresholds.max_register_spills
            || unnecessary_global_loads > 0
            || redundant_shared_stores > 0
        {
            let total_waste = register_spills + unnecessary_global_loads + redundant_shared_stores;
            Some(MudaDetection {
                muda: GpuMuda::Transport {
                    register_spills,
                    unnecessary_global_loads,
                    redundant_shared_stores,
                },
                impact_pct: (total_waste as f64).min(100.0),
                description: format!(
                    "Data movement waste: {register_spills} register spills, \
                     {unnecessary_global_loads} unnecessary global loads, \
                     {redundant_shared_stores} redundant shared stores"
                ),
                recommendation: if register_spills > 0 {
                    "Reduce register pressure: decrease tile size, use shared memory, or reduce live variables".to_string()
                } else {
                    "Review memory access patterns for redundant loads/stores".to_string()
                },
            })
        } else {
            None
        }
    }

    /// Detect Waiting muda from stall cycle data.
    pub fn detect_waiting(
        &self,
        barrier_stall_cycles: u64,
        memory_stall_cycles: u64,
        pipeline_bubbles: u64,
        warp_scheduler_idle_pct: f64,
    ) -> Option<MudaDetection> {
        if warp_scheduler_idle_pct > self.thresholds.max_scheduler_idle_pct
            || barrier_stall_cycles > 0
            || memory_stall_cycles > 0
        {
            let impact =
                warp_scheduler_idle_pct.max(if memory_stall_cycles > 0 { 10.0 } else { 0.0 });
            Some(MudaDetection {
                muda: GpuMuda::Waiting {
                    barrier_stall_cycles,
                    memory_stall_cycles,
                    pipeline_bubbles,
                    warp_scheduler_idle_pct,
                },
                impact_pct: impact,
                description: format!(
                    "Hardware idle: scheduler {warp_scheduler_idle_pct:.1}% idle, \
                     {memory_stall_cycles} memory stall cycles, \
                     {barrier_stall_cycles} barrier stall cycles"
                ),
                recommendation: if memory_stall_cycles > barrier_stall_cycles {
                    "Increase warps per SM for latency hiding, or improve data locality".to_string()
                } else {
                    "Reduce barrier synchronization or overlap compute with data movement"
                        .to_string()
                },
            })
        } else {
            None
        }
    }

    /// Detect Motion muda from branch divergence data.
    pub fn detect_motion(
        &self,
        divergent_branches: u64,
        branch_efficiency_pct: f64,
        loop_overhead_cycles: u64,
    ) -> Option<MudaDetection> {
        if branch_efficiency_pct < self.thresholds.min_branch_efficiency_pct
            || divergent_branches > 0
        {
            Some(MudaDetection {
                muda: GpuMuda::Motion {
                    divergent_branches,
                    branch_efficiency_pct,
                    loop_overhead_cycles,
                },
                impact_pct: 100.0 - branch_efficiency_pct,
                description: format!(
                    "Control flow waste: {divergent_branches} divergent branches, \
                     {branch_efficiency_pct:.1}% branch efficiency"
                ),
                recommendation:
                    "Ensure warp-uniform branching; move data-dependent branches outside warp"
                        .to_string(),
            })
        } else {
            None
        }
    }

    /// Detect Inventory muda from resource allocation data.
    pub fn detect_inventory(
        &self,
        unused_shared_memory_bytes: u64,
        unused_registers_per_thread: u32,
        occupancy_loss_pct: f64,
    ) -> Option<MudaDetection> {
        if occupancy_loss_pct > self.thresholds.max_occupancy_loss_pct
            || unused_shared_memory_bytes > 0
            || unused_registers_per_thread > 0
        {
            Some(MudaDetection {
                muda: GpuMuda::Inventory {
                    unused_shared_memory_bytes,
                    unused_registers_per_thread,
                    occupancy_loss_pct,
                },
                impact_pct: occupancy_loss_pct,
                description: format!(
                    "Resource waste: {unused_shared_memory_bytes} bytes unused smem, \
                     {unused_registers_per_thread} unused regs/thread, \
                     {occupancy_loss_pct:.1}% occupancy loss"
                ),
                recommendation: "Reduce shared memory or register allocation to improve occupancy"
                    .to_string(),
            })
        } else {
            None
        }
    }

    /// Detect Defects muda from numerical error data.
    pub fn detect_defects(
        &self,
        nan_count: u64,
        inf_count: u64,
        precision_loss_bits: f64,
    ) -> Option<MudaDetection> {
        if nan_count > self.thresholds.max_nan_inf_count
            || inf_count > self.thresholds.max_nan_inf_count
            || precision_loss_bits > 1.0
        {
            Some(MudaDetection {
                muda: GpuMuda::Defects { nan_count, inf_count, precision_loss_bits },
                impact_pct: if nan_count > 0 || inf_count > 0 {
                    100.0
                } else {
                    precision_loss_bits * 10.0
                },
                description: format!(
                    "Numerical defects: {nan_count} NaN, {inf_count} Inf, \
                     {precision_loss_bits:.1} bits precision loss"
                ),
                recommendation: if nan_count > 0 {
                    "Investigate NaN source: likely division by zero or log(negative)".to_string()
                } else {
                    "Consider using higher precision for accumulation".to_string()
                },
            })
        } else {
            None
        }
    }

    /// Detect Overproduction muda from padding/inactive thread data.
    pub fn detect_overproduction(
        &self,
        padding_waste_pct: f64,
        inactive_thread_pct: f64,
        unused_output_elements: u64,
    ) -> Option<MudaDetection> {
        if padding_waste_pct > self.thresholds.max_padding_waste_pct
            || inactive_thread_pct > self.thresholds.max_padding_waste_pct
        {
            Some(MudaDetection {
                muda: GpuMuda::Overproduction {
                    padding_waste_pct,
                    inactive_thread_pct,
                    unused_output_elements,
                },
                impact_pct: padding_waste_pct.max(inactive_thread_pct),
                description: format!(
                    "Overproduction: {padding_waste_pct:.1}% padding waste, \
                     {inactive_thread_pct:.1}% inactive threads"
                ),
                recommendation: "Adjust tile size to match problem dimensions; use predication for partial tiles".to_string(),
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_register_spills() {
        let detector = MudaDetector::new();
        let result = detector.detect_transport(5, 0, 0);
        assert!(result.is_some());
        let detection = result.unwrap();
        assert!(matches!(detection.muda, GpuMuda::Transport { register_spills: 5, .. }));
    }

    #[test]
    fn test_no_transport_waste() {
        let detector = MudaDetector::new();
        let result = detector.detect_transport(0, 0, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_detect_warp_divergence() {
        let detector = MudaDetector::new();
        let result = detector.detect_motion(10, 75.0, 100);
        assert!(result.is_some());
        let detection = result.unwrap();
        assert!(matches!(detection.muda, GpuMuda::Motion { divergent_branches: 10, .. }));
    }

    #[test]
    fn test_detect_nan_defects() {
        let detector = MudaDetector::new();
        let result = detector.detect_defects(3, 0, 0.0);
        assert!(result.is_some());
        assert_eq!(result.unwrap().impact_pct, 100.0);
    }

    #[test]
    fn test_no_defects_clean() {
        let detector = MudaDetector::new();
        let result = detector.detect_defects(0, 0, 0.5);
        assert!(result.is_none());
    }

    #[test]
    fn test_detect_overproduction() {
        let detector = MudaDetector::new();
        let result = detector.detect_overproduction(25.0, 15.0, 1024);
        assert!(result.is_some());
        assert_eq!(result.unwrap().impact_pct, 25.0);
    }

    #[test]
    fn test_custom_thresholds() {
        let thresholds = MudaThresholds { max_register_spills: 10, ..Default::default() };
        let detector = MudaDetector::with_thresholds(thresholds);
        // 5 spills should NOT trigger with threshold 10
        let result = detector.detect_transport(5, 0, 0);
        assert!(result.is_none());
        // 11 spills should trigger
        let result = detector.detect_transport(11, 0, 0);
        assert!(result.is_some());
    }
}
