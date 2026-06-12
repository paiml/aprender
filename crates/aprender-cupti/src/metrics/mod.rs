//! GPU metrics collection via CUPTI.
//!
//! Provides access to hardware performance counters including SM utilization,
//! warp efficiency, memory throughput, and instruction mix.

use std::collections::HashMap;

/// Identifier for a CUPTI metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricId {
    /// SM (Streaming Multiprocessor) utilization
    SmUtilization,
    /// Achieved occupancy (warps per SM)
    AchievedOccupancy,
    /// Memory throughput (bytes/sec)
    DramThroughput,
    /// L1 cache hit rate
    L1HitRate,
    /// L2 cache hit rate
    L2HitRate,
    /// Warp execution efficiency
    WarpExecutionEfficiency,
    /// Branch efficiency (non-divergent branches)
    BranchEfficiency,
    /// Global memory load efficiency
    GlobalLoadEfficiency,
    /// Global memory store efficiency
    GlobalStoreEfficiency,
    /// Shared memory load efficiency
    SharedLoadEfficiency,
    /// Shared memory store efficiency
    SharedStoreEfficiency,
    /// Tensor core utilization
    TensorCoreUtilization,
    /// FP32 operations per second
    Flop32,
    /// FP16 operations per second
    Flop16,
    /// Instructions executed per cycle (IPC)
    Ipc,
    /// Custom metric by name
    Custom(u32),
}

impl MetricId {
    /// Get the CUPTI metric name string.
    pub fn cupti_name(&self) -> &'static str {
        match self {
            MetricId::SmUtilization => "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            MetricId::AchievedOccupancy => "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
            MetricId::DramThroughput => "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            MetricId::L1HitRate => "l1tex__t_sector_hit_rate.pct",
            MetricId::L2HitRate => "lts__t_sector_hit_rate.pct",
            MetricId::WarpExecutionEfficiency => "smsp__thread_inst_executed_per_inst_executed.pct",
            MetricId::BranchEfficiency => "smsp__sass_average_branch_targets_threads_uniform.pct",
            MetricId::GlobalLoadEfficiency => "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct_of_peak_sustained_elapsed",
            MetricId::GlobalStoreEfficiency => "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct_of_peak_sustained_elapsed",
            MetricId::SharedLoadEfficiency => "smsp__sass_average_data_bytes_per_sector_mem_shared_op_ld.pct_of_peak_sustained_elapsed",
            MetricId::SharedStoreEfficiency => "smsp__sass_average_data_bytes_per_sector_mem_shared_op_st.pct_of_peak_sustained_elapsed",
            MetricId::TensorCoreUtilization => "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            MetricId::Flop32 => "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum",
            MetricId::Flop16 => "smsp__sass_thread_inst_executed_op_fp16_pred_on.sum",
            MetricId::Ipc => "sm__inst_executed.avg.per_cycle_active",
            MetricId::Custom(_) => "custom",
        }
    }
}

/// Value of a collected metric.
#[derive(Debug, Clone)]
pub enum MetricValue {
    /// Percentage value (0.0-100.0)
    Percent(f64),
    /// Absolute count
    Count(u64),
    /// Rate (per second)
    Rate(f64),
    /// Bytes
    Bytes(u64),
    /// Throughput (bytes/second)
    Throughput(f64),
}

impl MetricValue {
    /// Get as f64 for comparisons.
    pub fn as_f64(&self) -> f64 {
        match self {
            MetricValue::Percent(v) => *v,
            MetricValue::Count(v) => *v as f64,
            MetricValue::Rate(v) => *v,
            MetricValue::Bytes(v) => *v as f64,
            MetricValue::Throughput(v) => *v,
        }
    }

    /// Check if value exceeds threshold.
    pub fn exceeds(&self, threshold: f64) -> bool {
        self.as_f64() > threshold
    }
}

/// SM (Streaming Multiprocessor) metrics.
#[derive(Debug, Clone, Default)]
pub struct SmMetrics {
    /// SM utilization percentage (0-100)
    pub utilization: f64,
    /// Achieved occupancy percentage (0-100)
    pub occupancy: f64,
    /// Active warps per cycle
    pub active_warps: f64,
    /// Active cycles
    pub active_cycles: u64,
    /// Elapsed cycles
    pub elapsed_cycles: u64,
    /// Instructions executed
    pub instructions_executed: u64,
}

impl SmMetrics {
    /// Calculate IPC (instructions per cycle).
    pub fn ipc(&self) -> f64 {
        if self.active_cycles > 0 {
            self.instructions_executed as f64 / self.active_cycles as f64
        } else {
            0.0
        }
    }

    /// Check if SM is underutilized (<50%).
    pub fn is_underutilized(&self) -> bool {
        self.utilization < 50.0
    }
}

/// Warp-level metrics.
#[derive(Debug, Clone, Default)]
pub struct WarpMetrics {
    /// Warp execution efficiency (0-100)
    pub execution_efficiency: f64,
    /// Branch efficiency (0-100)
    pub branch_efficiency: f64,
    /// Warp stall reasons and percentages
    pub stall_reasons: HashMap<WarpStallReason, f64>,
    /// Average active threads per warp (max 32)
    pub active_threads_per_warp: f64,
}

impl WarpMetrics {
    /// Check if there's significant warp divergence.
    pub fn has_divergence(&self) -> bool {
        self.execution_efficiency < 75.0 || self.branch_efficiency < 90.0
    }

    /// Get primary stall reason.
    pub fn primary_stall_reason(&self) -> Option<WarpStallReason> {
        self.stall_reasons
            .iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(reason, _)| *reason)
    }
}

/// Reasons for warp stalls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WarpStallReason {
    /// Waiting for instruction fetch
    InstructionFetch,
    /// Waiting for memory access
    MemoryDependency,
    /// Waiting for execution dependency
    ExecutionDependency,
    /// Synchronization barrier
    Synchronization,
    /// Texture fetch
    TextureFetch,
    /// Other/unknown
    Other,
}

/// Memory metrics.
#[derive(Debug, Clone, Default)]
pub struct MemoryMetrics {
    /// DRAM throughput (GB/s)
    pub dram_throughput: f64,
    /// L1 cache hit rate (0-100)
    pub l1_hit_rate: f64,
    /// L2 cache hit rate (0-100)
    pub l2_hit_rate: f64,
    /// Global load efficiency (0-100)
    pub global_load_efficiency: f64,
    /// Global store efficiency (0-100)
    pub global_store_efficiency: f64,
    /// Shared memory throughput (GB/s)
    pub shared_throughput: f64,
}

impl MemoryMetrics {
    /// Check if memory is a bottleneck (high throughput, low efficiency).
    pub fn is_memory_bound(&self) -> bool {
        // Memory-bound if high DRAM usage but low cache hit rates
        self.dram_throughput > 100.0 && (self.l1_hit_rate < 50.0 || self.l2_hit_rate < 50.0)
    }

    /// Check for coalescing issues.
    pub fn has_coalescing_issues(&self) -> bool {
        self.global_load_efficiency < 80.0 || self.global_store_efficiency < 80.0
    }
}

/// Collected metrics snapshot.
#[derive(Debug, Clone, Default)]
pub struct MetricsSnapshot {
    /// SM metrics
    pub sm: SmMetrics,
    /// Warp metrics
    pub warp: WarpMetrics,
    /// Memory metrics
    pub memory: MemoryMetrics,
    /// Tensor core utilization (0-100)
    pub tensor_core_utilization: f64,
    /// Timestamp when metrics were collected (ns)
    pub timestamp_ns: u64,
}

impl MetricsSnapshot {
    /// Get overall GPU efficiency score (0-100).
    pub fn efficiency_score(&self) -> f64 {
        // Weighted average of key metrics
        let sm_weight = 0.3;
        let occupancy_weight = 0.2;
        let memory_weight = 0.3;
        let warp_weight = 0.2;

        sm_weight * self.sm.utilization
            + occupancy_weight * self.sm.occupancy
            + memory_weight * (self.memory.l1_hit_rate + self.memory.l2_hit_rate) / 2.0
            + warp_weight * self.warp.execution_efficiency
    }

    /// Identify primary performance bottleneck.
    pub fn primary_bottleneck(&self) -> Bottleneck {
        if self.sm.is_underutilized() {
            if self.memory.is_memory_bound() {
                Bottleneck::Memory
            } else if self.warp.has_divergence() {
                Bottleneck::WarpDivergence
            } else {
                Bottleneck::LaunchOverhead
            }
        } else if self.memory.has_coalescing_issues() {
            Bottleneck::MemoryCoalescing
        } else if self.sm.occupancy < 50.0 {
            Bottleneck::Occupancy
        } else {
            Bottleneck::Compute
        }
    }
}

/// GPU performance bottleneck types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottleneck {
    /// Compute-bound (good - GPU is fully utilized)
    Compute,
    /// Memory bandwidth limited
    Memory,
    /// Memory coalescing issues
    MemoryCoalescing,
    /// Low occupancy
    Occupancy,
    /// Warp divergence
    WarpDivergence,
    /// Kernel launch overhead dominates
    LaunchOverhead,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_id_name() {
        assert_eq!(
            MetricId::SmUtilization.cupti_name(),
            "sm__throughput.avg.pct_of_peak_sustained_elapsed"
        );
    }

    #[test]
    fn test_metric_value() {
        let pct = MetricValue::Percent(75.5);
        assert!((pct.as_f64() - 75.5).abs() < 0.001);
        assert!(pct.exceeds(50.0));
        assert!(!pct.exceeds(80.0));
    }

    #[test]
    fn test_sm_metrics_ipc() {
        let sm = SmMetrics {
            active_cycles: 1000,
            instructions_executed: 4000,
            ..Default::default()
        };
        assert!((sm.ipc() - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_warp_divergence() {
        let mut warp = WarpMetrics {
            execution_efficiency: 90.0,
            branch_efficiency: 95.0,
            ..Default::default()
        };
        assert!(!warp.has_divergence());

        warp.execution_efficiency = 60.0;
        assert!(warp.has_divergence());
    }

    #[test]
    fn test_bottleneck_detection() {
        let mut snapshot = MetricsSnapshot::default();
        snapshot.sm.utilization = 30.0;
        snapshot.memory.dram_throughput = 200.0;
        snapshot.memory.l1_hit_rate = 30.0;

        assert_eq!(snapshot.primary_bottleneck(), Bottleneck::Memory);
    }

    #[test]
    fn test_efficiency_score() {
        let mut snapshot = MetricsSnapshot::default();
        snapshot.sm.utilization = 80.0;
        snapshot.sm.occupancy = 70.0;
        snapshot.memory.l1_hit_rate = 60.0;
        snapshot.memory.l2_hit_rate = 80.0;
        snapshot.warp.execution_efficiency = 90.0;

        let score = snapshot.efficiency_score();
        assert!(score > 50.0 && score < 100.0);
    }
}
