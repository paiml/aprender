//! OTLP data types — compute blocks, GPU kernels, memory transfers, metrics snapshots.

use crate::metrics::Registry;

/// Compute block metadata for tracing (Sprint 32)
///
/// Represents a block of statistical computation containing multiple
/// Trueno SIMD operations (e.g., mean, stddev, percentiles).
#[derive(Debug, Clone)]
pub struct ComputeBlock {
    /// Operation name (e.g., "`calculate_statistics`", "`detect_anomalies`")
    pub operation: &'static str,
    /// Total duration of the block in microseconds
    pub duration_us: u64,
    /// Number of elements processed
    pub elements: usize,
    /// Whether this block exceeded the slow threshold (>100μs)
    pub is_slow: bool,
}

/// GPU kernel metadata for tracing (Sprint 37)
///
/// Represents a single GPU kernel execution (compute shader, render pass, etc.)
/// captured via wgpu timestamp queries.
#[derive(Debug, Clone)]
pub struct GpuKernel {
    /// Kernel name (e.g., "`sum_aggregation`", "`matrix_multiply`")
    pub kernel: String,
    /// Total duration in microseconds
    pub duration_us: u64,
    /// GPU backend (always "wgpu" for Phase 1)
    pub backend: &'static str,
    /// Workgroup size for compute shaders (e.g., "`[256,1,1]`")
    pub workgroup_size: Option<String>,
    /// Number of elements processed (if known)
    pub elements: Option<usize>,
    /// Whether this kernel exceeded the slow threshold (>100μs)
    pub is_slow: bool,
}

/// GPU memory transfer direction (Sprint 39 - Phase 4)
///
/// Represents the direction of CPU↔GPU data movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// CPU → GPU (buffer upload, `write_buffer`)
    CpuToGpu,
    /// GPU → CPU (buffer download/readback, `map_async`)
    GpuToCpu,
}

impl TransferDirection {
    /// Get string representation of transfer direction
    pub fn as_str(&self) -> &'static str {
        match self {
            TransferDirection::CpuToGpu => "cpu_to_gpu",
            TransferDirection::GpuToCpu => "gpu_to_cpu",
        }
    }
}

/// GPU memory transfer metadata for tracing (Sprint 39 - Phase 4)
///
/// Represents a single CPU↔GPU memory transfer operation captured via wall-clock timing.
/// Tracks buffer uploads (CPU→GPU) and downloads (GPU→CPU) to identify `PCIe` bandwidth
/// bottlenecks.
#[derive(Debug, Clone)]
pub struct GpuMemoryTransfer {
    /// Transfer name/label (e.g., "`mesh_data_upload`", "`framebuffer_readback`")
    pub label: String,
    /// Transfer direction (CPU→GPU or GPU→CPU)
    pub direction: TransferDirection,
    /// Number of bytes transferred
    pub bytes: usize,
    /// Total duration in microseconds
    pub duration_us: u64,
    /// Calculated bandwidth in MB/s
    pub bandwidth_mbps: f64,
    /// Optional buffer usage hint (e.g., "VERTEX", "UNIFORM", "STORAGE")
    pub buffer_usage: Option<String>,
    /// Whether this transfer exceeded the slow threshold (>100μs)
    pub is_slow: bool,
}

/// Metrics snapshot for OTLP export (Sprint 56)
///
/// Contains all metrics collected at a point in time for export.
#[derive(Debug, Clone, Default)]
pub struct MetricsSnapshot {
    /// Timestamp in nanoseconds since epoch
    pub timestamp_nanos: u64,
    /// Counter metrics
    pub counters: Vec<CounterSnapshot>,
    /// Gauge metrics
    pub gauges: Vec<GaugeSnapshot>,
    /// Histogram metrics
    pub histograms: Vec<HistogramSnapshot>,
}

/// Snapshot of a counter metric
#[derive(Debug, Clone)]
pub struct CounterSnapshot {
    pub name: String,
    pub labels: Vec<(String, String)>,
    pub value: u64,
}

/// Snapshot of a gauge metric
#[derive(Debug, Clone)]
pub struct GaugeSnapshot {
    pub name: String,
    pub labels: Vec<(String, String)>,
    pub value: i64,
}

/// Snapshot of a histogram metric
#[derive(Debug, Clone)]
pub struct HistogramSnapshot {
    pub name: String,
    pub labels: Vec<(String, String)>,
    pub count: u64,
    pub sum: f64,
    pub buckets: Vec<(f64, u64)>, // (le, cumulative_count)
}

impl MetricsSnapshot {
    /// Create a new metrics snapshot from a registry
    pub fn from_registry(registry: &Registry) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp_nanos =
            SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);

        let counters = registry
            .counters()
            .iter()
            .map(|c| CounterSnapshot {
                name: c.name().to_string(),
                labels: c.labels().iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                value: c.get(),
            })
            .collect();

        let gauges = registry
            .gauges()
            .iter()
            .map(|g| GaugeSnapshot {
                name: g.name().to_string(),
                labels: g.labels().iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                value: g.get(),
            })
            .collect();

        let histograms = registry
            .histograms()
            .iter()
            .map(|h| {
                let cumulative = h.cumulative_counts();
                let buckets: Vec<(f64, u64)> = h
                    .buckets()
                    .iter()
                    .zip(cumulative.iter())
                    .map(|(&bound, &count)| (bound, count))
                    .collect();

                HistogramSnapshot {
                    name: h.name().to_string(),
                    labels: h.labels().iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                    count: h.get_count(),
                    sum: h.get_sum(),
                    buckets,
                }
            })
            .collect();

        MetricsSnapshot { timestamp_nanos, counters, gauges, histograms }
    }

    /// Check if snapshot is empty
    pub fn is_empty(&self) -> bool {
        self.counters.is_empty() && self.gauges.is_empty() && self.histograms.is_empty()
    }

    /// Get total metric count
    pub fn len(&self) -> usize {
        self.counters.len() + self.gauges.len() + self.histograms.len()
    }
}

impl GpuMemoryTransfer {
    /// Create a new GPU memory transfer record
    ///
    /// Automatically calculates bandwidth from bytes and duration.
    ///
    /// # Arguments
    ///
    /// * `label` - Transfer name/label
    /// * `direction` - Transfer direction (CPU→GPU or GPU→CPU)
    /// * `bytes` - Number of bytes transferred
    /// * `duration_us` - Transfer duration in microseconds
    /// * `buffer_usage` - Optional buffer usage hint
    /// * `threshold_us` - Slow threshold for adaptive sampling
    ///
    /// # Returns
    ///
    /// New `GpuMemoryTransfer` with calculated bandwidth
    pub fn new(
        label: String,
        direction: TransferDirection,
        bytes: usize,
        duration_us: u64,
        buffer_usage: Option<String>,
        threshold_us: u64,
    ) -> Self {
        // Calculate bandwidth: MB/s = (bytes / 1_048_576) / (duration_us / 1_000_000)
        // Simplified: (bytes * 1_000_000) / (duration_us * 1_048_576)
        let bandwidth_mbps = if duration_us > 0 {
            (bytes as f64 * 1_000_000.0) / (duration_us as f64 * 1_048_576.0)
        } else {
            0.0
        };

        GpuMemoryTransfer {
            label,
            direction,
            bytes,
            duration_us,
            bandwidth_mbps,
            buffer_usage,
            is_slow: duration_us > threshold_us,
        }
    }
}

/// Configuration for OTLP exporter (Sprint 36: added batch config)
#[derive(Debug, Clone)]
pub struct OtlpConfig {
    /// OTLP endpoint URL (e.g., "<http://localhost:4317>")
    pub endpoint: String,
    /// Service name for traces
    pub service_name: String,
    /// Maximum number of spans per batch (default: 512)
    pub batch_size: usize,
    /// Maximum batch delay in milliseconds (default: 1000ms)
    pub batch_delay_ms: u64,
    /// Maximum queue size (default: 2048)
    pub queue_size: usize,
}

impl OtlpConfig {
    /// Create a new OTLP configuration with default batching settings
    pub fn new(endpoint: String, service_name: String) -> Self {
        contract_pre_error_handling!(endpoint);
        OtlpConfig {
            endpoint,
            service_name,
            batch_size: 512,
            batch_delay_ms: 1000,
            queue_size: 2048,
        }
    }

    /// Performance preset: Balanced (default)
    pub fn balanced(endpoint: String, service_name: String) -> Self {
        Self::new(endpoint, service_name)
    }

    /// Performance preset: Aggressive (max throughput)
    pub fn aggressive(endpoint: String, service_name: String) -> Self {
        OtlpConfig {
            endpoint,
            service_name,
            batch_size: 2048,
            batch_delay_ms: 5000,
            queue_size: 8192,
        }
    }

    /// Performance preset: Low-latency (min delay)
    pub fn low_latency(endpoint: String, service_name: String) -> Self {
        OtlpConfig { endpoint, service_name, batch_size: 128, batch_delay_ms: 100, queue_size: 512 }
    }

    /// Set custom batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set custom batch delay
    pub fn with_batch_delay_ms(mut self, delay_ms: u64) -> Self {
        self.batch_delay_ms = delay_ms;
        self
    }

    /// Set custom queue size
    pub fn with_queue_size(mut self, size: usize) -> Self {
        self.queue_size = size;
        self
    }
}
