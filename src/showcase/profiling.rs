
// ============================================================================
// Profiling Integration
// ============================================================================

/// Component timing for profiling
#[derive(Debug, Clone)]
pub struct ComponentTiming {
    /// Component name
    pub name: String,
    /// Total duration
    pub duration: Duration,
    /// Call count
    pub calls: u64,
}

/// Profiling collector for GPU kernel analysis
#[derive(Debug, Default)]
pub struct ProfilingCollector {
    /// Component timings
    timings: Vec<ComponentTiming>,
    /// Start time
    start: Option<Instant>,
}

impl ProfilingCollector {
    /// Create new collector
    pub fn new() -> Self {
        Self::default()
    }

    /// Start profiling
    pub fn start(&mut self) {
        self.start = Some(Instant::now());
    }

    /// Record component timing
    pub fn record(&mut self, name: &str, duration: Duration, calls: u64) {
        self.timings.push(ComponentTiming {
            name: name.to_string(),
            duration,
            calls,
        });
    }

    /// Generate hotspots (>5% of total time)
    pub fn into_hotspots(self) -> Vec<ProfilingHotspot> {
        let total: Duration = self.timings.iter().map(|t| t.duration).sum();
        let total_nanos = total.as_nanos() as f64;

        if total_nanos == 0.0 {
            return Vec::new();
        }

        self.timings
            .into_iter()
            .filter_map(|t| {
                let percentage = (t.duration.as_nanos() as f64 / total_nanos) * 100.0;
                if percentage > 5.0 {
                    let avg_per_call = if t.calls > 0 {
                        Duration::from_nanos((t.duration.as_nanos() / u128::from(t.calls)) as u64)
                    } else {
                        Duration::ZERO
                    };

                    let (explanation, is_expected) = explain_component(&t.name, percentage);

                    Some(ProfilingHotspot {
                        component: t.name,
                        time: t.duration,
                        percentage,
                        call_count: t.calls,
                        avg_per_call,
                        explanation,
                        is_expected,
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Explain profiling component
fn explain_component(name: &str, percentage: f64) -> (String, bool) {
    match name {
        "Q4K_GEMV" | "MatMul" | "GEMM" | "TensorCore" => (
            format!(
                "Matrix ops at {:.1}% - expected for transformer inference",
                percentage
            ),
            true,
        ),
        "Attention" | "FlashAttention" | "IncrementalAttention" => (
            format!(
                "Attention at {:.1}% - normal for autoregressive decode",
                percentage
            ),
            true,
        ),
        "RMSNorm" | "LayerNorm" => {
            if percentage > 15.0 {
                (
                    "Normalization high - megakernel fusion recommended".to_string(),
                    false,
                )
            } else {
                ("Normalization within normal range".to_string(), true)
            }
        }
        "KernelLaunch" => (
            "Kernel launch overhead - CUDA graphs recommended (PAR-037)".to_string(),
            false,
        ),
        "MemcpyH2D" | "MemcpyD2H" | "Transfer" => (
            "Memory transfer - persistent buffers recommended (PAR-038)".to_string(),
            false,
        ),
        "KVCache" | "KV_Cache" => {
            if percentage > 20.0 {
                (
                    "KV cache overhead high - FP16/ZRAM recommended".to_string(),
                    false,
                )
            } else {
                ("KV cache within normal range".to_string(), true)
            }
        }
        "SwiGLU" | "FFN" => (
            format!("FFN at {:.1}% - expected for transformer", percentage),
            true,
        ),
        "Embedding" => (
            "Embedding lookup - expected at inference start".to_string(),
            true,
        ),
        "Sampling" | "TopK" | "TopP" => (
            "Sampling overhead - expected for token generation".to_string(),
            true,
        ),
        _ => {
            if percentage > 20.0 {
                (
                    format!("Unknown at {:.1}% - investigate", percentage),
                    false,
                )
            } else {
                (String::new(), true)
            }
        }
    }
}

// ============================================================================
// PMAT Verification
// ============================================================================

/// PMAT verification result
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // Each bool corresponds to a spec verification point
pub struct PmatVerification {
    /// Point 41: ≥25% faster than llama.cpp
    pub point_41_pass: bool,
    /// Point 42: ≥60 tok/s minimum
    pub point_42_pass: bool,
    /// Point 49: CV <5% consistency
    pub point_49_pass: bool,
    /// 2x Ollama target
    pub ollama_2x_pass: bool,
    /// All checks passed
    pub all_pass: bool,
}

impl PmatVerification {
    /// Verify benchmark results against spec
    pub fn verify(runner: &ShowcaseRunner) -> Self {
        let apr_tps = runner
            .apr_gguf_stats
            .as_ref()
            .or(runner.apr_native_stats.as_ref())
            .map_or(0.0, |s| s.mean_throughput);

        let apr_cv = runner
            .apr_gguf_stats
            .as_ref()
            .or(runner.apr_native_stats.as_ref())
            .map_or(1.0, |s| s.cv);

        let llamacpp_tps = runner
            .llamacpp_stats
            .as_ref()
            .map_or(200.0, |s| s.mean_throughput);

        let ollama_tps = runner
            .ollama_stats
            .as_ref()
            .map_or(318.0, |s| s.mean_throughput);

        let point_41_pass = apr_tps >= llamacpp_tps * 1.25;
        let point_42_pass = apr_tps >= 60.0;
        let point_49_pass = apr_cv < 0.05;
        let ollama_2x_pass = apr_tps >= ollama_tps * 2.0;

        let all_pass = point_41_pass && point_42_pass && point_49_pass;

        Self {
            point_41_pass,
            point_42_pass,
            point_49_pass,
            ollama_2x_pass,
            all_pass,
        }
    }

    /// Generate verification report
    pub fn to_report(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();

        out.push_str("PMAT Verification Results:\n");
        out.push_str("─────────────────────────────────────\n");
        let _ = writeln!(
            out,
            "Point 41 (≥1.25x llama.cpp): {}",
            if self.point_41_pass {
                "✓ PASS"
            } else {
                "✗ FAIL"
            }
        );
        let _ = writeln!(
            out,
            "Point 42 (≥60 tok/s):        {}",
            if self.point_42_pass {
                "✓ PASS"
            } else {
                "✗ FAIL"
            }
        );
        let _ = writeln!(
            out,
            "Point 49 (CV <5%):           {}",
            if self.point_49_pass {
                "✓ PASS"
            } else {
                "✗ FAIL"
            }
        );
        let _ = writeln!(
            out,
            "2x Ollama Target:            {}",
            if self.ollama_2x_pass {
                "✓ PASS"
            } else {
                "○ PENDING"
            }
        );
        out.push_str("─────────────────────────────────────\n");
        let _ = writeln!(
            out,
            "Overall: {}",
            if self.all_pass {
                "✓ ALL PASS"
            } else {
                "✗ NEEDS WORK"
            }
        );

        out
    }
}

// ============================================================================
// Renacer Integration (feature-gated)
// ============================================================================

/// Renacer-based profiler for deep GPU kernel analysis
///
/// When the `showcase-profile` feature is enabled, this wraps renacer's
/// CUDA tracer and time attribution for detailed GPU kernel profiling.
///
/// # Usage
///
/// ```rust,ignore
/// use aprender::showcase::RenacerProfiler;
///
/// let profiler = RenacerProfiler::new()?;
/// profiler.start();
/// // ... run GPU inference ...
/// let hotspots = profiler.finish()?;
/// ```
#[cfg(feature = "showcase-profile")]
pub mod profiler {
    use super::{explain_component, Duration, ProfilingHotspot};
    use renacer::time_attribution::Hotspot;

    /// Renacer-based GPU profiler configuration
    #[derive(Debug, Clone)]
    pub struct RenacerProfilerConfig {
        /// Minimum duration threshold for CUDA kernel tracing (microseconds)
        pub threshold_us: u64,
        /// Whether to trace all kernels (debug mode)
        pub trace_all: bool,
        /// Device ID to trace
        pub device_id: u32,
    }

    impl Default for RenacerProfilerConfig {
        fn default() -> Self {
            Self {
                threshold_us: 100,
                trace_all: false,
                device_id: 0,
            }
        }
    }

    /// Convert renacer hotspots to showcase profiling hotspots
    pub fn convert_hotspots(renacer_hotspots: &[Hotspot]) -> Vec<ProfilingHotspot> {
        renacer_hotspots
            .iter()
            .map(|h| {
                let (explanation, is_expected) = explain_component(&h.cluster, h.percentage);
                ProfilingHotspot {
                    component: h.cluster.clone(),
                    time: h.time,
                    percentage: h.percentage,
                    call_count: 0, // renacer doesn't track call count
                    avg_per_call: Duration::ZERO,
                    explanation,
                    is_expected,
                }
            })
            .collect()
    }

    /// Re-export renacer types for convenience
    pub use renacer::cuda_tracer::CudaTracerConfig;
    pub use renacer::time_attribution::{identify_hotspots, Hotspot as RenacerHotspot};
}

/// Stub module when renacer is not available
#[cfg(not(feature = "showcase-profile"))]
pub mod profiler {
    /// Stub config when showcase-profile is disabled
    #[derive(Debug, Clone, Default)]
    pub struct RenacerProfilerConfig {
        /// Minimum duration threshold
        pub threshold_us: u64,
        /// Trace all kernels
        pub trace_all: bool,
        /// Device ID
        pub device_id: u32,
    }
}
