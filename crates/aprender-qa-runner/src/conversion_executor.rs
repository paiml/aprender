/// Configuration for conversion executor
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct ConversionConfig {
    /// Test all format pairs
    pub test_all_pairs: bool,
    /// Test round-trips
    pub test_round_trips: bool,
    /// Test multi-hop conversion chains (T-QKV-04)
    pub test_multi_hop: bool,
    /// Test tensor cardinality after conversion (MR-CARD)
    pub test_cardinality: bool,
    /// Test tensor name preservation after conversion (T-QKV-02)
    pub test_tensor_names: bool,
    /// Test idempotency of double-conversion (MR-IDEM)
    pub test_idempotency: bool,
    /// Test commutativity of conversion paths (MR-COM)
    pub test_commutativity: bool,
    /// Backends to test
    pub backends: Vec<Backend>,
    /// Use CPU only (no GPU)
    pub no_gpu: bool,
}

impl Default for ConversionConfig {
    /// Create a default config with all test types enabled on CPU and GPU
    fn default() -> Self {
        Self {
            test_all_pairs: true,
            test_round_trips: true,
            test_multi_hop: true,
            test_cardinality: true,
            test_tensor_names: true,
            test_idempotency: true,
            test_commutativity: true,
            backends: vec![Backend::Cpu, Backend::Gpu],
            no_gpu: false,
        }
    }
}

/// Configuration constructors for conversion test scenarios
impl ConversionConfig {
    /// Create config for CPU-only testing
    #[must_use]
    pub fn cpu_only() -> Self {
        Self {
            test_all_pairs: true,
            test_round_trips: true,
            test_multi_hop: true,
            test_cardinality: true,
            test_tensor_names: true,
            test_idempotency: true,
            test_commutativity: true,
            backends: vec![Backend::Cpu],
            no_gpu: true,
        }
    }
}

/// Executor for running P0 format conversion tests
#[derive(Debug)]
pub struct ConversionExecutor {
    /// Conversion test configuration flags
    config: ConversionConfig,
    /// Path to the conversion binary executable
    binary: String,
    /// Output directory for conversion artifacts (ISO-OUT-001)
    output_dir: Option<PathBuf>,
}

include!("conversion_executor_impl.rs");
include!("conversion_execution_result.rs");
