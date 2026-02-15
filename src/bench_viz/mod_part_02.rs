
// ============================================================================
// Benchmark Grid (2×3)
// ============================================================================

/// 2×3 Benchmark comparison grid with rich visualization
#[derive(Debug, Clone, Default)]
pub struct BenchmarkGrid {
    /// Row 1, Col 1: APR server serving GGUF format
    pub gguf_apr: Option<BenchMeasurement>,
    /// Row 1, Col 2: Ollama serving GGUF format
    pub gguf_ollama: Option<BenchMeasurement>,
    /// Row 1, Col 3: llama.cpp serving GGUF format
    pub gguf_llamacpp: Option<BenchMeasurement>,

    /// Row 2, Col 1: APR server serving native .apr format
    pub apr_native: Option<BenchMeasurement>,
    /// Row 2, Col 2: APR server serving GGUF (for comparison)
    pub apr_gguf: Option<BenchMeasurement>,
    /// Row 2, Col 3: Baseline measurement (Ollama/llama.cpp)
    pub apr_baseline: Option<BenchMeasurement>,

    /// Profiling hotspots
    pub hotspots: Vec<ProfilingHotspot>,

    /// Model name
    pub model_name: String,
    /// Model parameters (e.g., "0.5B")
    pub model_params: String,
    /// Quantization type (e.g., "Q4_K_M")
    pub quantization: String,

    /// GPU name
    pub gpu_name: String,
    /// GPU VRAM in GB
    pub gpu_vram_gb: f64,

    /// Configuration
    pub config: BenchConfig,
}
