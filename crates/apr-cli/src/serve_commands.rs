
/// Inference server subcommands (plan/run).
///
/// `apr serve plan` computes VRAM budget, throughput estimates, and contract
/// verification before starting a server. `apr serve run` launches the server.
#[derive(Subcommand, Debug)]
pub enum ServeCommands {
    /// Pre-flight inference capacity plan (VRAM budget, roofline, contracts)
    ///
    /// Inspects model metadata, detects GPU hardware, and produces a capacity
    /// plan showing whether the model fits in VRAM with the requested batch size.
    /// No weights are loaded — header-only inspection.
    ///
    /// Accepts local files (.gguf, .apr, .safetensors) or HuggingFace repo IDs
    /// (hf://org/repo or org/repo). For HF repos, only the ~2KB config.json is
    /// fetched — no weight download needed.
    Plan {
        /// Model source: local path or HuggingFace repo (hf://org/repo, org/repo)
        #[arg(value_name = "MODEL")]
        model: String,
        /// Detect GPU via nvidia-smi for VRAM budget
        #[arg(long)]
        gpu: bool,
        /// Target batch size for throughput estimation
        #[arg(long, default_value = "1")]
        batch_size: usize,
        /// Sequence length for KV cache estimation
        #[arg(long, default_value = "4096")]
        seq_len: usize,
        /// Output format: text, json, yaml
        #[arg(long, default_value = "text")]
        format: String,
        /// Quantization override for HF models (e.g., Q4_K_M, Q6_K, F16)
        #[arg(long)]
        quant: Option<String>,
    },
    /// Start inference server (REST API, streaming, metrics)
    Run {
        /// Path to model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Disable CORS
        #[arg(long)]
        no_cors: bool,
        /// Disable Prometheus metrics endpoint
        #[arg(long)]
        no_metrics: bool,
        /// Disable GPU acceleration
        #[arg(long)]
        no_gpu: bool,
        /// Force GPU acceleration (requires CUDA)
        #[arg(long)]
        gpu: bool,
        /// Enable batched GPU inference for 2X+ throughput
        #[arg(long)]
        batch: bool,
        /// Enable inference tracing (PMAT-SHOWCASE-METHODOLOGY-001)
        #[arg(long)]
        trace: bool,
        /// Trace detail level (none, basic, layer)
        #[arg(long, value_name = "LEVEL", default_value = "basic")]
        trace_level: String,
        /// Enable inline Roofline profiling (adds X-Profile headers)
        #[arg(long)]
        profile: bool,
    },
}
