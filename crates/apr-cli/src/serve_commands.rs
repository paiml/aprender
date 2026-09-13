
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
        ///
        /// Not required with `--list-devices`, which asks what this BUILD can
        /// dispatch to and needs no model to answer.
        #[arg(value_name = "FILE", required_unless_present = "list_devices")]
        file: Option<PathBuf>,
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
        /// Force GPU acceleration (DEPRECATED — use --gpu-layers)
        ///
        /// PERF-021: a boolean accelerator request has no observable
        /// resolution. Honoured and ignored look identical from outside, which
        /// is how #2696 shipped for three releases. Kept so existing scripts
        /// keep working; it means `--gpu-layers all`.
        #[arg(long)]
        gpu: bool,
        /// Layers to offload: a number, `auto`, `all`, or `0`.
        ///
        /// A QUANTITY, not a flag, so the request has a resolution the server
        /// can report: `all` on a model that does not fit is an error, `auto`
        /// offloads what fits and says how many. `auto` is the only value
        /// auto-fit may modify — an explicit number or `all` is a user
        /// instruction and is never lowered silently (I-17).
        ///
        /// Mirrors llama.cpp's `-ngl`, which takes an integer, `auto` or `all`
        /// and reports what it resolved. Neither comparator has a boolean.
        #[arg(long, value_name = "N|auto|all|0")]
        gpu_layers: Option<String>,
        /// List the accelerators this BUILD can dispatch to, then exit.
        ///
        /// Answers "what does this binary actually support" without starting a
        /// server — the question a user with #2696 could not ask.
        #[arg(long)]
        list_devices: bool,
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
        // PMAT-332 / #2583: shared `--backend` declaration (see `BackendArg`).
        // This site used to declare `#[arg(long, value_name = "BACKEND")]` with no
        // `value_parser`, so `--backend nonsense` parsed and the server silently
        // started on the default backend.
        #[command(flatten)]
        backend: BackendArg,
        /// PMAT-485: OTLP endpoint for distributed tracing export (Jaeger/Tempo)
        ///
        /// When set, inference spans (W3C Trace Context) are exported via OTLP.
        /// Each request = parent span, each layer = child span with TensorStats.
        /// Example: --otlp-endpoint http://localhost:4317
        #[arg(long, value_name = "URL")]
        otlp_endpoint: Option<String>,
        /// GH-286: Max context/sequence length for KV cache. Lower = less RSS.
        #[arg(long, default_value = "4096")]
        context_length: usize,
        /// GH-286: Skip FP8 weight cache warmup. Saves ~1.5 GB RSS.
        #[arg(long)]
        no_fp8_cache: bool,
        /// Enable Ollama compatibility mode (port 11434, added endpoints)
        #[arg(long)]
        ollama_compat: bool,
    },
}
