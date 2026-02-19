    /// Manage canary tests for regression
    Canary {
        #[command(subcommand)]
        command: CanaryCommands,
    },
    /// Export model to other formats
    Export {
        /// Path to .apr model file
        #[arg(value_name = "FILE", required_unless_present = "list_formats")]
        file: Option<PathBuf>,
        /// Output format (safetensors, gguf, mlx, onnx, openvino, coreml)
        #[arg(long, default_value = "safetensors")]
        format: String,
        /// Output file/directory path
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Apply quantization during export (int8, int4, fp16)
        #[arg(long)]
        quantize: Option<String>,
        /// List all supported export formats
        #[arg(long)]
        list_formats: bool,
        /// Batch export to multiple formats (comma-separated: gguf,mlx,safetensors)
        #[arg(long)]
        batch: Option<String>,
        /// Output in JSON format
        #[arg(long)]
        json: bool,
    },
    /// Import from external formats (hf://org/repo, local files, URLs)
    Import {
        /// Source: hf://org/repo, local file, or URL
        #[arg(value_name = "SOURCE")]
        source: String,
        /// Output .apr file path (default: derived from source name)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Model architecture (whisper, llama, bert, auto)
        #[arg(long, default_value = "auto")]
        arch: String,
        /// Quantization (int8, int4, fp16)
        #[arg(long)]
        quantize: Option<String>,
        /// Strict mode: reject unverified architectures and fail on validation errors
        #[arg(long)]
        strict: bool,
        /// Preserve Q4K quantization for fused kernel inference (GGUF only)
        /// Uses realizar's Q4K converter instead of dequantizing to F32
        #[arg(long)]
        preserve_q4k: bool,
        /// PMAT-232: External tokenizer.json for weights-only GGUF files.
        /// Required if the GGUF has no embedded tokenizer vocabulary.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// F-GT-001: Enforce provenance chain. Rejects pre-baked GGUF imports
        /// (only SafeTensors sources allowed). Ensures single-provenance testing.
        #[arg(long)]
        enforce_provenance: bool,
        /// GH-223: Allow import without config.json (default: error).
        /// Without config.json, hyperparameters like rope_theta are inferred from
        /// tensor shapes and may be wrong, producing garbage output.
        #[arg(long)]
        allow_no_config: bool,
    },
    /// Download and cache model from HuggingFace (Ollama-like UX)
    Pull {
        /// Model reference (alias, hf:// URI, or org/repo)
        #[arg(value_name = "MODEL")]
        model_ref: String,
        /// Force re-download even if cached
        #[arg(long)]
        force: bool,
    },
    /// List cached models
    #[command(name = "list", alias = "ls")]
    List,
    /// Remove model from cache
    #[command(name = "rm", alias = "remove")]
    Rm {
        /// Model reference to remove
        #[arg(value_name = "MODEL")]
        model_ref: String,
    },
    /// Convert/optimize model
    Convert {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Quantize to format (int8, int4, fp16, q4k)
        #[arg(long)]
        quantize: Option<String>,
        /// Compress output (none, zstd, zstd-max, lz4)
        #[arg(long)]
        compress: Option<String>,
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
        /// Force overwrite existing files
        #[arg(short, long)]
        force: bool,
    },
    /// Merge multiple models
    Merge {
        /// Model files to merge
        #[arg(value_name = "FILES", num_args = 2..)]
        files: Vec<PathBuf>,
        /// Merge strategy (average, weighted, slerp, ties, dare)
        #[arg(long, default_value = "average")]
        strategy: String,
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
        /// Weights for weighted merge (comma-separated, e.g., "0.7,0.3")
        #[arg(long, value_delimiter = ',')]
        weights: Option<Vec<f32>>,
        /// Base model for TIES/DARE (task vectors computed as delta from base)
        #[arg(long)]
        base_model: Option<PathBuf>,
        /// DARE drop probability (default: 0.9)
        #[arg(long, default_value = "0.9")]
        drop_rate: f32,
        /// TIES trim density threshold (default: 0.2)
        #[arg(long, default_value = "0.2")]
        density: f32,
        /// RNG seed for DARE (default: 42)
        #[arg(long, default_value = "42")]
        seed: u64,
    },
    /// Quantize model weights (GH-243)
    Quantize {
        /// Input model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Quantization scheme: int8, int4, fp16, q4k
        #[arg(long, short = 's', default_value = "int4")]
        scheme: String,
        /// Output file path (required unless --plan)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Output format override (apr, gguf, safetensors)
        #[arg(long)]
        format: Option<String>,
        /// Batch quantization (comma-separated schemes)
        #[arg(long)]
        batch: Option<String>,
        /// Plan mode (estimate only, no execution)
        #[arg(long)]
        plan: bool,
        /// Force overwrite existing files
        #[arg(short, long)]
        force: bool,
    },
    /// Fine-tune model with LoRA/QLoRA (GH-244)
    Finetune {
        /// Input model file
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
        /// Fine-tuning method: auto, full, lora, qlora
        #[arg(long, short = 'm', default_value = "auto")]
        method: String,
        /// LoRA rank (default: auto-selected)
        #[arg(long, short = 'r')]
        rank: Option<u32>,
        /// Available VRAM in GB
        #[arg(long, default_value = "16.0")]
        vram: f64,
        /// Plan mode (estimate only)
        #[arg(long)]
        plan: bool,
        /// Training data file (JSONL format)
        #[arg(long, short = 'd', value_name = "FILE")]
        data: Option<PathBuf>,
        /// Output path (adapter dir or merged model)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Adapter path for merge mode
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// Merge adapter into base model
        #[arg(long)]
        merge: bool,
        /// Training epochs
        #[arg(long, default_value = "3")]
        epochs: u32,
        /// Learning rate
        #[arg(long, default_value = "0.0002")]
        learning_rate: f64,
        /// Model size for planning (e.g., "7B", "1.5B")
        #[arg(long, value_name = "SIZE")]
        model_size: Option<String>,
    },
    /// Prune model (structured/unstructured pruning) (GH-247)
    Prune {
        /// Input model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Pruning method: magnitude, structured, depth, width, wanda, sparsegpt
        #[arg(long, short = 'm', default_value = "magnitude")]
        method: String,
        /// Target pruning ratio (0-1)
        #[arg(long, default_value = "0.5")]
        target_ratio: f32,
        /// Sparsity level (0-1)
        #[arg(long, default_value = "0.0")]
        sparsity: f32,
        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Layers to remove for depth pruning (e.g., "20-24")
        #[arg(long)]
        remove_layers: Option<String>,
        /// Analyze mode (identify pruning opportunities)
        #[arg(long)]
        analyze: bool,
        /// Plan mode (estimate only)
        #[arg(long)]
        plan: bool,
        /// Calibration data file
        #[arg(long, value_name = "FILE")]
        calibration: Option<PathBuf>,
    },
    /// Knowledge distillation (teacher -> student) (GH-247)
    Distill {
        /// Teacher model file
        #[arg(value_name = "TEACHER")]
        teacher: PathBuf,
        /// Student model file
        #[arg(long, value_name = "FILE")]
        student: Option<PathBuf>,
        /// Training data file
        #[arg(long, short = 'd', value_name = "FILE")]
        data: Option<PathBuf>,
        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Distillation strategy: standard, progressive, ensemble
        #[arg(long, default_value = "standard")]
        strategy: String,
        /// Temperature for softmax scaling
        #[arg(long, default_value = "3.0")]
        temperature: f64,
        /// Alpha weight for KL vs task loss
        #[arg(long, default_value = "0.7")]
        alpha: f64,
        /// Training epochs
        #[arg(long, default_value = "3")]
        epochs: u32,
        /// Plan mode (estimate only)
        #[arg(long)]
        plan: bool,
    },
    /// Interactive terminal UI
    Tui {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: Option<PathBuf>,
    },
    /// Model self-test: 10-stage pipeline integrity check (APR-TRACE-001)
    Check {
        /// Path to model file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Disable GPU acceleration
        #[arg(long)]
        no_gpu: bool,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Extended analysis, profiling, QA, and visualization commands
    #[command(flatten)]
    Extended(ExtendedCommands),
}
