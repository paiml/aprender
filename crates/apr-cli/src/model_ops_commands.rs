
#[derive(Subcommand, Debug)]
pub enum ModelOpsCommands {
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
        /// Fine-tuning task: classify (sequence classification)
        #[arg(long)]
        task: Option<String>,
        /// Number of classes for classification task
        #[arg(long, default_value = "5")]
        num_classes: usize,
        /// Output format for checkpoints: apr, safetensors, or both (comma-separated)
        #[arg(long, value_name = "FORMAT", default_value = "apr,safetensors")]
        checkpoint_format: String,
        /// Oversample minority classes to match majority (for imbalanced datasets)
        #[arg(long)]
        oversample: bool,
        /// Maximum sequence length for GPU buffer allocation (lower = less VRAM)
        #[arg(long, value_name = "LEN")]
        max_seq_len: Option<usize>,
        /// Quantize frozen weights to NF4 (4-bit) for QLoRA training (~8x VRAM savings)
        #[arg(long)]
        quantize_nf4: bool,
        /// GPU indices for data-parallel training (e.g., "0,1" for dual GPU)
        #[arg(long, value_name = "INDICES")]
        gpus: Option<String>,
        /// GPU backend selection: auto, cuda, wgpu
        #[arg(long, default_value = "auto")]
        gpu_backend: String,
        /// Distributed training role: coordinator or worker
        #[arg(long, value_name = "ROLE")]
        role: Option<String>,
        /// Address to bind (coordinator) or connect to (worker)
        #[arg(long, value_name = "ADDR")]
        bind: Option<String>,
        /// Coordinator address for worker nodes (e.g., "intel:9000")
        #[arg(long, value_name = "ADDR")]
        coordinator: Option<String>,
        /// Expected number of workers (coordinator only)
        #[arg(long, value_name = "N")]
        expect_workers: Option<usize>,
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
    /// Knowledge distillation (teacher -> student) (GH-247, ALB-011)
    Distill {
        /// Teacher model file (positional, for file-based mode)
        #[arg(value_name = "TEACHER")]
        teacher: Option<PathBuf>,
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
        /// YAML config file for two-stage distillation (ALB-011)
        #[arg(long, value_name = "FILE")]
        config: Option<PathBuf>,
        /// Distillation stage: precompute (extract teacher logits) or train (student KD)
        #[arg(long, value_name = "STAGE")]
        stage: Option<String>,
    },
}
