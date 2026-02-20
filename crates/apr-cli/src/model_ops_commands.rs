
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
}
