
/// ALB-028: Pipeline orchestration subcommands — wraps forjar DAG engine.
///
/// `apr pipeline plan/apply/status/validate` maps to forjar commands,
/// keeping sovereign stack tools decoupled.
#[derive(Subcommand, Debug)]
pub enum PipelineCommands {
    /// Show execution plan — validate manifest, display DAG and resource estimates.
    Plan {
        /// Path to forjar pipeline manifest (YAML)
        #[arg(value_name = "MANIFEST")]
        manifest: PathBuf,

        /// Target specific machine
        #[arg(short, long)]
        machine: Option<String>,

        /// Filter to resources with this tag
        #[arg(short, long)]
        tag: Option<String>,

        /// Show estimated change cost per resource type
        #[arg(long)]
        cost: bool,
    },

    /// Execute the pipeline — converge all resources to desired state.
    Apply {
        /// Path to forjar pipeline manifest (YAML)
        #[arg(value_name = "MANIFEST")]
        manifest: PathBuf,

        /// Target specific machine
        #[arg(short, long)]
        machine: Option<String>,

        /// Filter to resources with this tag
        #[arg(short, long)]
        tag: Option<String>,

        /// Number of parallel SSH sessions (default: 5)
        #[arg(short, long)]
        parallel: Option<u32>,

        /// Continue past failures (best-effort mode)
        #[arg(long)]
        keep_going: bool,
    },

    /// Show current pipeline state — converged, pending, or failed resources.
    Status {
        /// Path to forjar pipeline manifest (YAML)
        #[arg(value_name = "MANIFEST")]
        manifest: PathBuf,
    },

    /// Validate pipeline manifest without connecting to machines.
    Validate {
        /// Path to forjar pipeline manifest (YAML)
        #[arg(value_name = "MANIFEST")]
        manifest: PathBuf,
    },
}
