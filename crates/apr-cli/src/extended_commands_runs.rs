// Experiment/runs sub-enums (extracted from `extended_commands.rs`
// to keep the PMAT-689 file-size invariant). Inlined via `include!()`
// so item visibility and `use` imports from the parent file are shared.

#[cfg(feature = "training")]
/// Subcommands for `apr runs` — experiment run management (ALB-050/051)
#[derive(Subcommand, Debug)]
pub enum RunsCommands {
    /// List all training experiment runs (with inline loss sparklines)
    Ls {
        /// Directory to scan for experiments (default: current dir)
        #[arg(long, value_name = "DIR")]
        dir: Option<PathBuf>,
        /// Read from global experiment registry (~/.entrenar/experiments.db)
        #[arg(long)]
        global: bool,
        /// Filter by status: running, completed, failed, all
        #[arg(long, default_value = "all")]
        status: String,
        /// Output as JSON
        #[arg(long)]
        json: bool,
        /// Maximum number of runs to show
        #[arg(long, default_value = "50")]
        limit: usize,
    },
    /// Show detailed metrics for a specific run (with braille loss curve)
    Show {
        /// Run ID
        #[arg(value_name = "RUN_ID")]
        run_id: String,
        /// Directory containing experiment DB
        #[arg(long, value_name = "DIR")]
        dir: Option<PathBuf>,
        /// Read from global registry
        #[arg(long)]
        global: bool,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Compare two runs side-by-side (loss curves, config diff, metrics)
    Diff {
        /// First run ID
        #[arg(value_name = "RUN_A")]
        run_a: String,
        /// Second run ID
        #[arg(value_name = "RUN_B")]
        run_b: String,
        /// Directory containing experiment DB
        #[arg(long, value_name = "DIR")]
        dir: Option<PathBuf>,
        /// Read from global registry
        #[arg(long)]
        global: bool,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}

#[cfg(feature = "training")]
/// Subcommands for `apr experiment` — interactive experiment browser (ALB-024)
#[derive(Subcommand, Debug)]
pub enum ExperimentCommands {
    /// Browse experiment history with interactive TUI (loss curves, params)
    View {
        /// Path to experiment database file
        #[arg(long, value_name = "FILE")]
        db: Option<PathBuf>,
        /// Read from global experiment registry (~/.entrenar/experiments.db)
        #[arg(long)]
        global: bool,
        /// Output as JSON (non-interactive)
        #[arg(long)]
        json: bool,
    },
}
