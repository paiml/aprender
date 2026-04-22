// CRUX-series lint subcommands (extracted from `extended_commands.rs`
// to keep the PMAT-689 file-size invariant).
//
// `LintCommands` is flattened into `ExtendedCommands` via
// `#[command(flatten)]` so each variant is still invoked top-level
// (`apr ollama-chat-lint`, `apr awq-lint`, ...).

/// CRUX-series lint subcommands (Ollama/DRY/AWQ/OOM/tool-use/GBNF).
#[derive(Subcommand, Debug)]
pub enum LintCommands {
    /// Lint an Ollama /api/chat response for schema + NDJSON invariants (CRUX-C-04)
    OllamaChatLint {
        /// Path to captured /api/chat response (JSON object, or NDJSON if --stream)
        #[arg(long, value_name = "FILE")]
        response_file: PathBuf,
        /// Treat input as NDJSON stream (one frame per line)
        #[arg(long)]
        stream: bool,
    },
    /// Lint a captured DRY-sampling observation (CRUX-C-23)
    DrySamplingLint {
        /// Path to observation JSON
        #[arg(long, value_name = "FILE")]
        observation_file: PathBuf,
    },
    /// Lint a captured AWQ quality/compression/flags observation (CRUX-B-08)
    AwqLint {
        /// Path to captured AWQ observation JSON
        #[arg(long, value_name = "FILE")]
        observation_file: PathBuf,
    },
    /// Lint a captured CUDA OOM postmortem report (CRUX-F-13)
    OomLint {
        /// Path to captured OOM postmortem JSON (e.g. /tmp/apr-oom-<ts>.json)
        #[arg(long, value_name = "FILE")]
        report_file: PathBuf,
        /// Optional captured stderr log to verify the OOM_REPORT breadcrumb
        #[arg(long, value_name = "FILE")]
        stderr_file: Option<PathBuf>,
    },
    /// Lint a captured OpenAI tool-use response (CRUX-C-11)
    ToolUseLint {
        /// Path to captured OpenAI tool-use response JSON
        #[arg(long, value_name = "FILE")]
        observation_file: PathBuf,
    },
    /// Lint a GBNF grammar-constrained observation (CRUX-C-10)
    GbnfLint {
        /// Path to captured GBNF observation JSON
        #[arg(long, value_name = "FILE")]
        observation_file: PathBuf,
    },
}
