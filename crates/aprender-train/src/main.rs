//! Entrenar CLI
//!
//! Single-command training entry point for the entrenar library.
//!
//! # Usage
//!
//! ```bash
//! # Train from config
//! entrenar train config.yaml
//!
//! # Train with overrides
//! entrenar train config.yaml --epochs 10 --lr 0.001
//!
//! # Validate config
//! entrenar validate config.yaml
//!
//! # Show config info
//! entrenar info config.yaml
//!
//! # Quantize model
//! entrenar quantize model.gguf --output model_q4.gguf
//!
//! # Merge models
//! entrenar merge model1.gguf model2.gguf --output merged.gguf
//! ```

use clap::Parser;
use entrenar::cli::{run_command, Cli};
use std::process::ExitCode;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() -> ExitCode {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let cli = Cli::parse();

    match run_command(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}
