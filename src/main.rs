// CLI binary is only available for native targets (not WASM)
#![cfg_attr(not(feature = "native"), allow(unused))]
#![allow(dead_code)]
#![allow(unused_imports)]

#[macro_use]
#[allow(unused_macros)]
mod generated_contracts;
#[macro_use]
#[allow(unused_macros)]
mod batuta_contracts;
#[cfg(feature = "native")]
mod analyzer;
#[cfg(feature = "native")]
mod ansi_colors;
#[cfg(feature = "native")]
mod backend;
#[cfg(feature = "native")]
mod bug_hunter;
#[cfg(feature = "native")]
mod cli;
#[cfg(feature = "native")]
mod comply;
#[cfg(feature = "native")]
mod config;
#[cfg(feature = "native")]
mod content;
#[cfg(feature = "native")]
mod data;
#[cfg(feature = "native")]
mod experiment;
#[cfg(feature = "native")]
mod hf;
#[cfg(feature = "native")]
mod numpy_converter;
#[cfg(feature = "native")]
mod oracle;
#[cfg(feature = "native")]
mod pacha;
#[cfg(feature = "native")]
mod parf;
#[cfg(feature = "native")]
mod pipeline;
#[cfg(feature = "native")]
mod pipeline_analysis;
#[cfg(feature = "native")]
mod playbook;
#[cfg(feature = "native")]
mod pytorch_converter;
#[cfg(feature = "native")]
mod report;
#[cfg(feature = "native")]
mod sklearn_converter;
#[cfg(feature = "native")]
mod stack;
#[cfg(feature = "native")]
mod timing;
#[cfg(feature = "native")]
mod tools;
#[cfg(feature = "native")]
mod types;
#[cfg(feature = "native")]
mod viz;

// Split modules for QA-002 compliance (<=500 lines per file)
#[cfg(feature = "native")]
#[path = "main_cli.rs"]
mod main_cli;
#[cfg(feature = "native")]
#[path = "main_dispatch.rs"]
mod main_dispatch;
#[cfg(feature = "native")]
#[path = "main_drift.rs"]
mod main_drift;
#[cfg(feature = "native")]
#[path = "main_oracle_args.rs"]
mod main_oracle_args;
#[cfg(feature = "native")]
#[path = "main_oracle_dispatch.rs"]
mod main_oracle_dispatch;

#[cfg(feature = "native")]
fn main() -> anyhow::Result<()> {
    use clap::Parser;
    use tracing::info;
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    let cli = main_cli::Cli::parse();

    // Initialize tracing
    let filter_layer = if cli.debug {
        tracing_subscriber::EnvFilter::new("debug")
    } else if cli.verbose {
        tracing_subscriber::EnvFilter::new("info")
    } else {
        tracing_subscriber::EnvFilter::new("warn")
    };

    tracing_subscriber::registry().with(filter_layer).with(tracing_subscriber::fmt::layer()).init();

    info!("Batuta v{}", env!("CARGO_PKG_VERSION"));

    // --allow-drift or --unsafe-skip-drift-check skip the check entirely
    if !cli.unsafe_skip_drift_check && !cli.allow_drift {
        main_drift::enforce_drift_check(cli.strict)?;
    }

    main_dispatch::dispatch_command(cli.command)
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!(
        "batuta CLI requires the 'native' feature. Build with: cargo build --features native"
    );
    std::process::exit(1);
}
