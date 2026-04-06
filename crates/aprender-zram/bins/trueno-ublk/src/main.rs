//! trueno-ublk - GPU-accelerated ZRAM replacement using ublk
//!
//! A drop-in replacement for Linux kernel ZRAM that uses trueno-zram-core
//! for SIMD/GPU-accelerated compression via the ublk interface.
//!
//! # Usage
//!
//! ```bash
//! # Create a 1TB compressed RAM disk
//! trueno-ublk create -s 1T -a lz4 --gpu
//!
//! # List devices
//! trueno-ublk list
//!
//! # Show stats
//! trueno-ublk stat /dev/ublkb0
//!
//! # Interactive dashboard
//! trueno-ublk top
//! ```

// Allow dead code in modules prepared for future integration
#![allow(dead_code)]

mod backend; // KERN-001: Kernel-cooperative tiered storage
mod cleanup;
mod cli;
mod daemon;
mod device;
mod perf; // PERF-001: Performance optimization module
mod stats;
mod tui;
mod ublk;
mod visualize; // VIZ-001: Renacer visualization integration

use clap::Parser;
use cli::{Cli, Commands};
use tracing_subscriber::EnvFilter;

fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt().with_env_filter(EnvFilter::from_default_env()).init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Create(args) => {
            // Perform startup cleanup before creating devices (DT-003)
            cleanup::cleanup_orphaned_devices().ok();
            cli::create::run(args)
        }
        Commands::List(args) => cli::list::run(args),
        Commands::Stat(args) => cli::stat::run(args),
        Commands::Reset(args) => cli::reset::run(args),
        Commands::Find => cli::find::run(),
        Commands::Compact(args) => cli::compact::run(args),
        Commands::Idle(args) => cli::idle::run(args),
        Commands::Writeback(args) => cli::writeback::run(args),
        Commands::Set(args) => cli::set::run(args),
        Commands::Top(args) => cli::top::run(args),
        Commands::Entropy(args) => cli::entropy::run(args),
        Commands::Benchmark(args) => cli::benchmark::run(args),
    }
}
