//! Realizar CLI - Pure Rust ML inference engine
//!
//! Thin entry point that delegates all logic to the library (T-COV-001).

use clap::Parser;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[tokio::main]
async fn main() -> realizar::error::Result<()> {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    realizar::cli::entrypoint(realizar::cli::Cli::parse()).await
}
