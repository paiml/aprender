//! alimentar CLI entry point

use std::process::ExitCode;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() -> ExitCode {
    #[cfg(not(target_arch = "wasm32"))]
    sovereign_update::hook!("alimentar"); // EPIC #4232: `alimentar update`, and the startup notice
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    alimentar::cli::run()
}
