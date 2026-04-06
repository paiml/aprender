//! GPU VRAM Ledger & Guard Example (GPU-SHARE-001/002)
//!
//! Demonstrates VRAM reservation ledger and guard for GPU sharing.
//!
//! ```bash
//! cargo run --example gpu_ledger
//! cargo run --example gpu_ledger -- --reserve 8000 --task "qlora-7b"
//! cargo run --example gpu_ledger -- --status
//! cargo run --example gpu_ledger -- --wait <MB>
//! cargo run --example gpu_ledger -- --guard <MB>
//! ```

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("GPU VRAM Ledger Example");
        println!();
        println!("Usage:");
        println!("  gpu_ledger                          # Auto-detect GPU, show status");
        println!("  gpu_ledger --status                 # Show ledger status");
        println!("  gpu_ledger --reserve <MB> --task <name>  # Reserve VRAM");
        println!("  gpu_ledger --release                # Release our reservation");
        println!("  gpu_ledger --wait <MB>              # Wait for VRAM");
        println!("  gpu_ledger --guard <MB>             # Guard demo (acquire + update actual)");
        return;
    }

    // Enable layer tracing
    entrenar::trace::TRACER.enable();

    // Auto-detect GPU
    let uuid = entrenar::gpu::ledger::detect_gpu_uuid();
    let total_mb = entrenar::gpu::ledger::detect_total_memory_mb();
    let mem_type = entrenar::gpu::ledger::detect_memory_type();

    println!("GPU: {uuid}");
    println!("Total: {total_mb} MB");
    println!("Type: {mem_type:?} (reserve factor: {:.0}%)", mem_type.reserve_factor() * 100.0);
    println!();

    let mut ledger =
        entrenar::gpu::ledger::VramLedger::new(uuid, total_mb, mem_type.reserve_factor())
            .with_profiling(true);

    // --status: show current reservations
    if args.iter().any(|a| a == "--status") {
        match entrenar::gpu::ledger::gpu_status_display(&ledger) {
            Ok(status) => print!("{status}"),
            Err(e) => eprintln!("Error: {e}"),
        }
        return;
    }

    // --reserve <MB> --task <name>: create a reservation
    if let Some(pos) = args.iter().position(|a| a == "--reserve") {
        let budget_mb: usize =
            args.get(pos + 1).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                eprintln!("--reserve requires a number (MB)");
                std::process::exit(1);
            });

        let task = args
            .iter()
            .position(|a| a == "--task")
            .and_then(|p| args.get(p + 1))
            .map_or("example", String::as_str);

        match ledger.try_reserve(budget_mb, task) {
            Ok(id) => {
                println!("Reserved {budget_mb} MB (id: {id})");
                println!("Press Enter to release...");
                let mut input = String::new();
                std::io::stdin().read_line(&mut input).ok();
                ledger.release().ok();
                println!("Released.");
            }
            Err(e) => eprintln!("Error: {e}"),
        }

        println!("\n{}", ledger.profiler_report());
        return;
    }

    // --wait <MB>: wait for VRAM availability
    if let Some(pos) = args.iter().position(|a| a == "--wait") {
        let budget_mb: usize =
            args.get(pos + 1).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                eprintln!("--wait requires a number (MB)");
                std::process::exit(1);
            });

        let config = entrenar::gpu::wait::WaitConfig::with_timeout_secs(60);
        let mut profiler = entrenar::gpu::profiler::GpuProfiler::new(true);

        match entrenar::gpu::wait::wait_for_vram(
            &mut ledger,
            budget_mb,
            "wait-example",
            &config,
            &mut profiler,
        ) {
            Ok(id) => {
                println!("Reserved {budget_mb} MB after waiting (id: {id})");
                ledger.release().ok();
            }
            Err(e) => eprintln!("Error: {e}"),
        }
        return;
    }

    // --guard <MB>: guard demo (acquire, update actual, release)
    if let Some(pos) = args.iter().position(|a| a == "--guard") {
        let budget_mb: usize =
            args.get(pos + 1).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                eprintln!("--guard requires a number (MB)");
                std::process::exit(1);
            });

        match ledger.try_reserve(budget_mb, "guard-demo") {
            Ok(id) => {
                println!("Guard acquired: {budget_mb} MB (id: {id})");
                // Simulate actual VRAM measurement (typically less than budget)
                let actual = budget_mb * 9 / 10;
                ledger.update_actual(actual).ok();
                println!("Updated actual: {actual} MB");
                println!("Status:");
                if let Ok(status) = entrenar::gpu::ledger::gpu_status_display(&ledger) {
                    print!("{status}");
                }
                ledger.release().ok();
                println!("Guard released.");
            }
            Err(e) => eprintln!("Error: {e}"),
        }

        // Show trace report
        let trace_report = entrenar::trace::TRACER.report();
        println!("{trace_report}");
        println!("{}", ledger.profiler_report());
        return;
    }

    // Default: show status
    match entrenar::gpu::ledger::gpu_status_display(&ledger) {
        Ok(status) => print!("{status}"),
        Err(e) => eprintln!("Error: {e}"),
    }
}
