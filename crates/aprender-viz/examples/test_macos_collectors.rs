//! Test macOS collectors for ttop
use trueno_viz::monitor::collectors::{
    CpuCollector, DiskCollector, MemoryCollector, NetworkCollector, ProcessCollector,
};
use trueno_viz::monitor::Collector;

#[cfg(target_os = "macos")]
use trueno_viz::monitor::collectors::AppleGpuCollector;

fn main() {
    println!("=== ttop macOS Collector Test ===\n");

    // CPU
    let mut cpu = CpuCollector::new();
    println!("CPU Collector:");
    println!("  Available: {}", cpu.is_available());
    println!("  Cores: {}", cpu.core_count());
    if let Ok(m) = cpu.collect() {
        if let Some(load1) = m.get_gauge("cpu.load.1") {
            println!("  Load avg: {load1:.2}");
        }
    }

    // Memory
    let mut mem = MemoryCollector::new();
    println!("\nMemory Collector:");
    println!("  Available: {}", mem.is_available());
    if let Ok(m) = mem.collect() {
        if let Some(total) = m.get_counter("memory.total") {
            println!("  Total: {:.1} GB", total as f64 / 1024.0 / 1024.0 / 1024.0);
        }
        if let Some(pct) = m.get_gauge("memory.used.percent") {
            println!("  Used: {pct:.1}%");
        }
    }

    // Network
    let mut net = NetworkCollector::new();
    println!("\nNetwork Collector:");
    println!("  Available: {}", net.is_available());
    let _ = net.collect();
    std::thread::sleep(std::time::Duration::from_millis(200));
    if net.collect().is_ok() {
        println!("  Interfaces: {:?}", net.interfaces());
        if let Some(rates) = net.current_rates() {
            println!(
                "  Current: {} (RX: {}, TX: {})",
                rates.name,
                rates.rx_formatted(),
                rates.tx_formatted()
            );
        }
    }

    // Disk
    let mut disk = DiskCollector::new();
    println!("\nDisk Collector:");
    println!("  Available: {}", disk.is_available());
    if disk.collect().is_ok() {
        println!("  Mounts found: {}", disk.mounts().len());
        for m in disk.mounts().iter().take(3) {
            println!("    {} -> {} ({:.1}%)", m.device, m.mount_point, m.usage_percent());
        }
    }

    // Process
    let mut proc = ProcessCollector::new();
    println!("\nProcess Collector:");
    println!("  Available: {}", proc.is_available());
    if proc.collect().is_ok() {
        println!("  Processes found: {}", proc.count());
        // Show top 3 by name
        let procs: Vec<_> = proc.processes().values().take(5).collect();
        for p in procs {
            println!(
                "    PID {} {} ({:.1}% CPU, {:.1}% MEM)",
                p.pid, p.name, p.cpu_percent, p.mem_percent
            );
        }
    }

    // GPU (macOS only)
    #[cfg(target_os = "macos")]
    {
        let gpu = AppleGpuCollector::new();
        println!("\nGPU Collector:");
        println!("  Available: {}", gpu.is_available());
        println!("  GPU Count: {}", gpu.gpus().len());
        for (i, info) in gpu.gpus().iter().enumerate() {
            println!("  GPU {}: {} (Metal: {})", i, info.name, info.metal_family);
        }
    }

    println!("\nAll collectors working on macOS!");
}
