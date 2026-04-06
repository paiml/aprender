//! NVML GPU Monitoring Test
//!
//! Tests real GPU monitoring via NVML on NVIDIA GPUs.
//!
//! Run with: cargo run --example nvml_test --no-default-features --features nvml

use entrenar::monitor::gpu::GpuMonitor;

fn main() {
    println!("Testing NVML GPU Monitor...\n");

    let monitor = GpuMonitor::new().expect("Failed to create GPU monitor");
    let num_devices = monitor.num_devices();

    println!("Detected {num_devices} GPU device(s)");
    println!("Mock mode: {}\n", monitor.is_mock());

    if num_devices == 0 {
        println!("No GPUs detected - NVML may not be initialized");
        return;
    }

    let metrics = monitor.sample();
    for m in &metrics {
        println!("GPU {}: {}", m.device_id, m.name);
        println!("  Utilization: {}%", m.utilization_percent);
        println!(
            "  Memory: {}/{} MB ({}%)",
            m.memory_used_mb, m.memory_total_mb, m.memory_utilization_percent
        );
        println!("  Temperature: {}C", m.temperature_celsius);
        println!(
            "  Power: {:.1}W / {:.1}W ({:.1}%)",
            m.power_watts,
            m.power_limit_watts,
            m.power_percent()
        );
        println!("  Clocks: {} MHz GPU, {} MHz Memory", m.clock_mhz, m.memory_clock_mhz);
        println!("  PCIe: {} KB/s TX, {} KB/s RX", m.pcie_tx_kbps, m.pcie_rx_kbps);
        println!("  Fan: {}%", m.fan_speed_percent);
        println!();
    }
}
