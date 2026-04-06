//! Top command - interactive TUI dashboard

use super::TopArgs;
use crate::tui;
use anyhow::Result;

pub fn run(args: TopArgs) -> Result<()> {
    if args.report {
        // Non-interactive report mode
        print_report(&args)?;
    } else {
        // Interactive TUI mode
        tui::run(args)?;
    }
    Ok(())
}

fn print_report(args: &TopArgs) -> Result<()> {
    use crate::device::UblkDevice;

    let devices = if let Some(ref path) = args.device {
        vec![UblkDevice::open(path)?]
    } else {
        UblkDevice::list_all()?
    };

    println!("trueno-ublk Status Report");
    println!("========================");
    println!();

    for device in &devices {
        let stats = device.stats();
        let config = device.config();

        let ratio = if stats.compr_data_size > 0 {
            stats.orig_data_size as f64 / stats.compr_data_size as f64
        } else {
            1.0
        };

        println!("Device: {}", device.name());
        println!("  Size:        {}", super::format_size(config.size));
        println!("  Algorithm:   {:?}", config.algorithm);
        println!("  Data:        {}", super::format_size(stats.orig_data_size));
        println!("  Compressed:  {}", super::format_size(stats.compr_data_size));
        println!("  Ratio:       {:.2}:1", ratio);
        println!("  Throughput:  {:.2} GB/s", stats.throughput_gbps);
        println!("  Backend:     {}", stats.simd_backend);
        println!("  GPU:         {}", if config.gpu_enabled { "yes" } else { "no" });
        println!();
    }

    Ok(())
}
