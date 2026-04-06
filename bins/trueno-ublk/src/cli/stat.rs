//! Stat command - shows device statistics

use super::StatArgs;
use crate::device::UblkDevice;
use anyhow::Result;
use serde::Serialize;

#[derive(Serialize)]
struct FullStats {
    mm_stat: MmStat,
    io_stat: IoStat,
    bd_stat: BdStat,
    extended: ExtendedStats,
}

#[derive(Serialize)]
struct MmStat {
    orig_data_size: u64,
    compr_data_size: u64,
    mem_used_total: u64,
    mem_limit: u64,
    mem_used_max: u64,
    same_pages: u64,
    pages_compacted: u64,
    huge_pages: u64,
    huge_pages_since: u64,
}

#[derive(Serialize)]
struct IoStat {
    failed_reads: u64,
    failed_writes: u64,
    invalid_io: u64,
    notify_free: u64,
}

#[derive(Serialize)]
struct BdStat {
    bd_count: u64,
    bd_reads: u64,
    bd_writes: u64,
}

#[derive(Serialize)]
struct ExtendedStats {
    gpu_pages: u64,
    simd_pages: u64,
    scalar_pages: u64,
    throughput_gbps: f64,
    avg_entropy: f64,
    simd_backend: String,
}

pub fn run(args: StatArgs) -> Result<()> {
    let devices = if let Some(ref path) = args.device {
        vec![UblkDevice::open(path)?]
    } else {
        UblkDevice::list_all()?
    };

    for device in &devices {
        let stats = device.stats();

        if args.mm_stat {
            // zram-compatible mm_stat format (9 fields)
            println!(
                "{} {} {} {} {} {} {} {} {}",
                stats.orig_data_size,
                stats.compr_data_size,
                stats.mem_used_total,
                stats.mem_limit,
                stats.mem_used_max,
                stats.same_pages,
                stats.pages_compacted,
                stats.huge_pages,
                stats.huge_pages_since,
            );
        } else if args.io_stat {
            // zram-compatible io_stat format (4 fields)
            println!(
                "{} {} {} {}",
                stats.failed_reads, stats.failed_writes, stats.invalid_io, stats.notify_free,
            );
        } else if args.bd_stat {
            // zram-compatible bd_stat format (3 fields)
            println!("{} {} {}", stats.bd_count, stats.bd_reads, stats.bd_writes,);
        } else if args.json {
            let full = FullStats {
                mm_stat: MmStat {
                    orig_data_size: stats.orig_data_size,
                    compr_data_size: stats.compr_data_size,
                    mem_used_total: stats.mem_used_total,
                    mem_limit: stats.mem_limit,
                    mem_used_max: stats.mem_used_max,
                    same_pages: stats.same_pages,
                    pages_compacted: stats.pages_compacted,
                    huge_pages: stats.huge_pages,
                    huge_pages_since: stats.huge_pages_since,
                },
                io_stat: IoStat {
                    failed_reads: stats.failed_reads,
                    failed_writes: stats.failed_writes,
                    invalid_io: stats.invalid_io,
                    notify_free: stats.notify_free,
                },
                bd_stat: BdStat {
                    bd_count: stats.bd_count,
                    bd_reads: stats.bd_reads,
                    bd_writes: stats.bd_writes,
                },
                extended: ExtendedStats {
                    gpu_pages: stats.gpu_pages,
                    simd_pages: stats.simd_pages,
                    scalar_pages: stats.scalar_pages,
                    throughput_gbps: stats.throughput_gbps,
                    avg_entropy: stats.avg_entropy,
                    simd_backend: stats.simd_backend.clone(),
                },
            };
            println!("{}", serde_json::to_string_pretty(&full)?);
        } else if args.entropy {
            print_entropy_distribution(device)?;
        } else if args.debug {
            print_debug_stats(device)?;
        } else {
            print_summary(device)?;
        }
    }

    Ok(())
}

fn print_summary(device: &UblkDevice) -> Result<()> {
    let stats = device.stats();
    let config = device.config();

    let ratio = if stats.compr_data_size > 0 {
        stats.orig_data_size as f64 / stats.compr_data_size as f64
    } else {
        1.0
    };

    println!("{}", device.name());
    println!("  Algorithm:   {:?}", config.algorithm);
    println!("  Disk size:   {}", super::format_size(config.size));
    println!("  Data:        {}", super::format_size(stats.orig_data_size));
    println!("  Compressed:  {}", super::format_size(stats.compr_data_size));
    println!("  Ratio:       {:.2}:1", ratio);
    println!("  Streams:     {}", config.streams);
    println!("  GPU:         {}", if config.gpu_enabled { "yes" } else { "no" });
    println!("  Backend:     {}", stats.simd_backend);
    println!("  Throughput:  {:.2} GB/s", stats.throughput_gbps);
    println!("  Zero pages:  {}", stats.same_pages);
    if let Some(ref mp) = device.mountpoint() {
        println!("  Mountpoint:  {}", mp);
    }

    Ok(())
}

fn print_entropy_distribution(device: &UblkDevice) -> Result<()> {
    let stats = device.stats();

    println!("Entropy Distribution for {}", device.name());
    println!("  Average entropy: {:.2} bits/byte", stats.avg_entropy);
    println!();
    println!("  Page routing:");
    println!("    Scalar (high entropy): {:>10} pages", stats.scalar_pages);
    println!("    SIMD (medium entropy): {:>10} pages", stats.simd_pages);
    println!("    GPU batch (low entropy): {:>8} pages", stats.gpu_pages);

    let total = stats.scalar_pages + stats.simd_pages + stats.gpu_pages;
    if total > 0 {
        println!();
        println!("  Distribution:");
        println!("    Scalar: {:>5.1}%", stats.scalar_pages as f64 / total as f64 * 100.0);
        println!("    SIMD:   {:>5.1}%", stats.simd_pages as f64 / total as f64 * 100.0);
        println!("    GPU:    {:>5.1}%", stats.gpu_pages as f64 / total as f64 * 100.0);
    }

    Ok(())
}

fn print_debug_stats(device: &UblkDevice) -> Result<()> {
    let stats = device.stats();

    println!("Debug Stats for {}", device.name());
    println!();
    println!("Memory Statistics (mm_stat):");
    println!("  orig_data_size:   {}", stats.orig_data_size);
    println!("  compr_data_size:  {}", stats.compr_data_size);
    println!("  mem_used_total:   {}", stats.mem_used_total);
    println!("  mem_limit:        {}", stats.mem_limit);
    println!("  mem_used_max:     {}", stats.mem_used_max);
    println!("  same_pages:       {}", stats.same_pages);
    println!("  pages_compacted:  {}", stats.pages_compacted);
    println!("  huge_pages:       {}", stats.huge_pages);
    println!("  huge_pages_since: {}", stats.huge_pages_since);
    println!();
    println!("I/O Statistics (io_stat):");
    println!("  failed_reads:   {}", stats.failed_reads);
    println!("  failed_writes:  {}", stats.failed_writes);
    println!("  invalid_io:     {}", stats.invalid_io);
    println!("  notify_free:    {}", stats.notify_free);
    println!();
    println!("Backing Device (bd_stat):");
    println!("  bd_count:  {}", stats.bd_count);
    println!("  bd_reads:  {}", stats.bd_reads);
    println!("  bd_writes: {}", stats.bd_writes);
    println!();
    println!("Extended (trueno-ublk):");
    println!("  gpu_pages:      {}", stats.gpu_pages);
    println!("  simd_pages:     {}", stats.simd_pages);
    println!("  scalar_pages:   {}", stats.scalar_pages);
    println!("  throughput:     {:.2} GB/s", stats.throughput_gbps);
    println!("  avg_entropy:    {:.2} bits/byte", stats.avg_entropy);
    println!("  simd_backend:   {}", stats.simd_backend);

    Ok(())
}
