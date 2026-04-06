//! List command - lists all trueno-ublk devices

use super::{format_size, ListArgs, OutputColumn};
use crate::device::UblkDevice;
use anyhow::Result;
use serde::Serialize;

#[derive(Serialize)]
struct DeviceInfo {
    name: String,
    disksize: u64,
    data: u64,
    compr: u64,
    algorithm: String,
    streams: usize,
    zero_pages: u64,
    total: u64,
    mem_limit: u64,
    mem_used: u64,
    migrated: u64,
    mountpoint: Option<String>,
    gpu: bool,
    throughput: f64,
    backend: String,
    entropy: f64,
}

pub fn run(args: ListArgs) -> Result<()> {
    let devices = UblkDevice::list_all()?;

    if args.json {
        let infos: Vec<DeviceInfo> = devices.iter().map(device_to_info).collect();
        println!("{}", serde_json::to_string_pretty(&infos)?);
        return Ok(());
    }

    let columns = if args.output_all {
        OutputColumn::all().to_vec()
    } else if let Some(ref output) = args.output {
        parse_columns(output)?
    } else {
        OutputColumn::default_columns().to_vec()
    };

    // Print header
    if !args.no_headers {
        let header: Vec<&str> = columns.iter().map(|c| c.header()).collect();
        if args.raw {
            println!("{}", header.join(" "));
        } else {
            print_row(&header, &column_widths(&columns));
        }
    }

    // Print devices
    for device in &devices {
        let values = format_device(device, &columns, args.bytes);
        if args.raw {
            println!("{}", values.join(" "));
        } else {
            print_row_values(&values, &column_widths(&columns));
        }
    }

    Ok(())
}

fn device_to_info(device: &UblkDevice) -> DeviceInfo {
    let stats = device.stats();
    DeviceInfo {
        name: device.name().to_string(),
        disksize: device.config().size,
        data: stats.orig_data_size,
        compr: stats.compr_data_size,
        algorithm: format!("{:?}", device.config().algorithm),
        streams: device.config().streams,
        zero_pages: stats.same_pages,
        total: stats.mem_used_total,
        mem_limit: stats.mem_limit,
        mem_used: stats.mem_used_max,
        migrated: stats.pages_compacted,
        mountpoint: device.mountpoint(),
        gpu: device.config().gpu_enabled,
        throughput: stats.throughput_gbps,
        backend: stats.simd_backend.clone(),
        entropy: stats.avg_entropy,
    }
}

fn parse_columns(s: &str) -> Result<Vec<OutputColumn>> {
    s.split(',')
        .map(|col| {
            let col = col.trim().to_uppercase();
            match col.as_str() {
                "NAME" => Ok(OutputColumn::Name),
                "DISKSIZE" => Ok(OutputColumn::Disksize),
                "DATA" => Ok(OutputColumn::Data),
                "COMPR" => Ok(OutputColumn::Compr),
                "ALGORITHM" => Ok(OutputColumn::Algorithm),
                "STREAMS" => Ok(OutputColumn::Streams),
                "ZERO-PAGES" => Ok(OutputColumn::ZeroPages),
                "TOTAL" => Ok(OutputColumn::Total),
                "MEM-LIMIT" => Ok(OutputColumn::MemLimit),
                "MEM-USED" => Ok(OutputColumn::MemUsed),
                "MIGRATED" => Ok(OutputColumn::Migrated),
                "MOUNTPOINT" => Ok(OutputColumn::Mountpoint),
                "GPU" => Ok(OutputColumn::Gpu),
                "THROUGHPUT" => Ok(OutputColumn::Throughput),
                "BACKEND" => Ok(OutputColumn::Backend),
                "ENTROPY" => Ok(OutputColumn::Entropy),
                _ => anyhow::bail!("Unknown column: {}", col),
            }
        })
        .collect()
}

fn format_device(device: &UblkDevice, columns: &[OutputColumn], bytes: bool) -> Vec<String> {
    let stats = device.stats();
    let config = device.config();

    columns
        .iter()
        .map(|col| match col {
            OutputColumn::Name => device.name().to_string(),
            OutputColumn::Disksize => {
                if bytes {
                    config.size.to_string()
                } else {
                    format_size(config.size)
                }
            }
            OutputColumn::Data => {
                if bytes {
                    stats.orig_data_size.to_string()
                } else {
                    format_size(stats.orig_data_size)
                }
            }
            OutputColumn::Compr => {
                if bytes {
                    stats.compr_data_size.to_string()
                } else {
                    format_size(stats.compr_data_size)
                }
            }
            OutputColumn::Algorithm => format!("{:?}", config.algorithm).to_lowercase(),
            OutputColumn::Streams => config.streams.to_string(),
            OutputColumn::ZeroPages => stats.same_pages.to_string(),
            OutputColumn::Total => {
                if bytes {
                    stats.mem_used_total.to_string()
                } else {
                    format_size(stats.mem_used_total)
                }
            }
            OutputColumn::MemLimit => {
                if stats.mem_limit == 0 {
                    "0".to_string()
                } else if bytes {
                    stats.mem_limit.to_string()
                } else {
                    format_size(stats.mem_limit)
                }
            }
            OutputColumn::MemUsed => {
                if bytes {
                    stats.mem_used_max.to_string()
                } else {
                    format_size(stats.mem_used_max)
                }
            }
            OutputColumn::Migrated => stats.pages_compacted.to_string(),
            OutputColumn::Mountpoint => device.mountpoint().unwrap_or_default(),
            OutputColumn::Gpu => {
                if config.gpu_enabled {
                    "yes".to_string()
                } else {
                    "no".to_string()
                }
            }
            OutputColumn::Throughput => format!("{:.1} GB/s", stats.throughput_gbps),
            OutputColumn::Backend => stats.simd_backend.clone(),
            OutputColumn::Entropy => format!("{:.1}", stats.avg_entropy),
        })
        .collect()
}

fn column_widths(columns: &[OutputColumn]) -> Vec<usize> {
    columns.iter().map(|c| c.header().len().max(12)).collect()
}

fn print_row(values: &[&str], widths: &[usize]) {
    for (val, width) in values.iter().zip(widths.iter()) {
        print!("{:>width$} ", val, width = width);
    }
    println!();
}

fn print_row_values(values: &[String], widths: &[usize]) {
    for (val, width) in values.iter().zip(widths.iter()) {
        print!("{:>width$} ", val, width = width);
    }
    println!();
}
