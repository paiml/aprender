//! Status command for zram devices.
//!
//! This is a pure shim that delegates to `trueno_zram_core::zram`.

use crate::output::OutputFormat;
use clap::Args;
use serde::Serialize;
use trueno_zram_core::zram::{format_size, SysfsOps, ZramOps};

/// Arguments for status command.
#[derive(Args)]
pub struct StatusArgs {
    /// Specific device to show (omit for all devices).
    #[arg(short, long)]
    pub device: Option<u32>,
}

/// Serializable status for JSON output.
#[derive(Debug, Serialize)]
struct StatusOutput {
    device: u32,
    disksize: u64,
    orig_data_size: u64,
    compr_data_size: u64,
    mem_used_total: u64,
    algorithm: String,
    ratio: f64,
}

/// Show zram device status.
pub fn status(args: &StatusArgs, format: OutputFormat) -> Result<(), Box<dyn std::error::Error>> {
    let ops = SysfsOps::new();

    let statuses: Vec<StatusOutput> = if let Some(dev) = args.device {
        let s = ops.status(dev)?;
        let ratio = s.compression_ratio();
        vec![StatusOutput {
            device: s.device,
            disksize: s.disksize,
            orig_data_size: s.orig_data_size,
            compr_data_size: s.compr_data_size,
            mem_used_total: s.mem_used_total,
            algorithm: s.algorithm,
            ratio,
        }]
    } else {
        ops.list()?
            .into_iter()
            .map(|s| {
                let ratio = s.compression_ratio();
                StatusOutput {
                    device: s.device,
                    disksize: s.disksize,
                    orig_data_size: s.orig_data_size,
                    compr_data_size: s.compr_data_size,
                    mem_used_total: s.mem_used_total,
                    algorithm: s.algorithm,
                    ratio,
                }
            })
            .collect()
    };

    match format {
        OutputFormat::Table => print_table(&statuses),
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&statuses)?);
        }
        OutputFormat::Raw => {
            for s in &statuses {
                println!(
                    "{} {} {} {} {} {}",
                    s.device,
                    s.disksize,
                    s.orig_data_size,
                    s.compr_data_size,
                    s.mem_used_total,
                    s.algorithm
                );
            }
        }
    }

    Ok(())
}

fn print_table(statuses: &[StatusOutput]) {
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "DEVICE", "DISKSIZE", "DATA", "COMPR", "TOTAL", "ALGO", "RATIO"
    );

    for s in statuses {
        println!(
            "zram{:<2} {:>10} {:>10} {:>10} {:>10} {:>8} {:>7.2}x",
            s.device,
            format_size(s.disksize),
            format_size(s.orig_data_size),
            format_size(s.compr_data_size),
            format_size(s.mem_used_total),
            &s.algorithm[..s.algorithm.len().min(8)],
            s.ratio
        );
    }
}
