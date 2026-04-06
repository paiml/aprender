//! Benchmark command for compression performance testing.
//!
//! This is a pure shim that delegates to `trueno_zram_core::benchmark`.

use clap::Args;
use trueno_zram_core::benchmark::{
    generate_test_pages, parse_algorithm, run_benchmark, DataPattern,
};

/// Arguments for benchmark command.
#[derive(Args)]
pub struct BenchmarkArgs {
    /// Number of pages to compress.
    #[arg(short, long, default_value = "10000")]
    pub pages: usize,

    /// Algorithm to benchmark (lz4, zstd, all).
    #[arg(short, long, default_value = "all")]
    pub algorithm: String,

    /// Data pattern (zero, random, text, mixed).
    #[arg(short = 'p', long, default_value = "mixed")]
    pub pattern: String,
}

/// Run compression benchmarks.
pub fn benchmark(args: &BenchmarkArgs) -> Result<(), Box<dyn std::error::Error>> {
    let pattern = DataPattern::parse(&args.pattern)
        .ok_or_else(|| format!("Unknown pattern: {}", args.pattern))?;

    let algorithms = parse_algorithm(&args.algorithm)
        .ok_or_else(|| format!("Unknown algorithm: {}", args.algorithm))?;

    println!("trueno-zram Compression Benchmark");
    println!("==================================");
    println!("Pages: {}", args.pages);
    println!("Pattern: {pattern:?}");
    println!();

    // Generate test data
    let pages = generate_test_pages(args.pages, pattern);

    println!(
        "{:<15} {:>10} {:>12} {:>12} {:>10}",
        "Algorithm", "Backend", "Compress", "Decompress", "Ratio"
    );
    println!("{}", "-".repeat(60));

    for algo in algorithms {
        let result = run_benchmark(algo, &pages)?;

        let compress_throughput = result.compress_throughput() / 1e9;
        let decompress_throughput = result.decompress_throughput() / 1e9;

        println!(
            "{:<15} {:>10} {:>10.2} GB/s {:>10.2} GB/s {:>9.2}x",
            format!("{:?}", result.algorithm),
            format!("{:?}", result.backend),
            compress_throughput,
            decompress_throughput,
            result.compression_ratio()
        );
    }

    Ok(())
}
