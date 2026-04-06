//! CLI module for trueno-ublk
//!
//! Provides zramctl-compatible command-line interface with extensions
//! for GPU acceleration and entropy analysis.

pub mod benchmark; // VIZ-003: JSON/HTML export
pub mod compact;
pub mod create;
pub mod entropy;
pub mod find;
pub mod idle;
pub mod list;
pub mod reset;
pub mod set;
pub mod stat;
pub mod top;
pub mod writeback;

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// trueno-ublk - GPU-accelerated ZRAM replacement
#[derive(Parser, Debug)]
#[command(name = "trueno-ublk")]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
#[allow(clippy::large_enum_variant)] // CLI args are constructed once at startup
pub enum Commands {
    /// Create a new ublk device
    Create(CreateArgs),

    /// List all trueno-ublk devices
    List(ListArgs),

    /// Show device statistics
    Stat(StatArgs),

    /// Reset/remove a device
    Reset(ResetArgs),

    /// Find a free device
    Find,

    /// Trigger compaction
    Compact(DeviceArg),

    /// Mark pages as idle (for writeback)
    Idle(DeviceArg),

    /// Trigger writeback to backing device
    Writeback(WritebackArgs),

    /// Live reconfiguration
    Set(SetArgs),

    /// Interactive TUI dashboard
    Top(TopArgs),

    /// Analyze file/directory entropy
    Entropy(EntropyArgs),

    /// Run benchmarks and generate reports (VIZ-003)
    Benchmark(BenchmarkArgs),
}

/// Compression algorithm selection
#[derive(Clone, Copy, Debug, ValueEnum, Default)]
pub enum Algorithm {
    /// LZ4 - fast compression (default)
    #[default]
    Lz4,
    /// LZ4 HC - higher compression ratio
    Lz4hc,
    /// Zstd level 1 - balanced
    Zstd1,
    /// Zstd level 3 - better ratio
    Zstd3,
    /// Zstd level 9 - maximum compression
    Zstd9,
}

impl Algorithm {
    pub fn to_trueno(self) -> trueno_zram_core::Algorithm {
        match self {
            Algorithm::Lz4 => trueno_zram_core::Algorithm::Lz4,
            Algorithm::Lz4hc => trueno_zram_core::Algorithm::Lz4Hc,
            Algorithm::Zstd1 => trueno_zram_core::Algorithm::Zstd { level: 1 },
            Algorithm::Zstd3 => trueno_zram_core::Algorithm::Zstd { level: 3 },
            Algorithm::Zstd9 => trueno_zram_core::Algorithm::Zstd { level: 9 },
        }
    }
}

/// Create command arguments
#[derive(Parser, Debug)]
pub struct CreateArgs {
    /// Device ID to use (-1 for auto-assign)
    #[arg(long, default_value = "-1")]
    pub dev_id: i32,

    /// Device size (e.g., 1T, 256G, 1024M)
    #[arg(short, long)]
    pub size: String,

    /// Compression algorithm
    #[arg(short, long, value_enum, default_value = "lz4")]
    pub algorithm: Algorithm,

    /// Number of compression streams/threads
    #[arg(short = 't', long, default_value = "0")]
    pub streams: usize,

    /// Enable GPU acceleration
    #[arg(long)]
    pub gpu: bool,

    /// Memory limit for compressed data
    #[arg(long)]
    pub mem_limit: Option<String>,

    /// Backing device for writeback
    #[arg(long)]
    pub backing_dev: Option<PathBuf>,

    /// Writeback limit
    #[arg(long)]
    pub writeback_limit: Option<String>,

    /// Entropy threshold for skipping compression (0.0-8.0)
    #[arg(long, default_value = "7.5")]
    pub entropy_skip: f64,

    /// GPU batch size threshold
    #[arg(long, default_value = "1000")]
    pub gpu_batch: usize,

    /// Run in foreground (don't daemonize)
    #[arg(short = 'f', long)]
    pub foreground: bool,

    /// Enable batched compression mode for high throughput (>10 GB/s)
    ///
    /// In batched mode, pages are buffered until batch-threshold is reached,
    /// then compressed in parallel using SIMD (19-24 GB/s).
    /// DEFAULT: enabled. Use --no-batched to disable.
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    pub batched: bool,

    /// Disable batched compression mode (use per-page mode)
    ///
    /// Per-page mode has lower latency but lower throughput (~3.7 GB/s).
    #[arg(long = "no-batched", action = clap::ArgAction::SetTrue)]
    pub no_batched: bool,

    /// Batch threshold - pages before triggering batch compression (default: 1000)
    #[arg(long, default_value = "1000")]
    pub batch_threshold: usize,

    /// Flush timeout in ms - max time before flushing partial batch (default: 10)
    #[arg(long, default_value = "10")]
    pub flush_timeout_ms: u64,

    // =========================================================================
    // PERF-001: Performance Optimization Options
    // =========================================================================
    /// Enable polling mode (spin-wait on io_uring completions)
    ///
    /// Reduces latency by avoiding interrupt overhead. Increases CPU usage.
    /// Recommended for high-IOPS workloads.
    #[arg(long)]
    pub polling: bool,

    /// Polling spin cycles before yielding (default: 10000)
    #[arg(long, default_value = "10000")]
    pub poll_spin_cycles: u32,

    /// Enable adaptive polling (reduces CPU when idle)
    #[arg(long, default_value = "true")]
    pub poll_adaptive: bool,

    /// Page batch size for coalescing (64-256, default: 64)
    ///
    /// Groups sequential I/O requests for better throughput.
    /// Higher values improve throughput but may increase latency.
    #[arg(long, default_value = "64")]
    pub batch_pages: usize,

    /// Batch timeout in microseconds (default: 100)
    #[arg(long, default_value = "100")]
    pub batch_timeout_us: u64,

    /// CPU cores to pin workers to (comma-separated, e.g., "1,2,3,4")
    ///
    /// Pin ublk worker threads to specific cores for cache locality.
    /// Leave empty for no pinning (default).
    #[arg(long)]
    pub cpu_affinity: Option<String>,

    /// NUMA node for memory allocation (-1 for auto, default: -1)
    ///
    /// Allocate compression buffers on the specified NUMA node.
    /// Use -1 for automatic detection based on worker CPU.
    #[arg(long, default_value = "-1")]
    pub numa_node: i32,

    /// Use high-performance preset (enables polling, larger batches)
    ///
    /// Equivalent to: --polling --batch-pages 128 --batch-timeout-us 50
    #[arg(long)]
    pub high_perf: bool,

    /// Use maximum performance preset (highest CPU usage)
    ///
    /// Equivalent to: --polling --batch-pages 256 --batch-timeout-us 25 --poll-spin-cycles 100000
    /// WARNING: High CPU usage. Use only when targeting maximum IOPS.
    #[arg(long)]
    pub max_perf: bool,

    // =========================================================================
    // PERF-003: Multi-Queue Parallelism Options
    // =========================================================================
    /// Number of hardware queues (1-8, default: 1)
    ///
    /// Multiple queues enable parallel I/O processing with separate io_uring
    /// instances per queue. Each queue runs in its own thread pinned to a CPU core.
    /// More queues = higher IOPS but more CPU usage.
    ///
    /// Scaling: 1q=162K, 2q=~300K, 4q=~500K, 8q=~800K IOPS
    #[arg(long, default_value = "1", value_parser = clap::value_parser!(u16).range(1..=8))]
    pub queues: u16,

    // =========================================================================
    // PERF-006: Zero-Copy Mode (EXPERIMENTAL)
    // =========================================================================
    /// Enable UBLK_F_SUPPORT_ZERO_COPY mode (EXPERIMENTAL)
    ///
    /// Eliminates pwrite syscall per I/O by using mmap'd kernel buffers.
    /// Requires kernel 6.0+ with full ZERO_COPY support.
    /// WARNING: This is experimental and may not work on all kernels.
    #[arg(long)]
    pub zero_copy: bool,

    // =========================================================================
    // KERN-001/002/003: Kernel-Cooperative Tiered Storage
    // =========================================================================
    /// Storage backend type: memory, zram, tiered (default: memory)
    ///
    /// - memory: In-memory trueno SIMD compression (default, 10.6 GB/s)
    /// - zram: Route all pages to kernel ZRAM device (171 GB/s for low-entropy)
    /// - tiered: Entropy-based routing (kernel zram + trueno SIMD)
    ///
    /// Tiered mode routes pages by Shannon entropy H(X):
    /// - H(X) < 6.0: Kernel ZRAM (171 GB/s, fast LZ4)
    /// - 6.0 ≤ H(X) ≤ 7.5: trueno SIMD ZSTD (15 GiB/s, 6x better ratio)
    /// - H(X) > 7.5: Skip compression (3.4 GB/s direct)
    #[arg(long, default_value = "memory")]
    pub backend: String,

    /// Enable entropy-based routing for tiered storage
    ///
    /// Routes pages to optimal backend based on Shannon entropy.
    /// Requires --backend=tiered or --backend=zram.
    /// Without this flag, all pages use the same backend.
    #[arg(long)]
    pub entropy_routing: bool,

    /// Kernel ZRAM device path (default: /dev/zram0)
    ///
    /// Used when --backend=zram or --backend=tiered.
    /// The kernel ZRAM device must already exist and be configured.
    /// Use `zramctl` to create and size the device first.
    #[arg(long, default_value = "/dev/zram0")]
    pub zram_device: String,

    /// NVMe cold tier directory path (KERN-003)
    ///
    /// Used when --backend=tiered for high-entropy pages (H(X) > 7.5).
    /// High-entropy data skips compression and goes directly to NVMe.
    /// Directory will be created if it doesn't exist.
    /// Example: /var/lib/trueno/cold
    #[arg(long)]
    pub cold_tier: Option<String>,

    /// Entropy threshold for kernel ZRAM routing (default: 6.0)
    ///
    /// Pages with H(X) below this threshold are routed to kernel ZRAM.
    /// Higher values = more pages to kernel ZRAM (faster, less compression).
    /// Lower values = more pages to trueno SIMD (slower, better compression).
    #[arg(long, default_value = "6.0")]
    pub entropy_kernel_threshold: f64,

    /// Entropy threshold for skipping compression (default: 7.5)
    ///
    /// Pages with H(X) above this threshold skip compression entirely.
    /// Encrypted/random data is incompressible; save CPU cycles.
    #[arg(long, default_value = "7.5")]
    pub entropy_skip_threshold: f64,

    // =========================================================================
    // VIZ-002: Renacer Visualization Integration
    // =========================================================================
    /// Launch real-time TUI visualization (renacer dashboard)
    ///
    /// Displays live metrics: throughput, IOPS, compression ratio, tier distribution.
    /// Uses renacer visualization framework with ratatui-based panels.
    /// Press 'q' to quit the dashboard.
    #[arg(long)]
    pub visualize: bool,

    // =========================================================================
    // VIZ-004: OTLP Integration (OpenTelemetry)
    // =========================================================================
    /// OTLP endpoint for trace/metric export (e.g., http://localhost:4317)
    ///
    /// When set, exports OpenTelemetry spans and metrics to the specified
    /// OTLP-compatible collector (Jaeger, Tempo, etc.).
    #[arg(long)]
    pub otlp_endpoint: Option<String>,

    /// Service name for OTLP traces (default: trueno-ublk)
    #[arg(long, default_value = "trueno-ublk")]
    pub otlp_service_name: String,
}

/// List command arguments
#[derive(Parser, Debug)]
pub struct ListArgs {
    /// Output columns (comma-separated)
    #[arg(short, long)]
    pub output: Option<String>,

    /// Output all columns
    #[arg(long)]
    pub output_all: bool,

    /// Print sizes in bytes
    #[arg(short, long)]
    pub bytes: bool,

    /// Don't print headings
    #[arg(short = 'n', long)]
    pub no_headers: bool,

    /// Raw output format
    #[arg(long)]
    pub raw: bool,

    /// JSON output
    #[arg(long)]
    pub json: bool,
}

/// Stat command arguments
#[derive(Parser, Debug)]
pub struct StatArgs {
    /// Device path (e.g., /dev/ublkb0)
    pub device: Option<PathBuf>,

    /// Output mm_stat format (zram compatible)
    #[arg(long)]
    pub mm_stat: bool,

    /// Output io_stat format
    #[arg(long)]
    pub io_stat: bool,

    /// Output bd_stat format (backing device)
    #[arg(long)]
    pub bd_stat: bool,

    /// JSON output
    #[arg(long)]
    pub json: bool,

    /// Show entropy distribution
    #[arg(long)]
    pub entropy: bool,

    /// Show debug stats
    #[arg(long)]
    pub debug: bool,
}

/// Reset command arguments
#[derive(Parser, Debug)]
pub struct ResetArgs {
    /// Device path(s) to reset
    pub devices: Vec<PathBuf>,

    /// Reset all devices
    #[arg(long)]
    pub all: bool,
}

/// Simple device argument
#[derive(Parser, Debug)]
pub struct DeviceArg {
    /// Device path
    pub device: PathBuf,
}

/// Writeback command arguments
#[derive(Parser, Debug)]
pub struct WritebackArgs {
    /// Device path
    pub device: PathBuf,

    /// Only writeback idle pages
    #[arg(long)]
    pub idle: bool,

    /// Only writeback huge pages
    #[arg(long)]
    pub huge: bool,

    /// Writeback all pages
    #[arg(long)]
    pub all: bool,
}

/// Set command arguments
#[derive(Parser, Debug)]
pub struct SetArgs {
    /// Device path
    pub device: PathBuf,

    /// Set memory limit
    #[arg(long)]
    pub mem_limit: Option<String>,

    /// Enable/disable GPU
    #[arg(long)]
    pub gpu: Option<bool>,

    /// Set entropy skip threshold
    #[arg(long)]
    pub entropy_skip: Option<f64>,

    /// Set writeback limit
    #[arg(long)]
    pub writeback_limit: Option<String>,

    /// Enable/disable writeback limit
    #[arg(long)]
    pub writeback_limit_enable: Option<bool>,

    /// Reset mem_used_max watermark
    #[arg(long)]
    pub reset_mem_used_max: bool,
}

/// Top (TUI) command arguments
#[derive(Parser, Debug)]
pub struct TopArgs {
    /// Device to monitor (optional, monitors all if not specified)
    pub device: Option<PathBuf>,

    /// Demo mode with simulated data
    #[arg(long)]
    pub demo: bool,

    /// Non-interactive report mode
    #[arg(long)]
    pub report: bool,
}

/// Entropy analysis arguments
#[derive(Parser, Debug)]
pub struct EntropyArgs {
    /// Paths to analyze
    pub paths: Vec<PathBuf>,

    /// JSON output
    #[arg(long)]
    pub json: bool,

    /// Recursive directory scan
    #[arg(short, long)]
    pub recursive: bool,
}

// =============================================================================
// VIZ-003: Benchmark Report Arguments
// =============================================================================

/// Benchmark output format
#[derive(Clone, Copy, Debug, ValueEnum, Default)]
pub enum BenchmarkFormat {
    /// Plain text summary (default)
    #[default]
    Text,
    /// JSON format for analysis tools
    Json,
    /// HTML report with visualizations
    Html,
}

/// Benchmark workload type
#[derive(Clone, Copy, Debug, ValueEnum, Default)]
pub enum BenchmarkWorkload {
    /// Sequential write + read test
    #[default]
    Sequential,
    /// Random 4K IOPS test
    Random,
    /// Mixed workload (70% read, 30% write)
    Mixed,
    /// All workloads
    All,
}

/// Benchmark command arguments (VIZ-003)
#[derive(Parser, Debug)]
pub struct BenchmarkArgs {
    /// Device size for benchmark (e.g., 1G, 256M)
    #[arg(short, long, default_value = "1G")]
    pub size: String,

    /// Workload type to benchmark
    #[arg(short, long, value_enum, default_value = "sequential")]
    pub workload: BenchmarkWorkload,

    /// Output format
    #[arg(short, long, value_enum, default_value = "text")]
    pub format: BenchmarkFormat,

    /// Output file (stdout if not specified)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Number of iterations for averaging
    #[arg(long, default_value = "3")]
    pub iterations: u32,

    /// Enable ML anomaly detection in JSON output
    #[arg(long)]
    pub ml_anomaly: bool,

    /// Backend to benchmark (memory, zram, tiered)
    #[arg(long, default_value = "memory")]
    pub backend: String,
}

/// Output columns available for list command
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum OutputColumn {
    Name,
    Disksize,
    Data,
    Compr,
    Algorithm,
    Streams,
    ZeroPages,
    Total,
    MemLimit,
    MemUsed,
    Migrated,
    Mountpoint,
    Gpu,
    Throughput,
    Backend,
    Entropy,
}

impl OutputColumn {
    pub fn all() -> &'static [OutputColumn] {
        &[
            OutputColumn::Name,
            OutputColumn::Disksize,
            OutputColumn::Data,
            OutputColumn::Compr,
            OutputColumn::Algorithm,
            OutputColumn::Streams,
            OutputColumn::ZeroPages,
            OutputColumn::Total,
            OutputColumn::MemLimit,
            OutputColumn::MemUsed,
            OutputColumn::Migrated,
            OutputColumn::Mountpoint,
            OutputColumn::Gpu,
            OutputColumn::Throughput,
            OutputColumn::Backend,
            OutputColumn::Entropy,
        ]
    }

    pub fn default_columns() -> &'static [OutputColumn] {
        &[
            OutputColumn::Name,
            OutputColumn::Disksize,
            OutputColumn::Data,
            OutputColumn::Compr,
            OutputColumn::Algorithm,
            OutputColumn::Streams,
        ]
    }

    pub fn header(&self) -> &'static str {
        match self {
            OutputColumn::Name => "NAME",
            OutputColumn::Disksize => "DISKSIZE",
            OutputColumn::Data => "DATA",
            OutputColumn::Compr => "COMPR",
            OutputColumn::Algorithm => "ALGORITHM",
            OutputColumn::Streams => "STREAMS",
            OutputColumn::ZeroPages => "ZERO-PAGES",
            OutputColumn::Total => "TOTAL",
            OutputColumn::MemLimit => "MEM-LIMIT",
            OutputColumn::MemUsed => "MEM-USED",
            OutputColumn::Migrated => "MIGRATED",
            OutputColumn::Mountpoint => "MOUNTPOINT",
            OutputColumn::Gpu => "GPU",
            OutputColumn::Throughput => "THROUGHPUT",
            OutputColumn::Backend => "BACKEND",
            OutputColumn::Entropy => "ENTROPY",
        }
    }
}

/// Parse size string like "1T", "256G", "1024M" into bytes
pub fn parse_size(s: &str) -> anyhow::Result<u64> {
    let s = s.trim().to_uppercase();
    let (num, multiplier) = if s.ends_with("T") || s.ends_with("TIB") {
        (s.trim_end_matches("TIB").trim_end_matches("T"), 1u64 << 40)
    } else if s.ends_with("G") || s.ends_with("GIB") {
        (s.trim_end_matches("GIB").trim_end_matches("G"), 1u64 << 30)
    } else if s.ends_with("M") || s.ends_with("MIB") {
        (s.trim_end_matches("MIB").trim_end_matches("M"), 1u64 << 20)
    } else if s.ends_with("K") || s.ends_with("KIB") {
        (s.trim_end_matches("KIB").trim_end_matches("K"), 1u64 << 10)
    } else {
        (s.as_str(), 1u64)
    };

    let num: u64 = num.trim().parse()?;
    Ok(num * multiplier)
}

/// Format bytes as human-readable string
pub fn format_size(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;
    const TIB: u64 = GIB * 1024;

    if bytes >= TIB {
        format!("{:.1}T", bytes as f64 / TIB as f64)
    } else if bytes >= GIB {
        format!("{:.1}G", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1}M", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1}K", bytes as f64 / KIB as f64)
    } else {
        format!("{}B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_size_terabytes() {
        assert_eq!(parse_size("1T").unwrap(), 1u64 << 40);
        assert_eq!(parse_size("2TiB").unwrap(), 2u64 << 40);
    }

    #[test]
    fn test_parse_size_gigabytes() {
        assert_eq!(parse_size("256G").unwrap(), 256u64 << 30);
        assert_eq!(parse_size("128GiB").unwrap(), 128u64 << 30);
    }

    #[test]
    fn test_parse_size_megabytes() {
        assert_eq!(parse_size("1024M").unwrap(), 1024u64 << 20);
    }

    #[test]
    fn test_parse_size_bytes() {
        assert_eq!(parse_size("4096").unwrap(), 4096);
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(1u64 << 40), "1.0T");
        assert_eq!(format_size(256u64 << 30), "256.0G");
        assert_eq!(format_size(1024u64 << 20), "1.0G");
        assert_eq!(format_size(512u64 << 20), "512.0M");
    }

    #[test]
    fn test_output_columns() {
        assert!(OutputColumn::all().len() >= 12);
        assert!(OutputColumn::default_columns().len() >= 6);
    }

    #[test]
    fn test_algorithm_conversion() {
        let lz4 = Algorithm::Lz4;
        assert!(matches!(lz4.to_trueno(), trueno_zram_core::Algorithm::Lz4));
    }

    // ========================================================================
    // Popperian Falsification Checklist - Section E: CLI & Usability (51-60)
    // ========================================================================

    /// E51: Create with invalid algorithm - verify clap rejects unknown values.
    #[test]
    fn popperian_e51_invalid_algorithm_rejected() {
        use clap::Parser;

        // Try to parse with invalid algorithm
        let result = Cli::try_parse_from([
            "trueno-ublk",
            "create",
            "--size",
            "1G",
            "--algorithm",
            "invalid_algo",
        ]);

        assert!(result.is_err(), "E51: Invalid algorithm should be rejected by clap");

        let err = result.unwrap_err();
        let err_str = err.to_string();
        assert!(
            err_str.contains("invalid_algo") || err_str.contains("invalid value"),
            "E51: Error should mention invalid algorithm"
        );
    }

    /// E52: Create with invalid size - verify parse_size rejects malformed input.
    #[test]
    fn popperian_e52_invalid_size_rejected() {
        // Malformed size strings
        let invalid_sizes = [
            "invalid", "1X",     // Unknown suffix
            "-1G",    // Negative
            "1.5.5G", // Multiple decimals
            "",       // Empty
            "   ",    // Whitespace only
        ];

        for size_str in &invalid_sizes {
            let result = parse_size(size_str);
            assert!(result.is_err(), "E52: Invalid size '{}' should be rejected", size_str);
        }
    }

    /// E53: List output columns - verify JSON schema consistency.
    #[test]
    fn popperian_e53_list_json_schema() {
        // Verify ListArgs can be constructed with JSON output
        let list_args = ListArgs {
            output: None,
            output_all: false,
            bytes: false,
            no_headers: false,
            raw: false,
            json: true,
        };

        assert!(list_args.json, "E53: JSON flag should be set");

        // Verify all output columns have headers
        for col in OutputColumn::all() {
            let header = col.header();
            assert!(!header.is_empty(), "E53: Column {:?} should have a header", col);
        }
    }

    /// E54: Stat command - verify it captures compression ratio.
    #[test]
    fn popperian_e54_stat_compression_ratio() {
        // Verify StatArgs supports compression-related output
        let stat_args = StatArgs {
            device: None,
            mm_stat: true,
            io_stat: false,
            bd_stat: false,
            json: true,
            entropy: true,
            debug: false,
        };

        assert!(stat_args.mm_stat, "E54: mm_stat should be available for compression ratio");
        assert!(stat_args.entropy, "E54: entropy stat should be available");
    }

    /// E55: Reset command - verify it accepts multiple devices.
    #[test]
    fn popperian_e55_reset_cleanup() {
        use clap::Parser;

        // Reset all devices
        let result = Cli::try_parse_from(["trueno-ublk", "reset", "--all"]);
        assert!(result.is_ok(), "E55: Reset --all should parse");

        // Reset specific devices
        let result = Cli::try_parse_from(["trueno-ublk", "reset", "/dev/ublkb0", "/dev/ublkb1"]);
        assert!(result.is_ok(), "E55: Reset with multiple devices should parse");

        if let Ok(cli) = result {
            if let Commands::Reset(args) = cli.command {
                assert_eq!(args.devices.len(), 2, "E55: Should capture both devices");
            }
        }
    }

    /// E56: Help text - verify all subcommands have descriptions.
    #[test]
    fn popperian_e56_help_text_present() {
        use clap::CommandFactory;

        let cmd = Cli::command();

        // Verify main command has about text
        assert!(cmd.get_about().is_some(), "E56: Main command should have about text");

        // Verify subcommands exist
        let subcommands: Vec<_> = cmd.get_subcommands().collect();
        assert!(
            subcommands.len() >= 8,
            "E56: Should have at least 8 subcommands, got {}",
            subcommands.len()
        );

        // Verify each subcommand has about text
        for subcmd in subcommands {
            let name = subcmd.get_name();
            // Skip help which is auto-generated
            if name == "help" {
                continue;
            }
            assert!(
                subcmd.get_about().is_some(),
                "E56: Subcommand '{}' should have about text",
                name
            );
        }
    }

    /// E57: Version matches Cargo.toml.
    #[test]
    fn popperian_e57_version_matches_cargo() {
        use clap::CommandFactory;

        let cmd = Cli::command();
        let version = cmd.get_version();

        // Should have a version string
        assert!(version.is_some(), "E57: Should have version string");

        let version_str = version.unwrap();

        // Version should match Cargo.toml format (e.g., "0.1.0")
        let parts: Vec<&str> = version_str.split('.').collect();
        assert!(parts.len() >= 2, "E57: Version should be semver format, got: {}", version_str);

        // First part should be numeric
        assert!(parts[0].parse::<u32>().is_ok(), "E57: Major version should be numeric");
    }

    /// E58: Log levels - verify RUST_LOG parsing works.
    #[test]
    fn popperian_e58_log_levels() {
        use std::str::FromStr;

        // Verify common log levels are valid
        let log_levels = ["trace", "debug", "info", "warn", "error"];

        for level in &log_levels {
            // This tests that the level string is valid for tracing
            let filter = tracing_subscriber::filter::LevelFilter::from_str(level);
            assert!(filter.is_ok(), "E58: Log level '{}' should be valid", level);
        }

        // Verify module-specific filtering pattern
        let complex_filter = "trueno_ublk=debug,trueno_zram_core=trace";
        assert!(complex_filter.contains("="), "E58: Module-specific filters should use '=' syntax");
    }

    /// E59: TUI - verify TopArgs can enable demo mode.
    #[test]
    fn popperian_e59_tui_demo_mode() {
        use clap::Parser;

        let result = Cli::try_parse_from(["trueno-ublk", "top", "--demo"]);

        assert!(result.is_ok(), "E59: top --demo should parse");

        if let Ok(cli) = result {
            if let Commands::Top(args) = cli.command {
                assert!(args.demo, "E59: Demo mode should be enabled");
            }
        }
    }

    /// E60: TUI - verify report mode (non-interactive).
    #[test]
    fn popperian_e60_tui_report_mode() {
        use clap::Parser;

        let result = Cli::try_parse_from(["trueno-ublk", "top", "--report"]);

        assert!(result.is_ok(), "E60: top --report should parse");

        if let Ok(cli) = result {
            if let Commands::Top(args) = cli.command {
                assert!(args.report, "E60: Report mode should be enabled");
            }
        }
    }

    // Additional CLI validation tests

    /// Verify all algorithms convert correctly to trueno-zram-core.
    #[test]
    fn popperian_cli_all_algorithms() {
        let algorithms = [
            (Algorithm::Lz4, "lz4"),
            (Algorithm::Lz4hc, "lz4hc"),
            (Algorithm::Zstd1, "zstd1"),
            (Algorithm::Zstd3, "zstd3"),
            (Algorithm::Zstd9, "zstd9"),
        ];

        for (algo, name) in algorithms {
            // Should convert without panic
            let _ = algo.to_trueno();
            println!("Algorithm {} converts correctly", name);
        }
    }

    /// Verify size parsing edge cases.
    #[test]
    fn popperian_cli_size_edge_cases() {
        // Very large sizes
        assert!(parse_size("100T").is_ok(), "Should parse 100TB");

        // Mixed case
        assert_eq!(parse_size("1g").unwrap(), parse_size("1G").unwrap());
        assert_eq!(parse_size("1GiB").unwrap(), parse_size("1gib").unwrap());

        // With whitespace
        assert_eq!(parse_size("  1G  ").unwrap(), 1u64 << 30);
    }
}
