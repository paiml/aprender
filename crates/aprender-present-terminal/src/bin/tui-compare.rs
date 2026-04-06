//! TUI Comparison Tool
//!
//! Compares two TUI outputs using CIEDE2000, CLD, and SSIM metrics.
//!
//! Run: cargo run --bin tui-compare --features tui-compare -- --help

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use clap::Parser;

use presentar_terminal::direct::CellBuffer;
use presentar_terminal::tools::{compare_tui, generate_report, TuiComparisonConfig};

/// TUI Comparison Tool - Pixel-perfect verification for terminal UIs using CIEDE2000, CLD, and SSIM metrics
#[derive(Parser)]
#[command(name = "tui-compare", version, about = "TUI pixel comparison using CIEDE2000 (ΔE00), CLD, and SSIM metrics", long_about = None)]
struct Cli {
    /// Reference file (.ans) or command to run
    #[arg(short, long)]
    reference: String,

    /// Target file (.ans) or command to run
    #[arg(short, long)]
    target: String,

    /// Terminal size (`WIDTHxHEIGHT`)
    #[arg(long, default_value = "120x40")]
    size: String,

    /// Output file for report
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// CLD threshold (0.0-1.0)
    #[arg(long, default_value = "0.01")]
    threshold_cld: f64,

    /// ΔE00 threshold
    #[arg(long, default_value = "2.0")]
    threshold_delta_e: f64,

    /// SSIM threshold (0.0-1.0)
    #[arg(long, default_value = "0.95")]
    threshold_ssim: f64,

    /// Generate diff visualization
    #[arg(long)]
    diff_output: Option<PathBuf>,

    /// Quiet mode (only output pass/fail)
    #[arg(short, long)]
    quiet: bool,
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();

    // Parse terminal size
    let (width, height) = parse_size(&cli.size)?;

    // Load or capture reference
    let reference_buffer = load_or_capture(&cli.reference, width, height)?;

    // Load or capture target
    let target_buffer = load_or_capture(&cli.target, width, height)?;

    // Configure comparison
    let config = TuiComparisonConfig {
        cld_threshold: cli.threshold_cld,
        delta_e_threshold: cli.threshold_delta_e,
        ssim_threshold: cli.threshold_ssim,
        ..Default::default()
    };

    // Run comparison
    let result = compare_tui(&reference_buffer, &target_buffer, &config);

    // Generate report
    let report = generate_report(&result, &config);

    // Output report
    if let Some(output_path) = &cli.output {
        fs::write(output_path, &report)?;
    }

    if !cli.quiet {
        print!("{report}");
    }

    // Generate diff visualization if requested
    if let Some(diff_path) = &cli.diff_output {
        let diff_text = generate_diff_text(&reference_buffer, &target_buffer, &result);
        fs::write(diff_path, diff_text)?;
    }

    // Exit with appropriate code
    if result.passed {
        if !cli.quiet {
            eprintln!("\n✓ Comparison PASSED");
        }
        Ok(())
    } else {
        if !cli.quiet {
            eprintln!("\n✗ Comparison FAILED");
        }
        std::process::exit(1);
    }
}

fn parse_size(size: &str) -> io::Result<(u16, u16)> {
    let parts: Vec<&str> = size.split('x').collect();
    if parts.len() != 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Size must be in WIDTHxHEIGHT format",
        ));
    }

    let width: u16 = parts[0]
        .parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Invalid width"))?;

    let height: u16 = parts[1]
        .parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Invalid height"))?;

    Ok((width, height))
}

fn load_or_capture(source: &str, width: u16, height: u16) -> io::Result<CellBuffer> {
    let path = PathBuf::from(source);

    if path.exists() {
        // Load from file
        load_ansi_file(&path, width, height)
    } else if source.contains(' ') || source.starts_with("./") {
        // Treat as command
        capture_command(source, width, height)
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("File not found: {source}"),
        ))
    }
}

fn load_ansi_file(path: &PathBuf, width: u16, height: u16) -> io::Result<CellBuffer> {
    let content = fs::read_to_string(path)?;
    Ok(parse_ansi_to_buffer(&content, width, height))
}

fn capture_command(cmd: &str, width: u16, height: u16) -> io::Result<CellBuffer> {
    // Run command and capture output
    let output = Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .env("COLUMNS", width.to_string())
        .env("LINES", height.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()?;

    let content = String::from_utf8_lossy(&output.stdout);
    Ok(parse_ansi_to_buffer(&content, width, height))
}

/// Skip an ANSI escape sequence (CSI sequence).
fn skip_ansi_escape(chars: &mut std::iter::Peekable<std::str::Chars>) {
    if chars.peek() == Some(&'[') {
        chars.next(); // consume '['
                      // Skip parameters and command
        while let Some(&c) = chars.peek() {
            chars.next();
            if c.is_ascii_alphabetic() {
                break;
            }
        }
    }
}

/// Parse ANSI escape sequences into a `CellBuffer`
fn parse_ansi_to_buffer(content: &str, width: u16, height: u16) -> CellBuffer {
    let mut buffer = CellBuffer::new(width, height);
    let mut x = 0u16;
    let mut y = 0u16;
    let mut chars = content.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '\x1b' => skip_ansi_escape(&mut chars),
            '\n' => {
                x = 0;
                y = y.saturating_add(1);
            }
            '\r' => x = 0,
            _ => {
                if x < width && y < height {
                    buffer.set_char(x, y, ch);
                    x = x.saturating_add(1);
                }
            }
        }
    }

    buffer
}

/// Generate text-based diff visualization
fn generate_diff_text(
    reference: &CellBuffer,
    target: &CellBuffer,
    result: &presentar_terminal::tools::TuiComparisonResult,
) -> String {
    let mut output = String::new();
    output.push_str("=== DIFF VISUALIZATION ===\n\n");

    if result.diff_cells.is_empty() {
        output.push_str("No differences found.\n");
        return output;
    }

    let _ = write!(
        output,
        "Found {} differing cells:\n\n",
        result.diff_cells.len()
    );

    // Group by row
    let mut by_row: std::collections::HashMap<u16, Vec<_>> = std::collections::HashMap::new();
    for cell in &result.diff_cells {
        by_row.entry(cell.y).or_default().push(cell);
    }

    let mut rows: Vec<_> = by_row.keys().copied().collect();
    rows.sort_unstable();

    for row in rows.iter().take(20) {
        // Limit output
        let _ = writeln!(output, "Row {row}:");

        // Show reference row
        output.push_str("  REF: ");
        for x in 0..reference.width().min(80) {
            let ch = reference
                .get(x, *row)
                .and_then(|c| c.symbol.chars().next())
                .unwrap_or(' ');
            output.push(ch);
        }
        output.push('\n');

        // Show target row
        output.push_str("  TGT: ");
        for x in 0..target.width().min(80) {
            let ch = target
                .get(x, *row)
                .and_then(|c| c.symbol.chars().next())
                .unwrap_or(' ');
            output.push(ch);
        }
        output.push('\n');

        // Show diff markers
        output.push_str("  DIF: ");
        let empty = Vec::new();
        let row_diffs = by_row.get(row).unwrap_or(&empty);
        for x in 0..reference.width().min(80) {
            if row_diffs.iter().any(|d| d.x == x) {
                output.push('^');
            } else {
                output.push(' ');
            }
        }
        output.push_str("\n\n");
    }

    if rows.len() > 20 {
        let _ = writeln!(
            output,
            "... and {} more rows with differences",
            rows.len() - 20
        );
    }

    output
}
