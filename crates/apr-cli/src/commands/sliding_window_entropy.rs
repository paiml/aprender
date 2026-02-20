
fn print_distribution(analysis: &DistributionAnalysis) {
    if analysis.total == 0 {
        println!("  {}", "No data".dimmed());
        return;
    }

    let max_count = analysis
        .histogram
        .iter()
        .map(|(_, _, c)| *c)
        .max()
        .unwrap_or(1);

    for (start, end, count) in &analysis.histogram {
        let bar_width = if max_count > 0 {
            (*count * 40) / max_count
        } else {
            0
        };
        let pct = if analysis.total > 0 {
            *count as f64 / analysis.total as f64 * 100.0
        } else {
            0.0
        };
        let bar = "█".repeat(bar_width);
        println!(
            "  {} {} {}",
            format!("[{start:>8.3}, {end:>8.3})").dimmed(),
            format!("{bar:<40}").green(),
            format!("{pct:>5.1}%").white().bold()
        );
    }

    println!();
    output::metric("Entropy", format!("{:.2} bits", analysis.entropy), "");
    output::metric("Kurtosis", format!("{:.2}", analysis.kurtosis), "");
    output::metric("Skewness", format!("{:.4}", analysis.skewness), "");
    output::metric("Min", format!("{:.6}", analysis.min), "");
    output::metric("Max", format!("{:.6}", analysis.max), "");
    output::metric("Mean", format!("{:.6}", analysis.mean), "");
    output::metric("Std", format!("{:.6}", analysis.std), "");
    if analysis.nan_count > 0 {
        println!(
            "  {} {} NaN values",
            "Warning:".red().bold(),
            output::count_fmt(analysis.nan_count)
        );
    }
    if analysis.inf_count > 0 {
        println!(
            "  {} {} Inf values",
            "Warning:".red().bold(),
            output::count_fmt(analysis.inf_count)
        );
    }
    if analysis.zero_count > 0 {
        output::metric(
            "Zero values",
            format!(
                "{} ({:.1}%)",
                output::count_fmt(analysis.zero_count),
                analysis.zero_count as f64 / analysis.total as f64 * 100.0
            ),
            "",
        );
    }
}

// ============================================================================
// --entropy: Byte entropy analysis
// ============================================================================

/// Compute Shannon entropy of byte distribution (0.0 = all same, 8.0 = uniform random)
fn compute_byte_entropy(bytes: &[u8]) -> f64 {
    if bytes.is_empty() {
        return 0.0;
    }

    let mut counts = [0u64; 256];
    for &b in bytes {
        counts[b as usize] += 1;
    }

    let total = bytes.len() as f64;
    let mut entropy = 0.0_f64;
    for &count in &counts {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Result of sliding-window entropy analysis over a byte buffer.
struct SlidingWindowEntropy {
    min_entropy: f64,
    max_entropy: f64,
    min_offset: usize,
    max_offset: usize,
    /// Regions where entropy < 1.0 (possible all-zeros or corruption).
    anomalous_regions: Vec<(usize, f64)>,
}

/// Compute sliding-window entropy statistics over `bytes` using 4KB windows.
fn compute_sliding_window_entropy(bytes: &[u8]) -> SlidingWindowEntropy {
    let window_size = 4096;
    let step = (bytes.len() / 100).max(window_size); // ~100 samples
    let mut result = SlidingWindowEntropy {
        min_entropy: f64::MAX,
        max_entropy: f64::MIN,
        min_offset: 0,
        max_offset: 0,
        anomalous_regions: Vec::new(),
    };

    let mut offset = 0;
    while offset + window_size <= bytes.len() {
        let e = compute_byte_entropy(&bytes[offset..offset + window_size]);

        if e < result.min_entropy {
            result.min_entropy = e;
            result.min_offset = offset;
        }
        if e > result.max_entropy {
            result.max_entropy = e;
            result.max_offset = offset;
        }
        if e < 1.0 {
            result.anomalous_regions.push((offset, e));
        }

        offset += step;
    }
    result
}

fn print_entropy_analysis(bytes: &[u8], format: FileFormat) {
    output::header(&format!(
        "Byte Entropy Analysis ({})",
        format_display_name(format)
    ));

    if bytes.is_empty() {
        println!("  {}", "Empty file".dimmed());
        return;
    }

    let total_entropy = compute_byte_entropy(bytes);
    output::metric(
        "Total entropy",
        format!("{total_entropy:.4} bits"),
        "(0.0 = uniform, 8.0 = random)",
    );
    output::metric("File size", output::format_size(bytes.len() as u64), "");

    // Expected entropy ranges by format
    let expected = match format {
        FileFormat::Gguf => "Q4K/Q6K: 7.5-8.0, F32: 5.0-7.5, F16: 6.0-7.5",
        FileFormat::Apr => "F32: 5.0-7.5, F16: 6.0-7.5",
        FileFormat::SafeTensors => "F32: 5.0-7.5, F16: 6.0-7.5, BF16: 6.0-7.5",
    };
    output::metric("Expected range", expected, "");

    // Sliding window analysis (4KB windows)
    if bytes.len() >= 4096 {
        let sw = compute_sliding_window_entropy(bytes);

        output::subheader("Sliding Window (4KB)");
        output::metric(
            "Min entropy",
            format!("{:.4} at 0x{:X}", sw.min_entropy, sw.min_offset),
            "",
        );
        output::metric(
            "Max entropy",
            format!("{:.4} at 0x{:X}", sw.max_entropy, sw.max_offset),
            "",
        );

        if !sw.anomalous_regions.is_empty() {
            println!(
                "\n  {} {} anomalous regions (entropy < 1.0):",
                "Warning:".yellow().bold(),
                sw.anomalous_regions.len()
            );
            for (off, e) in sw.anomalous_regions.iter().take(5) {
                println!("    0x{off:08X}: entropy={e:.4} (possible all-zeros or padding)");
            }
            if sw.anomalous_regions.len() > 5 {
                println!("    ... and {} more", sw.anomalous_regions.len() - 5);
            }
        }
    }
}

// ============================================================================
// --contract: Layout contract overlay
// ============================================================================

fn print_contract_overlay(info: &GgufInfo) {
    use aprender::format::layout_contract::contract;

    output::header("Layout Contract Overlay (GGUF → APR)");

    let layout = contract();
    let mut pass_count = 0;
    let mut miss_count = 0;

    let headers = &["GGUF Name", "APR Name", "Transpose", "Critical", "Status"];
    let mut rows = Vec::new();

    for tensor in &info.tensors {
        if let Some(tc) = layout.get_gguf_contract(&tensor.name) {
            let status = output::badge_pass("Mapped");
            rows.push(vec![
                tensor.name.clone(),
                tc.apr_name.to_string(),
                if tc.should_transpose {
                    "Yes".to_string()
                } else {
                    "No".to_string()
                },
                if tc.is_critical {
                    "CRITICAL".to_string()
                } else {
                    "-".to_string()
                },
                status,
            ]);
            pass_count += 1;
        } else {
            miss_count += 1;
        }
    }

    if !rows.is_empty() {
        println!("{}", output::table(headers, &rows));
    }

    println!();
    output::metric("Mapped tensors", output::count_fmt(pass_count), "");
    output::metric(
        "Unmapped tensors",
        output::count_fmt(miss_count),
        "(norm weights, etc.)",
    );
    output::metric("Total", output::count_fmt(info.tensors.len()), "");
}

// ============================================================================
// Utility functions
// ============================================================================

/// Convert IEEE 754 half-precision (f16) to single-precision (f32)
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // +/- zero
            return f32::from_bits(sign << 31);
        }
        // Subnormal: normalize
        let mut e = 0_i32;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let exp32 = (127 - 15 + 1 - e) as u32;
        let mant32 = (m & 0x3FF) << 13;
        return f32::from_bits((sign << 31) | (exp32 << 23) | mant32);
    }

    if exp == 31 {
        if mant == 0 {
            // +/- infinity
            return f32::from_bits((sign << 31) | (0xFF << 23));
        }
        // NaN
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13));
    }

    // Normal number: (exp - 15 + 127) rewritten to avoid u32 underflow
    let exp32 = exp + 112; // 112 = 127 - 15
    let mant32 = mant << 13;
    f32::from_bits((sign << 31) | (exp32 << 23) | mant32)
}

/// GGML dtype name from u32 discriminant
fn ggml_dtype_name(dtype: u32) -> &'static str {
    match dtype {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        _ => "Unknown",
    }
}

/// Print an annotated hex field
fn print_annotated_field(offset: usize, bytes: &[u8], label: &str, value: &str) {
    print!("  {}", format!("{offset:08X}: ").dimmed());

    // Show up to 8 bytes of hex
    let display_len = bytes.len().min(8);
    for b in &bytes[..display_len] {
        print!("{}", format!("{b:02X} ").yellow());
    }
    if bytes.len() > 8 {
        print!("{}", ".. ".yellow());
    }

    // Pad to align annotations (8 bytes * 3 chars = 24, plus ".. " = 27)
    let hex_width = display_len * 3 + if bytes.len() > 8 { 3 } else { 0 };
    let padding = 28usize.saturating_sub(hex_width);
    print!("{:width$}", "", width = padding);

    println!("{}: {}", label.white().bold(), value.cyan());
}

/// Parse hex offset string (supports "0x" prefix)
pub(crate) fn parse_hex_offset(s: &str) -> Result<usize, String> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        usize::from_str_radix(hex, 16).map_err(|e| format!("Invalid hex offset: {e}"))
    } else {
        s.parse::<usize>()
            .map_err(|e| format!("Invalid offset: {e}"))
    }
}

// ============================================================================
// Preserved helpers (APR tensor display)
// ============================================================================

/// List tensor names (v2 reader)
#[allow(clippy::unnecessary_wraps)]
#[allow(clippy::disallowed_methods)]
fn list_tensors_v2(
    reader: &AprV2Reader,
    filtered: &[&str],
    json_output: bool,
) -> Result<(), CliError> {
    if json_output {
        let json = serde_json::json!({
            "tensors": filtered,
            "count": filtered.len()
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        println!("{}", "Tensors:".bold());
        for name in filtered {
            if let Some(entry) = reader.get_tensor(name) {
                println!(
                    "  {}: {:?} ({} bytes, dtype={:?})",
                    name.cyan(),
                    entry.shape,
                    entry.size,
                    entry.dtype
                );
            } else {
                println!("  {}", name);
            }
        }
        println!("\n{} tensors total", filtered.len().to_string().cyan());
    }
    Ok(())
}

/// Print tensor header information (v2 reader)
fn print_tensor_header_v2(entry: &TensorIndexEntry) {
    println!("{}", "═".repeat(70));
    println!("{}: {}", "Tensor".bold(), entry.name.cyan());
    println!("{}", "═".repeat(70));

    let num_elements: usize = entry.shape.iter().product();
    println!(
        "{}: {:?} = {} elements",
        "Shape".bold(),
        entry.shape,
        num_elements.to_string().green()
    );
    println!("{}: {:?}", "Dtype".bold(), entry.dtype);
    println!(
        "{}: 0x{:08X} ({} bytes)",
        "Offset".bold(),
        entry.offset,
        entry.offset
    );
    println!(
        "{}: {} bytes",
        "Size".bold(),
        entry.size.to_string().yellow()
    );
}

/// Check for tensor anomalies and print warnings
fn print_tensor_anomalies(min: f32, max: f32, mean: f32, std: f32) {
    if min.is_nan() || max.is_nan() || mean.is_nan() {
        println!("  {} NaN values detected!", "Warning:".red());
    }
    if min.is_infinite() || max.is_infinite() {
        println!("  {} Infinite values detected!", "Warning:".red());
    }
    if std < 1e-10 {
        println!(
            "  {} Very low variance - possible collapsed weights!",
            "Warning:".yellow()
        );
    }
}

/// Print statistics for tensor data
fn print_tensor_stats(data: &[f32]) {
    println!();
    println!("{}", "Statistics:".bold());
    let (min, max, mean, std) = compute_stats(data);
    println!("  min={min:.6}  max={max:.6}  mean={mean:.6}  std={std:.6}");
    print_tensor_anomalies(min, max, mean, std);
}

/// Print a hex dump row for a chunk of float values
fn print_hex_row(chunk: &[&f32], row_offset: usize) {
    print!("{}", format!("{row_offset:08X}: ").dimmed());

    for &val in chunk {
        let bytes = val.to_le_bytes();
        for b in &bytes {
            print!("{}", format!("{b:02X} ").yellow());
        }
    }

    let padding = (4 - chunk.len()) * 12;
    print!("{:width$}", "", width = padding);

    print!("{}", " | ".dimmed());
    for &val in chunk {
        let color_val = if *val == 0.0 {
            format!("{val:>10.4} ").dimmed().to_string()
        } else if *val < 0.0 {
            format!("{val:>10.4} ").red().to_string()
        } else {
            format!("{val:>10.4} ").green().to_string()
        };
        print!("{color_val}");
    }
    println!();
}
