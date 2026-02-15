
fn print_safetensors_file_header(bytes: &[u8]) {
    if bytes.len() < 9 {
        println!("  {} File too small for SafeTensors header", "Error:".red());
        return;
    }

    let header_len = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    print_annotated_field(
        0,
        &bytes[0..8],
        "header_length",
        &format!("{header_len} bytes"),
    );

    let header_end = (8 + header_len as usize).min(bytes.len());
    let preview_end = header_end.min(8 + 200); // Show first 200 chars of JSON
    if let Ok(json_preview) = std::str::from_utf8(&bytes[8..preview_end]) {
        println!("\n  {} (first 200 chars):", "JSON Header".bold());
        // Pretty-print if it's valid JSON
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(
            std::str::from_utf8(&bytes[8..header_end]).unwrap_or(""),
        ) {
            if let Ok(pretty) = serde_json::to_string_pretty(&parsed) {
                for (i, line) in pretty.lines().take(20).enumerate() {
                    println!("    {line}");
                    if i == 19 {
                        println!("    ...");
                    }
                }
            }
        } else {
            println!("    {json_preview}");
        }
    }
}

// ============================================================================
// --raw: Format-aware xxd
// ============================================================================

fn print_raw_hex(bytes: &[u8], offset: usize, limit: usize, width: usize) {
    let width = if width == 0 { 16 } else { width };
    let start = offset.min(bytes.len());
    let end = (start + limit).min(bytes.len());
    let slice = &bytes[start..end];

    if slice.is_empty() {
        println!("{}", "No bytes to display at this offset".yellow());
        return;
    }

    for (i, chunk) in slice.chunks(width).enumerate() {
        let addr = start + i * width;
        print_raw_hex_row(addr, chunk, width);
    }

    if end < bytes.len() {
        println!(
            "... {} more bytes",
            output::count_fmt(bytes.len() - end).dimmed()
        );
    }
}

/// Print a single raw hex dump row: offset | hex bytes | ASCII.
fn print_raw_hex_row(addr: usize, chunk: &[u8], width: usize) {
    print!("{}", format!("{addr:08X}: ").dimmed());

    // Hex bytes with midpoint separator
    for (j, &b) in chunk.iter().enumerate() {
        if j == width / 2 && width >= 8 {
            print!(" ");
        }
        print!("{}", format!("{b:02X} ").yellow());
    }
    // Pad short rows
    let missing = width - chunk.len();
    for j in 0..missing {
        if chunk.len() + j == width / 2 && width >= 8 {
            print!(" ");
        }
        print!("   ");
    }

    // ASCII column
    print!(" |");
    for &b in chunk {
        if b.is_ascii_graphic() || b == b' ' {
            print!("{}", (b as char).to_string().white());
        } else {
            print!("{}", ".".dimmed());
        }
    }
    for _ in 0..missing {
        print!(" ");
    }
    println!("|");
}

// ============================================================================
// --blocks: Quantization super-block inspection
// ============================================================================

/// Q4K block size: 144 bytes per 256 elements
const Q4K_BLOCK_SIZE: usize = 144;
/// Q6K block size: 210 bytes per 256 elements
const Q6K_BLOCK_SIZE: usize = 210;
/// Q8_0 block size: 34 bytes per 32 elements
const Q8_0_BLOCK_SIZE: usize = 34;

fn print_tensor_blocks(
    file_bytes: &[u8],
    tensor: &GgufTensorEntry,
    byte_offset: usize,
) -> Result<(), CliError> {
    let dtype_name = ggml_dtype_name(tensor.dtype);
    let dims_str: Vec<String> = tensor
        .dims
        .iter()
        .map(std::string::ToString::to_string)
        .collect();

    output::header(&format!(
        "Block View: {} ({}, [{}])",
        tensor.name,
        dtype_name,
        dims_str.join(", ")
    ));

    match tensor.dtype {
        12 => print_q4k_blocks(file_bytes, byte_offset, 3), // Q4K, show 3 blocks
        14 => print_q6k_blocks(file_bytes, byte_offset, 3), // Q6K
        8 => print_q8_0_blocks(file_bytes, byte_offset, 3), // Q8_0
        _ => {
            println!(
                "  {}",
                output::badge_info(&format!("Block view not applicable for dtype {dtype_name}"))
            );
        }
    }
    Ok(())
}

fn print_q4k_blocks(file_bytes: &[u8], base_offset: usize, count: usize) {
    for block_idx in 0..count {
        let offset = base_offset + block_idx * Q4K_BLOCK_SIZE;
        if offset + Q4K_BLOCK_SIZE > file_bytes.len() {
            println!(
                "  {} Block #{block_idx} exceeds file bounds",
                "Warn:".yellow()
            );
            break;
        }
        let block = &file_bytes[offset..offset + Q4K_BLOCK_SIZE];

        println!(
            "\n  {} (256 elements, {Q4K_BLOCK_SIZE} bytes):",
            format!("Q4_K Super-Block #{block_idx}").cyan().bold()
        );

        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        print_annotated_field(0, &block[0..2], "d (scale)", &format!("{d:.5} (f16)"));

        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        print_annotated_field(2, &block[2..4], "dmin", &format!("{dmin:.5} (f16)"));

        print_annotated_field(4, &block[4..16], "scales[0-11]", "12 sub-block scales");
        print_annotated_field(
            16,
            &block[16..Q4K_BLOCK_SIZE],
            "qs[0-127]",
            "4-bit packed (256 values)",
        );
    }
}

fn print_q6k_blocks(file_bytes: &[u8], base_offset: usize, count: usize) {
    for block_idx in 0..count {
        let offset = base_offset + block_idx * Q6K_BLOCK_SIZE;
        if offset + Q6K_BLOCK_SIZE > file_bytes.len() {
            println!(
                "  {} Block #{block_idx} exceeds file bounds",
                "Warn:".yellow()
            );
            break;
        }
        let block = &file_bytes[offset..offset + Q6K_BLOCK_SIZE];

        println!(
            "\n  {} (256 elements, {Q6K_BLOCK_SIZE} bytes):",
            format!("Q6_K Super-Block #{block_idx}").cyan().bold()
        );

        print_annotated_field(
            0,
            &block[0..128],
            "ql[0-127]",
            "low 4 bits (256 values, 2/byte)",
        );
        print_annotated_field(
            128,
            &block[128..192],
            "qh[0-63]",
            "high 2 bits (256 values, 4/byte)",
        );
        print_annotated_field(192, &block[192..208], "scales[0-15]", "16 sub-block scales");

        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        print_annotated_field(208, &block[208..210], "d (scale)", &format!("{d:.5} (f16)"));
    }
}

fn print_q8_0_blocks(file_bytes: &[u8], base_offset: usize, count: usize) {
    for block_idx in 0..count {
        let offset = base_offset + block_idx * Q8_0_BLOCK_SIZE;
        if offset + Q8_0_BLOCK_SIZE > file_bytes.len() {
            println!(
                "  {} Block #{block_idx} exceeds file bounds",
                "Warn:".yellow()
            );
            break;
        }
        let block = &file_bytes[offset..offset + Q8_0_BLOCK_SIZE];

        println!(
            "\n  {} (32 elements, {Q8_0_BLOCK_SIZE} bytes):",
            format!("Q8_0 Block #{block_idx}").cyan().bold()
        );

        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        print_annotated_field(0, &block[0..2], "scale", &format!("{scale:.5} (f16)"));
        print_annotated_field(
            2,
            &block[2..Q8_0_BLOCK_SIZE],
            "quants[0-31]",
            "i8 values [-128..127]",
        );
    }
}

// ============================================================================
// --distribution: Histogram + entropy + kurtosis
// ============================================================================

struct DistributionAnalysis {
    histogram: Vec<(f64, f64, usize)>, // (bin_start, bin_end, count)
    total: usize,
    entropy: f64,
    kurtosis: f64,
    skewness: f64,
    nan_count: usize,
    inf_count: usize,
    zero_count: usize,
    min: f32,
    max: f32,
    mean: f64,
    std: f64,
}

/// First-pass scan of f32 data: counts NaN/Inf/zero, min/max, sum, valid count.
struct ValueScan {
    nan_count: usize,
    inf_count: usize,
    zero_count: usize,
    min: f32,
    max: f32,
    sum: f64,
    valid_count: usize,
}

fn scan_values(data: &[f32]) -> ValueScan {
    let mut s = ValueScan {
        nan_count: 0,
        inf_count: 0,
        zero_count: 0,
        min: f32::INFINITY,
        max: f32::NEG_INFINITY,
        sum: 0.0,
        valid_count: 0,
    };
    for &x in data {
        if x.is_nan() {
            s.nan_count += 1;
            continue;
        }
        if x.is_infinite() {
            s.inf_count += 1;
            continue;
        }
        if x == 0.0 {
            s.zero_count += 1;
        }
        s.min = s.min.min(x);
        s.max = s.max.max(x);
        s.sum += f64::from(x);
        s.valid_count += 1;
    }
    s
}

/// Second-pass: variance, skewness, kurtosis from mean.
fn compute_moments(data: &[f32], mean: f64, valid_count: usize) -> (f64, f64, f64) {
    let (mut m2, mut m3, mut m4) = (0.0_f64, 0.0_f64, 0.0_f64);
    for &x in data {
        if x.is_nan() || x.is_infinite() {
            continue;
        }
        let d = f64::from(x) - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    let variance = m2 / valid_count as f64;
    let std = variance.sqrt();
    let skewness = if std > 0.0 {
        (m3 / valid_count as f64) / (std * std * std)
    } else {
        0.0
    };
    let kurtosis = if std > 0.0 {
        (m4 / valid_count as f64) / (variance * variance)
    } else {
        0.0
    };
    (std, skewness, kurtosis)
}

/// Build histogram bins for valid (non-NaN, non-Inf) values.
fn build_histogram(
    data: &[f32],
    min: f32,
    max: f32,
    num_bins: usize,
    valid_count: usize,
) -> (Vec<(f64, f64, usize)>, f64) {
    let range = f64::from(max) - f64::from(min);
    let bin_width = if range > 0.0 {
        range / num_bins as f64
    } else {
        1.0
    };
    let mut bins = vec![0usize; num_bins];
    for &x in data {
        if x.is_nan() || x.is_infinite() {
            continue;
        }
        let idx = (((f64::from(x) - f64::from(min)) / bin_width) as usize).min(num_bins - 1);
        bins[idx] += 1;
    }
    let histogram: Vec<(f64, f64, usize)> = bins
        .iter()
        .enumerate()
        .map(|(i, &count)| {
            let start = f64::from(min) + i as f64 * bin_width;
            (start, start + bin_width, count)
        })
        .collect();
    let entropy: f64 = histogram
        .iter()
        .filter(|(_, _, c)| *c > 0)
        .map(|(_, _, c)| {
            let p = *c as f64 / valid_count as f64;
            -p * p.log2()
        })
        .sum();
    (histogram, entropy)
}

fn empty_distribution(total: usize, scan: &ValueScan) -> DistributionAnalysis {
    DistributionAnalysis {
        histogram: Vec::new(),
        total,
        entropy: 0.0,
        kurtosis: 0.0,
        skewness: 0.0,
        nan_count: scan.nan_count,
        inf_count: scan.inf_count,
        zero_count: scan.zero_count,
        min: 0.0,
        max: 0.0,
        mean: 0.0,
        std: 0.0,
    }
}

fn compute_distribution(data: &[f32]) -> DistributionAnalysis {
    if data.is_empty() {
        let empty_scan = ValueScan {
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            sum: 0.0,
            valid_count: 0,
        };
        return empty_distribution(0, &empty_scan);
    }

    let scan = scan_values(data);
    if scan.valid_count == 0 {
        return empty_distribution(data.len(), &scan);
    }

    let mean = scan.sum / scan.valid_count as f64;
    let (std, skewness, kurtosis) = compute_moments(data, mean, scan.valid_count);
    let (histogram, entropy) = build_histogram(data, scan.min, scan.max, 10, scan.valid_count);

    DistributionAnalysis {
        histogram,
        total: data.len(),
        entropy,
        kurtosis,
        skewness,
        nan_count: scan.nan_count,
        inf_count: scan.inf_count,
        zero_count: scan.zero_count,
        min: scan.min,
        max: scan.max,
        mean,
        std,
    }
}
