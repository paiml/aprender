//! Hex Forensics — Binary Model Inspection Demo
//!
//! Demonstrates the algorithms behind `apr hex`: Shannon entropy, value
//! distribution analysis, f16→f32 conversion, and quantization block layout.
//!
//! Toyota Way Alignment:
//! - **Genchi Genbutsu**: Go and see the actual bytes
//! - **Jidoka**: Stop on anomalies — flag NaN, corruption, zero regions
//! - **Visualization**: Make binary structure visible
//!
//! Run with: `cargo run --example hex_forensics`
//!
//! CLI equivalents:
//! ```bash
//! apr hex model.gguf --header              # Annotated file header
//! apr hex model.gguf --raw --width 32      # Format-aware xxd
//! apr hex model.gguf --blocks --tensor q   # Q4K/Q6K super-block view
//! apr hex model.gguf --distribution        # Histogram + entropy + kurtosis
//! apr hex model.gguf --entropy             # Per-region byte entropy
//! apr hex model.gguf --contract            # Layout contract overlay
//! ```

fn main() {
    println!("=== Hex Forensics: Binary Model Inspection ===\n");

    // Part 1: Format Detection via Magic Bytes
    format_detection_demo();

    // Part 2: IEEE 754 Half-Precision (f16) Conversion
    f16_conversion_demo();

    // Part 3: Shannon Entropy for Corruption Detection
    entropy_demo();

    // Part 4: Value Distribution Analysis
    distribution_demo();

    // Part 5: Quantization Block Layout
    block_layout_demo();

    println!("\n=== Hex Forensics Demo Complete ===");
}

// ============================================================================
// Part 1: Format Detection
// ============================================================================

fn format_detection_demo() {
    println!("--- Part 1: Format Detection via Magic Bytes ---\n");

    let samples: &[(&[u8], &str)] = &[
        (b"GGUF", "GGUF (quantized LLM weights)"),
        (b"APR\x00", "APR (Aprender native format)"),
        (b"\x08\x00\x00\x00", "SafeTensors (8-byte LE header length)"),
        (b"\x89PNG", "PNG (not a model file)"),
    ];

    for (magic, desc) in samples {
        let detected = match &magic[..4.min(magic.len())] {
            b"GGUF" => "GGUF",
            b"APR\x00" => "APR",
            _ => {
                // SafeTensors: first 8 bytes are u64 LE header length
                if magic.len() >= 4 {
                    let len = u32::from_le_bytes([magic[0], magic[1], magic[2], magic[3]]);
                    if len < 100_000_000 {
                        "SafeTensors (probable)"
                    } else {
                        "Unknown"
                    }
                } else {
                    "Unknown"
                }
            }
        };
        println!(
            "  {:02X} {:02X} {:02X} {:02X} → {} ({})",
            magic[0], magic[1], magic[2], magic[3], detected, desc
        );
    }
    println!();
}

// ============================================================================
// Part 2: f16 → f32 Conversion
// ============================================================================

/// IEEE 754 half-precision to single-precision conversion
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31); // ±0
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
            return f32::from_bits((sign << 31) | (0xFF << 23)); // ±Inf
        }
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13)); // NaN
    }

    // Normal: exp + 112 avoids u32 underflow (112 = 127 - 15)
    let exp32 = exp + 112;
    let mant32 = mant << 13;
    f32::from_bits((sign << 31) | (exp32 << 23) | mant32)
}

fn f16_conversion_demo() {
    println!("--- Part 2: IEEE 754 Half-Precision (f16 → f32) ---\n");

    let test_cases: &[(u16, &str)] = &[
        (0x0000, "+0.0"),
        (0x8000, "-0.0"),
        (0x3C00, "1.0"),
        (0xBC00, "-1.0"),
        (0x3800, "0.5"),
        (0x7C00, "+Inf"),
        (0xFC00, "-Inf"),
        (0x7E00, "NaN"),
        (0x0001, "smallest subnormal"),
    ];

    for (bits, label) in test_cases {
        let result = f16_to_f32(*bits);
        println!("  0x{bits:04X} ({label:>20}) → {result:.10}");
    }
    println!();
}

// ============================================================================
// Part 3: Shannon Entropy
// ============================================================================

/// Shannon entropy: H = -Σ p(x) * log2(p(x))
/// 0.0 = all bytes identical, 8.0 = perfectly uniform random
fn compute_byte_entropy(bytes: &[u8]) -> f64 {
    if bytes.is_empty() {
        return 0.0;
    }
    let mut counts = [0u64; 256];
    for &b in bytes {
        counts[b as usize] += 1;
    }
    let len = bytes.len() as f64;
    let mut entropy = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / len;
            entropy -= p * p.log2();
        }
    }
    entropy
}

fn entropy_demo() {
    println!("--- Part 3: Shannon Entropy for Corruption Detection ---\n");

    // All zeros — entropy 0.0 (corruption signal)
    let all_zeros = vec![0u8; 4096];
    println!(
        "  All zeros (4KB):     {:.4} bits  ← ANOMALY: possible corruption",
        compute_byte_entropy(&all_zeros)
    );

    // Single repeated byte — entropy 0.0
    let repeated = vec![0x42u8; 4096];
    println!(
        "  Repeated 0x42 (4KB): {:.4} bits  ← ANOMALY: uninitialized memory",
        compute_byte_entropy(&repeated)
    );

    // Two alternating values — entropy 1.0
    let alternating: Vec<u8> = (0..4096).map(|i| if i % 2 == 0 { 0xAA } else { 0x55 }).collect();
    println!(
        "  Alternating AA/55:   {:.4} bits  ← Low: patterned data",
        compute_byte_entropy(&alternating)
    );

    // Pseudo-random (good quantized weights) — entropy ~7.8-8.0
    let pseudo_random: Vec<u8> = (0..4096u64)
        .map(|i| ((i.wrapping_mul(1103515245).wrapping_add(12345)) >> 16) as u8)
        .collect();
    println!(
        "  Pseudo-random (LCG): {:.4} bits  ← Normal: healthy quantized weights",
        compute_byte_entropy(&pseudo_random)
    );

    // Uniform random — entropy ~8.0 (theoretical max)
    // Simulate with all 256 byte values equally distributed
    let uniform: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
    println!(
        "  Uniform (0..255):    {:.4} bits  ← Max: perfectly uniform",
        compute_byte_entropy(&uniform)
    );

    println!("\n  Expected ranges by dtype:");
    println!("    Q4K/Q6K: 7.5-8.0 (high entropy, packed bits)");
    println!("    F32:     5.0-7.5 (lower, IEEE 754 exponent bias)");
    println!("    F16:     6.0-7.5 (moderate, smaller exponent field)");
    println!();
}

// ============================================================================
// Part 4: Distribution Analysis
// ============================================================================

fn distribution_demo() {
    println!("--- Part 4: Value Distribution Analysis ---\n");

    // Simulate typical neural network weight distribution (near-zero Gaussian)
    let weights: Vec<f32> = (0..10_000)
        .map(|i| {
            // Simple deterministic pseudo-normal via Box-Muller-like transform
            let x = (i as f32 / 10_000.0) * 2.0 - 1.0;
            x * 0.02 // Scale to typical weight range
        })
        .collect();

    let n = weights.len() as f64;
    let mean: f64 = weights.iter().map(|x| *x as f64).sum::<f64>() / n;
    let variance: f64 = weights.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    let skewness: f64 =
        weights.iter().map(|x| ((*x as f64 - mean) / std).powi(3)).sum::<f64>() / n;
    let kurtosis: f64 =
        weights.iter().map(|x| ((*x as f64 - mean) / std).powi(4)).sum::<f64>() / n;

    let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Simple histogram
    let num_bins = 10;
    let bin_width = (max - min) as f64 / num_bins as f64;
    let mut bins = vec![0usize; num_bins];
    for &w in &weights {
        let idx = ((w as f64 - min as f64) / bin_width) as usize;
        let idx = idx.min(num_bins - 1);
        bins[idx] += 1;
    }

    println!("  Simulated weight distribution (10,000 values):\n");
    for (i, &count) in bins.iter().enumerate() {
        let lo = min as f64 + i as f64 * bin_width;
        let hi = lo + bin_width;
        let bar_width = count * 40 / weights.len();
        let bar: String = "█".repeat(bar_width);
        let pct = count as f64 / weights.len() as f64 * 100.0;
        println!("  [{lo:>8.4}, {hi:>8.4})  {bar:<40} {pct:>5.1}%");
    }

    println!("\n  Statistics:");
    println!("    Min:      {min:.6}");
    println!("    Max:      {max:.6}");
    println!("    Mean:     {mean:.6}");
    println!("    Std:      {std:.6}");
    println!("    Skewness: {skewness:.4}");
    println!("    Kurtosis: {kurtosis:.4} (Gaussian = 3.0)");
    println!();
}

// ============================================================================
// Part 5: Quantization Block Layout
// ============================================================================

fn block_layout_demo() {
    println!("--- Part 5: Quantization Super-Block Layout ---\n");

    println!("  Q4_K (144 bytes per 256 elements):");
    println!("    Offset  Size  Field");
    println!("    0x00      2   d (scale, f16)");
    println!("    0x02      2   dmin (min scale, f16)");
    println!("    0x04     12   scales[0-11] (sub-block scales)");
    println!("    0x10    128   qs[0-127] (4-bit packed, 256 values)");
    println!("    Total:  144 bytes = 4.5 bits/element");

    println!("\n  Q6_K (210 bytes per 256 elements):");
    println!("    Offset  Size  Field");
    println!("    0x00    128   ql[0-127] (low 4 bits, 256 values packed 2/byte)");
    println!("    0x80     64   qh[0-63] (high 2 bits, 256 values packed 4/byte)");
    println!("    0xC0     16   scales[0-15] (sub-block scales)");
    println!("    0xD0      2   d (scale, f16)");
    println!("    Total:  210 bytes = 6.56 bits/element");

    println!("\n  Q8_0 (34 bytes per 32 elements):");
    println!("    Offset  Size  Field");
    println!("    0x00      2   d (scale, f16)");
    println!("    0x02     32   qs[0-31] (8-bit signed quants)");
    println!("    Total:   34 bytes = 8.5 bits/element");

    // Demonstrate dequantization for Q8_0 (simplest)
    println!("\n  Q8_0 Dequantization Example:");
    let d_f16: u16 = 0x2C00; // ~0.03125 in f16
    let d = f16_to_f32(d_f16);
    let quants: [i8; 8] = [10, -5, 32, 0, -128, 127, 64, -64];
    print!("    d = {d:.5}, quants = [");
    for (i, &q) in quants.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{q}");
    }
    println!("]");
    print!("    values = [");
    for (i, &q) in quants.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        let val = d * q as f32;
        print!("{val:.5}");
    }
    println!("]");
}
