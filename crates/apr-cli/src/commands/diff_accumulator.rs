
/// Print diff summary with diagnosis.
#[allow(clippy::too_many_arguments)]
fn print_diff_summary(
    results: &[TensorValueStats],
    identical: usize,
    transposed: usize,
    critical: usize,
    large: usize,
    medium: usize,
) {
    let sep = "╠══════════════════════════════════════════════════════════════════════════════╣";
    println!("{}", sep.cyan());
    println!(
        "{}",
        "║                              SUMMARY                                          ║"
            .cyan()
            .bold()
    );
    println!("{}", sep.cyan());
    println!("║ Tensors compared: {:<58} ║", results.len());
    println!(
        "║ Identical: {:<65} ║",
        format!("{identical}").green().to_string()
    );
    println!(
        "║ Transposed (layout diff): {:<50} ║",
        if transposed > 0 {
            format!("{transposed}").cyan().to_string()
        } else {
            "0".dimmed().to_string()
        }
    );
    println!(
        "║ Critical differences: {:<54} ║",
        if critical > 0 {
            format!("{critical}").red().bold().to_string()
        } else {
            "0".green().to_string()
        }
    );
    println!(
        "║ Large differences: {:<57} ║",
        if large > 0 {
            format!("{large}").red().to_string()
        } else {
            "0".green().to_string()
        }
    );
    println!(
        "║ Medium differences: {:<56} ║",
        if medium > 0 {
            format!("{medium}").yellow().to_string()
        } else {
            "0".green().to_string()
        }
    );

    println!("{}", sep.cyan());
    print_diff_diagnosis(results, identical, transposed, critical, large, medium);
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════════╝".cyan()
    );
}

/// Print the diagnosis section of the diff summary.
fn print_diff_diagnosis(
    results: &[TensorValueStats],
    identical: usize,
    transposed: usize,
    critical: usize,
    large: usize,
    medium: usize,
) {
    if critical > 0 {
        println!(
            "║ {} ║",
            "DIAGNOSIS: Critical value differences detected!"
                .red()
                .bold()
        );
        println!("║ {:<75} ║", "Possible causes:".yellow());
        println!(
            "║ {:<75} ║",
            "  - Different quantization/dequantization algorithms"
        );
        println!(
            "║ {:<75} ║",
            "  - Tensor layout mismatch (row-major vs column-major)"
        );
        println!("║ {:<75} ║", "  - Corrupted weights during conversion");
    } else if large > 0 {
        println!(
            "║ {} ║",
            "DIAGNOSIS: Large value differences - may affect inference quality"
                .yellow()
                .bold()
        );
        let transposed_with_diffs = results
            .iter()
            .filter(|r| {
                let is_t = r.shape_a.len() == 2
                    && r.shape_b.len() == 2
                    && r.shape_a[0] == r.shape_b[1]
                    && r.shape_a[1] == r.shape_b[0];
                is_t && r.status != TensorDiffStatus::Transposed
            })
            .count();
        if transposed_with_diffs > 0 {
            println!(
                "║ {:<75} ║",
                "NOTE: Differences in transposed tensors may be expected when".cyan()
            );
            println!(
                "║ {:<75} ║",
                "comparing GGUF (col-major) to APR (row-major) linearly.".cyan()
            );
        }
    } else if medium > 0 {
        println!(
            "║ {} ║",
            "DIAGNOSIS: Medium differences - likely acceptable quantization variance".blue()
        );
    } else if transposed > 0 && identical > 0 {
        println!(
            "║ {} ║",
            "DIAGNOSIS: Values identical, shapes transposed (format layout diff)"
                .cyan()
                .bold()
        );
    } else {
        println!(
            "║ {} ║",
            "DIAGNOSIS: Tensors are nearly identical".green().bold()
        );
    }
}

/// Look up element in data_b at the transposed position corresponding to index `i` in data_a.
fn lookup_transposed_element(
    data_b: &[f32],
    shape_a: &[usize],
    shape_b: &[usize],
    i: usize,
) -> Option<f32> {
    let cols_a = shape_a[1];
    let row = i / cols_a;
    let col = i % cols_a;
    let cols_b = shape_b[1];
    let j = row * cols_b + col;
    if j < data_b.len() {
        Some(data_b[j])
    } else {
        None
    }
}

/// Classify a diff value into identical/small/medium/large buckets.
fn classify_diff(
    diff: f32,
    identical: &mut usize,
    small: &mut usize,
    medium: &mut usize,
    large: &mut usize,
) {
    if diff == 0.0 {
        *identical += 1;
    } else if diff < 0.001 {
        *small += 1;
    } else if diff < 0.01 {
        *medium += 1;
    } else {
        *large += 1;
    }
}

/// Accumulator for element-wise diff statistics.
struct DiffAccumulator {
    sum_diff: f64,
    sum_sq_diff: f64,
    max_diff: f32,
    dot_product: f64,
    norm_a: f64,
    norm_b: f64,
    identical_count: usize,
    small_diff_count: usize,
    medium_diff_count: usize,
    large_diff_count: usize,
}

impl DiffAccumulator {
    fn new() -> Self {
        Self {
            sum_diff: 0.0,
            sum_sq_diff: 0.0,
            max_diff: 0.0,
            dot_product: 0.0,
            norm_a: 0.0,
            norm_b: 0.0,
            identical_count: 0,
            small_diff_count: 0,
            medium_diff_count: 0,
            large_diff_count: 0,
        }
    }

    /// Accumulate a pair of finite values. Returns false if NaN/Inf (counted as large diff).
    fn accumulate(&mut self, a: f32, b: f32) {
        if a.is_nan() || b.is_nan() || a.is_infinite() || b.is_infinite() {
            self.large_diff_count += 1;
            return;
        }
        let diff = (a - b).abs();
        self.sum_diff += diff as f64;
        self.sum_sq_diff += (diff as f64) * (diff as f64);
        self.max_diff = self.max_diff.max(diff);
        self.dot_product += (a as f64) * (b as f64);
        self.norm_a += (a as f64) * (a as f64);
        self.norm_b += (b as f64) * (b as f64);
        classify_diff(
            diff,
            &mut self.identical_count,
            &mut self.small_diff_count,
            &mut self.medium_diff_count,
            &mut self.large_diff_count,
        );
    }

    fn mean_diff(&self, n: usize) -> f32 {
        (self.sum_diff / n as f64) as f32
    }
    fn rmse(&self, n: usize) -> f32 {
        ((self.sum_sq_diff / n as f64).sqrt()) as f32
    }

    fn cosine_similarity(&self) -> f32 {
        if self.norm_a > 0.0 && self.norm_b > 0.0 {
            (self.dot_product / (self.norm_a.sqrt() * self.norm_b.sqrt())) as f32
        } else {
            0.0
        }
    }
}

/// Build an empty `TensorValueStats` for zero-element tensors.
fn empty_tensor_stats(name: &str, shape_a: &[usize], shape_b: &[usize]) -> TensorValueStats {
    TensorValueStats {
        name: name.to_string(),
        shape_a: shape_a.to_vec(),
        shape_b: shape_b.to_vec(),
        element_count: 0,
        mean_diff: 0.0,
        max_diff: 0.0,
        rmse: 0.0,
        cosine_similarity: 0.0,
        identical_count: 0,
        small_diff_count: 0,
        medium_diff_count: 0,
        large_diff_count: 0,
        status: TensorDiffStatus::Critical,
    }
}

fn compute_tensor_diff_stats(
    name: &str,
    shape_a: &[usize],
    shape_b: &[usize],
    data_a: &[f32],
    data_b: &[f32],
    transpose_aware: bool,
) -> TensorValueStats {
    let element_count = data_a.len().min(data_b.len());
    if element_count == 0 {
        return empty_tensor_stats(name, shape_a, shape_b);
    }

    let is_transpose = shape_a.len() == 2
        && shape_b.len() == 2
        && shape_a[0] == shape_b[1]
        && shape_a[1] == shape_b[0];
    let use_transpose = transpose_aware && is_transpose && shape_a.len() == 2;

    let mut acc = DiffAccumulator::new();
    for i in 0..element_count {
        let a = data_a[i];
        let b = if use_transpose {
            match lookup_transposed_element(data_b, shape_a, shape_b, i) {
                Some(val) => val,
                None => continue,
            }
        } else {
            data_b[i]
        };
        acc.accumulate(a, b);
    }

    let status = TensorDiffStatus::from_diff_info(
        acc.max_diff,
        shape_a,
        shape_b,
        acc.identical_count,
        element_count,
    );

    TensorValueStats {
        name: name.to_string(),
        shape_a: shape_a.to_vec(),
        shape_b: shape_b.to_vec(),
        element_count,
        mean_diff: acc.mean_diff(element_count),
        max_diff: acc.max_diff,
        rmse: acc.rmse(element_count),
        cosine_similarity: acc.cosine_similarity(),
        identical_count: acc.identical_count,
        small_diff_count: acc.small_diff_count,
        medium_diff_count: acc.medium_diff_count,
        large_diff_count: acc.large_diff_count,
        status,
    }
}

fn print_tensor_diff_row(stats: &TensorValueStats) {
    let status_str = stats.status.colored_string();
    let name_truncated = truncate_str(&stats.name, 40);

    // Color max_diff based on severity
    let max_diff_str = format!("{:.6}", stats.max_diff);
    let max_diff_colored = match stats.status {
        TensorDiffStatus::Identical | TensorDiffStatus::NearlyIdentical => max_diff_str.green(),
        TensorDiffStatus::SmallDiff | TensorDiffStatus::Transposed => max_diff_str.cyan(),
        TensorDiffStatus::MediumDiff => max_diff_str.yellow(),
        TensorDiffStatus::LargeDiff | TensorDiffStatus::Critical => max_diff_str.red(),
    };

    // Color cosine similarity
    let cos_str = format!("{:.6}", stats.cosine_similarity);
    let cos_colored = if stats.cosine_similarity > 0.9999 {
        cos_str.green()
    } else if stats.cosine_similarity > 0.999 {
        cos_str.blue()
    } else if stats.cosine_similarity > 0.99 {
        cos_str.yellow()
    } else {
        cos_str.red()
    };

    println!("║ [{}] {:<40} ║", status_str, name_truncated);
    println!(
        "║   max_diff={} mean_diff={:.6} rmse={:.6} cos_sim={} ║",
        max_diff_colored, stats.mean_diff, stats.rmse, cos_colored
    );

    // Check for shape mismatch and if it's a transpose
    let shape_match = stats.shape_a == stats.shape_b;
    let is_transpose = !shape_match
        && stats.shape_a.len() == 2
        && stats.shape_b.len() == 2
        && stats.shape_a[0] == stats.shape_b[1]
        && stats.shape_a[1] == stats.shape_b[0];

    if !shape_match {
        if is_transpose {
            println!(
                "║   {} shapes: {:?} vs {:?} {} ║",
                "TRANSPOSED".yellow(),
                stats.shape_a,
                stats.shape_b,
                "(row-major vs col-major)".dimmed()
            );
        } else {
            println!(
                "║   {} shapes: {:?} vs {:?} ║",
                "SHAPE MISMATCH".red().bold(),
                stats.shape_a,
                stats.shape_b
            );
        }
    }

    // Show distribution if there are differences
    if stats.status != TensorDiffStatus::Identical {
        let total = stats.element_count;
        let ident_pct = 100.0 * stats.identical_count as f64 / total as f64;
        let small_pct = 100.0 * stats.small_diff_count as f64 / total as f64;
        let med_pct = 100.0 * stats.medium_diff_count as f64 / total as f64;
        let large_pct = 100.0 * stats.large_diff_count as f64 / total as f64;

        println!(
            "║   dist: {:.1}% ident, {:.1}% small, {:.1}% med, {:.1}% large ({} elems)  ║",
            ident_pct, small_pct, med_pct, large_pct, total
        );
    }

    println!(
        "{}",
        "╠──────────────────────────────────────────────────────────────────────────────╣".dimmed()
    );
}

// ============================================================================
// Helper Functions
// ============================================================================

fn normalize_tensor_name(name: &str) -> String {
    // Normalize different naming conventions
    name.replace("blk.", "model.layers.")
        .replace(".attn_q.", ".self_attn.q_proj.")
        .replace(".attn_k.", ".self_attn.k_proj.")
        .replace(".attn_v.", ".self_attn.v_proj.")
        .replace(".attn_output.", ".self_attn.o_proj.")
        .replace(".ffn_gate.", ".mlp.gate_proj.")
        .replace(".ffn_up.", ".mlp.up_proj.")
        .replace(".ffn_down.", ".mlp.down_proj.")
        .replace(".attn_norm.", ".input_layernorm.")
        .replace(".ffn_norm.", ".post_attention_layernorm.")
}

fn truncate_path(path: &str, max_len: usize) -> String {
    if path.len() <= max_len {
        path.to_string()
    } else {
        format!("...{}", &path[path.len() - max_len + 3..])
    }
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

// ============================================================================
// Path Validation
// ============================================================================

fn validate_paths(path1: &Path, path2: &Path) -> Result<(), CliError> {
    for path in [path1, path2] {
        if !path.exists() {
            return Err(CliError::FileNotFound(path.to_path_buf()));
        }
        if !path.is_file() {
            return Err(CliError::NotAFile(path.to_path_buf()));
        }
    }
    Ok(())
}
