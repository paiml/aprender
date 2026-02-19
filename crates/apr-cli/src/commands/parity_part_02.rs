/// SPC color level for threshold-based formatting.
#[derive(Clone, Copy)]
enum SpcColor { GreenBold, Green, Yellow, Red, RedBold }

/// Apply SPC color to a pre-formatted string.
fn apply_spc_color(s: &str, level: SpcColor) -> String {
    match level {
        SpcColor::GreenBold => s.green().bold().to_string(),
        SpcColor::Green => s.green().to_string(),
        SpcColor::Yellow => s.yellow().to_string(),
        SpcColor::Red => s.red().to_string(),
        SpcColor::RedBold => s.red().bold().to_string(),
    }
}

/// KL divergence thresholds (lower is better): <0.001=GreenBold, <KL_MAX=Green, <0.1=Yellow, <1.0=Red, else RedBold.
const KL_THRESHOLDS: [(f64, SpcColor); 4] = [
    (0.001, SpcColor::GreenBold), (KL_DIV_MAX, SpcColor::Green),
    (0.1, SpcColor::Yellow), (1.0, SpcColor::Red),
];

/// Sigma thresholds (higher is better, checked descending): >=6=GreenBold, >=SIGMA_MIN=Green, >=2=Yellow, >=1=Red, else RedBold.
const SIGMA_THRESHOLDS: [(f64, SpcColor); 4] = [
    (6.0, SpcColor::GreenBold), (SIGMA_MIN as f64, SpcColor::Green),
    (2.0, SpcColor::Yellow), (1.0, SpcColor::Red),
];

fn color_lower_is_better(val: f64, thresholds: &[(f64, SpcColor)], fallback: SpcColor) -> SpcColor {
    for &(t, color) in thresholds {
        if val < t { return color; }
    }
    fallback
}

fn color_higher_is_better(val: f64, thresholds: &[(f64, SpcColor)], fallback: SpcColor) -> SpcColor {
    for &(t, color) in thresholds {
        if val >= t { return color; }
    }
    fallback
}

fn format_kl(kl: f64) -> String {
    let color = color_lower_is_better(kl, &KL_THRESHOLDS, SpcColor::RedBold);
    let precision = if kl < 0.1 { 5 } else if kl < 1.0 { 3 } else { 2 };
    apply_spc_color(&format!("{kl:>8.*}", precision), color)
}

fn format_sigma(sigma: f64) -> String {
    let color = color_higher_is_better(sigma, &SIGMA_THRESHOLDS, SpcColor::RedBold);
    apply_spc_color(&format!("{sigma:>6.1}σ "), color)
}

/// Print summary statistics
fn print_summary(metrics: &[SpcMetrics]) {
    let n = metrics.len();
    if n == 0 {
        return;
    }

    let pass_count = metrics.iter().filter(|m| m.verdict().is_pass()).count();
    let warn_count = metrics
        .iter()
        .filter(|m| matches!(m.verdict(), Verdict::WarnArgmax | Verdict::WarnOutOfSpec))
        .count();
    let fail_count = metrics.iter().filter(|m| m.verdict().is_fail()).count();

    let avg_cos: f32 = metrics.iter().map(|m| m.cosine_similarity).sum::<f32>() / n as f32;
    let min_cos: f32 = metrics
        .iter()
        .map(|m| m.cosine_similarity)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0);
    let avg_kl: f64 = metrics.iter().map(|m| m.kl_divergence).sum::<f64>() / n as f64;
    let max_diff: f32 = metrics
        .iter()
        .map(|m| m.max_abs_diff)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0);
    let avg_sigma: f64 = metrics.iter().map(|m| m.sigma_level).sum::<f64>() / n as f64;
    let avg_cpk: f64 = metrics.iter().map(SpcMetrics::cpk).sum::<f64>() / n as f64;

    let avg_oos: f64 = metrics.iter().map(|m| m.out_of_spec_count).sum::<usize>() as f64 / n as f64;
    let avg_vocab = metrics.iter().map(|m| m.vocab_size).sum::<usize>() as f64 / n as f64;
    let ppm = if avg_vocab > 0.0 {
        (avg_oos / avg_vocab) * 1_000_000.0
    } else {
        0.0
    };

    eprintln!();
    eprintln!(
        "{}",
        "══════════════════════════════════════════════════════════════════════"
            .cyan()
            .bold()
    );
    eprintln!("{}", "  STATISTICAL PROCESS CONTROL SUMMARY".cyan().bold());
    eprintln!(
        "{}",
        "══════════════════════════════════════════════════════════════════════"
            .cyan()
            .bold()
    );

    // Yield
    eprintln!();
    let yield_pct = (pass_count as f64 / n as f64) * 100.0;
    let yield_color = color_higher_is_better(
        yield_pct,
        &[(100.0, SpcColor::GreenBold), (85.0, SpcColor::Yellow)],
        SpcColor::RedBold,
    );
    let yield_str = apply_spc_color(&format!("{yield_pct:.1}%"), yield_color);
    eprintln!(
        "  Yield:              {} ({} pass, {} warn, {} fail out of {})",
        yield_str, pass_count, warn_count, fail_count, n,
    );

    // Defect rate in PPM (parts per million)
    let ppm_color = color_lower_is_better(
        ppm,
        &[(3.4, SpcColor::GreenBold), (100.0, SpcColor::Yellow), (10000.0, SpcColor::Red)],
        SpcColor::RedBold,
    );
    let precision = if ppm < 100.0 { 1 } else { 0 };
    let ppm_str = apply_spc_color(&format!("{ppm:.precision$} PPM"), ppm_color);
    eprintln!(
        "  Defect rate:        {} (logits outside tolerance)",
        ppm_str
    );

    // Key metrics
    eprintln!();
    eprintln!(
        "  {} {}",
        "Cosine similarity:".white().bold(),
        format_metric_range(avg_cos, min_cos, COSINE_SIM_MIN, true)
    );
    eprintln!(
        "  {} {}",
        "KL divergence:    ".white().bold(),
        format_metric_single(avg_kl, KL_DIV_MAX, false)
    );
    eprintln!(
        "  {} {}",
        "Max abs diff:     ".white().bold(),
        format_metric_single(max_diff as f64, TOLERANCE_ABS as f64, false)
    );
    eprintln!(
        "  {} {}",
        "Sigma level:      ".white().bold(),
        format_sigma_summary(avg_sigma)
    );
    eprintln!(
        "  {} {}",
        "Cpk:              ".white().bold(),
        format_cpk(avg_cpk)
    );

    // Control limits legend
    eprintln!();
    eprintln!(
        "{}",
        "  Control limits (from layer-parity-v1.yaml):".dimmed()
    );
    eprintln!(
        "    {} cosine ≥ {:.3}  {} KL div ≤ {:.3}  {} |diff| ≤ {:.1}  {} σ ≥ {:.0}  {} Cpk ≥ 1.33",
        "USL:".dimmed(),
        COSINE_SIM_MIN,
        "USL:".dimmed(),
        KL_DIV_MAX,
        "USL:".dimmed(),
        TOLERANCE_ABS,
        "LSL:".dimmed(),
        SIGMA_MIN,
        "LSL:".dimmed(),
    );

    // Overall verdict
    eprintln!();
    if fail_count == 0 && warn_count == 0 {
        eprintln!(
            "  {}",
            "  PARITY PROVEN: GPU inference is mathematically equivalent to CPU  "
                .on_green()
                .black()
                .bold()
        );
    } else if fail_count == 0 {
        eprintln!(
            "  {}",
            "  PARITY WARNING: GPU matches CPU but with elevated noise  "
                .on_yellow()
                .black()
                .bold()
        );
    } else {
        eprintln!(
            "  {}",
            "  PARITY DISPROVEN: GPU computes a DIFFERENT function than CPU  "
                .on_red()
                .white()
                .bold()
        );
    }
    eprintln!();
}

/// Color a value relative to a threshold: green if pass, red if fail, yellow if marginal.
fn color_vs_threshold(val: f32, threshold: f32, higher_is_better: bool) -> SpcColor {
    let pass = if higher_is_better { val >= threshold } else { val <= threshold };
    if pass { SpcColor::Green } else { SpcColor::RedBold }
}

fn format_metric_range(avg: f32, min: f32, threshold: f32, higher_is_better: bool) -> String {
    let pass = if higher_is_better { min >= threshold } else { avg <= threshold };
    let thr_s = format!("{threshold:.3}");
    let avg_c = apply_spc_color(&format!("{avg:.6}"), color_vs_threshold(avg, threshold, higher_is_better));
    let min_c = apply_spc_color(&format!("{min:.6}"), color_vs_threshold(min, threshold, higher_is_better));
    if pass {
        format!("avg={avg_c} min={min_c} (threshold: {})", thr_s.dimmed())
    } else {
        format!("avg={avg_c} min={min_c} (threshold: {} {})", thr_s.dimmed(), "VIOLATED".red().bold())
    }
}

fn format_metric_single(val: f64, threshold: f64, lower_is_better: bool) -> String {
    let pass = if lower_is_better { val <= threshold } else { val >= threshold };
    let color = if pass { SpcColor::Green } else { SpcColor::RedBold };
    let val_c = apply_spc_color(&format!("{val:.6}"), color);
    let thr_s = format!("{threshold:.3}");
    if pass {
        format!("{val_c} (threshold: {})", thr_s.dimmed())
    } else {
        format!("{val_c} (threshold: {} {})", thr_s.dimmed(), "VIOLATED".red().bold())
    }
}

/// Sigma yield labels indexed by integer sigma level.
const SIGMA_YIELD_LABELS: &[&str] = &[
    "out of control", "68.3% yield", "95.4% yield",
    "99.73% yield", "99.994% yield", "99.99994% yield",
];

fn format_sigma_summary(sigma: f64) -> String {
    let s = format!("{sigma:.2}σ");
    let idx = (sigma as usize).min(SIGMA_YIELD_LABELS.len() - 1);
    let label = if sigma >= 6.0 { "world class" } else { SIGMA_YIELD_LABELS[idx] };
    let color = color_higher_is_better(sigma, &SIGMA_THRESHOLDS, SpcColor::RedBold);
    let colored_s = apply_spc_color(&s, color);
    if sigma >= 2.0 {
        format!("{colored_s} ({})", label.dimmed())
    } else {
        format!("{colored_s} ({} {})", label, "VIOLATED".red().bold())
    }
}

/// Cpk capability thresholds and their labels.
const CPK_LEVELS: &[(f64, SpcColor, &str)] = &[
    (2.0, SpcColor::GreenBold, "world class, ≥2.0"),
    (1.33, SpcColor::Green, "capable, ≥1.33"),
    (1.0, SpcColor::Yellow, "marginal, needs improvement"),
];

fn format_cpk(cpk: f64) -> String {
    for &(threshold, color, label) in CPK_LEVELS {
        if cpk >= threshold {
            return format!("{} ({label})", apply_spc_color(&format!("{cpk:.3}"), color));
        }
    }
    format!(
        "{} ({} {})",
        apply_spc_color(&format!("{cpk:.3}"), SpcColor::RedBold),
        "incapable",
        "VIOLATED".red().bold()
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPUTATION
// ═══════════════════════════════════════════════════════════════════════════════

fn compute_metrics(
    cpu_logits: &[f32],
    gpu_logits: &[f32],
    position: usize,
    token_id: u32,
) -> SpcMetrics {
    let n = cpu_logits.len();

    let cpu_nan = cpu_logits.iter().filter(|x| x.is_nan()).count();
    let gpu_nan = gpu_logits.iter().filter(|x| x.is_nan()).count();

    let cpu_argmax = argmax(cpu_logits);
    let gpu_argmax = argmax(gpu_logits);

    let cpu_top = cpu_logits
        .get(cpu_argmax as usize)
        .copied()
        .unwrap_or(f32::NAN);
    let gpu_top = gpu_logits
        .get(gpu_argmax as usize)
        .copied()
        .unwrap_or(f32::NAN);

    // Element-wise differences
    let mut sum_abs_diff: f64 = 0.0;
    let mut sum_sq_diff: f64 = 0.0;
    let mut max_abs_diff: f32 = 0.0;
    let mut max_diff_idx: usize = 0;
    let mut out_of_spec = 0usize;

    for (i, (c, g)) in cpu_logits.iter().zip(gpu_logits.iter()).enumerate() {
        let d = (c - g).abs();
        let d = if d.is_nan() { f32::INFINITY } else { d };
        sum_abs_diff += d as f64;
        sum_sq_diff += (d as f64) * (d as f64);
        if d > max_abs_diff {
            max_abs_diff = d;
            max_diff_idx = i;
        }
        if d > TOLERANCE_ABS {
            out_of_spec += 1;
        }
    }

    let mean_abs_diff = (sum_abs_diff / n as f64) as f32;
    let rmse = (sum_sq_diff / n as f64).sqrt() as f32;

    // Cosine similarity
    let cosine_similarity = cosine_sim(cpu_logits, gpu_logits);

    // KL divergence on softmax distributions
    let kl_divergence = kl_div_softmax(cpu_logits, gpu_logits);

    // Sigma level: how many standard deviations is max_diff from mean?
    let sigma_level = if rmse > 1e-10 {
        // Process sigma: tolerance / process spread
        TOLERANCE_ABS as f64 / rmse as f64
    } else {
        99.0 // Perfect
    };

    SpcMetrics {
        position,
        token_id,
        cpu_argmax,
        gpu_argmax,
        _cpu_top_logit: cpu_top,
        _gpu_top_logit: gpu_top,
        max_abs_diff,
        _max_diff_idx: max_diff_idx,
        mean_abs_diff,
        rmse,
        cosine_similarity,
        kl_divergence,
        sigma_level,
        cpu_nan,
        gpu_nan,
        out_of_spec_count: out_of_spec,
        vocab_size: n,
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i as u32)
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        (dot / denom) as f32
    }
}

/// KL divergence: D_KL(P || Q) where P=softmax(cpu), Q=softmax(gpu)
fn kl_div_softmax(cpu_logits: &[f32], gpu_logits: &[f32]) -> f64 {
    // Numerically stable softmax
    let cpu_max = cpu_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let gpu_max = gpu_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let cpu_exp: Vec<f64> = cpu_logits
        .iter()
        .map(|x| ((*x - cpu_max) as f64).exp())
        .collect();
    let gpu_exp: Vec<f64> = gpu_logits
        .iter()
        .map(|x| ((*x - gpu_max) as f64).exp())
        .collect();

    let cpu_sum: f64 = cpu_exp.iter().sum();
    let gpu_sum: f64 = gpu_exp.iter().sum();

    if cpu_sum < 1e-30 || gpu_sum < 1e-30 {
        return f64::INFINITY;
    }

    let mut kl: f64 = 0.0;
    let eps = 1e-10;
    for (p_unnorm, q_unnorm) in cpu_exp.iter().zip(gpu_exp.iter()) {
        let p = p_unnorm / cpu_sum;
        let q = (q_unnorm / gpu_sum).max(eps);
        if p > eps {
            kl += p * (p / q).ln();
        }
    }

    kl.max(0.0) // Clamp numerical noise
}
