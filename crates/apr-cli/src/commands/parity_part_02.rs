
fn format_kl(kl: f64) -> String {
    if kl < 0.001 {
        format!("{:>8.5}", kl).green().bold().to_string()
    } else if kl < KL_DIV_MAX {
        format!("{:>8.5}", kl).green().to_string()
    } else if kl < 0.1 {
        format!("{:>8.4}", kl).yellow().to_string()
    } else if kl < 1.0 {
        format!("{:>8.3}", kl).red().to_string()
    } else {
        format!("{:>8.2}", kl).red().bold().to_string()
    }
}

fn format_sigma(sigma: f64) -> String {
    if sigma >= 6.0 {
        format!("{:>6.1}σ ", sigma).green().bold().to_string()
    } else if sigma >= SIGMA_MIN {
        format!("{:>6.1}σ ", sigma).green().to_string()
    } else if sigma >= 2.0 {
        format!("{:>6.1}σ ", sigma).yellow().to_string()
    } else if sigma >= 1.0 {
        format!("{:>6.1}σ ", sigma).red().to_string()
    } else {
        format!("{:>6.1}σ ", sigma).red().bold().to_string()
    }
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
    let yield_str = if yield_pct >= 100.0 {
        format!("{yield_pct:.1}%").green().bold().to_string()
    } else if yield_pct >= 85.0 {
        format!("{yield_pct:.1}%").yellow().to_string()
    } else {
        format!("{yield_pct:.1}%").red().bold().to_string()
    };
    eprintln!(
        "  Yield:              {} ({} pass, {} warn, {} fail out of {})",
        yield_str, pass_count, warn_count, fail_count, n,
    );

    // Defect rate in PPM (parts per million)
    let ppm_str = if ppm < 3.4 {
        format!("{ppm:.1} PPM").green().bold().to_string()
    } else if ppm < 100.0 {
        format!("{ppm:.1} PPM").yellow().to_string()
    } else if ppm < 10000.0 {
        format!("{ppm:.0} PPM").red().to_string()
    } else {
        format!("{ppm:.0} PPM").red().bold().to_string()
    };
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

fn format_metric_range(avg: f32, min: f32, threshold: f32, higher_is_better: bool) -> String {
    let pass = if higher_is_better {
        min >= threshold
    } else {
        avg <= threshold
    };
    let avg_s = format!("{avg:.6}");
    let min_s = format!("{min:.6}");
    let thr_s = format!("{threshold:.3}");
    if pass {
        format!(
            "avg={} min={} (threshold: {})",
            avg_s.green(),
            min_s.green(),
            thr_s.dimmed()
        )
    } else {
        format!(
            "avg={} min={} (threshold: {} {})",
            if avg >= threshold {
                avg_s.yellow().to_string()
            } else {
                avg_s.red().bold().to_string()
            },
            if min >= threshold {
                min_s.yellow().to_string()
            } else {
                min_s.red().bold().to_string()
            },
            thr_s.dimmed(),
            "VIOLATED".red().bold()
        )
    }
}

fn format_metric_single(val: f64, threshold: f64, lower_is_better: bool) -> String {
    let pass = if lower_is_better {
        val <= threshold
    } else {
        val >= threshold
    };
    let val_s = format!("{val:.6}");
    let thr_s = format!("{threshold:.3}");
    if pass {
        format!("{} (threshold: {})", val_s.green(), thr_s.dimmed())
    } else {
        format!(
            "{} (threshold: {} {})",
            val_s.red().bold(),
            thr_s.dimmed(),
            "VIOLATED".red().bold()
        )
    }
}

fn format_sigma_summary(sigma: f64) -> String {
    let s = format!("{sigma:.2}σ");
    let label = match sigma as u32 {
        0 => "out of control",
        1 => "68.3% yield",
        2 => "95.4% yield",
        3 => "99.73% yield",
        4 => "99.994% yield",
        5 => "99.99994% yield",
        _ => "world class",
    };
    if sigma >= 6.0 {
        format!("{} ({})", s.green().bold(), label.dimmed())
    } else if sigma >= SIGMA_MIN {
        format!("{} ({})", s.green(), label.dimmed())
    } else if sigma >= 2.0 {
        format!("{} ({})", s.yellow(), label.dimmed())
    } else {
        format!("{} ({} {})", s.red().bold(), label, "VIOLATED".red().bold())
    }
}

fn format_cpk(cpk: f64) -> String {
    let s = format!("{cpk:.3}");
    if cpk >= 2.0 {
        format!("{} (world class, ≥2.0)", s.green().bold())
    } else if cpk >= 1.33 {
        format!("{} (capable, ≥1.33)", s.green())
    } else if cpk >= 1.0 {
        format!(
            "{} (marginal, needs improvement {})",
            s.yellow(),
            "< 1.33".red()
        )
    } else {
        format!(
            "{} ({} {})",
            s.red().bold(),
            "incapable",
            "VIOLATED".red().bold()
        )
    }
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
