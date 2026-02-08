//! PMAT-232: GPU/CPU Parity — Statistical Process Control for Inference
//!
//! Toyota Way principles applied:
//!   - Jidoka (自働化): Stop-the-line on parity failure — GPU CANNOT serve if it diverges
//!   - Mieruka (見える化): Visual management — problems visible at a glance
//!   - Genchi Genbutsu (現地現物): Go and see — run BOTH backends, compare numbers
//!   - Kaizen (改善): Statistical controls that improve over time
//!
//! Six Sigma SPC metrics:
//!   - Cosine similarity: measures directional agreement (1.0 = identical direction)
//!   - KL divergence: measures distribution distance (0.0 = identical distribution)
//!   - Sigma level: process capability (≥3σ required, ≥6σ is world-class)
//!   - Cpk: process capability index (≥1.33 required for production)
//!
//! See contracts/layer-parity-v1.yaml for the full specification.

use crate::error::{CliError, Result};
use colored::Colorize;
use std::path::Path;

// ═══════════════════════════════════════════════════════════════════════════════
// CONTRACT TOLERANCES (from layer-parity-v1.yaml)
// ═══════════════════════════════════════════════════════════════════════════════

/// Max absolute logit difference for PASS (quantized GEMV tolerance)
const TOLERANCE_ABS: f32 = 1.0;
/// Cosine similarity minimum for PASS
const COSINE_SIM_MIN: f32 = 0.999;
/// KL divergence maximum for PASS (softmax distribution distance)
const KL_DIV_MAX: f64 = 0.01;
/// Sigma level minimum for production (3σ = 99.73% within limits)
const SIGMA_MIN: f64 = 3.0;

// ═══════════════════════════════════════════════════════════════════════════════
// SPC METRICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Statistical Process Control metrics for one token position
struct SpcMetrics {
    /// Token position
    position: usize,
    /// Token ID processed
    token_id: u32,
    /// CPU argmax token
    cpu_argmax: u32,
    /// GPU argmax token
    gpu_argmax: u32,
    /// CPU logit at CPU argmax
    _cpu_top_logit: f32,
    /// GPU logit at GPU argmax
    _gpu_top_logit: f32,
    /// Maximum absolute difference across all logits
    max_abs_diff: f32,
    /// Index of maximum difference
    _max_diff_idx: usize,
    /// Mean absolute difference
    mean_abs_diff: f32,
    /// Root mean square error
    rmse: f32,
    /// Cosine similarity between logit vectors (1.0 = identical direction)
    cosine_similarity: f32,
    /// KL divergence: D_KL(CPU_softmax || GPU_softmax)
    kl_divergence: f64,
    /// Sigma level: how many σ the max difference is from the mean
    sigma_level: f64,
    /// NaN count in CPU logits
    cpu_nan: usize,
    /// NaN count in GPU logits
    gpu_nan: usize,
    /// Number of logits where abs_diff > TOLERANCE_ABS
    out_of_spec_count: usize,
    /// Total logit count
    vocab_size: usize,
}

impl SpcMetrics {
    /// Overall parity verdict
    fn verdict(&self) -> Verdict {
        if self.cpu_nan > 0 || self.gpu_nan > 0 {
            Verdict::FailNan
        } else if self.cosine_similarity < 0.9 {
            Verdict::FailCatastrophic
        } else if self.cosine_similarity < COSINE_SIM_MIN {
            Verdict::FailDivergent
        } else if self.max_abs_diff > TOLERANCE_ABS {
            Verdict::WarnOutOfSpec
        } else if self.cpu_argmax != self.gpu_argmax {
            Verdict::WarnArgmax
        } else {
            Verdict::Pass
        }
    }

    /// Cpk (process capability index): min(USL-μ, μ-LSL) / 3σ
    /// For our case: how well centered the differences are within tolerance
    fn cpk(&self) -> f64 {
        if self.rmse < 1e-10 {
            return 99.9; // Perfect
        }
        let sigma = self.rmse as f64;
        let tolerance = TOLERANCE_ABS as f64;
        // One-sided: distance to upper spec limit / 3σ
        (tolerance - self.mean_abs_diff as f64) / (3.0 * sigma)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    /// Perfect parity within tolerance
    Pass,
    /// Argmax differs but logits are close
    WarnArgmax,
    /// Some logits outside tolerance
    WarnOutOfSpec,
    /// Cosine similarity below threshold — different computation
    FailDivergent,
    /// Cosine similarity below 0.9 — catastrophically wrong
    FailCatastrophic,
    /// NaN detected
    FailNan,
}

impl Verdict {
    fn symbol(&self) -> String {
        match self {
            Self::Pass => "  PASS  ".on_green().black().bold().to_string(),
            Self::WarnArgmax => "  WARN  ".on_yellow().black().bold().to_string(),
            Self::WarnOutOfSpec => " OOS(!) ".on_yellow().black().bold().to_string(),
            Self::FailDivergent => "  FAIL  ".on_red().white().bold().to_string(),
            Self::FailCatastrophic => " FATAL! ".on_red().white().bold().to_string(),
            Self::FailNan => "  NaN!  ".on_magenta().white().bold().to_string(),
        }
    }

    fn is_pass(&self) -> bool {
        matches!(self, Self::Pass)
    }

    fn is_fail(&self) -> bool {
        matches!(
            self,
            Self::FailDivergent | Self::FailCatastrophic | Self::FailNan
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// AUTO-DIAGNOSIS
// ═══════════════════════════════════════════════════════════════════════════════

/// Analyze failure patterns and produce human-readable diagnosis
fn auto_diagnose(metrics: &[SpcMetrics], hidden_dim: usize, num_heads: usize, kv_heads: usize) {
    let failures: Vec<_> = metrics.iter().filter(|m| m.verdict().is_fail()).collect();
    if failures.is_empty() {
        return;
    }

    eprintln!();
    eprintln!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════╗"
            .red()
            .bold()
    );
    eprintln!(
        "{}",
        "║                    AUTO-DIAGNOSIS (Five Whys)                       ║"
            .red()
            .bold()
    );
    eprintln!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════╝"
            .red()
            .bold()
    );

    // Pattern 1: Position 0 fails → core layer computation, NOT KV cache
    let pos0_fails = metrics
        .first()
        .is_some_and(|m| m.verdict().is_fail());

    // Pattern 2: Divergence grows with position → KV cache accumulation error
    let diffs: Vec<f32> = metrics.iter().map(|m| m.max_abs_diff).collect();
    let growing = diffs.windows(2).filter(|w| w[1] > w[0] * 1.5).count() > diffs.len() / 2;

    // Pattern 3: High cosine similarity but different argmax → close call
    let high_cos_wrong_argmax = failures
        .iter()
        .any(|m| m.cosine_similarity > 0.99 && m.cpu_argmax != m.gpu_argmax);

    // Pattern 4: Low cosine similarity → completely different computation
    let catastrophic = failures.iter().any(|m| m.cosine_similarity < 0.9);

    // Pattern 5: All positions fail uniformly → systematic error
    let all_fail = metrics.iter().all(|m| m.verdict().is_fail());

    // GQA analysis
    let gqa_ratio = if kv_heads > 0 {
        num_heads / kv_heads
    } else {
        0
    };
    let head_dim = if num_heads > 0 {
        hidden_dim / num_heads
    } else {
        0
    };

    eprintln!();
    eprintln!("{}", "  Architecture context:".cyan().bold());
    eprintln!(
        "    hidden_dim={}, heads={}, kv_heads={}, head_dim={}, GQA_ratio={}",
        hidden_dim, num_heads, kv_heads, head_dim, gqa_ratio,
    );
    eprintln!();

    if pos0_fails {
        eprintln!(
            "  {} {}",
            "WHY 1:".red().bold(),
            "Position 0 diverges — bug is in CORE LAYER computation"
        );
        eprintln!(
            "         {}",
            "(KV cache is empty at pos 0, so cache is NOT the cause)".dimmed()
        );
    }

    if catastrophic {
        let avg_cos: f32 =
            failures.iter().map(|m| m.cosine_similarity).sum::<f32>() / failures.len() as f32;
        eprintln!(
            "  {} GPU computes a COMPLETELY DIFFERENT function (avg cosine={:.4})",
            "WHY 2:".red().bold(),
            avg_cos,
        );
        eprintln!(
            "         {}",
            "This is NOT a rounding error. Likely: wrong dimensions, swapped args, or wrong kernel."
                .dimmed()
        );
    } else if high_cos_wrong_argmax {
        eprintln!(
            "  {} {}",
            "WHY 2:".yellow().bold(),
            "Logit DIRECTION matches but MAGNITUDE differs — scaling/normalization bug"
        );
    }

    if all_fail && pos0_fails {
        eprintln!(
            "  {} {}",
            "WHY 3:".red().bold(),
            "ALL positions fail from pos 0 — systematic error in GPU forward pass"
        );
        eprintln!(
            "         {}",
            "Not a progressive drift. The FIRST layer computation is already wrong.".dimmed()
        );
    } else if growing {
        eprintln!(
            "  {} {}",
            "WHY 3:".yellow().bold(),
            "Divergence GROWS with position — error accumulates through KV cache"
        );
    }

    // Dimension-specific diagnosis
    if catastrophic && pos0_fails {
        eprintln!();
        eprintln!("{}", "  Likely root causes (ranked by probability):".cyan().bold());
        eprintln!(
            "    {} Workspace buffer sized for wrong hidden_dim (expected {})",
            "1.".white().bold(),
            hidden_dim,
        );
        eprintln!(
            "    {} Attention head_dim={} or GQA ratio={} miscalculated in GPU kernels",
            "2.".white().bold(),
            head_dim,
            gqa_ratio,
        );
        eprintln!(
            "    {} GEMV kernel grid/block dimensions wrong for hidden_dim={}",
            "3.".white().bold(),
            hidden_dim,
        );
        eprintln!(
            "    {} RoPE frequency table wrong for head_dim={}",
            "4.".white().bold(),
            head_dim,
        );

        eprintln!();
        eprintln!("{}", "  Falsification tests:".cyan().bold());
        eprintln!(
            "    {} Run with hidden_dim=1536 model (1.5B) — if PASS, bug is dimension-dependent",
            "T1:".white().bold(),
        );
        eprintln!(
            "    {} Compare GPU workspace allocation vs hidden_dim*sizeof(f32)",
            "T2:".white().bold(),
        );
        eprintln!(
            "    {} Check if kv_dim={} matches GPU buffer allocation",
            "T3:".white().bold(),
            kv_heads * head_dim,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VISUAL OUTPUT
// ═══════════════════════════════════════════════════════════════════════════════

/// Print the SPC control chart header
fn print_header() {
    eprintln!(
        "{}",
        "┌─────┬────────┬──────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐"
            .dimmed()
    );
    eprintln!(
        "│ {} │ {} │ {} │ {} │ {} │ {} │ {} │ {} │",
        " Pos ".dimmed(),
        " Token".dimmed(),
        " Argmax CPU→GPU ".dimmed(),
        " Max Diff".dimmed(),
        "Cos Sim  ".dimmed(),
        " KL Div ".dimmed(),
        " Sigma  ".dimmed(),
        " Verdict".dimmed(),
    );
    eprintln!(
        "{}",
        "├─────┼────────┼──────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤"
            .dimmed()
    );
}

/// Print one row of the SPC control chart
fn print_row(m: &SpcMetrics) {
    let verdict = m.verdict();

    // Argmax comparison with color
    let argmax_str = if m.cpu_argmax == m.gpu_argmax {
        format!("{:>6} = {:>6}", m.cpu_argmax, m.gpu_argmax)
            .green()
            .to_string()
    } else {
        format!("{:>6} ! {:>6}", m.cpu_argmax, m.gpu_argmax)
            .red()
            .bold()
            .to_string()
    };

    // Max diff with color + bar
    let diff_str = format_diff(m.max_abs_diff);

    // Cosine similarity with color
    let cos_str = format_cosine(m.cosine_similarity);

    // KL divergence with color
    let kl_str = format_kl(m.kl_divergence);

    // Sigma level with color
    let sigma_str = format_sigma(m.sigma_level);

    eprintln!(
        "{} {:>3} {} {:>6} {} {} {} {} {} {} {} {} {} {} {}",
        "│".dimmed(),
        m.position,
        "│".dimmed(),
        m.token_id,
        "│".dimmed(),
        argmax_str,
        "│".dimmed(),
        diff_str,
        "│".dimmed(),
        cos_str,
        "│".dimmed(),
        kl_str,
        "│".dimmed(),
        sigma_str,
        "│".dimmed(),
    );

    // Verdict column
    if !verdict.is_pass() {
        eprintln!(
            "{}      {}        {}                  {}          {}          {}          {}          {} {}",
            "│".dimmed(),
            "".dimmed(),
            "".dimmed(),
            "".dimmed(),
            "".dimmed(),
            "".dimmed(),
            "".dimmed(),
            verdict.symbol(),
            "│".dimmed(),
        );
    }
}

fn print_footer() {
    eprintln!(
        "{}",
        "└─────┴────────┴──────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘"
            .dimmed()
    );
}

/// Divergence bar visualization
fn format_diff(diff: f32) -> String {
    if diff < 0.01 {
        format!("{:>8.4}", diff).green().to_string()
    } else if diff < 0.1 {
        format!("{:>8.4}", diff).green().to_string()
    } else if diff < TOLERANCE_ABS {
        format!("{:>8.4}", diff).yellow().to_string()
    } else if diff < 5.0 {
        format!("{:>8.3}", diff).red().to_string()
    } else {
        format!("{:>8.2}", diff).red().bold().to_string()
    }
}

fn format_cosine(cos: f32) -> String {
    if cos >= 0.9999 {
        format!("{:>8.6}", cos).green().bold().to_string()
    } else if cos >= COSINE_SIM_MIN {
        format!("{:>8.6}", cos).green().to_string()
    } else if cos >= 0.99 {
        format!("{:>8.6}", cos).yellow().to_string()
    } else if cos >= 0.9 {
        format!("{:>8.4}", cos).red().to_string()
    } else {
        format!("{:>8.4}", cos).red().bold().to_string()
    }
}

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
    let avg_cpk: f64 = metrics.iter().map(|m| m.cpk()).sum::<f64>() / n as f64;

    let avg_oos: f64 =
        metrics.iter().map(|m| m.out_of_spec_count).sum::<usize>() as f64 / n as f64;
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
    eprintln!(
        "{}",
        "  STATISTICAL PROCESS CONTROL SUMMARY".cyan().bold()
    );
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
    eprintln!("  Defect rate:        {} (logits outside tolerance)", ppm_str);

    // Key metrics
    eprintln!();
    eprintln!("  {} {}", "Cosine similarity:".white().bold(), format_metric_range(avg_cos, min_cos, COSINE_SIM_MIN, true));
    eprintln!("  {} {}", "KL divergence:    ".white().bold(), format_metric_single(avg_kl, KL_DIV_MAX, false));
    eprintln!("  {} {}", "Max abs diff:     ".white().bold(), format_metric_single(max_diff as f64, TOLERANCE_ABS as f64, false));
    eprintln!("  {} {}", "Sigma level:      ".white().bold(), format_sigma_summary(avg_sigma));
    eprintln!("  {} {}", "Cpk:              ".white().bold(), format_cpk(avg_cpk));

    // Control limits legend
    eprintln!();
    eprintln!("{}", "  Control limits (from layer-parity-v1.yaml):".dimmed());
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
        format!("{} ({} {})", s.red().bold(), "incapable", "VIOLATED".red().bold())
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
    let cpu_max = cpu_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let gpu_max = gpu_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let cpu_exp: Vec<f64> = cpu_logits.iter().map(|x| ((*x - cpu_max) as f64).exp()).collect();
    let gpu_exp: Vec<f64> = gpu_logits.iter().map(|x| ((*x - gpu_max) as f64).exp()).collect();

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

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN ENTRY POINT
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
pub fn run(file: &Path, prompt: &str, assert: bool, verbose: bool) -> Result<()> {
    use realizar::gguf::{
        MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
    };

    if !file.exists() {
        return Err(CliError::FileNotFound(file.to_path_buf()));
    }

    // ── Header ──────────────────────────────────────────────────────────────
    eprintln!();
    eprintln!(
        "{}",
        "══════════════════════════════════════════════════════════════════════"
            .cyan()
            .bold()
    );
    eprintln!(
        "  {}  {}",
        "apr parity".cyan().bold(),
        "GPU/CPU Statistical Process Control".white()
    );
    eprintln!(
        "{}",
        "══════════════════════════════════════════════════════════════════════"
            .cyan()
            .bold()
    );

    // ── Load model ──────────────────────────────────────────────────────────
    let mapped = MappedGGUFModel::from_path(file)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to map model: {e}")))?;

    let tokens = mapped
        .model
        .encode(prompt)
        .unwrap_or_else(|| vec![1u32]);

    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create model: {e}")))?;

    let config = &model.config;
    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let kv_heads = config.num_kv_heads;
    let head_dim = if num_heads > 0 { hidden_dim / num_heads } else { 0 };
    let kv_dim = kv_heads * head_dim;
    let num_layers = config.num_layers;
    let gqa_ratio = if kv_heads > 0 { num_heads / kv_heads } else { 0 };

    eprintln!();
    eprintln!("  {} {}", "Model:".white().bold(), file.display());
    eprintln!("  {} {:?}", "Prompt:".white().bold(), prompt);
    eprintln!(
        "  {} {} tokens: {:?}",
        "Tokens:".white().bold(),
        tokens.len(),
        &tokens[..tokens.len().min(20)],
    );
    eprintln!(
        "  {} hidden={} heads={} kv_heads={} head_dim={} GQA={} layers={} vocab={}",
        "Arch:".white().bold(),
        hidden_dim, num_heads, kv_heads, head_dim, gqa_ratio, num_layers, config.vocab_size,
    );

    let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)
        .map_err(|e| CliError::ValidationFailed(format!("CUDA init failed: {e}")))?;

    eprintln!(
        "  {} {} ({} MB VRAM)",
        "GPU:".white().bold(),
        cuda_model.device_name().green(),
        cuda_model.vram_mb(),
    );

    let max_seq = tokens.len() + 1;

    // ── Run parity check ────────────────────────────────────────────────────
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, max_seq);
    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, max_seq);
    cuda_model.executor_mut().reset_kv_cache_gpu();

    eprintln!();
    print_header();

    let mut all_metrics = Vec::new();

    for (pos, &token_id) in tokens.iter().enumerate() {
        let cpu_logits = cuda_model
            .model()
            .forward_single_with_cache(token_id, &mut cpu_cache, pos)
            .map_err(|e| {
                CliError::InferenceFailed(format!("CPU forward failed at pos {pos}: {e}"))
            })?;

        let gpu_logits = cuda_model
            .forward_gpu_resident(token_id, &mut gpu_cache, pos)
            .map_err(|e| {
                CliError::InferenceFailed(format!("GPU forward failed at pos {pos}: {e}"))
            })?;

        let m = compute_metrics(&cpu_logits, &gpu_logits, pos, token_id);
        print_row(&m);

        if verbose && m.verdict().is_fail() {
            eprintln!(
                "{}     {} mean_diff={:.6} rmse={:.6} oos={}/{} {}",
                "│".dimmed(),
                "".dimmed(),
                m.mean_abs_diff,
                m.rmse,
                m.out_of_spec_count,
                m.vocab_size,
                "│".dimmed(),
            );
        }

        all_metrics.push(m);
    }

    print_footer();

    // ── Summary statistics ──────────────────────────────────────────────────
    print_summary(&all_metrics);

    // ── Auto-diagnosis ──────────────────────────────────────────────────────
    auto_diagnose(&all_metrics, hidden_dim, num_heads, kv_heads);

    // ── Exit code ───────────────────────────────────────────────────────────
    let has_failures = all_metrics.iter().any(|m| m.verdict().is_fail());
    if has_failures && assert {
        Err(CliError::ValidationFailed(
            "PARITY DISPROVEN: GPU/CPU divergence exceeds tolerance".to_string(),
        ))
    } else {
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
pub fn run(_file: &Path, _prompt: &str, _assert: bool, _verbose: bool) -> Result<()> {
    Err(CliError::FeatureDisabled(
        "cuda feature required for parity check".to_string(),
    ))
}
