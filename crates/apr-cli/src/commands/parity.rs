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

#[allow(clippy::trivially_copy_pass_by_ref)]
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
    let pos0_fails = metrics.first().is_some_and(|m| m.verdict().is_fail());

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
            "  {} Position 0 diverges — bug is in CORE LAYER computation",
            "WHY 1:".red().bold(),
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
            "  {} Logit DIRECTION matches but MAGNITUDE differs — scaling/normalization bug",
            "WHY 2:".yellow().bold(),
        );
    }

    if all_fail && pos0_fails {
        eprintln!(
            "  {} ALL positions fail from pos 0 — systematic error in GPU forward pass",
            "WHY 3:".red().bold(),
        );
        eprintln!(
            "         {}",
            "Not a progressive drift. The FIRST layer computation is already wrong.".dimmed()
        );
    } else if growing {
        eprintln!(
            "  {} Divergence GROWS with position — error accumulates through KV cache",
            "WHY 3:".yellow().bold(),
        );
    }

    // Dimension-specific diagnosis
    if catastrophic && pos0_fails {
        eprintln!();
        eprintln!(
            "{}",
            "  Likely root causes (ranked by probability):"
                .cyan()
                .bold()
        );
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
    if diff < 0.1 {
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

include!("spc_color.rs");
include!("parity_03.rs");
