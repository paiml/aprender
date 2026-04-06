//! ComputeBrick Scoring Framework (section 29 of compute-block-tui-cbtop.md)

use std::fmt;

use super::widget::Color;
use super::Brick;

// ============================================================================
// ComputeBrick Scoring Framework (§29 of compute-block-tui-cbtop.md)
// ============================================================================

/// ComputeBrick quality score (0-100)
///
/// Scoring categories per §29.1:
/// - Performance: 40 pts (GFLOP/s throughput)
/// - Efficiency: 25 pts (backend utilization)
/// - Correctness: 20 pts (assertions, numerical accuracy)
/// - Stability: 15 pts (CV < 5%)
///
/// Reference: [Hennessy & Patterson, 2017] "Computer Architecture: A Quantitative Approach"
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrickScore {
    /// Performance score (0-40): GFLOP/s throughput vs theoretical peak
    pub performance: u8,
    /// Efficiency score (0-25): Backend utilization, memory efficiency
    pub efficiency: u8,
    /// Correctness score (0-20): Assertions passing, numerical accuracy
    pub correctness: u8,
    /// Stability score (0-15): CV < 5%, reproducibility
    pub stability: u8,
}

impl BrickScore {
    /// Create a new BrickScore with explicit values
    pub const fn new(performance: u8, efficiency: u8, correctness: u8, stability: u8) -> Self {
        Self {
            performance: if performance > 40 { 40 } else { performance },
            efficiency: if efficiency > 25 { 25 } else { efficiency },
            correctness: if correctness > 20 { 20 } else { correctness },
            stability: if stability > 15 { 15 } else { stability },
        }
    }

    /// Create a perfect score (100/100)
    pub const fn perfect() -> Self {
        Self::new(40, 25, 20, 15)
    }

    /// Create a zero score (0/100)
    pub const fn zero() -> Self {
        Self::new(0, 0, 0, 0)
    }

    /// Total score (0-100)
    pub const fn total(&self) -> u8 {
        self.performance + self.efficiency + self.correctness + self.stability
    }

    /// Letter grade based on total score (F501-F505 criteria)
    pub fn grade(&self) -> BrickGrade {
        match self.total() {
            90..=100 => BrickGrade::A,
            80..=89 => BrickGrade::B,
            70..=79 => BrickGrade::C,
            60..=69 => BrickGrade::D,
            _ => BrickGrade::F,
        }
    }

    /// Performance score as percentage (0.0-1.0)
    pub fn performance_pct(&self) -> f64 {
        self.performance as f64 / 40.0
    }

    /// Efficiency score as percentage (0.0-1.0)
    pub fn efficiency_pct(&self) -> f64 {
        self.efficiency as f64 / 25.0
    }

    /// Correctness score as percentage (0.0-1.0)
    pub fn correctness_pct(&self) -> f64 {
        self.correctness as f64 / 20.0
    }

    /// Stability score as percentage (0.0-1.0)
    pub fn stability_pct(&self) -> f64 {
        self.stability as f64 / 15.0
    }

    /// Calculate performance score from GFLOP/s vs theoretical peak
    /// Per §29.2: `min(40, (actual / theoretical) * 40)`
    pub fn score_performance(actual_gflops: f64, theoretical_gflops: f64) -> u8 {
        if theoretical_gflops <= 0.0 {
            return 0;
        }
        let ratio = actual_gflops / theoretical_gflops;
        (ratio * 40.0).min(40.0) as u8
    }

    /// Calculate performance score from speedup vs scalar baseline
    /// Per §29.2: `log2(speedup) * 5` capped at 20
    pub fn score_speedup(speedup: f64) -> u8 {
        if speedup <= 1.0 {
            return 0;
        }
        (speedup.log2() * 5.0).min(20.0) as u8
    }

    /// Calculate stability score from Coefficient of Variation
    /// Per §29.5: CV < 5% = 8 pts, CV < 10% = 4 pts, else 0
    pub fn score_cv(cv_percent: f64) -> u8 {
        if cv_percent < 5.0 {
            8
        } else if cv_percent < 10.0 {
            4
        } else {
            0
        }
    }

    /// Render a progress bar for a component (TUI display)
    pub fn render_bar(value: u8, max: u8, width: usize) -> String {
        let ratio = value as f64 / max as f64;
        let filled = (ratio * width as f64).round() as usize;
        let empty = width.saturating_sub(filled);
        format!("{}{}", "█".repeat(filled), "░".repeat(empty))
    }
}

impl Default for BrickScore {
    fn default() -> Self {
        Self::zero()
    }
}

impl fmt::Display for BrickScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/100 ({})", self.total(), self.grade().description())
    }
}

/// Letter grade for BrickScore
///
/// Note: Ordering is reversed (A > B > C > D > F) to match intuitive grade comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrickGrade {
    /// Failing (<60)
    F,
    /// Needs Improvement (60-69)
    D,
    /// Acceptable (70-79)
    C,
    /// Good (80-89)
    B,
    /// Excellent (90-100)
    A,
}

impl PartialOrd for BrickGrade {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BrickGrade {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Order by grade quality: A > B > C > D > F
        let self_val = match self {
            Self::A => 4,
            Self::B => 3,
            Self::C => 2,
            Self::D => 1,
            Self::F => 0,
        };
        let other_val = match other {
            Self::A => 4,
            Self::B => 3,
            Self::C => 2,
            Self::D => 1,
            Self::F => 0,
        };
        self_val.cmp(&other_val)
    }
}

impl BrickGrade {
    /// Human-readable description
    pub const fn description(&self) -> &'static str {
        match self {
            Self::A => "Excellent",
            Self::B => "Good",
            Self::C => "Acceptable",
            Self::D => "Needs Improvement",
            Self::F => "Failing",
        }
    }

    /// Single character representation
    pub const fn letter(&self) -> char {
        match self {
            Self::A => 'A',
            Self::B => 'B',
            Self::C => 'C',
            Self::D => 'D',
            Self::F => 'F',
        }
    }

    /// Color for TUI display (Andon-style)
    pub const fn color(&self) -> Color {
        match self {
            Self::A | Self::B => Color::ANDON_GREEN,
            Self::C => Color::ANDON_YELLOW,
            Self::D | Self::F => Color::ANDON_RED,
        }
    }
}

impl fmt::Display for BrickGrade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.letter())
    }
}

/// Trait extension for scoring ComputeBricks
///
/// Any Brick that implements Scorable can report its quality score.
pub trait Scorable: Brick {
    /// Calculate the current quality score
    fn score(&self) -> BrickScore;

    /// Generate a detailed score report (TUI-friendly)
    fn score_report(&self) -> String {
        let score = self.score();
        let perf_bar = BrickScore::render_bar(score.performance, 40, 20);
        let eff_bar = BrickScore::render_bar(score.efficiency, 25, 20);
        let corr_bar = BrickScore::render_bar(score.correctness, 20, 20);
        let stab_bar = BrickScore::render_bar(score.stability, 15, 20);

        format!(
            "╭──────────────────────────────────────────────────────╮\n\
             │      ComputeBrick Score: {:<24} │\n\
             ├──────────────────────────────────────────────────────┤\n\
             │ Performance: {:>5}/40  {} {:>3.0}% │\n\
             │ Efficiency:  {:>5}/25  {} {:>3.0}% │\n\
             │ Correctness: {:>5}/20  {} {:>3.0}% │\n\
             │ Stability:   {:>5}/15  {} {:>3.0}% │\n\
             ├──────────────────────────────────────────────────────┤\n\
             │ TOTAL SCORE: {:>6}/100                 Grade: {} │\n\
             ╰──────────────────────────────────────────────────────╯",
            self.brick_name(),
            score.performance,
            perf_bar,
            score.performance_pct() * 100.0,
            score.efficiency,
            eff_bar,
            score.efficiency_pct() * 100.0,
            score.correctness,
            corr_bar,
            score.correctness_pct() * 100.0,
            score.stability,
            stab_bar,
            score.stability_pct() * 100.0,
            score.total(),
            score.grade()
        )
    }
}
