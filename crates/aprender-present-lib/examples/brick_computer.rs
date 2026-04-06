//! Brick Computer Demo - SIMD Test Visualization
//!
//! Dynamic TUI showing Brick Architecture where each brick is a TEST
//! that runs actual SIMD compute operations:
//! - Bricks cycle: Idle → Computing → Pass/Fail
//! - Real SIMD operations (dot product, matrix ops)
//! - Parallel test execution with varying durations
//! - Inspired by trueno-viz and btop polish
//!
//! Run with: cargo run --example `brick_computer` -p presentar

use std::io::{self, Write};
use std::thread;
use std::time::{Duration, Instant};

// ============================================================================
// ANSI ESCAPE CODES
// ============================================================================

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const BLINK: &str = "\x1b[5m";

// Status colors (24-bit true color)
const GREEN: &str = "\x1b[38;2;74;222;128m";
const YELLOW: &str = "\x1b[38;2;250;204;21m";
const RED: &str = "\x1b[38;2;248;113;113m";
const CYAN: &str = "\x1b[38;2;34;211;238m";
const MAGENTA: &str = "\x1b[38;2;232;121;249m";
const BLUE: &str = "\x1b[38;2;96;165;250m";
const GRAY: &str = "\x1b[38;2;107;114;128m";
const WHITE: &str = "\x1b[38;2;248;250;252m";
const ORANGE: &str = "\x1b[38;2;251;146;60m";

// Background colors
const BG_GREEN: &str = "\x1b[48;2;22;163;74m";
const BG_YELLOW: &str = "\x1b[48;2;202;138;4m";
const BG_RED: &str = "\x1b[48;2;220;38;38m";
const BG_BLUE: &str = "\x1b[48;2;59;130;246m";
#[allow(dead_code)]
const BG_CYAN: &str = "\x1b[48;2;6;182;212m";
const BG_DARK: &str = "\x1b[48;2;30;41;59m";
const BG_DARKER: &str = "\x1b[48;2;15;23;42m";

// Box drawing
const BOX_TL: &str = "╭";
const BOX_TR: &str = "╮";
const BOX_BL: &str = "╰";
const BOX_BR: &str = "╯";
const BOX_H: &str = "─";
const BOX_V: &str = "│";

// Block and progress characters
#[allow(dead_code)]
const BLOCK_FULL: &str = "█";
const BLOCK_LIGHT: &str = "░";
const SPINNER: [&str; 8] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];
const PROGRESS: [&str; 8] = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];

// Braille for sparklines
const BRAILLE: [char; 8] = ['⣀', '⣄', '⣤', '⣦', '⣶', '⣷', '⣿', '⡿'];

// ============================================================================
// SIMD COMPUTE (Actual operations)
// ============================================================================

/// Actual SIMD dot product computation
fn simd_dot_product(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    // Manual SIMD-style: process 8 lanes
    let mut sum = 0.0f32;
    for i in 0..8 {
        sum += a[i] * b[i];
    }
    sum
}

/// SIMD matrix-vector multiply (8x8 * 8)
fn simd_matvec(matrix: &[[f32; 8]; 8], vec: &[f32; 8]) -> [f32; 8] {
    let mut result = [0.0f32; 8];
    for i in 0..8 {
        result[i] = simd_dot_product(&matrix[i], vec);
    }
    result
}

/// SIMD reduction (sum all lanes)
fn simd_reduce_sum(v: &[f32; 8]) -> f32 {
    v.iter().sum()
}

// ============================================================================
// TEST BRICK REPRESENTATION
// ============================================================================

#[derive(Clone, Copy, PartialEq)]
enum TestState {
    Idle,
    Queued,
    Running,
    Pass,
    Fail,
    Flaky,
}

#[derive(Clone, Copy, PartialEq)]
enum TestKind {
    DotProduct,
    MatVec,
    Reduce,
    Softmax,
    Attention,
    LayerNorm,
    Gelu,
    Transpose,
}

impl TestKind {
    #[allow(clippy::trivially_copy_pass_by_ref)]
    const fn name(&self) -> &'static str {
        match self {
            Self::DotProduct => "dot",
            Self::MatVec => "mvec",
            Self::Reduce => "red",
            Self::Softmax => "soft",
            Self::Attention => "attn",
            Self::LayerNorm => "norm",
            Self::Gelu => "gelu",
            Self::Transpose => "T",
        }
    }

    const fn from_index(i: usize) -> Self {
        match i % 8 {
            0 => Self::DotProduct,
            1 => Self::MatVec,
            2 => Self::Reduce,
            3 => Self::Softmax,
            4 => Self::Attention,
            5 => Self::LayerNorm,
            6 => Self::Gelu,
            _ => Self::Transpose,
        }
    }
}

struct TestBrick {
    id: usize,
    kind: TestKind,
    state: TestState,
    progress: u8,        // 0-100
    duration_ms: u32,    // How long this test takes
    elapsed_ms: u32,     // Time spent so far
    result: Option<f32>, // Computed result
    expected: f32,       // Expected result
    run_count: u32,
    pass_count: u32,
    fail_count: u32,
}

impl TestBrick {
    const fn new(id: usize) -> Self {
        let kind = TestKind::from_index(id);
        Self {
            id,
            kind,
            state: TestState::Idle,
            progress: 0,
            duration_ms: 50 + ((id * 17) % 100) as u32, // Varying durations
            elapsed_ms: 0,
            result: None,
            expected: 0.0,
            run_count: 0,
            pass_count: 0,
            fail_count: 0,
        }
    }

    fn start(&mut self) {
        self.state = TestState::Running;
        self.progress = 0;
        self.elapsed_ms = 0;
        self.result = None;
        self.run_count += 1;
    }

    fn tick(&mut self, delta_ms: u32, frame: u64) {
        match self.state {
            TestState::Idle => {
                // Random chance to queue
                if pseudo_random(frame + self.id as u64) % 20 == 0 {
                    self.state = TestState::Queued;
                }
            }
            TestState::Queued => {
                // Random chance to start
                if pseudo_random(frame + self.id as u64 * 3) % 5 == 0 {
                    self.start();
                }
            }
            TestState::Running => {
                self.elapsed_ms += delta_ms;
                self.progress =
                    ((self.elapsed_ms as f32 / self.duration_ms as f32) * 100.0).min(100.0) as u8;

                if self.elapsed_ms >= self.duration_ms {
                    // Execute actual SIMD compute
                    let (result, expected) = self.execute_simd(frame);
                    self.result = Some(result);
                    self.expected = expected;

                    // Check if test passed (with small epsilon)
                    let passed = (result - expected).abs() < 0.001;

                    // 5% flaky rate for drama
                    let is_flaky = pseudo_random(frame * 7 + self.id as u64) % 20 == 0;

                    if is_flaky {
                        self.state = TestState::Flaky;
                        self.fail_count += 1;
                    } else if passed {
                        self.state = TestState::Pass;
                        self.pass_count += 1;
                    } else {
                        self.state = TestState::Fail;
                        self.fail_count += 1;
                    }
                }
            }
            TestState::Pass | TestState::Fail | TestState::Flaky => {
                // Stay in terminal state for a bit, then reset
                if pseudo_random(frame + self.id as u64 * 11) % 30 == 0 {
                    self.state = TestState::Idle;
                    self.progress = 0;
                }
            }
        }
    }

    fn execute_simd(&self, seed: u64) -> (f32, f32) {
        // Generate test data based on seed
        let mut a = [0.0f32; 8];
        let mut b = [0.0f32; 8];
        for i in 0..8 {
            a[i] = ((seed + i as u64) % 100) as f32 / 10.0;
            b[i] = ((seed * 3 + i as u64) % 100) as f32 / 10.0;
        }

        match self.kind {
            TestKind::DotProduct => {
                let result = simd_dot_product(&a, &b);
                let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                (result, expected)
            }
            TestKind::Reduce => {
                let result = simd_reduce_sum(&a);
                let expected: f32 = a.iter().sum();
                (result, expected)
            }
            TestKind::MatVec => {
                let matrix = [[1.0f32; 8]; 8]; // Identity-ish
                let result = simd_matvec(&matrix, &a);
                (result[0], a.iter().sum())
            }
            TestKind::Softmax => {
                // Simplified softmax check
                let max_val = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = a.iter().map(|x| (x - max_val).exp()).sum();
                (sum, sum) // Always passes
            }
            TestKind::Attention => {
                let qk = simd_dot_product(&a, &b);
                let scaled = qk / 8.0f32.sqrt();
                (scaled, scaled)
            }
            TestKind::LayerNorm => {
                let mean = simd_reduce_sum(&a) / 8.0;
                let variance: f32 = a.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 8.0;
                (variance.sqrt(), variance.sqrt())
            }
            TestKind::Gelu => {
                let x = a[0];
                let gelu =
                    0.5 * x * (1.0 + (0.797_884_6 * 0.044_715_f32.mul_add(x.powi(3), x)).tanh());
                (gelu, gelu)
            }
            TestKind::Transpose => {
                // Check identity
                (a[0], a[0])
            }
        }
    }
}

/// Simple PRNG for animation variation
const fn pseudo_random(seed: u64) -> u64 {
    let mut x = seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

// ============================================================================
// BRICK COMPUTER (Test Runner)
// ============================================================================

struct BrickComputer {
    bricks: Vec<TestBrick>,
    frame: u64,
    start_time: Instant,
    history: Vec<f64>, // Pass rate history for sparkline
    total_pass: u64,
    total_fail: u64,
}

impl BrickComputer {
    fn new(count: usize) -> Self {
        Self {
            bricks: (0..count).map(TestBrick::new).collect(),
            frame: 0,
            start_time: Instant::now(),
            history: Vec::with_capacity(60),
            total_pass: 0,
            total_fail: 0,
        }
    }

    fn tick(&mut self, delta_ms: u32) {
        self.frame += 1;

        for brick in &mut self.bricks {
            let prev_state = brick.state;
            brick.tick(delta_ms, self.frame);

            // Track completions
            if prev_state == TestState::Running {
                match brick.state {
                    TestState::Pass => {
                        self.total_pass += 1;
                    }
                    TestState::Fail | TestState::Flaky => {
                        self.total_fail += 1;
                    }
                    _ => {}
                }
            }
        }

        // Update history
        let total = self.total_pass + self.total_fail;
        let pass_rate = if total > 0 {
            self.total_pass as f64 / total as f64
        } else {
            1.0
        };
        self.history.push(pass_rate);
        if self.history.len() > 60 {
            self.history.remove(0);
        }
    }

    fn running_count(&self) -> usize {
        self.bricks
            .iter()
            .filter(|b| b.state == TestState::Running)
            .count()
    }

    fn pass_count(&self) -> usize {
        self.bricks
            .iter()
            .filter(|b| b.state == TestState::Pass)
            .count()
    }

    fn fail_count(&self) -> usize {
        self.bricks
            .iter()
            .filter(|b| matches!(b.state, TestState::Fail | TestState::Flaky))
            .count()
    }

    fn tests_per_second(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            (self.total_pass + self.total_fail) as f64 / elapsed
        } else {
            0.0
        }
    }
}

// ============================================================================
// RENDERING
// ============================================================================

fn clear_screen() {
    print!("\x1b[2J\x1b[H");
}

fn move_to(x: usize, y: usize) {
    print!("\x1b[{y};{x}H");
}

fn draw_box(title: &str, x: usize, y: usize, w: usize, h: usize, color: &str) {
    move_to(x, y);
    print!("{}{}{}{}", color, BOX_TL, BOX_H.repeat(w - 2), BOX_TR);

    if !title.is_empty() {
        move_to(x + 2, y);
        print!("{BOLD}{color} {title} {RESET}");
    }

    for row in 1..h - 1 {
        move_to(x, y + row);
        print!("{color}{BOX_V}");
        move_to(x + w - 1, y + row);
        print!("{color}{BOX_V}");
    }

    move_to(x, y + h - 1);
    print!(
        "{}{}{}{}{}",
        color,
        BOX_BL,
        BOX_H.repeat(w - 2),
        BOX_BR,
        RESET
    );
}

fn draw_brick(brick: &TestBrick, x: usize, y: usize, frame: u64) {
    let spinner_idx = ((frame / 2) % 8) as usize;
    let progress_idx = (brick.progress as usize * 7 / 100).min(7);

    let (bg, fg, line1, line2, line3) = match brick.state {
        TestState::Idle => (
            BG_DARKER,
            GRAY,
            format!("  {BLOCK_LIGHT}  "),
            format!(" {:^4}", brick.kind.name()),
            format!("  {BLOCK_LIGHT}  "),
        ),
        TestState::Queued => (
            BG_DARK,
            YELLOW,
            "  ◌  ".to_string(),
            format!(" {:^4}", brick.kind.name()),
            " wait".to_string(),
        ),
        TestState::Running => (
            BG_BLUE,
            WHITE,
            format!("  {}  ", SPINNER[spinner_idx]),
            format!(" {:^4}", brick.kind.name()),
            format!(" {:>3}%", brick.progress),
        ),
        TestState::Pass => (
            BG_GREEN,
            WHITE,
            "  ✓  ".to_string(),
            format!(" {:^4}", brick.kind.name()),
            " PASS".to_string(),
        ),
        TestState::Fail => (
            BG_RED,
            WHITE,
            "  ✗  ".to_string(),
            format!(" {:^4}", brick.kind.name()),
            " FAIL".to_string(),
        ),
        TestState::Flaky => (
            BG_YELLOW,
            WHITE,
            "  ⚡ ".to_string(),
            format!(" {:^4}", brick.kind.name()),
            "FLAKY".to_string(),
        ),
    };

    // Draw 3-line brick with pulsing for running state
    let pulse = if brick.state == TestState::Running {
        if (frame / 4) % 2 == 0 {
            BOLD
        } else {
            ""
        }
    } else {
        ""
    };

    move_to(x, y);
    print!("{bg}{fg}{pulse}{line1}{RESET}");
    move_to(x, y + 1);
    print!("{bg}{fg}{pulse}{line2}{RESET}");
    move_to(x, y + 2);
    print!("{bg}{fg}{pulse}{line3}{RESET}");

    // Progress bar under brick when running
    #[allow(clippy::branches_sharing_code)]
    if brick.state == TestState::Running {
        move_to(x, y + 3);
        print!("{CYAN}");
        for i in 0..6 {
            if i * 100 / 6 < brick.progress as usize {
                print!("{}", PROGRESS[progress_idx]);
            } else {
                print!("{BLOCK_LIGHT}");
            }
        }
        print!("{RESET}");
    } else {
        move_to(x, y + 3);
        print!("      ");
    }
}

fn draw_sparkline(data: &[f64], x: usize, y: usize, width: usize) {
    move_to(x, y);

    if data.is_empty() {
        for _ in 0..width {
            print!("{}{}{}", GRAY, BRAILLE[0], RESET);
        }
        return;
    }

    let start = data.len().saturating_sub(width);
    for i in 0..width {
        let idx = start + i;
        let val = data.get(idx).copied().unwrap_or(1.0);
        let level = ((val * 7.0) as usize).min(7);

        let color = if val >= 0.95 {
            GREEN
        } else if val >= 0.80 {
            YELLOW
        } else {
            RED
        };

        print!("{}{}{}", color, BRAILLE[level], RESET);
    }
}

#[allow(dead_code)]
fn draw_meter(label: &str, value: f64, max: f64, x: usize, y: usize, width: usize, color: &str) {
    move_to(x, y);
    let bar_w = width - 12;
    let filled = ((value / max) * bar_w as f64).min(bar_w as f64) as usize;

    print!("{BOLD}{label:>8}{RESET} ");
    print!("{color}");
    for i in 0..bar_w {
        if i < filled {
            print!("{BLOCK_FULL}");
        } else {
            print!("{GRAY}{BLOCK_LIGHT}");
        }
    }
    print!("{RESET}");
}

fn draw_stats(computer: &BrickComputer, x: usize, y: usize, frame: u64) {
    let running = computer.running_count();
    let passed = computer.pass_count();
    let failed = computer.fail_count();

    move_to(x, y);
    print!(
        "{}Tests/sec:{} {:>8.1}",
        BOLD,
        RESET,
        computer.tests_per_second()
    );

    move_to(x, y + 1);
    print!(
        "{}Total:{} {:>12}",
        BOLD,
        RESET,
        computer.total_pass + computer.total_fail
    );

    move_to(x, y + 3);
    let spinner = SPINNER[((frame / 2) % 8) as usize];
    print!("{BOLD}Running:{RESET} ");
    if running > 0 {
        print!("{CYAN}{spinner} {running}{RESET}");
    } else {
        print!("{GRAY}0{RESET}");
    }

    move_to(x, y + 4);
    print!(
        "{}Pass:{} {}{}{} {}Fail:{} {}{}{}",
        BOLD,
        RESET,
        GREEN,
        passed,
        RESET,
        BOLD,
        RESET,
        if failed > 0 { RED } else { GRAY },
        failed,
        RESET
    );

    // Pass rate
    let total = computer.total_pass + computer.total_fail;
    let rate = if total > 0 {
        computer.total_pass as f64 / total as f64 * 100.0
    } else {
        100.0
    };
    move_to(x, y + 6);
    let rate_color = if rate >= 95.0 {
        GREEN
    } else if rate >= 80.0 {
        YELLOW
    } else {
        RED
    };
    print!("{BOLD}Pass Rate:{RESET} {rate_color}{rate:.1}%{RESET}");
}

fn draw_header(frame: u64) {
    move_to(1, 1);
    let pulse = if (frame / 8) % 2 == 0 { BOLD } else { "" };
    print!("{pulse}{MAGENTA}  BRICK COMPUTER  {RESET}");
    print!("{DIM}  SIMD Test Runner  {RESET}");
    print!("{CYAN}  presentar v0.2  {RESET}");
}

fn draw_legend(y: usize, frame: u64) {
    move_to(2, y);
    print!("{BOLD}Legend:{RESET} ");

    let spinner = SPINNER[((frame / 2) % 8) as usize];

    print!("{BG_DARK}{YELLOW} ◌ {RESET} Queue ");
    print!("{BG_BLUE}{WHITE} {spinner} {RESET} Run ");
    print!("{BG_GREEN}{WHITE} ✓ {RESET} Pass ");
    print!("{BG_RED}{WHITE} ✗ {RESET} Fail ");
    print!("{BG_YELLOW}{WHITE} ⚡{RESET} Flaky");
}

fn draw_jidoka(computer: &BrickComputer, x: usize, y: usize, frame: u64) {
    let any_fail = computer.fail_count() > 0;
    let any_running = computer.running_count() > 0;

    move_to(x, y);
    if any_fail {
        let blink = if (frame / 4) % 2 == 0 { BLINK } else { "" };
        print!("{blink}{RED}● JIDOKA HALT{RESET} - Test failure detected, investigating...");
    } else if any_running {
        let spinner = SPINNER[((frame / 2) % 8) as usize];
        print!(
            "{}{} JIDOKA{} - {} Tests executing SIMD compute...",
            CYAN,
            spinner,
            RESET,
            computer.running_count()
        );
    } else {
        print!("{BOLD}{GREEN}● JIDOKA PASS{RESET} - All bricks lit, render allowed");
    }

    // Show what SIMD ops are running
    move_to(x, y + 1);
    print!("{DIM}SIMD Ops:{RESET} ");
    for brick in &computer.bricks {
        if brick.state == TestState::Running {
            print!("{}{}{} ", ORANGE, brick.kind.name(), RESET);
        }
    }
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    // 16 test bricks (2 rows of 8)
    let brick_count = 16;
    let mut computer = BrickComputer::new(brick_count);

    // Hide cursor
    print!("\x1b[?25l");
    clear_screen();

    let tick_ms = 50u32; // 20 FPS
    let mut frame = 0u64;

    // Run until Ctrl+C
    loop {
        clear_screen();

        // Header
        draw_header(frame);

        // SIMD Test Bricks - 2 rows of 8
        draw_box("SIMD TEST BRICKS", 2, 3, 58, 12, CYAN);

        // Row 1: Bricks 0-7
        for i in 0..8 {
            let bx = 4 + (i * 7);
            draw_brick(&computer.bricks[i], bx, 5, frame);
        }

        // Row 2: Bricks 8-15
        for i in 8..16 {
            let bx = 4 + ((i - 8) * 7);
            draw_brick(&computer.bricks[i], bx, 10, frame);
        }

        // Stats panel
        draw_box("PERFORMANCE", 62, 3, 24, 12, MAGENTA);
        draw_stats(&computer, 64, 5, frame);

        // Sparkline panel
        draw_box("PASS RATE HISTORY", 2, 16, 84, 4, BLUE);
        move_to(4, 18);
        print!("{DIM}Rate:{RESET} ");
        draw_sparkline(&computer.history, 11, 18, 70);

        // JIDOKA status
        draw_box("VERIFICATION STATUS", 2, 21, 84, 4, GREEN);
        draw_jidoka(&computer, 4, 23, frame);

        // Legend
        draw_legend(26, frame);

        // Footer
        move_to(2, 28);
        print!("{DIM}Frame {frame} │ Press Ctrl+C to exit{RESET}");

        io::stdout().flush().ok();

        // Tick simulation
        computer.tick(tick_ms);
        frame += 1;

        thread::sleep(Duration::from_millis(u64::from(tick_ms)));
    }
}
