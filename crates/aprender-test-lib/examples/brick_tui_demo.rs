//! Brick Architecture TUI Demo
//!
//! Visual demonstration of bricks "lighting up" as tests pass.
//! Shows real-time verification with performance metrics.
//!
//! Run with: cargo run --example brick_tui_demo -p jugar-probar --features tui

use jugar_probar::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use std::io::{self, Write};
use std::thread;
use std::time::{Duration, Instant};

// ANSI color codes
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const CYAN: &str = "\x1b[36m";
const MAGENTA: &str = "\x1b[35m";
const BG_GREEN: &str = "\x1b[42m";
#[allow(dead_code)]
const BG_YELLOW: &str = "\x1b[43m";
const BG_RED: &str = "\x1b[41m";
const BLACK: &str = "\x1b[30m";

/// Visual brick representation
struct VisualBrick {
    name: &'static str,
    symbol: char,
    assertions: Vec<BrickAssertion>,
    budget_ms: u32,
    will_pass: bool,
    latency_ms: u32,
}

impl VisualBrick {
    fn new(
        name: &'static str,
        symbol: char,
        budget_ms: u32,
        will_pass: bool,
        latency_ms: u32,
    ) -> Self {
        Self {
            name,
            symbol,
            assertions: vec![
                BrickAssertion::TextVisible,
                BrickAssertion::ContrastRatio(4.5),
                BrickAssertion::MaxLatencyMs(budget_ms),
            ],
            budget_ms,
            will_pass,
            latency_ms,
        }
    }
}

impl Brick for VisualBrick {
    fn brick_name(&self) -> &'static str {
        self.name
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &self.assertions
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(self.budget_ms)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        if self.will_pass {
            for assertion in &self.assertions {
                passed.push(assertion.clone());
            }
        } else {
            passed.push(BrickAssertion::TextVisible);
            failed.push((
                BrickAssertion::ContrastRatio(4.5),
                "Contrast ratio 3.2 < 4.5".into(),
            ));
        }

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_millis(self.latency_ms as u64),
        }
    }

    fn to_html(&self) -> String {
        format!(
            "<div class=\"brick brick-{}\">{}</div>",
            self.name, self.symbol
        )
    }

    fn to_css(&self) -> String {
        format!(".brick-{} {{ display: block; }}", self.name)
    }
}

fn clear_screen() {
    print!("\x1b[2J\x1b[H");
    io::stdout().flush().ok();
}

fn draw_brick_row(bricks: &[VisualBrick], states: &[Option<bool>]) {
    // Draw top border
    print!("    ");
    for _ in bricks {
        print!("+-------");
    }
    println!("+");

    // Draw brick symbols with status
    print!("    ");
    for (i, brick) in bricks.iter().enumerate() {
        let (bg, fg) = match states[i] {
            None => (DIM, " "),
            Some(true) => (BG_GREEN, BLACK),
            Some(false) => (BG_RED, BLACK),
        };
        print!("|{}{} {} {} {}", bg, fg, brick.symbol, brick.symbol, RESET);
    }
    println!("|");

    // Draw brick names
    print!("    ");
    for (i, brick) in bricks.iter().enumerate() {
        let (bg, fg) = match states[i] {
            None => (DIM, " "),
            Some(true) => (BG_GREEN, BLACK),
            Some(false) => (BG_RED, BLACK),
        };
        let name = &brick.name[..brick.name.len().min(5)];
        print!("|{}{}{:^7}{}", bg, fg, name, RESET);
    }
    println!("|");

    // Draw bottom border
    print!("    ");
    for _ in bricks {
        print!("+-------");
    }
    println!("+");
}

fn draw_header(title: &str) {
    println!();
    println!("  {BOLD}{CYAN}========================================{RESET}");
    println!("  {BOLD}{CYAN}  {}{RESET}", title);
    println!("  {BOLD}{CYAN}========================================{RESET}");
    println!();
}

fn draw_metrics(brick: &VisualBrick, verification: &BrickVerification, elapsed: Duration) {
    let status = if verification.is_valid() {
        format!("{GREEN}PASS{RESET}")
    } else {
        format!("{RED}FAIL{RESET}")
    };

    let budget_status = if elapsed.as_millis() as u32 <= brick.budget_ms {
        format!("{GREEN}{:>3}ms{RESET}", elapsed.as_millis())
    } else {
        format!("{RED}{:>3}ms{RESET}", elapsed.as_millis())
    };

    println!(
        "    {BOLD}{:12}{RESET} [{status}] budget: {:>3}ms actual: {budget_status} assertions: {}/{} ",
        brick.name,
        brick.budget_ms,
        verification.passed.len(),
        verification.passed.len() + verification.failed.len()
    );

    for (assertion, reason) in &verification.failed {
        println!("      {RED}x {assertion:?}: {reason}{RESET}");
    }
}

fn draw_summary(total_pass: usize, total_fail: usize, total_time: Duration) {
    println!();
    println!("  {BOLD}Summary{RESET}");
    println!("  -------");
    println!(
        "    Bricks: {GREEN}{total_pass} passed{RESET}, {}{total_fail} failed{RESET}",
        if total_fail > 0 { RED } else { GREEN }
    );
    println!("    Total verification time: {:?}", total_time);
    println!(
        "    Status: {}{}{}",
        BOLD,
        if total_fail == 0 {
            format!("{GREEN}ALL BRICKS LIT{RESET}")
        } else {
            format!("{RED}SOME BRICKS DARK{RESET}")
        },
        RESET
    );
    println!();
}

fn main() {
    // Define our visual bricks
    let bricks = vec![
        VisualBrick::new("Status", 'S', 50, true, 12),
        VisualBrick::new("Wave", 'W', 100, true, 45),
        VisualBrick::new("Audio", 'A', 150, true, 67),
        VisualBrick::new("Trans", 'T', 600, true, 234),
        VisualBrick::new("Error", 'E', 50, false, 15),
        VisualBrick::new("Model", 'M', 200, true, 89),
    ];

    let mut states: Vec<Option<bool>> = vec![None; bricks.len()];
    let mut total_pass = 0;
    let mut total_fail = 0;
    let start = Instant::now();

    clear_screen();
    draw_header("BRICK ARCHITECTURE - LIVE VERIFICATION");

    println!("  {DIM}Verifying {} bricks...{RESET}", bricks.len());
    println!();

    draw_brick_row(&bricks, &states);
    println!();

    // Animate brick verification
    for (i, brick) in bricks.iter().enumerate() {
        // Simulate verification delay
        thread::sleep(Duration::from_millis(150));

        let _verify_start = Instant::now();
        let verification = brick.verify();
        let _elapsed = Duration::from_millis(brick.latency_ms as u64);

        states[i] = Some(verification.is_valid());

        if verification.is_valid() {
            total_pass += 1;
        } else {
            total_fail += 1;
        }

        // Redraw
        clear_screen();
        draw_header("BRICK ARCHITECTURE - LIVE VERIFICATION");

        println!(
            "  {MAGENTA}Verifying brick {}/{}: {}{RESET}",
            i + 1,
            bricks.len(),
            brick.name
        );
        println!();

        draw_brick_row(&bricks, &states);
        println!();

        println!("  {BOLD}Verification Results{RESET}");
        println!("  --------------------");

        for (j, b) in bricks.iter().enumerate() {
            if j <= i {
                let v = b.verify();
                let d = Duration::from_millis(b.latency_ms as u64);
                draw_metrics(b, &v, d);
            }
        }
    }

    // Final summary
    thread::sleep(Duration::from_millis(300));
    clear_screen();
    draw_header("BRICK ARCHITECTURE - VERIFICATION COMPLETE");

    draw_brick_row(&bricks, &states);

    println!();
    println!("  {BOLD}Final Verification Results{RESET}");
    println!("  -------------------------");

    for brick in &bricks {
        let verification = brick.verify();
        let elapsed = Duration::from_millis(brick.latency_ms as u64);
        draw_metrics(brick, &verification, elapsed);
    }

    draw_summary(total_pass, total_fail, start.elapsed());

    // Budget breakdown
    println!("  {BOLD}Budget Breakdown{RESET}");
    println!("  ----------------");
    let total_budget: u32 = bricks.iter().map(|b| b.budget_ms).sum();
    let total_used: u32 = bricks.iter().map(|b| b.latency_ms).sum();
    let utilization = (total_used as f64 / total_budget as f64) * 100.0;

    println!("    Total budget:  {:>6}ms", total_budget);
    println!("    Total used:    {:>6}ms", total_used);
    println!(
        "    Utilization:   {:>5.1}% {}",
        utilization,
        if utilization <= 80.0 {
            format!("{GREEN}(healthy){RESET}")
        } else if utilization <= 100.0 {
            format!("{YELLOW}(warning){RESET}")
        } else {
            format!("{RED}(exceeded!){RESET}")
        }
    );

    // Visual budget bar
    print!("    [");
    let bar_width = 40;
    let filled = ((utilization / 100.0) * bar_width as f64).min(bar_width as f64) as usize;
    for i in 0..bar_width {
        if i < filled {
            if utilization <= 80.0 {
                print!("{GREEN}#{RESET}");
            } else if utilization <= 100.0 {
                print!("{YELLOW}#{RESET}");
            } else {
                print!("{RED}#{RESET}");
            }
        } else {
            print!("{DIM}.{RESET}");
        }
    }
    println!("]");

    println!();
    println!("  {DIM}Demo complete. Press Ctrl+C to exit.{RESET}");
    println!();
}
