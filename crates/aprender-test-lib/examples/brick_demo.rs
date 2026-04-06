//! Brick Architecture Demo
//!
//! Demonstrates the Brick Architecture where tests ARE the interface.
//! Widgets are defined by their assertions and performance budgets.
//!
//! Run with: cargo run --example brick_demo -p jugar-probar

#![allow(clippy::expect_used)]

use jugar_probar::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use jugar_probar::brick_house::BrickHouseBuilder;
use std::sync::Arc;
use std::time::Duration;

/// A simple status brick that displays a status message
struct StatusBrick {
    message: String,
    is_visible: bool,
}

impl StatusBrick {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            is_visible: true,
        }
    }
}

impl Brick for StatusBrick {
    fn brick_name(&self) -> &'static str {
        "StatusBrick"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        // Define what must be true for this brick to be valid
        &[
            BrickAssertion::TextVisible,
            BrickAssertion::ContrastRatio(4.5),
        ]
    }

    fn budget(&self) -> BrickBudget {
        // 50ms budget for status display
        BrickBudget::uniform(50)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        // Verify text visibility assertion
        if self.is_visible && !self.message.is_empty() {
            passed.push(BrickAssertion::TextVisible);
        } else {
            failed.push((BrickAssertion::TextVisible, "Text not visible".into()));
        }

        // Assume contrast ratio passes (would check actual colors in real impl)
        passed.push(BrickAssertion::ContrastRatio(4.5));

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(50),
        }
    }

    fn to_html(&self) -> String {
        format!(
            r#"<div class="status-brick" role="status">{}</div>"#,
            self.message
        )
    }

    fn to_css(&self) -> String {
        r#".status-brick {
    padding: 8px 16px;
    background: #1a1a2e;
    color: #eaeaea;
    border-radius: 4px;
    font-size: 14px;
}"#
        .into()
    }
}

/// A waveform visualization brick
struct WaveformBrick {
    samples: Vec<f32>,
    is_active: bool,
}

impl WaveformBrick {
    fn new(samples: Vec<f32>) -> Self {
        Self {
            samples,
            is_active: true,
        }
    }
}

// Static assertions for WaveformBrick
static WAVEFORM_ASSERTIONS: [BrickAssertion; 1] = [BrickAssertion::MaxLatencyMs(100)];

impl Brick for WaveformBrick {
    fn brick_name(&self) -> &'static str {
        "WaveformBrick"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &WAVEFORM_ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        // 100ms budget for waveform rendering
        BrickBudget::uniform(100)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        if self.is_active {
            passed.push(BrickAssertion::MaxLatencyMs(100));
        } else {
            failed.push((
                BrickAssertion::MaxLatencyMs(100),
                "Waveform not active".into(),
            ));
        }

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(100),
        }
    }

    fn to_html(&self) -> String {
        format!(
            r#"<canvas class="waveform" width="400" height="100" data-samples="{}"></canvas>"#,
            self.samples.len()
        )
    }

    fn to_css(&self) -> String {
        r#".waveform {
    background: #0f0f23;
    border: 1px solid #333;
}"#
        .into()
    }
}

/// A transcription display brick
struct TranscriptionBrick {
    text: String,
    is_final: bool,
}

impl TranscriptionBrick {
    fn new(text: impl Into<String>, is_final: bool) -> Self {
        Self {
            text: text.into(),
            is_final,
        }
    }
}

impl Brick for TranscriptionBrick {
    fn brick_name(&self) -> &'static str {
        "TranscriptionBrick"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::TextVisible,
            BrickAssertion::Focusable, // For accessibility
        ]
    }

    fn budget(&self) -> BrickBudget {
        // 600ms budget for transcription (can be slow)
        BrickBudget::uniform(600)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let failed = Vec::new();

        // Transcription is always "visible" even if empty (shows placeholder)
        passed.push(BrickAssertion::TextVisible);
        passed.push(BrickAssertion::Focusable);

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(200),
        }
    }

    fn to_html(&self) -> String {
        let class = if self.is_final { "final" } else { "interim" };
        format!(
            r#"<p class="transcription {}" tabindex="0">{}</p>"#,
            class, self.text
        )
    }

    fn to_css(&self) -> String {
        r#".transcription {
    font-size: 18px;
    line-height: 1.6;
    padding: 16px;
}
.transcription.final { color: #fff; }
.transcription.interim { color: #888; font-style: italic; }"#
            .into()
    }
}

fn main() {
    println!("=== Brick Architecture Demo ===\n");

    // Demo 1: Individual brick verification
    println!("1. Individual Brick Verification");
    println!("   -----------------------------");

    let status = StatusBrick::new("Recording...");
    let verification = status.verify();
    println!(
        "   StatusBrick: {} (score: {:.0}%)",
        if verification.is_valid() {
            "VALID"
        } else {
            "INVALID"
        },
        verification.score() * 100.0
    );

    let waveform = WaveformBrick::new(vec![0.1, 0.5, 0.3, 0.8, 0.2]);
    let verification = waveform.verify();
    println!(
        "   WaveformBrick: {} (score: {:.0}%)",
        if verification.is_valid() {
            "VALID"
        } else {
            "INVALID"
        },
        verification.score() * 100.0
    );

    let transcription = TranscriptionBrick::new("Hello, this is a test transcription.", true);
    let verification = transcription.verify();
    println!(
        "   TranscriptionBrick: {} (score: {:.0}%)\n",
        if verification.is_valid() {
            "VALID"
        } else {
            "INVALID"
        },
        verification.score() * 100.0
    );

    // Demo 2: BrickHouse composition with budget
    println!("2. BrickHouse Composition");
    println!("   -----------------------");

    let status_brick = Arc::new(StatusBrick::new("Listening..."));
    let waveform_brick = Arc::new(WaveformBrick::new(vec![0.2, 0.6, 0.4, 0.9, 0.1]));
    let transcription_brick = Arc::new(TranscriptionBrick::new(
        "Testing speech recognition...",
        false,
    ));

    let mut house = BrickHouseBuilder::new("whisper-app")
        .budget_ms(1000) // 1 second total budget
        .brick(status_brick, 50) // 50ms for status
        .brick(waveform_brick, 100) // 100ms for waveform
        .brick(transcription_brick, 600) // 600ms for transcription
        .build()
        .expect("Failed to build brick house");

    println!("   House: {}", house.name());
    println!("   Total budget: {}ms", house.budget().total_ms);
    println!("   Brick count: {}", house.brick_count());
    println!("   Remaining budget: {}ms\n", house.remaining_budget_ms());

    // Demo 3: Verify all bricks
    println!("3. House-wide Verification");
    println!("   ------------------------");

    let verifications = house.verify_all();
    for (name, verification) in &verifications {
        println!(
            "   {}: {} assertions passed, {} failed",
            name,
            verification.passed.len(),
            verification.failed.len()
        );
    }
    println!("   Can render: {}\n", house.can_render());

    // Demo 4: Render with budget tracking
    println!("4. Render with Budget Tracking");
    println!("   ----------------------------");

    match house.render() {
        Ok(html) => {
            println!("   Render SUCCESS");
            println!("   HTML length: {} chars", html.len());

            if let Some(report) = house.last_report() {
                println!("   Budget utilization: {:.1}%", report.utilization());
                println!("   Total used: {}ms", report.total_used_ms);
                println!("   Within budget: {}", report.within_budget());
            }
        }
        Err(e) => {
            println!("   Render FAILED: {}", e);
        }
    }

    // Demo 5: Generate combined CSS
    println!("\n5. Combined CSS Output");
    println!("   --------------------");
    let css = house.to_css();
    println!("   CSS length: {} chars", css.len());
    println!("   First 100 chars: {}...", &css[..css.len().min(100)]);

    println!("\n=== Demo Complete ===");
}
