//! Content Creation Tooling Demo
//!
//! Demonstrates the Content Creation Tooling module from spec v1.1.0
//! for generating structured prompts for educational content.
//!
//! # Run
//! ```bash
//! cargo run --features native --example content_demo
//! ```

use batuta::content::{
    ContentType, ContentValidator, CourseLevel, EmitConfig, ModelContext, PromptEmitter,
    SourceContext, SourceSnippet, TokenBudget, ValidationSeverity,
};
use std::path::PathBuf;
use std::str::FromStr;

fn print_section_header(num: u32, title: &str) {
    println!("\n┌──────────────────────────────────────────────────────────────┐");
    println!("│ {}. {:58}│", num, title);
    println!("└──────────────────────────────────────────────────────────────┘\n");
}

fn demo_content_types() -> anyhow::Result<()> {
    print_section_header(1, "CONTENT TYPES");

    for ct in ContentType::all() {
        let range = ct.target_length();
        let length_desc = if range.start == 0 && range.end == 0 {
            "N/A".to_string()
        } else {
            format!("{}-{}", range.start, range.end)
        };
        println!("  {} ({}) - {} [{}]", ct.name(), ct.code(), ct.output_format(), length_desc);
    }

    println!("\n  Parsing from string:");
    let parsed = ContentType::from_str("bch")?;
    println!("    'bch' -> {} ({})", parsed.name(), parsed.code());

    Ok(())
}

fn demo_token_budgeting() {
    print_section_header(2, "TOKEN BUDGETING (Heijunka - Level Loading)");

    let budget = TokenBudget::new(ModelContext::Claude200K)
        .with_source_context(10_000)
        .with_rag_context(5_000)
        .with_output_target(15_000);

    println!("{}", budget.format_display("Claude Sonnet"));

    match budget.validate() {
        Ok(()) => println!("  ✓ Budget is valid\n"),
        Err(e) => println!("  ✗ Budget error: {}\n", e),
    }

    println!("  Model Context Windows:");
    let models = [
        ("Claude 200K", ModelContext::Claude200K),
        ("Gemini Pro", ModelContext::GeminiPro),
        ("GPT-4 Turbo", ModelContext::Gpt4Turbo),
    ];
    for (name, model) in models {
        println!("    {:15} {:>10} tokens", name, model.window_size());
    }
}

fn demo_source_context() {
    print_section_header(3, "SOURCE CONTEXT (Genchi Genbutsu - Go and See)");

    let mut source_ctx = SourceContext::new();
    source_ctx.add_snippet(SourceSnippet {
        path: PathBuf::from("src/lib.rs"),
        lines: Some((1, 15)),
        content: r#"/// Error handling with Result
pub fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("Division by zero".to_string())
    } else {
        Ok(a / b)
    }
}"#
        .to_string(),
        tokens: 85,
    });

    println!("  Source snippets added: {}", source_ctx.snippets.len());
    println!("  Total tokens used: {}", source_ctx.total_tokens);
    println!("\n  Formatted for prompt:");
    println!("  {}", "-".repeat(50));
    for line in source_ctx.format_for_prompt().lines().take(10) {
        println!("  {}", line);
    }
    println!("  ...");
}

fn demo_content_validation() {
    print_section_header(4, "CONTENT VALIDATION (Jidoka - Built-in Quality)");

    let validator = ContentValidator::new(ContentType::BookChapter);

    let good_content = r#"# Error Handling in Rust

Rust provides two primary mechanisms for handling errors: `Result<T, E>` and `panic!`.

## The Result Type

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::read_to_string("config.toml")?;
    println!("Config: {}", file);
    Ok(())
}
```

> **Note**: The `?` operator propagates errors automatically.

## Propagating Errors

### Using the ? Operator

The `?` operator unwraps `Ok` values or returns `Err` early.
"#;

    println!("  Testing GOOD content:");
    let result = validator.validate(good_content);
    println!("    Passed: {}", result.passed);
    println!("    Score: {}/100", result.score);
    println!("    Violations: {}", result.violations.len());

    let bad_content = r#"# Chapter Title

In this chapter, we will learn about error handling.

## Section 1

```
some code without language
```

#### Skipped H3

This section covers the basics.
"#;

    println!("\n  Testing BAD content:");
    let result = validator.validate(bad_content);
    println!("    Passed: {}", result.passed);
    println!("    Score: {}/100", result.score);
    println!("    Violations: {}", result.violations.len());

    for v in &result.violations {
        let severity = match v.severity {
            ValidationSeverity::Critical => "CRIT",
            ValidationSeverity::Error => "ERR ",
            ValidationSeverity::Warning => "WARN",
            ValidationSeverity::Info => "INFO",
        };
        println!("      [{}] {} @ {}", severity, v.constraint, v.location);
    }
}

fn demo_prompt_emission() -> anyhow::Result<()> {
    print_section_header(5, "PROMPT EMISSION");

    let emitter = PromptEmitter::new();

    let config = EmitConfig::new(ContentType::BookChapter)
        .with_title("SIMD Optimization in Rust")
        .with_audience("Systems programmers")
        .with_word_count(4000);

    let prompt = emitter.emit(&config)?;

    println!("  Generated Book Chapter Prompt:");
    println!("  {}", "-".repeat(50));
    for line in prompt.lines().take(20) {
        println!("  {}", line);
    }
    println!("  ... ({} total chars)", prompt.len());

    println!("\n  Generated Blog Post Prompt (summary):");
    let blog_config = EmitConfig::new(ContentType::BlogPost)
        .with_title("Why Rust for ML Inference")
        .with_word_count(1500);
    let blog_prompt = emitter.emit(&blog_config)?;
    println!("    Total length: {} chars", blog_prompt.len());
    println!("    Contains TOML: {}", blog_prompt.contains("TOML"));
    println!("    Contains SEO: {}", blog_prompt.contains("SEO"));

    println!("\n  Generated Presentar Demo Prompt (summary):");
    let demo_config =
        EmitConfig::new(ContentType::PresentarDemo).with_title("Shell Autocomplete WASM");
    let demo_prompt = emitter.emit(&demo_config)?;
    println!("    Total length: {} chars", demo_prompt.len());
    println!("    Contains WASM: {}", demo_prompt.contains("wasm"));
    println!("    Contains WCAG: {}", demo_prompt.contains("WCAG"));

    Ok(())
}

fn demo_course_levels() -> anyhow::Result<()> {
    print_section_header(6, "COURSE LEVELS (Configurable Detailed Outlines)");

    println!("  Available Course Levels:");
    println!("  {}", "-".repeat(50));

    let levels = [
        ("Short", CourseLevel::Short),
        ("Standard", CourseLevel::Standard),
        ("Extended", CourseLevel::Extended),
    ];

    for (name, level) in &levels {
        println!(
            "    {:10} │ {} week(s) │ {} modules │ {} videos/module",
            name,
            level.weeks(),
            level.modules(),
            level.videos_per_module()
        );
    }

    println!("\n  Parsing from string:");
    let parsed: CourseLevel = "short".parse()?;
    println!("    'short' -> {} weeks, {} modules", parsed.weeks(), parsed.modules());
    let parsed: CourseLevel = "extended".parse()?;
    println!("    'extended' -> {} weeks, {} modules", parsed.weeks(), parsed.modules());

    let emitter = PromptEmitter::new();
    demo_course_level_prompts(&emitter)?;

    Ok(())
}

fn demo_course_level_prompts(emitter: &PromptEmitter) -> anyhow::Result<()> {
    println!("\n  Generating Detailed Outlines with Course Levels:");
    println!("  {}", "-".repeat(50));

    let course_configs = [
        ("SHORT", "Quick Start Guide", CourseLevel::Short, "1 week", "2 modules", false),
        ("STANDARD", "Complete Course", CourseLevel::Standard, "3 weeks", "3 modules", true),
        (
            "EXTENDED",
            "Comprehensive Masterclass",
            CourseLevel::Extended,
            "6 weeks",
            "6 modules",
            true,
        ),
    ];

    for (label, title, level, weeks, modules, expect_weekly) in course_configs {
        let config = EmitConfig::new(ContentType::DetailedOutline)
            .with_title(title)
            .with_course_level(level);
        let prompt = emitter.emit(&config)?;

        println!("\n  {} Course:", label);
        println!("    Total prompt length: {} chars", prompt.len());
        println!("    Contains '{}': {}", weeks, prompt.contains(weeks));
        println!("    Contains '{}': {}", modules, prompt.contains(modules));
        println!(
            "    Has weekly objectives: {} (expected: {})",
            prompt.contains("weeks:\n"),
            expect_weekly
        );
    }

    // Show structure from standard course
    let standard_config = EmitConfig::new(ContentType::DetailedOutline)
        .with_title("Complete Course")
        .with_course_level(CourseLevel::Standard);
    let standard_prompt = emitter.emit(&standard_config)?;

    println!("\n  Sample Structure Requirements (Standard):");
    for line in standard_prompt.lines() {
        if line.contains("**Duration**")
            || line.contains("**Modules**")
            || line.contains("**Per Module**")
            || line.contains("**Learning Objectives**")
        {
            println!("    {}", line.trim());
        }
    }

    Ok(())
}

fn print_summary() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Toyota Way Principles Applied:                               ║");
    println!("║                                                                ║");
    println!("║  • Jidoka     - Quality gates catch errors automatically      ║");
    println!("║  • Poka-Yoke  - Structural constraints prevent mistakes       ║");
    println!("║  • Genchi Genbutsu - Source context grounds content           ║");
    println!("║  • Heijunka   - Token budgeting levels context usage          ║");
    println!("║  • Kaizen     - Dynamic templates enable improvement          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
}

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Content Creation Tooling Demo (Spec v1.1.0)              ║");
    println!("║     Toyota Way Principles for LLM Content Generation         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    demo_content_types()?;
    demo_token_budgeting();
    demo_source_context();
    demo_content_validation();
    demo_prompt_emission()?;
    demo_course_levels()?;
    print_summary();

    Ok(())
}
