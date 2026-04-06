//! Web Validation Demo
//!
//! Demonstrates Probar's Rust-native web validation capabilities:
//! - HTML validation (structure, nesting, attributes)
//! - CSS linting (syntax, selectors, values)
//! - JavaScript linting (minimal WASM loader only)
//! - Zero-JS policy enforcement
//!
//! Run with: cargo run --example web_validation_demo -p jugar-probar

use jugar_probar::web::{CssBuilder, CssRule, HtmlBuilder, JsBuilder, WebBundle, WebValidator};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Web Validation Demo (Rust-Native Tooling)              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    demo_html_validation();
    demo_css_validation();
    demo_js_validation();
    demo_web_bundle();
    demo_zero_js_policy();
}

/// Demonstrate HTML validation
fn demo_html_validation() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  1. HTML Validation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Valid HTML using the builder
    let valid_html = HtmlBuilder::new()
        .title("My WASM Game")
        .canvas("game-canvas", 800, 600)
        .build();

    match valid_html {
        Ok(html) => {
            println!("  Valid HTML Example:");
            println!("  ┌─────────────────────────────────────────────────────────┐");
            println!("  │ Title: {:42} │", html.title);
            println!("  │ Elements: {:40} │", html.elements.len());
            println!("  └─────────────────────────────────────────────────────────┘");

            let html_result = WebValidator::validate_html(&html);
            println!("\n  Validation Result:");
            println!("    ├─ Valid: {}", html_result.is_valid());
            println!("    ├─ Errors: {}", html_result.errors.len());
            println!("    └─ Warnings: {}", html_result.warnings.len());

            if html_result.is_valid() {
                println!("\n    ✓ HTML validation passed!");
            }
        }
        Err(e) => {
            println!("  Error building HTML: {}", e);
        }
    }
    println!();
}

/// Demonstrate CSS linting
fn demo_css_validation() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  2. CSS Linting");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Valid CSS using the builder
    let valid_css = CssBuilder::new()
        .reset() // Add CSS reset
        .fullscreen_body()
        .responsive_canvas("game-canvas")
        .rule(
            CssRule::new(".control-panel")
                .declaration("display", "flex")
                .declaration("gap", "1rem")
                .declaration("padding", "1rem"),
        )
        .rule(
            CssRule::new(".btn")
                .declaration("padding", "12px 24px")
                .declaration("font-size", "16px")
                .declaration("cursor", "pointer")
                .declaration("border", "none")
                .declaration("border-radius", "4px"),
        )
        .build();

    match valid_css {
        Ok(css) => {
            println!("  Valid CSS Example:");
            println!("  ┌─────────────────────────────────────────────────────────┐");
            println!("  │ Rules: {:44} │", css.rules.len());
            println!(
                "  │ Output size: {:38} │",
                format!("{} bytes", css.content.len())
            );
            println!("  └─────────────────────────────────────────────────────────┘");

            let css_result = WebValidator::lint_css(&css);
            println!("\n  Lint Result:");
            println!("    ├─ Valid: {}", css_result.is_valid());
            println!("    ├─ Errors: {}", css_result.errors.len());
            println!("    └─ Warnings: {}", css_result.warnings.len());

            // Show generated CSS snippet
            println!("\n  Generated CSS (first rule):");
            if let Some(rule) = css.rules.first() {
                println!("    {} {{", rule.selector);
                for (prop, val) in rule.declarations.iter().take(3) {
                    println!("      {}: {};", prop, val);
                }
                if rule.declarations.len() > 3 {
                    println!("      /* ... {} more */", rule.declarations.len() - 3);
                }
                println!("    }}");
            }
        }
        Err(e) => {
            println!("  Error building CSS: {}", e);
        }
    }
    println!();
}

/// Demonstrate JavaScript linting (minimal WASM loader)
fn demo_js_validation() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  3. JavaScript Linting (WASM Loader Only)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Minimal WASM loader (the ONLY JS allowed)
    let wasm_loader = JsBuilder::new("game.wasm", "game-canvas")
        .memory(256, 1024) // 16MB initial, 64MB max
        .entry_point("main")
        .build();

    match wasm_loader {
        Ok(js) => {
            println!("  WASM Loader (minimal JS):");
            println!("  ┌─────────────────────────────────────────────────────────┐");
            println!(
                "  │ Line count: {:39} │",
                format!("{} (max 20 allowed)", js.line_count)
            );
            println!(
                "  │ Within limit: {:37} │",
                if js.within_limit() { "✓" } else { "✗" }
            );
            println!("  └─────────────────────────────────────────────────────────┘");

            let js_result = WebValidator::lint_js(&js);
            println!("\n  Lint Result:");
            println!("    ├─ Valid: {}", js_result.is_valid());
            println!("    ├─ Errors: {}", js_result.errors.len());
            println!("    ├─ Warnings: {}", js_result.warnings.len());
            println!(
                "    └─ Security issues: {}",
                js_result.security_issues.len()
            );

            // Show the generated JS
            println!("\n  Generated JS:");
            for (i, line) in js.content.lines().take(5).enumerate() {
                let truncated = if line.len() > 55 {
                    format!("{}...", &line[..52])
                } else {
                    line.to_string()
                };
                println!("    {:2}. {}", i + 1, truncated);
            }
            if js.line_count > 5 {
                println!("    ... ({} more lines)", js.line_count - 5);
            }
        }
        Err(e) => {
            println!("  Error: {} (JS exceeded line limit!)", e);
        }
    }
    println!();
}

/// Demonstrate complete web bundle validation
fn demo_web_bundle() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  4. Complete Web Bundle");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create complete bundle
    let html = HtmlBuilder::new()
        .title("WASM Game")
        .canvas("game", 800, 600)
        .build()
        .expect("HTML build failed");

    let css = CssBuilder::new()
        .reset()
        .responsive_canvas("game")
        .build()
        .expect("CSS build failed");

    let js = JsBuilder::new("game.wasm", "game")
        .build()
        .expect("JS build failed");

    let bundle = WebBundle::new(html, css, js);

    println!("  Bundle Validation:");
    println!("    ├─ HTML valid: {}", bundle.validation.html.is_valid());
    println!("    ├─ CSS valid: {}", bundle.validation.css.is_valid());
    println!("    ├─ JS valid: {}", bundle.validation.js.is_valid());
    println!("    └─ Bundle valid: {}", bundle.is_valid());

    println!("\n  Bundle Output:");
    println!("    ├─ HTML size: {} bytes", bundle.html.content.len());
    println!("    ├─ CSS size: {} bytes", bundle.css.content.len());
    println!("    ├─ JS size: {} bytes", bundle.js.content.len());
    println!(
        "    └─ Total (single file): {} bytes",
        bundle.to_single_file().len()
    );

    println!("\n  Validation Report:");
    println!("    ├─ Total errors: {}", bundle.validation.error_count());
    println!(
        "    └─ Total warnings: {}",
        bundle.validation.warning_count()
    );
    println!();
}

/// Demonstrate zero-JS policy enforcement
fn demo_zero_js_policy() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  5. Zero-JS Policy Enforcement");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("  Policy Rules:");
    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │ 1. JavaScript limited to WASM loader only (<20 lines)   │");
    println!("  │ 2. No application logic in JavaScript                   │");
    println!("  │ 3. No npm/node_modules dependencies                     │");
    println!("  │ 4. No JS bundlers (webpack, rollup, etc.)              │");
    println!("  │ 5. All game code compiles to .wasm                     │");
    println!("  └─────────────────────────────────────────────────────────┘");

    println!("\n  Validation Stack (100% Rust):");
    println!("    ┌─────────────────┐");
    println!("    │  HtmlValidator  │ ─► Structure, nesting, a11y");
    println!("    ├─────────────────┤");
    println!("    │   CssLinter     │ ─► Syntax, selectors, values");
    println!("    ├─────────────────┤");
    println!("    │    JsLinter     │ ─► Minimal loader check");
    println!("    ├─────────────────┤");
    println!("    │ ZeroJsValidator │ ─► NO .js files in output");
    println!("    └─────────────────┘");

    println!("\n  No External Tools:");
    println!("    ✗ eslint      → Rust JsLinter");
    println!("    ✗ prettier    → Rust formatters");
    println!("    ✗ stylelint   → Rust CssLinter");
    println!("    ✗ htmlhint    → Rust HtmlValidator");

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Demo complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}
