//! Web Builders Demo
//!
//! Demonstrates Probar's type-safe web asset builders:
//! - HtmlBuilder: Accessible HTML generation
//! - CssBuilder: Structured CSS with variables
//! - JsBuilder: Minimal WASM loader (max 20 lines)
//! - WebBundle: Complete validated bundle
//!
//! Run with: cargo run --example web_builders_demo -p jugar-probar

use jugar_probar::web::{CssBuilder, CssRule, HtmlBuilder, JsBuilder, WebBundle};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Web Builders Demo (Type-Safe APIs)                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    demo_html_builder();
    demo_css_builder();
    demo_js_builder();
    demo_web_bundle();
}

/// Demonstrate HtmlBuilder
fn demo_html_builder() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  1. HtmlBuilder - Accessible HTML Generation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Build a complete HTML document
    let html = HtmlBuilder::new()
        .title("My WASM Game")
        .canvas("game-canvas", 800, 600)
        .build()
        .expect("HTML build failed");

    println!("  Generated HTML:");
    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!(
        "  │ Title: {}                                    │",
        html.title
    );
    println!("  │ Elements: {:41} │", html.elements.len());
    println!("  │ Size: {:45} │", format!("{} bytes", html.content.len()));
    println!("  └─────────────────────────────────────────────────────────┘");

    println!("\n  HTML Output (first 300 chars):");
    let preview: String = html.content.chars().take(300).collect();
    for line in preview.lines().take(10) {
        println!("    {}", line);
    }

    println!("\n  Builder Features:");
    println!("    ├─ .title(\"...\") - Set page title");
    println!("    ├─ .canvas(id, width, height) - Add canvas element");
    println!("    ├─ Auto-adds DOCTYPE, charset, viewport");
    println!("    ├─ Auto-adds lang=\"en\" for accessibility");
    println!("    └─ WCAG-compliant attributes on interactive elements");
    println!();
}

/// Demonstrate CssBuilder
fn demo_css_builder() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  2. CssBuilder - Structured Stylesheet Generation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Build a complete CSS stylesheet
    let css = CssBuilder::new()
        // Preset methods
        .reset() // Modern CSS reset
        .fullscreen_body() // Body fills viewport
        .responsive_canvas("game") // Canvas sizing
        // Custom rules with CssRule
        .rule(
            CssRule::new(".ui-panel")
                .declaration("background", "rgba(0, 0, 0, 0.8)")
                .declaration("border-radius", "8px")
                .declaration("padding", "16px"),
        )
        .rule(
            CssRule::new(".btn")
                .declaration("padding", "12px 24px")
                .declaration("font-size", "16px")
                .declaration("cursor", "pointer")
                .declaration("transition", "all 0.2s ease"),
        )
        .build()
        .expect("CSS build failed");

    println!("  Generated CSS:");
    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │ Rules: {:44} │", css.rules.len());
    println!("  │ Variables: {:41} │", css.variables.len());
    println!("  │ Size: {:45} │", format!("{} bytes", css.content.len()));
    println!("  └─────────────────────────────────────────────────────────┘");

    println!("\n  CSS Output:");
    for (i, line) in css.content.lines().take(15).enumerate() {
        if line.len() > 60 {
            println!("    {:2}. {}...", i + 1, &line[..57]);
        } else {
            println!("    {:2}. {}", i + 1, line);
        }
    }
    if css.content.lines().count() > 15 {
        println!("    ... ({} more lines)", css.content.lines().count() - 15);
    }

    println!("\n  Builder Features:");
    println!("    ├─ .reset() - Modern CSS reset (box-sizing, margins)");
    println!("    ├─ .fullscreen_body() - Body fills viewport");
    println!("    ├─ .responsive_canvas(id) - Canvas sizing");
    println!("    ├─ .rule(CssRule) - Add custom rules");
    println!("    └─ CssRule::new(selector).declaration(prop, val)");
    println!();
}

/// Demonstrate JsBuilder (minimal WASM loader)
fn demo_js_builder() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  3. JsBuilder - Minimal WASM Loader (Max 20 Lines)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Build minimal WASM loader
    let js = JsBuilder::new("game.wasm", "game-canvas")
        .memory(256, 1024) // 16MB initial, 64MB max (pages * 64KB)
        .entry_point("main") // WASM export to call
        .build()
        .expect("JS build failed");

    println!("  Generated JavaScript:");
    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!(
        "  │ Line count: {:39} │",
        format!("{} (limit: 20)", js.line_count)
    );
    println!(
        "  │ Within limit: {:37} │",
        if js.within_limit() {
            "✓ YES"
        } else {
            "✗ NO"
        }
    );
    println!("  │ Size: {:45} │", format!("{} bytes", js.content.len()));
    println!("  └─────────────────────────────────────────────────────────┘");

    println!("\n  Complete JS Output:");
    for (i, line) in js.content.lines().enumerate() {
        if line.len() > 60 {
            println!("    {:2}. {}...", i + 1, &line[..57]);
        } else {
            println!("    {:2}. {}", i + 1, line);
        }
    }

    println!("\n  Builder Features:");
    println!("    ├─ JsBuilder::new(wasm_path, canvas_id)");
    println!("    ├─ .memory(initial_pages, max_pages)");
    println!("    ├─ .entry_point(\"main\") - WASM function to call");
    println!("    ├─ Enforces <20 line limit (Zero-JS policy)");
    println!("    └─ Security checks block eval, innerHTML, etc.");
    println!();
}

/// Demonstrate WebBundle
fn demo_web_bundle() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  4. WebBundle - Complete Validated Bundle");
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

    println!("  Bundle Summary:");
    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!(
        "  │ HTML: {:36} │",
        format!(
            "{} bytes, {} elements",
            bundle.html.content.len(),
            bundle.html.elements.len()
        )
    );
    println!(
        "  │ CSS: {:37} │",
        format!(
            "{} bytes, {} rules",
            bundle.css.content.len(),
            bundle.css.rules.len()
        )
    );
    println!(
        "  │ JS: {:38} │",
        format!(
            "{} bytes, {} lines",
            bundle.js.content.len(),
            bundle.js.line_count
        )
    );
    println!("  └─────────────────────────────────────────────────────────┘");

    println!("\n  Validation Results:");
    println!(
        "    ├─ HTML: {}",
        if bundle.validation.html.is_valid() {
            "✓ Valid"
        } else {
            "✗ Invalid"
        }
    );
    println!(
        "    ├─ CSS:  {}",
        if bundle.validation.css.is_valid() {
            "✓ Valid"
        } else {
            "✗ Invalid"
        }
    );
    println!(
        "    ├─ JS:   {}",
        if bundle.validation.js.is_valid() {
            "✓ Valid"
        } else {
            "✗ Invalid"
        }
    );
    println!(
        "    └─ All:  {}",
        if bundle.is_valid() {
            "✓ Valid"
        } else {
            "✗ Invalid"
        }
    );

    println!("\n  Single-File Output:");
    let single = bundle.to_single_file();
    println!("    Total size: {} bytes", single.len());
    println!("\n    Preview (first 500 chars):");
    let preview: String = single.chars().take(500).collect();
    for line in preview.lines().take(15) {
        if line.len() > 60 {
            println!("      {}...", &line[..57]);
        } else {
            println!("      {}", line);
        }
    }

    println!("\n  Bundle Features:");
    println!("    ├─ WebBundle::new(html, css, js) - Combines assets");
    println!("    ├─ Auto-validates all assets on construction");
    println!("    ├─ .is_valid() - Check if bundle passes all validation");
    println!("    ├─ .to_single_file() - Inline CSS/JS into HTML");
    println!("    └─ Zero external dependencies (no npm, no bundlers)");

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Demo complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}
