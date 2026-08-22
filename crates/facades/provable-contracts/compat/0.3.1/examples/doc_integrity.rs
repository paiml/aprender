//! Document integrity validation example.
//!
//! Demonstrates mathematical enforcement of markdown and SVG structure.
//!
//! ```bash
//! cargo run --example doc_integrity
//! ```

use provable_contracts::doc_integrity::*;

fn main() {
    // --- Heading hierarchy ---
    println!("=== Heading Hierarchy ===");
    let good_md = "# Title\n## Section\n### Subsection\n## Another\n";
    let bad_md = "# Title\n### Skipped H2\n";
    let multi_h1 = "# First\n# Second\n";

    assert!(validate_heading_hierarchy(good_md).is_empty(), "valid hierarchy");
    let v = validate_heading_hierarchy(bad_md);
    println!("H1→H3 skip: {} violation(s) — {}", v.len(), v[0].message);
    let v = validate_heading_hierarchy(multi_h1);
    println!("Multiple H1: {} violation(s) — {}", v.len(), v[0].message);

    // --- Link safety ---
    println!("\n=== Link Safety ===");
    let safe = "[docs](https://docs.rs)\n";
    let xss = "[click](javascript:alert(1))\n";
    assert!(validate_links(safe).is_empty());
    let v = validate_links(xss);
    println!("XSS link: {}", v[0].message);

    // --- Code fence language tags ---
    println!("\n=== Code Fences ===");
    let tagged = "```rust\nfn main() {}\n```\n";
    let bare = "```\ncode\n```\n";
    assert!(validate_code_fences(tagged).is_empty());
    let v = validate_code_fences(bare);
    println!("Bare fence: {}", v[0].message);

    // --- Table column parity ---
    println!("\n=== Table Parity ===");
    let good_table = "| A | B |\n|---|---|\n| 1 | 2 |\n";
    let bad_table = "| A | B |\n|---|---|\n| 1 | 2 | 3 |\n";
    assert!(validate_tables(good_table).is_empty());
    let v = validate_tables(bad_table);
    println!("Column mismatch: {}", v[0].message);

    // --- SVG safety ---
    println!("\n=== SVG Safety ===");
    let safe_svg = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect/></svg>"#;
    let xss_svg = "<svg><script>alert(1)</script></svg>";
    assert!(validate_svg(safe_svg).is_empty());
    let v = validate_svg(xss_svg);
    println!("SVG violations: {}", v.iter().map(|v| v.message.as_str()).collect::<Vec<_>>().join(", "));

    // --- README drift ---
    println!("\n=== README Drift ===");
    let actual = "# Project\n\nHello world.\n";
    let generated = "# Project\n\nUpdated content.\n";
    let drift = detect_readme_drift(actual, generated);
    println!("Stale: {}, diff lines: {}", drift.stale, drift.diff_lines);

    // --- Required sections ---
    println!("\n=== Required Sections ===");
    let full = "# Project\n## Installation\n## Usage\n## License\n";
    let missing = "# Project\n## Usage\n";
    assert!(validate_required_sections(full, &["Installation", "Usage", "License"]).is_empty());
    let m = validate_required_sections(missing, &["Installation", "License"]);
    println!("Missing: {:?}", m);

    println!("\n✅ All document integrity checks demonstrated.");
}
