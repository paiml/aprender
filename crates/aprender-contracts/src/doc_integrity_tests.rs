use super::*;
use std::io::Write as _;

// =========================================================================
// FALSIFY-DOC-001: H1->H3 skip detected as violation
// =========================================================================
#[test]
fn falsify_doc_001_heading_skip() {
    let md = "# Title\n### Skipped H2\n";
    let violations = validate_heading_hierarchy(md);
    assert!(
        !violations.is_empty(),
        "heading skip H1->H3 must be detected"
    );
    assert!(violations.iter().any(|v| v.message.contains("skip")));
}

// =========================================================================
// FALSIFY-DOC-002: Multiple H1s detected as violation
// =========================================================================
#[test]
fn falsify_doc_002_duplicate_h1() {
    let md = "# Title\n# Another Title\n";
    let violations = validate_heading_hierarchy(md);
    assert!(!violations.is_empty(), "duplicate H1 must be detected");
    assert!(
        violations
            .iter()
            .any(|v| v.message.contains("duplicate H1"))
    );
}

// =========================================================================
// FALSIFY-DOC-003: javascript:alert() link flagged
// =========================================================================
#[test]
fn falsify_doc_003_javascript_link() {
    let md = "[click](javascript:alert(1))\n";
    let violations = validate_links(md);
    assert!(!violations.is_empty(), "javascript: link must be flagged");
    assert!(violations.iter().any(|v| v.message.contains("javascript:")));
}

// =========================================================================
// FALSIFY-DOC-004: Bare ``` flagged as violation
// =========================================================================
#[test]
fn falsify_doc_004_bare_code_fence() {
    let md = "```\ncode\n```\n";
    let violations = validate_code_fences(md);
    assert!(!violations.is_empty(), "bare code fence must be detected");
    assert!(violations.iter().any(|v| v.rule == "code-fence-language"));
}

// =========================================================================
// FALSIFY-DOC-005: Mismatched column count flagged
// =========================================================================
#[test]
fn falsify_doc_005_table_column_mismatch() {
    let md = "| A | B |\n|---|---|\n| 1 | 2 | 3 |\n";
    let violations = validate_tables(md);
    assert!(
        !violations.is_empty(),
        "table column mismatch must be detected"
    );
    assert!(violations.iter().any(|v| v.rule == "table-column-parity"));
}

// =========================================================================
// FALSIFY-DOC-006: SVG with <script> tag flagged
// =========================================================================
#[test]
fn falsify_doc_006_svg_script_injection() {
    let svg = "<svg><script>alert(1)</script></svg>";
    let violations = validate_svg(svg);
    assert!(!violations.is_empty(), "SVG with <script> must be flagged");
    assert!(violations.iter().any(|v| v.message.contains("<script>")));
}

// =========================================================================
// FALSIFY-DOC-007: Modified README detected as stale
// =========================================================================
#[test]
fn falsify_doc_007_readme_drift() {
    let actual = "# Old content\n";
    let generated = "# New content\n";
    let result = detect_readme_drift(actual, generated);
    assert!(result.stale, "stale README must be detected");
    assert!(result.diff_lines > 0);
}

// =========================================================================
// FALSIFY-DOC-008: SVG without viewBox flagged
// =========================================================================
#[test]
fn falsify_doc_008_svg_missing_viewbox() {
    let svg = "<svg xmlns='http://www.w3.org/2000/svg'><rect/></svg>";
    let violations = validate_svg(svg);
    assert!(
        violations.iter().any(|v| v.message.contains("viewBox")),
        "missing viewBox must be detected"
    );
}

// =========================================================================
// FALSIFY-DOC-009: README without License section flagged
// =========================================================================
#[test]
fn falsify_doc_009_missing_required_section() {
    let md = "# Project\n## Usage\nHello\n";
    let missing = validate_required_sections(md, &["License"]);
    assert!(
        !missing.is_empty(),
        "missing License section must be detected"
    );
    assert!(missing.contains(&"License".to_string()));
}

// =========================================================================
// Additional coverage: heading hierarchy — valid document
// =========================================================================
#[test]
fn heading_hierarchy_valid() {
    let md = "# Title\n## Section\n### Sub\n## Another\n";
    let violations = validate_heading_hierarchy(md);
    assert!(
        violations.is_empty(),
        "valid hierarchy should pass: {:?}",
        violations
    );
}

#[test]
fn heading_hierarchy_first_not_h1() {
    let md = "## Not H1\n### Sub\n";
    let violations = validate_heading_hierarchy(md);
    assert!(
        violations
            .iter()
            .any(|v| v.message.contains("first heading must be H1"))
    );
}

#[test]
fn heading_hierarchy_empty_document() {
    let md = "No headings here.\n";
    let violations = validate_heading_hierarchy(md);
    assert!(violations.is_empty());
}

// =========================================================================
// Additional coverage: links
// =========================================================================
#[test]
fn link_valid() {
    let md = "[text](https://example.com)\n";
    let violations = validate_links(md);
    assert!(
        violations.is_empty(),
        "valid link should pass: {:?}",
        violations
    );
}

#[test]
fn link_empty_url() {
    let md = "[text]()\n";
    let violations = validate_links(md);
    assert!(violations.iter().any(|v| v.message.contains("empty")));
}

#[test]
fn link_unescaped_space() {
    let md = "[text](https://example.com/path with spaces)\n";
    let violations = validate_links(md);
    assert!(violations.iter().any(|v| v.message.contains("space")));
}

#[test]
fn image_link_validated() {
    let md = "![alt](javascript:evil)\n";
    let violations = validate_links(md);
    assert!(!violations.is_empty(), "image links must also be checked");
}

// =========================================================================
// Additional coverage: code fences
// =========================================================================
#[test]
fn code_fence_with_language() {
    let md = "```rust\nfn main() {}\n```\n";
    let violations = validate_code_fences(md);
    assert!(
        violations.is_empty(),
        "tagged opening fence should produce zero violations: {:?}",
        violations
    );
}

#[test]
fn code_fence_closing_not_flagged() {
    // Only opening fences need language tags; closing fences are always bare.
    let md = "```rust\nfn main() {}\n```\n";
    let violations = validate_code_fences(md);
    assert!(violations.is_empty());
}

// =========================================================================
// Fence-awareness: headings inside code blocks must be ignored
// =========================================================================
#[test]
fn heading_inside_code_fence_ignored() {
    let md = "# Title\n```bash\n# this is a comment, not a heading\n```\n## Section\n";
    let violations = validate_heading_hierarchy(md);
    assert!(
        violations.is_empty(),
        "hash-comments inside code fences must not be treated as headings: {:?}",
        violations
    );
}

#[test]
fn link_inside_code_fence_ignored() {
    let md = "# Title\n```markdown\n[text](javascript:evil)\n```\n";
    let violations = validate_links(md);
    assert!(
        violations.is_empty(),
        "links inside code fences must not be validated: {:?}",
        violations
    );
}

#[test]
fn table_inside_code_fence_ignored() {
    let md = "# Title\n```text\n| A | B |\n|---|\n| 1 | 2 | 3 |\n```\n";
    let violations = validate_tables(md);
    assert!(
        violations.is_empty(),
        "tables inside code fences must not be validated: {:?}",
        violations
    );
}

#[test]
fn required_section_inside_fence_not_counted() {
    let md = "# Project\n```bash\n# License\n```\n";
    let missing = validate_required_sections(md, &["License"]);
    assert!(
        !missing.is_empty(),
        "heading inside code fence must not satisfy required section check"
    );
}

// =========================================================================
// Additional coverage: tables
// =========================================================================
#[test]
fn table_valid() {
    let md = "| A | B |\n|---|---|\n| 1 | 2 |\n";
    let violations = validate_tables(md);
    assert!(
        violations.is_empty(),
        "valid table should pass: {:?}",
        violations
    );
}

#[test]
fn table_separator_mismatch() {
    let md = "| A | B | C |\n|---|---|\n| 1 | 2 | 3 |\n";
    let violations = validate_tables(md);
    assert!(violations.iter().any(|v| v.rule == "table-column-parity"));
}

// =========================================================================
// Additional coverage: SVG
// =========================================================================
#[test]
fn svg_valid() {
    let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect/></svg>"#;
    let violations = validate_svg(svg);
    assert!(
        violations.is_empty(),
        "valid SVG should pass: {:?}",
        violations
    );
}

#[test]
fn svg_missing_xmlns() {
    let svg = r#"<svg viewBox="0 0 100 100"><rect/></svg>"#;
    let violations = validate_svg(svg);
    assert!(violations.iter().any(|v| v.message.contains("xmlns")));
}

#[test]
fn svg_no_svg_element() {
    let violations = validate_svg("<div>not svg</div>");
    assert!(violations.iter().any(|v| v.message.contains("<svg>")));
}

#[test]
fn svg_foreign_object() {
    let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><foreignObject>hi</foreignObject></svg>"#;
    let violations = validate_svg(svg);
    assert!(
        violations
            .iter()
            .any(|v| v.message.contains("foreignObject"))
    );
}

// =========================================================================
// Additional coverage: required sections
// =========================================================================
#[test]
fn required_sections_all_present() {
    let md = "# Project\n## Installation\n## Usage\n## License\n";
    let missing = validate_required_sections(md, &["Installation", "Usage", "License"]);
    assert!(missing.is_empty(), "all present: {:?}", missing);
}

#[test]
fn required_sections_case_insensitive() {
    let md = "# Project\n## license\n";
    let missing = validate_required_sections(md, &["License"]);
    assert!(missing.is_empty(), "case-insensitive match should work");
}

// =========================================================================
// Additional coverage: drift detection
// =========================================================================
#[test]
fn drift_identical() {
    let content = "# Title\n\nBody text.\n";
    let result = detect_readme_drift(content, content);
    assert!(!result.stale);
    assert_eq!(result.diff_lines, 0);
}

#[test]
fn drift_trailing_whitespace_normalized() {
    let actual = "# Title  \nBody.\n";
    let generated = "# Title\nBody.\n";
    let result = detect_readme_drift(actual, generated);
    assert!(!result.stale, "trailing whitespace should be normalized");
}

#[test]
fn drift_different_line_count() {
    let actual = "# Title\n";
    let generated = "# Title\nExtra line\n";
    let result = detect_readme_drift(actual, generated);
    assert!(result.stale);
    assert_eq!(result.diff_lines, 1);
}

// =========================================================================
// Additional coverage: validate_document dispatch
// =========================================================================
#[test]
fn validate_document_md() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("test.md");
    let mut f = std::fs::File::create(&path).expect("create");
    f.write_all(b"# Title\n## Section\n").expect("write");
    drop(f);

    let violations = validate_document(&path);
    assert!(violations.is_empty(), "valid .md: {:?}", violations);
}

#[test]
fn validate_document_svg() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("test.svg");
    let mut f = std::fs::File::create(&path).expect("create");
    f.write_all(br#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect/></svg>"#)
        .expect("write");
    drop(f);

    let violations = validate_document(&path);
    assert!(violations.is_empty(), "valid .svg: {:?}", violations);
}

#[test]
fn validate_document_unsupported() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("test.txt");
    std::fs::File::create(&path).expect("create");

    let violations = validate_document(&path);
    assert!(violations.iter().any(|v| v.rule == "unsupported-extension"));
}

#[test]
fn validate_document_missing_file() {
    let path = std::path::Path::new("/tmp/nonexistent_doc_integrity_test.md");
    let violations = validate_document(path);
    assert!(violations.iter().any(|v| v.rule == "io-error"));
}
