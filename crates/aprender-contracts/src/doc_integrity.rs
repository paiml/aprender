//! Document integrity validation for Markdown and SVG files.
//!
//! Enforces structural invariants from the `document-integrity-v1` contract:
//! heading hierarchy, link well-formedness, code fence language tags, GFM
//! table column parity, SVG structural safety, required sections, and
//! README drift detection.

use std::path::Path;

/// A single document validation violation with location and rule info.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocViolation {
    pub line: usize,
    pub rule: &'static str,
    pub message: String,
}

/// Result of comparing an actual README against a generated one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriftResult {
    pub stale: bool,
    pub diff_lines: usize,
}

/// Validate heading hierarchy: first heading is H1, exactly one H1, no level skips.
pub fn validate_heading_hierarchy(md: &str) -> Vec<DocViolation> {
    let mut violations = Vec::new();
    let mut headings: Vec<(usize, usize)> = Vec::new();
    let mut in_fence = false;

    for (idx, line) in md.lines().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        if !trimmed.starts_with('#') {
            continue;
        }
        let hashes = trimmed.bytes().take_while(|&b| b == b'#').count();
        if hashes > 6 {
            continue;
        }
        let rest = &trimmed[hashes..];
        if !rest.is_empty() && !rest.starts_with(' ') {
            continue;
        }
        headings.push((idx + 1, hashes));
    }

    if headings.is_empty() {
        return violations;
    }

    if headings[0].1 != 1 {
        violations.push(DocViolation {
            line: headings[0].0,
            rule: "heading-hierarchy",
            message: format!("first heading must be H1, found H{}", headings[0].1),
        });
    }

    for &(line, level) in &headings[1..] {
        if level == 1 {
            violations.push(DocViolation {
                line,
                rule: "heading-hierarchy",
                message: "duplicate H1 — exactly one H1 allowed per document".into(),
            });
        }
    }

    for i in 1..headings.len() {
        let (_, prev) = headings[i - 1];
        let (line, curr) = headings[i];
        if curr > prev + 1 {
            violations.push(DocViolation {
                line,
                rule: "heading-hierarchy",
                message: format!(
                    "heading level skip: H{curr} follows H{prev} (expected H{} or lower)",
                    prev + 1
                ),
            });
        }
    }

    violations
}

/// Validate `[text](url)` and `![alt](src)` links for empty URLs, `javascript:` scheme, and spaces.
pub fn validate_links(md: &str) -> Vec<DocViolation> {
    let mut violations = Vec::new();
    let mut in_fence = false;

    for (idx, line) in md.lines().enumerate() {
        let trimmed_check = line.trim_start();
        if trimmed_check.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        let line_num = idx + 1;
        let bytes = line.as_bytes();
        let len = bytes.len();
        let mut i = 0;

        while i < len {
            if i + 1 < len && bytes[i] == b']' && bytes[i + 1] == b'(' && bytes[..i].contains(&b'[')
            {
                let url_start = i + 2;
                let mut depth = 1u32;
                let mut url_end = url_start;
                while url_end < len && depth > 0 {
                    match bytes[url_end] {
                        b'(' => depth += 1,
                        b')' => depth -= 1,
                        _ => {}
                    }
                    if depth > 0 {
                        url_end += 1;
                    }
                }
                let url = &line[url_start..url_end];
                if url.is_empty() {
                    violations.push(DocViolation {
                        line: line_num,
                        rule: "link-wellformedness",
                        message: "link URL is empty".into(),
                    });
                } else {
                    if url.starts_with("javascript:") {
                        violations.push(DocViolation {
                            line: line_num,
                            rule: "link-wellformedness",
                            message: format!("link URL uses javascript: scheme (XSS risk): {url}"),
                        });
                    }
                    if url.contains(' ') {
                        violations.push(DocViolation {
                            line: line_num,
                            rule: "link-wellformedness",
                            message: format!("link URL contains unescaped space: {url}"),
                        });
                    }
                }
                i = url_end + 1;
            } else {
                i += 1;
            }
        }
    }
    violations
}

/// Validate code fences — flags bare triple-backtick fences without a language tag.
///
/// Tracks open/close state so only opening fences (not closing fences) are
/// checked for a language tag.
pub fn validate_code_fences(md: &str) -> Vec<DocViolation> {
    let mut violations = Vec::new();
    let mut in_fence = false;

    for (idx, line) in md.lines().enumerate() {
        let trimmed = line.trim();
        if let Some(after_fence) = trimmed.strip_prefix("```") {
            if in_fence {
                // Closing fence — no language tag expected.
                in_fence = false;
            } else {
                // Opening fence — check for language tag.
                in_fence = true;
                if after_fence.trim().is_empty() {
                    violations.push(DocViolation {
                        line: idx + 1,
                        rule: "code-fence-language",
                        message: "code fence without language tag".into(),
                    });
                }
            }
        }
    }
    violations
}

/// Validate GFM tables — checks column count consistency across header, separator, and rows.
pub fn validate_tables(md: &str) -> Vec<DocViolation> {
    let mut violations = Vec::new();
    let lines: Vec<&str> = md.lines().collect();
    let mut i = 0;
    let mut in_fence = false;

    while i < lines.len() {
        let trimmed_check = lines[i].trim_start();
        if trimmed_check.starts_with("```") {
            in_fence = !in_fence;
            i += 1;
            continue;
        }
        if in_fence {
            i += 1;
            continue;
        }
        let line = lines[i].trim();
        if !line.starts_with('|') {
            i += 1;
            continue;
        }
        let header_cols = count_table_columns(line);
        if i + 1 >= lines.len() {
            i += 1;
            continue;
        }
        let sep_line = lines[i + 1].trim();
        if !is_table_separator(sep_line) {
            i += 1;
            continue;
        }

        let sep_cols = count_table_columns(sep_line);
        if sep_cols != header_cols {
            violations.push(DocViolation {
                line: i + 2,
                rule: "table-column-parity",
                message: format!("separator has {sep_cols} columns, header has {header_cols}"),
            });
        }

        let mut j = i + 2;
        while j < lines.len() {
            let row = lines[j].trim();
            if !row.starts_with('|') {
                break;
            }
            let row_cols = count_table_columns(row);
            if row_cols != header_cols {
                violations.push(DocViolation {
                    line: j + 1,
                    rule: "table-column-parity",
                    message: format!("row has {row_cols} columns, header has {header_cols}"),
                });
            }
            j += 1;
        }
        i = j;
    }
    violations
}

fn count_table_columns(row: &str) -> usize {
    let trimmed = row.trim();
    let inner = trimmed.strip_prefix('|').unwrap_or(trimmed);
    let inner = inner.strip_suffix('|').unwrap_or(inner);
    if inner.trim().is_empty() {
        return 0;
    }
    inner.split('|').count()
}

fn is_table_separator(line: &str) -> bool {
    let trimmed = line.trim();
    if !trimmed.contains('|') || !trimmed.contains('-') {
        return false;
    }
    let inner = trimmed.strip_prefix('|').unwrap_or(trimmed);
    let inner = inner.strip_suffix('|').unwrap_or(inner);
    inner.split('|').all(|cell| {
        let c = cell.trim();
        !c.is_empty() && c.chars().all(|ch| ch == '-' || ch == ':')
    })
}

/// Validate SVG structural safety (string-based, no XML parser).
pub fn validate_svg(content: &str) -> Vec<DocViolation> {
    let mut violations = Vec::new();
    let lower = content.to_ascii_lowercase();

    if !lower.contains("<svg") {
        violations.push(DocViolation {
            line: 1,
            rule: "svg-structural-safety",
            message: "missing <svg> root element".into(),
        });
        return violations;
    }
    if !content.contains("viewBox") {
        violations.push(DocViolation {
            line: 1,
            rule: "svg-structural-safety",
            message: "missing viewBox attribute on <svg>".into(),
        });
    }
    for (tag, msg) in [
        ("<script", "SVG contains <script> tag (XSS risk)"),
        ("<foreignobject", "SVG contains <foreignObject> tag"),
    ] {
        if lower.contains(tag) {
            for (idx, line) in content.lines().enumerate() {
                if line.to_ascii_lowercase().contains(tag) {
                    violations.push(DocViolation {
                        line: idx + 1,
                        rule: "svg-structural-safety",
                        message: msg.into(),
                    });
                }
            }
        }
    }
    let has_xmlns = content.contains("xmlns=\"http://www.w3.org/2000/svg\"")
        || content.contains("xmlns='http://www.w3.org/2000/svg'");
    if !has_xmlns {
        violations.push(DocViolation {
            line: 1,
            rule: "svg-structural-safety",
            message: "missing xmlns=\"http://www.w3.org/2000/svg\" namespace".into(),
        });
    }
    violations
}

/// Check that each required section name appears as a heading. Returns missing names.
pub fn validate_required_sections(md: &str, required: &[&str]) -> Vec<String> {
    let mut in_fence = false;
    let headings: Vec<String> = md
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            if trimmed.starts_with("```") {
                in_fence = !in_fence;
                return None;
            }
            if in_fence {
                return None;
            }
            if trimmed.starts_with('#') {
                let text = trimmed.trim_start_matches('#').trim();
                if !text.is_empty() {
                    return Some(text.to_string());
                }
            }
            None
        })
        .collect();

    required
        .iter()
        .filter(|&&s| !headings.iter().any(|h| h.eq_ignore_ascii_case(s)))
        .map(ToString::to_string)
        .collect()
}

/// Detect drift between actual and generated README content.
pub fn detect_readme_drift(actual: &str, generated: &str) -> DriftResult {
    let norm = |s: &str| -> Vec<String> { s.lines().map(|l| l.trim_end().to_string()).collect() };
    let a = norm(actual);
    let g = norm(generated);
    let max_len = a.len().max(g.len());
    let mut diff_count = 0usize;
    for i in 0..max_len {
        if a.get(i).map_or("", String::as_str) != g.get(i).map_or("", String::as_str) {
            diff_count += 1;
        }
    }
    DriftResult {
        stale: diff_count > 0,
        diff_lines: diff_count,
    }
}

/// Dispatch to md or svg validator based on file extension.
pub fn validate_document(path: &Path) -> Vec<DocViolation> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let read = |p: &Path| -> Result<String, Vec<DocViolation>> {
        std::fs::read_to_string(p).map_err(|e| {
            vec![DocViolation {
                line: 0,
                rule: "io-error",
                message: format!("failed to read file: {e}"),
            }]
        })
    };
    match ext {
        "md" | "markdown" => {
            let content = match read(path) {
                Ok(c) => c,
                Err(v) => return v,
            };
            let mut v = validate_heading_hierarchy(&content);
            v.extend(validate_links(&content));
            v.extend(validate_code_fences(&content));
            v.extend(validate_tables(&content));
            v
        }
        "svg" => match read(path) {
            Ok(c) => validate_svg(&c),
            Err(v) => v,
        },
        _ => vec![DocViolation {
            line: 0,
            rule: "unsupported-extension",
            message: format!("unsupported file extension: .{ext}"),
        }],
    }
}

#[cfg(test)]
#[path = "doc_integrity_tests.rs"]
mod tests;
