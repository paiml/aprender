// SHIP-TWO-001 — `document-integrity-v1` algorithm-level PARTIAL
// discharge for FALSIFY-DOC-001..015 (closes 15/15 sweep).
//
// Contract: `contracts/document-integrity-v1.yaml`.
//
// Bundles 15 verdict fns over markdown / YAML / SVG / media
// integrity rules. Each verdict is a self-contained decision rule;
// the runtime impl can be built atop these primitives.

// ===========================================================================
// DOC-001 — Heading hierarchy: no H1→H3 skip
// DOC-002 — Heading hierarchy: at most one H1
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Doc001Verdict { Pass, Fail }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Doc002Verdict { Pass, Fail }

/// Extract heading levels (1..=6) from markdown.
fn extract_heading_levels(md: &str) -> Vec<u8> {
    md.lines()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            let hashes = trimmed.bytes().take_while(|b| *b == b'#').count();
            if hashes == 0 || hashes > 6 { return None; }
            // Must be followed by a space (or end-of-line).
            let after = trimmed.as_bytes().get(hashes).copied();
            if after == Some(b' ') || after.is_none() {
                Some(hashes as u8)
            } else {
                None
            }
        })
        .collect()
}

/// Pass iff no consecutive heading skips a level
/// (e.g., `## A` → `#### B` is Fail; `#` → `##` is Pass).
#[must_use]
pub fn verdict_from_heading_no_skip(md: &str) -> Doc001Verdict {
    let levels = extract_heading_levels(md);
    if levels.is_empty() { return Doc001Verdict::Pass; }
    let mut prev: Option<u8> = None;
    for h in levels {
        if let Some(p) = prev {
            if h > p + 1 { return Doc001Verdict::Fail; }
        }
        prev = Some(h);
    }
    Doc001Verdict::Pass
}

/// Pass iff the document contains at most one H1 line.
#[must_use]
pub fn verdict_from_single_h1(md: &str) -> Doc002Verdict {
    #[allow(clippy::naive_bytecount)]
    let h1_count = extract_heading_levels(md).iter().filter(|l| **l == 1).count();
    if h1_count <= 1 { Doc002Verdict::Pass } else { Doc002Verdict::Fail }
}

// ===========================================================================
// DOC-003 — Link well-formedness: no `javascript:` schemes
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Doc003Verdict { Pass, Fail }

/// Pass iff none of the links use a `javascript:` (case-insensitive) scheme.
#[must_use]
pub fn verdict_from_link_safety(links: &[&str]) -> Doc003Verdict {
    for link in links {
        let lc = link.trim().to_ascii_lowercase();
        if lc.starts_with("javascript:") { return Doc003Verdict::Fail; }
    }
    Doc003Verdict::Pass
}

// ===========================================================================
// DOC-004 — Code fence language: bare ``` is a violation
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Doc004Verdict { Pass, Fail }

/// Pass iff every code fence has a non-empty language tag.
/// We classify fences as "opening" (odd index) and "closing" (even index)
/// in encounter order: only opens require a language.
#[must_use]
pub fn verdict_from_code_fence_language(md: &str) -> Doc004Verdict {
    let mut in_fence = false;
    for line in md.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("```") {
            if !in_fence {
                // Opening fence — must have a non-empty language tag.
                if rest.trim().is_empty() { return Doc004Verdict::Fail; }
                in_fence = true;
            } else {
                // Closing fence.
                in_fence = false;
            }
        }
    }
    Doc004Verdict::Pass
}

// ===========================================================================
// DOC-005 — Table column parity
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Doc005Verdict { Pass, Fail }

fn count_table_cells(line: &str) -> usize {
    let trimmed = line.trim();
    if !trimmed.starts_with('|') || !trimmed.ends_with('|') { return 0; }
    let inner = &trimmed[1..trimmed.len() - 1];
    inner.split('|').count()
}

/// Pass iff every table row has the same column count as its header.
/// Looks at contiguous block of `| ... |` rows.
#[must_use]
pub fn verdict_from_table_column_parity(md: &str) -> Doc005Verdict {
    let mut header_cols: Option<usize> = None;
    for line in md.lines() {
        let cols = count_table_cells(line);
        if cols == 0 {
            // End of table block.
            header_cols = None;
            continue;
        }
        match header_cols {
            None => header_cols = Some(cols),
            Some(h) if cols != h => return Doc005Verdict::Fail,
            Some(_) => {}
        }
    }
    Doc005Verdict::Pass
}

// ===========================================================================
// DOC-006 — SVG safety: no `<script>` tags
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Doc006Verdict { Pass, Fail }

/// Pass iff the SVG body does NOT contain a `<script>` (or
/// `<SCRIPT>`) opening tag.
#[must_use]
pub fn verdict_from_svg_no_script(svg: &str) -> Doc006Verdict {
    let lc = svg.to_ascii_lowercase();
    if lc.contains("<script") { Doc006Verdict::Fail } else { Doc006Verdict::Pass }
}

// ===========================================================================
// DOC-007 — README drift: actual vs generated must match
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Doc007Verdict { Pass, Fail }

/// Pass iff `actual == generated` byte-exactly. Drift is Fail.
#[must_use]
pub fn verdict_from_readme_drift(actual: &str, generated: &str) -> Doc007Verdict {
    if actual == generated { Doc007Verdict::Pass } else { Doc007Verdict::Fail }
}

// ===========================================================================
// DOC-008 — SVG must declare a viewBox
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Doc008Verdict { Pass, Fail }

/// Pass iff the SVG root contains `viewBox=` (case-insensitive).
#[must_use]
pub fn verdict_from_svg_viewbox(svg: &str) -> Doc008Verdict {
    if svg.to_ascii_lowercase().contains("viewbox=") {
        Doc008Verdict::Pass
    } else {
        Doc008Verdict::Fail
    }
}

// ===========================================================================
// DOC-009 — Required sections present
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Doc009Verdict { Pass, Fail }

/// Pass iff every required section name appears as a markdown heading
/// (`## License`, `### License`, etc.) in `md`.
#[must_use]
pub fn verdict_from_required_sections(md: &str, required: &[&str]) -> Doc009Verdict {
    if required.is_empty() { return Doc009Verdict::Fail; }
    for section in required {
        let target = section.to_ascii_lowercase();
        let found = md.lines().any(|line| {
            let t = line.trim();
            t.starts_with('#') && t.to_ascii_lowercase().contains(&target)
        });
        if !found { return Doc009Verdict::Fail; }
    }
    Doc009Verdict::Pass
}

// ===========================================================================
// DOC-010 — YAML structural validity: balanced delimiters
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Doc010Verdict { Pass, Fail }

/// Pass iff `{` `}`, `[` `]` counts are balanced (no unclosed brace).
/// Quick structural sanity check; not a full YAML parser.
#[must_use]
pub fn verdict_from_yaml_balanced_delim(yaml: &str) -> Doc010Verdict {
    let mut stack: Vec<char> = Vec::new();
    for ch in yaml.chars() {
        match ch {
            '{' | '[' => stack.push(ch),
            '}' => {
                if stack.pop() != Some('{') { return Doc010Verdict::Fail; }
            }
            ']' => {
                if stack.pop() != Some('[') { return Doc010Verdict::Fail; }
            }
            _ => {}
        }
    }
    if stack.is_empty() { Doc010Verdict::Pass } else { Doc010Verdict::Fail }
}

// ===========================================================================
// DOC-011 — YAML duplicate top-level keys
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Doc011Verdict { Pass, Fail }

/// Pass iff no top-level YAML key (lines starting at column 0,
/// ending with `:`) appears more than once.
#[must_use]
pub fn verdict_from_yaml_no_duplicate_keys(yaml: &str) -> Doc011Verdict {
    let mut seen: Vec<&str> = Vec::new();
    for line in yaml.lines() {
        if line.is_empty() || line.starts_with(' ') || line.starts_with('\t') || line.starts_with('#') {
            continue;
        }
        if let Some(colon) = line.find(':') {
            let key = &line[..colon];
            if seen.contains(&key) { return Doc011Verdict::Fail; }
            seen.push(key);
        }
    }
    Doc011Verdict::Pass
}

// ===========================================================================
// DOC-012 — Magic-bytes / extension agreement
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Doc012Verdict { Pass, Fail }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatId { Png, Jpeg, Gif, Webp, Unknown }

#[must_use]
pub fn detect_format_by_magic(bytes: &[u8]) -> FormatId {
    if bytes.len() < 4 { return FormatId::Unknown; }
    if bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47]) { return FormatId::Png; }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) { return FormatId::Jpeg; }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") { return FormatId::Gif; }
    if bytes.len() >= 12 && &bytes[8..12] == b"WEBP" { return FormatId::Webp; }
    FormatId::Unknown
}

#[must_use]
#[allow(clippy::case_sensitive_file_extension_comparisons)] // filename pre-lowercased
pub fn format_from_extension(filename: &str) -> FormatId {
    let lower = filename.to_ascii_lowercase();
    if lower.ends_with(".png") { return FormatId::Png; }
    if lower.ends_with(".jpg") || lower.ends_with(".jpeg") { return FormatId::Jpeg; }
    if lower.ends_with(".gif") { return FormatId::Gif; }
    if lower.ends_with(".webp") { return FormatId::Webp; }
    FormatId::Unknown
}

/// Pass iff `format_from_extension(filename) == detect_format_by_magic(bytes)`
/// and both are non-`Unknown`.
#[must_use]
pub fn verdict_from_magic_extension_match(filename: &str, bytes: &[u8]) -> Doc012Verdict {
    let by_ext = format_from_extension(filename);
    let by_magic = detect_format_by_magic(bytes);
    if by_ext == FormatId::Unknown || by_magic == FormatId::Unknown {
        return Doc012Verdict::Fail;
    }
    if by_ext == by_magic { Doc012Verdict::Pass } else { Doc012Verdict::Fail }
}

// ===========================================================================
// DOC-013 — Image dimension bounds
// ===========================================================================

pub const AC_DOC_013_MAX_WIDTH: u32 = 8192;
pub const AC_DOC_013_MAX_HEIGHT: u32 = 8192;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Doc013Verdict { Pass, Fail }

/// Pass iff both dimensions are non-zero AND both ≤ 8192.
#[must_use]
pub const fn verdict_from_image_dimensions(width: u32, height: u32) -> Doc013Verdict {
    if width == 0 || height == 0 { return Doc013Verdict::Fail; }
    if width > AC_DOC_013_MAX_WIDTH || height > AC_DOC_013_MAX_HEIGHT {
        return Doc013Verdict::Fail;
    }
    Doc013Verdict::Pass
}

// ===========================================================================
// DOC-014 — Animation frame-count bounds
// ===========================================================================

pub const AC_DOC_014_MAX_FRAMES: u32 = 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Doc014Verdict { Pass, Fail }

/// Pass iff `frame_count > 0 AND frame_count <= 1000`.
#[must_use]
pub const fn verdict_from_animation_frame_count(frame_count: u32) -> Doc014Verdict {
    if frame_count == 0 { return Doc014Verdict::Fail; }
    if frame_count > AC_DOC_014_MAX_FRAMES { return Doc014Verdict::Fail; }
    Doc014Verdict::Pass
}

// ===========================================================================
// DOC-015 — Animation must not be infinite-loop
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Doc015Verdict { Pass, Fail }

/// Pass iff `loop_count > 0` (per GIF/APNG convention, 0 == infinite).
#[must_use]
pub const fn verdict_from_animation_loop_count(loop_count: u32) -> Doc015Verdict {
    if loop_count == 0 { Doc015Verdict::Fail } else { Doc015Verdict::Pass }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- DOC-001 ----------------------------------------------------------

    #[test] fn doc001_pass_canonical() { assert_eq!(verdict_from_heading_no_skip("# A\n## B\n### C\n"), Doc001Verdict::Pass); }
    #[test] fn doc001_pass_h1_only() { assert_eq!(verdict_from_heading_no_skip("# Just one\n"), Doc001Verdict::Pass); }
    #[test] fn doc001_pass_no_headings() { assert_eq!(verdict_from_heading_no_skip("plain text\n"), Doc001Verdict::Pass); }
    #[test] fn doc001_fail_h1_to_h3_skip() { assert_eq!(verdict_from_heading_no_skip("# Title\n### Skipped H2\n"), Doc001Verdict::Fail); }
    #[test] fn doc001_fail_skip_two_levels() { assert_eq!(verdict_from_heading_no_skip("## A\n#### B\n"), Doc001Verdict::Fail); }

    // ----- DOC-002 ----------------------------------------------------------

    #[test] fn doc002_pass_one_h1() { assert_eq!(verdict_from_single_h1("# Title\n## Sub\n"), Doc002Verdict::Pass); }
    #[test] fn doc002_pass_zero_h1() { assert_eq!(verdict_from_single_h1("## Title\n## More\n"), Doc002Verdict::Pass); }
    #[test] fn doc002_fail_two_h1() { assert_eq!(verdict_from_single_h1("# Title\n# Another Title\n"), Doc002Verdict::Fail); }

    // ----- DOC-003 ----------------------------------------------------------

    #[test] fn doc003_pass_safe_links() {
        let links = ["https://example.com", "http://example.com", "/relative"];
        assert_eq!(verdict_from_link_safety(&links), Doc003Verdict::Pass);
    }

    #[test] fn doc003_fail_javascript() {
        let links = ["javascript:alert(1)"];
        assert_eq!(verdict_from_link_safety(&links), Doc003Verdict::Fail);
    }

    #[test] fn doc003_fail_uppercase_javascript() {
        let links = ["JAVASCRIPT:alert(1)"];
        assert_eq!(verdict_from_link_safety(&links), Doc003Verdict::Fail);
    }

    #[test] fn doc003_pass_empty() {
        assert_eq!(verdict_from_link_safety(&[]), Doc003Verdict::Pass);
    }

    // ----- DOC-004 ----------------------------------------------------------

    #[test] fn doc004_pass_with_lang() {
        let md = "Some text\n```rust\nfn main() {}\n```\n";
        assert_eq!(verdict_from_code_fence_language(md), Doc004Verdict::Pass);
    }

    #[test] fn doc004_fail_bare_fence() {
        let md = "Some text\n```\ncode\n```\n";
        assert_eq!(verdict_from_code_fence_language(md), Doc004Verdict::Fail);
    }

    #[test] fn doc004_pass_no_fences() {
        assert_eq!(verdict_from_code_fence_language("plain text\n"), Doc004Verdict::Pass);
    }

    // ----- DOC-005 ----------------------------------------------------------

    #[test] fn doc005_pass_uniform() {
        let md = "| A | B |\n|---|---|\n| 1 | 2 |\n";
        assert_eq!(verdict_from_table_column_parity(md), Doc005Verdict::Pass);
    }

    #[test] fn doc005_fail_extra_column() {
        let md = "| A | B |\n|---|---|\n| 1 | 2 | 3 |\n";
        assert_eq!(verdict_from_table_column_parity(md), Doc005Verdict::Fail);
    }

    // ----- DOC-006 ----------------------------------------------------------

    #[test] fn doc006_pass_clean_svg() {
        let svg = "<svg><rect/></svg>";
        assert_eq!(verdict_from_svg_no_script(svg), Doc006Verdict::Pass);
    }

    #[test] fn doc006_fail_lowercase_script() {
        let svg = "<svg><script>alert(1)</script></svg>";
        assert_eq!(verdict_from_svg_no_script(svg), Doc006Verdict::Fail);
    }

    #[test] fn doc006_fail_uppercase_script() {
        let svg = "<svg><SCRIPT>alert(1)</SCRIPT></svg>";
        assert_eq!(verdict_from_svg_no_script(svg), Doc006Verdict::Fail);
    }

    // ----- DOC-007 ----------------------------------------------------------

    #[test] fn doc007_pass_match() {
        assert_eq!(verdict_from_readme_drift("# Same\n", "# Same\n"), Doc007Verdict::Pass);
    }

    #[test] fn doc007_fail_drift() {
        assert_eq!(verdict_from_readme_drift("# Old\n", "# New\n"), Doc007Verdict::Fail);
    }

    // ----- DOC-008 ----------------------------------------------------------

    #[test] fn doc008_pass_with_viewbox() {
        let svg = "<svg viewBox='0 0 100 100'></svg>";
        assert_eq!(verdict_from_svg_viewbox(svg), Doc008Verdict::Pass);
    }

    #[test] fn doc008_fail_missing_viewbox() {
        let svg = "<svg xmlns='http://www.w3.org/2000/svg'><rect/></svg>";
        assert_eq!(verdict_from_svg_viewbox(svg), Doc008Verdict::Fail);
    }

    // ----- DOC-009 ----------------------------------------------------------

    #[test] fn doc009_pass_has_license() {
        let md = "# Project\n## License\nMIT\n";
        assert_eq!(verdict_from_required_sections(md, &["License"]), Doc009Verdict::Pass);
    }

    #[test] fn doc009_fail_missing_license() {
        let md = "# Project\n## Usage\nHello\n";
        assert_eq!(verdict_from_required_sections(md, &["License"]), Doc009Verdict::Fail);
    }

    #[test] fn doc009_fail_empty_required_list() {
        // Vacuous truth not accepted.
        let md = "# Project\n";
        assert_eq!(verdict_from_required_sections(md, &[]), Doc009Verdict::Fail);
    }

    // ----- DOC-010 ----------------------------------------------------------

    #[test] fn doc010_pass_balanced() {
        assert_eq!(verdict_from_yaml_balanced_delim("a: [1, 2, 3]\n"), Doc010Verdict::Pass);
    }

    #[test] fn doc010_fail_unclosed_brace() {
        assert_eq!(verdict_from_yaml_balanced_delim("key: {unclosed\n"), Doc010Verdict::Fail);
    }

    #[test] fn doc010_fail_extra_close() {
        assert_eq!(verdict_from_yaml_balanced_delim("a: }\n"), Doc010Verdict::Fail);
    }

    // ----- DOC-011 ----------------------------------------------------------

    #[test] fn doc011_pass_unique_keys() {
        assert_eq!(verdict_from_yaml_no_duplicate_keys("name: foo\nver: 1\n"), Doc011Verdict::Pass);
    }

    #[test] fn doc011_fail_duplicate() {
        assert_eq!(verdict_from_yaml_no_duplicate_keys("name: foo\nname: bar\n"), Doc011Verdict::Fail);
    }

    #[test] fn doc011_pass_nested_same_name() {
        // Indented "name" is at non-top level — not a duplicate.
        let yaml = "outer:\n  name: a\n  name: b\n";
        assert_eq!(verdict_from_yaml_no_duplicate_keys(yaml), Doc011Verdict::Pass);
    }

    // ----- DOC-012 ----------------------------------------------------------

    #[test] fn doc012_pass_png() {
        let bytes = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(verdict_from_magic_extension_match("image.png", &bytes), Doc012Verdict::Pass);
    }

    #[test] fn doc012_fail_png_ext_jpeg_magic() {
        // Classic spoofed extension.
        let bytes = [0xFF, 0xD8, 0xFF, 0xE0];
        assert_eq!(verdict_from_magic_extension_match("image.png", &bytes), Doc012Verdict::Fail);
    }

    #[test] fn doc012_pass_jpg_canonical() {
        let bytes = [0xFF, 0xD8, 0xFF, 0xE0];
        assert_eq!(verdict_from_magic_extension_match("photo.jpg", &bytes), Doc012Verdict::Pass);
    }

    #[test] fn doc012_fail_unknown_magic() {
        let bytes = [0x00, 0x01, 0x02, 0x03];
        assert_eq!(verdict_from_magic_extension_match("image.png", &bytes), Doc012Verdict::Fail);
    }

    // ----- DOC-013 ----------------------------------------------------------

    #[test] fn doc013_pass_normal() {
        assert_eq!(verdict_from_image_dimensions(1920, 1080), Doc013Verdict::Pass);
    }

    #[test] fn doc013_pass_at_max() {
        assert_eq!(verdict_from_image_dimensions(8192, 8192), Doc013Verdict::Pass);
    }

    #[test] fn doc013_fail_oversized_width() {
        assert_eq!(verdict_from_image_dimensions(16384, 1080), Doc013Verdict::Fail);
    }

    #[test] fn doc013_fail_zero_height() {
        assert_eq!(verdict_from_image_dimensions(1920, 0), Doc013Verdict::Fail);
    }

    // ----- DOC-014 ----------------------------------------------------------

    #[test] fn doc014_pass_normal_count() {
        assert_eq!(verdict_from_animation_frame_count(60), Doc014Verdict::Pass);
    }

    #[test] fn doc014_pass_at_max() {
        assert_eq!(verdict_from_animation_frame_count(AC_DOC_014_MAX_FRAMES), Doc014Verdict::Pass);
    }

    #[test] fn doc014_fail_too_many() {
        assert_eq!(verdict_from_animation_frame_count(5000), Doc014Verdict::Fail);
    }

    #[test] fn doc014_fail_zero() {
        assert_eq!(verdict_from_animation_frame_count(0), Doc014Verdict::Fail);
    }

    // ----- DOC-015 ----------------------------------------------------------

    #[test] fn doc015_pass_finite_loop() {
        assert_eq!(verdict_from_animation_loop_count(1), Doc015Verdict::Pass);
        assert_eq!(verdict_from_animation_loop_count(10), Doc015Verdict::Pass);
    }

    #[test] fn doc015_fail_infinite_loop() {
        assert_eq!(verdict_from_animation_loop_count(0), Doc015Verdict::Fail);
    }

    // ----- Provenance pins ---------------------------------------------------

    #[test] fn provenance_dimensions() {
        assert_eq!(AC_DOC_013_MAX_WIDTH, 8192);
        assert_eq!(AC_DOC_013_MAX_HEIGHT, 8192);
    }

    #[test] fn provenance_frame_max() {
        assert_eq!(AC_DOC_014_MAX_FRAMES, 1000);
    }
}
