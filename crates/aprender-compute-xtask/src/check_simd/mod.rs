/// SIMD Attribute Checker - Pre-commit validation
///
/// Validates SIMD code properties to ensure correctness and performance:
/// 1. [CRITICAL] Missing #[`target_feature`] attributes
/// 2. [ERROR] Attribute-intrinsic mismatch
/// 3. [WARNING] Missing SAFETY comments
/// 4. [WARNING] Missing #[inline] attributes
///
/// Bug instances found: 104 functions missing #[`target_feature`]
/// Performance impact: 5.9x slower to missing 21x speedup potential
use anyhow::{Context, Result};
use colored::Colorize;
use regex::Regex;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViolationLevel {
    Critical, // Blocks commit, causes severe performance bugs
    Error,    // Blocks commit, causes correctness issues
    Warning,  // Reports but doesn't block
}

#[derive(Debug)]
struct Violation {
    level: ViolationLevel,
    filepath: PathBuf,
    line_num: usize,
    function_name: String,
    message: String,
    fix_suggestion: String,
}

struct IntrinsicPattern {
    pattern: Regex,
    required_feature: &'static str,
    _name: &'static str,
}

impl IntrinsicPattern {
    fn new(pattern: &str, required_feature: &'static str, name: &'static str) -> Self {
        Self {
            pattern: Regex::new(pattern).expect("Invalid regex pattern"),
            required_feature,
            _name: name,
        }
    }
}

/// Get SIMD intrinsic patterns for a backend
fn get_intrinsic_patterns(backend: &str) -> Option<IntrinsicPattern> {
    match backend {
        "sse2" => Some(IntrinsicPattern::new(r"_mm_\w+", "sse2", "SSE2")),
        "avx2" => Some(IntrinsicPattern::new(r"_mm256_\w+", "avx2", "AVX2")),
        "avx512" => Some(IntrinsicPattern::new(r"_mm512_\w+", "avx512f", "AVX512")),
        "neon" => Some(IntrinsicPattern::new(
            r"v(?:ld|st|add|sub|mul|div)\w*q_f32",
            "neon",
            "NEON",
        )),
        _ => None,
    }
}

/// FMA intrinsics that require additional 'fma' feature
fn is_fma_intrinsic(intrinsic: &str) -> bool {
    matches!(
        intrinsic,
        "_mm256_fmadd_ps"
            | "_mm256_fmsub_ps"
            | "_mm256_fnmadd_ps"
            | "_mm256_fnmsub_ps"
            | "_mm_fmadd_ps"
            | "_mm_fmsub_ps"
            | "_mm_fnmadd_ps"
            | "_mm_fnmsub_ps"
    )
}

/// Check if #[`target_feature`] attribute exists within 15 lines before function
fn check_target_feature_attribute(lines: &[String], fn_line: usize) -> Option<String> {
    let target_feature_re =
        Regex::new(r#"#\[target_feature\(enable\s*=\s*"([^"]+)"\)\]"#).expect("Invalid regex");

    let start = fn_line.saturating_sub(15);
    for line in &lines[start..fn_line] {
        if let Some(caps) = target_feature_re.captures(line) {
            return Some(caps[1].to_string());
        }
    }
    None
}

/// Check if SAFETY comment exists within 10 lines before function
fn has_safety_comment(lines: &[String], fn_line: usize) -> bool {
    let safety_re = Regex::new(r"//\s*SAFETY:").expect("Invalid regex");

    let start = fn_line.saturating_sub(10);
    for line in &lines[start..fn_line] {
        if safety_re.is_match(line) {
            return true;
        }
    }
    false
}

/// Check if #[inline] attribute exists within 15 lines before function
fn has_inline_attribute(lines: &[String], fn_line: usize) -> bool {
    let inline_re = Regex::new(r"#\[inline(?:\(always\))?\]").expect("Invalid regex");

    let start = fn_line.saturating_sub(15);
    for line in &lines[start..fn_line] {
        if inline_re.is_match(line) {
            return true;
        }
    }
    false
}

/// Find all SIMD intrinsics used in a function body
fn find_intrinsics_in_function(
    lines: &[String],
    fn_start: usize,
    pattern: &Regex,
) -> (usize, HashSet<String>) {
    let mut intrinsics = HashSet::new();
    let mut brace_count = 0;
    let mut fn_end = fn_start;

    for (offset, line) in lines[fn_start..].iter().enumerate() {
        brace_count += i32::try_from(line.matches('{').count()).unwrap_or(i32::MAX);
        brace_count -= i32::try_from(line.matches('}').count()).unwrap_or(i32::MAX);

        // Collect intrinsics
        for cap in pattern.captures_iter(line) {
            intrinsics.insert(cap[0].to_string());
        }

        // Found function end
        if brace_count == 0 && offset > 0 {
            fn_end = fn_start + offset;
            break;
        }
    }

    (fn_end, intrinsics)
}

/// Check if attribute matches the intrinsics actually used
fn check_attribute_mismatch(feature: &str, intrinsics: &HashSet<String>) -> Option<String> {
    let has_avx512 = intrinsics.iter().any(|i| i.starts_with("_mm512_"));
    let has_avx2 = intrinsics.iter().any(|i| i.starts_with("_mm256_"));
    let _has_sse2 = intrinsics.iter().any(|i| i.starts_with("_mm_"));

    if has_avx512 && !feature.contains("avx512f") {
        return Some(format!(
            "Using AVX-512 intrinsics but attribute is '{}' (should be 'avx512f')",
            feature
        ));
    }

    if has_avx2 && feature == "sse2" {
        return Some(
            "Using AVX2 intrinsics but attribute is 'sse2' (should be 'avx2')".to_string(),
        );
    }

    if !has_avx512 && feature.contains("avx512f") {
        return Some("Attribute is 'avx512f' but no AVX-512 intrinsics found".to_string());
    }

    None
}

/// Check if FMA intrinsics are used without 'fma' feature
fn check_fma_feature(feature: &str, intrinsics: &HashSet<String>) -> Option<String> {
    let uses_fma = intrinsics.iter().any(|i| is_fma_intrinsic(i));

    if uses_fma && !feature.contains("fma") {
        return Some(
            "Using FMA intrinsics (_mm256_fmadd_ps, etc.) but 'fma' feature not enabled"
                .to_string(),
        );
    }

    None
}

/// Check an unsafe function with SIMD intrinsics for all violation types.
///
/// Produces CRITICAL (missing `target_feature`), ERROR (attribute mismatch, missing FMA),
/// and WARNING (missing SAFETY comment, missing inline) violations.
fn check_unsafe_function_violations(
    filepath: &Path,
    lines: &[String],
    fn_name: &str,
    fn_line: usize,
    intrinsics: &HashSet<String>,
    required_feature: &str,
) -> Vec<Violation> {
    let mut violations = Vec::new();

    let target_feature = check_target_feature_attribute(lines, fn_line);

    match target_feature {
        None => {
            // [CRITICAL] Missing #[target_feature]
            violations.push(Violation {
                level: ViolationLevel::Critical,
                filepath: filepath.to_path_buf(),
                line_num: fn_line + 1, // 1-indexed
                function_name: fn_name.to_string(),
                message: format!(
                    "Missing #[target_feature] attribute (uses {} SIMD intrinsics)",
                    intrinsics.len()
                ),
                fix_suggestion: format!(
                    "Add #[target_feature(enable = \"{}\")] above function",
                    required_feature
                ),
            });
        }
        Some(feature) => {
            // [ERROR] Attribute-intrinsic mismatch
            if let Some(msg) = check_attribute_mismatch(&feature, intrinsics) {
                violations.push(Violation {
                    level: ViolationLevel::Error,
                    filepath: filepath.to_path_buf(),
                    line_num: fn_line + 1,
                    function_name: fn_name.to_string(),
                    message: msg,
                    fix_suggestion: "Correct #[target_feature] attribute to match intrinsics used"
                        .to_string(),
                });
            }

            // [ERROR] FMA intrinsics without FMA feature
            if let Some(msg) = check_fma_feature(&feature, intrinsics) {
                violations.push(Violation {
                    level: ViolationLevel::Error,
                    filepath: filepath.to_path_buf(),
                    line_num: fn_line + 1,
                    function_name: fn_name.to_string(),
                    message: msg,
                    fix_suggestion:
                        "Add 'fma' to target_feature: #[target_feature(enable = \"avx2,fma\")]"
                            .to_string(),
                });
            }
        }
    }

    // [WARNING] Missing SAFETY comment
    if !has_safety_comment(lines, fn_line) {
        violations.push(Violation {
            level: ViolationLevel::Warning,
            filepath: filepath.to_path_buf(),
            line_num: fn_line + 1,
            function_name: fn_name.to_string(),
            message: "Missing SAFETY comment for unsafe function with SIMD".to_string(),
            fix_suggestion: "Add // SAFETY: comment explaining why unsafe code is correct"
                .to_string(),
        });
    }

    // [WARNING] Missing #[inline] attribute
    if !has_inline_attribute(lines, fn_line) {
        violations.push(Violation {
            level: ViolationLevel::Warning,
            filepath: filepath.to_path_buf(),
            line_num: fn_line + 1,
            function_name: fn_name.to_string(),
            message: "Missing #[inline] attribute on SIMD hot path".to_string(),
            fix_suggestion: "Add #[inline] above function for better optimization".to_string(),
        });
    }

    violations
}

/// Check a single backend file for SIMD violations
fn check_file(filepath: &Path, backend: &str) -> Result<Vec<Violation>> {
    let Some(pattern) = get_intrinsic_patterns(backend) else {
        return Ok(vec![]);
    };

    let content = std::fs::read_to_string(filepath)
        .with_context(|| format!("Failed to read {}", filepath.display()))?;

    let lines: Vec<String> = content
        .lines()
        .map(std::string::ToString::to_string)
        .collect();
    let unsafe_fn_re = Regex::new(r"^\s*unsafe\s+fn\s+(\w+)").expect("Invalid regex");

    let mut violations = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        if let Some(caps) = unsafe_fn_re.captures(&lines[i]) {
            let fn_name = caps[1].to_string();
            let fn_line = i;
            let (fn_end, intrinsics) =
                find_intrinsics_in_function(&lines, fn_line, &pattern.pattern);

            if !intrinsics.is_empty() {
                violations.extend(check_unsafe_function_violations(
                    filepath,
                    &lines,
                    &fn_name,
                    fn_line,
                    &intrinsics,
                    pattern.required_feature,
                ));
            }

            i = fn_end.max(i + 1);
        } else {
            i += 1;
        }
    }

    Ok(violations)
}

/// Print a group of violations with a header.
fn print_violation_group(violations: &[&Violation], icon: &str, label: &str, color_red: bool) {
    if violations.is_empty() {
        return;
    }
    println!();
    let sep = "=".repeat(60);
    if color_red {
        println!("{}", sep.red());
        println!(
            "{}",
            format!("{icon} {label} ({})", violations.len())
                .red()
                .bold()
        );
        println!("{}", sep.red());
    } else {
        println!("{}", sep.yellow());
        println!(
            "{}",
            format!("{icon} {label} ({})", violations.len())
                .yellow()
                .bold()
        );
        println!("{}", sep.yellow());
    }
    println!();

    for v in violations.iter().take(10) {
        println!(
            "  {}:{} - {}()",
            v.filepath.display(),
            v.line_num,
            v.function_name
        );
        println!("     Problem: {}", v.message);
        println!("     Fix: {}", v.fix_suggestion);
        println!();
    }
    if violations.len() > 10 {
        println!("  {} more...", violations.len() - 10);
        println!();
    }
}

/// Print the summary counts and final pass/block status.
fn print_summary(critical: &[&Violation], errors: &[&Violation], warnings: &[&Violation]) {
    println!();
    println!("{}", "=".repeat(60).blue());
    println!("{}", "SUMMARY".blue().bold());
    println!("{}", "=".repeat(60).blue());
    println!();

    if !critical.is_empty() {
        println!(
            "  {} CRITICAL - Compiler CANNOT emit SIMD instructions",
            critical.len()
        );
    }
    if !errors.is_empty() {
        println!(
            "  {} ERRORS - Incorrect or incompatible attributes",
            errors.len()
        );
    }
    if !warnings.is_empty() {
        println!(
            "  {} WARNINGS - Best practices not followed",
            warnings.len()
        );
    }
    println!();
}

/// Run SIMD attribute checker on all backend files
pub fn run() -> Result<()> {
    println!("{}", "SIMD Property Checker (Rust)".blue().bold());
    println!("{}", "=".repeat(60).blue());
    println!();

    let backend_files = [
        ("src/backends/sse2.rs", "sse2"),
        ("src/backends/avx2.rs", "avx2"),
        ("src/backends/avx512.rs", "avx512"),
        ("src/backends/neon.rs", "neon"),
    ];

    let mut all_violations = Vec::new();
    for (filepath, backend) in &backend_files {
        let path = Path::new(filepath);
        if path.exists() {
            all_violations.extend(check_file(path, backend)?);
        }
    }

    if all_violations.is_empty() {
        println!(
            "{}",
            "PASS: All SIMD property checks passed!".green().bold()
        );
        println!();
        return Ok(());
    }

    let critical: Vec<_> = all_violations
        .iter()
        .filter(|v| v.level == ViolationLevel::Critical)
        .collect();
    let errors: Vec<_> = all_violations
        .iter()
        .filter(|v| v.level == ViolationLevel::Error)
        .collect();
    let warnings: Vec<_> = all_violations
        .iter()
        .filter(|v| v.level == ViolationLevel::Warning)
        .collect();

    print_violation_group(&critical, "CRITICAL", "CRITICAL VIOLATIONS", true);
    print_violation_group(&errors, "ERROR", "ERRORS", true);
    print_violation_group(&warnings, "WARNING", "WARNINGS", false);
    print_summary(&critical, &errors, &warnings);

    if !critical.is_empty() || !errors.is_empty() {
        anyhow::bail!(
            "SIMD validation failed: {} critical, {} errors",
            critical.len(),
            errors.len()
        );
    }

    println!(
        "{}",
        "COMMIT ALLOWED - Only warnings present".green().bold()
    );
    println!();
    Ok(())
}

#[cfg(test)]
mod tests;
