//! Panic Path Detection Demo
//!
//! Demonstrates the panic path linter (PROBAR-WASM-006) that detects
//! panic-inducing patterns in WASM code:
//! - `unwrap()` and `expect()` calls
//! - `panic!()`, `todo!()`, `unimplemented!()` macros
//! - `unreachable!()` macro (warning)
//! - Direct indexing without bounds checking (warning)
//!
//! Run with: cargo run --example panic_paths_demo -p jugar-probar

use jugar_probar::lint::{lint_panic_paths, LintSeverity, PanicPathSummary};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Panic Path Detection Demo (PROBAR-WASM-006)            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    demo_clean_code();
    demo_panic_methods();
    demo_panic_macros();
    demo_index_operations();
    demo_test_module_skip();
    demo_summary();
}

/// Demonstrate clean code that passes lint
fn demo_clean_code() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  1. Clean Code (No Panic Paths)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let clean_code = r#"
fn safe_divide(a: i32, b: i32) -> Option<i32> {
    if b == 0 {
        None
    } else {
        Some(a / b)
    }
}

fn get_first(items: &[i32]) -> Option<&i32> {
    items.get(0)  // Safe: uses .get() instead of indexing
}

fn parse_config(s: &str) -> Result<Config, ParseError> {
    let value = s.parse::<i32>().map_err(|_| ParseError::InvalidFormat)?;
    Ok(Config { value })
}

struct Config { value: i32 }
struct ParseError { }
impl ParseError { const InvalidFormat: Self = Self {}; }
"#;

    let report = lint_panic_paths(clean_code, "clean.rs").expect("parse failed");
    println!("  Source: clean.rs");
    println!("  Lines analyzed: {}", report.lines_analyzed);
    println!("  Errors: {}", report.error_count());
    println!("  Warnings: {}", report.warning_count());

    if report.errors.is_empty() {
        println!("\n  ✓ No panic paths detected - WASM safe!\n");
    }
}

/// Demonstrate detection of unwrap() and expect()
fn demo_panic_methods() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  2. Panic Methods Detection (unwrap/expect)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let bad_code = r#"
fn dangerous_code() {
    let config = std::env::var("CONFIG").unwrap();  // WASM-PANIC-001
    let port: u16 = config.parse().expect("invalid port");  // WASM-PANIC-002
    let file = std::fs::read_to_string("data.txt").unwrap();  // WASM-PANIC-001
}
"#;

    let report = lint_panic_paths(bad_code, "dangerous.rs").expect("parse failed");
    println!("  Source: dangerous.rs");
    println!("  Lines analyzed: {}", report.lines_analyzed);
    println!();

    for error in &report.errors {
        let icon = match error.severity {
            LintSeverity::Error => "✗",
            LintSeverity::Warning => "⚠",
            LintSeverity::Info => "ℹ",
        };
        println!(
            "  {} [{}] Line {}: {}",
            icon, error.rule, error.line, error.message
        );
        if let Some(ref suggestion) = error.suggestion {
            println!("    └─ Suggestion: {}", suggestion);
        }
    }
    println!();
}

/// Demonstrate detection of panic macros
fn demo_panic_macros() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  3. Panic Macro Detection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let macro_code = r#"
fn process(state: State) {
    match state {
        State::Valid => handle_valid(),
        State::Invalid => panic!("invalid state"),  // WASM-PANIC-003
        State::Unknown => unreachable!(),  // WASM-PANIC-004 (warning)
    }
}

fn future_feature() {
    todo!("implement OAuth2 support");  // WASM-PANIC-005
}

fn legacy_api() {
    unimplemented!();  // WASM-PANIC-006
}

enum State { Valid, Invalid, Unknown }
fn handle_valid() {}
"#;

    let report = lint_panic_paths(macro_code, "macros.rs").expect("parse failed");
    println!("  Source: macros.rs");
    println!();

    println!("  Detected panic macros:");
    for error in &report.errors {
        let severity = match error.severity {
            LintSeverity::Error => "ERROR  ",
            LintSeverity::Warning => "WARNING",
            LintSeverity::Info => "INFO   ",
        };
        println!("    {} │ {} │ {}", error.rule, severity, error.message);
    }
    println!();
}

/// Demonstrate detection of index operations
fn demo_index_operations() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  4. Index Operation Detection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let index_code = r#"
fn get_item(items: &[i32], idx: usize) -> i32 {
    items[idx]  // WASM-PANIC-007: can panic on out-of-bounds
}

fn get_char(s: &str, idx: usize) -> char {
    s.chars().nth(idx).unwrap()  // Double problem: index + unwrap
}
"#;

    let report = lint_panic_paths(index_code, "indexing.rs").expect("parse failed");
    println!("  Source: indexing.rs");
    println!();

    let summary = PanicPathSummary::from_report(&report);
    println!("  Summary:");
    println!("    ├─ Index operations: {}", summary.index_count);
    println!("    ├─ unwrap() calls: {}", summary.unwrap_count);
    println!(
        "    └─ Total warnings: {}",
        summary.total() - summary.error_count()
    );
    println!();

    println!("  Safe alternatives:");
    println!("    ├─ items.get(idx) -> Option<&T>");
    println!("    ├─ items.get(idx).copied() -> Option<T>");
    println!("    └─ items.get(idx).ok_or(Error)? -> T");
    println!();
}

/// Demonstrate that test modules are skipped
fn demo_test_module_skip() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  5. Test Module Handling");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let test_code = r#"
fn production_code() -> Option<i32> {
    Some(42)  // Clean
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_example() {
        let x = Some(5);
        let y = x.unwrap();  // OK in tests - skipped!
        assert_eq!(y, 5);
    }
}
"#;

    let report = lint_panic_paths(test_code, "with_tests.rs").expect("parse failed");
    println!("  Source: with_tests.rs (contains #[cfg(test)] module)");
    println!("  Errors found: {}", report.error_count());
    println!();

    if report.errors.is_empty() {
        println!("  ✓ Test modules are automatically skipped");
        println!("    (unwrap/expect allowed in tests for convenience)");
    }
    println!();
}

/// Demonstrate summary statistics
fn demo_summary() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  6. Summary Statistics");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mixed_code = r#"
fn example() {
    let a = Some(1).unwrap();
    let b = Some(2).unwrap();
    let c = Some(3).expect("c");
    panic!("oops");
    todo!();
    unreachable!();
    let arr = [1, 2, 3];
    let _ = arr[0];
    let _ = arr[1];
}
"#;

    let report = lint_panic_paths(mixed_code, "mixed.rs").expect("parse failed");
    let summary = PanicPathSummary::from_report(&report);

    println!("  Panic Path Summary for mixed.rs:");
    println!("  ┌─────────────────────────────────┐");
    println!("  │ Category          │ Count       │");
    println!("  ├─────────────────────────────────┤");
    println!("  │ unwrap()          │ {:>11} │", summary.unwrap_count);
    println!("  │ expect()          │ {:>11} │", summary.expect_count);
    println!("  │ panic!()          │ {:>11} │", summary.panic_count);
    println!("  │ todo!()           │ {:>11} │", summary.todo_count);
    println!(
        "  │ unimplemented!()  │ {:>11} │",
        summary.unimplemented_count
    );
    println!(
        "  │ unreachable!()    │ {:>11} │",
        summary.unreachable_count
    );
    println!("  │ Index operations  │ {:>11} │", summary.index_count);
    println!("  ├─────────────────────────────────┤");
    println!("  │ Total (errors)    │ {:>11} │", summary.error_count());
    println!("  │ Total (all)       │ {:>11} │", summary.total());
    println!("  └─────────────────────────────────┘");
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Demo complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}
