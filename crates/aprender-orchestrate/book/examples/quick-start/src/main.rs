//! Quick Start Example for Batuta
//!
//! This example demonstrates the basic workflow patterns used in Batuta.

use anyhow::Result;

/// Represents a project analysis result
#[derive(Debug)]
struct AnalysisResult {
    languages: Vec<Language>,
    total_files: usize,
    total_lines: usize,
}

#[derive(Debug)]
struct Language {
    name: String,
    files: usize,
    lines: usize,
}

/// Analyze a project's language composition
fn analyze_project() -> Result<AnalysisResult> {
    // In real usage, batuta scans your project directory
    // This is a demonstration of the output format
    Ok(AnalysisResult {
        languages: vec![
            Language {
                name: "Rust".to_string(),
                files: 42,
                lines: 3500,
            },
            Language {
                name: "Python".to_string(),
                files: 15,
                lines: 1200,
            },
        ],
        total_files: 57,
        total_lines: 4700,
    })
}

/// Display analysis results in a formatted table
fn display_results(result: &AnalysisResult) {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  📊 Project Analysis Results");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("  Language       Files    Lines");
    println!("  ──────────────────────────────");

    for lang in &result.languages {
        println!("  {:<12} {:>6} {:>8}", lang.name, lang.files, lang.lines);
    }

    println!("  ──────────────────────────────");
    println!("  {:<12} {:>6} {:>8}", "Total", result.total_files, result.total_lines);
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}

fn main() -> Result<()> {
    println!();
    println!("🦀 Batuta Quick Start Example");
    println!();

    let result = analyze_project()?;
    display_results(&result);

    println!();
    println!("✅ Analysis complete!");
    println!();
    println!("Next steps:");
    println!("  1. Run: batuta analyze --languages");
    println!("  2. Run: batuta analyze --tdg");
    println!("  3. Run: batuta oracle \"your question\"");
    println!();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_project() {
        let result = analyze_project().unwrap();
        assert!(!result.languages.is_empty());
        assert!(result.total_files > 0);
    }
}
