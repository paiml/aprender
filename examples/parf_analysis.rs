//! PARF (Pattern and Reference Finder) Analysis Example (BATUTA-012)
//!
//! Demonstrates cross-codebase pattern analysis and reference finding
//! for understanding code dependencies, usage patterns, and migration planning.
//!
//! Run with: cargo run --example parf_analysis

use batuta::parf::{CodePattern, ParfAnalyzer, SymbolKind};
use std::path::Path;

/// Display a limited list of items with "... and N more" suffix
fn display_limited<T, F>(items: &[T], limit: usize, display_fn: F)
where
    F: Fn(usize, &T),
{
    for (i, item) in items.iter().take(limit).enumerate() {
        display_fn(i + 1, item);
    }
    if items.len() > limit {
        println!("  ... and {} more", items.len() - limit);
    }
}

/// Display patterns by category
fn display_pattern_category<'a, P>(
    name: &str,
    patterns: &'a [CodePattern],
    predicate: P,
    limit: usize,
) where
    P: Fn(&'a CodePattern) -> Option<String>,
{
    let matching: Vec<_> = patterns.iter().filter_map(&predicate).collect();
    println!("{}: {}", name, matching.len());
    if !matching.is_empty() {
        display_limited(&matching, limit, |i, msg| {
            println!("  {}. {}", i, msg);
        });
    }
    println!();
}

fn main() {
    println!("🔍 PARF Analysis Demo (BATUTA-012)");
    println!("===================================\n");

    let mut analyzer = ParfAnalyzer::new();
    let codebase_path = Path::new("src");

    // Index codebase
    println!("📁 Indexing codebase...");
    if let Err(e) = analyzer.index_codebase(codebase_path) {
        eprintln!("   ❌ Indexing failed: {}", e);
        return;
    }
    println!("   ✅ Indexing complete\n");

    // Generate report
    println!("📊 Analysis Report");
    println!("------------------\n");
    println!("{}", analyzer.generate_report());

    // Symbol reference analysis
    display_symbol_references(&analyzer);

    // Pattern detection
    display_patterns(&analyzer);

    // Dependency analysis
    display_dependencies(&analyzer);

    // Dead code analysis
    display_dead_code(&analyzer);

    // Use case summary
    display_use_cases();
}

fn display_symbol_references(analyzer: &ParfAnalyzer) {
    println!("\n🔎 Symbol Reference Analysis");
    println!("----------------------------\n");

    let symbols = [
        ("BackendSelector", SymbolKind::Class),
        ("select_with_moe", SymbolKind::Function),
        ("NumPyConverter", SymbolKind::Class),
    ];

    for (symbol, kind) in symbols {
        let refs = analyzer.find_references(symbol, kind);
        println!("Symbol: {} ({:?})", symbol, kind);
        println!("  References found: {}", refs.len());

        if !refs.is_empty() {
            println!("  Sample references:");
            display_limited(&refs, 3, |i, r| {
                println!("    {}. {}:{}", i, r.file.display(), r.line);
            });
        }
        println!();
    }
}

fn display_patterns(analyzer: &ParfAnalyzer) {
    println!("🎯 Code Pattern Detection");
    println!("-------------------------\n");

    let patterns = analyzer.detect_patterns();

    display_pattern_category(
        "Technical Debt (markers)",
        &patterns,
        |p| {
            if let CodePattern::TechDebt { message, file, line } = p {
                Some(format!("{}:{} - {}", file.display(), line, message))
            } else {
                None
            }
        },
        5,
    );

    display_pattern_category(
        "Error Handling Issues",
        &patterns,
        |p| {
            if let CodePattern::ErrorHandling { pattern, file, line } = p {
                Some(format!("{}:{} - {}", file.display(), line, pattern))
            } else {
                None
            }
        },
        5,
    );

    display_pattern_category(
        "Resource Management",
        &patterns,
        |p| {
            if let CodePattern::ResourceManagement { resource_type, file, line } = p {
                Some(format!("{}:{} - {} resource", file.display(), line, resource_type))
            } else {
                None
            }
        },
        5,
    );

    display_pattern_category(
        "Deprecated APIs",
        &patterns,
        |p| {
            if let CodePattern::DeprecatedApi { api, file, line } = p {
                Some(format!("{}:{} - {}", file.display(), line, api))
            } else {
                None
            }
        },
        5,
    );
}

fn display_dependencies(analyzer: &ParfAnalyzer) {
    println!("📦 Dependency Analysis");
    println!("---------------------\n");

    let dependencies = analyzer.analyze_dependencies();
    println!("Total dependencies: {}", dependencies.len());

    if !dependencies.is_empty() {
        println!("Sample dependencies:");
        display_limited(&dependencies, 10, |i, dep| {
            println!("  {}. {} → {} ({:?})", i, dep.from.display(), dep.to.display(), dep.kind);
        });
    }
    println!();
}

fn display_dead_code(analyzer: &ParfAnalyzer) {
    println!("💀 Dead Code Analysis");
    println!("--------------------\n");

    let dead_code = analyzer.find_dead_code();
    println!("Potentially unused symbols: {}", dead_code.len());

    if !dead_code.is_empty() {
        println!("\nTop candidates for removal:");
        display_limited(&dead_code, 10, |i, dc| {
            println!("  {}. {} ({:?}) in {}:{}", i, dc.symbol, dc.kind, dc.file.display(), dc.line);
            println!("     Reason: {}", dc.reason);
        });
    }
    println!();
}

fn display_use_cases() {
    println!("💡 PARF Use Cases");
    println!("----------------\n");

    let use_cases = [
        (
            "Code Understanding",
            &[
                "Find all usages of a function or type",
                "Understand call graphs and dependencies",
                "Navigate unfamiliar codebases",
            ],
        ),
        (
            "Refactoring",
            &[
                "Identify dead code for safe removal",
                "Find all references before renaming",
                "Detect duplicate patterns for consolidation",
            ],
        ),
        (
            "Migration Planning",
            &[
                "Map dependencies for phased migration",
                "Identify technical debt to address",
                "Find deprecated APIs for replacement",
            ],
        ),
        (
            "Code Quality",
            &[
                "Detect error handling anti-patterns",
                "Find resource leaks (unclosed files)",
                "Track work-in-progress items",
            ],
        ),
    ];

    for (category, items) in use_cases {
        println!("  {}:", category);
        for item in items {
            println!("     • {}", item);
        }
        println!();
    }

    println!("✅ Analysis Complete!");
    println!("   PARF provides actionable insights for:");
    println!("   - Migration planning");
    println!("   - Refactoring safety");
    println!("   - Code quality improvement");
    println!("   - Technical debt reduction");
}
