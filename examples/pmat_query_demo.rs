#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    dead_code,
    clippy::needless_borrows_for_generic_args,
    clippy::match_single_binding,
    clippy::single_char_pattern
)]
//! PMAT Query + Oracle Integration Demo
//!
//! Demonstrates function-level quality-annotated code search via `pmat query`,
//! integrated into the Oracle CLI as `--pmat-query`. Shows how to parse PMAT
//! JSON output, apply quality filters, and combine with RAG document retrieval.
//!
//! Run with: cargo run --example pmat_query_demo --features native

fn main() {
    println!("PMAT Query + Oracle Integration Demo");
    println!("Function-Level Quality-Annotated Code Search\n");

    println!("{}", "=".repeat(60));
    println!("1. PARSING PMAT QUERY JSON OUTPUT");
    println!("{}\n", "=".repeat(60));

    // Simulate pmat query JSON output
    let pmat_json = r#"[
        {
            "file_path": "src/pipeline.rs",
            "function_name": "validate_stage",
            "signature": "fn validate_stage(&self, stage: &Stage) -> Result<()>",
            "doc_comment": "Validates a pipeline stage using Jidoka stop-on-error.",
            "start_line": 142,
            "end_line": 185,
            "language": "Rust",
            "tdg_score": 92.5,
            "tdg_grade": "A",
            "complexity": 4,
            "big_o": "O(n)",
            "satd_count": 0,
            "loc": 43,
            "relevance_score": 0.87,
            "source": null
        },
        {
            "file_path": "src/backend.rs",
            "function_name": "select_backend",
            "signature": "fn select_backend(&self, workload: &Workload) -> Backend",
            "doc_comment": "Cost-based backend selection using 5x PCIe rule.",
            "start_line": 88,
            "end_line": 145,
            "language": "Rust",
            "tdg_score": 78.3,
            "tdg_grade": "B",
            "complexity": 8,
            "big_o": "O(n log n)",
            "satd_count": 1,
            "loc": 57,
            "relevance_score": 0.72,
            "source": null
        },
        {
            "file_path": "src/serve/failover.rs",
            "function_name": "handle_timeout",
            "signature": "fn handle_timeout(&mut self, endpoint: &str) -> Result<Response>",
            "doc_comment": null,
            "start_line": 201,
            "end_line": 230,
            "language": "Rust",
            "tdg_score": 55.0,
            "tdg_grade": "C",
            "complexity": 12,
            "big_o": "O(n)",
            "satd_count": 2,
            "loc": 29,
            "relevance_score": 0.65,
            "source": null
        }
    ]"#;

    let results: Vec<serde_json::Value> = serde_json::from_str(pmat_json).unwrap();
    println!("Parsed {} function results from pmat query JSON\n", results.len());

    for (i, r) in results.iter().enumerate() {
        let grade = r["tdg_grade"].as_str().unwrap_or("?");
        let grade_color = match grade {
            "A" => "\x1b[92m",
            "B" => "\x1b[32m",
            "C" => "\x1b[93m",
            "D" => "\x1b[33m",
            "F" => "\x1b[91m",
            _ => "\x1b[0m",
        };

        let tdg_score = r["tdg_score"].as_f64().unwrap_or(0.0);
        let bar_filled = (tdg_score / 10.0).round() as usize;
        let bar_empty = 10_usize.saturating_sub(bar_filled);
        let bar = format!("{}{}", "\u{2588}".repeat(bar_filled), "\u{2591}".repeat(bar_empty));

        println!(
            "{}. {}[{}]\x1b[0m {}:{} {} {} {:.1}",
            i + 1,
            grade_color,
            grade,
            r["file_path"].as_str().unwrap_or("?"),
            r["start_line"],
            r["function_name"].as_str().unwrap_or("?"),
            bar,
            tdg_score
        );
        println!(
            "   {} Complexity: {} | Big-O: {} | SATD: {}",
            r["signature"].as_str().unwrap_or(""),
            r["complexity"],
            r["big_o"].as_str().unwrap_or("?"),
            r["satd_count"]
        );
        if let Some(doc) = r["doc_comment"].as_str() {
            println!("   {}", doc);
        }
        println!();
    }

    println!("{}", "=".repeat(60));
    println!("2. QUALITY FILTERING");
    println!("{}\n", "=".repeat(60));

    // Filter to grade A only
    let grade_a: Vec<&serde_json::Value> =
        results.iter().filter(|r| r["tdg_grade"].as_str() == Some("A")).collect();
    println!("Grade A functions: {}/{}\n", grade_a.len(), results.len());

    // Filter by complexity
    let low_complexity: Vec<&serde_json::Value> =
        results.iter().filter(|r| r["complexity"].as_u64().unwrap_or(0) <= 5).collect();
    println!("Low complexity (<=5): {}/{}\n", low_complexity.len(), results.len());

    // Filter by SATD (Self-Admitted Technical Debt)
    let no_debt: Vec<&serde_json::Value> =
        results.iter().filter(|r| r["satd_count"].as_u64().unwrap_or(0) == 0).collect();
    println!("No technical debt (SATD=0): {}/{}\n", no_debt.len(), results.len());

    println!("{}", "=".repeat(60));
    println!("3. OUTPUT FORMATS");
    println!("{}\n", "=".repeat(60));

    // JSON envelope
    let json_envelope = serde_json::json!({
        "query": "error handling",
        "source": "pmat",
        "result_count": results.len(),
        "results": results,
    });
    println!(
        "JSON envelope keys: {:?}\n",
        json_envelope.as_object().unwrap().keys().collect::<Vec<_>>()
    );

    // Markdown table
    println!("Markdown table output:\n");
    println!("| # | Grade | File | Function | TDG | Complexity | Big-O |");
    println!("|---|-------|------|----------|-----|------------|-------|");
    for (i, r) in results.iter().enumerate() {
        println!(
            "| {} | {} | {}:{} | `{}` | {:.1} | {} | {} |",
            i + 1,
            r["tdg_grade"].as_str().unwrap_or("?"),
            r["file_path"].as_str().unwrap_or("?"),
            r["start_line"],
            r["function_name"].as_str().unwrap_or("?"),
            r["tdg_score"].as_f64().unwrap_or(0.0),
            r["complexity"],
            r["big_o"].as_str().unwrap_or("?"),
        );
    }

    println!();
    println!("{}", "=".repeat(60));
    println!("4. HYBRID SEARCH (PMAT + RAG)");
    println!("{}\n", "=".repeat(60));

    println!("Combined search uses RRF (Reciprocal Rank Fusion, k=60):\n");
    println!("  +-----------+     +----------+");
    println!("  | pmat      |     | RAG      |");
    println!("  | query     |     | index    |");
    println!("  | (funcs)   |     | (docs)   |");
    println!("  +-----+-----+     +----+-----+");
    println!("        |                |");
    println!("        v                v");
    println!("  +-----------------------------+");
    println!("  |   RRF Fusion (k=60)         |");
    println!("  | [fn] + [doc] interleaved    |");
    println!("  +-----------------------------+");

    println!("\n\nCLI examples:");
    println!("  batuta oracle --pmat-query \"error handling\"");
    println!("  batuta oracle --pmat-query \"serialize\" --pmat-min-grade A");
    println!("  batuta oracle --pmat-query \"cache\" --pmat-max-complexity 10");
    println!("  batuta oracle --pmat-query \"allocator\" --pmat-include-source");
    println!("  batuta oracle --pmat-query \"error\" --rag  # RRF-fused view");
    println!("  batuta oracle --pmat-query \"error\" --format json");
    println!("  batuta oracle --pmat-query \"error\" --format markdown");
    println!("  batuta oracle --pmat-query \"tokenizer\" --pmat-all-local  # cross-project");

    println!();
    println!("{}", "=".repeat(60));
    println!("5. QUALITY SIGNALS EXPLAINED");
    println!("{}\n", "=".repeat(60));

    println!("  Signal       | Description                          | Source");
    println!("  -------------|--------------------------------------|--------");
    println!("  TDG Score    | Technical Debt Grade (0-100)          | pmat");
    println!("  TDG Grade    | Letter grade (A-F)                   | pmat");
    println!("  Complexity   | McCabe cyclomatic complexity          | pmat");
    println!("  Big-O        | Asymptotic complexity                | pmat");
    println!("  SATD Count   | Self-Admitted Technical Debt markers  | pmat");
    println!("  LOC          | Lines of code                        | pmat");
    println!("  Relevance    | Query relevance score (0-1)          | pmat");

    println!("\n\nGrade thresholds:");
    println!("  \x1b[92mA\x1b[0m: TDG >= 80  (excellent quality)");
    println!("  \x1b[32mB\x1b[0m: TDG >= 60  (good quality)");
    println!("  \x1b[93mC\x1b[0m: TDG >= 40  (acceptable)");
    println!("  \x1b[33mD\x1b[0m: TDG >= 20  (needs improvement)");
    println!("  \x1b[91mF\x1b[0m: TDG <  20  (critical issues)");

    println!();
    println!("{}", "=".repeat(60));
    println!("6. VERSION 2.0 ENHANCEMENTS");
    println!("{}\n", "=".repeat(60));

    println!("  Enhancement                | Description");
    println!("  ---------------------------|------------------------------------------");
    println!("  RRF-Fused Ranking          | Interleave [fn]+[doc] via RRF (k=60)");
    println!("  Cross-Project Search       | --pmat-all-local searches ~/src projects");
    println!("  Result Caching             | FNV hash key, mtime invalidation");
    println!("  Quality Summary            | Grade dist, avg complexity, total SATD");
    println!("  RAG Backlinks              | See-also links from code to documentation");

    // Quality summary demo
    println!("\n\nQuality summary example:");
    let grades: std::collections::HashMap<&str, usize> =
        [("A", 3), ("B", 2), ("C", 1)].iter().copied().collect();
    let parts: Vec<String> =
        grades.iter().filter(|(_, &v)| v > 0).map(|(k, v)| format!("{}{}", v, k)).collect();
    println!(
        "  Summary: {} | Avg complexity: 5.2 | Total SATD: 2 | Complexity: 1-12",
        parts.join(" ")
    );

    println!();
    println!("{}", "=".repeat(60));
    println!("7. GIT HISTORY SEARCH (-G / --git-history)");
    println!("{}\n", "=".repeat(60));

    println!("RRF-fused git history combines code search with commit history.\n");
    println!("CLI: pmat query \"error handling\" -G --limit 3\n");

    // Simulate git history output
    let commits = [
        (
            "6a99f95",
            "fix",
            "fix(safety): replace critical unwrap() calls with proper error handling",
            0.724,
            "Noah Gift",
            "2026-01-30",
            &[("src/cli/stack.rs", "B", 24, 3), ("src/experiment/tree.rs", "A", 8, 0)][..],
        ),
        (
            "8748f08",
            "fix",
            "fix(examples): Replace unwrap() with proper error handling",
            0.672,
            "Noah Gift",
            "2025-12-07",
            &[
                ("examples/mcp_demo.rs", "B", 2, 0),
                ("examples/stack_diagnostics_demo.rs", "A", 2, 0),
            ][..],
        ),
        (
            "604587a",
            "fix",
            "fix(examples): Add proper error handling to content_demo",
            0.664,
            "Noah Gift",
            "2025-12-07",
            &[("examples/content_demo.rs", "A", 1, 0)][..],
        ),
    ];

    for (i, (hash, tag, msg, score, author, date, files)) in commits.iter().enumerate() {
        let tag_color = match *tag {
            "feat" => "\x1b[32m",
            "fix" => "\x1b[31m",
            "test" => "\x1b[33m",
            _ => "\x1b[0m",
        };
        println!(
            "  {}. \x1b[33m{}\x1b[0m {}[{}]\x1b[0m \x1b[1;37m{}\x1b[0m \x1b[1;32m({:.3})\x1b[0m",
            i + 1,
            hash,
            tag_color,
            tag,
            msg,
            score
        );
        println!("     \x1b[36m{}\x1b[0m \x1b[2m{}\x1b[0m", author, date);
        let file_strs: Vec<String> = files
            .iter()
            .map(|(path, grade, faults, fixes)| {
                let mut s = format!("\x1b[2;36m{}\x1b[0m \x1b[32m[{}]\x1b[0m", path, grade);
                if *faults > 0 {
                    s.push_str(&format!(" \x1b[35mfaults:{}\x1b[0m", faults));
                }
                if *fixes > 0 {
                    s.push_str(&format!(" \x1b[31m({} fixes)\x1b[0m", fixes));
                }
                s
            })
            .collect();
        println!("     {}\n", file_strs.join(", "));
    }

    println!("{}", "=".repeat(60));
    println!("8. HOTSPOTS");
    println!("{}\n", "=".repeat(60));

    println!("Top changed files across all commits with fix counts and author ownership:\n");

    let hotspots = [
        ("Cargo.toml", 61, 14.2, 4, "Noah Gift", 97),
        ("src/main.rs", 60, 13.9, 5, "Noah Gift", 90),
        ("Cargo.lock", 58, 13.5, 3, "Noah Gift", 97),
        ("src/cli/oracle.rs", 37, 8.6, 5, "Noah Gift", 100),
        ("README.md", 30, 7.0, 0, "Noah Gift", 93),
    ];

    for (path, commits, pct, fixes, author, ownership) in &hotspots {
        let fix_str =
            if *fixes > 0 { format!(" \x1b[31m{} fixes\x1b[0m", fixes) } else { String::new() };
        println!(
            "  \x1b[2;36m{:<45}\x1b[0m \x1b[33m{:>3} commits ({:>4.1}%)\x1b[0m{} \x1b[36m{}:{}%\x1b[0m",
            path, commits, pct, fix_str, author, ownership
        );
    }

    println!();
    println!("{}", "=".repeat(60));
    println!("9. DEFECT INTRODUCTION");
    println!("{}\n", "=".repeat(60));

    println!("Feature commits that needed fixes within 30 days:\n");

    let defects = [
        ("5a3798f", &["Cargo.lock", "Cargo.toml"][..], 9),
        ("6763cf2", &["src/cli/oracle.rs", "src/main.rs"][..], 8),
        ("4c4e962", &["book/src/part3/oracle-mode.md", "examples/rag_oracle_demo.rs"][..], 8),
    ];

    for (hash, files, fixes) in &defects {
        let file_list =
            files.iter().map(|f| format!("\x1b[2;36m{}\x1b[0m", f)).collect::<Vec<_>>().join(", ");
        println!(
            "  \x1b[33m{}\x1b[0m {} \x1b[31m{} fixes within 30d\x1b[0m",
            hash, file_list, fixes
        );
    }

    println!();
    println!("{}", "=".repeat(60));
    println!("10. CHURN VELOCITY");
    println!("{}\n", "=".repeat(60));

    println!("Commits per week over 16-week window (red = unstable):\n");

    let churn = [
        ("Cargo.toml", 3.9),
        ("src/main.rs", 3.9),
        ("Cargo.lock", 3.7),
        ("src/cli/oracle.rs", 2.4),
        ("README.md", 1.9),
    ];

    for (path, rate) in &churn {
        let color = if *rate >= 3.0 {
            "\x1b[1;31m"
        } else if *rate >= 2.0 {
            "\x1b[33m"
        } else {
            "\x1b[2m"
        };
        println!("  \x1b[2;36m{:<45}\x1b[0m {}{:.1}/wk\x1b[0m", path, color, rate);
    }

    println!();
    println!("{}", "=".repeat(60));
    println!("11. CO-CHANGE COUPLING");
    println!("{}\n", "=".repeat(60));

    println!("Files that always change together (Jaccard similarity):\n");

    let coupling = [
        ("Cargo.lock", "Cargo.toml", 50, 0.72),
        ("Cargo.toml", "src/main.rs", 17, 0.16),
        ("Cargo.lock", "src/main.rs", 15, 0.15),
        ("src/lib.rs", "src/main.rs", 13, 0.18),
    ];

    for (a, b, co_changes, jaccard) in &coupling {
        let color = if *jaccard >= 0.5 { "\x1b[1;31m" } else { "\x1b[2m" };
        println!(
            "  \x1b[2;36m{}\x1b[0m <-> \x1b[2;36m{}\x1b[0m {}({} co-changes, J={:.2})\x1b[0m",
            a, b, color, co_changes, jaccard
        );
    }

    println!();
    println!("{}", "=".repeat(60));
    println!("12. ENRICHMENT FLAGS REFERENCE");
    println!("{}\n", "=".repeat(60));

    println!("  Flag           | Description                              | Source");
    println!("  ---------------|------------------------------------------|--------");
    println!("  -G             | Git history RRF fusion (commits+code)    | git log");
    println!("  --churn        | Git volatility (90-day commit count)     | git log");
    println!("  --duplicates   | Code clone detection (MinHash+LSH)       | AST");
    println!("  --entropy      | Pattern diversity (repetitive vs unique) | AST");
    println!("  --faults       | Fault annotations (unwrap, panic, etc.)  | AST");

    println!("\n\nCombined enrichment (full audit):");
    println!("  pmat query \"error handling\" --churn --duplicates --entropy --faults -G");

    println!("\nDone.");
}
