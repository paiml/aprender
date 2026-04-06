//! Bug Hunter Demo
//!
//! Demonstrates bug-hunter's proactive bug detection capabilities, including
//! the new GPU/CUDA kernel bug detection and test debt patterns.
//!
//! Run with: cargo run --example bug_hunter_demo --features native

use std::path::Path;

fn main() {
    println!("Bug Hunter Demo");
    println!("Proactive Bug Detection with Falsification-Driven Analysis\n");

    println!("{}", "=".repeat(70));
    println!("1. DEFECT CATEGORIES");
    println!("{}\n", "=".repeat(70));

    // Show all defect categories
    let categories = [
        ("TraitBounds", "Missing or incorrect trait bounds"),
        ("AstTransform", "Syntax/structure issues, macro expansion bugs"),
        ("OwnershipBorrow", "Ownership/lifetime errors"),
        ("ConfigurationErrors", "Config/environment issues"),
        ("ConcurrencyBugs", "Race conditions, data races"),
        ("SecurityVulnerabilities", "Security issues"),
        ("TypeErrors", "Type mismatches"),
        ("MemorySafety", "Memory bugs (use-after-free, etc.)"),
        ("LogicErrors", "Logic errors"),
        ("PerformanceIssues", "Performance issues"),
        ("GpuKernelBugs", "GPU/CUDA kernel bugs (PTX, memory access, dimension limits)"),
        ("SilentDegradation", "Silent degradation (fallbacks that hide failures)"),
        ("TestDebt", "Test debt (skipped/ignored tests indicating known bugs)"),
        ("HiddenDebt", "Hidden debt (euphemisms like 'placeholder', 'stub', 'demo')"),
        ("ContractGap", "Contract verification gaps (unbound, partial, missing proofs)"),
        ("ModelParityGap", "Model parity gaps (missing oracles, failed claims, incomplete ops)"),
    ];

    for (name, desc) in categories {
        println!("  {:24} {}", name, desc);
    }

    println!("\n{}", "=".repeat(70));
    println!("2. GPU/CUDA KERNEL BUG PATTERNS (NEW)");
    println!("{}\n", "=".repeat(70));

    // GPU/CUDA patterns
    let gpu_patterns = [
        ("CUDA_ERROR", "Critical", "0.9", "CUDA runtime errors in comments"),
        ("INVALID_PTX", "Critical", "0.95", "Invalid PTX generation issues"),
        ("PTX error", "Critical", "0.9", "PTX compilation errors"),
        ("kernel fail", "High", "0.8", "Kernel execution failures"),
        ("cuBLAS fallback", "High", "0.7", "cuBLAS fallback paths"),
        ("cuDNN fallback", "High", "0.7", "cuDNN fallback paths"),
        ("hidden_dim >=", "High", "0.7", "Dimension-related GPU bugs"),
    ];

    println!("These patterns detect GPU kernel issues documented in comments:");
    println!();
    for (pattern, severity, susp, desc) in gpu_patterns {
        println!("  {:20} [{:8}] ({}) {}", pattern, severity, susp, desc);
    }

    println!("\n{}", "=".repeat(70));
    println!("3. SILENT DEGRADATION PATTERNS (NEW)");
    println!("{}\n", "=".repeat(70));

    let degradation_patterns = [
        (".unwrap_or_else(|_|", "High", "0.7", "Silent error swallowing"),
        ("if let Err(_) =", "Medium", "0.5", "Unchecked error handling"),
        ("Err(_) => {}", "High", "0.75", "Empty error handlers"),
        ("Ok(_) => {}", "Medium", "0.4", "Empty success handlers"),
        ("// fallback", "Medium", "0.5", "Documented fallback paths"),
        ("// degraded", "High", "0.7", "Documented degradation"),
    ];

    println!("These patterns detect silent fallbacks that hide failures:");
    println!();
    for (pattern, severity, susp, desc) in degradation_patterns {
        println!("  {:20} [{:8}] ({}) {}", pattern, severity, susp, desc);
    }

    println!("\n{}", "=".repeat(70));
    println!("4. TEST DEBT PATTERNS (NEW)");
    println!("{}\n", "=".repeat(70));

    let test_debt_patterns = [
        ("#[ignore]", "High", "0.7", "Ignored tests"),
        ("// skip", "Medium", "0.5", "Skipped tests"),
        ("// broken", "High", "0.8", "Known broken tests"),
        ("// fails", "High", "0.75", "Known failing tests"),
        ("// disabled", "Medium", "0.6", "Disabled tests"),
        ("test removed", "Critical", "0.9", "Removed tests"),
        ("were removed", "Critical", "0.9", "Tests removed from codebase"),
        ("tests hang", "Critical", "0.9", "Hanging test documentation"),
        ("hang during", "High", "0.8", "Compilation/runtime hangs"),
    ];

    println!("These patterns detect test debt indicating known bugs:");
    println!();
    for (pattern, severity, susp, desc) in test_debt_patterns {
        println!("  {:20} [{:8}] ({}) {}", pattern, severity, susp, desc);
    }

    println!("\n{}", "=".repeat(70));
    println!("5. HIDDEN DEBT PATTERNS (NEW - PMAT #149)");
    println!("{}\n", "=".repeat(70));

    let hidden_debt_patterns = [
        ("placeholder", "High", "0.75", "Placeholder implementations"),
        ("stub", "High", "0.7", "Stub functions"),
        ("dummy", "High", "0.7", "Dummy values/objects"),
        ("fake", "Medium", "0.6", "Fake implementations"),
        ("mock", "Medium", "0.5", "Mock objects (in prod code)"),
        ("simplified", "Medium", "0.6", "Simplified implementations"),
        ("for demonstration", "High", "0.75", "Demo-only code"),
        ("demo only", "High", "0.8", "Demo-only code"),
        ("not implemented", "Critical", "0.9", "Unimplemented features"),
        ("unimplemented", "Critical", "0.9", "Unimplemented features"),
        ("temporary", "Medium", "0.6", "Temporary solutions"),
        ("hardcoded", "Medium", "0.5", "Hardcoded values"),
        ("workaround", "Medium", "0.6", "Workarounds for issues"),
        ("quick fix", "High", "0.7", "Quick fixes"),
        ("bandaid", "High", "0.7", "Band-aid solutions"),
        ("kludge", "High", "0.75", "Kludge code"),
        ("tech debt", "High", "0.8", "Acknowledged tech debt"),
    ];

    println!("These patterns detect euphemisms that hide technical debt:");
    println!("(Addresses PMAT issue #149 - aprender placeholder bug)\n");
    for (pattern, severity, susp, desc) in hidden_debt_patterns {
        println!("  {:20} [{:8}] ({}) {}", pattern, severity, susp, desc);
    }

    println!("\n{}", "=".repeat(70));
    println!("6. REAL-WORLD EXAMPLE: realizar GPU KERNEL BUGS");
    println!("{}\n", "=".repeat(70));

    println!("Bug-hunter detected these issues in realizar (CUDA inference runtime):\n");

    let example_findings = [
        (
            "tests.rs:7562",
            "INVALID_PTX",
            "Critical",
            "fused_qkv_into test removed - CUDA_ERROR_INVALID_PTX",
        ),
        (
            "tests.rs:9099",
            "INVALID_PTX",
            "Critical",
            "fused_gate_up_into test removed - CUDA_ERROR_INVALID_PTX",
        ),
        (
            "tests.rs:10629",
            "INVALID_PTX",
            "Critical",
            "q8_quantize_async skipped - CUDA_ERROR_INVALID_PTX",
        ),
        (
            "tests.rs:6026",
            "were removed",
            "Critical",
            "COV-013 tests removed - hang during kernel compilation",
        ),
        ("layer.rs:1177", "PTX error", "Critical", "PTX generation error documented"),
        ("graphed.rs:1055", "CUDA_ERROR", "Critical", "CUDA error handling path"),
    ];

    for (location, pattern, severity, desc) in example_findings {
        println!("  [{:8}] {} : {}", severity, location, pattern);
        println!("            {}", desc);
        println!();
    }

    println!("{}", "=".repeat(70));
    println!("7. CLI USAGE");
    println!("{}\n", "=".repeat(70));

    println!("# Run analysis on a project");
    println!("batuta bug-hunter analyze /path/to/project\n");

    println!("# Output as JSON for CI/CD integration");
    println!("batuta bug-hunter analyze . --format json\n");

    println!("# Filter by minimum suspiciousness");
    println!("batuta bug-hunter analyze . --min-suspiciousness 0.7\n");

    println!("# Filter GPU/TestDebt findings");
    println!("batuta bug-hunter analyze . --format json | jq '.findings | map(select(.category == \"GpuKernelBugs\" or .category == \"TestDebt\"))'\n");

    println!("{}", "=".repeat(70));
    println!("8. CACHING & PERFORMANCE");
    println!("{}\n", "=".repeat(70));

    println!("Bug-hunter uses FNV-1a cache keys with mtime invalidation:");
    println!("  - First run: Full analysis (~50s for large projects)");
    println!("  - Cached run: ~30ms (560x speedup)");
    println!("  - Cache invalidates when source files change");
    println!("  - Cache key includes: mode, targets, suspiciousness, pmat quality,");
    println!("    contracts flags, and model parity flags");
    println!();
    println!("Cache location: .pmat/bug-hunter-cache/");

    println!("\n{}", "=".repeat(70));
    println!("9. PARALLEL SCANNING");
    println!("{}\n", "=".repeat(70));

    let num_cpus = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);

    println!("Bug-hunter uses std::thread::scope for parallel file scanning:");
    println!("  - Detected {} CPU cores", num_cpus);
    println!("  - Files are chunked across threads");
    println!("  - Each thread scans patterns independently");
    println!("  - Results are merged with globally unique IDs");

    println!("\n{}", "=".repeat(70));
    println!("10. CONTRACT GAP & MODEL PARITY ANALYSIS (BH-26/BH-27)");
    println!("{}\n", "=".repeat(70));

    println!("BH-26: Contract Verification Gap Analysis");
    println!("  Analyzes provable-contracts binding registries for:");
    println!("  - Bindings with status 'not_implemented' (High, 0.8)");
    println!("  - Bindings with status 'partial' (Medium, 0.6)");
    println!("  - Contract YAMLs with no binding reference (Medium, 0.5)");
    println!("  - Proof obligations with <50% falsification test coverage (Low, 0.4)");
    println!();
    println!("  CLI usage:");
    println!("    batuta bug-hunter analyze . --contracts-auto");
    println!("    batuta bug-hunter analyze . --contracts /path/to/provable-contracts/contracts");
    println!();

    println!("BH-27: Model Parity Gap Analysis");
    println!("  Analyzes tiny-model-ground-truth directory for:");
    println!("  - Missing oracle files (model/prompt combinations) (Medium, 0.6)");
    println!("  - CLAIMS.md FAIL claims (High, 0.8)");
    println!("  - CLAIMS.md Deferred claims (Low, 0.4)");
    println!("  - Missing or empty oracle-ops directories (Low, 0.4)");
    println!();
    println!("  CLI usage:");
    println!("    batuta bug-hunter analyze . --model-parity-auto");
    println!("    batuta bug-hunter analyze . --model-parity /path/to/tiny-model-ground-truth");
    println!();
    println!("  Combined:");
    println!("    batuta bug-hunter analyze . --contracts-auto --model-parity-auto");
    println!();
    println!("  Suspiciousness filtering applies to BH-26/27 findings:");
    println!("    batuta bug-hunter analyze . --contracts-auto --min-suspiciousness 0.7");
    println!("    # Only shows not_implemented (0.8), filters partial (0.6) and low (0.4)");
    println!();
    println!("  Stack-wide analysis with contract/parity flags:");
    println!("    batuta bug-hunter stack --contracts-auto --model-parity-auto");

    println!("\n{}", "=".repeat(70));
    println!("11. NEW FEATURES (2026)");
    println!("{}\n", "=".repeat(70));

    println!("Bug-hunter now includes these advanced features:\n");

    println!("  GIT BLAME INTEGRATION:");
    println!("    - Each finding shows author, commit hash, and date");
    println!("    - Helps identify who introduced issues");
    println!("    - Example: Blame: Noah Gift (b40b402) 2026-02-03\n");

    println!("  DIFF MODE:");
    println!("    - Compare against a baseline to show only new issues");
    println!("    - batuta bug-hunter diff --base main");
    println!("    - batuta bug-hunter diff --since 7d\n");

    println!("  TREND TRACKING:");
    println!("    - Track tech debt over time with snapshots");
    println!("    - batuta bug-hunter trend --weeks 12");
    println!("    - batuta bug-hunter trend --snapshot\n");

    println!("  AUTO-TRIAGE:");
    println!("    - Group findings by root cause (directory + pattern)");
    println!("    - batuta bug-hunter triage\n");

    println!("  COVERAGE WEIGHTING:");
    println!("    - Boost suspiciousness for uncovered code paths");
    println!("    - batuta bug-hunter analyze --coverage lcov.info --coverage-weight 0.7\n");

    println!("  PMAT QUALITY WEIGHTING:");
    println!("    - Weight findings by code quality metrics");
    println!("    - batuta bug-hunter analyze --pmat-quality --quality-weight 0.5\n");

    println!("  ALLOWLIST CONFIG:");
    println!("    - Suppress intentional patterns via .pmat/bug-hunter.toml");
    println!("    - [[allow]]");
    println!("    - file = \"src/optim/*.rs\"");
    println!("    - pattern = \"unimplemented\"");
    println!("    - reason = \"Batch optimizers don't support step()\"\n");

    println!("  MULTI-LANGUAGE SUPPORT:");
    println!("    - Python: eval(), except:, pickle.loads, shell=True");
    println!("    - TypeScript: any, @ts-ignore, innerHTML, it.skip");
    println!("    - Go: _ = err, panic(, exec.Command(, interface{{}}");
    println!("    - Rust: unwrap(), unsafe {{}}, transmute, panic!\n");

    println!("{}", "=".repeat(70));
    println!("DEMO COMPLETE");
    println!("{}", "=".repeat(70));

    // Demonstrate programmatic API if the module is available
    #[cfg(feature = "native")]
    {
        use batuta::bug_hunter::{hunt, HuntConfig, HuntMode};

        println!("\n{}", "=".repeat(70));
        println!("12. PROGRAMMATIC API DEMO");
        println!("{}\n", "=".repeat(70));

        let config =
            HuntConfig { mode: HuntMode::Quick, min_suspiciousness: 0.5, ..Default::default() };

        println!("Running quick scan on current directory...\n");
        let result = hunt(Path::new("."), config);

        println!("Mode: {:?}", result.mode);
        println!("Findings: {}", result.findings.len());
        println!("Duration: {}ms", result.duration_ms);

        if !result.findings.is_empty() {
            println!("\nTop 5 findings:");
            for finding in result.findings.iter().take(5) {
                println!("  [{}] {} : {}", finding.severity, finding.file.display(), finding.title);
            }
        }
    }
}
