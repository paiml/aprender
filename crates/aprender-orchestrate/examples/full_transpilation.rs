//! Full Transpilation Pipeline Example
//!
//! Demonstrates end-to-end transpilation using external tools:
//! - Depyler for Python → Rust
//! - Bashrs for Shell → Rust
//! - Decy for C/C++ → Rust (if available)
//! - PMAT for quality analysis
//!
//! Run with: cargo run --example full_transpilation

use batuta::tools::ToolRegistry;

fn main() {
    println!("🔧 Batuta Full Transpilation Pipeline Demo");
    println!("==========================================\n");

    // Detect all available tools
    println!("📋 Detecting available tools...");
    let registry = ToolRegistry::detect();

    let available = registry.available_tools();
    if available.is_empty() {
        println!("   ⚠️  No transpilation tools detected!");
        println!("\n   Install tools:");
        println!("   - Depyler: cargo install depyler");
        println!("   - Bashrs:  cargo install bashrs");
        println!("   - Decy:    cargo install decy");
        println!("   - PMAT:    cargo install pmat");
        return;
    }

    println!("   ✅ Found {} tools:", available.len());
    for tool in &available {
        println!("      • {}", tool);
    }
    println!();

    // Show tool versions
    println!("📊 Tool Versions:");
    println!("─────────────────");

    if let Some(depyler) = &registry.depyler {
        println!(
            "  Depyler: {} ({})",
            depyler.version.as_ref().unwrap_or(&"unknown".to_string()),
            depyler.path
        );
    }

    if let Some(bashrs) = &registry.bashrs {
        println!(
            "  Bashrs:  {} ({})",
            bashrs.version.as_ref().unwrap_or(&"unknown".to_string()),
            bashrs.path
        );
    }

    if let Some(decy) = &registry.decy {
        println!(
            "  Decy:    {} ({})",
            decy.version.as_ref().unwrap_or(&"unknown".to_string()),
            decy.path
        );
    }

    if let Some(pmat) = &registry.pmat {
        println!(
            "  PMAT:    {} ({})",
            pmat.version.as_ref().unwrap_or(&"unknown".to_string()),
            pmat.path
        );
    }

    if let Some(ruchy) = &registry.ruchy {
        println!(
            "  Ruchy:   {} ({})",
            ruchy.version.as_ref().unwrap_or(&"unknown".to_string()),
            ruchy.path
        );
    }

    println!();

    // Demonstrate transpiler capabilities
    println!("🎯 Transpilation Capabilities:");
    println!("───────────────────────────────");

    if registry.depyler.is_some() {
        println!("  ✅ Python → Rust (Depyler)");
        println!("     • NumPy → Trueno tensor operations");
        println!("     • sklearn → Aprender ML algorithms");
        println!("     • PyTorch → Realizar inference");
        println!("     • Full project structure generation");
        println!("     • Type inference and safety guarantees");
    } else {
        println!("  ❌ Python → Rust (install depyler)");
    }
    println!();

    if registry.bashrs.is_some() {
        println!("  ✅ Shell → Rust (Bashrs)");
        println!("     • POSIX shell script transpilation");
        println!("     • Formal verification support");
        println!("     • Deterministic execution guarantees");
        println!("     • Standalone binary compilation");
        println!("     • ShellCheck integration");
    } else {
        println!("  ❌ Shell → Rust (install bashrs)");
    }
    println!();

    if registry.decy.is_some() {
        println!("  ✅ C/C++ → Rust (Decy)");
        println!("     • Unsafe code generation with safety comments");
        println!("     • Memory safety analysis");
        println!("     • StaticFixer integration");
        println!("     • Incremental transpilation support");
    } else {
        println!("  ❌ C/C++ → Rust (install decy)");
    }
    println!();

    // Quality analysis capabilities
    if registry.pmat.is_some() {
        println!("  ✅ Quality Analysis (PMAT)");
        println!("     • TDG scoring (Technical Debt Grade)");
        println!("     • Complexity analysis");
        println!("     • Code quality metrics");
        println!("     • Adaptive analysis (Muda elimination)");
    } else {
        println!("  ❌ Quality Analysis (install pmat)");
    }
    println!();

    // Ruchy capabilities
    if registry.ruchy.is_some() {
        println!("  ✅ Ruchy Scripting");
        println!("     • Ruby-like syntax with Rust performance");
        println!("     • Gradual typing (permissive/strict modes)");
        println!("     • Interactive REPL");
        println!("     • Formal verification (provability)");
    } else {
        println!("  ❌ Ruchy Scripting (install ruchy)");
    }
    println!();

    // Example workflow
    println!("📝 Example Workflow:");
    println!("────────────────────");
    println!();

    println!("1. Analyze Python project:");
    println!("   batuta analyze --languages --tdg /path/to/python_project");
    println!();

    println!("2. Transpile to Rust:");
    println!("   batuta transpile --input /path/to/python_project \\");
    println!("                     --output /path/to/rust_project");
    println!();

    println!("3. Optimize with MoE backend selection:");
    println!("   batuta optimize --enable-gpu --enable-simd \\");
    println!("                   /path/to/rust_project");
    println!();

    println!("4. Validate with syscall tracing:");
    println!("   batuta validate --trace-syscalls \\");
    println!("                   /path/to/rust_project");
    println!();

    println!("5. Build production binary:");
    println!("   batuta build --release /path/to/rust_project");
    println!();

    println!("6. Generate migration report:");
    println!("   batuta report --format html --output report.html");
    println!();

    // Integration with ML converters
    println!("🤖 ML Library Conversion:");
    println!("──────────────────────────");
    println!();

    println!("For Python projects using ML libraries:");
    println!("  • NumPy operations → Trueno (SIMD/GPU)");
    println!("  • sklearn algorithms → Aprender");
    println!("  • PyTorch inference → Realizar");
    println!();

    println!("Batuta automatically detects and converts:");
    println!("  ✓ np.add(a, b) → trueno::add(&a, &b)");
    println!("  ✓ sklearn.LinearRegression() → aprender::LinearRegression::new()");
    println!("  ✓ model.generate() → realizar::generate_text()");
    println!();

    // Toyota Way principles
    println!("🏭 Toyota Way Integration:");
    println!("───────────────────────────");
    println!();

    println!("Batuta implements Toyota Production System principles:");
    println!("  • Jidoka: Stop-the-line quality (error on warnings)");
    println!("  • Muda: Waste elimination (adaptive analysis)");
    println!("  • Kaizen: Continuous improvement (MoE optimization)");
    println!("  • Andon: Problem visualization (PARF analysis)");
    println!("  • Heijunka: Level scheduling (pipeline orchestration)");
    println!();

    // EXTREME TDD
    println!("⚡ EXTREME TDD Quality Gates:");
    println!("──────────────────────────────");
    println!();

    println!("All transpilations pass through:");
    println!("  1. Pre-commit checks (< 30s)");
    println!("  2. Fast test suite (< 5min)");
    println!("  3. Coverage analysis (< 10min)");
    println!("  4. Mutation testing (optional)");
    println!();

    if registry.pmat.is_some() {
        println!("Run quality checks:");
        println!("  make pre-commit  # Fast checks before commit");
        println!("  make test-fast   # Full test suite");
        println!("  make coverage    # Coverage report");
        println!("  make tdg         # TDG score with PMAT");
        println!();
    }

    // Missing tools guidance
    let missing_tools: Vec<&str> = vec!["decy", "depyler", "bashrs", "pmat"]
        .into_iter()
        .filter(|tool| match *tool {
            "decy" => registry.decy.is_none(),
            "depyler" => registry.depyler.is_none(),
            "bashrs" => registry.bashrs.is_none(),
            "pmat" => registry.pmat.is_none(),
            _ => false,
        })
        .collect();

    if !missing_tools.is_empty() {
        println!("📦 Install Missing Tools:");
        println!("─────────────────────────");

        let instructions = registry.get_installation_instructions(&missing_tools);
        for instruction in instructions {
            println!("  {}", instruction);
        }
        println!();
    }

    // Success message
    println!("✅ Batuta Orchestration Framework Ready!");
    println!();
    println!("   Convert ANY project (Python/C/C++/Shell) to Rust with:");
    println!("   • Automatic transpilation");
    println!("   • ML library conversion (NumPy/sklearn/PyTorch)");
    println!("   • MoE backend optimization (SIMD/GPU)");
    println!("   • Quality analysis and reporting");
    println!("   • Formal verification support");
    println!();
    println!("   Start with: batuta analyze <project_path>");
}
