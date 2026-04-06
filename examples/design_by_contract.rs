//! Design by Contract Example
//!
//! Demonstrates batuta's DbC enforcement: contract gap analysis via the bug
//! hunter, and dependency graph validation via the stack module.
//!
//! Run with: cargo run --example design_by_contract --features native

fn main() {
    println!("Design by Contract in Batuta");
    println!("{}\n", "=".repeat(50));

    // -----------------------------------------------------------------
    // 1. Contract Gap Analysis (BH-26)
    // -----------------------------------------------------------------
    println!("1. Contract Gap Analysis (BH-26)\n");
    println!("   The bug hunter scans provable-contracts binding");
    println!("   registries and produces BH-CONTRACT findings:\n");

    let statuses = [
        ("not_implemented", "High", 0.8, "No implementation exists"),
        ("partial", "Medium", 0.6, "Binding is incomplete"),
        ("(unbound)", "Medium", 0.5, "Contract has no binding"),
        ("(<50% proofs)", "Low", 0.4, "Insufficient falsification tests"),
    ];
    for (status, severity, susp, desc) in statuses {
        println!("   {:20} [{:6}] ({:.1}) {}", status, severity, susp, desc);
    }

    // -----------------------------------------------------------------
    // 2. Stack Dependency Graph Validation
    // -----------------------------------------------------------------
    #[cfg(feature = "native")]
    {
        use batuta::stack::DependencyGraph;

        println!("\n{}", "=".repeat(50));
        println!("2. Dependency Graph Validation\n");

        let graph = DependencyGraph::new();
        assert!(!graph.has_cycles(), "Empty graph must be acyclic");
        println!("   Empty graph: acyclic = {}", !graph.has_cycles());

        let order = graph.topological_order().unwrap_or_default();
        println!("   Topological order (empty): {:?}", order);

        println!("\n   Expected publish order for the full stack:");
        let expected = ["trueno", "aprender", "realizar", "entrenar", "apr-cli"];
        for (i, name) in expected.iter().enumerate() {
            println!("     {}. {}", i + 1, name);
        }
    }

    // -----------------------------------------------------------------
    // 3. Programmatic Bug Hunt with Contract Flags
    // -----------------------------------------------------------------
    #[cfg(feature = "native")]
    {
        use batuta::bug_hunter::{hunt, HuntConfig, HuntMode};
        use std::path::Path;

        println!("\n{}", "=".repeat(50));
        println!("3. Programmatic Contract Gap Hunt\n");

        let config = HuntConfig {
            mode: HuntMode::Quick,
            contracts_auto: true,
            min_suspiciousness: 0.5,
            ..Default::default()
        };

        let result = hunt(Path::new("."), config);
        let contract_findings: Vec<_> =
            result.findings.iter().filter(|f| f.id.starts_with("BH-CONTRACT")).collect();

        println!("   Total findings: {}", result.findings.len());
        println!("   Contract gaps:  {}", contract_findings.len());
        for f in contract_findings.iter().take(5) {
            println!("     [{}] {}", f.severity, f.title);
        }
    }

    println!("\n{}", "=".repeat(50));
    println!("Done.");
}
