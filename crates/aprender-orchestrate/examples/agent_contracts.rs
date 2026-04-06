//! Agent Contract Verification Demo
//!
//! Demonstrates provable design-by-contract verification:
//! - Parse YAML contracts (agent-loop-v1.yaml)
//! - Verify all 16 invariant test bindings exist
//! - Generate a verification report
//!
//! Run with: `cargo run --example agent_contracts --features agents`

#[cfg(feature = "agents")]
fn main() {
    use batuta::agent::contracts::{parse_contract, verify_bindings};

    println!("Agent Design-by-Contract Verification");
    println!("======================================");
    println!();

    // Parse the contract YAML
    let yaml = include_str!("../contracts/agent-loop-v1.yaml");
    let contract = parse_contract(yaml).expect("failed to parse contract");

    println!("Contract: {} v{}", contract.contract.name, contract.contract.version,);
    println!("Module:   {}", contract.contract.module);
    println!("Invariants: {}", contract.invariants.len(),);
    println!();

    // Display all invariants
    println!("--- Invariant Summary ---");
    println!();
    for inv in &contract.invariants {
        println!("  {} — {}", inv.id, inv.name);
        println!("    {}", inv.description.trim());
        let eq_line =
            inv.equation.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("(none)");
        println!("    Eq: {eq_line}");
        println!("    Bind: {}", inv.test_binding,);
        println!();
    }

    // Verify bindings against known test names
    // In CI, these would come from `cargo test --list`
    let known_tests = vec![
        "agent::guard::tests::test_iteration_limit",
        "agent::guard::tests::test_counters",
        "agent::runtime::tests::test_capability_denied_handled",
        "agent::guard::tests::test_pingpong_detection",
        "agent::guard::tests::test_cost_budget",
        "agent::guard::tests::test_consecutive_max_tokens",
        "agent::runtime::tests::test_conversation_stored_in_memory",
        "agent::pool::tests::test_pool_capacity_limit",
        "agent::pool::tests::test_pool_fan_out_fan_in",
        "agent::pool::tests::test_pool_join_all",
        "agent::tool::tests::test_sanitize_output_system_injection",
        "agent::tool::spawn::tests::test_spawn_tool_depth_limit",
        "agent::tool::network::tests::test_blocked_host",
        "agent::tool::inference::tests::test_inference_tool_timeout",
        "agent::runtime::tests_advanced::test_sovereign_privacy_blocks_network",
        "agent::guard::tests::test_token_budget_exhausted",
    ];

    let known: Vec<String> = known_tests.iter().map(|s| (*s).to_string()).collect();
    let result = verify_bindings(&contract, &known);

    println!("--- Verification Result ---");
    println!();
    println!("{}", result.report());

    // Display quality thresholds
    println!("--- Quality Thresholds ---");
    println!();
    println!("  Coverage target:  {}%", contract.verification.coverage_target,);
    println!("  Mutation target:  {}%", contract.verification.mutation_target,);
    println!("  Max cyclomatic:   {}", contract.verification.complexity_max_cyclomatic,);
    println!("  Max cognitive:    {}", contract.verification.complexity_max_cognitive,);
    println!("  Unit test paths:  {}", contract.verification.unit_tests.len(),);

    if result.all_verified() {
        println!();
        println!("All {} invariants verified.", result.total_invariants);
    } else {
        println!();
        eprintln!(
            "FAIL: {}/{} invariants verified",
            result.verified_bindings, result.total_invariants,
        );
        std::process::exit(1);
    }
}

#[cfg(not(feature = "agents"))]
fn main() {
    eprintln!("Enable `agents` feature: cargo run --example agent_contracts --features agents");
    std::process::exit(1);
}
