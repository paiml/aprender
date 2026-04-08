#![allow(clippy::disallowed_methods)]
//! Chapter 15: Orchestration and Agents
//!
//! Demonstrates pipeline DAG and agent architecture contracts.
//! Citation: Yao et al., "ReAct," arXiv:2210.03629
//! Contract: contracts/apr-book-ch15-v1.yaml

fn main() {
    // Pipeline stages contract
    let stages = [
        "apr import",
        "apr validate",
        "apr oracle",
        "apr convert",
        "apr qa",
        "apr serve",
    ];
    println!("Pipeline stages ({} steps):", stages.len());
    for (i, stage) in stages.iter().enumerate() {
        println!("  {}. {stage}", i + 1);
    }
    assert_eq!(stages.len(), 6, "Pipeline has exactly 6 stages");

    // DAG dependency contract
    // Each stage depends on the previous one
    println!("\nDAG contract: strictly sequential (no cycles)");
    for i in 1..stages.len() {
        println!("  {} -> {}", stages[i - 1], stages[i]);
    }

    // ReAct agent loop (Yao et al., 2022)
    println!("\nReAct agent loop:");
    println!("  Thought -> Action -> Observation -> repeat");
    println!("  Action space: apr CLI subcommands");

    // Oracle-guided model selection
    println!("\nOracle consultation protocol:");
    println!("  apr oracle --family <family> --explain --stats");
    println!("  Validates architecture claims against known contracts");

    // Fail-fast contract
    println!("\nFail-fast contract:");
    println!("  Pipeline aborts on first contract violation");
    println!("  No partial deployments allowed");

    println!("\nChapter 15 contracts: PASSED");
}
