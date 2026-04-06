#![allow(clippy::disallowed_methods)]
//! PAR-201: Execution Path Graph Demo
//!
//! Run with: cargo run --example execution_graph

use std::thread::sleep;
use std::time::Duration;
use trueno::{BrickId, BrickProfiler, ExecutionNode, PtxRegistry};

fn main() {
    println!("=== PAR-201: Execution Path Graph Demo ===\n");

    // Create profiler with graph enabled
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    profiler.enable_graph();

    // Simulate a transformer forward pass with graph recording
    println!("Recording execution graph for 2 transformer layers...\n");

    for layer in 0..2 {
        // Push layer scope
        profiler.graph_push_scope(ExecutionNode::Layer { index: layer });

        // RmsNorm - push brick scope to show kernel as child
        let timer = profiler.start_brick(BrickId::RmsNorm);
        sleep(Duration::from_micros(50));
        profiler.stop_brick(timer, 4096);
        profiler.graph_push_scope(ExecutionNode::Brick {
            id: BrickId::RmsNorm,
            timing_ns: 50_000,
            elements: 4096,
        });
        profiler.graph_record_kernel(
            "rmsnorm_kernel",
            0x1234567890ABCDEF,
            (16, 1, 1),
            (256, 1, 1),
            1024,
        );
        profiler.graph_pop_scope();

        // QKV Projection
        let timer = profiler.start_brick(BrickId::QkvProjection);
        sleep(Duration::from_micros(200));
        profiler.stop_brick(timer, 4096);
        profiler.graph_push_scope(ExecutionNode::Brick {
            id: BrickId::QkvProjection,
            timing_ns: 200_000,
            elements: 4096,
        });
        profiler.graph_record_kernel(
            "batched_q4k_gemv",
            0xDEADBEEFCAFEBABE,
            (32, 1, 1),
            (256, 1, 1),
            4096,
        );
        profiler.graph_pop_scope();

        // Attention
        let timer = profiler.start_brick(BrickId::AttentionScore);
        sleep(Duration::from_micros(150));
        profiler.stop_brick(timer, 4096);
        profiler.graph_push_scope(ExecutionNode::Brick {
            id: BrickId::AttentionScore,
            timing_ns: 150_000,
            elements: 4096,
        });
        profiler.graph_record_kernel(
            "incremental_attention",
            0xFEEDFACE12345678,
            (8, 1, 1),
            (256, 1, 1),
            2048,
        );
        profiler.graph_pop_scope();

        // FFN
        let timer = profiler.start_brick(BrickId::GateProjection);
        sleep(Duration::from_micros(300));
        profiler.stop_brick(timer, 4096);
        profiler.graph_push_scope(ExecutionNode::Brick {
            id: BrickId::GateProjection,
            timing_ns: 300_000,
            elements: 4096,
        });
        profiler.graph_record_kernel(
            "batched_q6k_gemv",
            0xABCDEF0123456789,
            (64, 1, 1),
            (256, 1, 1),
            8192,
        );
        profiler.graph_pop_scope();

        // Pop layer scope
        profiler.graph_pop_scope();
    }

    // Get the execution graph
    let graph = profiler.execution_graph();

    // Print graph statistics
    println!("Graph Statistics:");
    println!("  Nodes: {}", graph.num_nodes());
    println!("  Edges: {}", graph.num_edges());
    println!("  Scope balanced: {}", graph.is_scope_balanced());

    // Find all kernel nodes
    let kernels: Vec<_> = graph.kernel_nodes().collect();
    println!("\nKernel Nodes ({}):", kernels.len());
    for (id, node) in &kernels {
        if let ExecutionNode::Kernel { name, ptx_hash, grid, block, shared_mem, .. } = node {
            println!("  [{:>2}] {} (hash: 0x{:016x})", id.0, name, ptx_hash);
            println!("       grid: {:?}, block: {:?}, smem: {}B", grid, block, shared_mem);
        }
    }

    // Find slowest brick with kernel
    if let Some((id, node, timing_ns)) = graph.slowest_kernel() {
        println!("\nSlowest Brick with Kernel:");
        println!("  Node ID: {}", id.0);
        println!("  Timing: {:.1}µs", timing_ns as f64 / 1000.0);
        if let ExecutionNode::Brick { id: brick_id, .. } = node {
            println!("  Brick: {}", brick_id.name());
        }
    }

    // Phase 9: Critical Path Analysis
    println!("\n=== Phase 9: Critical Path Analysis ===\n");
    let (critical_path, total_ns) = graph.critical_path();
    println!(
        "Critical Path: {:.2}ms ({} nodes)",
        total_ns as f64 / 1_000_000.0,
        critical_path.len()
    );

    let slack = graph.compute_slack();
    let high_slack: Vec<_> = slack.iter().filter(|(_, &s)| s > 0).collect();
    if !high_slack.is_empty() {
        println!("\nParallelization Opportunities ({} nodes with slack):", high_slack.len());
    }

    // Formatted summary
    println!("\n{}", graph.critical_path_summary());

    // PTX Registry demo
    println!("\n=== PTX Registry Demo ===\n");
    let mut registry = PtxRegistry::new();

    let ptx_rmsnorm = r#"
.version 7.0
.target sm_80
.entry rmsnorm_kernel(.param .u64 out, .param .u64 in, .param .u32 n) {
    // ... kernel code ...
    ret;
}
"#;

    let ptx_gemv = r#"
.version 7.0
.target sm_80
.entry batched_q4k_gemv(.param .u64 out, .param .u64 in, .param .u64 weights) {
    // ... kernel code ...
    ret;
}
"#;

    registry.register(
        "rmsnorm_kernel",
        ptx_rmsnorm,
        Some(std::path::Path::new("src/kernels/norm.ptx")),
    );
    registry.register("batched_q4k_gemv", ptx_gemv, None);

    println!("Registered {} kernels", registry.len());
    for hash in registry.hashes() {
        if let Some(name) = registry.lookup_name(hash) {
            let path = registry
                .lookup_path(hash)
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "N/A".into());
            println!("  0x{:016x} -> {} ({})", hash, name, path);
        }
    }

    // Export to DOT format
    println!("\n=== DOT Export (Graphviz) ===\n");
    let dot = profiler.graph_to_dot();
    println!("DOT output ({} bytes):", dot.len());
    println!("{}", &dot[..dot.len().min(500)]);
    if dot.len() > 500 {
        println!("... ({} more bytes)", dot.len() - 500);
    }

    println!("\nTo visualize with Graphviz:");
    println!("  cargo run --example execution_graph > /tmp/graph.dot");
    println!("  dot -Tsvg /tmp/graph.dot -o /tmp/graph.svg");
    println!("  firefox /tmp/graph.svg");

    // Headless ASCII tree (no dependencies, works in CI/CD)
    println!("\n=== ASCII Tree (headless mode) ===\n");
    let ascii_tree = graph.to_ascii_tree();
    println!("{}", ascii_tree);

    // TUI visualization via presentar-terminal (optional)
    #[cfg(feature = "presentar-tui")]
    {
        println!("\n=== TUI TreeNode (presentar-terminal) ===\n");
        let tree_node = graph.to_tree_node();
        println!(
            "TreeNode structure: {} nodes, depth {}",
            tree_node.count_nodes(),
            tree_node.depth()
        );
        println!("(Use Tree widget for interactive TUI display)");
    }

    println!("\n✓ PAR-201 Execution Path Graph working correctly");
}
