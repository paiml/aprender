//! Graph algorithms example (Phase 2)
//!
//! Demonstrates BFS, find_callers, and PageRank on a call graph
//!
//! Run with: cargo run --example graph_algorithms

use trueno_graph::{bfs, find_callers, pagerank, CsrGraph, NodeId};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦀 trueno-graph Phase 2: Graph Algorithms\n");

    let graph = build_call_graph()?;

    let reachable = demo_bfs(&graph)?;
    let callers = demo_find_callers(&graph)?;
    let (ranked, scores) = demo_pagerank(&graph)?;

    demo_critical_path(&graph, &scores)?;
    let diff = demo_parquet_roundtrip(&graph, &scores).await?;

    print_summary(&reachable, &callers, &ranked, diff);

    Ok(())
}

fn build_call_graph() -> Result<CsrGraph, Box<dyn std::error::Error>> {
    println!("📊 Building call graph...");
    let mut graph = CsrGraph::new();

    // Main function calls
    graph.add_edge(NodeId(0), NodeId(1), 1.0)?; // main → parse_args
    graph.add_edge(NodeId(0), NodeId(2), 1.0)?; // main → validate_config
    graph.add_edge(NodeId(0), NodeId(3), 1.0)?; // main → execute

    // parse_args dependencies
    graph.add_edge(NodeId(1), NodeId(2), 1.0)?; // parse_args → validate_config
    graph.add_edge(NodeId(1), NodeId(4), 1.0)?; // parse_args → read_file

    // execute dependencies
    graph.add_edge(NodeId(3), NodeId(5), 1.0)?; // execute → process_data
    graph.add_edge(NodeId(3), NodeId(6), 1.0)?; // execute → write_output

    // process_data dependencies
    graph.add_edge(NodeId(5), NodeId(7), 1.0)?; // process_data → analyze
    graph.add_edge(NodeId(5), NodeId(8), 1.0)?; // process_data → transform

    // Circular dependency (common in real code)
    graph.add_edge(NodeId(8), NodeId(5), 0.5)?; // transform → process_data

    // Add node names
    let names = vec![
        "main",
        "parse_args",
        "validate_config",
        "execute",
        "read_file",
        "process_data",
        "write_output",
        "analyze",
        "transform",
    ];

    for (id, name) in names.iter().enumerate() {
        graph.set_node_name(NodeId(id as u32), (*name).to_string());
    }

    println!("  ✅ Graph built: {} nodes, {} edges\n", graph.num_nodes(), graph.num_edges());

    Ok(graph)
}

fn demo_bfs(graph: &CsrGraph) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    println!("🔍 BFS: Finding all functions reachable from main()...");
    let reachable = bfs(graph, NodeId(0))?;
    println!("  Found {} reachable functions:", reachable.len());
    for &node in &reachable {
        let name = graph.get_node_name(NodeId(node)).unwrap_or("unknown");
        println!("    → {}", name);
    }
    Ok(reachable)
}

fn demo_find_callers(graph: &CsrGraph) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    println!("\n🔍 Finding all callers of validate_config()...");
    let callers = find_callers(graph, NodeId(2), 10)?;
    println!("  Found {} callers:", callers.len());
    for &caller in &callers {
        let name = graph.get_node_name(NodeId(caller)).unwrap_or("unknown");
        println!("    ← {}", name);
    }
    Ok(callers)
}

fn demo_pagerank(
    graph: &CsrGraph,
) -> Result<(Vec<(&str, f32)>, Vec<f32>), Box<dyn std::error::Error>> {
    println!("\n📊 Computing PageRank importance scores...");
    let scores = pagerank(graph, 20, 1e-6)?;

    // Create ranked list
    let mut ranked: Vec<_> = (0..graph.num_nodes())
        .map(|i| {
            let name = graph.get_node_name(NodeId(i as u32)).unwrap_or("unknown");
            (name, scores[i])
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  Top 5 most important functions:");
    for (i, (name, score)) in ranked.iter().take(5).enumerate() {
        println!("    {}. {} (score: {:.4})", i + 1, name, score);
    }

    Ok((ranked, scores))
}

fn demo_critical_path(graph: &CsrGraph, scores: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Finding all callers of process_data() (depth 5)...");
    let process_callers = find_callers(graph, NodeId(5), 5)?;
    println!("  Found {} functions in call chain:", process_callers.len());
    for &caller in &process_callers {
        let name = graph.get_node_name(NodeId(caller)).unwrap_or("unknown");
        let score = scores[caller as usize];
        println!("    ← {} (importance: {:.4})", name, score);
    }
    Ok(())
}

async fn demo_parquet_roundtrip(
    graph: &CsrGraph,
    scores: &[f32],
) -> Result<f32, Box<dyn std::error::Error>> {
    println!("\n💾 Saving to Parquet...");
    let path = std::env::temp_dir().join("call_graph");
    graph.write_parquet(&path).await?;
    println!("  ✅ Saved to {}_edges.parquet", path.display());
    println!("  ✅ Saved to {}_nodes.parquet", path.display());

    println!("\n📂 Loading from Parquet...");
    let loaded = CsrGraph::read_parquet(&path).await?;
    let loaded_scores = pagerank(&loaded, 20, 1e-6)?;

    // Verify PageRank is consistent
    let diff: f32 = scores.iter().zip(loaded_scores.iter()).map(|(a, b)| (a - b).abs()).sum();

    println!("  ✅ Loaded: {} nodes, {} edges", loaded.num_nodes(), loaded.num_edges());
    println!("  ✅ PageRank consistency: {:.2e} (sum of absolute differences)", diff);

    Ok(diff)
}

fn print_summary(reachable: &[u32], callers: &[u32], ranked: &[(&str, f32)], diff: f32) {
    println!("\n✨ Phase 2 algorithms complete!");
    println!("\nSummary:");
    println!("  • BFS found {} reachable functions from main()", reachable.len());
    println!("  • validate_config() has {} callers", callers.len());
    println!("  • Most important: {} (score: {:.4})", ranked[0].0, ranked[0].1);
    println!("  • Parquet roundtrip verified (diff: {:.2e})", diff);
}
