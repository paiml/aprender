//! Simple example demonstrating trueno-graph usage
//!
//! Run with: cargo run --example simple_graph

use trueno_graph::{CsrGraph, NodeId};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦀 trueno-graph MVP Example\n");

    // 1. Build a call graph
    println!("📊 Building call graph...");
    let mut graph = CsrGraph::new();

    // Add edges (function calls)
    graph.add_edge(NodeId(0), NodeId(1), 1.0)?; // main → parse_args
    graph.add_edge(NodeId(0), NodeId(2), 1.0)?; // main → validate
    graph.add_edge(NodeId(0), NodeId(3), 1.0)?; // main → execute
    graph.add_edge(NodeId(1), NodeId(2), 2.0)?; // parse_args → validate (called twice)
    graph.add_edge(NodeId(3), NodeId(4), 1.0)?; // execute → cleanup

    // Add node names
    graph.set_node_name(NodeId(0), "main".to_string());
    graph.set_node_name(NodeId(1), "parse_args".to_string());
    graph.set_node_name(NodeId(2), "validate".to_string());
    graph.set_node_name(NodeId(3), "execute".to_string());
    graph.set_node_name(NodeId(4), "cleanup".to_string());

    println!("  ✅ Graph built: {} nodes, {} edges\n", graph.num_nodes(), graph.num_edges());

    // 2. Query neighbors
    println!("🔍 Querying graph...");

    let main_callees = graph.outgoing_neighbors(NodeId(0))?;
    println!("  main() calls: {:?}", main_callees);
    for callee in main_callees {
        let name = graph.get_node_name(NodeId(*callee)).unwrap_or("unknown");
        println!("    → {}", name);
    }

    let validate_callers = graph.incoming_neighbors(NodeId(2))?;
    println!("\n  validate() called by: {:?}", validate_callers);
    for &caller in validate_callers {
        let name = graph.get_node_name(NodeId(caller)).unwrap_or("unknown");
        println!("    ← {}", name);
    }

    // 3. Persist to Parquet
    println!("\n💾 Saving to Parquet...");
    let path = std::env::temp_dir().join("example_graph");
    graph.write_parquet(&path).await?;
    println!("  ✅ Saved to {}_edges.parquet", path.display());
    println!("  ✅ Saved to {}_nodes.parquet", path.display());

    // 4. Load from Parquet
    println!("\n📂 Loading from Parquet...");
    let loaded = CsrGraph::read_parquet(&path).await?;
    println!("  ✅ Loaded: {} nodes, {} edges", loaded.num_nodes(), loaded.num_edges());

    // Verify roundtrip
    assert_eq!(loaded.num_nodes(), graph.num_nodes());
    assert_eq!(loaded.num_edges(), graph.num_edges());
    assert_eq!(loaded.get_node_name(NodeId(0)), Some("main"));

    println!("\n✨ Example complete!");

    Ok(())
}
