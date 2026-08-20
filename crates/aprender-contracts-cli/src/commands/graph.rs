use std::path::Path;
use std::str::FromStr;

use provable_contracts::graph::{dependency_graph, graph_nodes, DependencyGraph};

use crate::contract_walk::collect_contracts;

/// Output format for the dependency graph rendering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphFormat {
    /// Human-readable text output.
    Text,
    /// Graphviz DOT language output.
    Dot,
    /// JSON structured output.
    Json,
    /// Mermaid diagram syntax output.
    Mermaid,
}

impl FromStr for GraphFormat {
    type Err = String;

    /// Parse a format name string into a `GraphFormat` variant
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "text" => Ok(Self::Text),
            "dot" => Ok(Self::Dot),
            "json" => Ok(Self::Json),
            "mermaid" => Ok(Self::Mermaid),
            other => Err(format!(
                "unknown format '{other}', expected 'text', 'dot', 'json', or 'mermaid'"
            )),
        }
    }
}

/// Load contracts from a directory and render their dependency graph
pub fn run(contract_dir: &Path, format: GraphFormat) -> Result<(), Box<dyn std::error::Error>> {
    let mut contracts = Vec::new();
    collect_contracts(contract_dir, &mut contracts);
    contracts.sort_by(|a, b| a.0.cmp(&b.0));

    let refs: Vec<(String, &provable_contracts::schema::Contract)> =
        contracts.iter().map(|(s, c)| (s.clone(), c)).collect();

    let graph = dependency_graph(&refs);

    match format {
        GraphFormat::Text => render_text(&graph),
        GraphFormat::Dot => render_dot(&graph),
        GraphFormat::Json => render_json(&graph),
        GraphFormat::Mermaid => render_mermaid(&graph),
    }

    if !graph.cycles.is_empty() {
        return Err("Dependency graph contains cycles".into());
    }
    Ok(())
}

/// Render the dependency graph as human-readable text to stdout
fn render_text(graph: &DependencyGraph) {
    let nodes = graph_nodes(graph);
    println!("Contract Dependency Graph");
    println!("=========================");
    println!("Nodes: {}", graph.nodes.len());
    println!();

    for node in &nodes {
        let deps = graph.edges.get(&node.stem).cloned().unwrap_or_default();
        if deps.is_empty() {
            println!("  {} (dependents: {}, deps: 0)", node.stem, node.dependents);
        } else {
            println!(
                "  {} → [{}] (dependents: {})",
                node.stem,
                deps.join(", "),
                node.dependents,
            );
        }
    }

    if !graph.cycles.is_empty() {
        println!();
        println!("CYCLES DETECTED:");
        for cycle in &graph.cycles {
            println!("  {}", cycle.join(" → "));
        }
    }

    if !graph.topo_order.is_empty() {
        println!();
        println!("Topological order:");
        for (i, node) in graph.topo_order.iter().enumerate() {
            println!("  {}: {node}", i + 1);
        }
    }
}

/// Render the dependency graph in Graphviz DOT format to stdout
fn render_dot(graph: &DependencyGraph) {
    println!("digraph contracts {{");
    println!("    rankdir=LR;");
    println!("    node [shape=box, style=rounded, fontname=\"Helvetica\"];");
    println!("    edge [color=\"#666666\"];");
    println!();

    // Emit all nodes (including isolated ones)
    for node in &graph.nodes {
        let deps = graph.edges.get(node).map_or(0, Vec::len);
        if deps == 0 && !is_depended_on(graph, node) {
            println!("    \"{node}\";");
        }
    }

    // Emit edges
    for (node, deps) in &graph.edges {
        for dep in deps {
            println!("    \"{node}\" -> \"{dep}\";");
        }
    }

    // Highlight cycles in red
    if !graph.cycles.is_empty() {
        println!();
        println!("    // Cycles detected");
        for cycle in &graph.cycles {
            for pair in cycle.windows(2) {
                println!(
                    "    \"{0}\" -> \"{1}\" [color=red, penwidth=2];",
                    pair[0], pair[1]
                );
            }
        }
    }

    println!("}}");
}

/// Render the dependency graph as a JSON object to stdout
fn render_json(graph: &DependencyGraph) {
    println!("{{");
    // Nodes array
    let nodes_json: Vec<String> = graph.nodes.iter().map(|n| format!("    \"{n}\"")).collect();
    println!("  \"nodes\": [");
    println!("{}", nodes_json.join(",\n"));
    println!("  ],");

    // Edges array
    println!("  \"edges\": [");
    let mut edge_lines = Vec::new();
    for (node, deps) in &graph.edges {
        for dep in deps {
            edge_lines.push(format!("    {{\"from\": \"{node}\", \"to\": \"{dep}\"}}"));
        }
    }
    println!("{}", edge_lines.join(",\n"));
    println!("  ],");

    // Topo order
    let topo_json: Vec<String> = graph
        .topo_order
        .iter()
        .map(|n| format!("    \"{n}\""))
        .collect();
    println!("  \"topo_order\": [");
    println!("{}", topo_json.join(",\n"));
    println!("  ],");

    // Cycles
    println!("  \"cycles\": [");
    let cycle_lines: Vec<String> = graph
        .cycles
        .iter()
        .map(|c| {
            let items: Vec<String> = c.iter().map(|n| format!("\"{n}\"")).collect();
            format!("    [{}]", items.join(", "))
        })
        .collect();
    println!("{}", cycle_lines.join(",\n"));
    println!("  ]");
    println!("}}");
}

/// Render the dependency graph in Mermaid diagram syntax to stdout
fn render_mermaid(graph: &DependencyGraph) {
    println!("graph TD");

    // Emit edges
    for (node, deps) in &graph.edges {
        let from = mermaid_id(node);
        if deps.is_empty() {
            println!("    {from}[\"{node}\"]");
        }
        for dep in deps {
            let to = mermaid_id(dep);
            println!("    {from}[\"{node}\"] --> {to}[\"{dep}\"]");
        }
    }

    if !graph.cycles.is_empty() {
        println!();
        println!("    %% Cycles detected");
        for cycle in &graph.cycles {
            let names: Vec<&str> = cycle.iter().map(String::as_str).collect();
            println!("    %% {}", names.join(" -> "));
        }
    }
}

/// Convert a contract stem to a valid Mermaid node ID (no hyphens).
fn mermaid_id(stem: &str) -> String {
    stem.replace('-', "_")
}

/// Check whether any other node in the graph depends on the given node
fn is_depended_on(graph: &DependencyGraph, node: &str) -> bool {
    graph
        .edges
        .values()
        .any(|deps| deps.iter().any(|d| d == node))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_format_from_str() {
        assert_eq!(GraphFormat::from_str("text").unwrap(), GraphFormat::Text);
        assert_eq!(GraphFormat::from_str("dot").unwrap(), GraphFormat::Dot);
        assert_eq!(GraphFormat::from_str("json").unwrap(), GraphFormat::Json);
        assert_eq!(
            GraphFormat::from_str("mermaid").unwrap(),
            GraphFormat::Mermaid
        );
        assert!(GraphFormat::from_str("xml").is_err());
    }

    #[test]
    fn test_from_str_other_format_returns_descriptive_error() {
        // Exercises the `other =>` catch-all arm in from_str
        let other = "yaml";
        let err = GraphFormat::from_str(other).unwrap_err();
        assert!(
            err.contains(other),
            "error should include the unrecognized format"
        );
        assert!(err.contains("unknown format"));
    }

    #[test]
    fn test_mermaid_id() {
        assert_eq!(mermaid_id("softmax-kernel-v1"), "softmax_kernel_v1");
        assert_eq!(mermaid_id("silu"), "silu");
    }
}
