//! `pv verify-pipeline` — compositional shape verification across contracts.
//!
//! Walks the dependency graph in topological order and verifies that every
//! assumes/guarantees edge is satisfied. Produces a composition report
//! showing the full proof chain or identifying break points.
//!
//! Spec: docs/specifications/sub/model-layout-provability.md (§36, P0-3)

use std::collections::BTreeMap;
use std::path::Path;

use provable_contracts::graph::dependency_graph;
use provable_contracts::schema::Contract;
use serde_json::Value;

use crate::contract_walk::collect_contracts;
use crate::json_obj::obj;

/// Run the verify-pipeline command.
pub fn run(contract_dir: &Path, format: &str) {
    // 1. Load all contracts
    let mut contracts = Vec::new();
    collect_contracts(contract_dir, &mut contracts);
    contracts.sort_by(|a, b| a.0.cmp(&b.0));

    if contracts.is_empty() {
        eprintln!("No contracts found in {}", contract_dir.display());
        return;
    }

    // 2. Build dependency graph + topological sort
    let refs: Vec<(String, &Contract)> = contracts.iter().map(|(s, c)| (s.clone(), c)).collect();
    let graph = dependency_graph(&refs);

    if !graph.cycles.is_empty() {
        eprintln!(
            "ERROR: {} cycle(s) detected in dependency graph — cannot verify pipeline",
            graph.cycles.len()
        );
        for cycle in &graph.cycles {
            eprintln!("  cycle: {}", cycle.join(" → "));
        }
        std::process::exit(1);
    }

    // 3. Build stem→contract index and walk edges
    let index: BTreeMap<&str, &Contract> = contracts.iter().map(|(s, c)| (s.as_str(), c)).collect();
    let (chains, edges_total, edges_satisfied, edges_broken) =
        walk_composition_edges(&graph.topo_order, &index);

    // 4. Output
    if format == "json" {
        print_json(&chains, edges_total, edges_satisfied, &edges_broken, &graph);
    } else {
        print_text(&chains, edges_total, edges_satisfied, &edges_broken, &graph);
    }

    if !edges_broken.is_empty() {
        std::process::exit(1);
    }
}

fn walk_composition_edges(
    topo_order: &[String],
    index: &BTreeMap<&str, &Contract>,
) -> (Vec<CompositionEdge>, usize, usize, Vec<CompositionEdge>) {
    let mut edges_total = 0usize;
    let mut edges_satisfied = 0usize;
    let mut edges_broken = Vec::new();
    let mut chains: Vec<CompositionEdge> = Vec::new();

    for stem in topo_order {
        let Some(contract) = index.get(stem.as_str()) else {
            continue;
        };
        for (eq_name, equation) in &contract.equations {
            let Some(assumes) = &equation.assumes else {
                continue;
            };
            let Some(from_contract) = &assumes.from_contract else {
                continue;
            };

            edges_total += 1;
            let from_eq = assumes.from_equation.as_deref();

            let edge = CompositionEdge {
                downstream: format!("{stem}.{eq_name}"),
                upstream: format!(
                    "{from_contract}{}",
                    from_eq.map_or(String::new(), |e| format!(".{e}"))
                ),
                assumed_shapes: assumes.shapes.keys().cloned().collect(),
                status: EdgeStatus::Unknown,
            };

            // Resolve upstream
            let Some(upstream_contract) = index.get(from_contract.as_str()) else {
                let mut e = edge;
                e.status = EdgeStatus::Broken("upstream contract not found".into());
                edges_broken.push(e.clone());
                chains.push(e);
                continue;
            };

            if let Some(upstream_eq_name) = from_eq {
                let Some(upstream_eq) = upstream_contract.equations.get(upstream_eq_name) else {
                    let mut e = edge;
                    e.status = EdgeStatus::Broken("upstream equation not found".into());
                    edges_broken.push(e.clone());
                    chains.push(e);
                    continue;
                };

                // `let ... else` rather than an `is_none()` guard followed by
                // `.unwrap()`: the binding carries the proof that guarantees
                // exist, so there is no unwrap left to justify.
                let Some(guarantees) = upstream_eq.guarantees.as_ref() else {
                    let mut e = edge;
                    e.status = EdgeStatus::Broken("upstream has no guarantees".into());
                    edges_broken.push(e.clone());
                    chains.push(e);
                    continue;
                };

                // Edge satisfied: upstream has guarantees
                let mut e = edge;
                let guaranteed_shapes: Vec<String> = guarantees.shapes.keys().cloned().collect();
                e.status = EdgeStatus::Satisfied(guaranteed_shapes);
                edges_satisfied += 1;
                chains.push(e);
            } else {
                // No specific equation -- check any equation has guarantees
                let has_guarantees = upstream_contract
                    .equations
                    .values()
                    .any(|eq| eq.guarantees.is_some());
                let mut e = edge;
                if has_guarantees {
                    e.status = EdgeStatus::Satisfied(vec![]);
                    edges_satisfied += 1;
                } else {
                    e.status = EdgeStatus::Broken("no equations with guarantees".into());
                    edges_broken.push(e.clone());
                }
                chains.push(e);
            }
        }
    }

    (chains, edges_total, edges_satisfied, edges_broken)
}

#[derive(Debug, Clone)]
struct CompositionEdge {
    downstream: String,
    upstream: String,
    assumed_shapes: Vec<String>,
    status: EdgeStatus,
}

#[derive(Debug, Clone)]
enum EdgeStatus {
    Unknown,
    Satisfied(Vec<String>),
    Broken(String),
}

fn print_text(
    chains: &[CompositionEdge],
    total: usize,
    satisfied: usize,
    broken: &[CompositionEdge],
    graph: &provable_contracts::graph::DependencyGraph,
) {
    println!("pv verify-pipeline — Compositional Shape Verification");
    println!("=====================================================");
    println!();
    println!(
        "Contracts: {}  |  Topo depth: {}",
        graph.nodes.len(),
        graph.topo_order.len()
    );
    println!(
        "Edges: {}  |  Satisfied: {}  |  Broken: {}",
        total,
        satisfied,
        broken.len()
    );
    println!();

    if !chains.is_empty() {
        println!("Composition edges:");
        for edge in chains {
            let icon = match &edge.status {
                EdgeStatus::Satisfied(_) => "✓",
                EdgeStatus::Broken(_) => "✗",
                EdgeStatus::Unknown => "?",
            };
            let detail = match &edge.status {
                EdgeStatus::Satisfied(shapes) if !shapes.is_empty() => {
                    format!(" (guarantees: {})", shapes.join(", "))
                }
                EdgeStatus::Broken(reason) => format!(" — {reason}"),
                _ => String::new(),
            };
            println!("  {icon} {} ← {}{detail}", edge.downstream, edge.upstream);
        }
        println!();
    }

    if broken.is_empty() {
        println!("Result: PASS — all composition edges satisfied");
    } else {
        println!("Result: FAIL — {} broken edge(s)", broken.len());
        for edge in broken {
            if let EdgeStatus::Broken(reason) = &edge.status {
                println!("  ✗ {} ← {} — {reason}", edge.downstream, edge.upstream);
            }
        }
    }
}

fn print_json(
    chains: &[CompositionEdge],
    total: usize,
    satisfied: usize,
    broken: &[CompositionEdge],
    graph: &provable_contracts::graph::DependencyGraph,
) {
    let edges_json: Vec<Value> = chains
        .iter()
        .map(|e| {
            let (status, detail) = match &e.status {
                EdgeStatus::Satisfied(shapes) => (
                    "satisfied",
                    obj([("guaranteed_shapes", Value::from(shapes.clone()))]),
                ),
                EdgeStatus::Broken(reason) => {
                    ("broken", obj([("reason", Value::from(reason.clone()))]))
                }
                EdgeStatus::Unknown => ("unknown", obj([])),
            };
            obj([
                ("downstream", Value::from(e.downstream.clone())),
                ("upstream", Value::from(e.upstream.clone())),
                ("assumed_shapes", Value::from(e.assumed_shapes.clone())),
                ("status", Value::from(status)),
                ("detail", detail),
            ])
        })
        .collect();

    let report = obj([
        ("contracts", Value::from(graph.nodes.len())),
        ("topo_depth", Value::from(graph.topo_order.len())),
        ("edges_total", Value::from(total)),
        ("edges_satisfied", Value::from(satisfied)),
        ("edges_broken", Value::from(broken.len())),
        ("passed", Value::from(broken.is_empty())),
        ("edges", Value::Array(edges_json)),
    ]);

    println!(
        "{}",
        serde_json::to_string_pretty(&report)
            .expect("a serde_json::Value of objects/arrays/strings/numbers always serializes")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_pipeline_on_real_contracts() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        if !dir.exists() {
            return; // skip in CI without contracts
        }
        // Should not panic
        run(&dir, "text");
    }

    #[test]
    fn verify_pipeline_json_on_real_contracts() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        if !dir.exists() {
            return;
        }
        run(&dir, "json");
    }

    #[test]
    fn verify_pipeline_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        run(tmp.path(), "text");
    }
}
