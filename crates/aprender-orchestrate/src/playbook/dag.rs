//! DAG construction from playbook deps/outs + after edges (PB-002)
//!
//! Builds a directed acyclic graph from implicit data dependencies (output→dep
//! matching) and explicit `after` edges. Reuses the fallback graph primitives
//! from `stack/graph.rs`.

use super::types::Playbook;
use anyhow::{bail, Result};
use std::collections::{HashMap, HashSet, VecDeque};

/// Playbook DAG with topological execution order
#[derive(Debug, Clone)]
pub struct PlaybookDag {
    /// Stages in topological execution order
    pub topo_order: Vec<String>,

    /// Map from stage → stages that must complete before it
    pub predecessors: HashMap<String, Vec<String>>,

    /// Map from stage → stages that depend on it
    pub successors: HashMap<String, Vec<String>>,
}

/// Build the execution DAG from a playbook
///
/// Algorithm:
/// 1. Build output_map: path → producing stage
/// 2. For each stage's deps, find the producing stage via output_map → add edge
/// 3. Add explicit `after` edges
/// 4. Check for cycles
/// 5. Topological sort
pub fn build_dag(playbook: &Playbook) -> Result<PlaybookDag> {
    let stage_names: Vec<String> = playbook.stages.keys().cloned().collect();

    // Initialize adjacency
    let mut predecessors: HashMap<String, Vec<String>> = HashMap::new();
    let mut successors: HashMap<String, Vec<String>> = HashMap::new();
    for name in &stage_names {
        predecessors.insert(name.clone(), Vec::new());
        successors.insert(name.clone(), Vec::new());
    }

    // Step 1: Build output_map (path → producing stage name)
    let mut output_map: HashMap<&str, &str> = HashMap::new();
    for (name, stage) in &playbook.stages {
        for out in &stage.outs {
            if let Some(existing) = output_map.insert(&out.path, name) {
                bail!(
                    "output path '{}' is produced by both '{}' and '{}'",
                    out.path,
                    existing,
                    name
                );
            }
        }
    }

    // Step 2: Implicit data dependency edges
    for (consumer_name, stage) in &playbook.stages {
        for dep in &stage.deps {
            if let Some(&producer_name) = output_map.get(dep.path.as_str()) {
                if producer_name != consumer_name {
                    add_edge(&mut predecessors, &mut successors, producer_name, consumer_name);
                }
            }
            // deps referencing external files (not produced by any stage) are fine
        }
    }

    // Step 3: Explicit `after` edges
    for (name, stage) in &playbook.stages {
        for after_name in &stage.after {
            add_edge(&mut predecessors, &mut successors, after_name, name);
        }
    }

    // Step 4: Cycle detection + topological sort (Kahn's algorithm)
    let topo_order = kahn_toposort(&stage_names, &predecessors, &successors)?;

    Ok(PlaybookDag { topo_order, predecessors, successors })
}

fn add_edge(
    predecessors: &mut HashMap<String, Vec<String>>,
    successors: &mut HashMap<String, Vec<String>>,
    from: &str,
    to: &str,
) {
    let preds = predecessors.entry(to.to_string()).or_default();
    if !preds.contains(&from.to_string()) {
        preds.push(from.to_string());
    }
    let succs = successors.entry(from.to_string()).or_default();
    if !succs.contains(&to.to_string()) {
        succs.push(to.to_string());
    }
}

/// Find nodes that become ready (in-degree drops to 0) after processing `node`.
fn collect_ready_successors(
    node: &str,
    names: &[String],
    predecessors: &HashMap<String, Vec<String>>,
    visited: &HashSet<String>,
    in_degree: &mut HashMap<&str, usize>,
) -> Vec<String> {
    let mut ready = Vec::new();
    for name in names {
        if visited.contains(name) {
            continue;
        }
        if let Some(preds) = predecessors.get(name.as_str()) {
            if preds.contains(&node.to_string()) {
                let deg = in_degree.get_mut(name.as_str()).expect("unexpected failure");
                *deg -= 1;
                if *deg == 0 {
                    ready.push(name.clone());
                }
            }
        }
    }
    ready.sort();
    ready
}

/// Kahn's topological sort with cycle detection
fn kahn_toposort(
    names: &[String],
    predecessors: &HashMap<String, Vec<String>>,
    _successors: &HashMap<String, Vec<String>>,
) -> Result<Vec<String>> {
    let mut in_degree: HashMap<&str, usize> = HashMap::new();
    for name in names {
        in_degree.insert(name, predecessors.get(name.as_str()).map_or(0, |p| p.len()));
    }

    let mut queue: VecDeque<String> = {
        let mut init: Vec<String> =
            names.iter().filter(|n| in_degree.get(n.as_str()) == Some(&0)).cloned().collect();
        init.sort();
        init.into()
    };

    let mut result = Vec::new();
    let mut visited: HashSet<String> = HashSet::new();

    while let Some(node) = queue.pop_front() {
        visited.insert(node.clone());
        result.push(node.clone());
        let next = collect_ready_successors(&node, names, predecessors, &visited, &mut in_degree);
        queue.extend(next);
    }

    if result.len() != names.len() {
        let cycle_stages: Vec<&str> =
            names.iter().filter(|n| !visited.contains(n.as_str())).map(|n| n.as_str()).collect();
        bail!("cycle detected in pipeline stages: {}", cycle_stages.join(" → "));
    }

    Ok(result)
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::playbook::parser::parse_playbook;

    #[test]
    fn test_PB002_linear_chain() {
        let yaml = r#"
version: "1.0"
name: chain
params: {}
targets: {}
stages:
  a:
    cmd: "echo a > /tmp/a.txt"
    deps: []
    outs:
      - path: /tmp/a.txt
  b:
    cmd: "cat /tmp/a.txt > /tmp/b.txt"
    deps:
      - path: /tmp/a.txt
    outs:
      - path: /tmp/b.txt
  c:
    cmd: "cat /tmp/b.txt > /tmp/c.txt"
    deps:
      - path: /tmp/b.txt
    outs:
      - path: /tmp/c.txt
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#;
        let pb = parse_playbook(yaml).expect("unexpected failure");
        let dag = build_dag(&pb).expect("unexpected failure");
        assert_eq!(dag.topo_order, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_PB002_parallel_stages() {
        let yaml = r#"
version: "1.0"
name: parallel
params: {}
targets: {}
stages:
  a:
    cmd: "echo a"
    deps: []
    outs:
      - path: /tmp/a.txt
  b:
    cmd: "echo b"
    deps: []
    outs:
      - path: /tmp/b.txt
  c:
    cmd: "echo c"
    deps: []
    outs:
      - path: /tmp/c.txt
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#;
        let pb = parse_playbook(yaml).expect("unexpected failure");
        let dag = build_dag(&pb).expect("unexpected failure");
        // All independent, alphabetical sort
        assert_eq!(dag.topo_order, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_PB002_diamond_dag() {
        let yaml = r#"
version: "1.0"
name: diamond
params: {}
targets: {}
stages:
  source:
    cmd: "echo src"
    deps: []
    outs:
      - path: /tmp/src.txt
  left:
    cmd: "echo left"
    deps:
      - path: /tmp/src.txt
    outs:
      - path: /tmp/left.txt
  right:
    cmd: "echo right"
    deps:
      - path: /tmp/src.txt
    outs:
      - path: /tmp/right.txt
  sink:
    cmd: "echo sink"
    deps:
      - path: /tmp/left.txt
      - path: /tmp/right.txt
    outs:
      - path: /tmp/sink.txt
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#;
        let pb = parse_playbook(yaml).expect("unexpected failure");
        let dag = build_dag(&pb).expect("unexpected failure");
        // source must be first, sink must be last
        assert_eq!(dag.topo_order[0], "source");
        assert_eq!(dag.topo_order[3], "sink");
        // left and right can be in either order
        let middle: HashSet<&str> = dag.topo_order[1..3].iter().map(|s| s.as_str()).collect();
        assert!(middle.contains("left"));
        assert!(middle.contains("right"));
    }

    #[test]
    fn test_PB002_cycle_detection() {
        let yaml = r#"
version: "1.0"
name: cycle
params: {}
targets: {}
stages:
  a:
    cmd: "echo a"
    deps:
      - path: /tmp/b.txt
    outs:
      - path: /tmp/a.txt
  b:
    cmd: "echo b"
    deps:
      - path: /tmp/a.txt
    outs:
      - path: /tmp/b.txt
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#;
        let pb = parse_playbook(yaml).expect("unexpected failure");
        let err = build_dag(&pb).unwrap_err();
        assert!(err.to_string().contains("cycle"));
    }

    #[test]
    fn test_PB002_after_edges() {
        let yaml = r#"
version: "1.0"
name: after
params: {}
targets: {}
stages:
  setup:
    cmd: "echo setup"
    deps: []
    outs:
      - path: /tmp/setup.txt
  work:
    cmd: "echo work"
    deps: []
    outs:
      - path: /tmp/work.txt
    after:
      - setup
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#;
        let pb = parse_playbook(yaml).expect("unexpected failure");
        let dag = build_dag(&pb).expect("unexpected failure");
        assert_eq!(dag.topo_order, vec!["setup", "work"]);
        assert_eq!(dag.predecessors["work"], vec!["setup"]);
        assert_eq!(dag.successors["setup"], vec!["work"]);
    }

    #[test]
    fn test_PB002_duplicate_output_path() {
        let yaml = r#"
version: "1.0"
name: dup
params: {}
targets: {}
stages:
  a:
    cmd: "echo a"
    deps: []
    outs:
      - path: /tmp/shared.txt
  b:
    cmd: "echo b"
    deps: []
    outs:
      - path: /tmp/shared.txt
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#;
        let pb = parse_playbook(yaml).expect("unexpected failure");
        let err = build_dag(&pb).unwrap_err();
        assert!(err.to_string().contains("produced by both"));
    }

    #[test]
    fn test_PB002_external_deps_no_edge() {
        let yaml = r#"
version: "1.0"
name: external
params: {}
targets: {}
stages:
  a:
    cmd: "echo a"
    deps:
      - path: /data/external.csv
    outs:
      - path: /tmp/a.txt
  b:
    cmd: "echo b"
    deps:
      - path: /data/another.csv
    outs:
      - path: /tmp/b.txt
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#;
        let pb = parse_playbook(yaml).expect("unexpected failure");
        let dag = build_dag(&pb).expect("unexpected failure");
        // Both are independent (deps are external)
        assert_eq!(dag.topo_order.len(), 2);
        assert!(dag.predecessors["a"].is_empty());
        assert!(dag.predecessors["b"].is_empty());
    }

    #[test]
    fn test_PB002_mixed_implicit_and_explicit() {
        let yaml = r#"
version: "1.0"
name: mixed
params: {}
targets: {}
stages:
  a:
    cmd: "echo a"
    deps: []
    outs:
      - path: /tmp/a.txt
  b:
    cmd: "echo b"
    deps:
      - path: /tmp/a.txt
    outs:
      - path: /tmp/b.txt
  c:
    cmd: "echo c"
    deps: []
    outs:
      - path: /tmp/c.txt
    after:
      - b
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#;
        let pb = parse_playbook(yaml).expect("unexpected failure");
        let dag = build_dag(&pb).expect("unexpected failure");
        assert_eq!(dag.topo_order, vec!["a", "b", "c"]);
    }
}
