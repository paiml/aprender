//! Upstream ancestry of a lineage node (EXT-001 EXT-07, FALSIFY-EXT-006).
//!
//! Every lineage edge points upstream -> downstream (`from_id` is the input),
//! so the ancestors of a node are found by following edges *into* it. A
//! diamond (a merge of two siblings of one base) is a DAG and is allowed; a
//! cycle is a corrupt graph and is refused rather than truncated.

use crate::error::{PachaError, Result};
use serde::Serialize;
use std::collections::HashSet;

/// One recorded lineage edge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StoredEdge {
    /// Upstream node (the input).
    pub from_id: String,
    /// Downstream node.
    pub to_id: String,
    /// Edge type (`base`, `dataset`, `produced`, `merged_from`, ...).
    pub edge_type: String,
    /// Edge metadata as recorded, if any.
    pub metadata: Option<serde_json::Value>,
}

/// Every upstream node of `root` and the edges between them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Ancestry {
    /// The node asked about.
    pub root: String,
    /// `root` first, then each ancestor once, in discovery order.
    pub nodes: Vec<String>,
    /// Each edge on a path into `root` once, in discovery order.
    pub edges: Vec<StoredEdge>,
}

/// Walk the ancestry of `root`, reading the edges into a node with `into`.
///
/// # Errors
///
/// [`PachaError::Validation`] if the graph above `root` has a cycle, or any
/// error `into` returns.
pub fn walk(root: &str, into: &mut dyn FnMut(&str) -> Result<Vec<StoredEdge>>) -> Result<Ancestry> {
    let mut out =
        Ancestry { root: root.to_string(), nodes: vec![root.to_string()], edges: Vec::new() };
    let mut done = HashSet::new();
    let mut path = vec![root.to_string()];
    visit(root, into, &mut done, &mut path, &mut out)?;
    Ok(out)
}

fn visit(
    node: &str,
    into: &mut dyn FnMut(&str) -> Result<Vec<StoredEdge>>,
    done: &mut HashSet<String>,
    path: &mut Vec<String>,
    out: &mut Ancestry,
) -> Result<()> {
    for edge in into(node)? {
        let parent = edge.from_id.clone();
        if path.contains(&parent) {
            path.push(parent);
            return Err(PachaError::Validation(format!(
                "lineage cycle: {}",
                path.iter().rev().cloned().collect::<Vec<_>>().join(" -> ")
            )));
        }
        out.edges.push(edge);
        if done.contains(&parent) {
            continue;
        }
        if !out.nodes.contains(&parent) {
            out.nodes.push(parent.clone());
        }
        path.push(parent.clone());
        visit(&parent, into, done, path, out)?;
        path.pop();
        done.insert(parent);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph<'a>(
        edges: &'a [(&'a str, &'a str)],
    ) -> impl FnMut(&str) -> Result<Vec<StoredEdge>> + 'a {
        move |to: &str| {
            Ok(edges
                .iter()
                .filter(|(_, t)| *t == to)
                .map(|(f, t)| StoredEdge {
                    from_id: (*f).into(),
                    to_id: (*t).into(),
                    edge_type: "e".into(),
                    metadata: None,
                })
                .collect())
        }
    }

    /// FALSIFY-EXT-006 (walk): a 3-hop chain gives 3 nodes and 2 edges.
    #[test]
    fn falsify_ext_006_three_hop_chain() {
        let a = walk("c", &mut graph(&[("a", "b"), ("b", "c")])).expect("walk");
        assert_eq!(a.nodes, ["c", "b", "a"]);
        assert_eq!(a.edges.len(), 2);
    }

    /// FALSIFY-EXT-006 (walk): a cycle is refused and named, not truncated.
    #[test]
    fn falsify_ext_006_cycle_refused() {
        let err = walk("c", &mut graph(&[("a", "b"), ("b", "c"), ("c", "a")]))
            .expect_err("cycle must be refused");
        assert!(err.to_string().contains("lineage cycle"), "{err}");
        let self_loop = walk("x", &mut graph(&[("x", "x")])).expect_err("self loop");
        assert!(self_loop.to_string().contains("x -> x"), "{self_loop}");
    }

    /// A diamond (merge of two siblings of one base) is a DAG: each node and
    /// each edge appears once, including the edges above the shared base.
    #[test]
    fn a_diamond_is_not_a_cycle() {
        let a = walk(
            "m",
            &mut graph(&[("s1", "m"), ("s2", "m"), ("b", "s1"), ("b", "s2"), ("r", "b")]),
        )
        .expect("walk");
        assert_eq!(a.nodes, ["m", "s1", "b", "r", "s2"]);
        assert_eq!(a.edges.len(), 5);
    }
}
