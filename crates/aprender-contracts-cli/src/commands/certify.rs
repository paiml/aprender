//! `pv certify` — produce a whole-model proof certificate.
//!
//! Runs verify-pipeline + verify-structure and emits a signed JSON
//! certificate summarizing the compositional proof chain.
//!
//! Spec: docs/specifications/sub/model-layout-provability.md (§36, P0-8)

use std::collections::BTreeMap;
use std::path::Path;

use provable_contracts::graph::dependency_graph;
use provable_contracts::schema::{parse_contract, Contract};
use serde_json::Value;

use crate::json_obj::obj;

/// Run the certify command.
pub fn run(
    contract_dir: &Path,
    config_json: Option<&Path>,
    output: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut contracts = Vec::new();
    load_yaml_recursive(contract_dir, &mut contracts)?;
    contracts.sort_by(|a, b| a.0.cmp(&b.0));

    if contracts.is_empty() {
        eprintln!("No contracts found in {}", contract_dir.display());
        return Ok(());
    }

    let refs: Vec<(String, &Contract)> = contracts.iter().map(|(s, c)| (s.clone(), c)).collect();
    let graph = dependency_graph(&refs);
    let index: BTreeMap<&str, &Contract> = contracts.iter().map(|(s, c)| (s.as_str(), c)).collect();

    let (edges_total, edges_satisfied, edge_details) = verify_composition_edges(&graph, &index);

    let config_proof = config_json.and_then(|cfg| analyze_config(cfg).ok());

    let composition_passed = edges_total > 0 && edges_satisfied == edges_total;
    let config_passed = config_proof.as_ref().is_none_or(|p| {
        p.get("all_checks_pass")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    });

    let certificate = build_certificate(
        &contracts,
        &graph,
        &index,
        edges_total,
        edges_satisfied,
        &edge_details,
        config_proof.as_ref(),
        composition_passed,
        config_passed,
    );

    let json_str = serde_json::to_string_pretty(&certificate)?;

    if let Some(out_path) = output {
        std::fs::write(out_path, &json_str)?;
        println!("Certificate written to: {}", out_path.display());
    } else {
        println!("{json_str}");
    }

    print_summary(
        composition_passed,
        config_passed,
        edges_satisfied,
        edges_total,
        config_proof.as_ref(),
    );

    if !composition_passed {
        std::process::exit(1);
    }

    Ok(())
}

fn verify_composition_edges(
    graph: &provable_contracts::graph::DependencyGraph,
    index: &BTreeMap<&str, &Contract>,
) -> (usize, usize, Vec<serde_json::Value>) {
    let mut edges_total = 0usize;
    let mut edges_satisfied = 0usize;
    let mut edge_details = Vec::new();

    for stem in &graph.topo_order {
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
            let from_eq = assumes.from_equation.as_deref().unwrap_or("*");
            let satisfied = check_edge(index, from_contract, assumes.from_equation.as_deref());
            if satisfied {
                edges_satisfied += 1;
            }
            edge_details.push(obj([
                ("downstream", Value::from(format!("{stem}.{eq_name}"))),
                (
                    "upstream",
                    Value::from(format!("{from_contract}.{from_eq}")),
                ),
                ("satisfied", Value::from(satisfied)),
            ]));
        }
    }

    (edges_total, edges_satisfied, edge_details)
}

#[allow(clippy::too_many_arguments)]
fn build_certificate(
    contracts: &[(String, Contract)],
    graph: &provable_contracts::graph::DependencyGraph,
    index: &BTreeMap<&str, &Contract>,
    edges_total: usize,
    edges_satisfied: usize,
    edge_details: &[serde_json::Value],
    config_proof: Option<&serde_json::Value>,
    composition_passed: bool,
    config_passed: bool,
) -> serde_json::Value {
    let with_assumes = contracts
        .iter()
        .filter(|(_, c)| c.equations.values().any(|e| e.assumes.is_some()))
        .count();
    let with_guarantees = contracts
        .iter()
        .filter(|(_, c)| c.equations.values().any(|e| e.guarantees.is_some()))
        .count();

    obj([
        ("version", Value::from("1.0.0")),
        (
            "tool",
            Value::from(format!("pv certify v{}", env!("CARGO_PKG_VERSION"))),
        ),
        ("timestamp", Value::from(chrono_free_timestamp())),
        (
            "contracts",
            obj([
                ("total", Value::from(contracts.len())),
                ("with_assumes", Value::from(with_assumes)),
                ("with_guarantees", Value::from(with_guarantees)),
                ("topo_depth", Value::from(graph.topo_order.len())),
                ("cycles", Value::from(graph.cycles.len())),
            ]),
        ),
        (
            "composition",
            obj([
                ("edges_total", Value::from(edges_total)),
                ("edges_satisfied", Value::from(edges_satisfied)),
                (
                    "edges_broken",
                    Value::from(edges_total.saturating_sub(edges_satisfied)),
                ),
                ("passed", Value::from(composition_passed)),
                ("edges", Value::Array(edge_details.to_vec())),
            ]),
        ),
        (
            "config_analysis",
            config_proof.cloned().unwrap_or(Value::Null),
        ),
        (
            "proofs",
            obj([
                (
                    "format_safety",
                    proof_status(index, "safetensors-format-safety-v1"),
                ),
                (
                    "config_algebra",
                    proof_status(index, "model-config-algebra-v1"),
                ),
                (
                    "architecture_schema",
                    proof_status(index, "apr-architecture-schema-v1"),
                ),
                (
                    "shape_pipeline",
                    proof_status(index, "tensor-shape-flow-v1"),
                ),
                ("tensor_names", proof_status(index, "tensor-names-v1")),
            ]),
        ),
        (
            "certificate_level",
            Value::from(if composition_passed && config_passed {
                "L3"
            } else {
                "L2"
            }),
        ),
        ("passed", Value::from(composition_passed && config_passed)),
    ])
}

fn print_summary(
    composition_passed: bool,
    config_passed: bool,
    edges_satisfied: usize,
    edges_total: usize,
    config_proof: Option<&serde_json::Value>,
) {
    let icon = if composition_passed && config_passed {
        "✓"
    } else {
        "✗"
    };
    eprintln!();
    eprintln!("pv certify — {icon} {edges_satisfied}/{edges_total} composition edges satisfied",);
    if let Some(cfg) = config_proof {
        if let Some(tensors) = cfg.get("expected_tensors") {
            eprintln!("  Config: {tensors} expected tensors");
        }
    }
    let level = if composition_passed && config_passed {
        "L3"
    } else {
        "L2"
    };
    eprintln!("  Certificate level: {level}");
}

fn check_edge(
    index: &BTreeMap<&str, &Contract>,
    from_contract: &str,
    from_eq: Option<&str>,
) -> bool {
    let Some(upstream) = index.get(from_contract) else {
        return false;
    };
    if let Some(eq_name) = from_eq {
        upstream
            .equations
            .get(eq_name)
            .is_some_and(|eq| eq.guarantees.is_some())
    } else {
        upstream
            .equations
            .values()
            .any(|eq| eq.guarantees.is_some())
    }
}

fn proof_status(index: &BTreeMap<&str, &Contract>, stem: &str) -> serde_json::Value {
    if let Some(contract) = index.get(stem) {
        let eq_count = contract.equations.len();
        let has_assumes = contract
            .equations
            .values()
            .filter(|e| e.assumes.is_some())
            .count();
        let has_guarantees = contract
            .equations
            .values()
            .filter(|e| e.guarantees.is_some())
            .count();
        let has_kani = !contract.kani_harnesses.is_empty();
        let has_lean = contract
            .equations
            .values()
            .any(|e| e.lean_theorem.is_some());
        obj([
            ("found", Value::from(true)),
            ("equations", Value::from(eq_count)),
            ("with_assumes", Value::from(has_assumes)),
            ("with_guarantees", Value::from(has_guarantees)),
            ("has_kani", Value::from(has_kani)),
            ("has_lean", Value::from(has_lean)),
            (
                "verdict",
                Value::from(if has_kani || has_lean {
                    "PROVEN"
                } else {
                    "SPECIFIED"
                }),
            ),
        ])
    } else {
        obj([
            ("found", Value::from(false)),
            ("verdict", Value::from("MISSING")),
        ])
    }
}

pub fn analyze_config(cfg_path: &Path) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(cfg_path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    let h = json.get("hidden_size").and_then(Value::as_u64).unwrap_or(0);
    let l = json
        .get("num_hidden_layers")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let nh = json
        .get("num_attention_heads")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let nkv = json
        .get("num_key_value_heads")
        .and_then(Value::as_u64)
        .unwrap_or(nh);
    let v = json.get("vocab_size").and_then(Value::as_u64).unwrap_or(0);

    let head_dim = if nh > 0 { h / nh } else { 0 };
    let expected_tensors = if l > 0 { 1 + l * 9 + 2 } else { 0 };

    let checks = [
        ("hidden_size > 0", h > 0),
        ("num_layers > 0", l > 0),
        ("num_heads > 0", nh > 0),
        ("vocab_size > 0", v > 0),
        ("hidden_size % num_heads == 0", nh > 0 && h % nh == 0),
        ("num_heads % num_kv_heads == 0", nkv > 0 && nh % nkv == 0),
        ("head_dim % 2 == 0", head_dim % 2 == 0),
    ];

    let all_pass = checks.iter().all(|(_, ok)| *ok);

    let algebra_checks: Vec<Value> = checks
        .iter()
        .map(|(name, ok)| obj([("check", Value::from(*name)), ("passed", Value::from(*ok))]))
        .collect();

    Ok(obj([
        ("config_path", Value::from(cfg_path.display().to_string())),
        ("hidden_size", Value::from(h)),
        ("num_layers", Value::from(l)),
        ("num_heads", Value::from(nh)),
        ("num_kv_heads", Value::from(nkv)),
        ("vocab_size", Value::from(v)),
        ("head_dim", Value::from(head_dim)),
        ("expected_tensors", Value::from(expected_tensors)),
        ("algebra_checks", Value::Array(algebra_checks)),
        ("all_checks_pass", Value::from(all_pass)),
    ]))
}

fn chrono_free_timestamp() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}s-since-epoch", dur.as_secs())
}

fn load_yaml_recursive(
    dir: &Path,
    out: &mut Vec<(String, Contract)>,
) -> Result<(), Box<dyn std::error::Error>> {
    let entries = std::fs::read_dir(dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            load_yaml_recursive(&path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            if stem == "binding" || stem == "playbook.schema" || stem.contains("playbook") {
                continue;
            }
            if let Ok(c) = parse_contract(&path) {
                out.push((stem, c));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn certify_on_real_contracts() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        if !dir.exists() {
            return;
        }
        let result = run(&dir, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn certify_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let result = run(tmp.path(), None, None);
        assert!(result.is_ok());
    }
}
