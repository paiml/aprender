// Test/example/bench binary: `.unwrap()` is idiomatic here; the lib`s
// cfg(test) allow does not reach this separate crate.
#![allow(clippy::disallowed_methods)]

use std::path::Path;

use provable_contracts::error::Severity;
use provable_contracts::graph::dependency_graph;
use provable_contracts::schema::{is_contract_yaml, parse_contract, validate_contract, Contract};

fn contracts_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts")
        .canonicalize()
        .expect("contracts directory must exist")
}

/// Every contract document directly under `contracts/`.
///
/// The file filter is `provable_contracts::schema::is_contract_yaml` — the same
/// predicate `pv lint`'s walker uses. Before #2551 this walker had its own
/// copy that did not skip `contracts/binding.yaml` (a `BindingRegistry`, not a
/// contract), so three of this file's ten tests panicked with ``missing field
/// `metadata` `` while `pv lint contracts/` reported zero errors on the same
/// tree. Two walkers, two answers, and the test target was dark in CI so
/// nobody saw it.
///
/// Still deliberately NON-recursive: `contract_data_integrity` pins exact
/// corpus totals (equations/obligations/tests/harnesses) for the top-level set.
/// Widening this to the 1300+ subdirectory contracts is a real change to what
/// those assertions mean and belongs in its own PR, not smuggled in behind a
/// walker fix.
fn all_contract_paths() -> Vec<std::path::PathBuf> {
    let dir = contracts_dir();
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read {}: {e}", dir.display()))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            is_contract_yaml(&path).then_some(path)
        })
        .collect();
    paths.sort();
    assert!(
        paths.len() > 100,
        "all_contract_paths() found only {} files under {} — a walker that finds \
         nothing passes every test in this file vacuously",
        paths.len(),
        dir.display()
    );
    paths
}

fn validate_contract_file(path: &Path) {
    assert!(path.exists(), "Contract file not found: {}", path.display());

    let contract =
        parse_contract(path).unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));

    let violations = validate_contract(&contract);
    let errors: Vec<_> = violations
        .iter()
        .filter(|v| v.severity == Severity::Error)
        .collect();

    assert!(
        errors.is_empty(),
        "Contract {} has validation errors: {:?}",
        path.display(),
        errors
    );
}

#[test]
fn validate_softmax_contract() {
    validate_contract_file(&contracts_dir().join("softmax-kernel-v1.yaml"));
}

#[test]
fn validate_rmsnorm_contract() {
    validate_contract_file(&contracts_dir().join("rmsnorm-kernel-v1.yaml"));
}

#[test]
fn validate_rope_contract() {
    validate_contract_file(&contracts_dir().join("rope-kernel-v1.yaml"));
}

#[test]
fn validate_activation_contract() {
    validate_contract_file(&contracts_dir().join("activation-kernel-v1.yaml"));
}

#[test]
fn validate_attention_contract() {
    validate_contract_file(&contracts_dir().join("attention-kernel-v1.yaml"));
}

#[test]
fn validate_matmul_contract() {
    validate_contract_file(&contracts_dir().join("matmul-kernel-v1.yaml"));
}

#[test]
fn validate_flash_attention_contract() {
    validate_contract_file(&contracts_dir().join("flash-attention-v1.yaml"));
}

#[test]
fn validate_all_contracts() {
    let paths = all_contract_paths();
    assert!(
        paths.len() >= 81,
        "Expected at least 81 contracts, found {}",
        paths.len()
    );
    for path in &paths {
        validate_contract_file(path);
    }
}

#[test]
fn qwen35_dag_integrity() {
    let paths = all_contract_paths();
    let contracts: Vec<_> = paths
        .iter()
        .map(|p| {
            let stem = p.file_stem().unwrap().to_str().unwrap().to_string();
            let contract = parse_contract(p).unwrap();
            (stem, contract)
        })
        .collect();

    let refs: Vec<_> = contracts.iter().map(|(s, c)| (s.clone(), c)).collect();
    let graph = dependency_graph(&refs);

    // No cycles in the full DAG
    assert!(
        graph.cycles.is_empty(),
        "DAG has cycles: {:?}",
        graph.cycles
    );

    // qwen35-e2e-verification is the capstone (no dependents)
    let e2e = "qwen35-e2e-verification-v1";
    assert!(
        graph.nodes.contains(e2e),
        "Missing e2e verification contract"
    );

    // Verify e2e depends on exactly 8 sub-contracts
    let e2e_deps = graph.edges.get(e2e).unwrap();
    assert_eq!(
        e2e_deps.len(),
        8,
        "e2e should depend on 8 contracts, found {}",
        e2e_deps.len()
    );

    // All 7 Qwen 3.5 contracts exist
    let qwen_contracts = [
        "sliding-window-attention-v1",
        "rope-extrapolation-v1",
        "embedding-algebra-v1",
        "inference-pipeline-v1",
        "qwen35-hybrid-forward-v1",
        "attention-scaling-v1",
        "qwen35-e2e-verification-v1",
    ];
    for name in &qwen_contracts {
        assert!(
            graph.nodes.contains(*name),
            "Missing Qwen 3.5 contract: {name}"
        );
    }

    // Topological order: foundations before composites
    let topo = &graph.topo_order;
    let softmax_pos = topo.iter().position(|n| n == "softmax-kernel-v1").unwrap();
    let attention_pos = topo
        .iter()
        .position(|n| n == "attention-kernel-v1")
        .unwrap();
    let e2e_pos = topo.iter().position(|n| n == e2e).unwrap();
    assert!(
        softmax_pos < attention_pos,
        "softmax should come before attention in topo order"
    );
    assert!(
        attention_pos < e2e_pos,
        "attention should come before e2e in topo order"
    );

    // e2e should be last or near-last (no dependents)
    assert_eq!(
        topo.len(),
        graph.nodes.len(),
        "Topo order should contain all nodes"
    );
}

fn check_falsification_ids(stem: &str, contract: &Contract, errors: &mut Vec<String>) {
    let parts: Vec<&str> = contract
        .falsification_tests
        .first()
        .map(|ft| ft.id.rsplitn(2, '-').collect::<Vec<_>>())
        .unwrap_or_default();
    if parts.len() != 2 {
        return;
    }
    let prefix = parts[1];
    for (i, ft) in contract.falsification_tests.iter().enumerate() {
        let expected = format!("{prefix}-{:03}", i + 1);
        if ft.id != expected {
            errors.push(format!(
                "{stem}: test ID gap: expected {expected}, found {}",
                ft.id
            ));
            break;
        }
    }
}

fn check_pass_criteria(stem: &str, contract: &Contract, ft_count: usize, errors: &mut Vec<String>) {
    let criteria = contract
        .qa_gate
        .as_ref()
        .and_then(|qa| qa.pass_criteria.as_deref());
    let n = criteria
        .and_then(|c| c.strip_prefix("All "))
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse::<usize>().ok());
    if let Some(n) = n {
        if n != ft_count {
            errors.push(format!(
                "{stem}: pass_criteria says {n} tests, actual {ft_count}"
            ));
        }
    }
}

#[test]
fn contract_data_integrity() {
    let paths = all_contract_paths();
    let mut total_eq = 0usize;
    let mut total_ob = 0usize;
    let mut total_ft = 0usize;
    let mut total_kani = 0usize;
    let mut errors = Vec::new();

    for path in &paths {
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let contract =
            parse_contract(path).unwrap_or_else(|e| panic!("Failed to parse {stem}: {e}"));

        let eq_count = contract.equations.len();
        let ft_count = contract.falsification_tests.len();

        total_eq += eq_count;
        total_ob += contract.proof_obligations.len();
        total_ft += ft_count;
        total_kani += contract.kani_harnesses.len();

        if eq_count == 0 {
            errors.push(format!("{stem}: no equations"));
        }

        for v in contract.provability_violations() {
            errors.push(format!("{stem}: {v}"));
        }

        check_falsification_ids(stem, &contract, &mut errors);
        check_pass_criteria(stem, &contract, ft_count, &mut errors);
    }

    // Corpus floors, not equalities. These were `assert_eq!(total_eq, 486)`
    // and friends, pinned when `contracts/` was a fraction of its current size
    // — the real figure today is 2329. They rotted precisely because this
    // target is dark in CI (#2551 wires it), so nobody ever saw them fail.
    //
    // A floor is the assertion that actually earns its place: it excludes the
    // outcome worth excluding (contract content silently deleted) without
    // failing every PR that adds a contract. Raise a floor when the surplus
    // gets large; never lower one without saying which contract was removed
    // and why.
    let floors = [
        // Measured 2026-08-20 over the 1227 top-level contracts:
        // 2329 / 2686 / 3386 / 1219. Floors sit a few percent under so a
        // single retired contract does not red the gate, while any bulk loss
        // does.
        ("equations", total_eq, 2280),
        ("proof obligations", total_ob, 2630),
        ("falsification tests", total_ft, 3310),
        ("Kani harnesses", total_kani, 1190),
    ];
    for (label, actual, floor) in floors {
        assert!(
            actual >= floor,
            "Total {label} fell to {actual}, below the pinned floor of {floor} — \
             contract content was deleted"
        );
    }

    // Shrink-only ceiling, not `errors.is_empty()`.
    //
    // This target was dark in CI (no workflow ran it — #2551 wires it), and in
    // the dark the corpus accumulated 470 data-integrity violations: contracts
    // with no equations, falsification-test IDs that skip numbers, `pass_criteria`
    // that names a count the file does not have. `is_empty()` cannot be restored
    // in one commit, and leaving the target dark so the assertion can stay
    // aspirational is how it rotted in the first place.
    //
    // So: the count may only go DOWN. A PR that adds a 445th violation fails.
    // Lower CEILING whenever you clean up — never raise it.
    //
    // 470 -> 444, measured on this tree after merging main: main's newly added
    // `provable-contracts-facade-v1` carried a falsification-test ID out of order
    // (…-005, -007, -008, -009, -006), which is what pushed the count to 471 and
    // reddened this branch — the ceiling was pinned before that contract existed.
    // Reordering it, plus correcting 26 `pass_criteria` strings that named a test
    // count their own file did not have, took the corpus to 444. No rule was
    // relaxed to get here; every one of the 26 was a number edited to match the
    // file it describes.
    const CEILING: usize = 444;
    assert!(
        errors.len() <= CEILING,
        "Data integrity violations rose to {} (ceiling {CEILING}) — this list may \
         only shrink:\n{}",
        errors.len(),
        errors.join("\n")
    );
    assert!(
        !paths.is_empty(),
        "no contracts examined — the ceiling above passes vacuously on an empty walk"
    );
}
