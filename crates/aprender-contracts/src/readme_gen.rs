//! Deterministic README.md and CI workflow generation for consumer projects.
//!
//! `pv generate --readme` produces a CONTRACT-README.md documenting
//! contract coverage. `pv generate --ci` produces a GitHub Actions
//! workflow for contract validation.
//!
//! **Determinism invariant:** Same inputs → identical byte-for-byte output.
//! No timestamps, git hashes, random IDs, or environment-dependent values.

use std::fmt::Write;

use crate::binding::BindingRegistry;
use crate::coverage::{coverage_report, CoverageReport};
use crate::proof_status::compute_proof_level;
use crate::schema::Contract;

/// Generate a deterministic CONTRACT-README.md for a consumer project.
///
/// Documents which contracts are bound, their proof levels, and gaps.
pub fn generate_readme(contracts: &[(String, &Contract)], binding: &BindingRegistry) -> String {
    let mut out = String::with_capacity(2048);

    let report = coverage_report(contracts, Some(binding));

    let _ = writeln!(out, "# Contract Coverage ({})\n", binding.target_crate);
    write_coverage_summary(&mut out, &report);
    write_binding_table(&mut out, contracts, binding);
    write_verification_summary(&mut out, contracts, binding);
    write_gaps(&mut out, contracts, binding);
    write_build_instructions(&mut out);

    out
}

fn write_coverage_summary(out: &mut String, report: &CoverageReport) {
    let total = report.totals.contracts;
    let with_tests = report
        .contracts
        .iter()
        .filter(|c| c.falsification_covered > 0)
        .count();
    let pct = (with_tests * 100).checked_div(total).unwrap_or(0);

    let _ = writeln!(
        out,
        "**Coverage:** {with_tests}/{total} contracts with tests ({pct}%)"
    );
    let _ = writeln!(
        out,
        "**Obligations:** {} | **Falsification:** {} | **Kani:** {}\n",
        report.totals.obligations, report.totals.falsification_tests, report.totals.kani_harnesses
    );
}

fn write_binding_table(
    out: &mut String,
    contracts: &[(String, &Contract)],
    binding: &BindingRegistry,
) {
    let _ = writeln!(out, "## Bound Contracts\n");
    let _ = writeln!(
        out,
        "| Contract | Equations | Obligations | Level | Status |"
    );
    let _ = writeln!(
        out,
        "|----------|-----------|-------------|-------|--------|"
    );

    for (stem, contract) in contracts {
        let contract_file = format!("{stem}.yaml");
        let bound_eqs: Vec<_> = binding
            .bindings
            .iter()
            .filter(|b| b.contract == contract_file)
            .collect();

        if bound_eqs.is_empty() {
            continue;
        }

        let level = compute_proof_level(contract, None);
        let status = if bound_eqs
            .iter()
            .all(|b| b.status == crate::binding::ImplStatus::Implemented)
        {
            "implemented"
        } else if bound_eqs
            .iter()
            .any(|b| b.status == crate::binding::ImplStatus::Implemented)
        {
            "partial"
        } else {
            "not_implemented"
        };

        let _ = writeln!(
            out,
            "| {stem} | {} | {} | {level} | {status} |",
            contract.equations.len(),
            contract.proof_obligations.len()
        );
    }
    let _ = writeln!(out);
}

fn write_verification_summary(
    out: &mut String,
    contracts: &[(String, &Contract)],
    _binding: &BindingRegistry,
) {
    let mut l1 = 0usize;
    let mut l2 = 0usize;
    let mut l3 = 0usize;
    let mut l4 = 0usize;
    let mut l5 = 0usize;

    for (_stem, contract) in contracts {
        match compute_proof_level(contract, None).to_string().as_str() {
            "L5" => l5 += 1,
            "L4" => l4 += 1,
            "L3" => l3 += 1,
            "L2" => l2 += 1,
            _ => l1 += 1,
        }
    }

    let _ = writeln!(out, "## Verification Ladder\n");
    let _ = writeln!(out, "| Level | Count | Method |");
    let _ = writeln!(out, "|-------|-------|--------|");
    let _ = writeln!(out, "| L5 | {l5} | Lean 4 theorem |");
    let _ = writeln!(out, "| L4 | {l4} | Kani BMC |");
    let _ = writeln!(out, "| L3 | {l3} | Kani + probar |");
    let _ = writeln!(out, "| L2 | {l2} | Falsification |");
    let _ = writeln!(out, "| L1 | {l1} | Type system |");
    let _ = writeln!(out);
}

fn write_gaps(out: &mut String, contracts: &[(String, &Contract)], binding: &BindingRegistry) {
    let mut gaps = Vec::new();

    for (stem, contract) in contracts {
        let contract_file = format!("{stem}.yaml");
        for eq_name in contract.equations.keys() {
            let bound = binding
                .bindings
                .iter()
                .any(|b| b.contract == contract_file && b.equation == *eq_name);
            if !bound {
                gaps.push((stem.as_str(), eq_name.as_str()));
            }
        }
    }

    if gaps.is_empty() {
        return;
    }

    let _ = writeln!(out, "## Unbound Equations\n");
    let _ = writeln!(out, "| Contract | Equation |");
    let _ = writeln!(out, "|----------|----------|");
    for (stem, eq) in &gaps {
        let _ = writeln!(out, "| {stem} | {eq} |");
    }
    let _ = writeln!(out);
}

fn write_build_instructions(out: &mut String) {
    let _ = writeln!(out, "## Build Integration\n");
    let _ = writeln!(out, "```toml");
    let _ = writeln!(out, "# Cargo.toml");
    let _ = writeln!(out, "[build-dependencies]");
    let _ = writeln!(out, "provable-contracts = \"0.1\"");
    let _ = writeln!(out);
    let _ = writeln!(out, "[dependencies]");
    let _ = writeln!(out, "provable-contracts-macros = \"0.1\"");
    let _ = writeln!(out, "```\n");
    let _ = writeln!(
        out,
        "See [provable-contracts](https://github.com/paiml/provable-contracts) for setup."
    );
}

/// Generate a deterministic GitHub Actions workflow for contract validation.
pub fn generate_ci_workflow(project_name: &str) -> String {
    let mut out = String::with_capacity(1024);

    let _ = writeln!(out, "# Auto-generated by pv generate --ci");
    let _ = writeln!(
        out,
        "# Deterministic output — regenerate with: pv generate --ci"
    );
    let _ = writeln!(out, "name: Contract Validation\n");
    let _ = writeln!(out, "on:");
    let _ = writeln!(out, "  push:");
    let _ = writeln!(out, "    branches: [main]");
    let _ = writeln!(out, "  pull_request:");
    let _ = writeln!(out, "    branches: [main]\n");
    let _ = writeln!(out, "jobs:");
    let _ = writeln!(out, "  contracts:");
    let _ = writeln!(out, "    name: Contract Validation");
    let _ = writeln!(out, "    runs-on: ubuntu-latest");
    let _ = writeln!(out, "    steps:");
    let _ = writeln!(out, "      - uses: actions/checkout@v4");
    let _ = writeln!(out, "      - uses: actions/checkout@v4");
    let _ = writeln!(out, "        with:");
    let _ = writeln!(out, "          repository: paiml/provable-contracts");
    let _ = writeln!(out, "          path: provable-contracts\n");
    let _ = writeln!(out, "      - uses: dtolnay/rust-toolchain@stable");
    let _ = writeln!(out, "      - uses: Swatinem/rust-cache@v2\n");
    let _ = writeln!(out, "      - name: Install pv");
    let _ = writeln!(
        out,
        "        run: cargo install --path provable-contracts/crates/provable-contracts-cli\n"
    );
    let _ = writeln!(out, "      - name: Validate contracts");
    let _ = writeln!(out, "        run: pv lint provable-contracts/contracts/\n");
    let _ = writeln!(out, "      - name: Verify bindings");
    let _ = writeln!(
        out,
        "        run: pv lint provable-contracts/contracts/ --binding provable-contracts/contracts/{project_name}/binding.yaml"
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::parse_contract_str;

    fn sample_contract() -> Contract {
        parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "test"
falsification_tests:
  - id: F-001
    rule: "test"
    prediction: "test"
    if_fails: "test"
kani_harnesses:
  - id: K-001
    obligation: "test"
    bound: 8
qa_gate:
  id: QA-001
  name: "Test"
  pass_criteria: "all pass"
"#,
        )
        .unwrap()
    }

    #[test]
    fn readme_is_deterministic() {
        let c = sample_contract();
        let contracts = vec![("test-v1".to_string(), &c)];
        let binding = BindingRegistry {
            version: "1.0.0".to_string(),
            target_crate: "test-crate".to_string(),
            critical_path: vec![],
            bindings: vec![crate::binding::KernelBinding {
                contract: "test-v1.yaml".to_string(),
                equation: "f".to_string(),
                module_path: None,
                function: None,
                signature: None,
                status: crate::binding::ImplStatus::Implemented,
                notes: None,
            }],
        };

        let out1 = generate_readme(&contracts, &binding);
        let out2 = generate_readme(&contracts, &binding);
        assert_eq!(out1, out2, "README output must be deterministic");
        assert!(out1.contains("# Contract Coverage (test-crate)"));
        assert!(out1.contains("test-v1"));
        assert!(out1.contains("implemented"));
    }

    #[test]
    fn ci_workflow_is_deterministic() {
        let out1 = generate_ci_workflow("aprender");
        let out2 = generate_ci_workflow("aprender");
        assert_eq!(out1, out2, "CI workflow must be deterministic");
        assert!(out1.contains("Contract Validation"));
        assert!(out1.contains("aprender/binding.yaml"));
    }
}
