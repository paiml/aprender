//! Every pre/postcondition the producer emits must be a Rust expression (#4371).
//!
//! `build.rs` emits `equations.*.{preconditions,postconditions}` from each
//! top-level `contracts/*.yaml` as `CONTRACT_<C>_<EQ>_{PRE,POST}_<i>`, and
//! `#[contract]` parses each value with `syn::parse_str::<Expr>` and refuses to
//! compile on one that is not an expression. So a prose condition is a latent
//! build break for the first function that binds its equation. This test walks
//! the same files with the same filter and parses every condition the same way,
//! naming each one that fails by contract/equation.
//!
//! Prose belongs in `prose_preconditions` / `prose_postconditions`, which no
//! producer emits, never in these two lists.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

#[derive(Deserialize, Default)]
struct ContractYaml {
    #[serde(default)]
    equations: BTreeMap<String, EquationYaml>,
}

#[derive(Deserialize, Default)]
struct EquationYaml {
    #[serde(default)]
    preconditions: Vec<String>,
    #[serde(default)]
    postconditions: Vec<String>,
}

/// The `build.rs` filter: a top-level `.yaml` whose name does not contain `binding`.
fn is_contract_yaml(path: &Path) -> bool {
    path.extension().and_then(|x| x.to_str()) == Some("yaml")
        && !path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().contains("binding"))
}

/// Each condition in one contract's text that `#[contract]` would reject, as
/// `"<contract>: <equation> <PRE|POST>_<i>: <condition>"`. A file that does not
/// deserialize emits nothing in `build.rs`, so it yields nothing here.
fn non_expression_conditions(contract: &str, yaml: &str) -> Vec<String> {
    let Ok(parsed) = serde_yaml::from_str::<ContractYaml>(yaml) else {
        return Vec::new();
    };
    let mut bad = Vec::new();
    for (eq_name, eq) in &parsed.equations {
        for (kind, list) in [("PRE", &eq.preconditions), ("POST", &eq.postconditions)] {
            for (i, cond) in list.iter().enumerate() {
                // A newline would also split the `cargo:rustc-env` line itself.
                if cond.contains('\n') || syn::parse_str::<syn::Expr>(cond).is_err() {
                    bad.push(format!("{contract}: {eq_name} {kind}_{i}: {cond}"));
                }
            }
        }
    }
    bad
}

#[test]
fn every_emitted_condition_is_a_rust_expression() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("contracts/ is readable")
        .flatten()
        .map(|e| e.path())
        .filter(|p| is_contract_yaml(p))
        .collect();
    files.sort();
    assert!(
        files.len() > 100,
        "found only {} contracts in {}; the walk is broken, not the contracts",
        files.len(),
        dir.display()
    );
    let mut bad = Vec::new();
    for path in &files {
        let name = path.file_name().expect("file name").to_string_lossy();
        let text = std::fs::read_to_string(path).expect("contract is readable");
        bad.extend(non_expression_conditions(&name, &text));
    }
    assert!(
        bad.is_empty(),
        "{} emitted condition(s) are not Rust expressions; `#[contract]` refuses \
         to compile on them. Rewrite each as an expression or move it to a field \
         the producer does not emit:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

#[test]
fn a_prose_condition_is_reported_by_contract_and_equation() {
    let yaml = "equations:\n  softmax:\n    preconditions:\n      - x.len() > 0\n      \
                - input must be finite\n    postconditions:\n      - 'out.iter().all(|v| v.is_finite())'\n";
    assert_eq!(
        non_expression_conditions("toy-v1.yaml", yaml),
        vec!["toy-v1.yaml: softmax PRE_1: input must be finite".to_string()]
    );
}

#[test]
fn a_multi_line_condition_is_reported() {
    let yaml = "equations:\n  e:\n    postconditions:\n      - |\n        a\n        && b\n";
    assert_eq!(non_expression_conditions("m.yaml", yaml).len(), 1);
}
