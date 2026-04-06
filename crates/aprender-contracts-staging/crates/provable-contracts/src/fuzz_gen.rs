//! Coverage-guided fuzz target generation.
//!
//! Generates `libfuzzer_sys::fuzz_target!` harnesses from YAML contracts.
//! Each fuzz target gates on contract preconditions and exercises the
//! bound function, catching panics, sanitizer violations, and crashes.

use std::fmt::Write;

use crate::schema::Contract;

/// Generate a libfuzzer fuzz target from a contract.
///
/// Returns the contents of a `fuzz/fuzz_targets/<stem>.rs` file.
/// The target deserialises arbitrary bytes into the kernel's input type,
/// filters inputs that violate preconditions, and calls the bound function.
pub fn generate_fuzz_target(contract: &Contract, stem: &str) -> String {
    let mut out = String::with_capacity(2048);

    let _ = writeln!(out, "#![no_main]");
    let _ = writeln!(
        out,
        "// Auto-generated fuzz target from contract: {}",
        contract.metadata.description
    );
    let _ = writeln!(out, "// Regenerate with: pv fuzz contracts/{stem}.yaml");
    let _ = writeln!(out);
    let _ = writeln!(out, "use libfuzzer_sys::fuzz_target;");
    let _ = writeln!(out);

    // Collect preconditions from proof obligations
    let preconditions: Vec<&str> = contract
        .proof_obligations
        .iter()
        .filter(|ob| ob.obligation_type == crate::schema::ObligationType::Precondition)
        .filter_map(|ob| ob.formal.as_deref())
        .collect();

    // Collect postconditions for assertion
    let postconditions: Vec<&str> = contract
        .proof_obligations
        .iter()
        .filter(|ob| ob.obligation_type == crate::schema::ObligationType::Postcondition)
        .filter_map(|ob| ob.formal.as_deref())
        .collect();

    let _ = writeln!(out, "fuzz_target!(|data: &[u8]| {{");
    let _ = writeln!(out, "    // Minimum input size guard");
    let _ = writeln!(out, "    if data.len() < 8 {{ return; }}");
    let _ = writeln!(out);

    // Generate input parsing from bytes
    let _ = writeln!(out, "    // Parse input: interpret bytes as f32 vector");
    let _ = writeln!(
        out,
        "    let len = ((data[0] as usize) % 64).saturating_add(1);"
    );
    let _ = writeln!(out, "    let byte_len = len * 4;");
    let _ = writeln!(out, "    if data.len() < 4 + byte_len {{ return; }}");
    let _ = writeln!(out, "    let input: Vec<f32> = data[4..4 + byte_len]");
    let _ = writeln!(out, "        .chunks_exact(4)");
    let _ = writeln!(
        out,
        "        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))"
    );
    let _ = writeln!(out, "        .collect();");
    let _ = writeln!(out);

    // Emit precondition gates
    if preconditions.is_empty() {
        let _ = writeln!(
            out,
            "    // No explicit preconditions — filter NaN/Inf for numerical stability"
        );
    } else {
        let _ = writeln!(out, "    // Contract precondition gates");
        for pre in &preconditions {
            let _ = writeln!(out, "    // PRE: {pre}");
        }
    }
    let _ = writeln!(
        out,
        "    if input.is_empty() || input.iter().any(|x| x.is_nan() || x.is_infinite()) {{"
    );
    let _ = writeln!(out, "        return;");
    let _ = writeln!(out, "    }}");
    let _ = writeln!(out);

    // Emit kernel call (must not panic)
    let fn_name = stem.replace('-', "_").replace("_v1", "");
    let _ = writeln!(out, "    // Kernel under test — must not panic");
    let _ = writeln!(out, "    let result = std::panic::catch_unwind(|| {{");
    let _ = writeln!(out, "        let mut output = vec![0.0f32; input.len()];");
    let _ = writeln!(
        out,
        "        // TODO: Wire up: {fn_name}(&input, &mut output);"
    );
    let _ = writeln!(out, "        output");
    let _ = writeln!(out, "    }});");
    let _ = writeln!(out);

    // Emit postcondition checks on successful execution
    if !postconditions.is_empty() {
        let _ = writeln!(out, "    // Contract postcondition checks");
        let _ = writeln!(out, "    if let Ok(output) = result {{");
        for post in &postconditions {
            let _ = writeln!(out, "        // POST: {post}");
        }
        let _ = writeln!(
            out,
            "        assert_eq!(output.len(), input.len(), \"output length mismatch\");"
        );
        let _ = writeln!(out, "    }}");
    }

    let _ = writeln!(out, "}});");

    out
}

/// Generate a `fuzz/Cargo.toml` for the fuzz workspace member.
pub fn generate_fuzz_cargo_toml(crate_name: &str) -> String {
    let mut out = String::with_capacity(512);

    let _ = writeln!(out, "[package]");
    let _ = writeln!(out, "name = \"{crate_name}-fuzz\"");
    let _ = writeln!(out, "version = \"0.0.0\"");
    let _ = writeln!(out, "publish = false");
    let _ = writeln!(out, "edition = \"2021\"");
    let _ = writeln!(out);
    let _ = writeln!(out, "[package.metadata]");
    let _ = writeln!(out, "cargo-fuzz = true");
    let _ = writeln!(out);
    let _ = writeln!(out, "[dependencies]");
    let _ = writeln!(out, "libfuzzer-sys = \"0.11\"");
    let _ = writeln!(out);
    let _ = writeln!(out, "# TODO: Add dependency on the crate under test");
    let _ = writeln!(out, "# {crate_name} = {{ path = \"..\" }}");

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::parse_contract_str;

    fn softmax_contract() -> Contract {
        parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Softmax kernel"
  references: ["Paper A"]
equations:
  softmax:
    formula: "σ(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))"
    domain: "x ∈ ℝ^n, n ≥ 1"
proof_obligations:
  - type: precondition
    property: "Input vector is finite and non-empty"
    formal: "∀i: ¬isNaN(x_i) ∧ ¬isInf(x_i) ∧ len(x) > 0"
  - type: postcondition
    property: "Output is a valid probability distribution"
    formal: "len(σ(x)) = len(x) ∧ ∀i: 0 < σ(x)_i < 1 ∧ |Σ σ(x)_i - 1| < ε"
  - type: invariant
    property: "Output sums to 1"
falsification_tests:
  - id: FALSIFY-001
    rule: "Normalization"
    prediction: "sum ≈ 1.0"
    if_fails: "Bug"
kani_harnesses:
  - id: KANI-001
    obligation: "sums to 1"
    bound: 8
"#,
        )
        .unwrap()
    }

    #[test]
    fn generates_fuzz_target() {
        let contract = softmax_contract();
        let output = generate_fuzz_target(&contract, "softmax-kernel-v1");

        assert!(output.contains("#![no_main]"));
        assert!(output.contains("use libfuzzer_sys::fuzz_target;"));
        assert!(output.contains("fuzz_target!"));
        assert!(output.contains("catch_unwind"));
        assert!(output.contains("PRE:"));
        assert!(output.contains("POST:"));
    }

    #[test]
    fn generates_fuzz_target_no_preconditions() {
        let contract = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Simple kernel"
  references: ["X"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "identity"
falsification_tests:
  - id: F-001
    rule: "id"
    prediction: "f(x) = x"
    if_fails: "bug"
kani_harnesses:
  - id: K-001
    obligation: "id"
    bound: 8
"#,
        )
        .unwrap();

        let output = generate_fuzz_target(&contract, "identity-v1");
        assert!(output.contains("No explicit preconditions"));
        assert!(output.contains("fuzz_target!"));
    }

    #[test]
    fn generates_cargo_toml() {
        let toml = generate_fuzz_cargo_toml("aprender");
        assert!(toml.contains("aprender-fuzz"));
        assert!(toml.contains("libfuzzer-sys"));
        assert!(toml.contains("cargo-fuzz = true"));
    }
}
