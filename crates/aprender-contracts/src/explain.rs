//! Contract explanation — chain-of-thought narrative for any contract.
//!
//! Unlike `book_gen` (reference tables), explain produces prose that walks
//! through the contract section by section, explaining the *why* behind
//! each element.

use std::fmt::Write;

use crate::binding::BindingRegistry;
use crate::proof_status::compute_proof_level;
use crate::schema::{Contract, ObligationType};

#[path = "explain_render.rs"]
mod explain_render;
use explain_render::{
    write_binding_status, write_enforcement, write_falsification_tests, write_kani_harnesses,
    write_kernel_phases, write_obligations, write_qa_gate, write_simd_dispatch,
    write_verification_ladder,
};

#[cfg(test)]
#[allow(clippy::all)]
#[path = "explain_tests.rs"]
mod tests;

/// Return a mathematical pattern description for the given obligation type.
pub fn obligation_pattern(ot: ObligationType) -> &'static str {
    match ot {
        ObligationType::Invariant => "∀x ∈ Domain: P(f(x)) — property holds for all inputs",
        ObligationType::Equivalence => "∀x: |f(x) - g(x)| < ε — two implementations agree",
        ObligationType::Bound => "∀x: a ≤ f(x)_i ≤ b — output range bounded",
        ObligationType::Monotonicity => "x_i > x_j → f(x)_i > f(x)_j — order preserved",
        ObligationType::Idempotency => "f(f(x)) = f(x) — applying twice gives same result",
        ObligationType::Linearity => "f(αx) = α·f(x) — homogeneous scaling",
        ObligationType::Symmetry => "f(permute(x)) related to f(x) — permutation property",
        ObligationType::Associativity => "(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c) — grouping invariant",
        ObligationType::Conservation => "Q(before) = Q(after) — conserved quantity",
        ObligationType::Ordering => "a ≤ b → f(a) ≤ f(b) — order relation maintained",
        ObligationType::Completeness => "∀ required elements present — completeness verified",
        ObligationType::Soundness => "∀x: P(x) → Q(f(x)) — soundness of transformation",
        ObligationType::Involution => "f(f(x)) = x — involution (self-inverse)",
        ObligationType::Determinism => "f(x) = f(x) — deterministic output for same input",
        ObligationType::Roundtrip => "decode(encode(x)) = x — roundtrip fidelity",
        ObligationType::StateMachine => "S × A → S — valid state transitions",
        ObligationType::Classification => "f(x) ∈ C — output belongs to valid class set",
        ObligationType::Independence => "P(A∩B) = P(A)·P(B) — statistical independence",
        ObligationType::Termination => "algorithm terminates in finite steps",
        // Eiffel DbC types
        ObligationType::Precondition => "P(input) — caller must guarantee before call",
        ObligationType::Postcondition => "P(in) → Q(out) — kernel guarantees if pre holds",
        ObligationType::Frame => "modifies(S) ∧ preserves(T\\S) — only S may change",
        ObligationType::LoopInvariant => "∀ iter i: P(state_i) — maintained across iterations",
        ObligationType::LoopVariant => "V(state) ∈ ℕ, strictly decreasing — termination witness",
        ObligationType::OldState => "Q(old(state), new(state)) — relates pre to post state",
        ObligationType::Subcontract => "weaken(pre) ∧ strengthen(post) — behavioral subtyping",
        ObligationType::Safety => "∀x: safe(f(x)) — memory/IO safety obligation",
        ObligationType::Liveness => "◇ P — eventually P holds (progress obligation)",
    }
}

/// Kani strategy explanation for the explain output.
fn strategy_explanation(strategy: &str) -> &str {
    match strategy {
        "exhaustive" => "verify for ALL inputs within bound",
        "stub_float" => {
            "assume Lean-proved postconditions on transcendentals, verify surrounding code"
        }
        "compositional" => "verify sub-kernels separately, compose proofs",
        "bounded_int" => "integer-only verification within bound",
        _ => "bounded model check",
    }
}

/// Generate a chain-of-thought narrative explanation for a contract.
///
/// `stem` is the contract filename without `.yaml`.
/// `binding` is an optional binding registry for binding status.
pub fn explain_contract(
    contract: &Contract,
    stem: &str,
    binding: Option<&BindingRegistry>,
) -> String {
    let mut out = String::with_capacity(4096);

    write_header(&mut out, contract, stem);
    write_what(&mut out, contract);
    write_equations(&mut out, contract);
    write_obligations(&mut out, contract);
    write_verification_ladder(&mut out, contract, binding);
    write_falsification_tests(&mut out, contract);
    write_kani_harnesses(&mut out, contract);
    write_kernel_phases(&mut out, contract);
    write_simd_dispatch(&mut out, contract);
    write_enforcement(&mut out, contract);
    write_qa_gate(&mut out, contract);
    write_type_invariants(&mut out, contract);
    write_coq_status(&mut out, contract);
    write_binding_status(&mut out, contract, stem, binding);

    out
}

fn write_type_invariants(out: &mut String, contract: &Contract) {
    if contract.type_invariants.is_empty() {
        return;
    }
    let _ = writeln!(out, "Type invariants (Meyer's class invariants)");
    let _ = writeln!(
        out,
        "  These predicates must hold for every instance at every stable state."
    );
    let _ = writeln!(out);
    for inv in &contract.type_invariants {
        let desc = inv
            .description
            .as_deref()
            .map(|d| format!(" — {d}"))
            .unwrap_or_default();
        let _ = writeln!(out, "  {} [{}]{}", inv.name, inv.type_name, desc);
        let _ = writeln!(out, "    Predicate: {}", inv.predicate);
        let _ = writeln!(out);
    }
}

fn write_coq_status(out: &mut String, contract: &Contract) {
    let Some(ref spec) = contract.coq_spec else {
        return;
    };
    let _ = writeln!(out, "Coq verification ({})", spec.module);
    if spec.obligations.is_empty() {
        let _ = writeln!(out, "  No obligation links defined — stubs only");
    } else {
        for ob in &spec.obligations {
            let _ = writeln!(out, "  {} → {} [{}]", ob.links_to, ob.coq_lemma, ob.status);
        }
    }
    let _ = writeln!(out);
}

/// Generate a markdown explanation with headers and LaTeX math.
#[allow(clippy::too_many_lines)]
pub fn explain_contract_markdown(
    contract: &Contract,
    stem: &str,
    binding: Option<&BindingRegistry>,
) -> String {
    let mut out = String::with_capacity(4096);
    let _ = writeln!(out, "# {stem}\n");
    let _ = writeln!(
        out,
        "**Version:** {} | **Description:** {}\n",
        contract.metadata.version, contract.metadata.description
    );

    if !contract.metadata.references.is_empty() {
        let _ = writeln!(out, "## References\n");
        for r in &contract.metadata.references {
            let _ = writeln!(out, "- {r}");
        }
        let _ = writeln!(out);
    }

    if !contract.equations.is_empty() {
        let _ = writeln!(out, "## Equations\n");
        for (name, eq) in &contract.equations {
            let _ = writeln!(out, "### {name}\n");
            let latex = crate::latex::math_to_latex(&eq.formula);
            let _ = writeln!(out, "$$\n{latex}\n$$\n");
            if let Some(ref dom) = eq.domain {
                let _ = writeln!(out, "**Domain:** ${}$\n", crate::latex::math_to_latex(dom));
            }
            if let Some(ref cod) = eq.codomain {
                let _ = writeln!(
                    out,
                    "**Codomain:** ${}$\n",
                    crate::latex::math_to_latex(cod)
                );
            }
            if !eq.invariants.is_empty() {
                let _ = writeln!(out, "**Invariants:**\n");
                for inv in &eq.invariants {
                    let _ = writeln!(out, "- ${}$", crate::latex::math_to_latex(inv));
                }
                let _ = writeln!(out);
            }
            if !eq.preconditions.is_empty() {
                let _ = writeln!(out, "**Preconditions:**\n");
                for pre in &eq.preconditions {
                    let _ = writeln!(out, "- `{pre}`");
                }
                let _ = writeln!(out);
            }
            if !eq.postconditions.is_empty() {
                let _ = writeln!(out, "**Postconditions:**\n");
                for post in &eq.postconditions {
                    let _ = writeln!(out, "- `{post}`");
                }
                let _ = writeln!(out);
            }
        }
    }

    if !contract.proof_obligations.is_empty() {
        let _ = writeln!(out, "## Proof Obligations\n");
        let _ = writeln!(out, "| # | Type | Property | Formal |");
        let _ = writeln!(out, "|---|------|----------|--------|");
        for (i, ob) in contract.proof_obligations.iter().enumerate() {
            let formal = ob.formal.as_deref().unwrap_or("");
            let _ = writeln!(
                out,
                "| {} | `{}` | {} | {} |",
                i + 1,
                ob.obligation_type,
                ob.property,
                formal
            );
        }
        let _ = writeln!(out);
    }

    // Verification summary
    let level = compute_proof_level(contract, None);
    let _ = writeln!(out, "## Verification\n");
    let _ = writeln!(out, "**Proof level:** {level}\n");
    if let Some(ref vs) = contract.verification_summary {
        let _ = writeln!(
            out,
            "- Lean: {}/{} proved",
            vs.l4_lean_proved, vs.total_obligations
        );
    }
    let _ = writeln!(out, "- Kani: {} harnesses", contract.kani_harnesses.len());
    let _ = writeln!(
        out,
        "- Tests: {} falsification\n",
        contract.falsification_tests.len()
    );

    // Binding status
    if let Some(registry) = binding {
        let contract_file = format!("{stem}.yaml");
        let has_bindings = contract.equations.keys().any(|eq| {
            registry
                .bindings
                .iter()
                .any(|b| b.contract == contract_file && b.equation == *eq)
        });
        if has_bindings {
            let _ = writeln!(out, "## Bindings ({})\n", registry.target_crate);
            let _ = writeln!(out, "| Equation | Status |");
            let _ = writeln!(out, "|----------|--------|");
            for eq_name in contract.equations.keys() {
                let status = registry
                    .bindings
                    .iter()
                    .find(|b| b.contract == contract_file && b.equation == *eq_name)
                    .map_or("missing", |b| match b.status {
                        crate::binding::ImplStatus::Implemented => "implemented",
                        crate::binding::ImplStatus::Partial => "partial",
                        crate::binding::ImplStatus::NotImplemented => "not_implemented",
                        crate::binding::ImplStatus::Pending => "pending",
                    });
                let _ = writeln!(out, "| {eq_name} | {status} |");
            }
            let _ = writeln!(out);
        }
    }

    out
}

/// Generate a JSON explanation of the contract.
// The `serde_json::json!` macro expands to internal `.unwrap()` calls on
// infallible `serde_json::Value` construction. Those are not author-written
// unwraps and cannot fail here, so the production `disallowed_methods` ban does
// not apply to them.
#[allow(clippy::disallowed_methods)]
pub fn explain_contract_json(
    contract: &Contract,
    stem: &str,
    binding: Option<&BindingRegistry>,
) -> String {
    let level = compute_proof_level(contract, None);

    let obligations: Vec<serde_json::Value> = contract
        .proof_obligations
        .iter()
        .map(|ob| {
            let mut obj = serde_json::json!({
                "type": ob.obligation_type.to_string(),
                "property": ob.property,
                "pattern": obligation_pattern(ob.obligation_type),
            });
            if let Some(ref f) = ob.formal {
                obj["formal"] = serde_json::json!(f);
            }
            if let Some(t) = ob.tolerance {
                obj["tolerance"] = serde_json::json!(t);
            }
            if let Some(ref lean) = ob.lean {
                obj["lean"] = serde_json::json!({
                    "theorem": lean.theorem,
                    "status": lean.status.to_string(),
                });
            }
            if let Some(ref req) = ob.requires {
                obj["requires"] = serde_json::json!(req);
            }
            if let Some(ref phase) = ob.applies_to_phase {
                obj["applies_to_phase"] = serde_json::json!(phase);
            }
            if let Some(ref parent) = ob.parent_contract {
                obj["parent_contract"] = serde_json::json!(parent);
            }
            obj
        })
        .collect();

    let json = serde_json::json!({
        "stem": stem,
        "version": contract.metadata.version,
        "description": contract.metadata.description,
        "references": contract.metadata.references,
        "depends_on": contract.metadata.depends_on,
        "equations": contract.equations.keys().collect::<Vec<_>>(),
        "proof_level": level.to_string(),
        "obligations": obligations,
        "falsification_tests": contract.falsification_tests.len(),
        "kani_harnesses": contract.kani_harnesses.len(),
        "binding": binding.map(|b| {
            let contract_file = format!("{stem}.yaml");
            let statuses: std::collections::BTreeMap<String, String> = contract
                .equations
                .keys()
                .map(|eq| {
                    let status = b.bindings.iter()
                        .find(|bi| bi.contract == contract_file && bi.equation == *eq)
                        .map_or_else(|| "missing".to_string(), |bi| bi.status.to_string());
                    (eq.clone(), status)
                })
                .collect();
            serde_json::json!({
                "target_crate": b.target_crate,
                "equations": statuses,
            })
        }),
    });

    serde_json::to_string_pretty(&json).unwrap_or_else(|_| "{}".to_string())
}

fn write_header(out: &mut String, contract: &Contract, stem: &str) {
    use crate::schema::ContractKind;
    let kind_tag = if contract.kind() == ContractKind::Kernel {
        String::new()
    } else {
        format!(" [{}]", contract.kind())
    };
    let _ = writeln!(out, "{stem} (v{}){kind_tag}", contract.metadata.version);
    let _ = writeln!(out, "{}", contract.metadata.description);
    let _ = writeln!(out);
}

fn write_what(out: &mut String, contract: &Contract) {
    let _ = writeln!(out, "What this contract specifies");

    let refs = &contract.metadata.references;
    if refs.is_empty() {
        let _ = writeln!(
            out,
            "  This contract specifies {}.",
            contract.metadata.description
        );
    } else {
        let _ = write!(
            out,
            "  This contract specifies {}. It derives from",
            contract.metadata.description
        );
        for (i, r) in refs.iter().enumerate() {
            if i == 0 {
                let _ = write!(out, " {r}");
            } else {
                let _ = write!(out, " and {r}");
            }
        }
        let _ = writeln!(out, ".");
    }

    if !contract.metadata.depends_on.is_empty() {
        let _ = write!(out, "  Depends on:");
        for dep in &contract.metadata.depends_on {
            let _ = write!(out, " {dep}");
        }
        let _ = writeln!(out);
    }

    let _ = writeln!(out);
}

fn write_equations(out: &mut String, contract: &Contract) {
    if contract.equations.is_empty() {
        return;
    }
    let _ = writeln!(out, "Governing equations");
    let _ = writeln!(out);

    for (name, eq) in &contract.equations {
        let _ = writeln!(out, "  {name}");
        let _ = writeln!(out, "    {}", eq.formula);

        if let Some(ref dom) = eq.domain {
            let _ = writeln!(out, "    Domain: {dom}");
        }
        if let Some(ref cod) = eq.codomain {
            let _ = writeln!(out, "    Range:  {cod}");
        }

        if !eq.invariants.is_empty() {
            let _ = writeln!(out);
            let _ = writeln!(out, "    Invariants:");
            for (i, inv) in eq.invariants.iter().enumerate() {
                let _ = writeln!(out, "      {}. {inv}", i + 1);
            }
        }

        if !eq.preconditions.is_empty() {
            let _ = writeln!(out);
            let _ = writeln!(out, "    Preconditions (caller must guarantee):");
            for pre in &eq.preconditions {
                let _ = writeln!(out, "      - {pre}");
            }
        }

        if !eq.postconditions.is_empty() {
            let _ = writeln!(out);
            let _ = writeln!(out, "    Postconditions (kernel guarantees):");
            for post in &eq.postconditions {
                let _ = writeln!(out, "      - {post}");
            }
        }

        if let Some(ref theorem) = eq.lean_theorem {
            let _ = writeln!(out, "    Lean theorem: {theorem}");
        }

        let _ = writeln!(out);
    }
}
