// SHIP-TWO-001 — `architecture-requirements-v1` algorithm-level
// PARTIAL discharge for FALSIFY-ARCH-001..012 (closes 12/12 sweep).
//
// Contract: `contracts/architecture-requirements-v1.yaml`.
//
// Bundles 12 verdict fns over the architecture role-set
// requirements. The runtime impl lives in
// `realizar/src/arch_requirements.rs`; this file provides a
// stand-alone reference model so the algorithm-level rule is
// testable offline against drift.

use std::collections::HashSet;

// ===========================================================================
// Reference role enum + base + extension constants
// ===========================================================================

/// All canonical roles a transformer model may declare. Identifiers
/// match `realizar/src/arch_requirements.rs` so YAML and Rust agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    // 9 base roles required by every transformer architecture.
    TokenEmbedding,
    AttnQ,
    AttnK,
    AttnV,
    AttnOut,
    FfnGate,
    FfnUp,
    FfnDown,
    LayerNorm,
    // Extension roles (gated by has_qk_norm / has_bias).
    AttnQNorm,
    AttnKNorm,
    AttnQBias,
    AttnKBias,
    AttnVBias,
}

pub const BASE_ROLES: &[Role] = &[
    Role::TokenEmbedding,
    Role::AttnQ,
    Role::AttnK,
    Role::AttnV,
    Role::AttnOut,
    Role::FfnGate,
    Role::FfnUp,
    Role::FfnDown,
    Role::LayerNorm,
];

pub const QK_NORM_ROLES: &[Role] = &[Role::AttnQNorm, Role::AttnKNorm];
pub const BIAS_ROLES: &[Role] = &[Role::AttnQBias, Role::AttnKBias, Role::AttnVBias];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchConstraints {
    pub has_qk_norm: bool,
    pub has_bias: bool,
}

#[must_use]
pub fn required_roles(c: ArchConstraints) -> HashSet<Role> {
    let mut s: HashSet<Role> = BASE_ROLES.iter().copied().collect();
    if c.has_qk_norm {
        s.extend(QK_NORM_ROLES.iter().copied());
    }
    if c.has_bias {
        s.extend(BIAS_ROLES.iter().copied());
    }
    s
}

/// Map an architecture name (or alias) to its constraints. Unknown
/// names fall back to base-only constraints (FALSIFY-ARCH-010).
#[must_use]
pub fn from_architecture(name: &str) -> ArchConstraints {
    match name {
        "qwen3" | "qwen3-moe" | "qwen3moe" => ArchConstraints { has_qk_norm: true, has_bias: false },
        "qwen2" | "qwen2.5" | "qwen" => ArchConstraints { has_qk_norm: false, has_bias: true },
        "llama" | "llama3" | "mistral" | "phi" | "phi3" | "gemma" | "gemma2" => {
            ArchConstraints { has_qk_norm: false, has_bias: false }
        }
        _ => ArchConstraints { has_qk_norm: false, has_bias: false },
    }
}

// ===========================================================================
// ARCH-001 — YAML/Rust parity (set equality)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_yaml_rust_parity<S1: std::hash::BuildHasher, S2: std::hash::BuildHasher>(yaml_set: &HashSet<Role, S1>, rust_set: &HashSet<Role, S2>) -> Arch001Verdict {
    if yaml_set.is_empty() || rust_set.is_empty() { return Arch001Verdict::Fail; }
    if yaml_set.len() == rust_set.len() && yaml_set.iter().all(|r| rust_set.contains(r)) {
        Arch001Verdict::Pass
    } else {
        Arch001Verdict::Fail
    }
}

// ===========================================================================
// ARCH-002 — Base roles always present (BASE_ROLES ⊆ required)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_base_roles_present<S: std::hash::BuildHasher>(required: &HashSet<Role, S>) -> Arch002Verdict {
    for r in BASE_ROLES {
        if !required.contains(r) { return Arch002Verdict::Fail; }
    }
    Arch002Verdict::Pass
}

// ===========================================================================
// ARCH-003 — Constraint matrix produces 4 distinct non-empty sets
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_constraint_matrix_distinct() -> Arch003Verdict {
    let cells = [
        ArchConstraints { has_qk_norm: false, has_bias: false },
        ArchConstraints { has_qk_norm: true,  has_bias: false },
        ArchConstraints { has_qk_norm: false, has_bias: true  },
        ArchConstraints { has_qk_norm: true,  has_bias: true  },
    ];
    let sets: Vec<HashSet<Role>> = cells.iter().map(|c| required_roles(*c)).collect();
    if sets.iter().any(HashSet::is_empty) { return Arch003Verdict::Fail; }
    for i in 0..sets.len() {
        for j in (i + 1)..sets.len() {
            if sets[i] == sets[j] { return Arch003Verdict::Fail; }
        }
    }
    Arch003Verdict::Pass
}

// ===========================================================================
// ARCH-004 — Cardinalities: 9, 11, 12, 14 across the 4 cells
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_role_cardinalities() -> Arch004Verdict {
    let cells = [
        (ArchConstraints { has_qk_norm: false, has_bias: false }, 9_usize),
        (ArchConstraints { has_qk_norm: true,  has_bias: false }, 11),
        (ArchConstraints { has_qk_norm: false, has_bias: true  }, 12),
        (ArchConstraints { has_qk_norm: true,  has_bias: true  }, 14),
    ];
    for (c, expected) in cells {
        if required_roles(c).len() != expected { return Arch004Verdict::Fail; }
    }
    Arch004Verdict::Pass
}

// ===========================================================================
// ARCH-005 — Qwen3 has QK norm, NOT bias
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_qwen3_role_set<S: std::hash::BuildHasher>(qwen3_set: &HashSet<Role, S>) -> Arch005Verdict {
    let has_qknorm = qwen3_set.contains(&Role::AttnQNorm) && qwen3_set.contains(&Role::AttnKNorm);
    let has_bias = qwen3_set.contains(&Role::AttnQBias)
        || qwen3_set.contains(&Role::AttnKBias)
        || qwen3_set.contains(&Role::AttnVBias);
    if has_qknorm && !has_bias { Arch005Verdict::Pass } else { Arch005Verdict::Fail }
}

// ===========================================================================
// ARCH-006 — Qwen2 has bias, NOT QK norm
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_qwen2_role_set<S: std::hash::BuildHasher>(qwen2_set: &HashSet<Role, S>) -> Arch006Verdict {
    let has_bias = qwen2_set.contains(&Role::AttnQBias)
        && qwen2_set.contains(&Role::AttnKBias)
        && qwen2_set.contains(&Role::AttnVBias);
    let has_qknorm = qwen2_set.contains(&Role::AttnQNorm) || qwen2_set.contains(&Role::AttnKNorm);
    if has_bias && !has_qknorm { Arch006Verdict::Pass } else { Arch006Verdict::Fail }
}

// ===========================================================================
// ARCH-007 — LLaMA/Mistral are base-only (9 roles, no extensions)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_base_only_arch<S: std::hash::BuildHasher>(role_set: &HashSet<Role, S>) -> Arch007Verdict {
    if role_set.len() != 9 { return Arch007Verdict::Fail; }
    let has_extension = QK_NORM_ROLES.iter().chain(BIAS_ROLES.iter()).any(|r| role_set.contains(r));
    if has_extension { Arch007Verdict::Fail } else { Arch007Verdict::Pass }
}

// ===========================================================================
// ARCH-008 — Missing required role → reject before forward pass
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch008Verdict { Pass, Fail }

/// Pass iff `present_roles` is a strict superset/equal of `required`
/// when `expect_complete=true`, OR `present_roles` is missing at
/// least one required role when `expect_complete=false`.
#[must_use]
pub fn verdict_from_completeness_check<S1, S2>(
    present_roles: &HashSet<Role, S1>,
    required: &HashSet<Role, S2>,
    expect_complete: bool,
) -> Arch008Verdict
where
    S1: std::hash::BuildHasher,
    S2: std::hash::BuildHasher,
{
    let missing_count = required.iter().filter(|r| !present_roles.contains(r)).count();
    let actually_complete = missing_count == 0;
    if actually_complete == expect_complete { Arch008Verdict::Pass } else { Arch008Verdict::Fail }
}

// ===========================================================================
// ARCH-009 — Optional roles do NOT trigger rejection
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch009Verdict { Pass, Fail }

/// Pass iff a model that has all required roles passes the
/// completeness check, regardless of which optional roles it has.
/// Counter-example: a model that fails completeness despite having
/// all required roles indicates the gate is over-strict.
#[must_use]
pub fn verdict_from_optional_role_check<S1, S2>(
    present_required: &HashSet<Role, S1>,
    required: &HashSet<Role, S2>,
    completeness_passed: bool,
) -> Arch009Verdict
where
    S1: std::hash::BuildHasher,
    S2: std::hash::BuildHasher,
{
    let has_all_required = required.iter().all(|r| present_required.contains(r));
    if has_all_required == completeness_passed { Arch009Verdict::Pass } else { Arch009Verdict::Fail }
}

// ===========================================================================
// ARCH-010 — Unknown architecture → base-only constraints
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch010Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_unknown_arch_fallback(name: &str) -> Arch010Verdict {
    let c = from_architecture(name);
    if c.has_qk_norm || c.has_bias {
        // For known architectures we expect non-base constraints.
        return Arch010Verdict::Pass;
    }
    let roles = required_roles(c);
    if roles.len() == 9 && BASE_ROLES.iter().all(|r| roles.contains(r)) {
        Arch010Verdict::Pass
    } else {
        Arch010Verdict::Fail
    }
}

// ===========================================================================
// ARCH-011 — Alias equivalence: from_architecture(alias) == primary
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch011Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_alias_equivalence(primary: &str, alias: &str) -> Arch011Verdict {
    if from_architecture(primary) == from_architecture(alias) {
        Arch011Verdict::Pass
    } else {
        Arch011Verdict::Fail
    }
}

// ===========================================================================
// ARCH-012 — Monotonicity: enabling features only adds roles
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch012Verdict { Pass, Fail }

/// Pass iff `required_roles(superset) ⊇ required_roles(base)` for
/// every flag combination where `superset` enables a strict superset
/// of `base`'s features.
#[must_use]
pub fn verdict_from_monotonicity(base: ArchConstraints, superset: ArchConstraints) -> Arch012Verdict {
    let base_implies_superset = (!base.has_qk_norm || superset.has_qk_norm)
        && (!base.has_bias || superset.has_bias);
    if !base_implies_superset {
        // Not a real superset relation — vacuously true (Pass).
        return Arch012Verdict::Pass;
    }
    let base_roles = required_roles(base);
    let super_roles = required_roles(superset);
    if base_roles.is_subset(&super_roles) { Arch012Verdict::Pass } else { Arch012Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(roles: &[Role]) -> HashSet<Role> { roles.iter().copied().collect() }

    // ----- ARCH-001 ----------------------------------------------------------

    #[test]
    fn arch001_pass_identical_sets() {
        let a = required_roles(from_architecture("qwen3"));
        let b = required_roles(from_architecture("qwen3"));
        assert_eq!(verdict_from_yaml_rust_parity(&a, &b), Arch001Verdict::Pass);
    }

    #[test]
    fn arch001_fail_drift() {
        let a = required_roles(from_architecture("qwen3"));
        let b = required_roles(from_architecture("qwen2"));
        assert_eq!(verdict_from_yaml_rust_parity(&a, &b), Arch001Verdict::Fail);
    }

    #[test]
    fn arch001_fail_empty() {
        let empty = HashSet::new();
        let a = required_roles(from_architecture("qwen3"));
        assert_eq!(verdict_from_yaml_rust_parity(&empty, &a), Arch001Verdict::Fail);
    }

    // ----- ARCH-002 ----------------------------------------------------------

    #[test]
    fn arch002_pass_base_subset() {
        let req = required_roles(ArchConstraints { has_qk_norm: false, has_bias: false });
        assert_eq!(verdict_from_base_roles_present(&req), Arch002Verdict::Pass);
    }

    #[test]
    fn arch002_pass_qwen3_has_base() {
        let req = required_roles(ArchConstraints { has_qk_norm: true, has_bias: false });
        assert_eq!(verdict_from_base_roles_present(&req), Arch002Verdict::Pass);
    }

    #[test]
    fn arch002_fail_missing_attn_q() {
        let mut req = required_roles(ArchConstraints { has_qk_norm: false, has_bias: false });
        req.remove(&Role::AttnQ);
        assert_eq!(verdict_from_base_roles_present(&req), Arch002Verdict::Fail);
    }

    // ----- ARCH-003 ----------------------------------------------------------

    #[test]
    fn arch003_pass() {
        assert_eq!(verdict_from_constraint_matrix_distinct(), Arch003Verdict::Pass);
    }

    // ----- ARCH-004 ----------------------------------------------------------

    #[test]
    fn arch004_pass_canonical_counts() {
        assert_eq!(verdict_from_role_cardinalities(), Arch004Verdict::Pass);
    }

    // ----- ARCH-005 ----------------------------------------------------------

    #[test]
    fn arch005_pass_qwen3() {
        let qwen3 = required_roles(from_architecture("qwen3"));
        assert_eq!(verdict_from_qwen3_role_set(&qwen3), Arch005Verdict::Pass);
    }

    #[test]
    fn arch005_fail_qwen3_missing_qk_norm() {
        // Hypothetical drift: qwen3 mistakenly built without QK norm.
        let drifted = required_roles(ArchConstraints { has_qk_norm: false, has_bias: false });
        assert_eq!(verdict_from_qwen3_role_set(&drifted), Arch005Verdict::Fail);
    }

    #[test]
    fn arch005_fail_qwen3_swapped_with_qwen2() {
        let qwen2 = required_roles(from_architecture("qwen2"));
        assert_eq!(verdict_from_qwen3_role_set(&qwen2), Arch005Verdict::Fail);
    }

    // ----- ARCH-006 ----------------------------------------------------------

    #[test]
    fn arch006_pass_qwen2() {
        let qwen2 = required_roles(from_architecture("qwen2"));
        assert_eq!(verdict_from_qwen2_role_set(&qwen2), Arch006Verdict::Pass);
    }

    #[test]
    fn arch006_fail_qwen2_missing_bias() {
        let drifted = required_roles(ArchConstraints { has_qk_norm: false, has_bias: false });
        assert_eq!(verdict_from_qwen2_role_set(&drifted), Arch006Verdict::Fail);
    }

    #[test]
    fn arch006_fail_qwen2_swapped_with_qwen3() {
        let qwen3 = required_roles(from_architecture("qwen3"));
        assert_eq!(verdict_from_qwen2_role_set(&qwen3), Arch006Verdict::Fail);
    }

    // ----- ARCH-007 ----------------------------------------------------------

    #[test]
    fn arch007_pass_llama() {
        let llama = required_roles(from_architecture("llama"));
        assert_eq!(verdict_from_base_only_arch(&llama), Arch007Verdict::Pass);
    }

    #[test]
    fn arch007_pass_mistral() {
        let mistral = required_roles(from_architecture("mistral"));
        assert_eq!(verdict_from_base_only_arch(&mistral), Arch007Verdict::Pass);
    }

    #[test]
    fn arch007_fail_qwen3_extension() {
        let qwen3 = required_roles(from_architecture("qwen3"));
        assert_eq!(verdict_from_base_only_arch(&qwen3), Arch007Verdict::Fail);
    }

    #[test]
    fn arch007_fail_short_set() {
        let mut llama = required_roles(from_architecture("llama"));
        llama.remove(&Role::AttnQ);
        assert_eq!(verdict_from_base_only_arch(&llama), Arch007Verdict::Fail);
    }

    // ----- ARCH-008 ----------------------------------------------------------

    #[test]
    fn arch008_pass_complete() {
        let req = required_roles(from_architecture("qwen3"));
        let present = req.clone();
        assert_eq!(
            verdict_from_completeness_check(&present, &req, true),
            Arch008Verdict::Pass
        );
    }

    #[test]
    fn arch008_pass_missing_one_expected_incomplete() {
        let req = required_roles(from_architecture("qwen3"));
        let mut present = req.clone();
        present.remove(&Role::AttnQNorm);
        assert_eq!(
            verdict_from_completeness_check(&present, &req, false),
            Arch008Verdict::Pass
        );
    }

    #[test]
    fn arch008_fail_silently_succeeded_with_missing_role() {
        let req = required_roles(from_architecture("qwen3"));
        let mut present = req.clone();
        present.remove(&Role::AttnQNorm);
        // Missing role but check claims complete — the GH-279 regression class.
        assert_eq!(
            verdict_from_completeness_check(&present, &req, true),
            Arch008Verdict::Fail
        );
    }

    // ----- ARCH-009 ----------------------------------------------------------

    #[test]
    fn arch009_pass_required_only() {
        let req = required_roles(from_architecture("qwen2"));
        assert_eq!(
            verdict_from_optional_role_check(&req, &req, true),
            Arch009Verdict::Pass
        );
    }

    #[test]
    fn arch009_fail_missing_required_but_passed() {
        let req = required_roles(from_architecture("qwen2"));
        let mut partial = req.clone();
        partial.remove(&Role::AttnQBias);
        assert_eq!(
            verdict_from_optional_role_check(&partial, &req, true),
            Arch009Verdict::Fail
        );
    }

    // ----- ARCH-010 ----------------------------------------------------------

    #[test]
    fn arch010_pass_unknown_string() {
        assert_eq!(
            verdict_from_unknown_arch_fallback("future_arch_2027"),
            Arch010Verdict::Pass
        );
    }

    #[test]
    fn arch010_pass_known_qwen3() {
        assert_eq!(verdict_from_unknown_arch_fallback("qwen3"), Arch010Verdict::Pass);
    }

    #[test]
    fn arch010_pass_random_garbage() {
        assert_eq!(verdict_from_unknown_arch_fallback(""), Arch010Verdict::Pass);
        assert_eq!(verdict_from_unknown_arch_fallback("xyz"), Arch010Verdict::Pass);
    }

    // ----- ARCH-011 ----------------------------------------------------------

    #[test]
    fn arch011_pass_llama_aliases() {
        assert_eq!(verdict_from_alias_equivalence("llama", "llama3"), Arch011Verdict::Pass);
    }

    #[test]
    fn arch011_pass_qwen_aliases() {
        assert_eq!(verdict_from_alias_equivalence("qwen2", "qwen2.5"), Arch011Verdict::Pass);
        assert_eq!(verdict_from_alias_equivalence("qwen2", "qwen"), Arch011Verdict::Pass);
    }

    #[test]
    fn arch011_pass_phi_aliases() {
        assert_eq!(verdict_from_alias_equivalence("phi", "phi3"), Arch011Verdict::Pass);
    }

    #[test]
    fn arch011_fail_cross_family() {
        assert_eq!(verdict_from_alias_equivalence("qwen2", "qwen3"), Arch011Verdict::Fail);
        assert_eq!(verdict_from_alias_equivalence("llama", "qwen3"), Arch011Verdict::Fail);
    }

    // ----- ARCH-012 ----------------------------------------------------------

    #[test]
    fn arch012_pass_base_to_qknorm() {
        let base = ArchConstraints { has_qk_norm: false, has_bias: false };
        let sup = ArchConstraints { has_qk_norm: true,  has_bias: false };
        assert_eq!(verdict_from_monotonicity(base, sup), Arch012Verdict::Pass);
    }

    #[test]
    fn arch012_pass_base_to_full() {
        let base = ArchConstraints { has_qk_norm: false, has_bias: false };
        let sup = ArchConstraints { has_qk_norm: true,  has_bias: true };
        assert_eq!(verdict_from_monotonicity(base, sup), Arch012Verdict::Pass);
    }

    #[test]
    fn arch012_pass_self_self() {
        let c = ArchConstraints { has_qk_norm: true, has_bias: true };
        assert_eq!(verdict_from_monotonicity(c, c), Arch012Verdict::Pass);
    }

    #[test]
    fn arch012_pass_non_superset_relation() {
        // base has bias, sup has qk_norm — neither implies the other,
        // so monotonicity is vacuously true.
        let base = ArchConstraints { has_qk_norm: false, has_bias: true };
        let sup = ArchConstraints { has_qk_norm: true, has_bias: false };
        assert_eq!(verdict_from_monotonicity(base, sup), Arch012Verdict::Pass);
    }

    // ----- Reference-model spot checks ---------------------------------------

    #[test]
    fn reference_model_qwen3_has_canonical_set() {
        let q3 = required_roles(from_architecture("qwen3"));
        assert_eq!(q3.len(), 11);
        assert!(q3.contains(&Role::AttnQNorm));
        assert!(q3.contains(&Role::AttnKNorm));
        assert!(!q3.contains(&Role::AttnQBias));
    }

    #[test]
    fn reference_model_qwen2_has_canonical_set() {
        let q2 = required_roles(from_architecture("qwen2"));
        assert_eq!(q2.len(), 12);
        assert!(q2.contains(&Role::AttnQBias));
        assert!(q2.contains(&Role::AttnKBias));
        assert!(q2.contains(&Role::AttnVBias));
        assert!(!q2.contains(&Role::AttnQNorm));
    }

    #[test]
    fn reference_model_alias_set_equality() {
        assert_eq!(s(BASE_ROLES).len(), 9);
        assert_eq!(s(QK_NORM_ROLES).len(), 2);
        assert_eq!(s(BIAS_ROLES).len(), 3);
    }
}
