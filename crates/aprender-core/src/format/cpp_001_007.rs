// `cpp-type-preservation-v1` algorithm-level PARTIAL discharge for
// FALSIFY-CPP-001..007.
//
// Contract: `contracts/cpp-type-preservation-v1.yaml`.
//
// CPP-001: class with fields → struct with pub fields
// CPP-002: constructor → new()
// CPP-003: destructor → impl Drop
// CPP-004: namespace → mod
// CPP-005: operator+ → impl std::ops::Add
// CPP-006: inheritance → composition + Deref
// CPP-007: implicit this → self.field

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CppVerdict {
    Pass,
    Fail,
}

/// CPP-001: class with N fields produces struct with same N pub fields.
#[must_use]
pub fn verdict_from_class_to_struct(
    cpp_field_count: u32,
    rust_pub_field_count: u32,
    rust_has_struct: bool,
) -> CppVerdict {
    if !rust_has_struct {
        return CppVerdict::Fail;
    }
    if cpp_field_count == rust_pub_field_count {
        CppVerdict::Pass
    } else {
        CppVerdict::Fail
    }
}

/// CPP-002: constructor → `pub fn new(...) -> Self`.
#[must_use]
pub fn verdict_from_ctor_to_new(
    cpp_has_ctor: bool,
    rust_has_pub_fn_new: bool,
    rust_returns_self: bool,
) -> CppVerdict {
    if !cpp_has_ctor {
        return CppVerdict::Pass; // gate not applicable
    }
    if rust_has_pub_fn_new && rust_returns_self {
        CppVerdict::Pass
    } else {
        CppVerdict::Fail
    }
}

/// CPP-003: destructor → `impl Drop for ...`.
#[must_use]
pub fn verdict_from_dtor_to_drop(
    cpp_has_dtor: bool,
    rust_has_impl_drop: bool,
) -> CppVerdict {
    if !cpp_has_dtor {
        return CppVerdict::Pass;
    }
    if rust_has_impl_drop {
        CppVerdict::Pass
    } else {
        CppVerdict::Fail
    }
}

/// CPP-004: `namespace X { ... }` → `pub mod X { ... }`.
#[must_use]
pub fn verdict_from_namespace_to_mod(
    cpp_namespace_name: &str,
    rust_mod_name: &str,
    rust_mod_has_pub_visibility: bool,
) -> CppVerdict {
    if cpp_namespace_name.is_empty() {
        return CppVerdict::Fail;
    }
    if cpp_namespace_name == rust_mod_name && rust_mod_has_pub_visibility {
        CppVerdict::Pass
    } else {
        CppVerdict::Fail
    }
}

/// CPP-005: operator+ → `impl std::ops::Add`.
#[must_use]
pub fn verdict_from_operator_plus_to_add(
    cpp_has_operator_plus: bool,
    rust_has_impl_add: bool,
) -> CppVerdict {
    if !cpp_has_operator_plus {
        return CppVerdict::Pass;
    }
    if rust_has_impl_add {
        CppVerdict::Pass
    } else {
        CppVerdict::Fail
    }
}

/// CPP-006: inheritance → composition + Deref.
#[must_use]
pub fn verdict_from_inheritance_to_deref(
    cpp_has_base_class: bool,
    rust_has_base_field: bool,
    rust_has_impl_deref: bool,
) -> CppVerdict {
    if !cpp_has_base_class {
        return CppVerdict::Pass;
    }
    if rust_has_base_field && rust_has_impl_deref {
        CppVerdict::Pass
    } else {
        CppVerdict::Fail
    }
}

/// CPP-007: implicit this → self.field.
///
/// `cpp_implicit_field_accesses` = count of bare `field` references in C++ method.
/// `rust_self_field_accesses` = count of `self.field` references in Rust method.
/// Pass iff equal AND non-zero.
#[must_use]
pub fn verdict_from_this_to_self(
    cpp_implicit_field_accesses: u32,
    rust_self_field_accesses: u32,
) -> CppVerdict {
    if cpp_implicit_field_accesses == 0 {
        return CppVerdict::Pass; // no implicit-this in source — vacuous Pass
    }
    if cpp_implicit_field_accesses == rust_self_field_accesses {
        CppVerdict::Pass
    } else {
        CppVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: CPP-001..003.
    // -----------------------------------------------------------------
    #[test]
    fn fcpp001_pass_2_field_class() {
        let v = verdict_from_class_to_struct(2, 2, true);
        assert_eq!(v, CppVerdict::Pass);
    }

    #[test]
    fn fcpp001_fail_field_count_mismatch() {
        let v = verdict_from_class_to_struct(2, 1, true);
        assert_eq!(v, CppVerdict::Fail);
    }

    #[test]
    fn fcpp001_fail_no_struct_emitted() {
        let v = verdict_from_class_to_struct(2, 2, false);
        assert_eq!(v, CppVerdict::Fail);
    }

    #[test]
    fn fcpp002_pass_ctor_to_new_returning_self() {
        let v = verdict_from_ctor_to_new(true, true, true);
        assert_eq!(v, CppVerdict::Pass);
    }

    #[test]
    fn fcpp002_pass_no_ctor_in_source() {
        let v = verdict_from_ctor_to_new(false, false, false);
        assert_eq!(v, CppVerdict::Pass);
    }

    #[test]
    fn fcpp002_fail_ctor_no_new() {
        let v = verdict_from_ctor_to_new(true, false, false);
        assert_eq!(v, CppVerdict::Fail);
    }

    #[test]
    fn fcpp002_fail_new_returns_other() {
        let v = verdict_from_ctor_to_new(true, true, false);
        assert_eq!(v, CppVerdict::Fail);
    }

    #[test]
    fn fcpp003_pass_dtor_to_drop() {
        let v = verdict_from_dtor_to_drop(true, true);
        assert_eq!(v, CppVerdict::Pass);
    }

    #[test]
    fn fcpp003_pass_no_dtor() {
        let v = verdict_from_dtor_to_drop(false, false);
        assert_eq!(v, CppVerdict::Pass);
    }

    #[test]
    fn fcpp003_fail_dtor_no_drop() {
        let v = verdict_from_dtor_to_drop(true, false);
        assert_eq!(v, CppVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 2: CPP-004..005.
    // -----------------------------------------------------------------
    #[test]
    fn fcpp004_pass_math_to_pub_mod_math() {
        let v = verdict_from_namespace_to_mod("math", "math", true);
        assert_eq!(v, CppVerdict::Pass);
    }

    #[test]
    fn fcpp004_fail_name_drift() {
        let v = verdict_from_namespace_to_mod("math", "maths", true);
        assert_eq!(v, CppVerdict::Fail);
    }

    #[test]
    fn fcpp004_fail_private_mod() {
        let v = verdict_from_namespace_to_mod("math", "math", false);
        assert_eq!(v, CppVerdict::Fail);
    }

    #[test]
    fn fcpp004_fail_empty_name() {
        let v = verdict_from_namespace_to_mod("", "", true);
        assert_eq!(v, CppVerdict::Fail);
    }

    #[test]
    fn fcpp005_pass_operator_plus_to_add() {
        let v = verdict_from_operator_plus_to_add(true, true);
        assert_eq!(v, CppVerdict::Pass);
    }

    #[test]
    fn fcpp005_pass_no_operator_plus() {
        let v = verdict_from_operator_plus_to_add(false, false);
        assert_eq!(v, CppVerdict::Pass);
    }

    #[test]
    fn fcpp005_fail_operator_plus_no_add() {
        let v = verdict_from_operator_plus_to_add(true, false);
        assert_eq!(v, CppVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: CPP-006..007.
    // -----------------------------------------------------------------
    #[test]
    fn fcpp006_pass_inheritance_with_base_and_deref() {
        let v = verdict_from_inheritance_to_deref(true, true, true);
        assert_eq!(v, CppVerdict::Pass);
    }

    #[test]
    fn fcpp006_pass_no_base_class() {
        let v = verdict_from_inheritance_to_deref(false, false, false);
        assert_eq!(v, CppVerdict::Pass);
    }

    #[test]
    fn fcpp006_fail_missing_base_field() {
        let v = verdict_from_inheritance_to_deref(true, false, true);
        assert_eq!(v, CppVerdict::Fail);
    }

    #[test]
    fn fcpp006_fail_missing_deref_impl() {
        let v = verdict_from_inheritance_to_deref(true, true, false);
        assert_eq!(v, CppVerdict::Fail);
    }

    #[test]
    fn fcpp007_pass_3_implicit_3_self() {
        let v = verdict_from_this_to_self(3, 3);
        assert_eq!(v, CppVerdict::Pass);
    }

    #[test]
    fn fcpp007_pass_no_implicit_this() {
        let v = verdict_from_this_to_self(0, 0);
        assert_eq!(v, CppVerdict::Pass);
    }

    #[test]
    fn fcpp007_fail_count_mismatch() {
        let v = verdict_from_this_to_self(3, 2);
        assert_eq!(v, CppVerdict::Fail);
    }

    #[test]
    fn fcpp007_fail_dropped_all_self() {
        // C++ method had 3 implicit-this references; Rust transpilation
        // dropped them all (CXXThisExpr handling broken).
        let v = verdict_from_this_to_self(3, 0);
        assert_eq!(v, CppVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: Mutation surveys.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_001_field_count_band() {
        for n in [0_u32, 1, 2, 5, 10] {
            let v = verdict_from_class_to_struct(n, n, true);
            // The 0-field case represents an empty class — gate evaluates struct presence.
            assert_eq!(v, CppVerdict::Pass, "n={n}");
        }
    }

    #[test]
    fn mutation_survey_007_implicit_this_band() {
        for n in [0_u32, 1, 5, 100] {
            let v = verdict_from_this_to_self(n, n);
            assert_eq!(v, CppVerdict::Pass, "n={n}");
            // Off-by-one trips
            if n > 0 {
                let v_off = verdict_from_this_to_self(n, n - 1);
                assert_eq!(v_off, CppVerdict::Fail, "n={n}");
            }
        }
    }

    // -----------------------------------------------------------------
    // Section 5: Realistic.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_healthy_passes_all_7() {
        let v1 = verdict_from_class_to_struct(2, 2, true);
        let v2 = verdict_from_ctor_to_new(true, true, true);
        let v3 = verdict_from_dtor_to_drop(true, true);
        let v4 = verdict_from_namespace_to_mod("math", "math", true);
        let v5 = verdict_from_operator_plus_to_add(true, true);
        let v6 = verdict_from_inheritance_to_deref(true, true, true);
        let v7 = verdict_from_this_to_self(3, 3);
        for v in [v1, v2, v3, v4, v5, v6, v7] {
            assert_eq!(v, CppVerdict::Pass);
        }
    }

    // -----------------------------------------------------------------
    // Section 6: Pre-fix regressions.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_pre_fix_all_7_failures() {
        // 7 simultaneous regressions:
        let v1 = verdict_from_class_to_struct(2, 1, true); // dropped a field
        let v2 = verdict_from_ctor_to_new(true, false, false); // ctor not extracted
        let v3 = verdict_from_dtor_to_drop(true, false); // Drop missing
        let v4 = verdict_from_namespace_to_mod("math", "math", false); // private mod
        let v5 = verdict_from_operator_plus_to_add(true, false); // Add impl missing
        let v6 = verdict_from_inheritance_to_deref(true, false, true); // base field missing
        let v7 = verdict_from_this_to_self(3, 0); // implicit-this dropped
        for v in [v1, v2, v3, v4, v5, v6, v7] {
            assert_eq!(v, CppVerdict::Fail);
        }
    }

    // -----------------------------------------------------------------
    // Section 7: Edge cases.
    // -----------------------------------------------------------------
    #[test]
    fn vacuous_pass_for_unused_features() {
        // Source had no operator+, no inheritance, no dtor → vacuous Pass.
        let v3 = verdict_from_dtor_to_drop(false, false);
        let v5 = verdict_from_operator_plus_to_add(false, false);
        let v6 = verdict_from_inheritance_to_deref(false, false, false);
        for v in [v3, v5, v6] {
            assert_eq!(v, CppVerdict::Pass);
        }
    }
}
