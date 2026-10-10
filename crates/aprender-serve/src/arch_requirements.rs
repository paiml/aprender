// PMAT-228: Generated from architecture-requirements-v1.yaml by build.rs.
// The include! pulls in the generated WeightRole enum, field_name() impl,
// const role arrays, and required_roles() function.
//
// Fallback: if build.rs can't find the YAML (CI/crates.io), this file
// won't compile. In that case, revert to the hand-written version from git.
include!(concat!(env!("OUT_DIR"), "/arch_requirements_generated.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_roles_count() {
        // Base: 2 norms + 4 attn proj + 3 ffn proj = 9
        assert_eq!(ROLES_NO_QK_NORM_NO_BIAS.len(), 9);
    }

    #[test]
    fn test_qk_norm_roles_count() {
        // Base 9 + 2 QK norms = 11
        assert_eq!(ROLES_QK_NORM_NO_BIAS.len(), 11);
    }

    #[test]
    fn test_bias_roles_count() {
        // Base 9 + 3 biases = 12
        assert_eq!(ROLES_NO_QK_NORM_BIAS.len(), 12);
    }

    #[test]
    fn test_both_roles_count() {
        // Base 9 + 2 QK norms + 3 biases = 14
        assert_eq!(ROLES_QK_NORM_AND_BIAS.len(), 14);
    }

    #[test]
    fn test_llama_requires_base_only() {
        let arch = ArchConstraints::from_architecture("llama");
        let roles = required_roles(&arch);
        assert_eq!(roles.len(), 9);
        assert!(!roles.contains(&WeightRole::AttnQNorm));
        assert!(!roles.contains(&WeightRole::AttnQBias));
    }

    #[test]
    fn test_qwen2_requires_bias() {
        let arch = ArchConstraints::from_architecture("qwen2");
        let roles = required_roles(&arch);
        assert!(roles.contains(&WeightRole::AttnQBias));
        assert!(roles.contains(&WeightRole::AttnKBias));
        assert!(roles.contains(&WeightRole::AttnVBias));
        assert!(!roles.contains(&WeightRole::AttnQNorm));
    }

    #[test]
    fn test_qwen3_requires_qk_norm() {
        let arch = ArchConstraints::from_architecture("qwen3");
        let roles = required_roles(&arch);
        assert!(roles.contains(&WeightRole::AttnQNorm));
        assert!(roles.contains(&WeightRole::AttnKNorm));
        assert!(!roles.contains(&WeightRole::AttnQBias));
    }

    #[test]
    fn test_mistral_requires_base_only() {
        let arch = ArchConstraints::from_architecture("mistral");
        let roles = required_roles(&arch);
        assert_eq!(roles.len(), 9);
    }

    #[test]
    fn test_all_architectures_have_base_roles() {
        let archs = [
            "llama", "qwen2", "qwen3", "mistral", "gemma", "phi", "phi3", "deepseek",
        ];
        let base = &[
            WeightRole::AttnNorm,
            WeightRole::FfnNorm,
            WeightRole::QProj,
            WeightRole::KProj,
            WeightRole::VProj,
            WeightRole::OProj,
            WeightRole::FfnGate,
            WeightRole::FfnUp,
            WeightRole::FfnDown,
        ];
        for arch_name in archs {
            let arch = ArchConstraints::from_architecture(arch_name);
            let roles = required_roles(&arch);
            for base_role in base {
                assert!(
                    roles.contains(base_role),
                    "Architecture '{}' missing base role {:?}",
                    arch_name,
                    base_role
                );
            }
        }
    }

    #[test]
    fn test_all_roles_have_field_names() {
        let arch = ArchConstraints::from_architecture("qwen3");
        for role in required_roles(&arch) {
            assert!(!role.field_name().is_empty());
        }
    }

    #[test]
    fn test_no_duplicate_roles() {
        let mut arch = ArchConstraints::from_architecture("qwen3");
        arch.has_bias = true; // Synthetic: both flags set
        let roles = required_roles(&arch);
        let mut seen = std::collections::HashSet::new();
        for role in roles {
            assert!(seen.insert(role), "Duplicate role: {:?}", role);
        }
    }

    const DENSE_FFN: [WeightRole; 3] =
        [WeightRole::FfnGate, WeightRole::FfnUp, WeightRole::FfnDown];

    /// FALSIFY-ARCH-013 (#5056): for every (qk_norm, bias), the MoE cell is the dense
    /// cell minus exactly the three dense FFN roles.
    #[test]
    fn falsify_arch_013_moe_substitutes_exactly_dense_ffn() {
        let mut arch = ArchConstraints::from_architecture("llama");
        for (qk, bias) in [(false, false), (true, false), (false, true), (true, true)] {
            arch.has_qk_norm = qk;
            arch.has_bias = bias;
            arch.is_moe = false;
            let dense = required_roles(&arch);
            arch.is_moe = true;
            let moe = required_roles(&arch);
            for role in DENSE_FFN {
                assert!(
                    !moe.contains(&role),
                    "({qk}, {bias}) MoE cell requires {role:?}"
                );
                assert!(
                    dense.contains(&role),
                    "({qk}, {bias}) dense cell lacks {role:?}"
                );
            }
            assert!(
                moe.iter().all(|r| dense.contains(r)),
                "({qk}, {bias}) MoE adds a role"
            );
            assert_eq!(dense.len(), moe.len() + DENSE_FFN.len(), "({qk}, {bias})");
        }
    }

    /// FALSIFY-ARCH-013: qwen3_moe selects moe_qk_norm_no_bias (8 roles), and so does
    /// its GGUF architecture string.
    #[test]
    fn falsify_arch_013_qwen3_moe_cell() {
        for name in ["qwen3_moe", "qwen3moe"] {
            let arch = ArchConstraints::from_architecture(name);
            assert!(arch.is_moe, "{name}");
            let roles = required_roles(&arch);
            assert_eq!(roles, ROLES_MOE_QK_NORM_NO_BIAS, "{name}");
            assert_eq!(roles.len(), 8, "{name}");
            assert!(roles.contains(&WeightRole::AttnQNorm), "{name}");
        }
    }

    #[test]
    fn test_moe_role_counts() {
        assert_eq!(ROLES_MOE_NO_QK_NORM_NO_BIAS.len(), 6);
        assert_eq!(ROLES_MOE_QK_NORM_NO_BIAS.len(), 8);
        assert_eq!(ROLES_MOE_NO_QK_NORM_BIAS.len(), 9);
        assert_eq!(ROLES_MOE_QK_NORM_AND_BIAS.len(), 11);
    }

    /// The code after the header comments, from the `use` line on.
    fn body(src: &str) -> &str {
        let at = src
            .find("use crate::gguf::ArchConstraints;")
            .expect("arch_requirements source has no `use crate::gguf::ArchConstraints;` line");
        &src[at..]
    }

    /// #5056 C358 1b: regenerating from the contract leaves no diff. The build regenerates
    /// `arch_requirements_generated.rs` from contracts/architecture-requirements-v1.yaml; the
    /// committed fallback snapshot must equal it byte for byte below the header, so a hand
    /// edit to either the snapshot or the contract alone turns this red.
    #[test]
    fn falsify_arch_req_fallback_equals_regenerated() {
        let generated = include_str!(concat!(env!("OUT_DIR"), "/arch_requirements_generated.rs"));
        let fallback = include_str!("arch_requirements_fallback.rs");
        let contract = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/architecture-requirements-v1.yaml");
        if contract.exists() {
            assert!(
                generated.contains("AUTO-GENERATED from architecture-requirements-v1.yaml"),
                "the contract exists but the build used the fallback, so this comparison would be vacuous"
            );
        }
        assert!(
            body(generated) == body(fallback),
            "src/arch_requirements_fallback.rs differs from the code generated from \
             contracts/architecture-requirements-v1.yaml. Regenerate it: copy \
             $OUT_DIR/arch_requirements_generated.rs below its header into the fallback"
        );
    }
}
