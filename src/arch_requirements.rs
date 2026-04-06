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
}
