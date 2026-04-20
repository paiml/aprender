//! Short-name alias resolution for `apr pull` (CRUX-A-01).
//!
//! Contract: `contracts/crux-A-01-v1.yaml` — closes FALSIFY-CRUX-A-01-001
//! (`apr pull <short> --dry-run` emits the resolved canonical URL).
//!
//! The alias map is embedded at compile time via `include_str!` so the
//! invariant "alias_map is loaded from configs/aliases.yaml" holds for every
//! build artifact (source build, `cargo install aprender`, release tarball).

use std::collections::BTreeMap;
use std::sync::OnceLock;

const ALIASES_YAML: &str = include_str!("../../../../configs/aliases.yaml");

fn aliases() -> &'static BTreeMap<String, String> {
    static MAP: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    MAP.get_or_init(|| {
        serde_yaml::from_str::<BTreeMap<String, String>>(ALIASES_YAML)
            .expect("CRUX-A-01: embedded configs/aliases.yaml must parse as str→str map")
    })
}

/// Resolve a short name to its canonical URL.
///
/// - If `name` already contains a scheme (`://`) it is returned as-is.
/// - Otherwise the embedded alias map is consulted; `None` is returned when
///   the short name is unknown (caller handles did-you-mean).
pub fn resolve_short_name(name: &str) -> Option<String> {
    if name.contains("://") {
        return Some(name.to_string());
    }
    aliases().get(name).cloned()
}

/// Borrow the full alias map (used by future `apr registry aliases --json`).
pub fn alias_map() -> &'static BTreeMap<String, String> {
    aliases()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_canonical_short_names() {
        for canonical in ["llama3", "mistral", "phi3", "qwen2"] {
            let url = resolve_short_name(canonical)
                .unwrap_or_else(|| panic!("CRUX-A-01: {canonical} must resolve"));
            assert!(
                url.starts_with("hf://") || url.starts_with("https://"),
                "CRUX-A-01: {canonical} → {url} must be fully-qualified"
            );
        }
    }

    #[test]
    fn unknown_short_name_returns_none() {
        assert!(resolve_short_name("not-a-real-model-xyz").is_none());
    }

    #[test]
    fn scheme_qualified_input_passes_through() {
        let input = "hf://org/repo/file.gguf";
        assert_eq!(resolve_short_name(input).as_deref(), Some(input));
    }

    #[test]
    fn resolution_is_deterministic() {
        let a = resolve_short_name("llama3");
        let b = resolve_short_name("llama3");
        assert_eq!(a, b, "CRUX-A-01: resolution must be deterministic");
    }
}
