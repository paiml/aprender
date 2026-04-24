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

/// Levenshtein edit distance between two ASCII strings (iterative, O(|a|·|b|) time,
/// O(min(|a|,|b|)) space).
///
/// Pure helper for FALSIFY-CRUX-A-01-003 (did-you-mean suggestions). No Unicode
/// normalization is performed — alias keys are ASCII-only by contract.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr: Vec<usize> = vec![0; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// Return alias-map keys within `max_distance` Levenshtein edits of `query`,
/// sorted ascending by distance (ties broken alphabetically). Used by
/// `apr pull <typo> --dry-run` to emit "did you mean …" hints.
///
/// Closes FALSIFY-CRUX-A-01-003 suggestion logic.
pub fn did_you_mean(query: &str, max_distance: usize) -> Vec<&'static str> {
    let mut hits: Vec<(usize, &str)> = aliases()
        .keys()
        .filter_map(|k| {
            let d = levenshtein(query, k);
            (d <= max_distance).then_some((d, k.as_str()))
        })
        .collect();
    hits.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    hits.into_iter().map(|(_, k)| k).collect()
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

    #[test]
    fn levenshtein_empty_strings() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", "xyz"), 3);
    }

    #[test]
    fn levenshtein_identity_and_basic_edits() {
        assert_eq!(levenshtein("llama3", "llama3"), 0);
        assert_eq!(levenshtein("lama3", "llama3"), 1); // insertion
        assert_eq!(levenshtein("llamma3", "llama3"), 1); // deletion
        assert_eq!(levenshtein("llana3", "llama3"), 1); // substitution
        assert_eq!(levenshtein("ab", "cd"), 2); // two substitutions
        assert_eq!(levenshtein("lam3", "llama3"), 2); // two insertions
    }

    #[test]
    fn did_you_mean_finds_llama3_for_typo() {
        let suggestions = did_you_mean("lama3", 2);
        assert!(
            suggestions.contains(&"llama3"),
            "CRUX-A-01 FALSIFY-003: 'lama3' must suggest 'llama3', got {suggestions:?}"
        );
    }

    #[test]
    fn did_you_mean_empty_when_too_far() {
        let suggestions = did_you_mean("completely-unrelated-xyz", 2);
        assert!(
            suggestions.is_empty(),
            "CRUX-A-01 FALSIFY-003: unrelated names must not produce suggestions, got {suggestions:?}"
        );
    }

    #[test]
    fn did_you_mean_ranks_closer_first() {
        // "qwem2" is 1 edit from "qwen2" and 2 edits from "qwen2.5"
        let suggestions = did_you_mean("qwem2", 2);
        assert_eq!(
            suggestions.first(),
            Some(&"qwen2"),
            "CRUX-A-01 FALSIFY-003: closest match must rank first, got {suggestions:?}"
        );
    }
}
