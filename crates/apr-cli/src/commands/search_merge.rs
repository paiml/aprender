//! Hub + local search-result merger for `apr search` (CRUX-A-23).
//!
//! Contract: `contracts/crux-A-23-v1.yaml`.
//!
//! Three pure algorithm-level necessary conditions:
//!
//! 1. `classify_source(repo, hub_hit, local_hit)` produces
//!    `Source::{Hub, Local, Both}` deterministically. A repo present
//!    in both halves MUST collapse to `Both` — never two rows.
//!
//! 2. `merge_search_results(hub, local)` joins the two input lists on
//!    `repo`, tags each row with its origin, sorts by (match_score DESC,
//!    downloads DESC, repo ASC), and returns a single deduplicated list.
//!    Invariant: output length == |hub ∪ local| (set union on repo).
//!
//! 3. `merge_search_results(&[], local)` returns the local half verbatim
//!    (with `Source::Local`). This is the algorithm-level necessary
//!    condition for FALSIFY-CRUX-A-23-001 (offline mode returns local-
//!    only results, never errors): when the Hub half is empty — which
//!    is exactly what `--offline` produces — the merger still yields
//!    the local cache entries as `Source::Local`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Origin of a search result row. `Both` wins over either half alone
/// so the caller does not double-render identical repos.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Source {
    Hub,
    Local,
    Both,
}

/// A search-result row before merge. Hub rows carry downloads/likes
/// counts; local rows set `cached=true` and typically have zero
/// downloads (we did not fetch Hub stats).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchHit {
    pub repo: String,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
    #[serde(default)]
    pub cached: bool,
}

/// A merged search-result row carrying origin tag and aggregated
/// downloads/likes from whichever half had the higher count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergedRow {
    pub repo: String,
    pub downloads: u64,
    pub likes: u64,
    pub source: Source,
    pub cached: bool,
}

/// Classify a repo's origin given which half(s) it appeared in.
pub fn classify_source(hub_hit: bool, local_hit: bool) -> Option<Source> {
    match (hub_hit, local_hit) {
        (true, true) => Some(Source::Both),
        (true, false) => Some(Source::Hub),
        (false, true) => Some(Source::Local),
        (false, false) => None,
    }
}

/// Merge Hub and local search halves into a single deduplicated list,
/// sorted deterministically.
///
/// Sort order: downloads DESC, then likes DESC, then repo ASC for
/// ties. Ties on (downloads, likes) are broken by repo name so
/// output is byte-stable regardless of input order — this is a
/// necessary condition for any downstream determinism tests.
pub fn merge_search_results(hub: &[SearchHit], local: &[SearchHit]) -> Vec<MergedRow> {
    let mut acc: BTreeMap<String, MergedRow> = BTreeMap::new();

    for h in hub {
        acc.insert(
            h.repo.clone(),
            MergedRow {
                repo: h.repo.clone(),
                downloads: h.downloads,
                likes: h.likes,
                source: Source::Hub,
                cached: h.cached,
            },
        );
    }

    for l in local {
        acc.entry(l.repo.clone())
            .and_modify(|row| {
                row.source = Source::Both;
                row.cached = true;
                row.downloads = row.downloads.max(l.downloads);
                row.likes = row.likes.max(l.likes);
            })
            .or_insert(MergedRow {
                repo: l.repo.clone(),
                downloads: l.downloads,
                likes: l.likes,
                source: Source::Local,
                cached: true,
            });
    }

    let mut rows: Vec<MergedRow> = acc.into_values().collect();
    rows.sort_by(|a, b| {
        b.downloads
            .cmp(&a.downloads)
            .then_with(|| b.likes.cmp(&a.likes))
            .then_with(|| a.repo.cmp(&b.repo))
    });
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(repo: &str, downloads: u64, likes: u64, cached: bool) -> SearchHit {
        SearchHit {
            repo: repo.into(),
            downloads,
            likes,
            cached,
        }
    }

    // ===== classify_source =====

    #[test]
    fn classify_hub_only() {
        assert_eq!(classify_source(true, false), Some(Source::Hub));
    }

    #[test]
    fn classify_local_only() {
        assert_eq!(classify_source(false, true), Some(Source::Local));
    }

    #[test]
    fn classify_both_halves() {
        assert_eq!(classify_source(true, true), Some(Source::Both));
    }

    #[test]
    fn classify_neither_returns_none() {
        assert_eq!(classify_source(false, false), None);
    }

    #[test]
    fn classify_is_deterministic() {
        for (h, l) in [(true, true), (true, false), (false, true), (false, false)] {
            assert_eq!(classify_source(h, l), classify_source(h, l));
        }
    }

    // ===== merge_search_results: dedup =====

    #[test]
    fn repo_in_both_halves_appears_once_as_both() {
        // FALSIFY-CRUX-A-23-002: a repo cached locally AND on Hub
        // appears exactly once with source=BOTH.
        let hub = vec![hit("bert-base-uncased", 1_000_000, 50, false)];
        let local = vec![hit("bert-base-uncased", 0, 0, true)];
        let out = merge_search_results(&hub, &local);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].source, Source::Both);
        assert_eq!(out[0].repo, "bert-base-uncased");
        assert!(out[0].cached);
    }

    #[test]
    fn hub_only_repo_keeps_hub_source() {
        let hub = vec![hit("gpt-4", 500, 10, false)];
        let out = merge_search_results(&hub, &[]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].source, Source::Hub);
        assert!(!out[0].cached);
    }

    #[test]
    fn local_only_repo_keeps_local_source() {
        // FALSIFY-CRUX-A-23-001 core sub-claim: local entries survive
        // an empty Hub half (the exact state --offline produces).
        let local = vec![hit("gpt2", 0, 0, true)];
        let out = merge_search_results(&[], &local);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].source, Source::Local);
        assert_eq!(out[0].repo, "gpt2");
        assert!(out[0].cached);
    }

    #[test]
    fn empty_hub_and_empty_local_returns_empty() {
        let out = merge_search_results(&[], &[]);
        assert!(out.is_empty());
    }

    // ===== merge_search_results: offline safety =====

    #[test]
    fn offline_mode_modeled_as_empty_hub_never_errors() {
        // --offline mode calls merge with hub=&[]. We prove the
        // merger returns Ok-shaped output with every local row
        // preserved.
        let local = vec![
            hit("gpt2", 0, 0, true),
            hit("bert-base-uncased", 0, 0, true),
        ];
        let out = merge_search_results(&[], &local);
        assert_eq!(out.len(), 2);
        for row in &out {
            assert_eq!(row.source, Source::Local);
            assert!(row.cached);
        }
    }

    #[test]
    fn local_entries_always_surface_even_with_nonempty_hub_miss() {
        // Hub may return matches for query "gpt" but miss the locally
        // cached gpt2. The merger must still surface gpt2.
        let hub = vec![
            hit("openai/gpt-oss", 5000, 10, false),
            hit("tiiuae/gpt-neox", 2000, 5, false),
        ];
        let local = vec![hit("gpt2", 0, 0, true)];
        let out = merge_search_results(&hub, &local);
        let repos: Vec<&str> = out.iter().map(|r| r.repo.as_str()).collect();
        assert!(repos.contains(&"gpt2"));
    }

    // ===== merge_search_results: sort order + determinism =====

    #[test]
    fn sort_is_by_downloads_descending() {
        let hub = vec![
            hit("c", 100, 0, false),
            hit("a", 300, 0, false),
            hit("b", 200, 0, false),
        ];
        let out = merge_search_results(&hub, &[]);
        let order: Vec<&str> = out.iter().map(|r| r.repo.as_str()).collect();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    #[test]
    fn sort_breaks_ties_by_likes_then_repo() {
        let hub = vec![
            hit("zzz", 100, 10, false),
            hit("aaa", 100, 10, false),
            hit("mmm", 100, 20, false),
        ];
        let out = merge_search_results(&hub, &[]);
        let order: Vec<&str> = out.iter().map(|r| r.repo.as_str()).collect();
        assert_eq!(order, vec!["mmm", "aaa", "zzz"]);
    }

    #[test]
    fn merge_is_deterministic() {
        let hub = vec![hit("a", 1, 1, false), hit("b", 2, 2, false)];
        let local = vec![hit("b", 0, 0, true)];
        assert_eq!(
            merge_search_results(&hub, &local),
            merge_search_results(&hub, &local)
        );
    }

    #[test]
    fn merge_preserves_max_downloads_on_both() {
        // When a repo appears in both halves, the merged row should
        // carry the higher downloads count (usually Hub, but if the
        // local half happens to carry a bigger number for any reason
        // we don't lose it).
        let hub = vec![hit("x", 100, 0, false)];
        let local = vec![hit("x", 999, 0, true)];
        let out = merge_search_results(&hub, &local);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].downloads, 999);
        assert_eq!(out[0].source, Source::Both);
    }

    #[test]
    fn merged_row_count_equals_union_of_repos() {
        // Fundamental invariant: output length is the size of
        // {hub.repo} ∪ {local.repo}. Overlapping repos collapse.
        let hub = vec![hit("a", 0, 0, false), hit("b", 0, 0, false)];
        let local = vec![hit("b", 0, 0, true), hit("c", 0, 0, true)];
        let out = merge_search_results(&hub, &local);
        // Union = {a, b, c}
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn merge_serializes_to_stable_json() {
        let hub = vec![hit("a", 1, 0, false)];
        let out = merge_search_results(&hub, &[]);
        let s = serde_json::to_string(&out).unwrap();
        let parsed: Vec<MergedRow> = serde_json::from_str(&s).unwrap();
        assert_eq!(out, parsed);
    }
}
