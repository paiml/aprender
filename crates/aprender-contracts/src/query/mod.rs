//! Contract query engine with BM25 semantic search.
//!
//! Provides fast lookup across all contracts with semantic ranking,
//! regex/literal search, and structured filters.
//!
//! Spec: `docs/specifications/sub/query.md`

pub mod cross_project;
mod index;
mod persist;
mod query_enrich;
pub mod registry;
mod types;
mod types_render;

pub use cross_project::CrossProjectIndex;
pub use index::ContractIndex;
pub use types::{
    DiffInfo, EquationBinding, ProjectCoverage, ProofStatusInfo, QueryOutput, QueryParams,
    QueryResult, ScoreInfo, SearchMode, ViolationInfo,
};

use crate::binding::BindingRegistry;
use query_enrich::{
    build_call_sites, build_coverage_map, build_violations, filter_by_project, filter_coverage,
    filter_violations,
};

/// Execute a query against a contract index.
pub fn execute(index: &ContractIndex, params: &QueryParams) -> QueryOutput {
    let scored_indices = match params.mode {
        SearchMode::Semantic => index.bm25_search(&params.query),
        SearchMode::Regex => {
            match index.regex_search(&params.query) {
                Ok(idxs) => idxs.into_iter().map(|i| (i, 1.0)).collect(),
                Err(_) => Vec::new(), // Invalid regex yields empty results
            }
        }
        SearchMode::Literal => index
            .literal_search(&params.query, params.case_sensitive)
            .into_iter()
            .map(|i| (i, 1.0))
            .collect(),
    };

    let binding = params.binding_path.as_ref().and_then(|p| {
        let content = std::fs::read_to_string(p).ok()?;
        serde_yaml::from_str::<BindingRegistry>(&content).ok()
    });

    let filtered = apply_filters(index, scored_indices, params, binding.as_ref());
    let total_matches = filtered.len();
    let limited: Vec<_> = filtered.into_iter().take(params.limit).collect();

    // Build cross-project index lazily when any cross-project flag is set
    let needs_xp = params.show_call_sites
        || params.show_violations
        || params.show_coverage_map
        || params.all_projects;
    let xp_index = if needs_xp {
        index.entries.first().map(|e| {
            let p = std::path::Path::new(&e.path);
            // Go up from contracts/foo.yaml to the repo root
            let contracts_dir = p.parent().unwrap_or(p);
            let repo_root = if contracts_dir
                .parent()
                .is_none_or(|p| p.as_os_str().is_empty())
            {
                std::path::Path::new(".")
            } else {
                contracts_dir
                    .parent()
                    .expect("parent is Some: the is_none_or branch above handles None")
            };
            let extra = params.include_project.as_ref().map(std::path::Path::new);
            cross_project::CrossProjectIndex::build_with_extra(repo_root, extra)
        })
    } else {
        None
    };

    let project_filter = params.project_filter.as_deref();
    let results: Vec<QueryResult> = limited
        .into_iter()
        .enumerate()
        .map(|(rank, (idx, relevance))| {
            build_result(
                index,
                params,
                binding.as_ref(),
                xp_index.as_ref(),
                project_filter,
                rank,
                idx,
                relevance,
            )
        })
        .collect();

    QueryOutput {
        query: params.query.clone(),
        total_matches,
        results,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_result(
    index: &ContractIndex,
    params: &QueryParams,
    binding: Option<&BindingRegistry>,
    xp_index: Option<&cross_project::CrossProjectIndex>,
    project_filter: Option<&str>,
    rank: usize,
    idx: usize,
    relevance: f64,
) -> QueryResult {
    let entry = &index.entries[idx];
    let (depends_on, depended_by) = graph_fields(index, entry, params.show_graph);
    QueryResult {
        rank: rank + 1,
        stem: entry.stem.clone(),
        path: clean_path(&entry.path),
        relevance,
        description: entry.description.clone(),
        kind: entry.kind,
        equations: entry.equations.clone(),
        obligation_count: entry.obligation_count,
        references: opt_vec(&entry.references, params.show_paper),
        depends_on,
        depended_by,
        score: params
            .show_score
            .then(|| query_enrich::build_score_info(entry))
            .flatten(),
        proof_status: params
            .show_proof_status
            .then(|| query_enrich::build_proof_status_info(entry))
            .flatten(),
        bindings: opt_binding(entry, binding, params.show_binding),
        diff: params
            .show_diff
            .then(|| query_enrich::build_diff_info(entry))
            .flatten(),
        pagerank: if params.show_pagerank {
            index.cached_pagerank(&entry.stem)
        } else {
            None
        },
        call_sites: filter_by_project(build_call_sites(&entry.stem, xp_index), project_filter),
        violations: filter_violations(
            build_violations(entry, xp_index, params.show_violations),
            project_filter,
        ),
        coverage_map: filter_coverage(
            build_coverage_map(&entry.stem, xp_index, params.show_coverage_map),
            project_filter,
        ),
    }
}

/// Clean a contract path for display: strip to `contracts/...` relative form.
fn clean_path(raw: &str) -> String {
    // Use the last occurrence of "contracts/" to handle paths like
    // /foo/contracts/crates/.../../../contracts/softmax-kernel-v1.yaml
    if let Some(idx) = raw.rfind("contracts/") {
        raw[idx..].to_string()
    } else {
        raw.to_string()
    }
}

fn opt_vec(source: &[String], include: bool) -> Vec<String> {
    if include {
        source.to_vec()
    } else {
        Vec::new()
    }
}

fn graph_fields(
    index: &ContractIndex,
    entry: &types::ContractEntry,
    show: bool,
) -> (Vec<String>, Vec<String>) {
    if !show {
        return (Vec::new(), Vec::new());
    }
    let deps = entry.depends_on.clone();
    let rev = index
        .depended_by(&entry.stem)
        .into_iter()
        .map(String::from)
        .collect();
    (deps, rev)
}

fn opt_binding(
    entry: &types::ContractEntry,
    binding: Option<&BindingRegistry>,
    show: bool,
) -> Vec<EquationBinding> {
    if show {
        query_enrich::build_binding_info(entry, binding)
    } else {
        Vec::new()
    }
}

fn apply_filters(
    index: &ContractIndex,
    results: Vec<(usize, f64)>,
    params: &QueryParams,
    binding: Option<&BindingRegistry>,
) -> Vec<(usize, f64)> {
    results
        .into_iter()
        .filter(|(idx, _)| {
            let entry = &index.entries[*idx];
            filter_obligation(entry, params.obligation_filter.as_ref())
                && filter_depends_on(entry, params.depends_on.as_ref())
                && filter_depended_by(index, entry, params.depended_by.as_ref())
                && filter_unproven(entry, params.unproven_only)
                && filter_min_score(index, entry, params.min_score)
                && filter_binding_gaps(entry, params.binding_gaps_only, binding)
                && filter_min_level(entry, params.min_level.as_deref())
                && filter_tier(entry, params.tier_filter)
                && filter_class(entry, params.class_filter)
                && filter_kind(entry, params.kind_filter)
        })
        .collect()
}

fn filter_kind(entry: &types::ContractEntry, kind: Option<crate::schema::ContractKind>) -> bool {
    match kind {
        Some(k) => entry.kind == k,
        None => true,
    }
}

fn filter_obligation(entry: &types::ContractEntry, obligation: Option<&String>) -> bool {
    match obligation {
        Some(ot) => entry.obligation_types.iter().any(|t| t == ot),
        None => true,
    }
}

fn filter_depends_on(entry: &types::ContractEntry, depends_on: Option<&String>) -> bool {
    match depends_on {
        Some(dep) => entry.depends_on.iter().any(|d| d == dep),
        None => true,
    }
}

fn filter_depended_by(
    index: &ContractIndex,
    entry: &types::ContractEntry,
    depended_by: Option<&String>,
) -> bool {
    match depended_by {
        Some(target) => index
            .get_by_stem(target)
            .is_some_and(|t| t.depends_on.contains(&entry.stem)),
        None => true,
    }
}

fn filter_min_score(
    index: &ContractIndex,
    entry: &types::ContractEntry,
    min_score: Option<f64>,
) -> bool {
    let Some(threshold) = min_score else {
        return true;
    };
    if let Some(cached) = index.cached_score(&entry.stem) {
        return cached >= threshold;
    }
    query_enrich::build_score_info(entry).is_some_and(|s| s.composite >= threshold)
}

fn filter_binding_gaps(
    entry: &types::ContractEntry,
    gaps_only: bool,
    binding: Option<&BindingRegistry>,
) -> bool {
    if !gaps_only {
        return true;
    }
    let Some(binding) = binding else {
        return false;
    };
    binding.bindings_for(&entry.stem).iter().any(|b| {
        b.status == crate::binding::ImplStatus::NotImplemented
            || b.status == crate::binding::ImplStatus::Partial
    })
}

fn filter_unproven(entry: &types::ContractEntry, unproven_only: bool) -> bool {
    if !unproven_only {
        return true;
    }
    entry.obligation_count > entry.kani_count
}

fn filter_tier(entry: &types::ContractEntry, tier: Option<u8>) -> bool {
    let Some(t) = tier else { return true };
    registry::tier_of(&entry.stem) == t
}

fn filter_class(entry: &types::ContractEntry, class: Option<char>) -> bool {
    let Some(c) = class else { return true };
    registry::classes_of(&entry.stem).contains(&c)
}

fn filter_min_level(entry: &types::ContractEntry, min_level: Option<&str>) -> bool {
    let Some(min) = min_level else { return true };
    let threshold = query_enrich::parse_proof_level(min);
    let path = std::path::Path::new(&entry.path);
    let Ok(contract) = crate::schema::parse_contract(path) else {
        return false;
    };
    let level = crate::proof_status::compute_proof_level(&contract, None);
    level >= threshold
}

#[cfg(test)]
mod tests {
    include!("query_tests.rs");
}
#[cfg(test)]
mod coverage_tests {
    include!("query_tests_coverage.rs");
}
#[cfg(test)]
mod render_tests {
    include!("query_tests_render.rs");
}
