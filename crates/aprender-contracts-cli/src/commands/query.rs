//! `pv query` — Search contracts by intent, regex, or literal match.

use std::path::Path;

use provable_contracts::query::{self, ContractIndex, QueryParams, SearchMode};
use provable_contracts::schema::ContractKind;

/// All parameters needed to execute a query from the CLI.
#[allow(clippy::struct_excessive_bools)]
pub struct QueryCliParams<'a> {
    pub contract_dir: &'a Path,
    pub query_str: &'a str,
    pub regex: bool,
    pub literal: bool,
    pub case_sensitive: bool,
    pub limit: usize,
    pub obligation: Option<&'a str>,
    pub min_score: Option<f64>,
    pub min_level: Option<String>,
    pub depends_on: Option<&'a str>,
    pub depended_by: Option<&'a str>,
    pub unproven: bool,
    pub show_score: bool,
    pub show_graph: bool,
    pub show_paper: bool,
    pub show_proof_status: bool,
    pub show_binding: bool,
    pub binding_gaps: bool,
    pub binding: Option<&'a Path>,
    pub show_diff: bool,
    pub show_pagerank: bool,
    pub show_call_sites: bool,
    pub show_violations: bool,
    pub show_coverage_map: bool,
    pub project_filter: Option<&'a str>,
    pub include_project: Option<&'a Path>,
    pub tier: Option<u8>,
    pub class: Option<char>,
    pub kind: Option<&'a str>,
    pub all_projects: bool,
    pub rebuild_index: bool,
    pub format: &'a str,
    pub exit_code: bool,
}

pub fn run(p: &QueryCliParams<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let index = ContractIndex::from_directory_opts(p.contract_dir, p.rebuild_index)?;

    let mode = if p.regex {
        SearchMode::Regex
    } else if p.literal {
        SearchMode::Literal
    } else {
        SearchMode::Semantic
    };

    let params = QueryParams {
        query: p.query_str.to_string(),
        mode,
        case_sensitive: p.case_sensitive,
        limit: p.limit,
        obligation_filter: p.obligation.map(String::from),
        min_score: p.min_score,
        depends_on: p.depends_on.map(String::from),
        depended_by: p.depended_by.map(String::from),
        unproven_only: p.unproven,
        show_score: p.show_score,
        show_graph: p.show_graph,
        show_paper: p.show_paper,
        show_proof_status: p.show_proof_status,
        show_binding: p.show_binding,
        binding_path: p.binding.map(|b| b.display().to_string()),
        binding_gaps_only: p.binding_gaps,
        show_diff: p.show_diff,
        show_pagerank: p.show_pagerank,
        show_call_sites: p.show_call_sites,
        show_violations: p.show_violations,
        show_coverage_map: p.show_coverage_map,
        min_level: p.min_level.clone(),
        project_filter: p.project_filter.map(String::from),
        include_project: p.include_project.map(|p| p.display().to_string()),
        tier_filter: p.tier,
        class_filter: p.class,
        kind_filter: p.kind.map(parse_kind).transpose()?,
        all_projects: p.all_projects,
    };

    let output = query::execute(&index, &params);

    match p.format {
        "json" => println!("{}", serde_json::to_string_pretty(&output)?),
        "markdown" => print!("{}", output.to_markdown()),
        _ => print!("{output}"),
    }

    if p.exit_code && output.results.is_empty() {
        return Err("No matching contracts found".into());
    }

    Ok(())
}

fn parse_kind(s: &str) -> Result<ContractKind, Box<dyn std::error::Error>> {
    match s.to_lowercase().as_str() {
        "kernel" => Ok(ContractKind::Kernel),
        "registry" => Ok(ContractKind::Registry),
        "model-family" | "modelfamily" => Ok(ContractKind::ModelFamily),
        "pattern" => Ok(ContractKind::Pattern),
        "schema" => Ok(ContractKind::Schema),
        other => Err(format!(
            "invalid --kind value '{other}': expected one of \
             kernel, registry, model-family, pattern, schema"
        )
        .into()),
    }
}
