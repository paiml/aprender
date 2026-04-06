//! Enrichment builder functions for query results.
//!
//! Split from mod.rs to keep file under 500 lines.

use super::cross_project;
use super::types::{
    self, CallSiteInfo, DiffInfo, EquationBinding, ProjectCoverage, ProofStatusInfo, ScoreInfo,
    ViolationInfo,
};
use crate::binding::BindingRegistry;
use crate::scoring;

pub(super) fn build_call_sites(
    stem: &str,
    xp_index: Option<&cross_project::CrossProjectIndex>,
) -> Vec<CallSiteInfo> {
    let Some(xp) = xp_index else {
        return Vec::new();
    };
    xp.call_sites_for(stem)
        .iter()
        .map(|cs| CallSiteInfo {
            project: cs.project.clone(),
            file: cs.file.clone(),
            line: cs.line,
            equation: cs.equation.clone(),
        })
        .collect()
}

pub(super) fn build_violations(
    entry: &types::ContractEntry,
    xp_index: Option<&cross_project::CrossProjectIndex>,
    show: bool,
) -> Vec<ViolationInfo> {
    if !show {
        return Vec::new();
    }
    let Some(xp) = xp_index else {
        return Vec::new();
    };
    let mut violations = Vec::new();

    // Check binding refs for unimplemented / partial bindings
    for br in xp.binding_refs_for(&entry.stem) {
        if br.status == "not_implemented" || br.status == "partial" {
            violations.push(ViolationInfo {
                project: br.project.clone(),
                kind: "binding_gap".to_string(),
                detail: format!("{}: {}", br.equation, br.status),
            });
        }
    }

    // If contract has unproven obligations, flag consumer projects
    if entry.obligation_count > entry.kani_count {
        let unproven = entry.obligation_count - entry.kani_count;
        let sites = xp.call_sites_for(&entry.stem);
        let mut projects: Vec<String> = sites.iter().map(|s| s.project.clone()).collect();
        projects.sort();
        projects.dedup();
        for proj in projects {
            violations.push(ViolationInfo {
                project: proj,
                kind: "unproven_obligations".to_string(),
                detail: format!(
                    "{unproven}/{} obligations lack Kani harnesses",
                    entry.obligation_count
                ),
            });
        }
    }

    violations
}

pub(super) fn build_coverage_map(
    stem: &str,
    xp_index: Option<&cross_project::CrossProjectIndex>,
    show: bool,
) -> Vec<ProjectCoverage> {
    if !show {
        return Vec::new();
    }
    let Some(xp) = xp_index else {
        return Vec::new();
    };

    let mut map: std::collections::HashMap<String, ProjectCoverage> =
        std::collections::HashMap::new();

    // Count call sites per project
    for cs in xp.call_sites_for(stem) {
        let entry = map
            .entry(cs.project.clone())
            .or_insert_with(|| ProjectCoverage {
                project: cs.project.clone(),
                call_sites: 0,
                binding_refs: 0,
                binding_implemented: 0,
                binding_total: 0,
            });
        entry.call_sites += 1;
    }

    // Count binding refs per project
    for br in xp.binding_refs_for(stem) {
        let entry = map
            .entry(br.project.clone())
            .or_insert_with(|| ProjectCoverage {
                project: br.project.clone(),
                call_sites: 0,
                binding_refs: 0,
                binding_implemented: 0,
                binding_total: 0,
            });
        entry.binding_refs += 1;
        entry.binding_total += 1;
        if br.status == "implemented" {
            entry.binding_implemented += 1;
        }
    }

    let mut result: Vec<_> = map.into_values().collect();
    result.sort_by(|a, b| a.project.cmp(&b.project));
    result
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn build_proof_status_info(entry: &types::ContractEntry) -> Option<ProofStatusInfo> {
    let path = std::path::Path::new(&entry.path);
    let contract = crate::schema::parse_contract(path).ok()?;
    let level = crate::proof_status::compute_proof_level(&contract, None);
    Some(ProofStatusInfo {
        level: level.to_string(),
        obligations: entry.obligation_count as u32,
        falsification_tests: entry.falsification_count as u32,
        kani_harnesses: entry.kani_count as u32,
        lean_proved: contract
            .verification_summary
            .as_ref()
            .map_or(0, |vs| vs.l4_lean_proved),
    })
}

pub(super) fn build_binding_info(
    entry: &types::ContractEntry,
    binding: Option<&BindingRegistry>,
) -> Vec<EquationBinding> {
    let Some(binding) = binding else {
        return entry
            .equations
            .iter()
            .map(|eq| EquationBinding {
                equation: eq.clone(),
                status: "no binding registry".into(),
                module_path: None,
            })
            .collect();
    };
    entry
        .equations
        .iter()
        .map(|eq| {
            let found = binding.find_binding(&entry.stem, eq);
            match found {
                Some(b) => EquationBinding {
                    equation: eq.clone(),
                    status: b.status.to_string(),
                    module_path: b.module_path.clone(),
                },
                None => EquationBinding {
                    equation: eq.clone(),
                    status: "unbound".into(),
                    module_path: None,
                },
            }
        })
        .collect()
}

pub(super) fn build_diff_info(entry: &types::ContractEntry) -> Option<DiffInfo> {
    let output = std::process::Command::new("git")
        .args(["log", "-1", "--format=%H %aI", "--"])
        .arg(&entry.path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let line = String::from_utf8(output.stdout).ok()?;
    let line = line.trim();
    let (hash, date) = line.split_once(' ')?;
    let date_part = date.split('T').next().unwrap_or(date);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    let days_ago = parse_iso_days_ago(date_part, now);
    Some(DiffInfo {
        last_modified: date_part.to_string(),
        days_ago,
        commit_hash: hash.to_string(),
    })
}

pub(super) fn build_score_info(entry: &types::ContractEntry) -> Option<ScoreInfo> {
    let path = std::path::Path::new(&entry.path);
    let contract = crate::schema::parse_contract(path).ok()?;
    let score = scoring::score_contract(&contract, None, &entry.stem);
    Some(ScoreInfo {
        composite: score.composite,
        grade: score.grade.to_string(),
        spec_depth: score.spec_depth,
        falsification: score.falsification_coverage,
        kani: score.kani_coverage,
        lean: score.lean_coverage,
        binding: score.binding_coverage,
    })
}

pub(super) fn parse_iso_days_ago(date: &str, now_epoch: u64) -> u64 {
    let parts: Vec<&str> = date.split('-').collect();
    if parts.len() != 3 {
        return 0;
    }
    let y: u64 = parts[0].parse().unwrap_or(0);
    let m: usize = parts[1].parse().unwrap_or(0);
    let d: u64 = parts[2].parse().unwrap_or(0);
    let days_from_epoch = y.saturating_sub(1970) * 365
        + y.saturating_sub(1969) / 4
        + month_days(m)
        + d.saturating_sub(1);
    let date_epoch = days_from_epoch * 86400;
    now_epoch.saturating_sub(date_epoch) / 86400
}

pub(super) fn month_days(m: usize) -> u64 {
    const CUMULATIVE: [u64; 13] = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    CUMULATIVE.get(m).copied().unwrap_or(0)
}

pub(super) fn parse_proof_level(s: &str) -> crate::proof_status::ProofLevel {
    match s.to_uppercase().as_str() {
        "L5" => crate::proof_status::ProofLevel::L5,
        "L4" => crate::proof_status::ProofLevel::L4,
        "L3" => crate::proof_status::ProofLevel::L3,
        "L2" => crate::proof_status::ProofLevel::L2,
        _ => crate::proof_status::ProofLevel::L1,
    }
}

pub(super) fn filter_by_project(
    sites: Vec<CallSiteInfo>,
    project: Option<&str>,
) -> Vec<CallSiteInfo> {
    let Some(p) = project else { return sites };
    sites.into_iter().filter(|s| s.project == p).collect()
}

pub(super) fn filter_violations(
    violations: Vec<ViolationInfo>,
    project: Option<&str>,
) -> Vec<ViolationInfo> {
    let Some(p) = project else { return violations };
    violations.into_iter().filter(|v| v.project == p).collect()
}

pub(super) fn filter_coverage(
    map: Vec<ProjectCoverage>,
    project: Option<&str>,
) -> Vec<ProjectCoverage> {
    let Some(p) = project else { return map };
    map.into_iter().filter(|c| c.project == p).collect()
}

#[cfg(test)]
#[path = "query_enrich_tests.rs"]
mod tests;
