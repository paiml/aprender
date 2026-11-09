//! Parse `gh pr list --json number,additions,deletions,labels,files,reviewDecision,statusCheckRollup`
//! into [`Candidate`]s. Pure: the caller runs `gh` (GraphQL, ETag, GH-1/GH-2) and passes the text.

use crate::Candidate;
use serde_json::Value;

/// The `--json` field list the parser expects.
pub const GH_FIELDS: &str =
    "number,additions,deletions,labels,files,reviewDecision,statusCheckRollup,headRefOid";

fn check_ok(c: &Value) -> bool {
    // CheckRun: status COMPLETED + conclusion; StatusContext: state.
    match (c["conclusion"].as_str(), c["state"].as_str()) {
        (Some(k), _) => matches!(k, "SUCCESS" | "NEUTRAL" | "SKIPPED"),
        (None, Some(s)) => s == "SUCCESS",
        _ => false,
    }
}

/// A real pass: all-skipped/neutral does not count as green.
fn is_success(c: &Value) -> bool {
    c["conclusion"].as_str() == Some("SUCCESS") || c["state"].as_str() == Some("SUCCESS")
}

fn parse_one(v: &Value) -> Result<Candidate, String> {
    let number = v["number"].as_u64().ok_or("missing number")?;
    let n = |k: &str| u32::try_from(v[k].as_u64().unwrap_or(u64::MAX)).unwrap_or(u32::MAX);
    let strs = |k: &str, f: &str| -> Vec<String> {
        v[k].as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x[f].as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };
    let checks = v["statusCheckRollup"].as_array();
    Ok(Candidate {
        number,
        additions: n("additions"),
        deletions: n("deletions"),
        green: checks.is_some_and(|c| c.iter().all(check_ok) && c.iter().any(is_success)),
        approved: v["reviewDecision"].as_str() == Some("APPROVED"),
        labels: strs("labels", "name"),
        files: strs("files", "path"),
        head: v["headRefOid"].as_str().unwrap_or_default().to_string(),
    })
}

/// Parse a JSON array of PRs. A malformed row fails the whole read (not_measured, L23).
///
/// # Errors
/// Returns the parse or field error text.
pub fn parse_prs(json: &str) -> Result<Vec<Candidate>, String> {
    let v: Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
    v.as_array()
        .ok_or_else(|| "expected a JSON array".to_string())?
        .iter()
        .map(parse_one)
        .collect()
}

#[cfg(test)]
mod tests;
