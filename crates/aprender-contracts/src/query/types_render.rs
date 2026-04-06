//! Display and Markdown rendering for query result types.
//!
//! Split from types.rs to keep per-function complexity under thresholds.

use super::types::{
    CallSiteInfo, DiffInfo, EquationBinding, ProjectCoverage, ProofStatusInfo, QueryOutput,
    QueryResult, ScoreInfo, ViolationInfo,
};

impl QueryResult {
    pub(crate) fn fmt_enrichment(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt_score(f, self.score.as_ref())?;
        fmt_proof_status(f, self.proof_status.as_ref())?;
        fmt_bindings(f, &self.bindings)?;
        fmt_diff(f, self.diff.as_ref())?;
        fmt_pagerank(f, self.pagerank)?;
        fmt_call_sites(f, &self.call_sites)?;
        fmt_violations(f, &self.violations)?;
        fmt_coverage_map(f, &self.coverage_map)?;
        Ok(())
    }

    pub(crate) fn to_markdown_item(&self) -> String {
        use crate::schema::ContractKind;
        let mut out = format!("### {}. {}\n\n", self.rank, self.stem);
        out.push_str(&format!("- **Relevance:** {:.2}\n", self.relevance));
        if self.kind != ContractKind::Kernel {
            out.push_str(&format!("- **Kind:** {}\n", self.kind));
        }
        if !self.equations.is_empty() {
            out.push_str(&format!("- **Equations:** {}\n", self.equations.join(", ")));
        }
        out.push_str(&format!("- **Obligations:** {}\n", self.obligation_count));
        md_score(&mut out, self.score.as_ref());
        md_proof_status(&mut out, self.proof_status.as_ref());
        md_bindings(&mut out, &self.bindings);
        md_diff(&mut out, self.diff.as_ref());
        md_pagerank(&mut out, self.pagerank);
        md_call_sites(&mut out, &self.call_sites);
        md_violations(&mut out, &self.violations);
        md_coverage_map(&mut out, &self.coverage_map);
        if !self.references.is_empty() {
            out.push_str(&format!("- **Papers:** {}\n", self.references.join("; ")));
        }
        if !self.depends_on.is_empty() {
            out.push_str(&format!(
                "- **Depends on:** {}\n",
                self.depends_on.join(", ")
            ));
        }
        if !self.depended_by.is_empty() {
            out.push_str(&format!(
                "- **Depended by:** {}\n",
                self.depended_by.join(", ")
            ));
        }
        out.push('\n');
        out
    }
}

impl std::fmt::Display for QueryResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::schema::ContractKind;
        let kind_tag = if self.kind == ContractKind::Kernel {
            String::new()
        } else {
            format!(" [{}]", self.kind)
        };
        writeln!(
            f,
            "[{}] {}{} (relevance: {:.2})",
            self.rank, self.stem, kind_tag, self.relevance
        )?;
        writeln!(f, "    {}", self.description)?;
        if !self.equations.is_empty() {
            writeln!(f, "    Equations: {}", self.equations.join(", "))?;
        }
        writeln!(f, "    Obligations: {}", self.obligation_count)?;
        if !self.references.is_empty() {
            writeln!(f, "    Papers: {}", self.references.join("; "))?;
        }
        self.fmt_enrichment(f)?;
        if !self.depends_on.is_empty() {
            writeln!(f, "    Depends on: {}", self.depends_on.join(", "))?;
        }
        if !self.depended_by.is_empty() {
            writeln!(f, "    Depended by: {}", self.depended_by.join(", "))?;
        }
        Ok(())
    }
}

impl QueryOutput {
    /// Render results as Markdown (for `--format markdown`).
    pub fn to_markdown(&self) -> String {
        let mut out = format!("## Query: \"{}\"\n\n", self.query);
        for r in &self.results {
            out.push_str(&r.to_markdown_item());
        }
        out.push_str(&format!("*{} matches*\n", self.total_matches));
        out
    }
}

impl std::fmt::Display for QueryOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for r in &self.results {
            write!(f, "{r}")?;
            writeln!(f, "    ---")?;
        }
        writeln!(f, "\n{} matches for \"{}\"", self.total_matches, self.query)
    }
}

// --- Text format helpers ---

fn fmt_score(f: &mut std::fmt::Formatter<'_>, score: Option<&ScoreInfo>) -> std::fmt::Result {
    let Some(s) = score else { return Ok(()) };
    writeln!(f, "    Score: {:.2} (Grade {})", s.composite, s.grade)?;
    writeln!(
        f,
        "    Spec: {:.2} | Falsify: {:.2} | Kani: {:.2} | Lean: {:.2} | Bind: {:.2}",
        s.spec_depth, s.falsification, s.kani, s.lean, s.binding
    )
}

fn fmt_proof_status(
    f: &mut std::fmt::Formatter<'_>,
    ps: Option<&ProofStatusInfo>,
) -> std::fmt::Result {
    let Some(ps) = ps else { return Ok(()) };
    writeln!(
        f,
        "    Proof Level: {} (ob:{} ft:{} kani:{} lean:{})",
        ps.level, ps.obligations, ps.falsification_tests, ps.kani_harnesses, ps.lean_proved
    )
}

fn fmt_bindings(f: &mut std::fmt::Formatter<'_>, bindings: &[EquationBinding]) -> std::fmt::Result {
    if bindings.is_empty() {
        return Ok(());
    }
    writeln!(f, "    Bindings:")?;
    for b in bindings {
        let loc = b.module_path.as_deref().unwrap_or("unbound");
        writeln!(f, "      {}: {} ({})", b.equation, b.status, loc)?;
    }
    Ok(())
}

fn fmt_diff(f: &mut std::fmt::Formatter<'_>, diff: Option<&DiffInfo>) -> std::fmt::Result {
    let Some(d) = diff else { return Ok(()) };
    writeln!(
        f,
        "    Last modified: {} ({} days ago, {})",
        d.last_modified,
        d.days_ago,
        &d.commit_hash[..7.min(d.commit_hash.len())]
    )
}

fn fmt_pagerank(f: &mut std::fmt::Formatter<'_>, pr: Option<f64>) -> std::fmt::Result {
    let Some(pr) = pr else { return Ok(()) };
    writeln!(f, "    PageRank: {pr:.4}")
}

fn fmt_call_sites(f: &mut std::fmt::Formatter<'_>, sites: &[CallSiteInfo]) -> std::fmt::Result {
    if sites.is_empty() {
        return Ok(());
    }
    writeln!(
        f,
        "    Call sites ({} projects):",
        count_unique_projects(sites)
    )?;
    for cs in sites {
        let eq = cs.equation.as_deref().unwrap_or("");
        writeln!(f, "      {}/{}:{} {eq}", cs.project, cs.file, cs.line)?;
    }
    Ok(())
}

fn fmt_violations(
    f: &mut std::fmt::Formatter<'_>,
    violations: &[ViolationInfo],
) -> std::fmt::Result {
    if violations.is_empty() {
        return Ok(());
    }
    writeln!(f, "    Violations ({}):", violations.len())?;
    for v in violations {
        writeln!(f, "      {}: {} — {}", v.project, v.kind, v.detail)?;
    }
    Ok(())
}

fn fmt_coverage_map(f: &mut std::fmt::Formatter<'_>, map: &[ProjectCoverage]) -> std::fmt::Result {
    if map.is_empty() {
        return Ok(());
    }
    writeln!(f, "    Coverage map:")?;
    for c in map {
        let bar = coverage_bar(c.binding_implemented, c.binding_total);
        writeln!(
            f,
            "      {:<12} {bar} ({} sites, {}/{} bound)",
            c.project, c.call_sites, c.binding_implemented, c.binding_total
        )?;
    }
    Ok(())
}

// --- Markdown format helpers ---

fn md_score(out: &mut String, score: Option<&ScoreInfo>) {
    let Some(s) = score else { return };
    out.push_str(&format!(
        "- **Score:** {:.2} (Grade {})\n",
        s.composite, s.grade
    ));
    out.push_str(&format!(
        "- Spec: {:.2} | Falsify: {:.2} | Kani: {:.2} | Lean: {:.2} | Bind: {:.2}\n",
        s.spec_depth, s.falsification, s.kani, s.lean, s.binding
    ));
}

fn md_proof_status(out: &mut String, ps: Option<&ProofStatusInfo>) {
    let Some(ps) = ps else { return };
    out.push_str(&format!(
        "- **Proof Level:** {} (ob:{} ft:{} kani:{} lean:{})\n",
        ps.level, ps.obligations, ps.falsification_tests, ps.kani_harnesses, ps.lean_proved
    ));
}

fn md_bindings(out: &mut String, bindings: &[EquationBinding]) {
    if bindings.is_empty() {
        return;
    }
    out.push_str("- **Bindings:**\n");
    for b in bindings {
        let loc = b.module_path.as_deref().unwrap_or("unbound");
        out.push_str(&format!("  - `{}`: {} (`{}`)\n", b.equation, b.status, loc));
    }
}

fn md_diff(out: &mut String, diff: Option<&DiffInfo>) {
    let Some(d) = diff else { return };
    out.push_str(&format!(
        "- **Last modified:** {} ({} days ago)\n",
        d.last_modified, d.days_ago
    ));
}

fn md_pagerank(out: &mut String, pr: Option<f64>) {
    let Some(pr) = pr else { return };
    out.push_str(&format!("- **PageRank:** {pr:.4}\n"));
}

fn md_call_sites(out: &mut String, sites: &[CallSiteInfo]) {
    if sites.is_empty() {
        return;
    }
    out.push_str(&format!(
        "- **Call sites** ({} projects):\n",
        count_unique_projects(sites)
    ));
    for cs in sites {
        let eq = cs
            .equation
            .as_deref()
            .map_or(String::new(), |e| format!(" eq={e}"));
        out.push_str(&format!(
            "  - `{}/{}:{}`{eq}\n",
            cs.project, cs.file, cs.line
        ));
    }
}

fn md_violations(out: &mut String, violations: &[ViolationInfo]) {
    if violations.is_empty() {
        return;
    }
    out.push_str(&format!("- **Violations ({}):**\n", violations.len()));
    for v in violations {
        out.push_str(&format!("  - `{}`: {} — {}\n", v.project, v.kind, v.detail));
    }
}

fn md_coverage_map(out: &mut String, map: &[ProjectCoverage]) {
    if map.is_empty() {
        return;
    }
    out.push_str("- **Coverage map:**\n");
    for c in map {
        let pct = if c.binding_total > 0 {
            #[allow(clippy::cast_precision_loss)]
            {
                c.binding_implemented as f64 / c.binding_total as f64 * 100.0
            }
        } else {
            0.0
        };
        out.push_str(&format!(
            "  - `{}`: {:.0}% ({}/{} bound, {} sites)\n",
            c.project, pct, c.binding_implemented, c.binding_total, c.call_sites
        ));
    }
}

// --- Shared helpers ---

pub(crate) fn coverage_bar(implemented: usize, total: usize) -> String {
    if total == 0 {
        return "--".to_string();
    }
    let filled = (implemented * 6 + total / 2) / total.max(1);
    let filled = filled.min(6);
    let empty = 6 - filled;
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

fn count_unique_projects(sites: &[CallSiteInfo]) -> usize {
    let mut seen = std::collections::HashSet::new();
    for s in sites {
        seen.insert(&s.project);
    }
    seen.len()
}
