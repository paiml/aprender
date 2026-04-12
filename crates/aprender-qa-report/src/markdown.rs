//! RAG-Optimized Markdown Export
//!
//! Generates markdown reports optimized for semantic chunking and RAG retrieval.
//! Uses headers and structure that align with batuta's SemanticChunker separators.
//!
//! # RAG Integration
//!
//! The generated markdown uses:
//! - `## ` and `### ` headers for semantic boundaries
//! - Structured data tables for easy extraction
//! - Code blocks for reproducible commands
//! - Consistent gate ID references for cross-linking

use apr_qa_runner::{Evidence, EvidenceCollector};

use crate::mqs::MqsScore;
use crate::popperian::PopperianScore;

/// Generate RAG-optimized markdown report for a model qualification
///
/// # Arguments
///
/// * `mqs` - Model Qualification Score
/// * `popperian` - Popperian falsification score
/// * `collector` - Evidence collector with test results
///
/// # Returns
///
/// Markdown string optimized for RAG indexing
#[must_use]
pub fn generate_rag_markdown(
    mqs: &MqsScore,
    popperian: &PopperianScore,
    collector: &EvidenceCollector,
) -> String {
    let mut md = String::with_capacity(8192);

    md.push_str(&format!("# Model Qualification: {}\n\n", mqs.model_id));

    write_summary_section(&mut md, mqs, popperian);
    write_gateway_section(&mut md, mqs);
    write_category_section(&mut md, mqs);
    write_falsifications_section(&mut md, popperian);
    write_test_results_section(&mut md, collector);
    write_penalties_section(&mut md, mqs);
    write_popperian_section(&mut md, popperian);
    write_metadata_section(&mut md, mqs);

    md
}

/// Write the summary section with MQS score and key metrics
fn write_summary_section(md: &mut String, mqs: &MqsScore, popperian: &PopperianScore) {
    md.push_str("## Summary\n\n");
    md.push_str(&format!(
        "- **MQS Score**: {}/1000 ({:.1} normalized, {})\n",
        mqs.raw_score, mqs.normalized_score, mqs.grade
    ));
    md.push_str(&format!("- **Status**: {}\n", qualification_status(mqs)));
    md.push_str(&format!(
        "- **Tests**: {} passed / {} failed / {} total\n",
        mqs.tests_passed, mqs.tests_failed, mqs.total_tests
    ));
    md.push_str(&format!(
        "- **Black Swans**: {}\n",
        popperian.black_swan_count
    ));
    md.push_str(&format!(
        "- **Corroboration Rate**: {:.1}%\n\n",
        popperian.corroboration_ratio * 100.0
    ));
}

/// Write the gateway checks table section
fn write_gateway_section(md: &mut String, mqs: &MqsScore) {
    md.push_str("## Gateway Checks\n\n");
    md.push_str("| Gateway | Status | Description |\n");
    md.push_str("|---------|--------|-------------|\n");
    for gw in &mqs.gateways {
        let status = if gw.passed { "✓ PASS" } else { "✗ FAIL" };
        let desc = if let Some(reason) = &gw.failure_reason {
            format!("{} - {}", gw.description, reason)
        } else {
            gw.description.clone()
        };
        md.push_str(&format!(
            "| {} | {} | {} |\n",
            escape_md_table(&gw.id),
            status,
            escape_md_table(&desc)
        ));
    }
    md.push('\n');
}

/// Write the category scores table section
fn write_category_section(md: &mut String, mqs: &MqsScore) {
    md.push_str("## Category Scores\n\n");
    md.push_str("| Category | Score | Max | Percentage |\n");
    md.push_str("|----------|-------|-----|------------|\n");
    for (cat, (score, max)) in mqs.categories.breakdown() {
        let pct = if max > 0 {
            (score as f64 / max as f64) * 100.0
        } else {
            0.0
        };
        md.push_str(&format!(
            "| {} | {} | {} | {:.1}% |\n",
            cat, score, max, pct
        ));
    }
    md.push('\n');
}

/// Write the falsifications section with gate IDs and evidence
fn write_falsifications_section(md: &mut String, popperian: &PopperianScore) {
    if popperian.falsifications.is_empty() {
        return;
    }
    md.push_str("## Falsifications\n\n");
    for (i, falsification) in popperian.falsifications.iter().enumerate() {
        md.push_str(&format!("### {}: {}\n\n", i + 1, falsification.gate_id));
        md.push_str(&format!("- **Hypothesis**: {}\n", falsification.hypothesis));
        md.push_str(&format!("- **Evidence**: {}\n", falsification.evidence));
        md.push_str(&format!("- **Severity**: {}/5\n", falsification.severity));
        if falsification.is_black_swan {
            md.push_str("- **Black Swan**: Yes (rare, high-impact failure)\n");
        }
        md.push_str(&format!(
            "- **Occurrences**: {}\n\n",
            falsification.occurrence_count
        ));
    }
}

/// Write the test results section grouped by category
fn write_test_results_section(md: &mut String, collector: &EvidenceCollector) {
    md.push_str("## Test Results by Category\n\n");
    for category in &["QUAL", "PERF", "STAB", "COMP", "EDGE", "REGR"] {
        let category_evidence: Vec<&Evidence> = collector
            .all()
            .iter()
            .filter(|e| extract_category(&e.gate_id) == *category)
            .collect();

        if category_evidence.is_empty() {
            continue;
        }

        md.push_str(&format!("### {} Tests\n\n", category));

        let passed = category_evidence
            .iter()
            .filter(|e| e.outcome.is_pass())
            .count();
        let total = category_evidence.len();
        md.push_str(&format!(
            "Pass rate: {}/{} ({:.1}%)\n\n",
            passed,
            total,
            (passed as f64 / total as f64) * 100.0
        ));

        write_category_failures(md, &category_evidence);
    }
}

/// Write failure details for a single category, capped at 10 entries
fn write_category_failures(md: &mut String, evidence: &[&Evidence]) {
    let failures: Vec<_> = evidence.iter().filter(|e| e.outcome.is_fail()).collect();

    if failures.is_empty() {
        return;
    }

    md.push_str("**Failures:**\n\n");
    for e in failures.iter().take(10) {
        md.push_str(&format!(
            "- `{}`: {} ({:?}, {}ms)\n",
            e.gate_id,
            e.reason.replace('`', "'"),
            e.outcome,
            e.metrics.duration_ms
        ));
    }
    if failures.len() > 10 {
        md.push_str(&format!(
            "- ... and {} more failures\n",
            failures.len() - 10
        ));
    }
    md.push('\n');
}

/// Write the penalties table section
fn write_penalties_section(md: &mut String, mqs: &MqsScore) {
    if mqs.penalties.is_empty() {
        return;
    }
    md.push_str("## Penalties Applied\n\n");
    md.push_str("| Code | Description | Points |\n");
    md.push_str("|------|-------------|--------|\n");
    for penalty in &mqs.penalties {
        md.push_str(&format!(
            "| {} | {} | -{} |\n",
            escape_md_table(&penalty.code),
            escape_md_table(&penalty.description),
            penalty.points
        ));
    }
    md.push_str(&format!(
        "\n**Total Penalty**: -{} points\n\n",
        mqs.total_penalty
    ));
}

/// Write the Popperian analysis section with statistical metrics
fn write_popperian_section(md: &mut String, popperian: &PopperianScore) {
    md.push_str("## Popperian Analysis\n\n");
    md.push_str(&format!(
        "- **Hypotheses Tested**: {}\n",
        popperian.hypotheses_tested
    ));
    md.push_str(&format!("- **Corroborated**: {}\n", popperian.corroborated));
    md.push_str(&format!("- **Falsified**: {}\n", popperian.falsified));
    md.push_str(&format!(
        "- **Severity-Weighted Score**: {:.2}\n",
        popperian.severity_weighted_score
    ));
    md.push_str(&format!(
        "- **Confidence Level**: {:.1}%\n",
        popperian.confidence_level * 100.0
    ));
    md.push_str(&format!(
        "- **Reproducibility Index**: {:.2}\n\n",
        popperian.reproducibility_index
    ));
}

/// Write the metadata section with model ID and qualification status
fn write_metadata_section(md: &mut String, mqs: &MqsScore) {
    md.push_str("## Metadata\n\n");
    md.push_str(&format!("- **Model ID**: {}\n", mqs.model_id));
    md.push_str(&format!("- **Gateways Passed**: {}\n", mqs.gateways_passed));
    md.push_str(&format!("- **Qualifies**: {}\n", mqs.qualifies()));
    md.push_str(&format!(
        "- **Production Ready**: {}\n",
        mqs.is_production_ready()
    ));
}

/// Generate a compact summary for index files
#[must_use]
pub fn generate_index_entry(mqs: &MqsScore) -> String {
    format!(
        "| {} | {}/1000 | {} | {} | {} |\n",
        escape_md_table(&mqs.model_id),
        mqs.raw_score,
        escape_md_table(&mqs.grade),
        qualification_status(mqs),
        if mqs.is_production_ready() {
            "Yes"
        } else {
            "No"
        }
    )
}

/// Get qualification status string.
///
/// Seven tiers (gateways must pass; first match wins):
/// - Gateway failure  -> REJECTED (Gateway Failure)
/// - Score >= 90      -> CERTIFIED
/// - Score >= 85      -> CERTIFIED (Conditional)
/// - Score >= 80      -> QUALIFIED (Conditional)
/// - Score >= 70      -> PROVISIONAL
/// - Score >= 60      -> UNDER REVIEW
/// - Score >= 50      -> NEEDS IMPROVEMENT
/// - Score < 50       -> REJECTED
///
/// Score-based qualification tiers (gateways must pass). Checked in order; first match wins.
const QUALIFICATION_TIERS: &[(f64, &str)] = &[
    (90.0, "CERTIFIED"),
    (85.0, "CERTIFIED (Conditional)"),
    (80.0, "QUALIFIED (Conditional)"),
    (70.0, "PROVISIONAL"),
    (60.0, "UNDER REVIEW"),
    (50.0, "NEEDS IMPROVEMENT"),
];

/// Determine qualification status string from MQS score tiers
fn qualification_status(mqs: &MqsScore) -> &'static str {
    if !mqs.gateways_passed {
        return "REJECTED (Gateway Failure)";
    }
    QUALIFICATION_TIERS
        .iter()
        .find(|&&(min_score, _)| mqs.normalized_score >= min_score)
        .map_or("REJECTED", |&(_, label)| label)
}

/// Extract MQS category from gate ID (delegates to canonical implementation).
fn extract_category(gate_id: &str) -> String {
    crate::MqsCalculator::extract_category(gate_id)
}

/// Generate evidence detail markdown for a single test
#[must_use]
pub fn generate_evidence_detail(evidence: &Evidence) -> String {
    let mut md = String::with_capacity(512);

    md.push_str(&format!("### {}\n\n", evidence.gate_id));
    md.push_str(&format!("- **Outcome**: {:?}\n", evidence.outcome));
    md.push_str(&format!("- **Reason**: {}\n", evidence.reason));
    md.push_str(&format!(
        "- **Duration**: {}ms\n",
        evidence.metrics.duration_ms
    ));

    if let Some(tps) = evidence.metrics.tokens_per_second {
        md.push_str(&format!("- **Tokens/sec**: {:.1}\n", tps));
    }
    if let Some(ttft) = evidence.metrics.time_to_first_token_ms {
        md.push_str(&format!("- **Time to First Token**: {:.1}ms\n", ttft));
    }
    if let Some(mem) = evidence.metrics.memory_peak_mb {
        md.push_str(&format!("- **Peak Memory**: {} MB\n", mem));
    }

    // Scenario details
    md.push_str("\n**Scenario**:\n");
    md.push_str(&format!("- Model: {}\n", evidence.scenario.model));
    md.push_str(&format!("- Backend: {:?}\n", evidence.scenario.backend));
    md.push_str(&format!("- Format: {:?}\n", evidence.scenario.format));
    md.push_str(&format!("- Seed: {}\n", evidence.scenario.seed));

    if !evidence.output.is_empty() {
        let output_preview: String = evidence.output.chars().take(200).collect();
        md.push_str(&format!(
            "\n**Output Preview**:\n```\n{}\n```\n",
            output_preview
        ));
    }

    md.push('\n');
    md
}

/// Escape characters that break markdown table cells (pipes, newlines, backslashes).
fn escape_md_table(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace('\n', " ")
        .replace('\r', "")
}

#[cfg(test)]
#[path = "markdown_tests.rs"]
mod markdown_tests;
