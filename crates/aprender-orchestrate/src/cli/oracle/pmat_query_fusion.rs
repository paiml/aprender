//! Quality summary computation and RRF fusion for PMAT Query.

use super::pmat_query_types::{FusedResult, PmatQueryResult, QualitySummary};

/// Compute quality distribution summary from results.
pub(super) fn compute_quality_summary(results: &[PmatQueryResult]) -> QualitySummary {
    let mut grades: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut total_complexity: u64 = 0;
    let mut total_satd: u32 = 0;
    let mut min_cx = u32::MAX;
    let mut max_cx = 0u32;

    for r in results {
        *grades.entry(r.tdg_grade.clone()).or_insert(0) += 1;
        total_complexity += r.complexity as u64;
        total_satd += r.satd_count;
        min_cx = min_cx.min(r.complexity);
        max_cx = max_cx.max(r.complexity);
    }

    let avg_complexity =
        if results.is_empty() { 0.0 } else { total_complexity as f64 / results.len() as f64 };

    if results.is_empty() {
        min_cx = 0;
    }

    QualitySummary { grades, avg_complexity, total_satd, complexity_range: (min_cx, max_cx) }
}

/// Format quality summary as a one-line string.
pub(super) fn format_summary_line(summary: &QualitySummary) -> String {
    let mut grade_parts: Vec<String> = Vec::new();
    for grade in &["A", "B", "C", "D", "F"] {
        if let Some(&count) = summary.grades.get(*grade) {
            grade_parts.push(format!("{}{}", count, grade));
        }
    }
    let grades = if grade_parts.is_empty() { "none".to_string() } else { grade_parts.join(" ") };
    format!(
        "{} | Avg complexity: {:.1} | Total SATD: {} | Complexity: {}-{}",
        grades,
        summary.avg_complexity,
        summary.total_satd,
        summary.complexity_range.0,
        summary.complexity_range.1
    )
}

/// Fuse PMAT and RAG results using Reciprocal Rank Fusion (k=60).
///
/// Returns interleaved results tagged by source type.
pub(super) fn rrf_fuse_results(
    pmat_results: &[PmatQueryResult],
    rag_results: &[crate::oracle::rag::RetrievalResult],
    top_k: usize,
) -> Vec<(FusedResult, f64)> {
    let k = 60.0_f64;
    let mut scores: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut items: std::collections::HashMap<String, FusedResult> =
        std::collections::HashMap::new();

    // Accumulate PMAT ranked list
    for (rank, r) in pmat_results.iter().enumerate() {
        let id = format!("fn:{}:{}", r.file_path, r.function_name);
        *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + rank as f64 + 1.0);
        items.entry(id).or_insert_with(|| FusedResult::Function(Box::new(r.clone())));
    }

    // Accumulate RAG ranked list
    for (rank, r) in rag_results.iter().enumerate() {
        let id = format!("doc:{}", r.id);
        *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + rank as f64 + 1.0);
        items.entry(id).or_insert_with(|| FusedResult::Document {
            component: r.component.clone(),
            source: r.source.clone(),
            score: r.score,
            content: r.content.clone(),
        });
    }

    // Normalize: max possible = 2 / (k + 1) if result is rank-1 in both lists
    let max_rrf = 2.0 / (k + 1.0);
    let mut fused: Vec<(FusedResult, f64)> = scores
        .into_iter()
        .filter_map(|(id, rrf_score)| {
            let normalized = (rrf_score / max_rrf).min(1.0);
            items.remove(&id).map(|item| (item, normalized))
        })
        .collect();

    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(top_k);
    fused
}
