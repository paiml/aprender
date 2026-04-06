//! Compute IR metrics from pre-judged results (no API calls needed)
//!
//! Reads judgments JSONL (produced by Claude Code or external judge)
//! and retrieval results JSONL, computes standard IR metrics.

use super::judge::{compute_aggregate_metrics, compute_by_domain_metrics};
use super::types::{
    AggregateMetrics, ChunkJudgment, EvalOutput, EvalRunConfig, JudgmentEntry, QueryResult,
    RetrievalResultEntry,
};
use std::collections::HashMap;

/// Compute metrics from pre-judged results
///
/// Takes retrieval results (with chunks) and judgment entries (with verdicts),
/// correlates them by (query, rank), and computes standard IR metrics.
pub fn compute_metrics_from_judgments(
    retrieval_results: &[RetrievalResultEntry],
    judgments: &[JudgmentEntry],
) -> EvalOutput {
    // Index judgments by (query, rank) for fast lookup
    let mut judgment_map: HashMap<(&str, usize), &JudgmentEntry> = HashMap::new();
    for j in judgments {
        judgment_map.insert((&j.query, j.rank), j);
    }

    let mut per_query = Vec::new();

    for entry in retrieval_results {
        let mut chunk_judgments = Vec::new();

        for (rank_idx, chunk) in entry.results.iter().enumerate() {
            let rank = rank_idx + 1;

            // Look up the judgment for this (query, rank) pair
            let (relevant, reasoning) =
                if let Some(j) = judgment_map.get(&(entry.query.as_str(), rank)) {
                    (j.relevant, j.reasoning.clone())
                } else {
                    // No judgment = not relevant (unjudged chunks count against)
                    (false, "no judgment provided".to_string())
                };

            chunk_judgments.push(ChunkJudgment {
                rank,
                score: chunk.score,
                source: chunk.source.clone(),
                relevant,
                reasoning,
            });
        }

        let mrr = compute_mrr(&chunk_judgments);
        let hit_5 = chunk_judgments.iter().take(5).any(|j| j.relevant);
        let relevant_count = chunk_judgments.iter().filter(|j| j.relevant).count();

        per_query.push(QueryResult {
            query: entry.query.clone(),
            domain: entry.domain.clone(),
            mrr,
            hit_5,
            relevant_count,
            total_results: entry.results.len(),
            latency_s: entry.latency_s,
            judgments: chunk_judgments,
        });
    }

    let aggregate = compute_aggregate_metrics(&per_query);
    let by_domain = compute_by_domain_metrics(&per_query);
    let timestamp = super::judge::chrono_now();

    EvalOutput {
        timestamp,
        config: EvalRunConfig {
            num_queries: retrieval_results.len(),
            top_k: retrieval_results.first().map(|r| r.results.len()).unwrap_or(10),
            judge_model: "claude-code".to_string(),
            cache_hits: 0,
            api_calls: 0,
        },
        aggregate,
        by_domain,
        per_query,
    }
}

fn compute_mrr(judgments: &[ChunkJudgment]) -> f64 {
    for j in judgments {
        if j.relevant {
            return 1.0 / j.rank as f64;
        }
    }
    0.0
}

/// Format metrics summary to stdout
#[allow(clippy::implicit_hasher)]
pub fn format_metrics_summary(
    agg: &AggregateMetrics,
    by_domain: &HashMap<String, AggregateMetrics>,
) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    s.push_str(&"=".repeat(60));
    s.push('\n');
    s.push_str("AGGREGATE RESULTS\n");
    s.push_str(&"=".repeat(60));
    s.push('\n');
    let _ = writeln!(s, "  Queries:       {}", agg.num_queries);
    let _ = writeln!(s, "  MRR:           {:.4}", agg.mrr);
    let _ = writeln!(s, "  NDCG@5:        {:.4}", agg.ndcg_5);
    let _ = writeln!(s, "  NDCG@10:       {:.4}", agg.ndcg_10);
    let _ = writeln!(s, "  Recall@5:      {:.4}", agg.recall_5);
    let _ = writeln!(s, "  Precision@5:   {:.4}", agg.precision_5);
    let _ = writeln!(s, "  Hit Rate@5:    {:.4}", agg.hit_rate_5);
    let _ = writeln!(s, "  Hit Rate@10:   {:.4}", agg.hit_rate_10);
    let _ = writeln!(s, "  MAP:           {:.4}", agg.map);
    let _ = writeln!(s, "  Latency:       {:.3}s", agg.mean_latency_s);
    s.push('\n');
    s.push_str("BY DOMAIN:\n");

    let mut domains: Vec<_> = by_domain.iter().collect();
    domains.sort_by(|(a, _), (b, _)| a.cmp(b));
    for (domain, m) in domains {
        let _ = writeln!(
            s,
            "  {domain:12}  MRR={:.3}  NDCG@5={:.3}  Hit@5={:.3}  (n={})",
            m.mrr, m.ndcg_5, m.hit_rate_5, m.num_queries
        );
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::types::RetrievedChunk;

    fn make_retrieval_entry(query: &str, num_results: usize) -> RetrievalResultEntry {
        RetrievalResultEntry {
            query: query.to_string(),
            domain: "test".to_string(),
            course: "test-course".to_string(),
            results: (0..num_results)
                .map(|i| RetrievedChunk {
                    content: format!("chunk {i}"),
                    source: Some(format!("/test/chunk{i}.srt")),
                    score: 1.0 - i as f32 * 0.1,
                    title: None,
                    start_secs: None,
                    end_secs: None,
                })
                .collect(),
            latency_s: 0.5,
        }
    }

    #[test]
    fn test_metrics_basic() {
        let results = vec![make_retrieval_entry("what is kubernetes?", 5)];
        let judgments = vec![
            JudgmentEntry {
                query: "what is kubernetes?".to_string(),
                rank: 1,
                relevant: false,
                reasoning: "off topic".to_string(),
                source: None,
                score: None,
            },
            JudgmentEntry {
                query: "what is kubernetes?".to_string(),
                rank: 2,
                relevant: true,
                reasoning: "discusses k8s".to_string(),
                source: None,
                score: None,
            },
        ];

        let output = compute_metrics_from_judgments(&results, &judgments);
        assert_eq!(output.per_query.len(), 1);
        assert!((output.per_query[0].mrr - 0.5).abs() < 0.001);
        assert!(output.per_query[0].hit_5);
    }

    #[test]
    fn test_metrics_no_relevant() {
        let results = vec![make_retrieval_entry("obscure query", 3)];
        let judgments = vec![JudgmentEntry {
            query: "obscure query".to_string(),
            rank: 1,
            relevant: false,
            reasoning: "not relevant".to_string(),
            source: None,
            score: None,
        }];

        let output = compute_metrics_from_judgments(&results, &judgments);
        assert!((output.per_query[0].mrr).abs() < 0.001);
        assert!(!output.per_query[0].hit_5);
    }

    #[test]
    fn test_metrics_missing_judgments() {
        let results = vec![make_retrieval_entry("test query", 5)];
        // No judgments at all — everything defaults to not relevant
        let judgments = vec![];

        let output = compute_metrics_from_judgments(&results, &judgments);
        assert!((output.aggregate.mrr).abs() < 0.001);
        assert!((output.aggregate.hit_rate_5).abs() < 0.001);
    }
}
