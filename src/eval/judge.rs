//! LLM-as-judge for content-based relevance scoring

use super::client::AnthropicClient;
use super::types::{
    AggregateMetrics, ChunkJudgment, EvalOutput, EvalRunConfig, JudgeCache, JudgeVerdict,
    QueryResult, RetrievalResultEntry,
};
use std::collections::HashMap;

const JUDGE_SYSTEM: &str = "You judge relevance for information retrieval evaluation.
Given a QUERY and DOCUMENT (video transcript chunk), decide if the document
is RELEVANT — contains information that helps answer the query, even partially.
RELEVANT: discusses the specific topic with substantive content.
NOT RELEVANT: merely mentions a keyword, covers a different topic, or is navigational.
Respond ONLY with JSON: {\"relevant\": true, \"reasoning\": \"brief explanation\"} or {\"relevant\": false, \"reasoning\": \"brief explanation\"}";

/// LLM-based relevance judge
pub struct RelevanceJudge {
    client: AnthropicClient,
    model: String,
    cache: JudgeCache,
}

impl RelevanceJudge {
    /// Create a new judge
    pub fn new(client: AnthropicClient, model: &str, cache: JudgeCache) -> Self {
        Self { client, model: model.to_string(), cache }
    }

    /// Judge whether a chunk is relevant to a query
    pub async fn judge(&mut self, query: &str, content: &str) -> Result<JudgeVerdict, String> {
        // Check cache first
        if let Some(cached) = self.cache.get(query, content) {
            return Ok(cached.clone());
        }

        let user_msg = format!("QUERY: {query}\nDOCUMENT:\n---\n{content}\n---");

        let result = self.client.complete(&self.model, Some(JUDGE_SYSTEM), &user_msg, 200).await?;

        let verdict = parse_verdict(&result.text)?;

        // Cache the result
        self.cache.insert(query, content, verdict.clone(), &self.model);

        Ok(verdict)
    }

    /// Get the current cache (for saving)
    pub fn cache(&self) -> &JudgeCache {
        &self.cache
    }

    /// Run full evaluation: judge all retrieval results and compute metrics
    pub async fn evaluate(
        &mut self,
        results: &[RetrievalResultEntry],
        top_k: usize,
    ) -> Result<EvalOutput, String> {
        let total = results.len();
        let mut per_query = Vec::new();
        let mut cache_hits = 0usize;
        let mut api_calls = 0usize;
        let _cache_size_before = self.cache.entries.len();

        for (i, entry) in results.iter().enumerate() {
            eprint!("[{}/{}] {}...", i + 1, total, &entry.query[..entry.query.len().min(60)]);

            let mut judgments = Vec::new();
            let chunks_to_judge = entry.results.len().min(top_k);

            for (rank, chunk) in entry.results.iter().take(chunks_to_judge).enumerate() {
                let was_cached = self.cache.get(&entry.query, &chunk.content).is_some();

                let verdict = self.judge(&entry.query, &chunk.content).await?;

                if was_cached {
                    cache_hits += 1;
                } else {
                    api_calls += 1;
                }

                judgments.push(ChunkJudgment {
                    rank: rank + 1,
                    score: chunk.score,
                    source: chunk.source.clone(),
                    relevant: verdict.relevant,
                    reasoning: verdict.reasoning,
                });
            }

            let relevant_count = judgments.iter().filter(|j| j.relevant).count();
            let mrr = compute_mrr(&judgments);
            let hit_5 = judgments.iter().take(5).any(|j| j.relevant);

            let status = if hit_5 { "HIT" } else { "MISS" };
            eprintln!(" [{status}] rel={relevant_count}/{chunks_to_judge} MRR={mrr:.2}");

            per_query.push(QueryResult {
                query: entry.query.clone(),
                domain: entry.domain.clone(),
                mrr,
                hit_5,
                relevant_count,
                total_results: entry.results.len(),
                latency_s: entry.latency_s,
                judgments,
            });
        }

        // Compute aggregates
        let aggregate = compute_aggregate_metrics(&per_query);
        let by_domain = compute_by_domain_metrics(&per_query);

        let timestamp = chrono_now();

        eprintln!("\n{}", format_summary(&aggregate, &by_domain));
        eprintln!(
            "Cache: {} hits, {} new calls ({} total cached)",
            cache_hits,
            api_calls,
            self.cache.entries.len()
        );

        Ok(EvalOutput {
            timestamp,
            config: EvalRunConfig {
                num_queries: total,
                top_k,
                judge_model: self.model.clone(),
                cache_hits,
                api_calls,
            },
            aggregate,
            by_domain,
            per_query,
        })
    }
}

fn parse_verdict(text: &str) -> Result<JudgeVerdict, String> {
    // Try to extract JSON from the response
    let trimmed = text.trim();

    // Try direct parse first
    if let Ok(v) = serde_json::from_str::<JudgeVerdict>(trimmed) {
        return Ok(v);
    }

    // Try to find JSON in the response (model sometimes wraps in markdown)
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            let json_str = &trimmed[start..=end];
            if let Ok(v) = serde_json::from_str::<JudgeVerdict>(json_str) {
                return Ok(v);
            }
        }
    }

    // Fallback: check for keywords
    let lower = trimmed.to_lowercase();
    if lower.contains("not relevant") || lower.contains("\"relevant\": false") {
        return Ok(JudgeVerdict { relevant: false, reasoning: trimmed.to_string() });
    }
    if lower.contains("relevant") || lower.contains("\"relevant\": true") {
        return Ok(JudgeVerdict { relevant: true, reasoning: trimmed.to_string() });
    }

    Err(format!("Could not parse judge response: {trimmed}"))
}

fn compute_mrr(judgments: &[ChunkJudgment]) -> f64 {
    for j in judgments {
        if j.relevant {
            return 1.0 / j.rank as f64;
        }
    }
    0.0
}

fn compute_ndcg(judgments: &[ChunkJudgment], k: usize) -> f64 {
    let dcg: f64 = judgments
        .iter()
        .take(k)
        .filter(|j| j.relevant)
        .map(|j| 1.0 / (j.rank as f64 + 1.0).log2())
        .sum();

    let relevant_count = judgments.iter().take(k).filter(|j| j.relevant).count();
    let idcg: f64 = (0..relevant_count.min(k)).map(|r| 1.0 / (r as f64 + 2.0).log2()).sum();

    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

fn compute_average_precision(judgments: &[ChunkJudgment]) -> f64 {
    let mut sum = 0.0;
    let mut rel_count: usize = 0;

    for (i, j) in judgments.iter().enumerate() {
        if j.relevant {
            rel_count += 1;
            sum += rel_count as f64 / (i + 1) as f64;
        }
    }

    let total_relevant = judgments.iter().filter(|j| j.relevant).count();
    if total_relevant == 0 {
        0.0
    } else {
        sum / total_relevant as f64
    }
}

/// Compute aggregate metrics across all queries (public for metrics module)
pub fn compute_aggregate_metrics(queries: &[QueryResult]) -> AggregateMetrics {
    if queries.is_empty() {
        return AggregateMetrics::default();
    }
    let n = queries.len() as f64;

    let mrr: f64 = queries.iter().map(|q| q.mrr).sum::<f64>() / n;
    let hit_5: f64 = queries.iter().filter(|q| q.hit_5).count() as f64 / n;

    let hit_10: f64 =
        queries.iter().filter(|q| q.judgments.iter().take(10).any(|j| j.relevant)).count() as f64
            / n;

    let ndcg_5: f64 = queries.iter().map(|q| compute_ndcg(&q.judgments, 5)).sum::<f64>() / n;

    let ndcg_10: f64 = queries.iter().map(|q| compute_ndcg(&q.judgments, 10)).sum::<f64>() / n;

    let recall_5: f64 = queries
        .iter()
        .map(|q| {
            let rel_in_5 = q.judgments.iter().take(5).filter(|j| j.relevant).count();
            let total_rel = q.judgments.iter().filter(|j| j.relevant).count().max(1);
            rel_in_5 as f64 / total_rel as f64
        })
        .sum::<f64>()
        / n;

    let precision_5: f64 = queries
        .iter()
        .map(|q| {
            let k = q.judgments.len().min(5);
            if k == 0 {
                return 0.0;
            }
            q.judgments.iter().take(5).filter(|j| j.relevant).count() as f64 / k as f64
        })
        .sum::<f64>()
        / n;

    let map: f64 = queries.iter().map(|q| compute_average_precision(&q.judgments)).sum::<f64>() / n;

    let mean_latency: f64 = queries.iter().map(|q| q.latency_s).sum::<f64>() / n;

    AggregateMetrics {
        num_queries: queries.len(),
        mrr: round4(mrr),
        ndcg_5: round4(ndcg_5),
        ndcg_10: round4(ndcg_10),
        recall_5: round4(recall_5),
        precision_5: round4(precision_5),
        hit_rate_5: round4(hit_5),
        hit_rate_10: round4(hit_10),
        map: round4(map),
        mean_latency_s: round4(mean_latency),
    }
}

/// Compute per-domain metrics (public for metrics module)
pub fn compute_by_domain_metrics(queries: &[QueryResult]) -> HashMap<String, AggregateMetrics> {
    let mut by_domain: HashMap<String, Vec<&QueryResult>> = HashMap::new();
    for q in queries {
        by_domain.entry(q.domain.clone()).or_default().push(q);
    }

    by_domain
        .into_iter()
        .map(|(domain, qs)| {
            let owned: Vec<QueryResult> = qs.into_iter().cloned().collect();
            (domain, compute_aggregate_metrics(&owned))
        })
        .collect()
}

fn format_summary(agg: &AggregateMetrics, by_domain: &HashMap<String, AggregateMetrics>) -> String {
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

fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

/// Simple UTC timestamp without chrono crate (public for metrics module)
pub fn chrono_now() -> String {
    // Simple UTC timestamp without chrono crate
    let dur =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let secs = dur.as_secs();
    // Basic ISO 8601 from epoch
    let days = secs / 86400;
    let remaining = secs % 86400;
    let hours = remaining / 3600;
    let minutes = (remaining % 3600) / 60;
    let seconds = remaining % 60;

    // Days since 1970-01-01
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    // Simple Gregorian calendar conversion
    let mut year = 1970;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let month_days: &[u64] = if is_leap(year) {
        &[31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        &[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 0;
    for (i, &md) in month_days.iter().enumerate() {
        if days < md {
            month = i as u64 + 1;
            break;
        }
        days -= md;
    }
    if month == 0 {
        month = 12;
    }
    (year, month, days + 1)
}

fn is_leap(year: u64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Compare two eval outputs and print deltas
pub fn compare_results(baseline: &EvalOutput, candidate: &EvalOutput) -> String {
    use std::fmt::Write;
    let b = &baseline.aggregate;
    let c = &candidate.aggregate;

    let mut s = String::new();
    s.push_str(&"=".repeat(60));
    s.push('\n');
    s.push_str("COMPARISON: baseline \u{2192} candidate\n");
    s.push_str(&"=".repeat(60));
    s.push('\n');

    let metrics = [
        ("MRR", b.mrr, c.mrr),
        ("NDCG@5", b.ndcg_5, c.ndcg_5),
        ("NDCG@10", b.ndcg_10, c.ndcg_10),
        ("Recall@5", b.recall_5, c.recall_5),
        ("Precision@5", b.precision_5, c.precision_5),
        ("Hit Rate@5", b.hit_rate_5, c.hit_rate_5),
        ("Hit Rate@10", b.hit_rate_10, c.hit_rate_10),
        ("MAP", b.map, c.map),
    ];

    for (name, base, cand) in metrics {
        let delta = cand - base;
        let arrow = if delta > 0.001 {
            "^"
        } else if delta < -0.001 {
            "v"
        } else {
            "="
        };
        let _ = writeln!(s, "  {name:14}  {base:.4} \u{2192} {cand:.4}  ({delta:+.4}) {arrow}");
    }

    s
}

/// Check if results meet minimum thresholds (regression gate)
pub fn check_gate(output: &EvalOutput, min_mrr: f64, min_hit5: f64) -> Result<(), String> {
    let a = &output.aggregate;
    let mut failures = Vec::new();

    if a.mrr < min_mrr {
        failures.push(format!("MRR {:.4} < {min_mrr:.4}", a.mrr));
    }
    if a.hit_rate_5 < min_hit5 {
        failures.push(format!("Hit@5 {:.4} < {min_hit5:.4}", a.hit_rate_5));
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!("Regression gate FAILED: {}", failures.join(", ")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_verdict_json() {
        let v = parse_verdict(r#"{"relevant": true, "reasoning": "discusses topic"}"#).unwrap();
        assert!(v.relevant);
        assert_eq!(v.reasoning, "discusses topic");
    }

    #[test]
    fn test_parse_verdict_wrapped() {
        let v = parse_verdict(
            r#"Here is my judgment:
{"relevant": false, "reasoning": "off topic"}
"#,
        )
        .unwrap();
        assert!(!v.relevant);
    }

    #[test]
    fn test_parse_verdict_markdown() {
        let v = parse_verdict(
            r#"```json
{"relevant": true, "reasoning": "discusses AWS Lambda"}
```"#,
        )
        .unwrap();
        assert!(v.relevant);
    }

    #[test]
    fn test_compute_mrr_first() {
        let judgments = vec![
            ChunkJudgment {
                rank: 1,
                score: 0.9,
                source: None,
                relevant: true,
                reasoning: String::new(),
            },
            ChunkJudgment {
                rank: 2,
                score: 0.8,
                source: None,
                relevant: false,
                reasoning: String::new(),
            },
        ];
        assert!((compute_mrr(&judgments) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_mrr_third() {
        let judgments = vec![
            ChunkJudgment {
                rank: 1,
                score: 0.9,
                source: None,
                relevant: false,
                reasoning: String::new(),
            },
            ChunkJudgment {
                rank: 2,
                score: 0.8,
                source: None,
                relevant: false,
                reasoning: String::new(),
            },
            ChunkJudgment {
                rank: 3,
                score: 0.7,
                source: None,
                relevant: true,
                reasoning: String::new(),
            },
        ];
        assert!((compute_mrr(&judgments) - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_mrr_none() {
        let judgments = vec![ChunkJudgment {
            rank: 1,
            score: 0.9,
            source: None,
            relevant: false,
            reasoning: String::new(),
        }];
        assert!((compute_mrr(&judgments)).abs() < 0.001);
    }

    #[test]
    fn test_check_gate_pass() {
        let output = EvalOutput {
            timestamp: String::new(),
            config: EvalRunConfig {
                num_queries: 10,
                top_k: 10,
                judge_model: String::new(),
                cache_hits: 0,
                api_calls: 10,
            },
            aggregate: AggregateMetrics {
                num_queries: 10,
                mrr: 0.6,
                hit_rate_5: 0.8,
                ..Default::default()
            },
            by_domain: HashMap::new(),
            per_query: Vec::new(),
        };
        assert!(check_gate(&output, 0.5, 0.7).is_ok());
    }

    #[test]
    fn test_check_gate_fail() {
        let output = EvalOutput {
            timestamp: String::new(),
            config: EvalRunConfig {
                num_queries: 10,
                top_k: 10,
                judge_model: String::new(),
                cache_hits: 0,
                api_calls: 10,
            },
            aggregate: AggregateMetrics {
                num_queries: 10,
                mrr: 0.3,
                hit_rate_5: 0.4,
                ..Default::default()
            },
            by_domain: HashMap::new(),
            per_query: Vec::new(),
        };
        assert!(check_gate(&output, 0.5, 0.7).is_err());
    }

    #[test]
    fn test_days_to_ymd() {
        // 2024-01-01 is day 19723
        let (y, m, d) = days_to_ymd(19723);
        assert_eq!((y, m, d), (2024, 1, 1));
    }
}
