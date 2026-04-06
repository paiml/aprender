//! Core types for the evaluation framework

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for eval operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalConfig {
    /// Claude model for generation/judging
    pub model: String,
    /// Number of query-chunk pairs to generate
    pub sample_size: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Top-k results to retrieve
    pub top_k: usize,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            model: "claude-sonnet-4-20250514".to_string(),
            sample_size: 250,
            seed: 42,
            top_k: 10,
        }
    }
}

/// A single ground truth entry (query paired with its source chunk)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthEntry {
    /// The evaluation query
    pub query: String,
    /// Full text of the source chunk this query was generated from
    pub chunk_content: String,
    /// File path of the source chunk
    pub chunk_source: String,
    /// Start time in seconds (for media chunks)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_start_secs: Option<f64>,
    /// End time in seconds (for media chunks)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_end_secs: Option<f64>,
    /// Domain classification
    pub domain: String,
    /// Course directory name
    pub course: String,
}

/// Raw retrieval results for a single query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResultEntry {
    /// The query that was run
    pub query: String,
    /// Domain classification
    pub domain: String,
    /// Course directory name
    pub course: String,
    /// Retrieved chunks with scores
    pub results: Vec<RetrievedChunk>,
    /// Query latency in seconds
    pub latency_s: f64,
}

/// A single retrieved chunk from a query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedChunk {
    /// Chunk text content
    pub content: String,
    /// Source file path
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Retrieval score
    pub score: f32,
    /// Title
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Start time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_secs: Option<f64>,
    /// End time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_secs: Option<f64>,
}

/// LLM judge verdict for a (query, chunk) pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeVerdict {
    /// Is the chunk relevant to the query?
    pub relevant: bool,
    /// Brief reasoning from the judge
    pub reasoning: String,
}

/// Single cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeCacheEntry {
    /// The verdict
    pub verdict: JudgeVerdict,
    /// Model used for judging
    pub model: String,
}

/// Persistent cache for LLM judge verdicts
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JudgeCache {
    /// Map from cache key (sha256 hex prefix) to verdict
    pub entries: HashMap<String, JudgeCacheEntry>,
}

impl JudgeCache {
    /// Load cache from a JSON file, or return empty cache
    pub fn load(path: &std::path::Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Save cache to a JSON file
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    /// Compute cache key from query + content
    pub fn cache_key(query: &str, content: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(query.as_bytes());
        hasher.update(b"|||");
        hasher.update(content.as_bytes());
        let result = hasher.finalize();
        hex::encode(&result[..8]) // 16 hex chars
    }

    /// Look up a cached verdict
    pub fn get(&self, query: &str, content: &str) -> Option<&JudgeVerdict> {
        let key = Self::cache_key(query, content);
        self.entries.get(&key).map(|e| &e.verdict)
    }

    /// Insert a verdict into the cache
    pub fn insert(&mut self, query: &str, content: &str, verdict: JudgeVerdict, model: &str) {
        let key = Self::cache_key(query, content);
        self.entries.insert(key, JudgeCacheEntry { verdict, model: model.to_string() });
    }
}

/// Inline hex encoding (avoid adding hex crate dep)
mod hex {
    pub(crate) fn encode(bytes: &[u8]) -> String {
        use std::fmt::Write;
        bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
    }
}

/// A single judgment entry (written by Claude Code or external judge)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgmentEntry {
    /// The query
    pub query: String,
    /// Rank of the chunk being judged (1-indexed)
    pub rank: usize,
    /// Whether the chunk is relevant
    pub relevant: bool,
    /// Brief reasoning
    pub reasoning: String,
    /// Source path (for correlation)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Retrieval score (for correlation)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

/// Eval output with full results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalOutput {
    /// Timestamp of the eval run
    pub timestamp: String,
    /// Config used
    pub config: EvalRunConfig,
    /// Aggregate metrics
    pub aggregate: AggregateMetrics,
    /// Per-domain metrics
    pub by_domain: HashMap<String, AggregateMetrics>,
    /// Per-query details
    pub per_query: Vec<QueryResult>,
}

/// Config recorded in eval output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalRunConfig {
    /// Number of queries evaluated
    pub num_queries: usize,
    /// Top-k used for retrieval
    pub top_k: usize,
    /// Model used for judging
    pub judge_model: String,
    /// Cache hits (saved API calls)
    pub cache_hits: usize,
    /// New API calls made
    pub api_calls: usize,
}

/// Aggregate metrics across queries
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregateMetrics {
    /// Number of queries
    pub num_queries: usize,
    /// Mean Reciprocal Rank
    pub mrr: f64,
    /// NDCG at k=5
    #[serde(rename = "ndcg@5")]
    pub ndcg_5: f64,
    /// NDCG at k=10
    #[serde(rename = "ndcg@10")]
    pub ndcg_10: f64,
    /// Recall at k=5
    #[serde(rename = "recall@5")]
    pub recall_5: f64,
    /// Precision at k=5
    #[serde(rename = "precision@5")]
    pub precision_5: f64,
    /// Hit rate at k=5
    #[serde(rename = "hit_rate@5")]
    pub hit_rate_5: f64,
    /// Hit rate at k=10
    #[serde(rename = "hit_rate@10")]
    pub hit_rate_10: f64,
    /// Mean Average Precision
    pub map: f64,
    /// Mean query latency
    pub mean_latency_s: f64,
}

/// Per-query result with judge details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// The query
    pub query: String,
    /// Domain
    pub domain: String,
    /// MRR for this query
    pub mrr: f64,
    /// Hit at k=5
    pub hit_5: bool,
    /// Number of relevant results in top-10
    pub relevant_count: usize,
    /// Total results
    pub total_results: usize,
    /// Latency
    pub latency_s: f64,
    /// Per-result judgments
    pub judgments: Vec<ChunkJudgment>,
}

/// Judgment for a single retrieved chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkJudgment {
    /// Rank position (1-indexed)
    pub rank: usize,
    /// Retrieval score
    pub score: f32,
    /// Source path
    pub source: Option<String>,
    /// Whether the judge deemed it relevant
    pub relevant: bool,
    /// Judge reasoning
    pub reasoning: String,
}
