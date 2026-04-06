//! Hybrid Retriever - BM25 + Dense with RRF Fusion
//!
//! Implements state-of-the-art hybrid retrieval following:
//! - Karpukhin et al. (2020) for dense retrieval
//! - Robertson & Zaragoza (2009) for BM25
//! - Cormack et al. (2009) for Reciprocal Rank Fusion
//!
//! # Performance
//!
//! - Query latency tracked via `profiling::GLOBAL_METRICS`
//! - Spans: `retrieve`, `bm25_search`, `dense_search`, `rrf_fuse`, `tokenize`
//! - Targets: p50 <20ms, p99 <100ms

use super::profiling::{record_query_latency, span};
use super::types::{Bm25Config, RetrievalResult, RrfConfig, ScoreBreakdown};
use super::DocumentIndex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Hybrid retriever combining sparse (BM25) and dense retrieval
#[derive(Debug)]
pub struct HybridRetriever {
    /// BM25 configuration
    bm25_config: Bm25Config,
    /// RRF configuration
    rrf_config: RrfConfig,
    /// Inverted index for BM25
    inverted_index: InvertedIndex,
    /// Average document length for BM25
    avg_doc_length: f64,
}

impl HybridRetriever {
    /// Create a new hybrid retriever
    pub fn new() -> Self {
        Self {
            bm25_config: Bm25Config::default(),
            rrf_config: RrfConfig::default(),
            inverted_index: InvertedIndex::new(),
            avg_doc_length: 0.0,
        }
    }

    /// Create with custom configuration
    pub fn with_config(bm25_config: Bm25Config, rrf_config: RrfConfig) -> Self {
        Self { bm25_config, rrf_config, inverted_index: InvertedIndex::new(), avg_doc_length: 0.0 }
    }

    /// Index a document for retrieval
    pub fn index_document(&mut self, doc_id: &str, content: &str) {
        self.inverted_index.add_document(doc_id, content);
        self.update_avg_doc_length();
    }

    /// Remove a document from the index
    pub fn remove_document(&mut self, doc_id: &str) {
        self.inverted_index.remove_document(doc_id);
        self.update_avg_doc_length();
    }

    /// Update average document length
    fn update_avg_doc_length(&mut self) {
        let total_length: usize = self.inverted_index.doc_lengths.values().sum();
        let doc_count = self.inverted_index.doc_lengths.len();
        self.avg_doc_length =
            if doc_count > 0 { total_length as f64 / doc_count as f64 } else { 0.0 };
    }

    /// Retrieve documents matching query
    ///
    /// Records latency metrics to `GLOBAL_METRICS` for performance monitoring.
    /// Target: p50 <20ms, p99 <100ms
    pub fn retrieve(
        &self,
        query: &str,
        _index: &DocumentIndex,
        top_k: usize,
    ) -> Vec<RetrievalResult> {
        let start = Instant::now();
        let _retrieve_span = span("retrieve");

        // Get BM25 results
        let bm25_results = {
            let _bm25_span = span("bm25_search");
            self.bm25_search(query, top_k * 2)
        };

        // Get dense results (TF-IDF cosine similarity)
        let dense_results = {
            let _dense_span = span("dense_search");
            self.dense_search(query, top_k * 2)
        };

        // Fuse with RRF
        let mut results = {
            let _fuse_span = span("rrf_fuse");
            self.rrf_fuse(&bm25_results, &dense_results, top_k)
        };

        // Apply component boosting
        {
            let _boost_span = span("component_boost");
            self.apply_component_boost(&mut results, query);
        }

        // Record query latency for performance tracking
        record_query_latency(start.elapsed());

        results
    }

    /// BM25 sparse retrieval
    fn bm25_search(&self, query: &str, top_k: usize) -> Vec<(String, f64)> {
        let query_terms = tokenize(query);
        let mut scores: HashMap<String, f64> = HashMap::new();

        let n = self.inverted_index.doc_lengths.len() as f64;

        for term in &query_terms {
            if let Some(postings) = self.inverted_index.index.get(term) {
                // IDF calculation: log((N - n + 0.5) / (n + 0.5))
                let df = postings.len() as f64;
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

                for (doc_id, tf) in postings {
                    let doc_len =
                        self.inverted_index.doc_lengths.get(doc_id).copied().unwrap_or(1) as f64;

                    // BM25 score
                    let k1 = self.bm25_config.k1 as f64;
                    let b = self.bm25_config.b as f64;
                    let tf_norm = (*tf as f64 * (k1 + 1.0))
                        / (*tf as f64
                            + k1 * (1.0 - b + b * doc_len / self.avg_doc_length.max(1.0)));

                    *scores.entry(doc_id.clone()).or_insert(0.0) += idf * tf_norm;
                }
            }
        }

        let mut results: Vec<_> = scores.into_iter().collect();
        sort_and_truncate(&mut results, top_k);

        results
    }

    /// Dense retrieval using TF-IDF cosine similarity
    ///
    /// Only iterates candidate documents (those containing at least one query term),
    /// not the full index. This makes it efficient even for large indices.
    fn dense_search(&self, query: &str, top_k: usize) -> Vec<(String, f64)> {
        let query_terms = tokenize(query);
        if query_terms.is_empty() {
            return vec![];
        }

        let n = self.inverted_index.doc_lengths.len() as f64;
        if n == 0.0 {
            return vec![];
        }

        // Build query TF-IDF vector + collect candidate docs
        let mut query_vec: HashMap<&str, f64> = HashMap::new();
        let mut candidates: HashSet<String> = HashSet::new();

        for term in &query_terms {
            if let Some(postings) = self.inverted_index.index.get(term.as_str()) {
                let df = postings.len() as f64;
                let idf = (n / df).ln() + 1.0; // smoothed IDF
                *query_vec.entry(term.as_str()).or_insert(0.0) += idf;
                candidates.extend(postings.keys().cloned());
            }
        }

        // Score each candidate doc by cosine similarity
        let query_norm: f64 = query_vec.values().map(|v| v * v).sum::<f64>().sqrt();
        if query_norm == 0.0 {
            return vec![];
        }

        let mut scores: Vec<(String, f64)> = candidates
            .into_iter()
            .filter_map(|doc_id| {
                let doc_len = *self.inverted_index.doc_lengths.get(&doc_id)? as f64;
                let mut dot = 0.0;
                let mut doc_norm_sq = 0.0;

                for term in &query_terms {
                    if let Some(postings) = self.inverted_index.index.get(term.as_str()) {
                        if let Some(&tf) = postings.get(&doc_id) {
                            let df = postings.len() as f64;
                            let idf = (n / df).ln() + 1.0;
                            let tfidf = (tf as f64 / doc_len.max(1.0)) * idf;
                            dot += tfidf * query_vec.get(term.as_str()).unwrap_or(&0.0);
                            doc_norm_sq += tfidf * tfidf;
                        }
                    }
                }

                let doc_norm = doc_norm_sq.sqrt();
                if doc_norm == 0.0 {
                    return None;
                }
                let cosine = dot / (query_norm * doc_norm);
                Some((doc_id, cosine))
            })
            .collect();

        sort_and_truncate(&mut scores, top_k);
        scores
    }

    /// Reciprocal Rank Fusion
    ///
    /// RRF score = Σ 1/(k + rank) for each retriever
    /// Following Cormack et al. (2009)
    fn rrf_fuse(
        &self,
        sparse_results: &[(String, f64)],
        dense_results: &[(String, f64)],
        top_k: usize,
    ) -> Vec<RetrievalResult> {
        let k = self.rrf_config.k as f64;
        let mut rrf_scores: HashMap<String, (f64, f64, f64)> = HashMap::new(); // (rrf, bm25, dense)

        // Accumulate RRF contribution from a single ranked list.
        // `set_field` stores the raw score into the appropriate tuple slot.
        let mut accumulate =
            |results: &[(String, f64)], set_field: fn(&mut (f64, f64, f64), f64)| {
                for (rank, (doc_id, raw_score)) in results.iter().enumerate() {
                    let entry = rrf_scores.entry(doc_id.clone()).or_insert((0.0, 0.0, 0.0));
                    entry.0 += 1.0 / (k + rank as f64 + 1.0);
                    set_field(entry, *raw_score);
                }
            };

        accumulate(sparse_results, |e, s| e.1 = s); // BM25
        accumulate(dense_results, |e, s| e.2 = s); // Dense

        // Convert to results
        let mut results: Vec<_> = rrf_scores
            .into_iter()
            .map(|(doc_id, (rrf_score, bm25_score, dense_score))| {
                // Normalize score to 0-1 range
                let max_rrf = 2.0 / (k + 1.0); // Max possible RRF score (rank 1 in both)
                let normalized_score = (rrf_score / max_rrf).min(1.0);

                let component = extract_component(&doc_id);
                let id = doc_id.clone();
                RetrievalResult {
                    id,
                    component,
                    source: doc_id,
                    content: String::new(), // Would be filled from index
                    score: normalized_score,
                    start_line: 1,
                    end_line: 1,
                    score_breakdown: ScoreBreakdown {
                        bm25_score,
                        dense_score,
                        rrf_score,
                        rerank_score: None,
                    },
                }
            })
            .collect();

        // Sort by score descending
        results.sort();
        results.truncate(top_k);

        results
    }

    /// Get index statistics
    pub fn stats(&self) -> RetrieverStats {
        RetrieverStats {
            total_documents: self.inverted_index.doc_lengths.len(),
            total_terms: self.inverted_index.index.len(),
            avg_doc_length: self.avg_doc_length,
        }
    }

    /// Boost results whose component matches a component name mentioned in the query.
    ///
    /// Extracts component names from `doc_lengths` keys (first path segment),
    /// sorts longest-first to handle hyphenated names (e.g., "trueno-ublk" before "trueno"),
    /// and applies a 1.5x multiplier to matching results, then re-sorts.
    fn apply_component_boost(&self, results: &mut [RetrievalResult], query: &str) {
        let query_lower = query.to_lowercase();

        // Collect unique component names from index, sorted longest first
        let mut components: Vec<String> = self
            .inverted_index
            .doc_lengths
            .keys()
            .filter_map(|k| k.split('/').next())
            .collect::<HashSet<_>>()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        components.sort_by_key(|c| std::cmp::Reverse(c.len()));

        // Find which components are mentioned in query
        let mentioned: Vec<String> =
            components.into_iter().filter(|c| query_lower.contains(&c.to_lowercase())).collect();

        if mentioned.is_empty() {
            return;
        }

        // Apply 1.5x boost to matching results
        for result in results.iter_mut() {
            if mentioned.iter().any(|m| result.component.eq_ignore_ascii_case(m)) {
                result.score = (result.score * 1.5).min(1.0);
            }
        }

        results.sort();
    }

    /// Convert to persisted format for serialization
    pub fn to_persisted(&self) -> super::persistence::PersistedIndex {
        super::persistence::PersistedIndex {
            inverted_index: self.inverted_index.index.clone(),
            doc_lengths: self.inverted_index.doc_lengths.clone(),
            bm25_config: self.bm25_config,
            rrf_config: self.rrf_config,
            avg_doc_length: self.avg_doc_length,
        }
    }

    /// Restore from persisted format
    pub fn from_persisted(persisted: super::persistence::PersistedIndex) -> Self {
        Self {
            bm25_config: persisted.bm25_config,
            rrf_config: persisted.rrf_config,
            inverted_index: InvertedIndex {
                index: persisted.inverted_index,
                doc_lengths: persisted.doc_lengths,
            },
            avg_doc_length: persisted.avg_doc_length,
        }
    }
}

impl Default for HybridRetriever {
    fn default() -> Self {
        Self::new()
    }
}

/// Retriever statistics
#[derive(Debug, Clone)]
pub struct RetrieverStats {
    /// Total documents indexed
    pub total_documents: usize,
    /// Total unique terms
    pub total_terms: usize,
    /// Average document length
    pub avg_doc_length: f64,
}

/// Inverted index for BM25
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct InvertedIndex {
    /// Term -> (doc_id -> term_frequency)
    pub index: HashMap<String, HashMap<String, usize>>,
    /// Document lengths
    pub doc_lengths: HashMap<String, usize>,
}

impl InvertedIndex {
    fn new() -> Self {
        Self::default()
    }

    fn add_document(&mut self, doc_id: &str, content: &str) {
        let tokens = tokenize(content);
        self.doc_lengths.insert(doc_id.to_string(), tokens.len());

        // Count term frequencies
        let mut term_freqs: HashMap<String, usize> = HashMap::new();
        for token in tokens {
            *term_freqs.entry(token).or_insert(0) += 1;
        }

        // Add to inverted index
        for (term, freq) in term_freqs {
            self.index.entry(term).or_default().insert(doc_id.to_string(), freq);
        }
    }

    fn remove_document(&mut self, doc_id: &str) {
        self.doc_lengths.remove(doc_id);

        // Remove from all posting lists
        for postings in self.index.values_mut() {
            postings.remove(doc_id);
        }

        // Clean up empty posting lists
        self.index.retain(|_, postings| !postings.is_empty());
    }
}

/// Stem a word using aprender's PorterStemmer when the `ml` feature is enabled,
/// falling back to simple suffix stripping otherwise.
#[cfg(feature = "ml")]
fn stem(word: &str) -> String {
    use aprender::text::stem::{PorterStemmer, Stemmer};
    PorterStemmer::new().stem(word).unwrap_or_else(|_| word.to_string())
}

/// Fallback suffix stripping when aprender is not available.
///
/// Strips the longest matching suffix while keeping the stem >= 3 characters.
#[cfg(not(feature = "ml"))]
fn stem(word: &str) -> String {
    if word.len() <= 3 {
        return word.to_string();
    }
    for suffix in &[
        "ization", "isation", "ation", "tion", "sion", "ment", "ness", "ible", "able", "ence",
        "ance", "zing", "ying", "ming", "ning", "ting", "ring", "ling", "sing", "ious", "eous",
        "mming", "ful", "ive", "ize", "ise", "ity", "ist", "ism", "ied", "ies", "ing", "ous",
        "ers", "est", "ely", "ory", "ant", "ent", "ial", "ual", "ly", "ed", "er", "al", "ic",
    ] {
        if let Some(s) = word.strip_suffix(suffix) {
            if s.len() >= 3 {
                return s.to_string();
            }
        }
    }
    word.to_string()
}

/// Check if a word is a stop word using aprender's StopWordsFilter when available.
#[cfg(feature = "ml")]
fn is_stop_word(word: &str) -> bool {
    use aprender::text::stopwords::StopWordsFilter;
    use std::sync::LazyLock;
    static FILTER: LazyLock<StopWordsFilter> = LazyLock::new(StopWordsFilter::english);
    FILTER.is_stop_word(word)
}

/// Fallback stop word check when aprender is not available.
#[cfg(not(feature = "ml"))]
fn is_stop_word(word: &str) -> bool {
    const STOP_WORDS: &[&str] = &[
        "the", "is", "at", "which", "on", "in", "to", "for", "of", "and", "or", "an", "be", "by",
        "as", "do", "if", "it", "no", "so", "up", "how", "can", "its", "has", "had", "was", "are",
        "were", "been", "have", "from", "this", "that", "with", "what", "when", "where", "will",
        "not", "but", "all", "each", "than",
    ];
    STOP_WORDS.contains(&word)
}

/// Tokenizer with stop-word filtering and stemming.
///
/// Splits on non-alphanumeric characters (preserving underscores),
/// removes single-character tokens and stop words, then applies stemming.
/// When the `ml` feature is enabled, uses aprender's PorterStemmer and
/// 171-word English stop words list. Otherwise falls back to simple suffix stripping.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty() && s.len() > 1)
        .filter(|s| !is_stop_word(s))
        .map(stem)
        .collect()
}

/// Sort `(id, score)` pairs by score descending and keep only the top `k`.
fn sort_and_truncate(results: &mut Vec<(String, f64)>, k: usize) {
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
}

/// Extract component name from doc_id
fn extract_component(doc_id: &str) -> String {
    doc_id.split('/').next().unwrap_or("unknown").to_string()
}

#[cfg(test)]
#[path = "retriever_tests.rs"]
mod tests;
