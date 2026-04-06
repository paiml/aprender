//! Curated arXiv database for citation matching
//!
//! Contains ~120 curated entries across ML, NLP, systems, DevOps, and cloud topics.
//! URLs are deterministic `https://arxiv.org/abs/{id}` — stable, no 404 risk.

use super::types::ArxivCitation;

/// Curated arXiv database compiled into the binary.
pub struct ArxivDatabase {
    entries: Vec<ArxivCitation>,
}

impl ArxivDatabase {
    /// Load the built-in curated database.
    pub fn builtin() -> Self {
        Self { entries: builtin_entries() }
    }

    /// Find citations by topic keyword (single topic, case-insensitive).
    pub fn find_by_topic(&self, topic: &str, limit: usize) -> Vec<ArxivCitation> {
        let topic_lower = topic.to_lowercase();
        let mut results: Vec<_> = self
            .entries
            .iter()
            .filter(|e| {
                e.topics.iter().any(|t| t.to_lowercase().contains(&topic_lower))
                    || e.title.to_lowercase().contains(&topic_lower)
            })
            .cloned()
            .collect();
        results.truncate(limit);
        results
    }

    /// Find citations by multiple keywords using Jaccard scoring.
    pub fn find_by_keywords(&self, keywords: &[&str], limit: usize) -> Vec<ArxivCitation> {
        let kw_lower: Vec<String> = keywords.iter().map(|k| k.to_lowercase()).collect();
        let kw_count = kw_lower.len() as f64;
        if kw_count == 0.0 {
            return Vec::new();
        }

        let mut scored: Vec<(f64, &ArxivCitation)> = self
            .entries
            .iter()
            .map(|entry| {
                let entry_topics: Vec<String> =
                    entry.topics.iter().map(|t| t.to_lowercase()).collect();
                let title_lower = entry.title.to_lowercase();

                let matches = kw_lower
                    .iter()
                    .filter(|kw| {
                        entry_topics.iter().any(|t| t.contains(kw.as_str()))
                            || title_lower.contains(kw.as_str())
                    })
                    .count() as f64;

                let union = kw_count + entry.topics.len() as f64 - matches;
                let score = if union > 0.0 { matches / union } else { 0.0 };
                (score, entry)
            })
            .filter(|(score, _)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(limit).map(|(_, e)| e.clone()).collect()
    }

    /// Get total number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if database is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

fn cite(
    id: &str,
    title: &str,
    authors: &str,
    year: u16,
    snippet: &str,
    topics: &[&str],
) -> ArxivCitation {
    ArxivCitation {
        arxiv_id: id.to_string(),
        title: title.to_string(),
        authors: authors.to_string(),
        year,
        url: format!("https://arxiv.org/abs/{id}"),
        abstract_snippet: snippet.to_string(),
        topics: topics.iter().map(|s| (*s).to_string()).collect(),
    }
}

fn builtin_entries() -> Vec<ArxivCitation> {
    include!("arxiv_entries.rs")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_database_size() {
        let db = ArxivDatabase::builtin();
        assert!(db.len() >= 100, "Expected at least 100 entries, got {}", db.len());
        assert!(!db.is_empty());
    }

    #[test]
    fn test_find_by_topic() {
        let db = ArxivDatabase::builtin();
        let results = db.find_by_topic("transformer", 5);
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
        // Should find "Attention Is All You Need"
        assert!(results.iter().any(|r| r.arxiv_id == "1706.03762"));
    }

    #[test]
    fn test_find_by_topic_case_insensitive() {
        let db = ArxivDatabase::builtin();
        let lower = db.find_by_topic("rag", 10);
        let upper = db.find_by_topic("RAG", 10);
        assert_eq!(lower.len(), upper.len());
    }

    #[test]
    fn test_find_by_keywords_jaccard() {
        let db = ArxivDatabase::builtin();
        let results = db.find_by_keywords(&["mlops", "pipeline", "ci/cd"], 5);
        assert!(!results.is_empty());
        // Results should be scored and sorted
    }

    #[test]
    fn test_find_by_keywords_empty() {
        let db = ArxivDatabase::builtin();
        let results = db.find_by_keywords(&[], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_by_topic_no_results() {
        let db = ArxivDatabase::builtin();
        let results = db.find_by_topic("xyznonexistent", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_arxiv_url_format() {
        let db = ArxivDatabase::builtin();
        for entry in &db.entries {
            assert!(entry.url.starts_with("https://arxiv.org/abs/"), "Bad URL: {}", entry.url);
            assert!(entry.url.ends_with(&entry.arxiv_id));
        }
    }

    #[test]
    fn test_all_entries_have_topics() {
        let db = ArxivDatabase::builtin();
        for entry in &db.entries {
            assert!(!entry.topics.is_empty(), "Entry {} has no topics", entry.arxiv_id);
        }
    }

    #[test]
    fn test_find_by_topic_limit() {
        let db = ArxivDatabase::builtin();
        let results = db.find_by_topic("deep learning", 2);
        assert!(results.len() <= 2);
    }
}
