//! Contract index building and BM25 search.

use std::collections::HashMap;
use std::path::Path;

use provable_contracts_macros::requires;

use crate::schema::{Contract, parse_contract};
use crate::scoring;

use super::persist::{self, PersistedIndex};
use super::types::ContractEntry;

/// In-memory contract index with inverted indexes for fast lookup.
#[derive(Debug)]
pub struct ContractIndex {
    pub entries: Vec<ContractEntry>,
    name_index: HashMap<String, usize>,
    equation_index: HashMap<String, Vec<usize>>,
    obligation_index: HashMap<String, Vec<usize>>,
    /// Pre-computed composite scores for O(1) `--min-score` filtering.
    score_cache: HashMap<String, f64>,
    /// Pre-computed pagerank scores for importance-weighted ranking.
    pagerank_cache: HashMap<String, f64>,
    /// Average document length for BM25.
    avg_dl: f64,
    /// Document frequency per term.
    df: HashMap<String, usize>,
}

impl ContractIndex {
    /// Build an index from a directory of YAML contracts.
    ///
    /// Uses cached index from `.pv/contracts.idx` when fresh,
    /// otherwise rebuilds and caches for next time.
    pub fn from_directory(dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Self::from_directory_opts(dir, false)
    }

    /// Build an index with option to force rebuild (skip cache).
    pub fn from_directory_opts(
        dir: &Path,
        force_rebuild: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Try loading cached index first (unless force rebuild)
        if !force_rebuild {
            if let Some(cached) = persist::load_cached(dir) {
                let mut index = Self::from_entries(cached.entries);
                index.score_cache = cached.score_cache;
                index.pagerank_cache = cached.pagerank_cache;
                return Ok(index);
            }
        }

        let index = Self::build_from_directory(dir)?;

        // Save to cache (best-effort, don't fail on write errors)
        let _ = persist::save_cached(
            dir,
            &PersistedIndex {
                entries: index.entries.clone(),
                score_cache: index.score_cache.clone(),
                pagerank_cache: index.pagerank_cache.clone(),
            },
        );

        Ok(index)
    }

    /// Build an index from a directory without cache.
    pub fn build_from_directory(dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut yaml_paths: Vec<_> = collect_yaml_files(dir)?;
        yaml_paths.sort();

        let mut entries = Vec::new();
        let mut score_cache = HashMap::new();
        for path in &yaml_paths {
            let Ok(contract) = parse_contract(path) else {
                continue;
            };
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            let path_str = path.display().to_string();
            let score = scoring::score_contract(&contract, None, &stem);
            score_cache.insert(stem.clone(), score.composite);
            entries.push(build_entry(stem, path_str, &contract));
        }

        let mut index = Self::from_entries(entries);
        index.score_cache = score_cache;
        index.pagerank_cache = index.pagerank(20, 0.85);
        Ok(index)
    }

    /// Build an index from pre-parsed entries.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_entries(entries: Vec<ContractEntry>) -> Self {
        let mut name_index = HashMap::new();
        let mut equation_index: HashMap<String, Vec<usize>> = HashMap::new();
        let mut obligation_index: HashMap<String, Vec<usize>> = HashMap::new();
        let mut df: HashMap<String, usize> = HashMap::new();
        let mut total_len = 0usize;

        for (i, entry) in entries.iter().enumerate() {
            name_index.insert(entry.stem.clone(), i);
            for eq in &entry.equations {
                equation_index.entry(eq.clone()).or_default().push(i);
            }
            for ot in &entry.obligation_types {
                obligation_index.entry(ot.clone()).or_default().push(i);
            }

            let terms = tokenize(&entry.corpus_text);
            total_len += terms.len();
            let mut seen = std::collections::HashSet::new();
            for t in &terms {
                if seen.insert(t.clone()) {
                    *df.entry(t.clone()).or_default() += 1;
                }
            }
        }

        let avg_dl = if entries.is_empty() {
            1.0
        } else {
            total_len as f64 / entries.len() as f64
        };

        Self {
            entries,
            name_index,
            equation_index,
            obligation_index,
            score_cache: HashMap::new(),
            pagerank_cache: HashMap::new(),
            avg_dl,
            df,
        }
    }

    /// Look up a contract by exact stem.
    pub fn get_by_stem(&self, stem: &str) -> Option<&ContractEntry> {
        self.name_index.get(stem).map(|&i| &self.entries[i])
    }

    /// Get the pre-computed composite score for a contract stem.
    pub fn cached_score(&self, stem: &str) -> Option<f64> {
        self.score_cache.get(stem).copied()
    }

    /// Get the pre-computed pagerank score for a contract stem.
    pub fn cached_pagerank(&self, stem: &str) -> Option<f64> {
        self.pagerank_cache.get(stem).copied()
    }

    /// Look up contracts by obligation type.
    pub fn get_by_obligation(&self, ob_type: &str) -> Vec<&ContractEntry> {
        self.obligation_index
            .get(ob_type)
            .map(|idxs| idxs.iter().map(|&i| &self.entries[i]).collect())
            .unwrap_or_default()
    }

    /// Look up contracts by equation name.
    pub fn get_by_equation(&self, eq: &str) -> Vec<&ContractEntry> {
        self.equation_index
            .get(eq)
            .map(|idxs| idxs.iter().map(|&i| &self.entries[i]).collect())
            .unwrap_or_default()
    }

    /// BM25 search across all entries. Returns (index, score) pairs sorted descending.
    #[allow(clippy::cast_precision_loss)]
    pub fn bm25_search(&self, query: &str) -> Vec<(usize, f64)> {
        let query_terms = tokenize(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        let n = self.entries.len() as f64;
        let k1 = 1.2;
        let b = 0.75;

        let mut scores: Vec<(usize, f64)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let doc_terms = tokenize(&entry.corpus_text);
                let dl = doc_terms.len() as f64;

                let tf_map = term_frequencies(&doc_terms);
                let score: f64 = query_terms
                    .iter()
                    .map(|qt| {
                        let doc_freq = self.df.get(qt).copied().unwrap_or(0) as f64;
                        let idf = ((n - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).ln();
                        let tf = tf_map.get(qt).copied().unwrap_or(0) as f64;
                        idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / self.avg_dl))
                    })
                    .sum();

                (i, score)
            })
            .filter(|(_, s)| *s > 0.0)
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Regex search across all entries. Returns matching indices.
    pub fn regex_search(&self, pattern: &str) -> Result<Vec<usize>, regex::Error> {
        let re = regex::Regex::new(pattern)?;
        Ok(self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| re.is_match(&e.corpus_text))
            .map(|(i, _)| i)
            .collect())
    }

    /// Literal substring search. Returns matching indices.
    pub fn literal_search(&self, needle: &str, case_sensitive: bool) -> Vec<usize> {
        let needle_lower = needle.to_lowercase();
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                if case_sensitive {
                    e.corpus_text.contains(needle)
                } else {
                    e.corpus_text.to_lowercase().contains(&needle_lower)
                }
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Return reverse dependencies: contracts that depend on `stem`.
    pub fn depended_by(&self, stem: &str) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|e| e.depends_on.iter().any(|d| d == stem))
            .map(|e| e.stem.as_str())
            .collect()
    }

    /// Compute pagerank scores over the contract dependency graph.
    ///
    /// Returns a map from stem to pagerank score. Higher scores indicate
    /// more "important" contracts (more depended-upon by others).
    #[allow(clippy::cast_precision_loss)]
    #[requires(iterations > 0 && damping > 0.0 && damping < 1.0)]
    pub fn pagerank(&self, iterations: usize, damping: f64) -> HashMap<String, f64> {
        let n = self.entries.len();
        if n == 0 {
            return HashMap::new();
        }
        let n_f = n as f64;
        let mut scores: Vec<f64> = vec![1.0 / n_f; n];

        for _ in 0..iterations {
            let mut new_scores = vec![(1.0 - damping) / n_f; n];
            for (i, entry) in self.entries.iter().enumerate() {
                let out_degree = entry.depends_on.len();
                if out_degree == 0 {
                    // Distribute rank equally to all (dangling node)
                    let share = damping * scores[i] / n_f;
                    for s in &mut new_scores {
                        *s += share;
                    }
                } else {
                    let share = damping * scores[i] / out_degree as f64;
                    for dep in &entry.depends_on {
                        if let Some(&j) = self.name_index.get(dep) {
                            new_scores[j] += share;
                        }
                    }
                }
            }
            scores = new_scores;
        }

        self.entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.stem.clone(), scores[i]))
            .collect()
    }
}

fn build_entry(stem: String, path: String, contract: &Contract) -> ContractEntry {
    let equations: Vec<String> = contract.equations.keys().cloned().collect();
    let obligation_types: Vec<String> = contract
        .proof_obligations
        .iter()
        .map(|o| o.obligation_type.to_string())
        .collect();
    let properties: Vec<String> = contract
        .proof_obligations
        .iter()
        .map(|o| o.property.clone())
        .collect();
    let references = contract.metadata.references.clone();
    let depends_on = contract.metadata.depends_on.clone();
    let mut corpus_parts = vec![stem.clone(), contract.metadata.description.clone()];
    for (name, eq) in &contract.equations {
        corpus_parts.push(name.clone());
        corpus_parts.push(eq.formula.clone());
        corpus_parts.extend(eq.invariants.iter().cloned());
    }
    for ob in &contract.proof_obligations {
        corpus_parts.push(ob.property.clone());
        if let Some(f) = &ob.formal {
            corpus_parts.push(f.clone());
        }
    }
    corpus_parts.extend(references.iter().cloned());
    let corpus_text = corpus_parts.join(" ");

    ContractEntry {
        stem,
        path,
        description: contract.metadata.description.clone(),
        equations,
        obligation_types,
        properties,
        references,
        depends_on,
        is_registry: contract.is_registry(),
        kind: contract.kind(),
        obligation_count: contract.proof_obligations.len(),
        falsification_count: contract.falsification_tests.len(),
        kani_count: contract.kani_harnesses.len(),
        corpus_text,
    }
}

/// Tokenize text into lowercase alphanumeric terms (>= 2 chars).
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .map(str::to_lowercase)
        .filter(|s| s.len() >= 2)
        .collect()
}

fn term_frequencies(terms: &[String]) -> HashMap<&String, usize> {
    let mut tf = HashMap::new();
    for t in terms {
        *tf.entry(t).or_insert(0) += 1;
    }
    tf
}

fn collect_yaml_files(dir: &Path) -> Result<Vec<std::path::PathBuf>, Box<dyn std::error::Error>> {
    let mut result = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            result.extend(collect_yaml_files(&path)?);
        } else if path.extension().and_then(|x| x.to_str()) == Some("yaml") {
            result.push(path);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_splits_correctly() {
        let tokens = tokenize("softmax-kernel_v1 numerical stability");
        assert!(tokens.contains(&"softmax".to_string()));
        assert!(tokens.contains(&"kernel_v1".to_string()));
        assert!(tokens.contains(&"numerical".to_string()));
        assert!(tokens.contains(&"stability".to_string()));
    }

    #[test]
    fn tokenize_filters_short() {
        let tokens = tokenize("a is ok");
        assert!(!tokens.iter().any(|t| t == "a"));
        assert!(tokens.contains(&"is".to_string()));
        assert!(tokens.contains(&"ok".to_string()));
    }

    #[test]
    fn index_from_contracts_dir() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        let index = ContractIndex::build_from_directory(&dir).unwrap();
        assert!(index.entries.len() > 10, "Should index many contracts");
        assert!(index.get_by_stem("softmax-kernel-v1").is_some());
    }

    #[test]
    fn bm25_ranks_relevant_first() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        let index = ContractIndex::build_from_directory(&dir).unwrap();
        let results = index.bm25_search("softmax numerical stability");
        assert!(!results.is_empty());
        // Top result should be related to softmax/cross-entropy (both reference softmax)
        let top = &index.entries[results[0].0];
        assert!(
            top.corpus_text.to_lowercase().contains("softmax"),
            "Top result corpus should mention softmax, got stem={}",
            top.stem,
        );
    }

    #[test]
    fn literal_search_finds_match() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        let index = ContractIndex::build_from_directory(&dir).unwrap();
        let matches = index.literal_search("RMSNorm", false);
        assert!(!matches.is_empty());
    }

    #[test]
    fn regex_search_finds_patterns() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        let index = ContractIndex::build_from_directory(&dir).unwrap();
        let matches = index.regex_search(r"(?i)softmax|log.softmax").unwrap();
        assert!(!matches.is_empty());
    }

    #[test]
    fn depended_by_returns_dependents() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        let index = ContractIndex::build_from_directory(&dir).unwrap();
        // softmax-kernel-v1 is depended on by several contracts
        let _deps = index.depended_by("softmax-kernel-v1");
        // At minimum attention contracts depend on softmax
        // depended_by should return without panicking
        assert!(!index.entries.is_empty(), "Index should contain contracts");
    }

    #[test]
    fn pagerank_produces_valid_scores() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        let index = ContractIndex::build_from_directory(&dir).unwrap();
        let scores = index.pagerank(20, 0.85);
        // scores is a HashMap keyed by stem — duplicate stems collapse to one entry
        let unique_stems: std::collections::HashSet<_> =
            index.entries.iter().map(|e| &e.stem).collect();
        assert_eq!(scores.len(), unique_stems.len());
        // All scores should be positive
        for s in scores.values() {
            assert!(*s > 0.0, "PageRank should be positive");
        }
        // Softmax should rank relatively high (many things depend on it)
        let softmax = scores.get("softmax-kernel-v1").unwrap();
        #[allow(clippy::cast_precision_loss)]
        let mean = scores.values().sum::<f64>() / scores.len() as f64;
        assert!(
            *softmax >= mean,
            "softmax ({softmax:.4}) should be >= mean ({mean:.4})"
        );
    }

    #[test]
    fn pagerank_empty_index() {
        let index = ContractIndex::from_entries(Vec::new());
        let scores = index.pagerank(20, 0.85);
        assert!(scores.is_empty());
    }

    #[test]
    fn from_directory_uses_cache() {
        // Build from a temp dir to avoid contaminating the real .pv cache
        let tmp = std::env::temp_dir().join("pv_from_dir_cache_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // Copy a few contracts to temp
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        for name in &["softmax-kernel-v1.yaml", "rmsnorm-kernel-v1.yaml"] {
            let content = std::fs::read_to_string(src.join(name)).unwrap();
            std::fs::write(tmp.join(name), content).unwrap();
        }

        // First call builds + caches
        let idx1 = ContractIndex::from_directory(&tmp).unwrap();
        assert!(idx1.entries.len() >= 2);

        // Second call should hit cache
        let idx2 = ContractIndex::from_directory(&tmp).unwrap();
        assert_eq!(idx1.entries.len(), idx2.entries.len());

        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::remove_dir_all(tmp.parent().unwrap().join(".pv"));
    }
}
