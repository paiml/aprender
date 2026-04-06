//! Code Duplication Detection Rule
//!
//! Detects significant code duplication across PAIML stack projects using MinHash+LSH.

use crate::comply::rule::{
    FixResult, RuleCategory, RuleResult, RuleViolation, StackComplianceRule, Suggestion,
    ViolationLevel,
};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Code duplication detection rule using MinHash+LSH
#[derive(Debug)]
pub struct DuplicationRule {
    /// Similarity threshold (0.0 to 1.0)
    similarity_threshold: f64,
    /// Minimum fragment size in lines
    min_fragment_size: usize,
    /// Number of MinHash permutations
    num_permutations: usize,
    /// File patterns to include
    include_patterns: Vec<String>,
    /// File patterns to exclude
    exclude_patterns: Vec<String>,
}

impl Default for DuplicationRule {
    fn default() -> Self {
        Self::new()
    }
}

impl DuplicationRule {
    /// Create a new duplication rule with default configuration
    pub fn new() -> Self {
        Self {
            similarity_threshold: 0.85,
            min_fragment_size: 50,
            num_permutations: 128,
            include_patterns: vec!["**/*.rs".to_string()],
            exclude_patterns: vec![
                "**/target/**".to_string(),
                "**/tests/**".to_string(),
                "**/benches/**".to_string(),
            ],
        }
    }

    /// Extract code fragments from a file for duplication analysis.
    ///
    /// Parses the file and extracts logical code blocks (functions, impls, structs)
    /// that meet the minimum fragment size requirement. Uses a sliding window
    /// approach with block depth tracking to identify boundaries.
    ///
    /// # Arguments
    /// * `path` - Path to the source file to analyze
    ///
    /// # Returns
    /// * `Ok(Vec<CodeFragment>)` - Extracted fragments meeting size threshold
    /// * `Err` - If file cannot be read
    fn extract_fragments(&self, path: &Path) -> anyhow::Result<Vec<CodeFragment>> {
        let content = std::fs::read_to_string(path)?;
        let lines: Vec<&str> = content.lines().collect();

        if lines.len() < self.min_fragment_size {
            return Ok(Vec::new());
        }

        let mut fragments = Vec::new();

        // Use sliding window to extract fragments
        // For efficiency, use function/impl boundaries when possible
        let mut current_start = 0;
        let mut in_block = false;
        let mut block_depth = 0;

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Track block depth
            block_depth += trimmed.matches('{').count();
            block_depth = block_depth.saturating_sub(trimmed.matches('}').count());

            // Detect function/impl boundaries
            if (trimmed.starts_with("fn ")
                || trimmed.starts_with("pub fn ")
                || trimmed.starts_with("impl ")
                || trimmed.starts_with("pub struct ")
                || trimmed.starts_with("struct "))
                && !in_block
            {
                // Start new fragment
                if i > current_start && i - current_start >= self.min_fragment_size {
                    fragments.push(CodeFragment {
                        path: path.to_path_buf(),
                        start_line: current_start + 1,
                        end_line: i,
                        content: lines[current_start..i].join("\n"),
                    });
                }
                current_start = i;
                in_block = true;
            }

            // End of block
            if in_block && block_depth == 0 && trimmed.ends_with('}') {
                if i - current_start >= self.min_fragment_size {
                    fragments.push(CodeFragment {
                        path: path.to_path_buf(),
                        start_line: current_start + 1,
                        end_line: i + 1,
                        content: lines[current_start..=i].join("\n"),
                    });
                }
                current_start = i + 1;
                in_block = false;
            }
        }

        // Capture remaining content
        if lines.len() - current_start >= self.min_fragment_size {
            fragments.push(CodeFragment {
                path: path.to_path_buf(),
                start_line: current_start + 1,
                end_line: lines.len(),
                content: lines[current_start..].join("\n"),
            });
        }

        Ok(fragments)
    }

    /// Compute MinHash signature for a code fragment.
    ///
    /// Uses locality-sensitive hashing to create a compact signature
    /// that can be compared for similarity. Fragments with similar
    /// content will have similar signatures.
    ///
    /// # Algorithm
    /// 1. Tokenize content into n-grams
    /// 2. Hash each token with multiple permutation functions
    /// 3. Keep minimum hash for each permutation (MinHash property)
    ///
    /// # Arguments
    /// * `fragment` - Code fragment to compute signature for
    ///
    /// # Returns
    /// MinHash signature with `num_permutations` values
    fn compute_minhash(&self, fragment: &CodeFragment) -> MinHashSignature {
        // Tokenize: extract words and n-grams
        let tokens = self.tokenize(&fragment.content);

        // Compute MinHash using multiple hash functions
        let mut signature = vec![u64::MAX; self.num_permutations];

        for token in tokens {
            for (i, sig) in signature.iter_mut().enumerate() {
                // Simple hash combining token and permutation index
                let hash = self.hash_token(&token, i as u64);
                if hash < *sig {
                    *sig = hash;
                }
            }
        }

        MinHashSignature { values: signature }
    }

    /// Tokenize code into n-grams for MinHash computation.
    ///
    /// Normalizes code by removing comments and extra whitespace,
    /// then extracts both individual tokens and 3-grams (sequences
    /// of 3 words) to capture structural similarity.
    ///
    /// # Arguments
    /// * `content` - Raw code content to tokenize
    ///
    /// # Returns
    /// Vector of token strings (words and 3-grams)
    fn tokenize(&self, content: &str) -> Vec<String> {
        let mut tokens = Vec::new();

        // Normalize: lowercase, remove extra whitespace
        let normalized: String = content
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with("//"))
            .collect::<Vec<_>>()
            .join(" ");

        // Extract words
        let words: Vec<&str> = normalized
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| !w.is_empty())
            .collect();

        // Generate 3-grams
        for window in words.windows(3) {
            tokens.push(window.join(" "));
        }

        // Also add individual significant tokens
        for word in &words {
            if word.len() > 3 {
                tokens.push((*word).to_string());
            }
        }

        tokens
    }

    /// Hash a token with a permutation index.
    ///
    /// Uses FNV-1a hash combined with the permutation index to create
    /// different hash functions for MinHash. This simulates independent
    /// hash functions required by the MinHash algorithm.
    ///
    /// # Arguments
    /// * `token` - Token string to hash
    /// * `perm` - Permutation index (0 to num_permutations-1)
    ///
    /// # Returns
    /// 64-bit hash value
    fn hash_token(&self, token: &str, perm: u64) -> u64 {
        // FNV-1a hash with permutation mixing
        let mut hash: u64 = 0xcbf29ce484222325;
        hash = hash.wrapping_mul(0x100000001b3);
        hash ^= perm;

        for byte in token.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }

        hash
    }

    /// Compute Jaccard similarity from MinHash signatures
    fn jaccard_similarity(&self, sig1: &MinHashSignature, sig2: &MinHashSignature) -> f64 {
        let matches = sig1.values.iter().zip(sig2.values.iter()).filter(|(a, b)| a == b).count();

        matches as f64 / self.num_permutations as f64
    }

    /// Find similar fragments using LSH
    fn find_duplicates(&self, fragments: &[CodeFragment]) -> Vec<DuplicateCluster> {
        if fragments.len() < 2 {
            return Vec::new();
        }

        // Compute signatures for all fragments
        let signatures: Vec<MinHashSignature> =
            fragments.iter().map(|f| self.compute_minhash(f)).collect();

        // Use LSH to find candidate pairs
        let mut similar_pairs: Vec<(usize, usize, f64)> = Vec::new();

        // Band-based LSH for faster candidate generation
        let bands = 20;
        let rows_per_band = self.num_permutations / bands;
        let mut buckets: HashMap<(usize, Vec<u64>), Vec<usize>> = HashMap::new();

        for (idx, sig) in signatures.iter().enumerate() {
            for band in 0..bands {
                let start = band * rows_per_band;
                let end = start + rows_per_band;
                let band_hash: Vec<u64> = sig.values[start..end].to_vec();
                buckets.entry((band, band_hash)).or_default().push(idx);
            }
        }

        // Check candidates
        let mut checked: HashSet<(usize, usize)> = HashSet::new();
        for indices in buckets.values() {
            for i in 0..indices.len() {
                for j in (i + 1)..indices.len() {
                    let idx1 = indices[i];
                    let idx2 = indices[j];
                    let key = (idx1.min(idx2), idx1.max(idx2));

                    if checked.contains(&key) {
                        continue;
                    }
                    checked.insert(key);

                    let sim = self.jaccard_similarity(&signatures[idx1], &signatures[idx2]);
                    if sim >= self.similarity_threshold {
                        similar_pairs.push((idx1, idx2, sim));
                    }
                }
            }
        }

        // Cluster similar fragments using union-find
        let mut clusters = self.cluster_fragments(fragments, &similar_pairs);

        // Filter to only significant clusters
        clusters.retain(|c| c.fragments.len() >= 2);
        clusters.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));

        clusters
    }

    /// Cluster fragments using union-find
    fn cluster_fragments(
        &self,
        fragments: &[CodeFragment],
        pairs: &[(usize, usize, f64)],
    ) -> Vec<DuplicateCluster> {
        let n = fragments.len();
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank: Vec<usize> = vec![0; n];

        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) {
            let px = find(parent, x);
            let py = find(parent, y);
            if px == py {
                return;
            }
            match rank[px].cmp(&rank[py]) {
                std::cmp::Ordering::Less => parent[px] = py,
                std::cmp::Ordering::Greater => parent[py] = px,
                std::cmp::Ordering::Equal => {
                    parent[py] = px;
                    rank[px] += 1;
                }
            }
        }

        // Union similar pairs
        for (i, j, _sim) in pairs {
            union(&mut parent, &mut rank, *i, *j);
        }

        // Group by cluster
        let mut cluster_map: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
        for (i, j, sim) in pairs {
            let root = find(&mut parent, *i);
            cluster_map.entry(root).or_default().push((*i, *sim));
            cluster_map.entry(root).or_default().push((*j, *sim));
        }

        // Build clusters
        let mut clusters = Vec::new();
        for (_root, members) in cluster_map {
            let mut seen = HashSet::new();
            let mut cluster_fragments = Vec::new();
            let mut max_sim = 0.0f64;

            for (idx, sim) in members {
                if seen.insert(idx) {
                    cluster_fragments.push(fragments[idx].clone());
                    max_sim = max_sim.max(sim);
                }
            }

            if cluster_fragments.len() >= 2 {
                clusters
                    .push(DuplicateCluster { fragments: cluster_fragments, similarity: max_sim });
            }
        }

        clusters
    }

    /// Check if a file matches include/exclude patterns
    fn should_include(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();

        // Check exclude patterns first
        for pattern in &self.exclude_patterns {
            if glob_match(pattern, &path_str) {
                return false;
            }
        }

        // Check include patterns
        for pattern in &self.include_patterns {
            if glob_match(pattern, &path_str) {
                return true;
            }
        }

        false
    }
}

/// Simple glob matching (supports ** and *)
fn glob_match(pattern: &str, path: &str) -> bool {
    let pattern_parts: Vec<&str> = pattern.split('/').collect();
    let path_parts: Vec<&str> = path.split('/').collect();

    glob_match_parts(&pattern_parts, &path_parts)
}

/// Handle ** glob pattern matching (recursive)
fn glob_match_doublestar(pattern: &[&str], path: &[&str]) -> bool {
    let pattern_rest = pattern.get(1..).unwrap_or(&[]);
    let path_rest = path.get(1..).unwrap_or(&[]);
    // ** matches zero directories: try rest of pattern with current path
    if glob_match_parts(pattern_rest, path) {
        return true;
    }
    // ** matches one+ directories: try same pattern with rest of path
    if !path.is_empty() && glob_match_parts(pattern, path_rest) {
        return true;
    }
    // ** matches current and move both forward
    !path.is_empty() && glob_match_parts(pattern_rest, path_rest)
}

fn glob_match_parts(pattern: &[&str], path: &[&str]) -> bool {
    if pattern.is_empty() {
        return path.is_empty();
    }

    if path.is_empty() {
        return pattern.iter().all(|p| *p == "**");
    }

    let pat_first = match pattern.first() {
        Some(p) => *p,
        None => return path.is_empty(),
    };

    if pat_first == "**" {
        return glob_match_doublestar(pattern, path);
    }

    let path_first = match path.first() {
        Some(p) => *p,
        None => return false,
    };
    let pattern_rest = pattern.get(1..).unwrap_or(&[]);
    let path_rest = path.get(1..).unwrap_or(&[]);

    // Regular segment: must match and continue
    segment_match(pat_first, path_first) && glob_match_parts(pattern_rest, path_rest)
}

fn segment_match(pattern: &str, segment: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    if pattern.contains('*') {
        // Simple wildcard matching
        let parts: Vec<&str> = pattern.split('*').collect();
        if parts.len() == 2 {
            segment.starts_with(parts[0]) && segment.ends_with(parts[1])
        } else {
            pattern == segment
        }
    } else {
        pattern == segment
    }
}

/// A code fragment for duplication analysis
#[derive(Debug, Clone)]
struct CodeFragment {
    path: std::path::PathBuf,
    start_line: usize,
    end_line: usize,
    content: String,
}

/// MinHash signature
#[derive(Debug)]
struct MinHashSignature {
    values: Vec<u64>,
}

/// A cluster of duplicate code fragments
#[derive(Debug)]
struct DuplicateCluster {
    fragments: Vec<CodeFragment>,
    similarity: f64,
}

impl StackComplianceRule for DuplicationRule {
    fn id(&self) -> &'static str {
        "code-duplication"
    }

    fn description(&self) -> &'static str {
        "Detects significant code duplication using MinHash+LSH"
    }

    fn help(&self) -> Option<&str> {
        Some(
            "Threshold: 85% similarity, Minimum: 50 lines\n\
             Uses MinHash+LSH for efficient near-duplicate detection",
        )
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Code
    }

    fn check(&self, project_path: &Path) -> anyhow::Result<RuleResult> {
        // Collect all source files
        let mut fragments = Vec::new();

        for entry in walkdir::WalkDir::new(project_path).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_file() && self.should_include(path) {
                match self.extract_fragments(path) {
                    Ok(frags) => fragments.extend(frags),
                    Err(_) => continue, // Skip files that can't be read
                }
            }
        }

        // Find duplicates
        let clusters = self.find_duplicates(&fragments);

        if clusters.is_empty() {
            return Ok(RuleResult::pass());
        }

        let mut violations = Vec::new();
        let mut suggestions = Vec::new();

        for (i, cluster) in clusters.iter().enumerate() {
            // Only report violations for very high similarity (likely copy-paste)
            if cluster.similarity >= 0.95 {
                let locations: Vec<String> = cluster
                    .fragments
                    .iter()
                    .map(|f| format!("{}:{}-{}", f.path.display(), f.start_line, f.end_line))
                    .collect();

                violations.push(
                    RuleViolation::new(
                        format!("DUP-{:03}", i + 1),
                        format!(
                            "High code duplication ({:.0}%) across {} locations",
                            cluster.similarity * 100.0,
                            cluster.fragments.len()
                        ),
                    )
                    .with_severity(ViolationLevel::Warning)
                    .with_location(locations.join(", ")),
                );
            } else {
                // Lower similarity is a suggestion
                let locations: Vec<String> = cluster
                    .fragments
                    .iter()
                    .take(3)
                    .map(|f| format!("{}:{}", f.path.display(), f.start_line))
                    .collect();

                suggestions.push(
                    Suggestion::new(format!(
                        "Similar code ({:.0}%) found in {} locations: {}",
                        cluster.similarity * 100.0,
                        cluster.fragments.len(),
                        locations.join(", ")
                    ))
                    .with_fix("Consider extracting to a shared module".to_string()),
                );
            }
        }

        if violations.is_empty() {
            Ok(RuleResult::pass_with_suggestions(suggestions))
        } else {
            let mut result = RuleResult::fail(violations);
            result.suggestions = suggestions;
            Ok(result)
        }
    }

    fn can_fix(&self) -> bool {
        false // Duplication requires manual refactoring
    }

    fn fix(&self, _project_path: &Path) -> anyhow::Result<FixResult> {
        Ok(FixResult::failure(
            "Auto-fix not supported for code duplication - manual refactoring required",
        ))
    }
}

#[cfg(test)]
#[path = "duplication_tests.rs"]
mod tests;
