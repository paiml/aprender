//! Query preprocessing strategies for improved retrieval.
//!
//! This module provides preprocessing techniques to enhance query effectiveness:
//! - HyDE (Hypothetical Document Embeddings): Generate hypothetical answers for better matching
//! - Multi-query expansion: Expand a single query into multiple related queries

use crate::Result;

/// A query preprocessor that transforms or expands queries before retrieval.
pub trait QueryPreprocessor: Send + Sync {
    /// Preprocess a query, potentially returning multiple expanded queries.
    fn preprocess(&self, query: &str) -> Result<Vec<String>>;

    /// Get the name of this preprocessor for debugging.
    fn name(&self) -> &str;
}

/// No-op preprocessor that returns the query unchanged.
#[derive(Debug, Clone, Default)]
pub struct PassthroughPreprocessor;

impl QueryPreprocessor for PassthroughPreprocessor {
    fn preprocess(&self, query: &str) -> Result<Vec<String>> {
        Ok(vec![query.to_string()])
    }

    fn name(&self) -> &str {
        "passthrough"
    }
}

/// HyDE (Hypothetical Document Embeddings) preprocessor.
///
/// Instead of searching with the query directly, HyDE generates a hypothetical
/// document that would answer the query, then uses that for retrieval.
/// This can improve semantic matching by generating content in the same
/// "language" as the documents being searched.
#[derive(Debug, Clone)]
pub struct HydePreprocessor<G: HypotheticalGenerator> {
    generator: G,
    include_original: bool,
}

/// Trait for generating hypothetical documents from queries.
pub trait HypotheticalGenerator: Send + Sync {
    /// Generate a hypothetical document that would answer the query.
    fn generate(&self, query: &str) -> Result<String>;
}

impl<G: HypotheticalGenerator> HydePreprocessor<G> {
    /// Create a new HyDE preprocessor with the given generator.
    pub fn new(generator: G) -> Self {
        Self { generator, include_original: false }
    }

    /// Include the original query alongside the hypothetical document.
    #[must_use]
    pub fn with_original_query(mut self, include: bool) -> Self {
        self.include_original = include;
        self
    }
}

impl<G: HypotheticalGenerator> QueryPreprocessor for HydePreprocessor<G> {
    fn preprocess(&self, query: &str) -> Result<Vec<String>> {
        let hypothetical = self.generator.generate(query)?;
        if self.include_original {
            Ok(vec![query.to_string(), hypothetical])
        } else {
            Ok(vec![hypothetical])
        }
    }

    fn name(&self) -> &str {
        "hyde"
    }
}

/// Anthropic-backed HyDE generator using Claude API.
///
/// Generates hypothetical documents by prompting Claude to write a passage
/// that would answer the given query. Requires `ANTHROPIC_API_KEY` environment
/// variable and the `eval` feature flag (reuses the eval Anthropic client).
#[cfg(feature = "eval")]
pub struct AnthropicHypotheticalGenerator {
    client: crate::eval::AnthropicClient,
    runtime: std::sync::Arc<tokio::runtime::Runtime>,
    model: String,
    max_tokens: u32,
}

#[cfg(feature = "eval")]
impl std::fmt::Debug for AnthropicHypotheticalGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicHypotheticalGenerator")
            .field("client", &"AnthropicClient{..}")
            .field("runtime", &"Runtime{..}")
            .field("model", &self.model)
            .field("max_tokens", &self.max_tokens)
            .finish()
    }
}

#[cfg(feature = "eval")]
impl AnthropicHypotheticalGenerator {
    /// Create a new Anthropic HyDE generator from environment.
    pub fn from_env() -> std::result::Result<Self, String> {
        let client = crate::eval::AnthropicClient::from_env()?;
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| format!("Failed to create tokio runtime: {e}"))?;
        Ok(Self {
            client,
            runtime: std::sync::Arc::new(runtime),
            model: "claude-haiku-4-5-20251001".to_string(),
            max_tokens: 256,
        })
    }

    /// Set the model to use for generation.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set maximum tokens for the hypothetical document.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }
}

#[cfg(feature = "eval")]
impl HypotheticalGenerator for AnthropicHypotheticalGenerator {
    fn generate(&self, query: &str) -> Result<String> {
        let system = "You are a technical content generator. Given a user query, write a short \
            passage (2-4 sentences) that directly answers the query as if it were an excerpt from \
            a lecture transcript or technical document. Output ONLY the passage text, no preamble \
            or formatting.";

        let result = self.runtime.block_on(self.client.complete(
            &self.model,
            Some(system),
            query,
            self.max_tokens,
        ));

        match result {
            Ok(completion) => Ok(completion.text),
            Err(e) => Err(crate::Error::InvalidConfig(format!("HyDE generation failed: {e}"))),
        }
    }
}

/// Mock HyDE generator for testing that creates a simple hypothetical answer.
#[derive(Debug, Clone, Default)]
pub struct MockHypotheticalGenerator {
    prefix: String,
}

impl MockHypotheticalGenerator {
    /// Create a new mock generator.
    pub fn new() -> Self {
        Self { prefix: "The answer is:".to_string() }
    }

    /// Set a custom prefix for generated hypotheticals.
    #[must_use]
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }
}

impl HypotheticalGenerator for MockHypotheticalGenerator {
    fn generate(&self, query: &str) -> Result<String> {
        Ok(format!("{} {}", self.prefix, query))
    }
}

/// Multi-query expansion preprocessor.
///
/// Expands a single query into multiple related queries using different
/// strategies. This can help retrieve documents that match different
/// phrasings or aspects of the original query.
#[derive(Debug, Clone)]
pub struct MultiQueryPreprocessor<E: QueryExpander> {
    expander: E,
    max_queries: usize,
    include_original: bool,
}

/// Trait for expanding queries into multiple related queries.
pub trait QueryExpander: Send + Sync {
    /// Expand a query into multiple related queries.
    fn expand(&self, query: &str) -> Result<Vec<String>>;
}

impl<E: QueryExpander> MultiQueryPreprocessor<E> {
    /// Create a new multi-query preprocessor.
    pub fn new(expander: E) -> Self {
        Self { expander, max_queries: 5, include_original: true }
    }

    /// Set the maximum number of expanded queries.
    #[must_use]
    pub fn with_max_queries(mut self, max: usize) -> Self {
        self.max_queries = max;
        self
    }

    /// Whether to include the original query in results.
    #[must_use]
    pub fn with_original_query(mut self, include: bool) -> Self {
        self.include_original = include;
        self
    }
}

impl<E: QueryExpander> QueryPreprocessor for MultiQueryPreprocessor<E> {
    fn preprocess(&self, query: &str) -> Result<Vec<String>> {
        let mut queries = if self.include_original { vec![query.to_string()] } else { vec![] };

        let expanded = self.expander.expand(query)?;
        for q in expanded {
            if queries.len() >= self.max_queries {
                break;
            }
            // Skip duplicates and, when original is excluded, skip expansions
            // that are identical to the original query
            if queries.contains(&q) || (!self.include_original && q == query) {
                continue;
            }
            queries.push(q);
        }

        Ok(queries)
    }

    fn name(&self) -> &str {
        "multi-query"
    }
}

/// Keyword-based query expander.
///
/// Expands queries by extracting key terms and creating variations.
#[derive(Debug, Clone, Default)]
pub struct KeywordExpander {
    stopwords: std::collections::HashSet<String>,
}

impl KeywordExpander {
    /// Create a new keyword expander with default stopwords.
    pub fn new() -> Self {
        let stopwords: std::collections::HashSet<String> = [
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
            "shall", "can", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
            "because", "until", "while", "what", "which", "who", "this", "that", "these", "those",
            "i", "me", "my", "myself", "we", "our", "you", "your", "he", "him", "she", "her", "it",
            "its", "they", "them", "their",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();

        Self { stopwords }
    }

    /// Extract keywords from text.
    fn extract_keywords(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty() && w.len() > 2 && !self.stopwords.contains(*w))
            .map(String::from)
            .collect()
    }
}

impl QueryExpander for KeywordExpander {
    fn expand(&self, query: &str) -> Result<Vec<String>> {
        let keywords = self.extract_keywords(query);
        let mut expansions = Vec::new();

        // Create query from just keywords
        if keywords.len() > 1 {
            expansions.push(keywords.join(" "));
        }

        // Create queries with individual important keywords emphasized
        for keyword in keywords.iter().take(3) {
            expansions.push(format!("{query} {keyword}"));
        }

        Ok(expansions)
    }
}

/// Synonym-based query expander.
///
/// Expands queries by replacing words with synonyms.
#[derive(Debug, Clone)]
pub struct SynonymExpander {
    synonyms: std::collections::HashMap<String, Vec<String>>,
}

impl SynonymExpander {
    /// Create a new synonym expander with the given synonym map.
    pub fn new(synonyms: std::collections::HashMap<String, Vec<String>>) -> Self {
        Self { synonyms }
    }

    /// Create an expander with default technical synonyms.
    pub fn with_technical_synonyms() -> Self {
        let mut synonyms = std::collections::HashMap::new();
        synonyms.insert(
            "error".to_string(),
            vec!["exception".to_string(), "failure".to_string(), "bug".to_string()],
        );
        synonyms
            .insert("function".to_string(), vec!["method".to_string(), "procedure".to_string()]);
        synonyms.insert(
            "create".to_string(),
            vec!["make".to_string(), "build".to_string(), "generate".to_string()],
        );
        synonyms.insert("delete".to_string(), vec!["remove".to_string(), "destroy".to_string()]);
        synonyms.insert(
            "update".to_string(),
            vec!["modify".to_string(), "change".to_string(), "edit".to_string()],
        );
        synonyms.insert(
            "find".to_string(),
            vec!["search".to_string(), "lookup".to_string(), "locate".to_string()],
        );
        synonyms.insert(
            "fast".to_string(),
            vec!["quick".to_string(), "rapid".to_string(), "speedy".to_string()],
        );
        synonyms.insert("slow".to_string(), vec!["sluggish".to_string(), "delayed".to_string()]);
        Self { synonyms }
    }
}

impl Default for SynonymExpander {
    fn default() -> Self {
        Self::with_technical_synonyms()
    }
}

impl QueryExpander for SynonymExpander {
    fn expand(&self, query: &str) -> Result<Vec<String>> {
        let mut expansions = Vec::new();
        let words: Vec<&str> = query.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            let lower = word.to_lowercase();
            if let Some(syns) = self.synonyms.get(&lower) {
                for syn in syns.iter().take(2) {
                    let mut new_words = words.clone();
                    new_words[i] = syn;
                    expansions.push(new_words.join(" "));
                }
            }
        }

        Ok(expansions)
    }
}

/// Chained preprocessor that applies multiple preprocessors in sequence.
#[allow(missing_debug_implementations)]
pub struct ChainedPreprocessor {
    preprocessors: Vec<Box<dyn QueryPreprocessor>>,
    deduplicate: bool,
    max_total: usize,
}

impl ChainedPreprocessor {
    /// Create a new chained preprocessor.
    pub fn new() -> Self {
        Self { preprocessors: Vec::new(), deduplicate: true, max_total: 10 }
    }

    /// Add a preprocessor to the chain.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn add<P: QueryPreprocessor + 'static>(mut self, preprocessor: P) -> Self {
        self.preprocessors.push(Box::new(preprocessor));
        self
    }

    /// Set maximum total queries to return.
    #[must_use]
    pub fn with_max_total(mut self, max: usize) -> Self {
        self.max_total = max;
        self
    }

    /// Whether to deduplicate queries.
    #[must_use]
    pub fn with_deduplicate(mut self, dedup: bool) -> Self {
        self.deduplicate = dedup;
        self
    }
}

impl Default for ChainedPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryPreprocessor for ChainedPreprocessor {
    fn preprocess(&self, query: &str) -> Result<Vec<String>> {
        if self.preprocessors.is_empty() {
            return Ok(vec![query.to_string()]);
        }

        let mut all_queries = Vec::new();

        for preprocessor in &self.preprocessors {
            let queries = preprocessor.preprocess(query)?;
            for q in queries {
                if all_queries.len() >= self.max_total {
                    break;
                }
                if !self.deduplicate || !all_queries.contains(&q) {
                    all_queries.push(q);
                }
            }
        }

        Ok(all_queries)
    }

    fn name(&self) -> &str {
        "chained"
    }
}

/// Query analyzer that extracts structured information from queries.
#[derive(Debug, Clone)]
pub struct QueryAnalyzer {
    intent_keywords: std::collections::HashMap<QueryIntent, Vec<String>>,
}

/// Detected intent of a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum QueryIntent {
    /// Looking for information or explanation.
    Informational,
    /// Looking for how to do something.
    HowTo,
    /// Looking for a definition.
    Definition,
    /// Looking for troubleshooting help.
    Troubleshooting,
    /// Looking to compare options.
    Comparison,
    /// Unknown intent.
    Unknown,
}

/// Analysis result for a query.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QueryAnalysis {
    /// Original query text.
    pub original: String,
    /// Detected intent.
    pub intent: QueryIntent,
    /// Extracted keywords.
    pub keywords: Vec<String>,
    /// Confidence score (0.0 - 1.0).
    pub confidence: f32,
}

impl QueryAnalyzer {
    /// Create a new query analyzer with default intent patterns.
    pub fn new() -> Self {
        let mut intent_keywords = std::collections::HashMap::new();

        intent_keywords.insert(
            QueryIntent::HowTo,
            vec![
                "how".to_string(),
                "tutorial".to_string(),
                "guide".to_string(),
                "steps".to_string(),
                "way".to_string(),
            ],
        );

        intent_keywords.insert(
            QueryIntent::Definition,
            vec![
                "what".to_string(),
                "define".to_string(),
                "meaning".to_string(),
                "definition".to_string(),
            ],
        );

        intent_keywords.insert(
            QueryIntent::Troubleshooting,
            vec![
                "error".to_string(),
                "fix".to_string(),
                "problem".to_string(),
                "issue".to_string(),
                "not working".to_string(),
                "failed".to_string(),
                "broken".to_string(),
            ],
        );

        intent_keywords.insert(
            QueryIntent::Comparison,
            vec![
                "vs".to_string(),
                "versus".to_string(),
                "compare".to_string(),
                "difference".to_string(),
                "better".to_string(),
            ],
        );

        Self { intent_keywords }
    }

    /// Analyze a query and return structured information.
    pub fn analyze(&self, query: &str) -> QueryAnalysis {
        let lower = query.to_lowercase();
        let mut best_intent = QueryIntent::Informational;
        let mut best_score = 0;

        for (intent, keywords) in &self.intent_keywords {
            let score = keywords.iter().filter(|kw| lower.contains(kw.as_str())).count();
            if score > best_score {
                best_score = score;
                best_intent = *intent;
            }
        }

        // Extract keywords
        let keywords: Vec<String> = lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty() && w.len() > 2)
            .map(String::from)
            .collect();

        let confidence =
            if best_score == 0 { 0.3 } else { (0.5 + 0.1 * best_score as f32).min(1.0) };

        QueryAnalysis {
            original: query.to_string(),
            intent: if best_score == 0 { QueryIntent::Unknown } else { best_intent },
            keywords,
            confidence,
        }
    }
}

impl Default for QueryAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Passthrough preprocessor tests

    #[test]
    fn test_passthrough_returns_original() {
        let preprocessor = PassthroughPreprocessor;
        let result = preprocessor.preprocess("test query").unwrap();
        assert_eq!(result, vec!["test query"]);
    }

    #[test]
    fn test_passthrough_name() {
        let preprocessor = PassthroughPreprocessor;
        assert_eq!(preprocessor.name(), "passthrough");
    }

    // HyDE preprocessor tests

    #[test]
    fn test_hyde_generates_hypothetical() {
        let generator = MockHypotheticalGenerator::new();
        let hyde = HydePreprocessor::new(generator);
        let result = hyde.preprocess("what is rust").unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].contains("The answer is:"));
        assert!(result[0].contains("what is rust"));
    }

    #[test]
    fn test_hyde_with_original() {
        let generator = MockHypotheticalGenerator::new();
        let hyde = HydePreprocessor::new(generator).with_original_query(true);
        let result = hyde.preprocess("test query").unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "test query");
        assert!(result[1].contains("The answer is:"));
    }

    #[test]
    fn test_hyde_custom_prefix() {
        let generator = MockHypotheticalGenerator::new().with_prefix("Answer:");
        let hyde = HydePreprocessor::new(generator);
        let result = hyde.preprocess("query").unwrap();
        assert!(result[0].starts_with("Answer:"));
    }

    #[test]
    fn test_hyde_name() {
        let generator = MockHypotheticalGenerator::new();
        let hyde = HydePreprocessor::new(generator);
        assert_eq!(hyde.name(), "hyde");
    }

    // Multi-query preprocessor tests

    #[test]
    fn test_multi_query_with_keyword_expander() {
        let expander = KeywordExpander::new();
        let multi = MultiQueryPreprocessor::new(expander);
        let result = multi.preprocess("rust programming language").unwrap();
        assert!(!result.is_empty());
        assert_eq!(result[0], "rust programming language"); // original first
    }

    #[test]
    fn test_multi_query_max_queries() {
        let expander = KeywordExpander::new();
        let multi = MultiQueryPreprocessor::new(expander).with_max_queries(2);
        let result = multi.preprocess("rust programming language").unwrap();
        assert!(result.len() <= 2);
    }

    #[test]
    fn test_multi_query_without_original() {
        let expander = KeywordExpander::new();
        let multi = MultiQueryPreprocessor::new(expander).with_original_query(false);
        let result = multi.preprocess("rust programming language").unwrap();
        assert!(!result.contains(&"rust programming language".to_string()));
    }

    #[test]
    fn test_multi_query_name() {
        let expander = KeywordExpander::new();
        let multi = MultiQueryPreprocessor::new(expander);
        assert_eq!(multi.name(), "multi-query");
    }

    // Keyword expander tests

    #[test]
    fn test_keyword_expander_extracts_keywords() {
        let expander = KeywordExpander::new();
        let keywords = expander.extract_keywords("the quick brown fox jumps");
        assert!(keywords.contains(&"quick".to_string()));
        assert!(keywords.contains(&"brown".to_string()));
        assert!(keywords.contains(&"jumps".to_string()));
        assert!(!keywords.contains(&"the".to_string())); // stopword
    }

    #[test]
    fn test_keyword_expander_filters_short_words() {
        let expander = KeywordExpander::new();
        let keywords = expander.extract_keywords("a go at it");
        assert!(keywords.is_empty() || !keywords.iter().any(|w| w.len() <= 2));
    }

    #[test]
    fn test_keyword_expander_expand() {
        let expander = KeywordExpander::new();
        let result = expander.expand("rust memory safety").unwrap();
        assert!(!result.is_empty());
    }

    // Synonym expander tests

    #[test]
    fn test_synonym_expander_basic() {
        let expander = SynonymExpander::with_technical_synonyms();
        let result = expander.expand("create a function").unwrap();
        assert!(!result.is_empty());
        // Should have variations with "make" or "build" instead of "create"
        assert!(result.iter().any(|q| q.contains("make") || q.contains("build")));
    }

    #[test]
    fn test_synonym_expander_no_synonyms() {
        let expander = SynonymExpander::with_technical_synonyms();
        let result = expander.expand("xyz abc def").unwrap();
        assert!(result.is_empty()); // no synonyms for these words
    }

    #[test]
    fn test_synonym_expander_custom_synonyms() {
        let mut synonyms = std::collections::HashMap::new();
        synonyms.insert("test".to_string(), vec!["check".to_string()]);
        let expander = SynonymExpander::new(synonyms);
        let result = expander.expand("test code").unwrap();
        assert!(result.iter().any(|q| q.contains("check")));
    }

    // Chained preprocessor tests

    #[test]
    fn test_chained_empty() {
        let chained = ChainedPreprocessor::new();
        let result = chained.preprocess("query").unwrap();
        assert_eq!(result, vec!["query"]);
    }

    #[test]
    fn test_chained_single() {
        let chained = ChainedPreprocessor::new().add(PassthroughPreprocessor);
        let result = chained.preprocess("query").unwrap();
        assert_eq!(result, vec!["query"]);
    }

    #[test]
    fn test_chained_multiple() {
        let chained = ChainedPreprocessor::new()
            .add(PassthroughPreprocessor)
            .add(HydePreprocessor::new(MockHypotheticalGenerator::new()));
        let result = chained.preprocess("query").unwrap();
        assert!(result.len() >= 2);
        assert!(result.contains(&"query".to_string()));
    }

    #[test]
    fn test_chained_deduplicates() {
        let chained = ChainedPreprocessor::new()
            .add(PassthroughPreprocessor)
            .add(PassthroughPreprocessor)
            .with_deduplicate(true);
        let result = chained.preprocess("query").unwrap();
        assert_eq!(result.len(), 1); // duplicates removed
    }

    #[test]
    fn test_chained_max_total() {
        let chained = ChainedPreprocessor::new()
            .add(MultiQueryPreprocessor::new(KeywordExpander::new()).with_max_queries(10))
            .with_max_total(3);
        let result = chained.preprocess("rust programming language tutorial").unwrap();
        assert!(result.len() <= 3);
    }

    #[test]
    fn test_chained_name() {
        let chained = ChainedPreprocessor::new();
        assert_eq!(chained.name(), "chained");
    }

    // Query analyzer tests

    #[test]
    fn test_analyzer_how_to() {
        let analyzer = QueryAnalyzer::new();
        let analysis = analyzer.analyze("how to write tests in rust");
        assert_eq!(analysis.intent, QueryIntent::HowTo);
        assert!(analysis.confidence > 0.5);
    }

    #[test]
    fn test_analyzer_definition() {
        let analyzer = QueryAnalyzer::new();
        let analysis = analyzer.analyze("what is a monad");
        assert_eq!(analysis.intent, QueryIntent::Definition);
    }

    #[test]
    fn test_analyzer_troubleshooting() {
        let analyzer = QueryAnalyzer::new();
        let analysis = analyzer.analyze("error compiling code fix");
        assert_eq!(analysis.intent, QueryIntent::Troubleshooting);
    }

    #[test]
    fn test_analyzer_comparison() {
        let analyzer = QueryAnalyzer::new();
        let analysis = analyzer.analyze("rust vs go comparison");
        assert_eq!(analysis.intent, QueryIntent::Comparison);
    }

    #[test]
    fn test_analyzer_unknown() {
        let analyzer = QueryAnalyzer::new();
        let analysis = analyzer.analyze("random words xyz");
        assert_eq!(analysis.intent, QueryIntent::Unknown);
        assert!(analysis.confidence < 0.5);
    }

    #[test]
    fn test_analyzer_extracts_keywords() {
        let analyzer = QueryAnalyzer::new();
        let analysis = analyzer.analyze("rust programming language");
        assert!(analysis.keywords.contains(&"rust".to_string()));
        assert!(analysis.keywords.contains(&"programming".to_string()));
        assert!(analysis.keywords.contains(&"language".to_string()));
    }

    #[test]
    fn test_query_analysis_serialization() {
        let analysis = QueryAnalysis {
            original: "test".to_string(),
            intent: QueryIntent::HowTo,
            keywords: vec!["test".to_string()],
            confidence: 0.8,
        };
        let json = serde_json::to_string(&analysis).unwrap();
        let deserialized: QueryAnalysis = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.original, "test");
        assert_eq!(deserialized.intent, QueryIntent::HowTo);
    }

    // Property-based tests
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_passthrough_preserves_input(query in "\\PC{1,100}") {
            let preprocessor = PassthroughPreprocessor;
            let result = preprocessor.preprocess(&query).unwrap();
            prop_assert_eq!(result.len(), 1);
            prop_assert_eq!(&result[0], &query);
        }

        #[test]
        fn prop_hyde_always_returns_something(query in "\\w{1,50}") {
            let hyde = HydePreprocessor::new(MockHypotheticalGenerator::new());
            let result = hyde.preprocess(&query).unwrap();
            prop_assert!(!result.is_empty());
        }

        #[test]
        fn prop_chained_respects_max_total(query in "\\w{1,50}", max in 1usize..20) {
            let chained = ChainedPreprocessor::new()
                .add(MultiQueryPreprocessor::new(KeywordExpander::new()))
                .add(HydePreprocessor::new(MockHypotheticalGenerator::new()))
                .with_max_total(max);
            let result = chained.preprocess(&query).unwrap();
            prop_assert!(result.len() <= max);
        }

        #[test]
        fn prop_analyzer_always_returns_analysis(query in "\\w{1,100}") {
            let analyzer = QueryAnalyzer::new();
            let analysis = analyzer.analyze(&query);
            prop_assert_eq!(analysis.original, query);
            prop_assert!(analysis.confidence >= 0.0 && analysis.confidence <= 1.0);
        }

        #[test]
        fn prop_keyword_expander_no_empty_results(
            w1 in "[a-z]{4,10}",
            w2 in "[a-z]{4,10}",
            w3 in "[a-z]{4,10}"
        ) {
            let expander = KeywordExpander::new();
            let query = format!("{w1} {w2} {w3}");
            let result = expander.expand(&query).unwrap();
            // All results should be non-empty strings
            for q in &result {
                prop_assert!(!q.is_empty());
            }
        }
    }
}
