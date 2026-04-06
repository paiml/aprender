//! Decision pattern store implementation with hybrid retrieval.
//!
//! Uses trueno-rag for BM25 lexical search combined with dense embeddings
//! and Reciprocal Rank Fusion (RRF) for optimal fix suggestions.

use super::{ChunkId, FixPattern, FixSuggestion, PatternStoreConfig, PatternStoreData};
use std::collections::HashMap;
use std::path::Path;
use trueno_rag::{
    chunk::FixedSizeChunker, embed::MockEmbedder, fusion::FusionStrategy,
    pipeline::RagPipelineBuilder, rerank::NoOpReranker, Document, RagPipeline,
};

/// Store for decision patterns with hybrid retrieval
///
/// Uses trueno-rag for BM25 + dense embedding retrieval with RRF fusion.
///
/// # Example
///
/// ```ignore
/// use entrenar::citl::{DecisionPatternStore, FixPattern};
///
/// let mut store = DecisionPatternStore::new()?;
///
/// // Index a fix pattern
/// let pattern = FixPattern::new("E0308", "- let x: i32 = \"hello\";\n+ let x: &str = \"hello\";")
///     .with_decision("type_mismatch_detected")
///     .with_decision("infer_correct_type");
/// store.index_fix(pattern)?;
///
/// // Get fix suggestions
/// let suggestions = store.suggest_fix("E0308", &["type_mismatch"], 5)?;
/// ```
pub struct DecisionPatternStore {
    /// RAG pipeline for hybrid retrieval
    pipeline: RagPipeline<MockEmbedder, NoOpReranker>,
    /// Pattern storage indexed by chunk ID
    patterns: HashMap<ChunkId, FixPattern>,
    /// Error code index for fast filtering
    error_index: HashMap<String, Vec<ChunkId>>,
    /// Configuration
    config: PatternStoreConfig,
}

impl DecisionPatternStore {
    /// Create a new pattern store with default configuration
    pub fn new() -> Result<Self, crate::Error> {
        Self::with_config(PatternStoreConfig::default())
    }

    /// Create a new pattern store with custom configuration
    pub fn with_config(config: PatternStoreConfig) -> Result<Self, crate::Error> {
        let pipeline = RagPipelineBuilder::new()
            .chunker(FixedSizeChunker::new(config.chunk_size, config.chunk_size / 8))
            .embedder(MockEmbedder::new(config.embedding_dim))
            .reranker(NoOpReranker::new())
            .fusion(FusionStrategy::RRF { k: config.rrf_k })
            .build()
            .map_err(|e| crate::Error::ConfigError(format!("RAG pipeline error: {e}")))?;

        Ok(Self { pipeline, patterns: HashMap::new(), error_index: HashMap::new(), config })
    }

    /// Index a fix pattern for later retrieval
    pub fn index_fix(&mut self, pattern: FixPattern) -> Result<(), crate::Error> {
        let chunk_id = pattern.id;
        let error_code = pattern.error_code.clone();

        // Create searchable document
        let doc = Document::new(pattern.to_searchable_text())
            .with_title(format!("Fix for {}", pattern.error_code));

        // Index in RAG pipeline
        self.pipeline
            .index_document(&doc)
            .map_err(|e| crate::Error::ConfigError(format!("Indexing error: {e}")))?;

        // Update error index
        self.error_index.entry(error_code).or_default().push(chunk_id);

        // Store pattern
        self.patterns.insert(chunk_id, pattern);

        Ok(())
    }

    /// Suggest fixes for a given error code and decision context
    ///
    /// # Arguments
    ///
    /// * `error_code` - The error code to find fixes for
    /// * `decision_context` - Recent decisions that led to the error
    /// * `k` - Maximum number of suggestions to return
    ///
    /// # Returns
    ///
    /// Vector of fix suggestions ranked by relevance
    pub fn suggest_fix(
        &self,
        error_code: &str,
        decision_context: &[String],
        k: usize,
    ) -> Result<Vec<FixSuggestion>, crate::Error> {
        // Build query from error code and decision context
        let context_str = decision_context.join(" ");
        let query = format!("{error_code} {context_str}");

        // Retrieve from RAG pipeline
        let results = self
            .pipeline
            .query(&query, k * 2) // Over-fetch for filtering
            .map_err(|e| crate::Error::ConfigError(format!("Query error: {e}")))?;

        // Filter by error code if we have patterns for it
        let relevant_patterns: Vec<_> = if let Some(pattern_ids) = self.error_index.get(error_code)
        {
            pattern_ids.iter().filter_map(|id| self.patterns.get(id)).collect()
        } else {
            // Return any patterns if no exact error code match
            self.patterns.values().collect()
        };

        // Match RAG results with our patterns (by content similarity)
        let mut suggestions: Vec<FixSuggestion> = Vec::new();

        for (rank, result) in results.iter().enumerate() {
            // Find matching pattern by comparing content
            for pattern in &relevant_patterns {
                let pattern_text = pattern.to_searchable_text();
                if result.chunk.content.contains(&pattern.error_code)
                    || pattern_text.contains(&result.chunk.content)
                {
                    suggestions.push(FixSuggestion::new(
                        (*pattern).clone(),
                        result.best_score(),
                        rank,
                    ));
                    break;
                }
            }
        }

        // If no RAG matches, fall back to error index
        if suggestions.is_empty() && !relevant_patterns.is_empty() {
            for (rank, pattern) in relevant_patterns.iter().take(k).enumerate() {
                suggestions.push(FixSuggestion::new(
                    (*pattern).clone(),
                    1.0 - (rank as f32 * 0.1),
                    rank,
                ));
            }
        }

        // Sort by weighted score and limit
        suggestions.sort_by(|a, b| {
            b.weighted_score().partial_cmp(&a.weighted_score()).unwrap_or(std::cmp::Ordering::Equal)
        });
        suggestions.truncate(k);

        // Re-assign ranks after sorting
        for (rank, suggestion) in suggestions.iter_mut().enumerate() {
            suggestion.rank = rank;
        }

        Ok(suggestions)
    }

    /// Get the number of indexed patterns
    #[must_use]
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Check if the store is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Get a pattern by ID
    #[must_use]
    pub fn get(&self, id: &ChunkId) -> Option<&FixPattern> {
        self.patterns.get(id)
    }

    /// Get a mutable pattern by ID
    pub fn get_mut(&mut self, id: &ChunkId) -> Option<&mut FixPattern> {
        self.patterns.get_mut(id)
    }

    /// Update a pattern's success/failure count
    pub fn record_outcome(&mut self, id: &ChunkId, success: bool) {
        if let Some(pattern) = self.patterns.get_mut(id) {
            if success {
                pattern.record_success();
            } else {
                pattern.record_failure();
            }
        }
    }

    /// Get all patterns for an error code
    #[must_use]
    pub fn patterns_for_error(&self, error_code: &str) -> Vec<&FixPattern> {
        self.error_index
            .get(error_code)
            .map(|ids| ids.iter().filter_map(|id| self.patterns.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &PatternStoreConfig {
        &self.config
    }

    /// Export all patterns to JSON
    pub fn export_json(&self) -> Result<String, crate::Error> {
        let patterns: Vec<_> = self.patterns.values().collect();
        serde_json::to_string_pretty(&patterns)
            .map_err(|e| crate::Error::Serialization(format!("JSON export error: {e}")))
    }

    /// Import patterns from JSON
    pub fn import_json(&mut self, json: &str) -> Result<usize, crate::Error> {
        let patterns: Vec<FixPattern> = serde_json::from_str(json)
            .map_err(|e| crate::Error::Serialization(format!("JSON import error: {e}")))?;

        let count = patterns.len();
        for pattern in patterns {
            self.index_fix(pattern)?;
        }

        Ok(count)
    }

    /// Save patterns to .apr format (aprender model format)
    ///
    /// Uses `ModelType::Custom` with compressed MessagePack serialization.
    /// The .apr format provides:
    /// - CRC32 checksum (integrity)
    /// - Optional zstd compression
    /// - Compatible with aprender ecosystem
    ///
    /// # Example
    ///
    /// ```ignore
    /// use entrenar::citl::DecisionPatternStore;
    ///
    /// let store = DecisionPatternStore::new()?;
    /// // ... index patterns ...
    /// store.save_apr("decision_patterns.apr")?;
    /// ```
    pub fn save_apr(&self, path: impl AsRef<Path>) -> Result<(), crate::Error> {
        use aprender::format::{save, Compression, ModelType, SaveOptions};

        // Collect patterns into serializable wrapper
        let patterns: Vec<FixPattern> = self.patterns.values().cloned().collect();
        let wrapper = PatternStoreData { version: 1, config: self.config.clone(), patterns };

        save(
            &wrapper,
            ModelType::Custom,
            path,
            SaveOptions::default().with_compression(Compression::ZstdDefault),
        )
        .map_err(|e| crate::Error::Serialization(format!("APR save error: {e}")))
    }

    /// Load patterns from .apr format
    ///
    /// Restores patterns and rebuilds the RAG index.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use entrenar::citl::DecisionPatternStore;
    ///
    /// let store = DecisionPatternStore::load_apr("decision_patterns.apr")?;
    /// let suggestions = store.suggest_fix("E0308", &["type_mismatch".into()], 5)?;
    /// ```
    pub fn load_apr(path: impl AsRef<Path>) -> Result<Self, crate::Error> {
        use aprender::format::{load, ModelType};

        let wrapper: PatternStoreData = load(path, ModelType::Custom)
            .map_err(|e| crate::Error::Serialization(format!("APR load error: {e}")))?;

        // Rebuild store with loaded config
        let mut store = Self::with_config(wrapper.config)?;

        // Re-index all patterns
        for pattern in wrapper.patterns {
            store.index_fix(pattern)?;
        }

        Ok(store)
    }
}

impl std::fmt::Debug for DecisionPatternStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionPatternStore")
            .field("pattern_count", &self.patterns.len())
            .field("error_codes", &self.error_index.keys().collect::<Vec<_>>())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}
