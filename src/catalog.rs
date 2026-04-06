//! Model Catalog and Discovery
//!
//! Unified interface for discovering models across local and remote registries.
//!
//! ## Features
//!
//! - Search across multiple sources (local, remote, HuggingFace)
//! - Filter by model type, size, task, quantization
//! - Sort by popularity, size, date, name
//! - Cached catalog for fast local queries
//!
//! ## Example
//!
//! ```rust,ignore
//! use pacha::catalog::{ModelCatalog, SearchQuery};
//!
//! let catalog = ModelCatalog::new()
//!     .with_local_registry(registry)
//!     .with_remote_registry(remote);
//!
//! let results = catalog.search(
//!     SearchQuery::new()
//!         .with_text("llama")
//!         .with_task(Task::TextGeneration)
//!         .with_max_size_gb(8.0)
//! ).await?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// CAT-001: Catalog Entry
// ============================================================================

/// A model entry in the catalog
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    /// Unique identifier
    pub id: String,
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Source (local, remote, huggingface)
    pub source: ModelSource,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Task category
    pub task: Option<Task>,
    /// Architecture (e.g., "llama", "mistral", "phi")
    pub architecture: Option<String>,
    /// Quantization type if quantized
    pub quantization: Option<String>,
    /// Context length
    pub context_length: Option<u32>,
    /// Number of parameters (approximate)
    pub parameters: Option<u64>,
    /// License
    pub license: Option<String>,
    /// Description
    pub description: Option<String>,
    /// Tags for filtering
    pub tags: Vec<String>,
    /// Download count (for popularity)
    pub downloads: u64,
    /// Last updated timestamp
    pub updated_at: Option<String>,
    /// URI for accessing the model
    pub uri: String,
}

impl CatalogEntry {
    /// Create a new catalog entry
    #[must_use]
    pub fn new(name: impl Into<String>, version: impl Into<String>, source: ModelSource) -> Self {
        let name = name.into();
        let version = version.into();
        let uri = match source {
            ModelSource::Local => format!("pacha://{name}:{version}"),
            ModelSource::Remote { ref host } => format!("pacha://{host}/{name}:{version}"),
            ModelSource::HuggingFace => format!("hf://{name}"),
        };

        Self {
            id: format!("{name}:{version}"),
            name,
            version,
            source,
            size_bytes: 0,
            task: None,
            architecture: None,
            quantization: None,
            context_length: None,
            parameters: None,
            license: None,
            description: None,
            tags: Vec::new(),
            downloads: 0,
            updated_at: None,
            uri,
        }
    }

    /// Set size in bytes
    #[must_use]
    pub fn with_size(mut self, bytes: u64) -> Self {
        self.size_bytes = bytes;
        self
    }

    /// Set task
    #[must_use]
    pub fn with_task(mut self, task: Task) -> Self {
        self.task = Some(task);
        self
    }

    /// Set architecture
    #[must_use]
    pub fn with_architecture(mut self, arch: impl Into<String>) -> Self {
        self.architecture = Some(arch.into());
        self
    }

    /// Set quantization
    #[must_use]
    pub fn with_quantization(mut self, quant: impl Into<String>) -> Self {
        self.quantization = Some(quant.into());
        self
    }

    /// Set context length
    #[must_use]
    pub fn with_context_length(mut self, length: u32) -> Self {
        self.context_length = Some(length);
        self
    }

    /// Set parameters
    #[must_use]
    pub fn with_parameters(mut self, params: u64) -> Self {
        self.parameters = Some(params);
        self
    }

    /// Set license
    #[must_use]
    pub fn with_license(mut self, license: impl Into<String>) -> Self {
        self.license = Some(license.into());
        self
    }

    /// Set description
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add tag
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add multiple tags
    #[must_use]
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags.extend(tags);
        self
    }

    /// Set downloads
    #[must_use]
    pub fn with_downloads(mut self, downloads: u64) -> Self {
        self.downloads = downloads;
        self
    }

    /// Get size in GB
    #[must_use]
    pub fn size_gb(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Check if model matches a text query
    #[must_use]
    pub fn matches_text(&self, query: &str) -> bool {
        let query = query.to_lowercase();
        self.name.to_lowercase().contains(&query)
            || self.description.as_ref().is_some_and(|d| d.to_lowercase().contains(&query))
            || self.tags.iter().any(|t| t.to_lowercase().contains(&query))
            || self.architecture.as_ref().is_some_and(|a| a.to_lowercase().contains(&query))
    }
}

/// Model source
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelSource {
    /// Local Pacha registry
    Local,
    /// Remote Pacha registry
    Remote {
        /// Registry host
        host: String,
    },
    /// HuggingFace Hub
    HuggingFace,
}

/// Task category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Task {
    /// Text generation (LLMs)
    TextGeneration,
    /// Text classification
    TextClassification,
    /// Question answering
    QuestionAnswering,
    /// Summarization
    Summarization,
    /// Translation
    Translation,
    /// Image classification
    ImageClassification,
    /// Object detection
    ObjectDetection,
    /// Image generation
    ImageGeneration,
    /// Speech recognition
    SpeechRecognition,
    /// Text to speech
    TextToSpeech,
    /// Embedding generation
    Embedding,
    /// Code generation
    CodeGeneration,
    /// Multi-modal
    MultiModal,
    /// Other/unknown
    Other,
}

impl Task {
    /// Get display name
    #[must_use]
    pub const fn display_name(&self) -> &'static str {
        match self {
            Self::TextGeneration => "Text Generation",
            Self::TextClassification => "Text Classification",
            Self::QuestionAnswering => "Question Answering",
            Self::Summarization => "Summarization",
            Self::Translation => "Translation",
            Self::ImageClassification => "Image Classification",
            Self::ObjectDetection => "Object Detection",
            Self::ImageGeneration => "Image Generation",
            Self::SpeechRecognition => "Speech Recognition",
            Self::TextToSpeech => "Text to Speech",
            Self::Embedding => "Embedding",
            Self::CodeGeneration => "Code Generation",
            Self::MultiModal => "Multi-Modal",
            Self::Other => "Other",
        }
    }
}

// ============================================================================
// CAT-002: Search Query
// ============================================================================

/// Search query for catalog
#[derive(Debug, Clone, Default)]
pub struct SearchQuery {
    /// Text search (name, description, tags)
    pub text: Option<String>,
    /// Filter by task
    pub task: Option<Task>,
    /// Filter by source
    pub source: Option<ModelSource>,
    /// Filter by architecture
    pub architecture: Option<String>,
    /// Filter by quantization
    pub quantization: Option<String>,
    /// Maximum size in GB
    pub max_size_gb: Option<f64>,
    /// Minimum size in GB
    pub min_size_gb: Option<f64>,
    /// Minimum context length
    pub min_context_length: Option<u32>,
    /// Filter by license
    pub license: Option<String>,
    /// Filter by tags (any match)
    pub tags: Vec<String>,
    /// Sort order
    pub sort: SortOrder,
    /// Maximum results
    pub limit: usize,
    /// Offset for pagination
    pub offset: usize,
}

impl SearchQuery {
    /// Create a new search query
    #[must_use]
    pub fn new() -> Self {
        Self { limit: 50, ..Default::default() }
    }

    /// Set text search
    #[must_use]
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Filter by task
    #[must_use]
    pub fn with_task(mut self, task: Task) -> Self {
        self.task = Some(task);
        self
    }

    /// Filter by source
    #[must_use]
    pub fn with_source(mut self, source: ModelSource) -> Self {
        self.source = Some(source);
        self
    }

    /// Filter by architecture
    #[must_use]
    pub fn with_architecture(mut self, arch: impl Into<String>) -> Self {
        self.architecture = Some(arch.into());
        self
    }

    /// Filter by quantization
    #[must_use]
    pub fn with_quantization(mut self, quant: impl Into<String>) -> Self {
        self.quantization = Some(quant.into());
        self
    }

    /// Set maximum size
    #[must_use]
    pub fn with_max_size_gb(mut self, gb: f64) -> Self {
        self.max_size_gb = Some(gb);
        self
    }

    /// Set minimum size
    #[must_use]
    pub fn with_min_size_gb(mut self, gb: f64) -> Self {
        self.min_size_gb = Some(gb);
        self
    }

    /// Set minimum context length
    #[must_use]
    pub fn with_min_context_length(mut self, length: u32) -> Self {
        self.min_context_length = Some(length);
        self
    }

    /// Filter by license
    #[must_use]
    pub fn with_license(mut self, license: impl Into<String>) -> Self {
        self.license = Some(license.into());
        self
    }

    /// Add tag filter
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set sort order
    #[must_use]
    pub fn with_sort(mut self, sort: SortOrder) -> Self {
        self.sort = sort;
        self
    }

    /// Set result limit
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set offset for pagination
    #[must_use]
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Check if an entry matches this query
    #[must_use]
    pub fn matches(&self, entry: &CatalogEntry) -> bool {
        self.matches_text_filter(entry)
            && self.matches_task_filter(entry)
            && self.matches_source_filter(entry)
            && self.matches_arch_filter(entry)
            && self.matches_quant_filter(entry)
            && self.matches_size_filter(entry)
            && self.matches_context_filter(entry)
            && self.matches_license_filter(entry)
            && self.matches_tag_filter(entry)
    }

    fn matches_text_filter(&self, entry: &CatalogEntry) -> bool {
        self.text.as_ref().map_or(true, |text| entry.matches_text(text))
    }

    fn matches_task_filter(&self, entry: &CatalogEntry) -> bool {
        self.task.map_or(true, |task| entry.task == Some(task))
    }

    fn matches_source_filter(&self, entry: &CatalogEntry) -> bool {
        self.source.as_ref().map_or(true, |source| &entry.source == source)
    }

    fn matches_arch_filter(&self, entry: &CatalogEntry) -> bool {
        self.architecture.as_ref().map_or(true, |arch| entry.architecture.as_ref() == Some(arch))
    }

    fn matches_quant_filter(&self, entry: &CatalogEntry) -> bool {
        self.quantization.as_ref().map_or(true, |quant| entry.quantization.as_ref() == Some(quant))
    }

    fn matches_size_filter(&self, entry: &CatalogEntry) -> bool {
        let within_max = self.max_size_gb.map_or(true, |max| entry.size_gb() <= max);
        let within_min = self.min_size_gb.map_or(true, |min| entry.size_gb() >= min);
        within_max && within_min
    }

    fn matches_context_filter(&self, entry: &CatalogEntry) -> bool {
        self.min_context_length.map_or(true, |min_ctx| entry.context_length.unwrap_or(0) >= min_ctx)
    }

    fn matches_license_filter(&self, entry: &CatalogEntry) -> bool {
        self.license.as_ref().map_or(true, |lic| entry.license.as_ref() == Some(lic))
    }

    fn matches_tag_filter(&self, entry: &CatalogEntry) -> bool {
        self.tags.is_empty() || self.tags.iter().any(|t| entry.tags.contains(t))
    }
}

/// Sort order for search results
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SortOrder {
    /// Sort by name (alphabetical)
    #[default]
    Name,
    /// Sort by downloads (most popular first)
    Downloads,
    /// Sort by size (smallest first)
    SizeAsc,
    /// Sort by size (largest first)
    SizeDesc,
    /// Sort by date (newest first)
    DateDesc,
    /// Sort by date (oldest first)
    DateAsc,
    /// Sort by parameters (smallest first)
    ParametersAsc,
    /// Sort by parameters (largest first)
    ParametersDesc,
}

// ============================================================================
// CAT-003: Search Results
// ============================================================================

/// Search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    /// Matching entries
    pub entries: Vec<CatalogEntry>,
    /// Total matches (before pagination)
    pub total: usize,
    /// Query offset
    pub offset: usize,
    /// Query limit
    pub limit: usize,
}

impl SearchResults {
    /// Create new search results
    #[must_use]
    pub fn new(entries: Vec<CatalogEntry>, total: usize, offset: usize, limit: usize) -> Self {
        Self { entries, total, offset, limit }
    }

    /// Check if there are more results
    #[must_use]
    pub fn has_more(&self) -> bool {
        self.offset + self.entries.len() < self.total
    }

    /// Get next page offset
    #[must_use]
    pub fn next_offset(&self) -> Option<usize> {
        if self.has_more() {
            Some(self.offset + self.limit)
        } else {
            None
        }
    }
}

// ============================================================================
// CAT-004: Model Catalog
// ============================================================================

/// Unified model catalog
#[derive(Debug, Default)]
pub struct ModelCatalog {
    /// In-memory cache of catalog entries
    entries: Vec<CatalogEntry>,
    /// Index by name for fast lookup
    by_name: HashMap<String, Vec<usize>>,
    /// Index by source
    by_source: HashMap<String, Vec<usize>>,
}

impl ModelCatalog {
    /// Create a new empty catalog
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entry to the catalog
    pub fn add(&mut self, entry: CatalogEntry) {
        let idx = self.entries.len();

        // Index by name
        self.by_name.entry(entry.name.clone()).or_default().push(idx);

        // Index by source
        let source_key = match &entry.source {
            ModelSource::Local => "local".to_string(),
            ModelSource::Remote { host } => format!("remote:{host}"),
            ModelSource::HuggingFace => "huggingface".to_string(),
        };
        self.by_source.entry(source_key).or_default().push(idx);

        self.entries.push(entry);
    }

    /// Get total number of entries
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if catalog is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get entry by index
    #[must_use]
    pub fn get(&self, idx: usize) -> Option<&CatalogEntry> {
        self.entries.get(idx)
    }

    /// Get entries by name
    #[must_use]
    pub fn get_by_name(&self, name: &str) -> Vec<&CatalogEntry> {
        self.by_name
            .get(name)
            .map(|indices| indices.iter().filter_map(|&i| self.entries.get(i)).collect())
            .unwrap_or_default()
    }

    /// Search the catalog
    #[must_use]
    pub fn search(&self, query: &SearchQuery) -> SearchResults {
        // Filter
        let mut matches: Vec<&CatalogEntry> =
            self.entries.iter().filter(|e| query.matches(e)).collect();

        let total = matches.len();

        // Sort
        match query.sort {
            SortOrder::Name => matches.sort_by(|a, b| a.name.cmp(&b.name)),
            SortOrder::Downloads => matches.sort_by(|a, b| b.downloads.cmp(&a.downloads)),
            SortOrder::SizeAsc => matches.sort_by(|a, b| a.size_bytes.cmp(&b.size_bytes)),
            SortOrder::SizeDesc => matches.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes)),
            SortOrder::DateDesc => {
                matches.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
            }
            SortOrder::DateAsc => {
                matches.sort_by(|a, b| a.updated_at.cmp(&b.updated_at));
            }
            SortOrder::ParametersAsc => {
                matches.sort_by(|a, b| a.parameters.cmp(&b.parameters));
            }
            SortOrder::ParametersDesc => {
                matches.sort_by(|a, b| b.parameters.cmp(&a.parameters));
            }
        }

        // Paginate
        let entries: Vec<CatalogEntry> =
            matches.into_iter().skip(query.offset).take(query.limit).cloned().collect();

        SearchResults::new(entries, total, query.offset, query.limit)
    }

    /// List all unique architectures
    #[must_use]
    pub fn architectures(&self) -> Vec<String> {
        let mut archs: Vec<_> =
            self.entries.iter().filter_map(|e| e.architecture.clone()).collect();
        archs.sort();
        archs.dedup();
        archs
    }

    /// List all unique tags
    #[must_use]
    pub fn tags(&self) -> Vec<String> {
        let mut tags: Vec<_> = self.entries.iter().flat_map(|e| e.tags.clone()).collect();
        tags.sort();
        tags.dedup();
        tags
    }

    /// List all unique licenses
    #[must_use]
    pub fn licenses(&self) -> Vec<String> {
        let mut licenses: Vec<_> = self.entries.iter().filter_map(|e| e.license.clone()).collect();
        licenses.sort();
        licenses.dedup();
        licenses
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> CatalogStats {
        let total_models = self.entries.len();
        let total_size: u64 = self.entries.iter().map(|e| e.size_bytes).sum();

        let local_count = self.by_source.get("local").map_or(0, Vec::len);
        let hf_count = self.by_source.get("huggingface").map_or(0, Vec::len);
        let remote_count = total_models - local_count - hf_count;

        let by_task: HashMap<String, usize> = {
            let mut map = HashMap::new();
            for entry in &self.entries {
                if let Some(task) = entry.task {
                    *map.entry(task.display_name().to_string()).or_insert(0) += 1;
                }
            }
            map
        };

        CatalogStats {
            total_models,
            total_size_bytes: total_size,
            local_count,
            remote_count,
            huggingface_count: hf_count,
            by_task,
            unique_architectures: self.architectures().len(),
            unique_tags: self.tags().len(),
        }
    }
}

/// Catalog statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogStats {
    /// Total number of models
    pub total_models: usize,
    /// Total size in bytes
    pub total_size_bytes: u64,
    /// Local models count
    pub local_count: usize,
    /// Remote models count
    pub remote_count: usize,
    /// HuggingFace models count
    pub huggingface_count: usize,
    /// Models by task
    pub by_task: HashMap<String, usize>,
    /// Number of unique architectures
    pub unique_architectures: usize,
    /// Number of unique tags
    pub unique_tags: usize,
}

impl CatalogStats {
    /// Get total size in GB
    #[must_use]
    pub fn total_size_gb(&self) -> f64 {
        self.total_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // CAT-001: Entry Tests
    // ========================================================================

    #[test]
    fn test_catalog_entry_new() {
        let entry = CatalogEntry::new("llama3", "8b", ModelSource::Local);
        assert_eq!(entry.name, "llama3");
        assert_eq!(entry.version, "8b");
        assert_eq!(entry.uri, "pacha://llama3:8b");
    }

    #[test]
    fn test_catalog_entry_builder() {
        let entry = CatalogEntry::new("mistral", "7b-q4", ModelSource::Local)
            .with_size(4_000_000_000)
            .with_task(Task::TextGeneration)
            .with_architecture("mistral")
            .with_quantization("Q4_K_M")
            .with_context_length(8192)
            .with_parameters(7_000_000_000)
            .with_license("Apache-2.0")
            .with_description("Mistral 7B quantized")
            .with_tag("llm")
            .with_downloads(10000);

        assert_eq!(entry.size_bytes, 4_000_000_000);
        assert_eq!(entry.task, Some(Task::TextGeneration));
        assert_eq!(entry.architecture, Some("mistral".to_string()));
        assert_eq!(entry.quantization, Some("Q4_K_M".to_string()));
        assert_eq!(entry.context_length, Some(8192));
        assert_eq!(entry.downloads, 10000);
    }

    #[test]
    fn test_catalog_entry_size_gb() {
        let entry =
            CatalogEntry::new("test", "1.0", ModelSource::Local).with_size(4 * 1024 * 1024 * 1024); // 4 GB

        assert!((entry.size_gb() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_catalog_entry_matches_text() {
        let entry = CatalogEntry::new("llama3-8b", "1.0", ModelSource::Local)
            .with_description("Meta's Llama 3 model")
            .with_tag("meta")
            .with_architecture("llama");

        assert!(entry.matches_text("llama"));
        assert!(entry.matches_text("LLAMA")); // Case insensitive
        assert!(entry.matches_text("meta"));
        assert!(entry.matches_text("Meta's"));
        assert!(!entry.matches_text("gpt"));
    }

    #[test]
    fn test_model_source_remote() {
        let entry = CatalogEntry::new(
            "model",
            "1.0",
            ModelSource::Remote { host: "registry.example.com".to_string() },
        );
        assert_eq!(entry.uri, "pacha://registry.example.com/model:1.0");
    }

    #[test]
    fn test_model_source_huggingface() {
        let entry = CatalogEntry::new("meta-llama/Llama-2-7b", "main", ModelSource::HuggingFace);
        assert_eq!(entry.uri, "hf://meta-llama/Llama-2-7b");
    }

    // ========================================================================
    // CAT-002: Query Tests
    // ========================================================================

    #[test]
    fn test_search_query_default() {
        let query = SearchQuery::new();
        assert!(query.text.is_none());
        assert!(query.task.is_none());
        assert_eq!(query.limit, 50);
        assert_eq!(query.offset, 0);
    }

    #[test]
    fn test_search_query_builder() {
        let query = SearchQuery::new()
            .with_text("llama")
            .with_task(Task::TextGeneration)
            .with_architecture("llama")
            .with_max_size_gb(8.0)
            .with_sort(SortOrder::Downloads)
            .with_limit(10);

        assert_eq!(query.text, Some("llama".to_string()));
        assert_eq!(query.task, Some(Task::TextGeneration));
        assert_eq!(query.architecture, Some("llama".to_string()));
        assert_eq!(query.max_size_gb, Some(8.0));
        assert_eq!(query.sort, SortOrder::Downloads);
        assert_eq!(query.limit, 10);
    }

    #[test]
    fn test_search_query_matches_text() {
        let entry = CatalogEntry::new("llama3", "1.0", ModelSource::Local);
        let query = SearchQuery::new().with_text("llama");

        assert!(query.matches(&entry));

        let query = SearchQuery::new().with_text("gpt");
        assert!(!query.matches(&entry));
    }

    #[test]
    fn test_search_query_matches_task() {
        let entry =
            CatalogEntry::new("test", "1.0", ModelSource::Local).with_task(Task::TextGeneration);

        let query = SearchQuery::new().with_task(Task::TextGeneration);
        assert!(query.matches(&entry));

        let query = SearchQuery::new().with_task(Task::ImageClassification);
        assert!(!query.matches(&entry));
    }

    #[test]
    fn test_search_query_matches_size() {
        let entry =
            CatalogEntry::new("test", "1.0", ModelSource::Local).with_size(4 * 1024 * 1024 * 1024); // 4 GB

        let query = SearchQuery::new().with_max_size_gb(8.0);
        assert!(query.matches(&entry));

        let query = SearchQuery::new().with_max_size_gb(2.0);
        assert!(!query.matches(&entry));

        let query = SearchQuery::new().with_min_size_gb(2.0);
        assert!(query.matches(&entry));

        let query = SearchQuery::new().with_min_size_gb(8.0);
        assert!(!query.matches(&entry));
    }

    #[test]
    fn test_search_query_matches_tags() {
        let entry =
            CatalogEntry::new("test", "1.0", ModelSource::Local).with_tag("llm").with_tag("meta");

        let query = SearchQuery::new().with_tag("llm");
        assert!(query.matches(&entry));

        let query = SearchQuery::new().with_tag("gpt");
        assert!(!query.matches(&entry));
    }

    // ========================================================================
    // CAT-003: Results Tests
    // ========================================================================

    #[test]
    fn test_search_results_has_more() {
        // 100 total, got 10 entries starting at 0 -> more available
        let entries: Vec<CatalogEntry> = (0..10)
            .map(|i| CatalogEntry::new(format!("m{i}"), "1.0", ModelSource::Local))
            .collect();
        let results = SearchResults::new(entries, 100, 0, 10);
        assert!(results.has_more());

        // 10 total, got 10 entries starting at 0 -> no more
        let entries: Vec<CatalogEntry> = (0..10)
            .map(|i| CatalogEntry::new(format!("m{i}"), "1.0", ModelSource::Local))
            .collect();
        let results = SearchResults::new(entries, 10, 0, 10);
        assert!(!results.has_more());
    }

    #[test]
    fn test_search_results_next_offset() {
        // 100 total, got 10 entries starting at 0 -> next offset is 10
        let entries: Vec<CatalogEntry> = (0..10)
            .map(|i| CatalogEntry::new(format!("m{i}"), "1.0", ModelSource::Local))
            .collect();
        let results = SearchResults::new(entries, 100, 0, 10);
        assert_eq!(results.next_offset(), Some(10));

        // 100 total, got 10 entries starting at 90 -> no next (at end)
        let entries: Vec<CatalogEntry> = (0..10)
            .map(|i| CatalogEntry::new(format!("m{i}"), "1.0", ModelSource::Local))
            .collect();
        let results = SearchResults::new(entries, 100, 90, 10);
        assert_eq!(results.next_offset(), None);
    }

    // ========================================================================
    // CAT-004: Catalog Tests
    // ========================================================================

    #[test]
    fn test_catalog_new() {
        let catalog = ModelCatalog::new();
        assert!(catalog.is_empty());
        assert_eq!(catalog.len(), 0);
    }

    #[test]
    fn test_catalog_add() {
        let mut catalog = ModelCatalog::new();
        catalog.add(CatalogEntry::new("llama3", "8b", ModelSource::Local));

        assert_eq!(catalog.len(), 1);
        assert!(!catalog.is_empty());
    }

    #[test]
    fn test_catalog_get_by_name() {
        let mut catalog = ModelCatalog::new();
        catalog.add(CatalogEntry::new("llama3", "8b", ModelSource::Local));
        catalog.add(CatalogEntry::new("llama3", "70b", ModelSource::Local));
        catalog.add(CatalogEntry::new("mistral", "7b", ModelSource::Local));

        let entries = catalog.get_by_name("llama3");
        assert_eq!(entries.len(), 2);

        let entries = catalog.get_by_name("gpt");
        assert!(entries.is_empty());
    }

    #[test]
    fn test_catalog_search() {
        let mut catalog = ModelCatalog::new();
        catalog.add(
            CatalogEntry::new("llama3-8b", "1.0", ModelSource::Local)
                .with_task(Task::TextGeneration)
                .with_downloads(1000),
        );
        catalog.add(
            CatalogEntry::new("llama3-70b", "1.0", ModelSource::Local)
                .with_task(Task::TextGeneration)
                .with_downloads(500),
        );
        catalog.add(
            CatalogEntry::new("clip", "1.0", ModelSource::Local)
                .with_task(Task::ImageClassification),
        );

        // Search by text
        let results = catalog.search(&SearchQuery::new().with_text("llama"));
        assert_eq!(results.total, 2);

        // Search by task
        let results = catalog.search(&SearchQuery::new().with_task(Task::TextGeneration));
        assert_eq!(results.total, 2);

        // Sort by downloads
        let results =
            catalog.search(&SearchQuery::new().with_text("llama").with_sort(SortOrder::Downloads));
        assert_eq!(results.entries[0].name, "llama3-8b"); // More downloads
    }

    #[test]
    fn test_catalog_search_pagination() {
        let mut catalog = ModelCatalog::new();
        for i in 0..25 {
            catalog.add(CatalogEntry::new(format!("model-{i}"), "1.0", ModelSource::Local));
        }

        let results = catalog.search(&SearchQuery::new().with_limit(10));
        assert_eq!(results.entries.len(), 10);
        assert_eq!(results.total, 25);
        assert!(results.has_more());

        let results = catalog.search(&SearchQuery::new().with_limit(10).with_offset(20));
        assert_eq!(results.entries.len(), 5);
        assert!(!results.has_more());
    }

    #[test]
    fn test_catalog_architectures() {
        let mut catalog = ModelCatalog::new();
        catalog.add(CatalogEntry::new("m1", "1.0", ModelSource::Local).with_architecture("llama"));
        catalog
            .add(CatalogEntry::new("m2", "1.0", ModelSource::Local).with_architecture("mistral"));
        catalog.add(CatalogEntry::new("m3", "1.0", ModelSource::Local).with_architecture("llama"));

        let archs = catalog.architectures();
        assert_eq!(archs.len(), 2);
        assert!(archs.contains(&"llama".to_string()));
        assert!(archs.contains(&"mistral".to_string()));
    }

    #[test]
    fn test_catalog_stats() {
        let mut catalog = ModelCatalog::new();
        catalog.add(
            CatalogEntry::new("m1", "1.0", ModelSource::Local)
                .with_size(1024)
                .with_task(Task::TextGeneration),
        );
        catalog.add(
            CatalogEntry::new("m2", "1.0", ModelSource::HuggingFace)
                .with_size(2048)
                .with_task(Task::TextGeneration),
        );

        let stats = catalog.stats();
        assert_eq!(stats.total_models, 2);
        assert_eq!(stats.total_size_bytes, 3072);
        assert_eq!(stats.local_count, 1);
        assert_eq!(stats.huggingface_count, 1);
    }

    // ========================================================================
    // Task Tests
    // ========================================================================

    #[test]
    fn test_task_display_name() {
        assert_eq!(Task::TextGeneration.display_name(), "Text Generation");
        assert_eq!(Task::CodeGeneration.display_name(), "Code Generation");
        assert_eq!(Task::ImageClassification.display_name(), "Image Classification");
    }

    // ========================================================================
    // Serialization Tests
    // ========================================================================

    #[test]
    fn test_catalog_entry_serialization() {
        let entry = CatalogEntry::new("llama3", "8b", ModelSource::Local)
            .with_task(Task::TextGeneration)
            .with_size(4_000_000_000);

        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("llama3"));
        assert!(json.contains("TextGeneration"));

        let parsed: CatalogEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "llama3");
    }

    #[test]
    fn test_search_results_serialization() {
        let results = SearchResults::new(
            vec![CatalogEntry::new("test", "1.0", ModelSource::Local)],
            1,
            0,
            10,
        );

        let json = serde_json::to_string(&results).unwrap();
        let parsed: SearchResults = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.total, 1);
        assert_eq!(parsed.entries.len(), 1);
    }

    #[test]
    fn test_catalog_stats_serialization() {
        let stats = CatalogStats {
            total_models: 100,
            total_size_bytes: 1024 * 1024 * 1024,
            local_count: 50,
            remote_count: 30,
            huggingface_count: 20,
            by_task: HashMap::from([("Text Generation".to_string(), 80)]),
            unique_architectures: 5,
            unique_tags: 10,
        };

        let json = serde_json::to_string(&stats).unwrap();
        let parsed: CatalogStats = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.total_models, 100);
    }
}
