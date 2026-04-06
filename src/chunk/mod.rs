//! Document chunking strategies for RAG pipelines

mod timestamp;
pub use timestamp::TimestampChunker;

use crate::{Document, DocumentId, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Stable replacement for `str::ceil_char_boundary` (unstable).
/// Returns the smallest index >= `i` that is a valid UTF-8 char boundary.
fn ceil_char_boundary(s: &str, i: usize) -> usize {
    if i >= s.len() {
        s.len()
    } else {
        let mut pos = i;
        while pos < s.len() && !s.is_char_boundary(pos) {
            pos += 1;
        }
        pos
    }
}

/// Unique chunk identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkId(pub uuid::Uuid);

impl ChunkId {
    /// Create a new random chunk ID
    #[must_use]
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl Default for ChunkId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ChunkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Metadata associated with a chunk
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Source document title
    pub title: Option<String>,
    /// Section/header hierarchy
    pub headers: Vec<String>,
    /// Page number (for PDFs)
    pub page: Option<usize>,
    /// Custom metadata
    pub custom: HashMap<String, serde_json::Value>,
}

/// A chunk of text from a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique chunk identifier
    pub id: ChunkId,
    /// Source document reference
    pub document_id: DocumentId,
    /// Chunk text content
    pub content: String,
    /// Character offset in source document (start)
    pub start_offset: usize,
    /// Character offset in source document (end)
    pub end_offset: usize,
    /// Metadata inherited from document
    pub metadata: ChunkMetadata,
    /// Embedding vector (populated after embedding)
    pub embedding: Option<Vec<f32>>,
}

impl Chunk {
    /// Create a new chunk
    #[must_use]
    pub fn new(
        document_id: DocumentId,
        content: String,
        start_offset: usize,
        end_offset: usize,
    ) -> Self {
        Self {
            id: ChunkId::new(),
            document_id,
            content,
            start_offset,
            end_offset,
            metadata: ChunkMetadata::default(),
            embedding: None,
        }
    }

    /// Get the length of the chunk in characters
    #[must_use]
    pub fn len(&self) -> usize {
        self.content.len()
    }

    /// Check if the chunk is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// Set the embedding vector
    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        // Contract: embedding-algebra-v1.yaml precondition (pv codegen)
        contract_pre_embedding_lookup!(embedding);
        self.embedding = Some(embedding);
    }
}

/// Chunking strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// Fixed-size chunks with overlap
    FixedSize {
        /// Target chunk size in characters
        chunk_size: usize,
        /// Overlap between consecutive chunks
        overlap: usize,
    },
    /// Split on sentence boundaries
    Sentence {
        /// Maximum sentences per chunk
        max_sentences: usize,
        /// Overlap sentences between chunks
        overlap_sentences: usize,
    },
    /// Split on paragraph boundaries
    Paragraph {
        /// Maximum paragraphs per chunk
        max_paragraphs: usize,
    },
    /// Recursive character splitting
    Recursive {
        /// Separators to try in order
        separators: Vec<String>,
        /// Target chunk size
        chunk_size: usize,
        /// Overlap between chunks
        overlap: usize,
    },
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        Self::Recursive {
            separators: vec![
                "\n\n".to_string(),
                "\n".to_string(),
                ". ".to_string(),
                " ".to_string(),
            ],
            chunk_size: 512,
            overlap: 50,
        }
    }
}

/// Trait for document chunkers
pub trait Chunker: Send + Sync {
    /// Split document into chunks
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>>;

    /// Estimate chunk count without materializing
    fn estimate_chunks(&self, document: &Document) -> usize;
}

/// Recursive chunker implementation
#[derive(Debug, Clone)]
pub struct RecursiveChunker {
    separators: Vec<String>,
    chunk_size: usize,
    overlap: usize,
}

impl RecursiveChunker {
    /// Create a new recursive chunker
    #[must_use]
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        Self {
            separators: vec![
                "\n\n".to_string(),
                "\n".to_string(),
                ". ".to_string(),
                " ".to_string(),
            ],
            chunk_size,
            overlap,
        }
    }

    /// Create with custom separators
    #[must_use]
    pub fn with_separators(mut self, separators: Vec<String>) -> Self {
        self.separators = separators;
        self
    }

    fn split_text(&self, text: &str, separator_idx: usize) -> Vec<String> {
        if text.len() <= self.chunk_size {
            return vec![text.to_string()];
        }

        if separator_idx >= self.separators.len() {
            // Fallback: split by characters
            return self.split_by_chars(text);
        }

        let separator = &self.separators[separator_idx];
        let parts: Vec<&str> = text.split(separator).collect();

        if parts.len() == 1 {
            // Separator not found, try next
            return self.split_text(text, separator_idx + 1);
        }

        self.merge_splits(&parts, separator, separator_idx)
    }

    fn merge_splits(&self, parts: &[&str], separator: &str, separator_idx: usize) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut current = String::new();

        for part in parts {
            let potential = if current.is_empty() {
                (*part).to_string()
            } else {
                format!("{current}{separator}{part}")
            };

            if potential.len() <= self.chunk_size {
                current = potential;
            } else if current.is_empty() {
                // Single part too large, recurse
                chunks.extend(self.split_text(part, separator_idx + 1));
            } else {
                chunks.push(current);
                current = (*part).to_string();
            }
        }

        if !current.is_empty() {
            if current.len() <= self.chunk_size {
                chunks.push(current);
            } else {
                chunks.extend(self.split_text(&current, separator_idx + 1));
            }
        }

        chunks
    }

    fn split_by_chars(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < chars.len() {
            let end = (start + self.chunk_size).min(chars.len());
            let chunk: String = chars[start..end].iter().collect();
            chunks.push(chunk);

            if end >= chars.len() {
                break;
            }

            // Move start, accounting for overlap
            start = if end > self.overlap { end - self.overlap } else { end };
        }

        chunks
    }

    fn apply_overlap(&self, chunks: Vec<String>) -> Vec<String> {
        if self.overlap == 0 || chunks.len() <= 1 {
            return chunks;
        }

        let mut result = Vec::with_capacity(chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            if i == 0 {
                result.push(chunk.clone());
            } else {
                // Add overlap from previous chunk
                let prev = &chunks[i - 1];
                let overlap_text = if prev.len() > self.overlap {
                    let start = prev.len() - self.overlap;
                    let start = ceil_char_boundary(prev, start);
                    &prev[start..]
                } else {
                    prev.as_str()
                };
                result.push(format!("{overlap_text}{chunk}"));
            }
        }
        result
    }
}

impl Chunker for RecursiveChunker {
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>> {
        if document.content.is_empty() {
            return Err(Error::EmptyDocument(
                document.title.clone().unwrap_or_else(|| "untitled".to_string()),
            ));
        }

        let text_chunks = self.split_text(&document.content, 0);
        let overlapped = self.apply_overlap(text_chunks);

        let mut offset = 0;
        let mut chunks = Vec::new();

        for content in overlapped {
            // Snap offset to a valid char boundary
            let safe_offset = ceil_char_boundary(&document.content, offset);
            // Find actual position in document
            let start = document.content[safe_offset..]
                .find(&content)
                .map_or(safe_offset, |pos| safe_offset + pos);
            let end = start + content.len();

            let mut chunk = Chunk::new(document.id, content, start, end);
            chunk.metadata.title = document.title.clone();

            chunks.push(chunk);
            offset = ceil_char_boundary(&document.content, start + 1);
        }

        Ok(chunks)
    }

    fn estimate_chunks(&self, document: &Document) -> usize {
        if document.content.is_empty() {
            return 0;
        }
        let effective_size = self.chunk_size.saturating_sub(self.overlap);
        if effective_size == 0 {
            return 1;
        }
        (document.content.len() + effective_size - 1) / effective_size
    }
}

/// Fixed-size chunker implementation
#[derive(Debug, Clone)]
pub struct FixedSizeChunker {
    chunk_size: usize,
    overlap: usize,
}

impl FixedSizeChunker {
    /// Create a new fixed-size chunker
    #[must_use]
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        Self { chunk_size, overlap }
    }
}

impl Chunker for FixedSizeChunker {
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>> {
        if document.content.is_empty() {
            return Err(Error::EmptyDocument(
                document.title.clone().unwrap_or_else(|| "untitled".to_string()),
            ));
        }

        let chars: Vec<char> = document.content.chars().collect();
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < chars.len() {
            let end = (start + self.chunk_size).min(chars.len());
            let content: String = chars[start..end].iter().collect();

            let byte_start = chars[..start].iter().collect::<String>().len();
            let byte_end = byte_start + content.len();

            let mut chunk = Chunk::new(document.id, content, byte_start, byte_end);
            chunk.metadata.title = document.title.clone();
            chunks.push(chunk);

            if end >= chars.len() {
                break;
            }

            let step = self.chunk_size.saturating_sub(self.overlap);
            start += if step == 0 { 1 } else { step };
        }

        Ok(chunks)
    }

    fn estimate_chunks(&self, document: &Document) -> usize {
        if document.content.is_empty() {
            return 0;
        }
        let step = self.chunk_size.saturating_sub(self.overlap);
        if step == 0 {
            return document.content.chars().count();
        }
        let char_count = document.content.chars().count();
        (char_count + step - 1) / step
    }
}

/// Semantic chunker that groups sentences by embedding similarity
pub struct SemanticChunker<E: crate::embed::Embedder> {
    embedder: E,
    /// Similarity threshold (0.0 to 1.0) - chunks split when similarity drops below this
    pub similarity_threshold: f32,
    /// Maximum chunk size in characters
    pub max_chunk_size: usize,
}

impl<E: crate::embed::Embedder> SemanticChunker<E> {
    /// Create a new semantic chunker
    pub fn new(embedder: E, similarity_threshold: f32, max_chunk_size: usize) -> Self {
        Self { embedder, similarity_threshold, max_chunk_size }
    }

    /// Split text into sentences
    fn split_sentences(text: &str) -> Vec<&str> {
        let mut sentences = Vec::new();
        let mut start = 0;

        for (i, c) in text.char_indices() {
            if c == '.' || c == '!' || c == '?' {
                let next_char = text[i + c.len_utf8()..].chars().next();
                if next_char.map_or(true, |nc| nc.is_whitespace()) {
                    let end = i + c.len_utf8();
                    let sentence = text[start..end].trim();
                    if !sentence.is_empty() {
                        sentences.push(sentence);
                    }
                    start = end;
                }
            }
        }

        let remaining = text[start..].trim();
        if !remaining.is_empty() {
            sentences.push(remaining);
        }

        sentences
    }
}

impl<E: crate::embed::Embedder> Chunker for SemanticChunker<E> {
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>> {
        if document.content.is_empty() {
            return Err(Error::EmptyDocument(
                document.title.clone().unwrap_or_else(|| "untitled".to_string()),
            ));
        }

        let sentences = Self::split_sentences(&document.content);
        if sentences.is_empty() {
            return Err(Error::EmptyDocument(
                document.title.clone().unwrap_or_else(|| "untitled".to_string()),
            ));
        }

        if sentences.len() == 1 {
            let content = sentences[0].to_string();
            let start_offset = document.content.find(&content).unwrap_or(0);
            let end_offset = start_offset + content.len();
            let mut chunk = Chunk::new(document.id, content, start_offset, end_offset);
            chunk.metadata.title = document.title.clone();
            return Ok(vec![chunk]);
        }

        // Embed all sentences
        let embeddings: Vec<Vec<f32>> = sentences
            .iter()
            .map(|s| {
                self.embedder.embed(s).unwrap_or_else(|e| {
                    eprintln!("Embedding failed for sentence: {e}");
                    vec![0.0; self.embedder.dimension()]
                })
            })
            .collect();

        let mut chunks = Vec::new();
        let mut current_sentences: Vec<&str> = vec![sentences[0]];
        let mut current_embedding = &embeddings[0];

        for i in 1..sentences.len() {
            let similarity = crate::embed::cosine_similarity(current_embedding, &embeddings[i]);
            let current_len: usize = current_sentences.iter().map(|s| s.len()).sum();

            if similarity < self.similarity_threshold
                || current_len + sentences[i].len() > self.max_chunk_size
            {
                // Create chunk from current sentences
                let content = current_sentences.join(" ");
                let start_offset = document.content.find(&content).unwrap_or(0);
                let end_offset = start_offset + content.len();
                let mut chunk = Chunk::new(document.id, content, start_offset, end_offset);
                chunk.metadata.title = document.title.clone();
                chunks.push(chunk);

                current_sentences = vec![sentences[i]];
                current_embedding = &embeddings[i];
            } else {
                current_sentences.push(sentences[i]);
            }
        }

        // Add remaining sentences
        if !current_sentences.is_empty() {
            let content = current_sentences.join(" ");
            let start_offset = document.content.find(&content).unwrap_or(0);
            let end_offset = start_offset + content.len();
            let mut chunk = Chunk::new(document.id, content, start_offset, end_offset);
            chunk.metadata.title = document.title.clone();
            chunks.push(chunk);
        }

        Ok(chunks)
    }

    fn estimate_chunks(&self, document: &Document) -> usize {
        if document.content.is_empty() {
            return 0;
        }
        // Rough estimate based on max chunk size
        let sentences = Self::split_sentences(&document.content);
        (sentences.len() + 2) / 3 // Assume average 3 sentences per chunk
    }
}

/// Structural chunker that respects document structure (headers, sections)
#[derive(Debug, Clone)]
pub struct StructuralChunker {
    /// Whether to respect headers when chunking
    pub respect_headers: bool,
    /// Maximum section size in characters
    pub max_section_size: usize,
}

impl StructuralChunker {
    /// Create a new structural chunker
    #[must_use]
    pub fn new(respect_headers: bool, max_section_size: usize) -> Self {
        Self { respect_headers, max_section_size }
    }

    /// Extract header text from a line
    fn extract_header(line: &str) -> Option<String> {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            // Markdown header
            let header = trimmed.trim_start_matches('#').trim();
            if !header.is_empty() {
                return Some(header.to_string());
            }
        }
        None
    }

    /// Check if a line is a header
    fn is_header(line: &str) -> bool {
        Self::extract_header(line).is_some()
    }

    /// Split document into sections by headers
    fn split_by_headers(text: &str) -> Vec<(Option<String>, String)> {
        let mut sections = Vec::new();
        let mut current_header: Option<String> = None;
        let mut current_content = String::new();

        for line in text.lines() {
            if Self::is_header(line) {
                // Save previous section if not empty
                if !current_content.trim().is_empty() || current_header.is_some() {
                    sections.push((current_header.take(), current_content.trim().to_string()));
                    current_content = String::new();
                }
                current_header = Self::extract_header(line);
                current_content.push_str(line);
                current_content.push('\n');
            } else {
                current_content.push_str(line);
                current_content.push('\n');
            }
        }

        // Add final section
        if !current_content.trim().is_empty() {
            sections.push((current_header, current_content.trim().to_string()));
        }

        sections
    }
}

impl Chunker for StructuralChunker {
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>> {
        if document.content.is_empty() {
            return Err(Error::EmptyDocument(
                document.title.clone().unwrap_or_else(|| "untitled".to_string()),
            ));
        }

        let sections = if self.respect_headers {
            Self::split_by_headers(&document.content)
        } else {
            vec![(None, document.content.clone())]
        };

        if sections.is_empty() {
            return Err(Error::EmptyDocument(
                document.title.clone().unwrap_or_else(|| "untitled".to_string()),
            ));
        }

        let mut chunks = Vec::new();
        // Hoist clones and constructors outside loop (CB-518, CB-520)
        let doc_title = document.title.clone();
        let doc_source = document.source.clone();
        let doc_metadata = document.metadata.clone();
        let sub_chunker = RecursiveChunker::new(self.max_section_size, 50);

        for (header, content) in sections {
            if content.is_empty() {
                continue;
            }

            // Split large sections if needed
            if content.len() > self.max_section_size {
                let sub_doc = Document {
                    id: document.id,
                    content,
                    title: doc_title.clone(),
                    source: doc_source.clone(),
                    metadata: doc_metadata.clone(),
                };
                if let Ok(sub_chunks) = sub_chunker.chunk(&sub_doc) {
                    for mut chunk in sub_chunks {
                        if let Some(ref h) = header {
                            chunk.metadata.headers.push(h.clone());
                        }
                        chunks.push(chunk);
                    }
                }
            } else {
                let start_offset = document.content.find(&content).unwrap_or(0);
                let end_offset = start_offset + content.len();
                let mut chunk = Chunk::new(document.id, content, start_offset, end_offset);
                chunk.metadata.title = doc_title.clone();
                if let Some(h) = header {
                    chunk.metadata.headers.push(h);
                }
                chunks.push(chunk);
            }
        }

        if chunks.is_empty() {
            // Fallback: return entire document as single chunk
            let content = document.content.clone();
            let mut chunk = Chunk::new(document.id, content, 0, document.content.len());
            chunk.metadata.title = document.title.clone();
            chunks.push(chunk);
        }

        Ok(chunks)
    }

    fn estimate_chunks(&self, document: &Document) -> usize {
        if document.content.is_empty() {
            return 0;
        }
        let sections = Self::split_by_headers(&document.content);
        sections.len().max(1)
    }
}

/// Paragraph-based chunker
#[derive(Debug, Clone)]
pub struct ParagraphChunker {
    max_paragraphs: usize,
}

impl ParagraphChunker {
    /// Create a new paragraph chunker
    #[must_use]
    pub fn new(max_paragraphs: usize) -> Self {
        Self { max_paragraphs }
    }

    /// Split text into paragraphs
    fn split_paragraphs(text: &str) -> Vec<&str> {
        text.split("\n\n").map(|p| p.trim()).filter(|p| !p.is_empty()).collect()
    }
}

impl Chunker for ParagraphChunker {
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>> {
        if document.content.is_empty() {
            return Err(Error::EmptyDocument(
                document.title.clone().unwrap_or_else(|| "untitled".to_string()),
            ));
        }

        let paragraphs = Self::split_paragraphs(&document.content);
        if paragraphs.is_empty() {
            return Err(Error::EmptyDocument(
                document.title.clone().unwrap_or_else(|| "untitled".to_string()),
            ));
        }

        let mut chunks = Vec::new();
        let mut i = 0;

        while i < paragraphs.len() {
            let end = (i + self.max_paragraphs).min(paragraphs.len());
            let content = paragraphs[i..end].join("\n\n");

            let start_offset = document.content.find(&content).unwrap_or(0);
            let end_offset = start_offset + content.len();

            let mut chunk = Chunk::new(document.id, content, start_offset, end_offset);
            chunk.metadata.title = document.title.clone();
            chunks.push(chunk);

            i = end;
        }

        Ok(chunks)
    }

    fn estimate_chunks(&self, document: &Document) -> usize {
        if document.content.is_empty() {
            return 0;
        }
        let paragraphs = Self::split_paragraphs(&document.content);
        if self.max_paragraphs == 0 {
            return paragraphs.len();
        }
        (paragraphs.len() + self.max_paragraphs - 1) / self.max_paragraphs
    }
}

/// Sentence-based chunker
#[derive(Debug, Clone)]
pub struct SentenceChunker {
    max_sentences: usize,
    overlap_sentences: usize,
}

impl SentenceChunker {
    /// Create a new sentence chunker
    #[must_use]
    pub fn new(max_sentences: usize, overlap_sentences: usize) -> Self {
        Self { max_sentences, overlap_sentences }
    }

    fn split_sentences(text: &str) -> Vec<&str> {
        let mut sentences = Vec::new();
        let mut start = 0;

        for (i, c) in text.char_indices() {
            if c == '.' || c == '!' || c == '?' {
                // Check for end of sentence
                let next_char = text[i + c.len_utf8()..].chars().next();
                if next_char.map_or(true, |nc| nc.is_whitespace() || nc.is_uppercase()) {
                    let end = i + c.len_utf8();
                    let sentence = text[start..end].trim();
                    if !sentence.is_empty() {
                        sentences.push(sentence);
                    }
                    start = end;
                }
            }
        }

        // Add remaining text
        let remaining = text[start..].trim();
        if !remaining.is_empty() {
            sentences.push(remaining);
        }

        sentences
    }
}

impl Chunker for SentenceChunker {
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>> {
        if document.content.is_empty() {
            return Err(Error::EmptyDocument(
                document.title.clone().unwrap_or_else(|| "untitled".to_string()),
            ));
        }

        let sentences = Self::split_sentences(&document.content);
        let mut chunks = Vec::new();
        let mut i = 0;

        while i < sentences.len() {
            let end = (i + self.max_sentences).min(sentences.len());
            let content = sentences[i..end].join(" ");

            let start_offset = document.content.find(&content).unwrap_or(0);
            let end_offset = start_offset + content.len();

            let mut chunk = Chunk::new(document.id, content, start_offset, end_offset);
            chunk.metadata.title = document.title.clone();
            chunks.push(chunk);

            let step = self.max_sentences.saturating_sub(self.overlap_sentences);
            i += if step == 0 { 1 } else { step };
        }

        Ok(chunks)
    }

    fn estimate_chunks(&self, document: &Document) -> usize {
        if document.content.is_empty() {
            return 0;
        }
        let sentences = Self::split_sentences(&document.content);
        let step = self.max_sentences.saturating_sub(self.overlap_sentences);
        if step == 0 {
            return sentences.len();
        }
        (sentences.len() + step - 1) / step
    }
}

#[cfg(test)]
mod tests;
