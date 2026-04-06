//! Semantic Chunker for Code-Aware Text Splitting
//!
//! Implements recursive character splitting with code-aware boundaries.
//! Based on LangChain text splitter patterns [1] and Chen et al. (2017) [21].

use super::fingerprint::ChunkerConfig;

/// Semantic chunker with code-aware splitting
///
/// Uses recursive character splitting with Rust/Markdown-aware separators.
/// Preserves context through configurable overlap.
#[derive(Debug, Clone)]
pub struct SemanticChunker {
    /// Target chunk size in characters (approximates tokens)
    chunk_size: usize,
    /// Overlap between chunks for context preservation
    chunk_overlap: usize,
    /// Code-aware separators ordered by priority (highest to lowest)
    separators: Vec<String>,
}

impl SemanticChunker {
    /// Create a new semantic chunker with custom settings
    pub fn new(chunk_size: usize, chunk_overlap: usize, separators: Vec<String>) -> Self {
        Self { chunk_size, chunk_overlap, separators }
    }

    /// Create from a ChunkerConfig
    pub fn from_config(config: &ChunkerConfig) -> Self {
        Self {
            chunk_size: config.chunk_size,
            chunk_overlap: config.chunk_overlap,
            separators: Self::default_separators(),
        }
    }

    /// Default separators for Rust/Markdown content
    fn default_separators() -> Vec<String> {
        vec![
            "\n## ".to_string(),     // Markdown H2
            "\n### ".to_string(),    // Markdown H3
            "\n#### ".to_string(),   // Markdown H4
            "\nfn ".to_string(),     // Rust function
            "\npub fn ".to_string(), // Rust public function
            "\nimpl ".to_string(),   // Rust impl block
            "\nstruct ".to_string(), // Rust struct
            "\nenum ".to_string(),   // Rust enum
            "\nmod ".to_string(),    // Rust module
            "\n```".to_string(),     // Code fence
            "\n\n".to_string(),      // Paragraph
            "\n".to_string(),        // Line
            " ".to_string(),         // Word
        ]
    }

    /// Split text into chunks
    pub fn split(&self, text: &str) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let mut current_pos = 0;

        while current_pos < text.len() {
            let (chunk_text, end_pos) = self.extract_chunk(text, current_pos);

            if !chunk_text.trim().is_empty() {
                let start_line = text[..current_pos].matches('\n').count() + 1;
                let end_line = start_line + chunk_text.matches('\n').count();

                chunks.push(Chunk {
                    content: chunk_text.to_string(),
                    start_offset: current_pos,
                    end_offset: end_pos,
                    start_line,
                    end_line,
                });
            }

            // Move forward, accounting for overlap
            let advance = if end_pos - current_pos > self.chunk_overlap {
                end_pos - current_pos - self.chunk_overlap
            } else {
                end_pos - current_pos
            };

            // Advance at least 1, ensuring we land on a char boundary
            let new_pos = current_pos + advance.max(1);
            current_pos = Self::find_next_char_boundary(text, new_pos);
        }

        chunks
    }

    /// Extract a single chunk starting at position
    fn extract_chunk(&self, text: &str, start: usize) -> (String, usize) {
        let remaining = &text[start..];
        let target_end = Self::find_char_boundary(text, (start + self.chunk_size).min(text.len()));

        // If remaining text fits in one chunk, return it all
        if start + remaining.len() <= target_end {
            return (remaining.to_string(), text.len());
        }

        // Find the best split point using separators
        let search_region = &text[start..target_end];

        for separator in &self.separators {
            if let Some(pos) = search_region.rfind(separator.as_str()) {
                if pos > 0 {
                    // Include the separator in the chunk
                    let end = start + pos + separator.len();
                    return (text[start..end].to_string(), end);
                }
            }
        }

        // No separator found, hard cut at nearest char boundary
        (text[start..target_end].to_string(), target_end)
    }

    /// Find the nearest valid UTF-8 character boundary at or before the given position
    fn find_char_boundary(text: &str, pos: usize) -> usize {
        if pos >= text.len() {
            return text.len();
        }
        // Walk backwards to find a char boundary
        let mut boundary = pos;
        while boundary > 0 && !text.is_char_boundary(boundary) {
            boundary -= 1;
        }
        boundary
    }

    /// Find the next valid UTF-8 character boundary at or after the given position
    fn find_next_char_boundary(text: &str, pos: usize) -> usize {
        if pos >= text.len() {
            return text.len();
        }
        // Walk forwards to find a char boundary
        let mut boundary = pos;
        while boundary < text.len() && !text.is_char_boundary(boundary) {
            boundary += 1;
        }
        boundary
    }

    /// Get configuration as ChunkerConfig
    pub fn config(&self) -> ChunkerConfig {
        let sep_refs: Vec<&str> = self.separators.iter().map(|s| s.as_str()).collect();
        ChunkerConfig::new(self.chunk_size, self.chunk_overlap, &sep_refs)
    }
}

impl Default for SemanticChunker {
    fn default() -> Self {
        Self { chunk_size: 512, chunk_overlap: 64, separators: Self::default_separators() }
    }
}

/// A text chunk with position metadata
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// Chunk content
    pub content: String,
    /// Start byte offset in source
    pub start_offset: usize,
    /// End byte offset in source
    pub end_offset: usize,
    /// Start line number (1-indexed)
    pub start_line: usize,
    /// End line number (1-indexed)
    pub end_line: usize,
}

impl Chunk {
    /// Get content hash for deduplication
    pub fn content_hash(&self) -> [u8; 32] {
        // Use same hash function as fingerprint
        let mut hash = [0u8; 32];
        let mut state: u64 = 0xcbf2_9ce4_8422_2325;
        for &byte in self.content.as_bytes() {
            state ^= byte as u64;
            state = state.wrapping_mul(0x0100_0000_01b3);
        }
        for i in 0..4 {
            let chunk = state.wrapping_add(i as u64).to_le_bytes();
            hash[i * 8..(i + 1) * 8].copy_from_slice(&chunk);
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunker_default() {
        let chunker = SemanticChunker::default();
        assert_eq!(chunker.chunk_size, 512);
        assert_eq!(chunker.chunk_overlap, 64);
        assert!(!chunker.separators.is_empty());
    }

    #[test]
    fn test_split_short_text() {
        let chunker = SemanticChunker::default();
        let text = "Short text";
        let chunks = chunker.split(text);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "Short text");
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 1);
    }

    #[test]
    fn test_split_markdown_headers() {
        let chunker = SemanticChunker::new(100, 10, vec!["\n## ".to_string()]);
        let text = "# Title\n\nIntro paragraph.\n\n## Section 1\n\nContent 1.\n\n## Section 2\n\nContent 2.";

        let chunks = chunker.split(text);

        // Should split at ## headers
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_split_rust_code() {
        let chunker = SemanticChunker::new(200, 20, vec!["\nfn ".to_string(), "\n\n".to_string()]);
        let text = r#"
fn foo() {
    println!("foo");
}

fn bar() {
    println!("bar");
}

fn baz() {
    println!("baz");
}
"#;

        let chunks = chunker.split(text);

        // Should preserve function boundaries
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            // Each chunk should be valid
            assert!(!chunk.content.is_empty());
        }
    }

    #[test]
    fn test_chunk_overlap() {
        let chunker = SemanticChunker::new(50, 10, vec![" ".to_string()]);
        let text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12";

        let chunks = chunker.split(text);

        // With overlap, subsequent chunks should share some content
        if chunks.len() > 1 {
            // Check that chunks have reasonable sizes
            for chunk in &chunks {
                assert!(chunk.content.len() <= chunker.chunk_size + 20); // Some tolerance
            }
        }
    }

    #[test]
    fn test_chunk_line_tracking() {
        let chunker = SemanticChunker::new(50, 5, vec!["\n".to_string()]);
        let text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5";

        let chunks = chunker.split(text);

        // First chunk should start at line 1
        assert_eq!(chunks[0].start_line, 1);
    }

    #[test]
    fn test_chunk_content_hash_deterministic() {
        let chunk1 = Chunk {
            content: "test content".to_string(),
            start_offset: 0,
            end_offset: 12,
            start_line: 1,
            end_line: 1,
        };
        let chunk2 = Chunk {
            content: "test content".to_string(),
            start_offset: 100, // Different offset, same content
            end_offset: 112,
            start_line: 5,
            end_line: 5,
        };

        assert_eq!(chunk1.content_hash(), chunk2.content_hash());
    }

    #[test]
    fn test_chunk_content_hash_different() {
        let chunk1 = Chunk {
            content: "content 1".to_string(),
            start_offset: 0,
            end_offset: 9,
            start_line: 1,
            end_line: 1,
        };
        let chunk2 = Chunk {
            content: "content 2".to_string(),
            start_offset: 0,
            end_offset: 9,
            start_line: 1,
            end_line: 1,
        };

        assert_ne!(chunk1.content_hash(), chunk2.content_hash());
    }

    #[test]
    fn test_empty_text() {
        let chunker = SemanticChunker::default();
        let chunks = chunker.split("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_whitespace_only() {
        let chunker = SemanticChunker::default();
        let chunks = chunker.split("   \n\n   \t   ");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_config_round_trip() {
        let chunker = SemanticChunker::new(256, 32, vec!["\n".to_string()]);
        let config = chunker.config();

        assert_eq!(config.chunk_size, 256);
        assert_eq!(config.chunk_overlap, 32);
    }

    #[test]
    fn test_large_document_chunking() {
        let chunker = SemanticChunker::new(100, 20, SemanticChunker::default_separators());

        // Create a large document
        let mut text = String::new();
        for i in 0..50 {
            text.push_str(&format!(
                "\n## Section {}\n\nThis is paragraph {} with some content.\n",
                i, i
            ));
        }

        let chunks = chunker.split(&text);

        // Should produce multiple chunks
        assert!(chunks.len() > 1);

        // All chunks should be non-empty
        for chunk in &chunks {
            assert!(!chunk.content.trim().is_empty());
        }
    }

    // Property-based tests for semantic chunker
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        // Strategy for generating reasonable chunk sizes
        fn chunk_size_strategy() -> impl Strategy<Value = usize> {
            32usize..=1024
        }

        // Strategy for generating reasonable overlap (must be less than chunk size)
        fn overlap_strategy(chunk_size: usize) -> impl Strategy<Value = usize> {
            0..=(chunk_size / 2)
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]

            /// Property: Chunking any text produces chunks whose content equals the original
            #[test]
            fn prop_chunking_preserves_content(text in ".{0,500}") {
                let chunker = SemanticChunker::default();
                let chunks = chunker.split(&text);

                // If there are chunks, they should contain content from the original
                for chunk in &chunks {
                    // Each chunk content should be found in original text (accounting for overlap)
                    prop_assert!(text.contains(chunk.content.trim()) || chunk.content.trim().is_empty());
                }
            }

            /// Property: All chunks have valid line numbers (start <= end)
            #[test]
            fn prop_chunk_lines_valid(text in ".{1,500}\n.{1,500}\n.{1,500}") {
                let chunker = SemanticChunker::default();
                let chunks = chunker.split(&text);

                for chunk in &chunks {
                    prop_assert!(chunk.start_line <= chunk.end_line,
                        "start_line {} > end_line {}", chunk.start_line, chunk.end_line);
                    prop_assert!(chunk.start_line >= 1,
                        "start_line {} should be >= 1", chunk.start_line);
                }
            }

            /// Property: Chunk offsets are valid (start <= end, within bounds)
            #[test]
            fn prop_chunk_offsets_valid(text in ".{10,500}") {
                let chunker = SemanticChunker::default();
                let chunks = chunker.split(&text);

                for chunk in &chunks {
                    prop_assert!(chunk.start_offset <= chunk.end_offset,
                        "start_offset {} > end_offset {}", chunk.start_offset, chunk.end_offset);
                    prop_assert!(chunk.end_offset <= text.len(),
                        "end_offset {} > text.len() {}", chunk.end_offset, text.len());
                }
            }

            /// Property: Content hashes are deterministic
            #[test]
            fn prop_content_hash_deterministic(content in ".{1,100}") {
                let chunk = Chunk {
                    content: content.clone(),
                    start_offset: 0,
                    end_offset: content.len(),
                    start_line: 1,
                    end_line: 1,
                };

                let hash1 = chunk.content_hash();
                let hash2 = chunk.content_hash();
                prop_assert_eq!(hash1, hash2);
            }

            /// Property: Different content produces different hashes (with high probability)
            #[test]
            fn prop_content_hash_different(
                content1 in "[a-z]{5,20}",
                content2 in "[A-Z]{5,20}"
            ) {
                // Ensure contents are different
                if content1 != content2 {
                    let chunk1 = Chunk {
                        content: content1.clone(),
                        start_offset: 0,
                        end_offset: content1.len(),
                        start_line: 1,
                        end_line: 1,
                    };
                    let chunk2 = Chunk {
                        content: content2.clone(),
                        start_offset: 0,
                        end_offset: content2.len(),
                        start_line: 1,
                        end_line: 1,
                    };

                    prop_assert_ne!(chunk1.content_hash(), chunk2.content_hash());
                }
            }

            /// Property: Custom chunk sizes are respected (approximately)
            #[test]
            fn prop_chunk_size_respected(
                chunk_size in chunk_size_strategy(),
                text_len in 100usize..2000
            ) {
                let overlap = chunk_size / 4;
                let chunker = SemanticChunker::new(chunk_size, overlap, vec![" ".to_string()]);

                // Generate text of specified length
                let text: String = (0..text_len).map(|i| if i % 10 == 0 { ' ' } else { 'a' }).collect();
                let chunks = chunker.split(&text);

                // Most chunks should be close to chunk_size (with some tolerance)
                for chunk in &chunks {
                    // Allow 2x tolerance for edge cases
                    prop_assert!(chunk.content.len() <= chunk_size * 2,
                        "chunk len {} > 2 * chunk_size {}", chunk.content.len(), chunk_size * 2);
                }
            }
        }
    }
}
