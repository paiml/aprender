//! Shared types for Coursera asset generation

use serde::{Deserialize, Serialize};

/// Parsed transcript input (from whisper-apr JSON or plain text)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptInput {
    /// Full transcript text
    pub text: String,
    /// Language code (e.g., "en")
    pub language: String,
    /// Time-aligned segments (empty for plain text)
    pub segments: Vec<TranscriptSegment>,
    /// Source file path
    pub source_path: String,
}

/// A time-aligned transcript segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptSegment {
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
    /// Segment text
    pub text: String,
}

/// An extracted concept with definition and context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    /// The technical term
    pub term: String,
    /// Brief definition
    pub definition: String,
    /// Context sentence from transcript
    pub context: String,
    /// Category classification
    pub category: ConceptCategory,
}

/// Category for technical concepts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConceptCategory {
    Algorithm,
    DataStructure,
    Tool,
    Pattern,
    Metric,
    General,
}

impl ConceptCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Algorithm => "Algorithm",
            Self::DataStructure => "Data Structure",
            Self::Tool => "Tool",
            Self::Pattern => "Pattern",
            Self::Metric => "Metric",
            Self::General => "General",
        }
    }
}

impl std::fmt::Display for ConceptCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A curated arXiv citation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArxivCitation {
    /// arXiv identifier (e.g., "1706.03762")
    pub arxiv_id: String,
    /// Paper title
    pub title: String,
    /// Author list (abbreviated)
    pub authors: String,
    /// Publication year
    pub year: u16,
    /// Canonical URL
    pub url: String,
    /// Brief abstract snippet
    pub abstract_snippet: String,
    /// Topic keywords for matching
    pub topics: Vec<String>,
}

/// A reflection question with Bloom's taxonomy level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionQuestion {
    /// The question text
    pub question: String,
    /// Bloom's taxonomy thinking level
    pub thinking_level: BloomLevel,
}

/// Bloom's taxonomy cognitive levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BloomLevel {
    Analysis,
    Synthesis,
    Evaluation,
    Application,
    Creation,
}

impl BloomLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Analysis => "Analysis",
            Self::Synthesis => "Synthesis",
            Self::Evaluation => "Evaluation",
            Self::Application => "Application",
            Self::Creation => "Creation",
        }
    }
}

impl std::fmt::Display for BloomLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A vocabulary entry extracted from transcripts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabularyEntry {
    /// The term
    pub term: String,
    /// Definition derived from context
    pub definition: String,
    /// Timestamp or position of first occurrence
    pub first_occurrence: String,
    /// Frequency count across transcripts
    pub frequency: usize,
    /// Category classification
    pub category: ConceptCategory,
}

/// Banner generation configuration
#[derive(Debug, Clone)]
pub struct BannerConfig {
    /// Course title displayed on banner
    pub course_title: String,
    /// Concept terms to display as bubbles
    pub concepts: Vec<String>,
    /// Output width in pixels
    pub width: u32,
    /// Output height in pixels
    pub height: u32,
}

impl Default for BannerConfig {
    fn default() -> Self {
        Self { course_title: String::new(), concepts: Vec::new(), width: 1200, height: 400 }
    }
}

/// Reflection reading output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionReading {
    /// Extracted themes from transcript
    pub themes: Vec<String>,
    /// Bloom's taxonomy questions
    pub questions: Vec<ReflectionQuestion>,
    /// Matched arXiv citations
    pub citations: Vec<ArxivCitation>,
}

/// Key concepts reading output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyConceptsReading {
    /// Extracted concepts with definitions
    pub concepts: Vec<Concept>,
    /// Code examples mentioned in transcript
    pub code_examples: Vec<CodeExample>,
}

/// A code example extracted from transcript context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExample {
    /// Language (bash, rust, python, typescript)
    pub language: String,
    /// Code snippet
    pub code: String,
    /// Related concept term
    pub related_concept: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concept_category_display() {
        assert_eq!(ConceptCategory::Algorithm.as_str(), "Algorithm");
        assert_eq!(ConceptCategory::DataStructure.as_str(), "Data Structure");
        assert_eq!(ConceptCategory::Tool.as_str(), "Tool");
        assert_eq!(ConceptCategory::Pattern.as_str(), "Pattern");
        assert_eq!(ConceptCategory::Metric.as_str(), "Metric");
        assert_eq!(ConceptCategory::General.as_str(), "General");
    }

    #[test]
    fn test_bloom_level_display() {
        assert_eq!(BloomLevel::Analysis.as_str(), "Analysis");
        assert_eq!(BloomLevel::Synthesis.as_str(), "Synthesis");
        assert_eq!(BloomLevel::Evaluation.as_str(), "Evaluation");
        assert_eq!(BloomLevel::Application.as_str(), "Application");
        assert_eq!(BloomLevel::Creation.as_str(), "Creation");
    }

    #[test]
    fn test_banner_config_default() {
        let config = BannerConfig::default();
        assert_eq!(config.width, 1200);
        assert_eq!(config.height, 400);
        assert!(config.course_title.is_empty());
        assert!(config.concepts.is_empty());
    }

    #[test]
    fn test_concept_category_format() {
        assert_eq!(format!("{}", ConceptCategory::Algorithm), "Algorithm");
    }

    #[test]
    fn test_bloom_level_format() {
        assert_eq!(format!("{}", BloomLevel::Analysis), "Analysis");
    }

    #[test]
    fn test_transcript_input_serialization() {
        let input = TranscriptInput {
            text: "Hello world".to_string(),
            language: "en".to_string(),
            segments: vec![TranscriptSegment {
                start: 0.0,
                end: 1.5,
                text: "Hello world".to_string(),
            }],
            source_path: "test.json".to_string(),
        };
        let json = serde_json::to_string(&input).expect("json serialize failed");
        assert!(json.contains("Hello world"));
        let parsed: TranscriptInput = serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(parsed.text, "Hello world");
        assert_eq!(parsed.segments.len(), 1);
    }

    #[test]
    fn test_vocabulary_entry_serialization() {
        let entry = VocabularyEntry {
            term: "MLOps".to_string(),
            definition: "Machine Learning Operations".to_string(),
            first_occurrence: "0:05".to_string(),
            frequency: 3,
            category: ConceptCategory::Pattern,
        };
        let json = serde_json::to_string(&entry).expect("json serialize failed");
        let parsed: VocabularyEntry = serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(parsed.term, "MLOps");
        assert_eq!(parsed.frequency, 3);
    }
}
