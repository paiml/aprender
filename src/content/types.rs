//! Content type definitions
//!
//! Content type taxonomy from spec section 2.1.

use super::ContentError;
use serde::{Deserialize, Serialize};
use std::ops::Range;
use std::str::FromStr;

/// Content type taxonomy from spec section 2.1
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContentType {
    /// High-Level Outline (HLO) - Course/book structure planning
    HighLevelOutline,
    /// Detailed Outline (DLO) - Section-level content planning
    DetailedOutline,
    /// Book Chapter (BCH) - Technical documentation (mdBook)
    BookChapter,
    /// Blog Post (BLP) - Technical articles
    BlogPost,
    /// Presentar Demo (PDM) - Interactive WASM demos
    PresentarDemo,
}

/// Course level configuration for detailed outlines
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum CourseLevel {
    /// Short course: 1 week, 2 modules, 3 videos each
    Short,
    /// Standard course: 3 weeks, 3 modules, 5 videos each (default)
    #[default]
    Standard,
    /// Extended course: 6 weeks, 6 modules, 5 videos each
    Extended,
    /// Custom configuration
    Custom { weeks: u8, modules: u8, videos_per_module: u8 },
}

impl CourseLevel {
    /// Get duration in weeks
    pub fn weeks(&self) -> u8 {
        match self {
            CourseLevel::Short => 1,
            CourseLevel::Standard => 3,
            CourseLevel::Extended => 6,
            CourseLevel::Custom { weeks, .. } => *weeks,
        }
    }

    /// Get number of modules
    pub fn modules(&self) -> u8 {
        match self {
            CourseLevel::Short => 2,
            CourseLevel::Standard => 3,
            CourseLevel::Extended => 6,
            CourseLevel::Custom { modules, .. } => *modules,
        }
    }

    /// Get videos per module
    pub fn videos_per_module(&self) -> u8 {
        match self {
            CourseLevel::Short => 3,
            CourseLevel::Standard => 5,
            CourseLevel::Extended => 5,
            CourseLevel::Custom { videos_per_module, .. } => *videos_per_module,
        }
    }
}

impl FromStr for CourseLevel {
    type Err = ContentError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "short" | "s" | "1" => Ok(CourseLevel::Short),
            "standard" | "std" | "3" => Ok(CourseLevel::Standard),
            "extended" | "ext" | "6" => Ok(CourseLevel::Extended),
            _ => Err(ContentError::InvalidContentType(format!(
                "Invalid course level: {}. Use: short, standard, extended",
                s
            ))),
        }
    }
}

impl ContentType {
    /// Get all metadata for this content type as a single lookup
    fn metadata(&self) -> (&'static str, &'static str, &'static str, Range<usize>) {
        match self {
            Self::HighLevelOutline => ("HLO", "High-Level Outline", "YAML/Markdown", 50..200),
            Self::DetailedOutline => ("DLO", "Detailed Outline", "YAML/Markdown", 200..1000),
            Self::BookChapter => ("BCH", "Book Chapter", "Markdown (mdBook)", 2000..8000),
            Self::BlogPost => ("BLP", "Blog Post", "Markdown + TOML", 500..3000),
            Self::PresentarDemo => ("PDM", "Presentar Demo", "HTML + YAML", 0..0),
        }
    }

    /// Get the short code for this content type
    pub fn code(&self) -> &'static str {
        self.metadata().0
    }

    /// Get the display name
    pub fn name(&self) -> &'static str {
        self.metadata().1
    }

    /// Get the output format
    pub fn output_format(&self) -> &'static str {
        self.metadata().2
    }

    /// Get the target length range (in words for text, lines for outlines)
    pub fn target_length(&self) -> Range<usize> {
        self.metadata().3
    }

    /// Get all content types
    pub fn all() -> Vec<ContentType> {
        vec![
            ContentType::HighLevelOutline,
            ContentType::DetailedOutline,
            ContentType::BookChapter,
            ContentType::BlogPost,
            ContentType::PresentarDemo,
        ]
    }
}

impl FromStr for ContentType {
    type Err = ContentError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "hlo" | "high-level-outline" | "outline" => Ok(ContentType::HighLevelOutline),
            "dlo" | "detailed-outline" | "detailed" => Ok(ContentType::DetailedOutline),
            "bch" | "book-chapter" | "chapter" => Ok(ContentType::BookChapter),
            "blp" | "blog-post" | "blog" => Ok(ContentType::BlogPost),
            "pdm" | "presentar-demo" | "demo" => Ok(ContentType::PresentarDemo),
            _ => Err(ContentError::InvalidContentType(s.to_string())),
        }
    }
}
