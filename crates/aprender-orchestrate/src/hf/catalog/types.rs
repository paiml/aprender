//! Core types for the HuggingFace Ecosystem Catalog
//!
//! Implements HF-QUERY-001 core type definitions.

use serde::{Deserialize, Serialize};

// ============================================================================
// HF-QUERY-001: Core Types
// ============================================================================

/// Category of HuggingFace component
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HfComponentCategory {
    /// Hub and client libraries
    Hub,
    /// Deployment and inference
    Deployment,
    /// Core ML libraries
    Library,
    /// Training and optimization
    Training,
    /// Collaboration and extras
    Collaboration,
    /// Community resources
    Community,
}

impl HfComponentCategory {
    /// Get all categories
    pub fn all() -> &'static [Self] {
        &[
            Self::Hub,
            Self::Deployment,
            Self::Library,
            Self::Training,
            Self::Collaboration,
            Self::Community,
        ]
    }

    /// Get display name
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Hub => "Hub & Client Libraries",
            Self::Deployment => "Deployment & Inference",
            Self::Library => "Core ML Libraries",
            Self::Training => "Training & Optimization",
            Self::Collaboration => "Collaboration & Extras",
            Self::Community => "Community Resources",
        }
    }
}

impl std::fmt::Display for HfComponentCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Asset type for course planning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetType {
    Video,
    Lab,
    Reading,
    Quiz,
    Discussion,
}

/// Course alignment metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CourseAlignment {
    /// Course number (1-5)
    pub course: u8,
    /// Week number (1-3)
    pub week: u8,
    /// Lesson identifiers (e.g., "1.1", "2.3")
    pub lessons: Vec<String>,
    /// Asset types used
    pub asset_types: Vec<AssetType>,
}

impl CourseAlignment {
    pub fn new(course: u8, week: u8) -> Self {
        Self { course, week, lessons: Vec::new(), asset_types: Vec::new() }
    }

    pub fn with_lessons(mut self, lessons: &[&str]) -> Self {
        self.lessons = lessons.iter().map(|s| (*s).to_string()).collect();
        self
    }

    pub fn with_assets(mut self, assets: &[AssetType]) -> Self {
        self.asset_types = assets.to_vec();
        self
    }
}

/// A component in the HuggingFace ecosystem (full metadata)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogComponent {
    /// Unique identifier (e.g., "transformers", "peft")
    pub id: String,
    /// Display name
    pub name: String,
    /// Category
    pub category: HfComponentCategory,
    /// Short description
    pub description: String,
    /// Documentation URL
    pub docs_url: String,
    /// GitHub repository URL
    pub repo_url: Option<String>,
    /// PyPI package name
    pub pypi_name: Option<String>,
    /// npm package name
    pub npm_name: Option<String>,
    /// Dependencies on other components
    pub dependencies: Vec<String>,
    /// Course alignments
    pub courses: Vec<CourseAlignment>,
    /// Related components
    pub related: Vec<String>,
    /// Tags for search
    pub tags: Vec<String>,
}

impl CatalogComponent {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        category: HfComponentCategory,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            category,
            description: String::new(),
            docs_url: String::new(),
            repo_url: None,
            pypi_name: None,
            npm_name: None,
            dependencies: Vec::new(),
            courses: Vec::new(),
            related: Vec::new(),
            tags: Vec::new(),
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn with_docs(mut self, url: impl Into<String>) -> Self {
        self.docs_url = url.into();
        self
    }

    pub fn with_repo(mut self, url: impl Into<String>) -> Self {
        self.repo_url = Some(url.into());
        self
    }

    pub fn with_pypi(mut self, name: impl Into<String>) -> Self {
        self.pypi_name = Some(name.into());
        self
    }

    pub fn with_npm(mut self, name: impl Into<String>) -> Self {
        self.npm_name = Some(name.into());
        self
    }

    pub fn with_deps(mut self, deps: &[&str]) -> Self {
        self.dependencies = deps.iter().map(|s| (*s).to_string()).collect();
        self
    }

    pub fn with_course(mut self, alignment: CourseAlignment) -> Self {
        self.courses.push(alignment);
        self
    }

    pub fn with_related(mut self, related: &[&str]) -> Self {
        self.related = related.iter().map(|s| (*s).to_string()).collect();
        self
    }

    pub fn with_tags(mut self, tags: &[&str]) -> Self {
        self.tags = tags.iter().map(|s| (*s).to_string()).collect();
        self
    }
}
