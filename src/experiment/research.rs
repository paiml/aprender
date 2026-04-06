//! Academic research support with ORCID, CRediT, and citation generation.
//!
//! This module provides types for academic metadata, contributor roles,
//! and citation generation in BibTeX and CITATION.cff formats.

use super::ExperimentError;
use serde::{Deserialize, Serialize};

/// ORCID identifier (Open Researcher and Contributor ID)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Orcid(String);

impl Orcid {
    /// Create and validate an ORCID
    pub fn new(orcid: impl Into<String>) -> Result<Self, ExperimentError> {
        let orcid = orcid.into();
        // ORCID format: 0000-0000-0000-000X where X can be 0-9 or X
        // Replaced regex-lite with string validation (DEP-REDUCE)
        if Self::is_valid_orcid(&orcid) {
            Ok(Self(orcid))
        } else {
            Err(ExperimentError::InvalidOrcid(orcid))
        }
    }

    /// Validate ORCID format without regex
    fn is_valid_orcid(orcid: &str) -> bool {
        let parts: Vec<&str> = orcid.split('-').collect();
        if parts.len() != 4 {
            return false;
        }
        // First 3 parts: exactly 4 digits
        for part in parts.iter().take(3) {
            if part.len() != 4 || !part.chars().all(|c| c.is_ascii_digit()) {
                return false;
            }
        }
        // Last part: 3 digits + (digit or X)
        let last = parts[3];
        if last.len() != 4 {
            return false;
        }
        let chars: Vec<char> = last.chars().collect();
        chars.iter().take(3).all(|c| c.is_ascii_digit())
            && (chars[3].is_ascii_digit() || chars[3] == 'X')
    }

    /// Get the ORCID string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// CRediT (Contributor Roles Taxonomy) roles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CreditRole {
    Conceptualization,
    DataCuration,
    FormalAnalysis,
    FundingAcquisition,
    Investigation,
    Methodology,
    ProjectAdministration,
    Resources,
    Software,
    Supervision,
    Validation,
    Visualization,
    WritingOriginalDraft,
    WritingReviewEditing,
}

/// Research contributor with ORCID and CRediT roles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchContributor {
    /// Full name
    pub name: String,
    /// ORCID identifier
    pub orcid: Option<Orcid>,
    /// Affiliation
    pub affiliation: String,
    /// CRediT roles
    pub roles: Vec<CreditRole>,
    /// Email (optional)
    pub email: Option<String>,
}

/// Research artifact with full academic metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchArtifact {
    /// Title
    pub title: String,
    /// Abstract
    pub abstract_text: String,
    /// Contributors with roles
    pub contributors: Vec<ResearchContributor>,
    /// Keywords
    pub keywords: Vec<String>,
    /// DOI if published
    pub doi: Option<String>,
    /// ArXiv ID if applicable
    pub arxiv_id: Option<String>,
    /// License
    pub license: String,
    /// Creation date
    pub created_at: String,
    /// Associated datasets
    pub datasets: Vec<String>,
    /// Associated code repositories
    pub code_repositories: Vec<String>,
    /// Pre-registration info
    pub pre_registration: Option<PreRegistration>,
}

/// Pre-registration for reproducible research
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreRegistration {
    /// Registration timestamp
    pub timestamp: String,
    /// Ed25519 signature of the registration
    pub signature: String,
    /// Public key used for signing
    pub public_key: String,
    /// Hash of the pre-registered hypotheses
    pub hypotheses_hash: String,
    /// Registry where registered (e.g., OSF, AsPredicted)
    pub registry: String,
    /// Registration ID
    pub registration_id: String,
}

/// Citation metadata for BibTeX/CFF generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationMetadata {
    /// Citation type
    pub citation_type: CitationType,
    /// Title
    pub title: String,
    /// Authors
    pub authors: Vec<String>,
    /// Year
    pub year: u16,
    /// Month (optional)
    pub month: Option<u8>,
    /// DOI
    pub doi: Option<String>,
    /// URL
    pub url: Option<String>,
    /// Journal/Conference name
    pub venue: Option<String>,
    /// Volume
    pub volume: Option<String>,
    /// Pages
    pub pages: Option<String>,
    /// Publisher
    pub publisher: Option<String>,
    /// Version (for software)
    pub version: Option<String>,
}

/// Citation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CitationType {
    Article,
    InProceedings,
    Book,
    Software,
    Dataset,
    Misc,
}

impl CitationMetadata {
    /// Generate BibTeX entry
    pub fn to_bibtex(&self, key: &str) -> String {
        let type_str = match self.citation_type {
            CitationType::Article => "article",
            CitationType::InProceedings => "inproceedings",
            CitationType::Book => "book",
            CitationType::Software => "software",
            CitationType::Dataset => "dataset",
            CitationType::Misc => "misc",
        };

        let mut bibtex = format!("@{}{{{},\n", type_str, key);
        bibtex.push_str(&format!("  title = {{{}}},\n", self.title));
        bibtex.push_str(&format!("  author = {{{}}},\n", self.authors.join(" and ")));
        bibtex.push_str(&format!("  year = {{{}}},\n", self.year));

        if let Some(month) = self.month {
            bibtex.push_str(&format!("  month = {{{}}},\n", month));
        }
        if let Some(ref doi) = self.doi {
            bibtex.push_str(&format!("  doi = {{{}}},\n", doi));
        }
        if let Some(ref url) = self.url {
            bibtex.push_str(&format!("  url = {{{}}},\n", url));
        }
        if let Some(ref venue) = self.venue {
            let field = match self.citation_type {
                CitationType::Article => "journal",
                CitationType::InProceedings => "booktitle",
                _ => "howpublished",
            };
            bibtex.push_str(&format!("  {} = {{{}}},\n", field, venue));
        }
        if let Some(ref volume) = self.volume {
            bibtex.push_str(&format!("  volume = {{{}}},\n", volume));
        }
        if let Some(ref pages) = self.pages {
            bibtex.push_str(&format!("  pages = {{{}}},\n", pages));
        }
        if let Some(ref publisher) = self.publisher {
            bibtex.push_str(&format!("  publisher = {{{}}},\n", publisher));
        }
        if let Some(ref version) = self.version {
            bibtex.push_str(&format!("  version = {{{}}},\n", version));
        }

        bibtex.push('}');
        bibtex
    }

    /// Generate CITATION.cff content
    pub fn to_cff(&self) -> String {
        let mut cff = String::from("cff-version: 1.2.0\n");
        cff.push_str(&format!("title: \"{}\"\n", self.title));
        cff.push_str("authors:\n");
        for author in &self.authors {
            cff.push_str(&format!("  - name: \"{}\"\n", author));
        }
        cff.push_str(&format!(
            "date-released: \"{}-{:02}-01\"\n",
            self.year,
            self.month.unwrap_or(1)
        ));

        if let Some(ref version) = self.version {
            cff.push_str(&format!("version: \"{}\"\n", version));
        }
        if let Some(ref doi) = self.doi {
            cff.push_str(&format!("doi: \"{}\"\n", doi));
        }
        if let Some(ref url) = self.url {
            cff.push_str(&format!("url: \"{}\"\n", url));
        }

        cff
    }
}
