//! Anonymization Config for double-blind (ENT-023)
//!
//! Provides anonymization capabilities for research artifacts
//! to support double-blind peer review.

use crate::research::artifact::{Author, ResearchArtifact};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Configuration for anonymization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnonymizationConfig {
    /// Salt for deterministic anonymous ID generation
    pub salt: String,
    /// Replacement text for author names
    pub author_replacement: String,
    /// Replacement text for affiliations
    pub affiliation_replacement: String,
    /// Whether to strip ORCID identifiers
    pub strip_orcid: bool,
    /// Whether to strip ROR identifiers
    pub strip_ror: bool,
    /// Whether to strip DOI
    pub strip_doi: bool,
    /// Custom patterns to redact (regex strings)
    pub redact_patterns: Vec<String>,
}

impl Default for AnonymizationConfig {
    fn default() -> Self {
        Self {
            salt: String::new(),
            author_replacement: "Anonymous Author".to_string(),
            affiliation_replacement: "Anonymous Institution".to_string(),
            strip_orcid: true,
            strip_ror: true,
            strip_doi: false,
            redact_patterns: Vec::new(),
        }
    }
}

impl AnonymizationConfig {
    /// Create a new anonymization config with a salt
    pub fn new(salt: impl Into<String>) -> Self {
        Self { salt: salt.into(), ..Default::default() }
    }

    /// Set the author replacement text
    pub fn with_author_replacement(mut self, replacement: impl Into<String>) -> Self {
        self.author_replacement = replacement.into();
        self
    }

    /// Set the affiliation replacement text
    pub fn with_affiliation_replacement(mut self, replacement: impl Into<String>) -> Self {
        self.affiliation_replacement = replacement.into();
        self
    }

    /// Set whether to strip DOI
    pub fn with_strip_doi(mut self, strip: bool) -> Self {
        self.strip_doi = strip;
        self
    }

    /// Add patterns to redact
    pub fn with_redact_patterns(
        mut self,
        patterns: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.redact_patterns.extend(patterns.into_iter().map(Into::into));
        self
    }

    /// Generate a deterministic anonymous ID from an original ID
    pub fn generate_anonymous_id(&self, original_id: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.salt.as_bytes());
        hasher.update(original_id.as_bytes());
        let hash = hasher.finalize();
        format!("anon-{}", hex::encode(hash.get(..8).unwrap_or(&hash))) // Use first 8 bytes for brevity
    }

    /// Anonymize a research artifact
    pub fn anonymize(&self, artifact: &ResearchArtifact) -> AnonymizedArtifact {
        let anonymous_id = self.generate_anonymous_id(&artifact.id);

        // Create anonymous authors
        let anonymous_authors: Vec<AnonymousAuthor> = artifact
            .authors
            .iter()
            .enumerate()
            .map(|(i, _)| AnonymousAuthor {
                placeholder: format!("{} {}", self.author_replacement, i + 1),
                affiliation_placeholder: self.affiliation_replacement.clone(),
            })
            .collect();

        // Strip DOI if configured
        let doi = if self.strip_doi { None } else { artifact.doi.clone() };

        // Anonymize description if present
        let description = artifact
            .description
            .as_ref()
            .map(|desc| anonymize_text_internal(desc, &artifact.authors, self));

        AnonymizedArtifact {
            anonymous_id,
            original_id_hash: self.hash_original_id(&artifact.id),
            title: artifact.title.clone(),
            authors: anonymous_authors,
            artifact_type: artifact.artifact_type,
            license: artifact.license.clone(),
            doi,
            version: artifact.version.clone(),
            description,
            keywords: artifact.keywords.clone(),
        }
    }

    /// Hash the original ID for later verification
    fn hash_original_id(&self, original_id: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.salt.as_bytes());
        hasher.update(b"original:");
        hasher.update(original_id.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Verify that an anonymous artifact came from a specific original ID
    pub fn verify_original_id(&self, artifact: &AnonymizedArtifact, original_id: &str) -> bool {
        let expected_hash = self.hash_original_id(original_id);
        artifact.original_id_hash == expected_hash
    }
}

/// Anonymous author placeholder
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnonymousAuthor {
    /// Placeholder name (e.g., "Anonymous Author 1")
    pub placeholder: String,
    /// Placeholder affiliation
    pub affiliation_placeholder: String,
}

/// Anonymized research artifact
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnonymizedArtifact {
    /// Anonymous ID (derived from original ID + salt)
    pub anonymous_id: String,
    /// Hash of original ID for later verification
    pub original_id_hash: String,
    /// Title (preserved)
    pub title: String,
    /// Anonymous author placeholders
    pub authors: Vec<AnonymousAuthor>,
    /// Artifact type (preserved)
    pub artifact_type: crate::research::artifact::ArtifactType,
    /// License (preserved)
    pub license: crate::research::artifact::License,
    /// DOI (may be stripped based on config)
    pub doi: Option<String>,
    /// Version (preserved)
    pub version: String,
    /// Description (preserved)
    pub description: Option<String>,
    /// Keywords (preserved)
    pub keywords: Vec<String>,
}

impl AnonymizedArtifact {
    /// Convert to a format suitable for double-blind export (JSON)
    pub fn to_double_blind_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_err| "{}".to_string())
    }

    /// Get author count
    pub fn author_count(&self) -> usize {
        self.authors.len()
    }
}

/// Anonymize a string by replacing author names
pub fn anonymize_text(text: &str, authors: &[Author], config: &AnonymizationConfig) -> String {
    anonymize_text_internal(text, authors, config)
}

/// Internal function for text anonymization
fn anonymize_text_internal(text: &str, authors: &[Author], config: &AnonymizationConfig) -> String {
    let mut result = text.to_string();

    for (i, author) in authors.iter().enumerate() {
        // Replace full name
        result = result.replace(&author.name, &format!("{} {}", config.author_replacement, i + 1));

        // Replace last name only
        let last_name = author.last_name();
        if last_name != author.name {
            result = result.replace(last_name, &format!("{} {}", config.author_replacement, i + 1));
        }

        // Replace affiliations
        for affiliation in &author.affiliations {
            result = result.replace(&affiliation.name, &config.affiliation_replacement);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research::artifact::{Affiliation, ArtifactType, ContributorRole, License};

    fn create_test_artifact() -> ResearchArtifact {
        let author1 = Author::new("Alice Smith")
            .with_orcid("0000-0002-1825-0097")
            .expect("operation should succeed")
            .with_role(ContributorRole::Conceptualization)
            .with_affiliation(
                Affiliation::new("MIT")
                    .with_ror_id("https://ror.org/03yrm5c26")
                    .expect("operation should succeed"),
            );

        let author2 =
            Author::new("Bob Jones").with_affiliation(Affiliation::new("Stanford University"));

        ResearchArtifact::new(
            "paper-2024-001",
            "Novel Deep Learning Architecture",
            ArtifactType::Paper,
            License::CcBy4,
        )
        .with_authors([author1, author2])
        .with_doi("10.1234/example.2024")
        .with_description("A groundbreaking paper by Alice Smith from MIT")
    }

    #[test]
    fn test_anonymize_removes_authors() {
        let artifact = create_test_artifact();
        let config = AnonymizationConfig::new("secret-salt");

        let anon = config.anonymize(&artifact);

        assert_eq!(anon.authors.len(), 2);
        assert_eq!(anon.authors[0].placeholder, "Anonymous Author 1");
        assert_eq!(anon.authors[1].placeholder, "Anonymous Author 2");
    }

    #[test]
    fn test_anonymize_removes_affiliations() {
        let artifact = create_test_artifact();
        let config = AnonymizationConfig::new("secret-salt");

        let anon = config.anonymize(&artifact);

        assert_eq!(anon.authors[0].affiliation_placeholder, "Anonymous Institution");
        assert_eq!(anon.authors[1].affiliation_placeholder, "Anonymous Institution");
    }

    #[test]
    fn test_anonymous_id_deterministic() {
        let config = AnonymizationConfig::new("fixed-salt");

        let id1 = config.generate_anonymous_id("paper-001");
        let id2 = config.generate_anonymous_id("paper-001");

        assert_eq!(id1, id2);
        assert!(id1.starts_with("anon-"));
    }

    #[test]
    fn test_salt_changes_ids() {
        let config1 = AnonymizationConfig::new("salt-1");
        let config2 = AnonymizationConfig::new("salt-2");

        let id1 = config1.generate_anonymous_id("paper-001");
        let id2 = config2.generate_anonymous_id("paper-001");

        assert_ne!(id1, id2);
    }

    #[test]
    fn test_double_blind_export() {
        let artifact = create_test_artifact();
        let config = AnonymizationConfig::new("review-salt");

        let anon = config.anonymize(&artifact);
        let json = anon.to_double_blind_json();

        // Should not contain author names
        assert!(!json.contains("Alice Smith"));
        assert!(!json.contains("Bob Jones"));

        // Should not contain affiliations
        assert!(!json.contains("MIT"));
        assert!(!json.contains("Stanford"));

        // Should contain anonymous placeholders
        assert!(json.contains("Anonymous Author"));
        assert!(json.contains("Anonymous Institution"));

        // Should preserve title
        assert!(json.contains("Novel Deep Learning Architecture"));
    }

    #[test]
    fn test_strip_doi_option() {
        let artifact = create_test_artifact();

        // Default: DOI preserved
        let config1 = AnonymizationConfig::new("salt");
        let anon1 = config1.anonymize(&artifact);
        assert!(anon1.doi.is_some());

        // With strip_doi: DOI removed
        let config2 = AnonymizationConfig::new("salt").with_strip_doi(true);
        let anon2 = config2.anonymize(&artifact);
        assert!(anon2.doi.is_none());
    }

    #[test]
    fn test_verify_original_id() {
        let artifact = create_test_artifact();
        let config = AnonymizationConfig::new("verification-salt");

        let anon = config.anonymize(&artifact);

        // Should verify with correct original ID
        assert!(config.verify_original_id(&anon, "paper-2024-001"));

        // Should fail with wrong original ID
        assert!(!config.verify_original_id(&anon, "paper-2024-002"));
    }

    #[test]
    fn test_custom_replacements() {
        let artifact = create_test_artifact();
        let config = AnonymizationConfig::new("salt")
            .with_author_replacement("Reviewer")
            .with_affiliation_replacement("Hidden University");

        let anon = config.anonymize(&artifact);

        assert_eq!(anon.authors[0].placeholder, "Reviewer 1");
        assert_eq!(anon.authors[0].affiliation_placeholder, "Hidden University");
    }

    #[test]
    fn test_anonymize_text() {
        let author = Author::new("Alice Smith").with_affiliation(Affiliation::new("MIT"));

        let text = "This paper by Alice Smith from MIT presents...";
        let config = AnonymizationConfig::new("salt");

        let anon_text = anonymize_text(text, &[author], &config);

        assert!(!anon_text.contains("Alice Smith"));
        assert!(!anon_text.contains("MIT"));
        assert!(anon_text.contains("Anonymous Author 1"));
        assert!(anon_text.contains("Anonymous Institution"));
    }

    #[test]
    fn test_anonymize_text_last_name() {
        let author = Author::new("Alice Marie Smith").with_affiliation(Affiliation::new("MIT"));

        let text = "Smith et al. demonstrated...";
        let config = AnonymizationConfig::new("salt");

        let anon_text = anonymize_text(text, &[author], &config);

        assert!(!anon_text.contains("Smith"));
        assert!(anon_text.contains("Anonymous Author 1"));
    }

    #[test]
    fn test_author_count() {
        let artifact = create_test_artifact();
        let config = AnonymizationConfig::new("salt");
        let anon = config.anonymize(&artifact);

        assert_eq!(anon.author_count(), 2);
    }

    #[test]
    fn test_preserved_fields() {
        let artifact = create_test_artifact();
        let config = AnonymizationConfig::new("salt");
        let anon = config.anonymize(&artifact);

        // These should be preserved
        assert_eq!(anon.title, artifact.title);
        assert_eq!(anon.version, artifact.version);
        assert_eq!(anon.artifact_type, artifact.artifact_type);
        assert_eq!(anon.license, artifact.license);
        assert_eq!(anon.keywords, artifact.keywords);
    }
}
