//! Tests for archive module.

#![allow(clippy::module_inception)]
#[cfg(test)]
mod tests {
    use crate::research::archive::{
        config::{FigshareConfig, ZenodoConfig},
        deposit::ArchiveDeposit,
        identifiers::{IdentifierScheme, RelatedIdentifier, RelationType},
        metadata::{DepositMetadata, ResourceType},
        provider::ArchiveProvider,
        result::{DepositError, DepositResult},
        validate_doi,
    };
    use crate::research::artifact::{ArtifactType, Author, License, ResearchArtifact};

    fn create_test_artifact() -> ResearchArtifact {
        ResearchArtifact::new(
            "dataset-001",
            "Test Dataset for Machine Learning",
            ArtifactType::Dataset,
            License::CcBy4,
        )
        .with_author(Author::new("Alice Smith"))
        .with_doi("10.1234/test")
        .with_description("A test dataset for ML research")
        .with_keywords(["machine learning", "dataset"])
    }

    #[test]
    fn test_zenodo_deposit_struct() {
        let artifact = create_test_artifact();
        let deposit = ArchiveDeposit::new(ArchiveProvider::Zenodo, artifact)
            .with_text_file("data.csv", "a,b,c\n1,2,3");

        assert_eq!(deposit.provider, ArchiveProvider::Zenodo);
        assert_eq!(deposit.files.len(), 1);
        assert_eq!(deposit.metadata.title, "Test Dataset for Machine Learning");
    }

    #[test]
    fn test_figshare_deposit_struct() {
        let artifact = create_test_artifact();
        let deposit = ArchiveDeposit::new(ArchiveProvider::Figshare, artifact);

        assert_eq!(deposit.provider, ArchiveProvider::Figshare);
    }

    #[test]
    fn test_doi_format_validation() {
        // Valid DOIs
        assert!(validate_doi("10.1234/test"));
        assert!(validate_doi("10.5281/zenodo.1234567"));
        assert!(validate_doi("10.1000/xyz123"));
        assert!(validate_doi("10.12345/some-complex.id_here"));

        // Invalid DOIs
        assert!(!validate_doi("1234/test")); // Missing 10.
        assert!(!validate_doi("10.123/test")); // Too few digits after 10.
        assert!(!validate_doi("")); // Empty
        assert!(!validate_doi("doi:10.1234/test")); // Has prefix
    }

    #[test]
    fn test_deposit_result_url_generation() {
        let result = DepositResult {
            doi: "10.5281/zenodo.1234567".to_string(),
            record_id: "1234567".to_string(),
            url: "https://zenodo.org/record/1234567".to_string(),
            provider: ArchiveProvider::Zenodo,
        };

        assert_eq!(result.generate_url(), "https://zenodo.org/record/1234567");
        assert_eq!(result.doi_url(), "https://doi.org/10.5281/zenodo.1234567");
    }

    #[test]
    fn test_zenodo_config() {
        let config =
            ZenodoConfig::new("test-token").with_sandbox(true).with_community("ml-research");

        assert_eq!(config.token, "test-token");
        assert!(config.sandbox);
        assert_eq!(config.community, Some("ml-research".to_string()));
        assert_eq!(config.base_url(), "https://sandbox.zenodo.org");
    }

    #[test]
    fn test_figshare_config() {
        let config = FigshareConfig::new("test-token").with_project(12345);

        assert_eq!(config.token, "test-token");
        assert_eq!(config.project_id, Some(12345));
    }

    #[test]
    fn test_deposit_metadata_from_artifact() {
        let artifact = create_test_artifact();
        let metadata = DepositMetadata::from_artifact(&artifact);

        assert_eq!(metadata.title, "Test Dataset for Machine Learning");
        assert_eq!(metadata.authors, vec!["Alice Smith"]);
        assert_eq!(metadata.keywords, vec!["machine learning", "dataset"]);
        assert_eq!(metadata.resource_type, ResourceType::Dataset);
    }

    #[test]
    fn test_resource_type_conversion() {
        assert_eq!(ResourceType::from_artifact_type(ArtifactType::Dataset), ResourceType::Dataset);
        assert_eq!(ResourceType::from_artifact_type(ArtifactType::Code), ResourceType::Software);
        assert_eq!(
            ResourceType::from_artifact_type(ArtifactType::Paper),
            ResourceType::Publication
        );
    }

    #[test]
    fn test_resource_type_zenodo() {
        assert_eq!(ResourceType::Dataset.zenodo_type(), "dataset");
        assert_eq!(ResourceType::Software.zenodo_type(), "software");
        assert_eq!(ResourceType::Publication.zenodo_type(), "publication");
    }

    #[test]
    fn test_related_identifier_creation() {
        let doi_rel = RelatedIdentifier::is_identical_to("10.1234/test");
        assert_eq!(doi_rel.scheme, IdentifierScheme::Doi);
        assert_eq!(doi_rel.relation, RelationType::IsIdenticalTo);

        let url_rel = RelatedIdentifier::is_supplement_to("https://example.com/data");
        assert_eq!(url_rel.scheme, IdentifierScheme::Url);
        assert_eq!(url_rel.relation, RelationType::IsSupplementTo);
    }

    #[test]
    fn test_provider_urls() {
        assert_eq!(ArchiveProvider::Zenodo.base_url(), "https://zenodo.org");
        assert_eq!(ArchiveProvider::Figshare.base_url(), "https://figshare.com");

        assert!(ArchiveProvider::Zenodo.sandbox_url().is_some());
        assert!(ArchiveProvider::Figshare.sandbox_url().is_none());
    }

    #[test]
    fn test_deposit_no_files_error() {
        let artifact = create_test_artifact();
        let deposit = ArchiveDeposit::new(ArchiveProvider::Zenodo, artifact);
        // No files added

        let result = deposit.deposit();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), DepositError::NoFiles));
    }

    #[test]
    fn test_deposit_mock_success() {
        let artifact = create_test_artifact();
        let deposit = ArchiveDeposit::new(ArchiveProvider::Zenodo, artifact)
            .with_text_file("README.md", "# Test");

        let result = deposit.deposit();
        assert!(result.is_ok());

        let result = result.expect("operation should succeed");
        assert!(result.doi.starts_with("10.5281/zenodo."));
        assert!(result.url.starts_with("https://zenodo.org/record/"));
    }

    #[test]
    fn test_provider_display() {
        assert_eq!(format!("{}", ArchiveProvider::Zenodo), "Zenodo");
        assert_eq!(format!("{}", ArchiveProvider::Figshare), "Figshare");
        assert_eq!(format!("{}", ArchiveProvider::Dryad), "Dryad");
    }

    #[test]
    fn test_provider_display_dataverse() {
        assert_eq!(format!("{}", ArchiveProvider::Dataverse), "Dataverse");
    }

    #[test]
    fn test_zenodo_config_production_url() {
        let config = ZenodoConfig::new("token");
        assert_eq!(config.base_url(), "https://zenodo.org");
    }

    #[test]
    fn test_provider_sandbox_url_all() {
        assert_eq!(ArchiveProvider::Zenodo.sandbox_url(), Some("https://sandbox.zenodo.org"));
        assert!(ArchiveProvider::Dryad.sandbox_url().is_none());
        assert!(ArchiveProvider::Dataverse.sandbox_url().is_none());
    }

    #[test]
    fn test_provider_api_endpoints() {
        assert!(ArchiveProvider::Zenodo.api_endpoint().contains("zenodo"));
        assert!(ArchiveProvider::Figshare.api_endpoint().contains("figshare"));
        assert!(ArchiveProvider::Dryad.api_endpoint().contains("dryad"));
        assert!(ArchiveProvider::Dataverse.api_endpoint().contains("dataverse"));
    }

    #[test]
    fn test_provider_base_urls() {
        assert_eq!(ArchiveProvider::Dryad.base_url(), "https://datadryad.org");
        assert_eq!(ArchiveProvider::Dataverse.base_url(), "https://dataverse.harvard.edu");
    }

    #[test]
    fn test_related_identifier_cites() {
        let rel = RelatedIdentifier::cites("10.1234/paper");
        assert_eq!(rel.relation, RelationType::Cites);
        assert_eq!(rel.scheme, IdentifierScheme::Doi);
        assert_eq!(rel.identifier, "10.1234/paper");
    }

    #[test]
    fn test_related_identifier_other_scheme() {
        // Identifier that doesn't start with "10." or "http"
        let rel = RelatedIdentifier::is_identical_to("arxiv:2301.12345");
        assert_eq!(rel.scheme, IdentifierScheme::Other);
    }

    #[test]
    fn test_resource_type_from_model_artifact() {
        assert_eq!(ResourceType::from_artifact_type(ArtifactType::Model), ResourceType::Software);
    }

    #[test]
    fn test_resource_type_from_notebook_artifact() {
        use crate::research::artifact::ArtifactType;
        assert_eq!(ResourceType::from_artifact_type(ArtifactType::Notebook), ResourceType::Other);
        assert_eq!(ResourceType::from_artifact_type(ArtifactType::Workflow), ResourceType::Other);
    }

    #[test]
    fn test_resource_type_zenodo_all_variants() {
        assert_eq!(ResourceType::Presentation.zenodo_type(), "presentation");
        assert_eq!(ResourceType::Poster.zenodo_type(), "poster");
        assert_eq!(ResourceType::Image.zenodo_type(), "image");
        assert_eq!(ResourceType::Video.zenodo_type(), "video");
        assert_eq!(ResourceType::Other.zenodo_type(), "other");
    }

    #[test]
    fn test_deposit_with_binary_file() {
        let artifact = create_test_artifact();
        let binary_data = vec![0u8, 1, 2, 3, 255, 254, 253];
        let deposit = ArchiveDeposit::new(ArchiveProvider::Zenodo, artifact)
            .with_file("data.bin", binary_data.clone());

        assert_eq!(deposit.files.len(), 1);
        assert_eq!(deposit.files[0].1, binary_data);
    }

    #[test]
    fn test_deposit_metadata_from_citation() {
        use crate::research::citation::CitationMetadata;

        let artifact = create_test_artifact();
        let citation =
            CitationMetadata::new(artifact, 2024).with_url("https://example.com/supplement");

        let metadata = DepositMetadata::from_citation(&citation);
        assert_eq!(metadata.title, "Test Dataset for Machine Learning");
        // Should have both the original DOI and the supplement URL
        assert!(!metadata.related_identifiers.is_empty());
    }

    #[test]
    fn test_deposit_error_display() {
        let err = DepositError::NoFiles;
        assert!(format!("{err}").contains("No files"));

        let err = DepositError::AuthenticationFailed;
        assert!(format!("{err}").contains("Authentication"));

        let err = DepositError::InvalidMetadata("missing title".to_string());
        assert!(format!("{err}").contains("missing title"));

        let err = DepositError::UploadFailed("timeout".to_string());
        assert!(format!("{err}").contains("timeout"));

        let err = DepositError::ApiError("rate limit".to_string());
        assert!(format!("{err}").contains("rate limit"));
    }

    #[test]
    fn test_deposit_result_serde_roundtrip() {
        let result = DepositResult {
            doi: "10.5281/zenodo.123".to_string(),
            record_id: "123".to_string(),
            url: "https://zenodo.org/record/123".to_string(),
            provider: ArchiveProvider::Figshare,
        };
        let json = serde_json::to_string(&result).expect("JSON serialization should succeed");
        let parsed: DepositResult =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(parsed.doi, result.doi);
        assert_eq!(parsed.provider, ArchiveProvider::Figshare);
    }

    #[test]
    fn test_zenodo_config_serde_roundtrip() {
        let config =
            ZenodoConfig::new("secret-token").with_sandbox(true).with_community("ml-research");
        let json = serde_json::to_string(&config).expect("JSON serialization should succeed");
        let parsed: ZenodoConfig =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(parsed.token, "secret-token");
        assert!(parsed.sandbox);
        assert_eq!(parsed.community, Some("ml-research".to_string()));
    }

    #[test]
    fn test_figshare_config_serde_roundtrip() {
        let config = FigshareConfig::new("api-key").with_project(99999);
        let json = serde_json::to_string(&config).expect("JSON serialization should succeed");
        let parsed: FigshareConfig =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(parsed.token, "api-key");
        assert_eq!(parsed.project_id, Some(99999));
    }

    #[test]
    fn test_deposit_metadata_serde_roundtrip() {
        let artifact = create_test_artifact();
        let metadata = DepositMetadata::from_artifact(&artifact);
        let json = serde_json::to_string(&metadata).expect("JSON serialization should succeed");
        let parsed: DepositMetadata =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(parsed.title, metadata.title);
        assert_eq!(parsed.authors, metadata.authors);
    }

    #[test]
    fn test_related_identifier_serde_roundtrip() {
        let rel = RelatedIdentifier::is_supplement_to("https://github.com/example/repo");
        let json = serde_json::to_string(&rel).expect("JSON serialization should succeed");
        let parsed: RelatedIdentifier =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(parsed.identifier, rel.identifier);
        assert_eq!(parsed.relation, RelationType::IsSupplementTo);
    }

    #[test]
    fn test_archive_provider_serde_roundtrip() {
        for provider in [
            ArchiveProvider::Zenodo,
            ArchiveProvider::Figshare,
            ArchiveProvider::Dryad,
            ArchiveProvider::Dataverse,
        ] {
            let json = serde_json::to_string(&provider).expect("JSON serialization should succeed");
            let parsed: ArchiveProvider =
                serde_json::from_str(&json).expect("JSON deserialization should succeed");
            assert_eq!(parsed, provider);
        }
    }

    #[test]
    fn test_deposit_metadata_without_description() {
        let artifact = ResearchArtifact::new(
            "no-desc-001",
            "Artifact Without Description",
            ArtifactType::Code,
            License::Mit,
        );
        // No description set
        let metadata = DepositMetadata::from_artifact(&artifact);
        // Should generate a default description
        assert!(metadata.description.contains("Artifact Without Description"));
    }

    #[test]
    fn test_is_supplement_to_with_doi() {
        let rel = RelatedIdentifier::is_supplement_to("10.5555/supplement");
        assert_eq!(rel.scheme, IdentifierScheme::Doi);
        assert_eq!(rel.relation, RelationType::IsSupplementTo);
    }

    #[test]
    fn test_multiple_files_deposit() {
        let artifact = create_test_artifact();
        let deposit = ArchiveDeposit::new(ArchiveProvider::Zenodo, artifact)
            .with_text_file("README.md", "# Documentation")
            .with_text_file("data.csv", "a,b,c")
            .with_file("model.bin", vec![1, 2, 3, 4]);

        assert_eq!(deposit.files.len(), 3);
    }
}
