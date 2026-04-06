//! Tests for research artifact types.

use super::*;

#[test]
fn test_orcid_validation_valid() {
    // Standard ORCID
    assert!(validate_orcid("0000-0002-1825-0097"));
    // ORCID with X checksum
    assert!(validate_orcid("0000-0001-5109-351X"));
    // All zeros
    assert!(validate_orcid("0000-0000-0000-0000"));
    // All nines
    assert!(validate_orcid("9999-9999-9999-9999"));
}

#[test]
fn test_orcid_validation_invalid() {
    // Missing hyphen
    assert!(!validate_orcid("000000021825-0097"));
    // Too short
    assert!(!validate_orcid("0000-0002-1825"));
    // Invalid character
    assert!(!validate_orcid("0000-0002-1825-009A"));
    // Lowercase x
    assert!(!validate_orcid("0000-0001-5109-351x"));
    // Empty
    assert!(!validate_orcid(""));
    // URL format (should just be the ID)
    assert!(!validate_orcid("https://orcid.org/0000-0002-1825-0097"));
}

#[test]
fn test_ror_id_validation() {
    // Valid ROR IDs
    assert!(validate_ror_id("https://ror.org/03yrm5c26"));
    assert!(validate_ror_id("https://ror.org/00hj8s172"));

    // Invalid ROR IDs
    assert!(!validate_ror_id("ror.org/03yrm5c26")); // Missing https://
    assert!(!validate_ror_id("https://ror.org/03yrm5c2")); // Too short
    assert!(!validate_ror_id("https://ror.org/03yrm5c267")); // Too long
    assert!(!validate_ror_id("https://ror.org/03YRM5C26")); // Uppercase
    assert!(!validate_ror_id("")); // Empty
}

#[test]
fn test_contributor_role_display() {
    assert_eq!(format!("{}", ContributorRole::Conceptualization), "Conceptualization");
    assert_eq!(format!("{}", ContributorRole::DataCuration), "Data curation");
    assert_eq!(format!("{}", ContributorRole::FormalAnalysis), "Formal analysis");
    assert_eq!(format!("{}", ContributorRole::WritingOriginal), "Writing – original draft");
    assert_eq!(format!("{}", ContributorRole::WritingReview), "Writing – review & editing");
}

#[test]
fn test_artifact_with_multiple_authors() {
    let author1 = Author::new("Alice Smith")
        .with_orcid("0000-0002-1825-0097")
        .expect("operation should succeed")
        .with_role(ContributorRole::Conceptualization)
        .with_role(ContributorRole::WritingOriginal);

    let author2 = Author::new("Bob Jones")
        .with_role(ContributorRole::Software)
        .with_role(ContributorRole::DataCuration);

    let artifact = ResearchArtifact::new(
        "artifact-001",
        "Novel Deep Learning Architecture",
        ArtifactType::Paper,
        License::CcBy4,
    )
    .with_authors([author1, author2])
    .with_doi("10.1000/xyz123")
    .with_keywords(["deep learning", "neural networks"]);

    assert_eq!(artifact.authors.len(), 2);
    assert_eq!(artifact.authors[0].name, "Alice Smith");
    assert_eq!(artifact.authors[0].orcid, Some("0000-0002-1825-0097".to_string()));
    assert_eq!(artifact.authors[0].roles.len(), 2);
    assert_eq!(artifact.authors[1].name, "Bob Jones");
    assert_eq!(artifact.authors[1].orcid, None);
    assert_eq!(artifact.doi, Some("10.1000/xyz123".to_string()));
    assert_eq!(artifact.keywords.len(), 2);
}

#[test]
fn test_author_last_name() {
    let author1 = Author::new("Alice Marie Smith");
    assert_eq!(author1.last_name(), "Smith");

    let author2 = Author::new("Madonna");
    assert_eq!(author2.last_name(), "Madonna");

    let author3 = Author::new("Ludwig van Beethoven");
    assert_eq!(author3.last_name(), "Beethoven");
}

#[test]
fn test_affiliation_with_ror() {
    let affiliation = Affiliation::new("MIT")
        .with_ror_id("https://ror.org/03yrm5c26")
        .expect("operation should succeed")
        .with_country("US");

    assert_eq!(affiliation.name, "MIT");
    assert_eq!(affiliation.ror_id, Some("https://ror.org/03yrm5c26".to_string()));
    assert_eq!(affiliation.country, Some("US".to_string()));
}

#[test]
fn test_affiliation_invalid_ror() {
    let result = Affiliation::new("MIT").with_ror_id("invalid-ror");
    assert!(result.is_err());
}

#[test]
fn test_author_invalid_orcid() {
    let result = Author::new("Alice Smith").with_orcid("invalid-orcid");
    assert!(result.is_err());
}

#[test]
fn test_artifact_type_display() {
    assert_eq!(format!("{}", ArtifactType::Dataset), "Dataset");
    assert_eq!(format!("{}", ArtifactType::Model), "Model");
    assert_eq!(format!("{}", ArtifactType::Code), "Code");
    assert_eq!(format!("{}", ArtifactType::Paper), "Paper");
    assert_eq!(format!("{}", ArtifactType::Notebook), "Notebook");
    assert_eq!(format!("{}", ArtifactType::Workflow), "Workflow");
}

#[test]
fn test_license_display() {
    assert_eq!(format!("{}", License::Mit), "MIT");
    assert_eq!(format!("{}", License::Apache2), "Apache-2.0");
    assert_eq!(format!("{}", License::CcBy4), "CC-BY-4.0");
    assert_eq!(format!("{}", License::Custom("LGPL-2.1".to_string())), "LGPL-2.1");
}

#[test]
fn test_author_with_affiliation() {
    let affiliation = Affiliation::new("Stanford University").with_country("US");

    let author = Author::new("Jane Doe").with_affiliation(affiliation);

    assert_eq!(author.affiliations.len(), 1);
    assert_eq!(author.affiliations[0].name, "Stanford University");
}

#[test]
fn test_role_deduplication() {
    let author = Author::new("Test Author")
        .with_role(ContributorRole::Software)
        .with_role(ContributorRole::Software)
        .with_roles([ContributorRole::Software, ContributorRole::Validation]);

    assert_eq!(author.roles.len(), 2);
    assert!(author.roles.contains(&ContributorRole::Software));
    assert!(author.roles.contains(&ContributorRole::Validation));
}
