//! Related identifiers and relation types.

use serde::{Deserialize, Serialize};

/// Related identifier for linking resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedIdentifier {
    /// The identifier value (DOI, URL, etc.)
    pub identifier: String,
    /// Relation type
    pub relation: RelationType,
    /// Identifier scheme (DOI, URL, etc.)
    pub scheme: IdentifierScheme,
}

impl RelatedIdentifier {
    /// Create a "is identical to" relation
    pub fn is_identical_to(identifier: impl Into<String>) -> Self {
        let id = identifier.into();
        let scheme = if id.starts_with("10.") {
            IdentifierScheme::Doi
        } else if id.starts_with("http") {
            IdentifierScheme::Url
        } else {
            IdentifierScheme::Other
        };

        Self { identifier: id, relation: RelationType::IsIdenticalTo, scheme }
    }

    /// Create a "is supplement to" relation
    pub fn is_supplement_to(identifier: impl Into<String>) -> Self {
        let id = identifier.into();
        let scheme =
            if id.starts_with("10.") { IdentifierScheme::Doi } else { IdentifierScheme::Url };

        Self { identifier: id, relation: RelationType::IsSupplementTo, scheme }
    }

    /// Create a "cites" relation
    pub fn cites(identifier: impl Into<String>) -> Self {
        Self {
            identifier: identifier.into(),
            relation: RelationType::Cites,
            scheme: IdentifierScheme::Doi,
        }
    }
}

/// Relation types (based on DataCite)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationType {
    IsCitedBy,
    Cites,
    IsSupplementTo,
    IsSupplementedBy,
    IsContinuedBy,
    Continues,
    IsDescribedBy,
    Describes,
    HasMetadata,
    IsMetadataFor,
    HasVersion,
    IsVersionOf,
    IsNewVersionOf,
    IsPreviousVersionOf,
    IsPartOf,
    HasPart,
    IsReferencedBy,
    References,
    IsDocumentedBy,
    Documents,
    IsCompiledBy,
    Compiles,
    IsVariantFormOf,
    IsOriginalFormOf,
    IsIdenticalTo,
    IsReviewedBy,
    Reviews,
    IsDerivedFrom,
    IsSourceOf,
    IsRequiredBy,
    Requires,
    IsObsoletedBy,
    Obsoletes,
}

/// Identifier scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdentifierScheme {
    Doi,
    Url,
    Orcid,
    Ror,
    Arxiv,
    Pmid,
    Other,
}
