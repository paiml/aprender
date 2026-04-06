//! Author with ORCID and contributor roles.

use serde::{Deserialize, Serialize};

use super::{validate_orcid, Affiliation, ContributorRole, ValidationError};

/// Author with ORCID and contributor roles
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Author {
    /// Full name
    pub name: String,
    /// ORCID identifier (optional)
    pub orcid: Option<String>,
    /// Institutional affiliations
    pub affiliations: Vec<Affiliation>,
    /// Contributor roles (CRediT taxonomy)
    pub roles: Vec<ContributorRole>,
}

impl Author {
    /// Create a new author
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), orcid: None, affiliations: Vec::new(), roles: Vec::new() }
    }

    /// Set the ORCID (validates format)
    pub fn with_orcid(mut self, orcid: impl Into<String>) -> Result<Self, ValidationError> {
        let id = orcid.into();
        if !validate_orcid(&id) {
            return Err(ValidationError::InvalidOrcid(id));
        }
        self.orcid = Some(id);
        Ok(self)
    }

    /// Add an affiliation
    pub fn with_affiliation(mut self, affiliation: Affiliation) -> Self {
        self.affiliations.push(affiliation);
        self
    }

    /// Add a contributor role
    pub fn with_role(mut self, role: ContributorRole) -> Self {
        if !self.roles.contains(&role) {
            self.roles.push(role);
        }
        self
    }

    /// Add multiple contributor roles
    pub fn with_roles(mut self, roles: impl IntoIterator<Item = ContributorRole>) -> Self {
        for role in roles {
            if !self.roles.contains(&role) {
                self.roles.push(role);
            }
        }
        self
    }

    /// Get the author's last name (for citation keys)
    pub fn last_name(&self) -> &str {
        self.name.split_whitespace().next_back().unwrap_or(&self.name)
    }
}
