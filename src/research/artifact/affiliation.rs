//! Institutional affiliation with optional ROR identifier.

use serde::{Deserialize, Serialize};

use super::{validate_ror_id, ValidationError};

/// Institutional affiliation with optional ROR identifier
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Affiliation {
    /// Institution name
    pub name: String,
    /// Research Organization Registry ID (optional)
    pub ror_id: Option<String>,
    /// Country code (ISO 3166-1 alpha-2)
    pub country: Option<String>,
}

impl Affiliation {
    /// Create a new affiliation
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), ror_id: None, country: None }
    }

    /// Set the ROR ID (validates format)
    pub fn with_ror_id(mut self, ror_id: impl Into<String>) -> Result<Self, ValidationError> {
        let ror = ror_id.into();
        if !validate_ror_id(&ror) {
            return Err(ValidationError::InvalidRorId(ror));
        }
        self.ror_id = Some(ror);
        Ok(self)
    }

    /// Set the country code
    pub fn with_country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }
}
