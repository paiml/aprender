//! Security advisory information.

use serde::{Deserialize, Serialize};

use super::Severity;

/// Security advisory information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Advisory {
    /// Advisory ID (e.g., RUSTSEC-2021-0001)
    pub id: String,

    /// Severity level
    pub severity: Severity,

    /// Short title/description
    pub title: String,
}

impl Advisory {
    /// Create a new advisory
    pub fn new(id: impl Into<String>, severity: Severity, title: impl Into<String>) -> Self {
        Self { id: id.into(), severity, title: title.into() }
    }
}
