//! Dataset split type

use serde::{Deserialize, Serialize};

/// Dataset split type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Split {
    /// Training split
    Train,
    /// Validation split
    Validation,
    /// Test split
    Test,
}

impl std::fmt::Display for Split {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Train => write!(f, "train"),
            Self::Validation => write!(f, "validation"),
            Self::Test => write!(f, "test"),
        }
    }
}
