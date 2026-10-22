//! TSP Instance YAML Configuration
//!
//! YAML-first architecture for TSP instances. Users can download, modify, and re-run
//! experiments without touching code.
//!
//! # Example YAML
//!
//! ```yaml
//! meta:
//!   id: "TSP-BAY-006"
//!   version: "1.0.0"
//!   description: "6-city Bay Area ground truth instance"
//!   source: "Google Maps (Dec 2024)"
//!   units: "miles"
//!   optimal_known: 115
//!
//! cities:
//!   - id: 0
//!     name: "San Francisco"
//!     alias: "SF"
//!     coords: { lat: 37.7749, lon: -122.4194 }
//!
//! matrix:
//!   - [0, 12, 48]
//!   - [12, 0, 42]
//!   - [48, 42, 0]
//!
//! algorithm:
//!   method: "grasp"
//!   params:
//!     rcl_size: 3
//!     restarts: 10
//!     two_opt: true
//!     seed: 42
//! ```

use serde::{Deserialize, Serialize};

/// Geographic coordinates (lat/lon).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Coords {
    /// Latitude in degrees.
    pub lat: f64,
    /// Longitude in degrees.
    pub lon: f64,
}

impl Default for Coords {
    fn default() -> Self {
        Self { lat: 0.0, lon: 0.0 }
    }
}

/// A city in the TSP instance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TspCity {
    /// Unique city identifier (0-indexed).
    pub id: usize,
    /// Full city name.
    pub name: String,
    /// Short alias (2-4 chars).
    pub alias: String,
    /// Geographic coordinates.
    pub coords: Coords,
}

impl TspCity {
    /// Create a new city.
    #[must_use]
    pub fn new(
        id: usize,
        name: impl Into<String>,
        alias: impl Into<String>,
        lat: f64,
        lon: f64,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            alias: alias.into(),
            coords: Coords { lat, lon },
        }
    }
}

/// Metadata about the TSP instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TspMeta {
    /// Unique instance identifier.
    pub id: String,
    /// Version string (semver).
    #[serde(default = "default_version")]
    pub version: String,
    /// Human-readable description.
    pub description: String,
    /// Data source (e.g., "Google Maps").
    #[serde(default)]
    pub source: String,
    /// Distance units (e.g., "miles", "km").
    #[serde(default = "default_units")]
    pub units: String,
    /// Known optimal solution (for verification).
    #[serde(default)]
    pub optimal_known: Option<u32>,
}

fn default_version() -> String {
    "1.0.0".to_string()
}

fn default_units() -> String {
    "miles".to_string()
}

impl Default for TspMeta {
    fn default() -> Self {
        Self {
            id: "TSP-UNNAMED".to_string(),
            version: default_version(),
            description: "Unnamed TSP instance".to_string(),
            source: String::new(),
            units: default_units(),
            optimal_known: None,
        }
    }
}

/// Algorithm parameters for TSP solving.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TspParams {
    /// RCL size for GRASP (restricted candidate list).
    #[serde(default = "default_rcl_size")]
    pub rcl_size: usize,
    /// Number of GRASP restarts.
    #[serde(default = "default_restarts")]
    pub restarts: usize,
    /// Enable 2-opt local search.
    #[serde(default = "default_two_opt")]
    pub two_opt: bool,
    /// Random seed for reproducibility.
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_rcl_size() -> usize {
    3
}

fn default_restarts() -> usize {
    10
}

fn default_two_opt() -> bool {
    true
}

fn default_seed() -> u64 {
    42
}

impl Default for TspParams {
    fn default() -> Self {
        Self {
            rcl_size: default_rcl_size(),
            restarts: default_restarts(),
            two_opt: default_two_opt(),
            seed: default_seed(),
        }
    }
}

/// Algorithm configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TspAlgorithmConfig {
    /// Algorithm method: `greedy`, `grasp`, or `brute_force`.
    #[serde(default = "default_method")]
    pub method: String,
    /// Algorithm-specific parameters.
    #[serde(default)]
    pub params: TspParams,
}

fn default_method() -> String {
    "grasp".to_string()
}

impl Default for TspAlgorithmConfig {
    fn default() -> Self {
        Self {
            method: default_method(),
            params: TspParams::default(),
        }
    }
}

/// Complete TSP instance configuration (YAML-first).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TspInstanceYaml {
    /// Instance metadata.
    pub meta: TspMeta,
    /// List of cities.
    pub cities: Vec<TspCity>,
    /// Distance matrix (n x n).
    pub matrix: Vec<Vec<u32>>,
    /// Algorithm configuration.
    #[serde(default)]
    pub algorithm: TspAlgorithmConfig,
}

/// Error types for TSP instance parsing/validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TspInstanceError {
    /// YAML parsing failed.
    ParseError(String),
    /// Matrix dimensions don't match city count.
    MatrixDimensionMismatch { expected: usize, got_rows: usize },
    /// Matrix row has wrong number of columns.
    MatrixRowMismatch {
        row: usize,
        expected: usize,
        got: usize,
    },
    /// Triangle inequality violated.
    TriangleInequalityViolation { i: usize, j: usize, k: usize },
    /// Matrix is not symmetric.
    AsymmetricMatrix {
        i: usize,
        j: usize,
        forward: u32,
        backward: u32,
    },
    /// Invalid city ID.
    InvalidCityId { id: usize, max: usize },
    /// IO error.
    IoError(String),
}

impl std::fmt::Display for TspInstanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "YAML parse error: {msg}"),
            Self::MatrixDimensionMismatch { expected, got_rows } => {
                write!(f, "Matrix dimension mismatch: expected {expected}x{expected}, got {got_rows} rows")
            }
            Self::MatrixRowMismatch { row, expected, got } => {
                write!(f, "Matrix row {row} has {got} columns, expected {expected}")
            }
            Self::TriangleInequalityViolation { i, j, k } => {
                write!(
                    f,
                    "Triangle inequality violated: d({i},{k}) > d({i},{j}) + d({j},{k})"
                )
            }
            Self::AsymmetricMatrix {
                i,
                j,
                forward,
                backward,
            } => {
                write!(
                    f,
                    "Asymmetric matrix: d({i},{j})={forward} != d({j},{i})={backward}"
                )
            }
            Self::InvalidCityId { id, max } => {
                write!(f, "Invalid city ID {id}, max is {max}")
            }
            Self::IoError(msg) => write!(f, "IO error: {msg}"),
        }
    }
}

impl std::error::Error for TspInstanceError {}

impl TspInstanceYaml {
    /// Parse TSP instance from YAML string.
    ///
    /// # Errors
    ///
    /// Returns `TspInstanceError::ParseError` if YAML is invalid.
    pub fn from_yaml(yaml: &str) -> Result<Self, TspInstanceError> {
        serde_yaml::from_str(yaml).map_err(|e| TspInstanceError::ParseError(e.to_string()))
    }

    /// Load TSP instance from a YAML file.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or YAML is invalid.
    pub fn from_yaml_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, TspInstanceError> {
        let content =
            std::fs::read_to_string(path).map_err(|e| TspInstanceError::IoError(e.to_string()))?;
        Self::from_yaml(&content)
    }

    /// Serialize to YAML string.
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    pub fn to_yaml(&self) -> Result<String, TspInstanceError> {
        serde_yaml::to_string(self).map_err(|e| TspInstanceError::ParseError(e.to_string()))
    }

    /// Number of cities in the instance.
    #[must_use]
    pub fn city_count(&self) -> usize {
        self.cities.len()
    }

    /// Get distance between two cities.
    ///
    /// # Panics
    ///
    /// Panics if city indices are out of bounds.
    #[must_use]
    pub fn distance(&self, from: usize, to: usize) -> u32 {
        self.matrix[from][to]
    }
}

mod checks;

// =============================================================================
// WASM Bindings (OR-001-06)
// =============================================================================

#[cfg(feature = "wasm")]
mod wasm;

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "wasm"))]
mod wasm_tests;
