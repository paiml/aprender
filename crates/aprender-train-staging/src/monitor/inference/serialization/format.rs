//! Trace format types and constants.

use serde::{Deserialize, Serialize};

/// Trace format for serialization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceFormat {
    /// APRT binary format (compact, fast)
    ///
    /// Header: 0x41505254 ("APRT"), version byte, path type
    Binary,

    /// Pretty-printed JSON
    Json,

    /// JSON Lines (one JSON per line)
    JsonLines,
}

/// Magic bytes for APRT binary format
pub const APRT_MAGIC: [u8; 4] = [0x41, 0x50, 0x52, 0x54]; // "APRT"

/// Current format version
pub const APRT_VERSION: u8 = 1;

/// Path type identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PathType {
    Linear = 0,
    Tree = 1,
    Forest = 2,
    KNN = 3,
    Neural = 4,
    Custom = 255,
}

impl From<u8> for PathType {
    fn from(value: u8) -> Self {
        match value {
            0 => PathType::Linear,
            1 => PathType::Tree,
            2 => PathType::Forest,
            3 => PathType::KNN,
            4 => PathType::Neural,
            5..=255 => PathType::Custom,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_type_from_u8_all_variants() {
        let cases: &[(u8, PathType)] = &[
            (0, PathType::Linear),
            (1, PathType::Tree),
            (2, PathType::Forest),
            (3, PathType::KNN),
            (4, PathType::Neural),
            (5, PathType::Custom),
            (128, PathType::Custom),
            (255, PathType::Custom),
        ];

        for &(input, expected) in cases {
            let result = PathType::from(input);
            // Syntactic match covering all arms from From<u8>
            let label = match input {
                0 => PathType::Linear,
                1 => PathType::Tree,
                2 => PathType::Forest,
                3 => PathType::KNN,
                4 => PathType::Neural,
                5..=255 => PathType::Custom,
            };
            assert_eq!(result, expected);
            assert_eq!(result, label);
        }
    }

    #[test]
    fn test_trace_format_variants() {
        let _binary = TraceFormat::Binary;
        let _json = TraceFormat::Json;
        let _jsonl = TraceFormat::JsonLines;
    }

    #[test]
    fn test_aprt_constants() {
        assert_eq!(APRT_MAGIC, [0x41, 0x50, 0x52, 0x54]);
        assert_eq!(APRT_VERSION, 1);
    }
}
