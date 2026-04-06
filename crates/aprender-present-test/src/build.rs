//! Build validation module for WASM bundle size and quality checks.
//!
//! Provides tools for validating WASM bundles meet size and quality requirements.

use crate::grade::{GateCheckResult, GateViolation, QualityGates, ViolationSeverity};
use std::path::Path;

// =============================================================================
// BuildInfo - TESTS FIRST
// =============================================================================

/// Information about a built WASM bundle.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BuildInfo {
    /// Path to the WASM file
    pub wasm_path: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// Estimated Brotli compressed size in bytes
    pub compressed_size_bytes: u64,
    /// Build mode (debug/release)
    pub mode: BuildMode,
    /// Target platform
    pub target: String,
    /// Build timestamp (Unix epoch seconds)
    pub timestamp: u64,
}

/// Build mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BuildMode {
    #[default]
    Debug,
    Release,
}

impl std::fmt::Display for BuildMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Debug => write!(f, "debug"),
            Self::Release => write!(f, "release"),
        }
    }
}

impl BuildInfo {
    /// Size in KB (rounded up).
    #[must_use]
    pub fn size_kb(&self) -> u32 {
        ((self.size_bytes + 1023) / 1024) as u32
    }

    /// Compressed size in KB (rounded up).
    #[must_use]
    pub fn compressed_size_kb(&self) -> u32 {
        ((self.compressed_size_bytes + 1023) / 1024) as u32
    }

    /// Compression ratio (0.0 to 1.0).
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.size_bytes == 0 {
            return 0.0;
        }
        self.compressed_size_bytes as f64 / self.size_bytes as f64
    }

    /// Whether bundle meets the size limit.
    #[must_use]
    pub fn meets_size_limit(&self, limit_kb: u32) -> bool {
        self.size_kb() <= limit_kb
    }
}

// =============================================================================
// BundleAnalyzer - TESTS FIRST
// =============================================================================

/// Analyzes WASM bundle for size and content.
#[derive(Debug, Default)]
pub struct BundleAnalyzer {
    /// Maximum allowed size in KB
    pub max_size_kb: u32,
    /// Expected compression ratio for Brotli
    pub expected_compression_ratio: f64,
    /// Forbidden patterns in the bundle
    pub forbidden_patterns: Vec<String>,
}

impl BundleAnalyzer {
    /// Create a new analyzer with default limits from the spec.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_size_kb: 500,                // Per spec: <500KB
            expected_compression_ratio: 0.3, // Typical Brotli ratio for WASM
            forbidden_patterns: vec![
                "PANIC_MESSAGE".to_string(), // Debug panics
                "debug_assert".to_string(),  // Debug assertions
            ],
        }
    }

    /// Analyze a WASM file.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read.
    pub fn analyze(&self, path: &Path) -> Result<BundleAnalysis, BundleError> {
        // Read file
        let data = std::fs::read(path).map_err(|e| BundleError::IoError(e.to_string()))?;

        let size_bytes = data.len() as u64;

        // Estimate Brotli compression (real compression would require brotli crate)
        let compressed_size = self.estimate_compressed_size(&data);

        // Check for WASM magic bytes
        let is_valid_wasm = data.len() >= 4 && &data[0..4] == b"\0asm";

        // Check for forbidden patterns
        let forbidden_found = self.find_forbidden_patterns(&data);

        // Parse sections if valid WASM
        let sections = if is_valid_wasm {
            self.parse_wasm_sections(&data)
        } else {
            Vec::new()
        };

        Ok(BundleAnalysis {
            info: BuildInfo {
                wasm_path: path.display().to_string(),
                size_bytes,
                compressed_size_bytes: compressed_size,
                mode: if size_bytes > 1_000_000 {
                    BuildMode::Debug
                } else {
                    BuildMode::Release
                },
                target: "wasm32-unknown-unknown".to_string(),
                timestamp: 0,
            },
            is_valid_wasm,
            sections,
            forbidden_found,
        })
    }

    /// Analyze raw WASM bytes.
    #[must_use]
    pub fn analyze_bytes(&self, data: &[u8]) -> BundleAnalysis {
        let size_bytes = data.len() as u64;
        let compressed_size = self.estimate_compressed_size(data);
        let is_valid_wasm = data.len() >= 4 && &data[0..4] == b"\0asm";
        let forbidden_found = self.find_forbidden_patterns(data);
        let sections = if is_valid_wasm {
            self.parse_wasm_sections(data)
        } else {
            Vec::new()
        };

        BundleAnalysis {
            info: BuildInfo {
                wasm_path: "<memory>".to_string(),
                size_bytes,
                compressed_size_bytes: compressed_size,
                mode: if size_bytes > 1_000_000 {
                    BuildMode::Debug
                } else {
                    BuildMode::Release
                },
                target: "wasm32-unknown-unknown".to_string(),
                timestamp: 0,
            },
            is_valid_wasm,
            sections,
            forbidden_found,
        }
    }

    /// Estimate Brotli compressed size.
    fn estimate_compressed_size(&self, data: &[u8]) -> u64 {
        // Rough estimate: WASM typically compresses to ~30% with Brotli
        let ratio = self.expected_compression_ratio;
        (data.len() as f64 * ratio) as u64
    }

    /// Find forbidden patterns in binary data.
    fn find_forbidden_patterns(&self, data: &[u8]) -> Vec<String> {
        let mut found = Vec::new();
        let data_str = String::from_utf8_lossy(data);

        for pattern in &self.forbidden_patterns {
            if data_str.contains(pattern) {
                found.push(pattern.clone());
            }
        }

        found
    }

    /// Parse WASM section headers.
    fn parse_wasm_sections(&self, data: &[u8]) -> Vec<WasmSection> {
        let mut sections = Vec::new();

        if data.len() < 8 {
            return sections;
        }

        // Skip magic (4 bytes) and version (4 bytes)
        let mut pos = 8;

        while pos < data.len() {
            if pos >= data.len() {
                break;
            }

            let section_id = data[pos];
            pos += 1;

            // Parse LEB128 size
            let (size, bytes_read) = Self::read_leb128(&data[pos..]);
            pos += bytes_read;

            if pos + size as usize > data.len() {
                break;
            }

            sections.push(WasmSection {
                id: section_id,
                name: Self::section_name(section_id),
                size,
            });

            pos += size as usize;
        }

        sections
    }

    /// Read LEB128 encoded unsigned integer.
    fn read_leb128(data: &[u8]) -> (u64, usize) {
        let mut result = 0u64;
        let mut shift = 0;
        let mut bytes_read = 0;

        for &byte in data {
            result |= u64::from(byte & 0x7F) << shift;
            bytes_read += 1;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift >= 64 {
                break;
            }
        }

        (result, bytes_read)
    }

    /// Get section name from ID.
    fn section_name(id: u8) -> String {
        match id {
            0 => "custom".to_string(),
            1 => "type".to_string(),
            2 => "import".to_string(),
            3 => "function".to_string(),
            4 => "table".to_string(),
            5 => "memory".to_string(),
            6 => "global".to_string(),
            7 => "export".to_string(),
            8 => "start".to_string(),
            9 => "element".to_string(),
            10 => "code".to_string(),
            11 => "data".to_string(),
            12 => "data_count".to_string(),
            _ => format!("unknown_{id}"),
        }
    }

    /// Check if analysis passes quality gates.
    #[must_use]
    pub fn check_gates(&self, analysis: &BundleAnalysis, gates: &QualityGates) -> GateCheckResult {
        let mut violations = Vec::new();

        // Check bundle size
        if analysis.info.size_kb() > gates.performance.max_bundle_size_kb {
            violations.push(GateViolation {
                gate: "max_bundle_size_kb".to_string(),
                expected: format!("<= {}KB", gates.performance.max_bundle_size_kb),
                actual: format!("{}KB", analysis.info.size_kb()),
                severity: ViolationSeverity::Error,
            });
        }

        // Check for forbidden patterns
        if !analysis.forbidden_found.is_empty() {
            violations.push(GateViolation {
                gate: "forbidden_patterns".to_string(),
                expected: "no debug symbols".to_string(),
                actual: format!("found: {}", analysis.forbidden_found.join(", ")),
                severity: ViolationSeverity::Warning,
            });
        }

        // Check WASM validity
        if !analysis.is_valid_wasm {
            violations.push(GateViolation {
                gate: "wasm_validity".to_string(),
                expected: "valid WASM file".to_string(),
                actual: "invalid WASM magic bytes".to_string(),
                severity: ViolationSeverity::Error,
            });
        }

        // Check for debug build (size heuristic)
        if analysis.info.mode == BuildMode::Debug {
            violations.push(GateViolation {
                gate: "build_mode".to_string(),
                expected: "release build".to_string(),
                actual: "debug build (size > 1MB)".to_string(),
                severity: ViolationSeverity::Warning,
            });
        }

        let passed = !violations
            .iter()
            .any(|v| v.severity == ViolationSeverity::Error);

        GateCheckResult { passed, violations }
    }
}

/// Result of bundle analysis.
#[derive(Debug, Clone, Default)]
pub struct BundleAnalysis {
    /// Build info
    pub info: BuildInfo,
    /// Whether file is valid WASM
    pub is_valid_wasm: bool,
    /// WASM sections found
    pub sections: Vec<WasmSection>,
    /// Forbidden patterns found
    pub forbidden_found: Vec<String>,
}

impl BundleAnalysis {
    /// Get code section size.
    #[must_use]
    pub fn code_size(&self) -> u64 {
        self.sections
            .iter()
            .find(|s| s.name == "code")
            .map_or(0, |s| s.size)
    }

    /// Get data section size.
    #[must_use]
    pub fn data_size(&self) -> u64 {
        self.sections
            .iter()
            .find(|s| s.name == "data")
            .map_or(0, |s| s.size)
    }

    /// Get custom sections total size.
    #[must_use]
    pub fn custom_size(&self) -> u64 {
        self.sections
            .iter()
            .filter(|s| s.name == "custom")
            .map(|s| s.size)
            .sum()
    }
}

/// WASM section info.
#[derive(Debug, Clone, Default)]
pub struct WasmSection {
    /// Section ID
    pub id: u8,
    /// Section name
    pub name: String,
    /// Section size in bytes
    pub size: u64,
}

/// Build validation errors.
#[derive(Debug)]
pub enum BundleError {
    /// IO error reading file
    IoError(String),
    /// Invalid WASM format
    InvalidFormat(String),
}

impl std::fmt::Display for BundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(msg) => write!(f, "IO error: {msg}"),
            Self::InvalidFormat(msg) => write!(f, "invalid format: {msg}"),
        }
    }
}

impl std::error::Error for BundleError {}

// =============================================================================
// SizeTracker - Track bundle size over time
// =============================================================================

/// Tracks bundle size changes over time.
#[derive(Debug, Clone, Default)]
pub struct SizeTracker {
    /// Historical size records
    pub records: Vec<SizeRecord>,
    /// Baseline size for comparison
    pub baseline_kb: Option<u32>,
}

/// Record of bundle size at a point in time.
#[derive(Debug, Clone)]
pub struct SizeRecord {
    /// Timestamp (Unix epoch)
    pub timestamp: u64,
    /// Size in KB
    pub size_kb: u32,
    /// Git commit hash (if available)
    pub commit: Option<String>,
    /// Label for this record
    pub label: String,
}

impl SizeTracker {
    /// Create new tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a size record.
    pub fn record(&mut self, size_kb: u32, label: &str, commit: Option<&str>) {
        self.records.push(SizeRecord {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            size_kb,
            commit: commit.map(String::from),
            label: label.to_string(),
        });
    }

    /// Set baseline for comparison.
    pub fn set_baseline(&mut self, size_kb: u32) {
        self.baseline_kb = Some(size_kb);
    }

    /// Get size change from baseline.
    #[must_use]
    pub fn change_from_baseline(&self, current_kb: u32) -> Option<i32> {
        self.baseline_kb.map(|b| current_kb as i32 - b as i32)
    }

    /// Get size change percentage from baseline.
    #[must_use]
    pub fn change_percentage(&self, current_kb: u32) -> Option<f64> {
        self.baseline_kb.map(|b| {
            if b == 0 {
                0.0
            } else {
                (f64::from(current_kb) - f64::from(b)) / f64::from(b) * 100.0
            }
        })
    }

    /// Get the minimum size ever recorded.
    #[must_use]
    pub fn min_size(&self) -> Option<u32> {
        self.records.iter().map(|r| r.size_kb).min()
    }

    /// Get the maximum size ever recorded.
    #[must_use]
    pub fn max_size(&self) -> Option<u32> {
        self.records.iter().map(|r| r.size_kb).max()
    }

    /// Get the latest record.
    #[must_use]
    pub fn latest(&self) -> Option<&SizeRecord> {
        self.records.last()
    }
}

// =============================================================================
// Tests - TDD Style
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // BuildInfo tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_build_info_default() {
        let info = BuildInfo::default();
        assert_eq!(info.size_bytes, 0);
        assert_eq!(info.compressed_size_bytes, 0);
        assert_eq!(info.mode, BuildMode::Debug);
    }

    #[test]
    fn test_build_info_size_kb() {
        let info = BuildInfo {
            size_bytes: 1024,
            ..Default::default()
        };
        assert_eq!(info.size_kb(), 1);

        let info = BuildInfo {
            size_bytes: 1025,
            ..Default::default()
        };
        assert_eq!(info.size_kb(), 2); // Rounds up
    }

    #[test]
    fn test_build_info_compression_ratio() {
        let info = BuildInfo {
            size_bytes: 1000,
            compressed_size_bytes: 300,
            ..Default::default()
        };
        assert!((info.compression_ratio() - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_build_info_compression_ratio_zero() {
        let info = BuildInfo {
            size_bytes: 0,
            compressed_size_bytes: 0,
            ..Default::default()
        };
        assert_eq!(info.compression_ratio(), 0.0);
    }

    #[test]
    fn test_build_info_meets_size_limit() {
        let info = BuildInfo {
            size_bytes: 400 * 1024, // 400KB
            ..Default::default()
        };
        assert!(info.meets_size_limit(500));
        assert!(!info.meets_size_limit(300));
    }

    #[test]
    fn test_build_mode_display() {
        assert_eq!(BuildMode::Debug.to_string(), "debug");
        assert_eq!(BuildMode::Release.to_string(), "release");
    }

    // -------------------------------------------------------------------------
    // BundleAnalyzer tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bundle_analyzer_new() {
        let analyzer = BundleAnalyzer::new();
        assert_eq!(analyzer.max_size_kb, 500);
        assert!((analyzer.expected_compression_ratio - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_bundle_analyzer_analyze_bytes_valid_wasm() {
        // Valid WASM header
        let mut data = vec![0x00, 0x61, 0x73, 0x6D]; // \0asm
        data.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // version 1

        let analyzer = BundleAnalyzer::new();
        let analysis = analyzer.analyze_bytes(&data);

        assert!(analysis.is_valid_wasm);
        assert_eq!(analysis.info.size_bytes, 8);
    }

    #[test]
    fn test_bundle_analyzer_analyze_bytes_invalid_wasm() {
        let data = b"not wasm data";
        let analyzer = BundleAnalyzer::new();
        let analysis = analyzer.analyze_bytes(data);

        assert!(!analysis.is_valid_wasm);
    }

    #[test]
    fn test_bundle_analyzer_forbidden_patterns() {
        let mut analyzer = BundleAnalyzer::new();
        analyzer.forbidden_patterns = vec!["SECRET".to_string()];

        let data = b"some data with SECRET inside";
        let analysis = analyzer.analyze_bytes(data);

        assert!(analysis.forbidden_found.contains(&"SECRET".to_string()));
    }

    #[test]
    fn test_bundle_analyzer_compression_estimate() {
        let analyzer = BundleAnalyzer::new();
        let data = vec![0u8; 10000]; // 10KB of zeros
        let analysis = analyzer.analyze_bytes(&data);

        // Should estimate ~30% compression
        assert!(analysis.info.compressed_size_bytes < analysis.info.size_bytes);
        let ratio = analysis.info.compression_ratio();
        assert!((ratio - 0.3).abs() < 0.1);
    }

    #[test]
    fn test_bundle_analyzer_check_gates_passes() {
        let analyzer = BundleAnalyzer::new();
        let gates = QualityGates::default();

        // Small valid WASM
        let mut data = vec![0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];
        data.extend_from_slice(&vec![0u8; 1000]); // Small bundle

        let analysis = analyzer.analyze_bytes(&data);
        let result = analyzer.check_gates(&analysis, &gates);

        assert!(result.passed);
    }

    #[test]
    fn test_bundle_analyzer_check_gates_size_fails() {
        let analyzer = BundleAnalyzer::new();
        let mut gates = QualityGates::default();
        gates.performance.max_bundle_size_kb = 1; // Very small limit

        let mut data = vec![0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];
        data.extend_from_slice(&vec![0u8; 5000]); // ~5KB bundle

        let analysis = analyzer.analyze_bytes(&data);
        let result = analyzer.check_gates(&analysis, &gates);

        assert!(!result.passed);
        assert!(result
            .violations
            .iter()
            .any(|v| v.gate == "max_bundle_size_kb"));
    }

    #[test]
    fn test_bundle_analyzer_check_gates_invalid_wasm() {
        let analyzer = BundleAnalyzer::new();
        let gates = QualityGates::default();

        let data = b"not wasm at all";
        let analysis = analyzer.analyze_bytes(data);
        let result = analyzer.check_gates(&analysis, &gates);

        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.gate == "wasm_validity"));
    }

    #[test]
    fn test_bundle_analyzer_leb128() {
        // Test single byte
        let data = [0x7F];
        let (val, bytes) = BundleAnalyzer::read_leb128(&data);
        assert_eq!(val, 127);
        assert_eq!(bytes, 1);

        // Test multi-byte
        let data = [0x80, 0x01];
        let (val, bytes) = BundleAnalyzer::read_leb128(&data);
        assert_eq!(val, 128);
        assert_eq!(bytes, 2);
    }

    #[test]
    fn test_bundle_analyzer_section_names() {
        assert_eq!(BundleAnalyzer::section_name(0), "custom");
        assert_eq!(BundleAnalyzer::section_name(1), "type");
        assert_eq!(BundleAnalyzer::section_name(10), "code");
        assert_eq!(BundleAnalyzer::section_name(11), "data");
        assert_eq!(BundleAnalyzer::section_name(255), "unknown_255");
    }

    // -------------------------------------------------------------------------
    // BundleAnalysis tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bundle_analysis_code_size() {
        let analysis = BundleAnalysis {
            sections: vec![
                WasmSection {
                    id: 10,
                    name: "code".to_string(),
                    size: 5000,
                },
                WasmSection {
                    id: 11,
                    name: "data".to_string(),
                    size: 1000,
                },
            ],
            ..Default::default()
        };

        assert_eq!(analysis.code_size(), 5000);
        assert_eq!(analysis.data_size(), 1000);
    }

    #[test]
    fn test_bundle_analysis_custom_size() {
        let analysis = BundleAnalysis {
            sections: vec![
                WasmSection {
                    id: 0,
                    name: "custom".to_string(),
                    size: 100,
                },
                WasmSection {
                    id: 0,
                    name: "custom".to_string(),
                    size: 200,
                },
            ],
            ..Default::default()
        };

        assert_eq!(analysis.custom_size(), 300);
    }

    #[test]
    fn test_bundle_analysis_empty_sections() {
        let analysis = BundleAnalysis::default();
        assert_eq!(analysis.code_size(), 0);
        assert_eq!(analysis.data_size(), 0);
        assert_eq!(analysis.custom_size(), 0);
    }

    // -------------------------------------------------------------------------
    // BundleError tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bundle_error_display() {
        let io_err = BundleError::IoError("file not found".to_string());
        assert!(io_err.to_string().contains("IO error"));

        let format_err = BundleError::InvalidFormat("bad magic".to_string());
        assert!(format_err.to_string().contains("invalid format"));
    }

    // -------------------------------------------------------------------------
    // SizeTracker tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_size_tracker_new() {
        let tracker = SizeTracker::new();
        assert!(tracker.records.is_empty());
        assert!(tracker.baseline_kb.is_none());
    }

    #[test]
    fn test_size_tracker_record() {
        let mut tracker = SizeTracker::new();
        tracker.record(100, "initial", None);
        tracker.record(110, "feature-a", Some("abc123"));

        assert_eq!(tracker.records.len(), 2);
        assert_eq!(tracker.records[0].size_kb, 100);
        assert_eq!(tracker.records[1].size_kb, 110);
        assert_eq!(tracker.records[1].commit, Some("abc123".to_string()));
    }

    #[test]
    fn test_size_tracker_baseline() {
        let mut tracker = SizeTracker::new();
        tracker.set_baseline(100);

        assert_eq!(tracker.change_from_baseline(110), Some(10));
        assert_eq!(tracker.change_from_baseline(90), Some(-10));
    }

    #[test]
    fn test_size_tracker_change_percentage() {
        let mut tracker = SizeTracker::new();
        tracker.set_baseline(100);

        let pct = tracker.change_percentage(110).unwrap();
        assert!((pct - 10.0).abs() < 0.01);

        let pct = tracker.change_percentage(50).unwrap();
        assert!((pct - (-50.0)).abs() < 0.01);
    }

    #[test]
    fn test_size_tracker_min_max() {
        let mut tracker = SizeTracker::new();
        tracker.record(100, "a", None);
        tracker.record(150, "b", None);
        tracker.record(120, "c", None);

        assert_eq!(tracker.min_size(), Some(100));
        assert_eq!(tracker.max_size(), Some(150));
    }

    #[test]
    fn test_size_tracker_latest() {
        let mut tracker = SizeTracker::new();
        tracker.record(100, "first", None);
        tracker.record(200, "second", None);

        let latest = tracker.latest().unwrap();
        assert_eq!(latest.size_kb, 200);
        assert_eq!(latest.label, "second");
    }

    #[test]
    fn test_size_tracker_empty_min_max() {
        let tracker = SizeTracker::new();
        assert!(tracker.min_size().is_none());
        assert!(tracker.max_size().is_none());
        assert!(tracker.latest().is_none());
    }

    #[test]
    fn test_size_tracker_baseline_zero() {
        let mut tracker = SizeTracker::new();
        tracker.set_baseline(0);

        assert_eq!(tracker.change_percentage(100), Some(0.0));
    }
}
