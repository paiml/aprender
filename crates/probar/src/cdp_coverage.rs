//! CDP Profiler-based Code Coverage (Issue #10)
//!
//! Provides line-level coverage tracking for browser-executed code (JS/WASM)
//! using Chrome DevTools Protocol's Profiler domain.
//!
//! ## Usage
//!
//! ```ignore
//! // Enable coverage collection
//! page.start_coverage().await?;
//!
//! // Navigate and interact
//! page.goto("http://localhost:8080/demo.html").await?;
//! page.click("#start_button").await?;
//!
//! // Get coverage data
//! let coverage = page.take_coverage().await?;
//! println!("Functions covered: {}", coverage.functions_covered());
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Coverage configuration
#[derive(Debug, Clone)]
pub struct CoverageConfig {
    /// Include call counts for each function
    pub call_count: bool,
    /// Include detailed range information
    pub detailed: bool,
    /// Allow coverage to be collected multiple times
    pub allow_triggered_updates: bool,
}

impl Default for CoverageConfig {
    fn default() -> Self {
        Self {
            call_count: true,
            detailed: true,
            allow_triggered_updates: false,
        }
    }
}

impl CoverageConfig {
    /// Create a new coverage config with defaults
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable call count tracking
    #[must_use]
    pub const fn with_call_count(mut self, enabled: bool) -> Self {
        self.call_count = enabled;
        self
    }

    /// Enable detailed range information
    #[must_use]
    pub const fn with_detailed(mut self, enabled: bool) -> Self {
        self.detailed = enabled;
        self
    }
}

/// A range of bytes/characters in a script that was covered
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageRange {
    /// Start offset (byte position)
    pub start_offset: u32,
    /// End offset (byte position)
    pub end_offset: u32,
    /// Number of times this range was executed
    pub count: u32,
}

/// Coverage data for a single function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCoverage {
    /// Function name (may be empty for anonymous functions)
    pub function_name: String,
    /// Ranges within this function that were covered
    pub ranges: Vec<CoverageRange>,
    /// Whether this function was called at all
    pub is_block_coverage: bool,
}

impl FunctionCoverage {
    /// Check if the function was executed at least once
    #[must_use]
    pub fn was_executed(&self) -> bool {
        self.ranges.iter().any(|r| r.count > 0)
    }

    /// Get total execution count across all ranges
    #[must_use]
    pub fn total_count(&self) -> u32 {
        self.ranges.iter().map(|r| r.count).sum()
    }

    /// Get the byte range covered
    #[must_use]
    pub fn byte_range(&self) -> Option<(u32, u32)> {
        if self.ranges.is_empty() {
            return None;
        }
        let start = self.ranges.iter().map(|r| r.start_offset).min()?;
        let end = self.ranges.iter().map(|r| r.end_offset).max()?;
        Some((start, end))
    }
}

/// Coverage data for a single script (JS file or WASM module)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptCoverage {
    /// Script ID from CDP
    pub script_id: String,
    /// Script URL
    pub url: String,
    /// Functions in this script
    pub functions: Vec<FunctionCoverage>,
}

impl ScriptCoverage {
    /// Count functions that were executed
    #[must_use]
    pub fn functions_executed(&self) -> usize {
        self.functions.iter().filter(|f| f.was_executed()).count()
    }

    /// Count total functions
    #[must_use]
    pub fn functions_total(&self) -> usize {
        self.functions.len()
    }

    /// Calculate coverage percentage
    #[must_use]
    pub fn coverage_percent(&self) -> f64 {
        if self.functions.is_empty() {
            return 100.0;
        }
        (self.functions_executed() as f64 / self.functions_total() as f64) * 100.0
    }

    /// Check if this is a WASM module
    #[must_use]
    pub fn is_wasm(&self) -> bool {
        std::path::Path::new(&self.url)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("wasm"))
            || self.url.contains("wasm")
    }
}

/// Complete coverage report from a test session
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoverageReport {
    /// Coverage data per script
    pub scripts: Vec<ScriptCoverage>,
    /// Timestamp when coverage was taken
    pub timestamp_ms: u64,
}

impl CoverageReport {
    /// Create empty report
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add script coverage
    pub fn add_script(&mut self, script: ScriptCoverage) {
        self.scripts.push(script);
    }

    /// Get total functions covered across all scripts
    #[must_use]
    pub fn functions_covered(&self) -> usize {
        self.scripts.iter().map(|s| s.functions_executed()).sum()
    }

    /// Get total functions across all scripts
    #[must_use]
    pub fn functions_total(&self) -> usize {
        self.scripts.iter().map(|s| s.functions_total()).sum()
    }

    /// Calculate overall coverage percentage
    #[must_use]
    pub fn coverage_percent(&self) -> f64 {
        let total = self.functions_total();
        if total == 0 {
            return 100.0;
        }
        (self.functions_covered() as f64 / total as f64) * 100.0
    }

    /// Get WASM-only coverage
    #[must_use]
    pub fn wasm_coverage(&self) -> WasmCoverage {
        let wasm_scripts: Vec<_> = self.scripts.iter().filter(|s| s.is_wasm()).collect();

        let functions_covered = wasm_scripts.iter().map(|s| s.functions_executed()).sum();
        let functions_total = wasm_scripts.iter().map(|s| s.functions_total()).sum();

        WasmCoverage {
            functions_covered,
            functions_total,
            scripts: wasm_scripts.into_iter().cloned().collect(),
        }
    }

    /// Get JS-only coverage (excluding WASM)
    #[must_use]
    pub fn js_coverage(&self) -> JsCoverage {
        let js_scripts: Vec<_> = self.scripts.iter().filter(|s| !s.is_wasm()).collect();

        let functions_covered = js_scripts.iter().map(|s| s.functions_executed()).sum();
        let functions_total = js_scripts.iter().map(|s| s.functions_total()).sum();

        JsCoverage {
            functions_covered,
            functions_total,
            scripts: js_scripts.into_iter().cloned().collect(),
        }
    }

    /// Filter to specific URL pattern
    #[must_use]
    pub fn filter_by_url(&self, pattern: &str) -> Self {
        Self {
            scripts: self
                .scripts
                .iter()
                .filter(|s| s.url.contains(pattern))
                .cloned()
                .collect(),
            timestamp_ms: self.timestamp_ms,
        }
    }

    /// Generate a summary string
    #[must_use]
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "Coverage: {:.1}% ({}/{})\n",
            self.coverage_percent(),
            self.functions_covered(),
            self.functions_total()
        ));

        for script in &self.scripts {
            let icon = if script.is_wasm() { "🦀" } else { "📜" };
            s.push_str(&format!(
                "  {} {} - {:.1}% ({}/{})\n",
                icon,
                script.url,
                script.coverage_percent(),
                script.functions_executed(),
                script.functions_total()
            ));
        }

        s
    }

    /// Get uncovered functions (useful for debugging)
    #[must_use]
    pub fn uncovered_functions(&self) -> Vec<(&str, &str)> {
        let mut result = Vec::new();
        for script in &self.scripts {
            for func in &script.functions {
                if !func.was_executed() && !func.function_name.is_empty() {
                    result.push((script.url.as_str(), func.function_name.as_str()));
                }
            }
        }
        result
    }

    /// Get covered functions with call counts
    #[must_use]
    pub fn covered_functions(&self) -> Vec<CoveredFunction> {
        let mut result = Vec::new();
        for script in &self.scripts {
            for func in &script.functions {
                if func.was_executed() {
                    result.push(CoveredFunction {
                        script_url: script.url.clone(),
                        function_name: func.function_name.clone(),
                        call_count: func.total_count(),
                    });
                }
            }
        }
        result
    }
}

/// A function that was covered during execution
#[derive(Debug, Clone)]
pub struct CoveredFunction {
    /// Script URL
    pub script_url: String,
    /// Function name
    pub function_name: String,
    /// Number of times called
    pub call_count: u32,
}

/// WASM-specific coverage data
#[derive(Debug, Clone)]
pub struct WasmCoverage {
    /// Functions covered in WASM modules
    pub functions_covered: usize,
    /// Total functions in WASM modules
    pub functions_total: usize,
    /// WASM scripts
    pub scripts: Vec<ScriptCoverage>,
}

impl WasmCoverage {
    /// Calculate coverage percentage
    #[must_use]
    pub fn coverage_percent(&self) -> f64 {
        if self.functions_total == 0 {
            return 100.0;
        }
        (self.functions_covered as f64 / self.functions_total as f64) * 100.0
    }
}

/// JS-specific coverage data
#[derive(Debug, Clone)]
pub struct JsCoverage {
    /// Functions covered in JS files
    pub functions_covered: usize,
    /// Total functions in JS files
    pub functions_total: usize,
    /// JS scripts
    pub scripts: Vec<ScriptCoverage>,
}

impl JsCoverage {
    /// Calculate coverage percentage
    #[must_use]
    pub fn coverage_percent(&self) -> f64 {
        if self.functions_total == 0 {
            return 100.0;
        }
        (self.functions_covered as f64 / self.functions_total as f64) * 100.0
    }
}

/// Source map entry for mapping WASM offsets to Rust source
#[derive(Debug, Clone)]
pub struct SourceMapEntry {
    /// WASM byte offset
    pub wasm_offset: u32,
    /// Source file path
    pub source_file: String,
    /// Line number (1-indexed)
    pub line: u32,
    /// Column number (0-indexed)
    pub column: u32,
}

/// Source map for WASM to Rust mapping
#[derive(Debug, Clone, Default)]
pub struct WasmSourceMap {
    /// Mappings from WASM offset to source location
    pub entries: Vec<SourceMapEntry>,
    /// Source file contents (for line extraction)
    pub sources: HashMap<String, Vec<String>>,
}

impl WasmSourceMap {
    /// Create empty source map
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up source location for a WASM offset
    #[must_use]
    pub fn lookup(&self, offset: u32) -> Option<&SourceMapEntry> {
        // Find the entry with the largest offset <= the target
        self.entries
            .iter()
            .filter(|e| e.wasm_offset <= offset)
            .max_by_key(|e| e.wasm_offset)
    }

    /// Map coverage ranges to source lines
    pub fn map_coverage(&self, coverage: &CoverageReport) -> LineCoverage {
        let mut line_coverage = LineCoverage::new();

        for script in &coverage.scripts {
            if !script.is_wasm() {
                continue;
            }

            for func in &script.functions {
                for range in &func.ranges {
                    // Map start and end offsets to source lines
                    if let Some(start_entry) = self.lookup(range.start_offset) {
                        line_coverage.mark_covered(
                            &start_entry.source_file,
                            start_entry.line,
                            range.count,
                        );
                    }
                    if let Some(end_entry) = self.lookup(range.end_offset) {
                        line_coverage.mark_covered(
                            &end_entry.source_file,
                            end_entry.line,
                            range.count,
                        );
                    }
                }
            }
        }

        line_coverage
    }
}

/// Line-level coverage data
#[derive(Debug, Clone, Default)]
pub struct LineCoverage {
    /// Coverage per file: file -> (line -> count)
    pub files: HashMap<String, HashMap<u32, u32>>,
}

impl LineCoverage {
    /// Create empty line coverage
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a line as covered
    pub fn mark_covered(&mut self, file: &str, line: u32, count: u32) {
        let file_coverage = self.files.entry(file.to_string()).or_default();
        let current = file_coverage.entry(line).or_insert(0);
        *current = (*current).saturating_add(count);
    }

    /// Check if a line was covered
    #[must_use]
    pub fn is_covered(&self, file: &str, line: u32) -> bool {
        self.files
            .get(file)
            .and_then(|f| f.get(&line))
            .is_some_and(|&count| count > 0)
    }

    /// Get coverage count for a line
    #[must_use]
    pub fn get_count(&self, file: &str, line: u32) -> u32 {
        self.files
            .get(file)
            .and_then(|f| f.get(&line))
            .copied()
            .unwrap_or(0)
    }

    /// Get covered lines for a file
    #[must_use]
    pub fn covered_lines(&self, file: &str) -> Vec<u32> {
        self.files
            .get(file)
            .map(|f| f.keys().copied().collect())
            .unwrap_or_default()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_coverage_config_default() {
        let config = CoverageConfig::default();
        assert!(config.call_count);
        assert!(config.detailed);
        assert!(!config.allow_triggered_updates);
    }

    #[test]
    fn test_coverage_config_builder() {
        let config = CoverageConfig::new()
            .with_call_count(false)
            .with_detailed(true);
        assert!(!config.call_count);
        assert!(config.detailed);
    }

    #[test]
    fn test_coverage_range() {
        let range = CoverageRange {
            start_offset: 0,
            end_offset: 100,
            count: 5,
        };
        assert_eq!(range.count, 5);
    }

    #[test]
    fn test_function_coverage_executed() {
        let func = FunctionCoverage {
            function_name: "test_fn".to_string(),
            ranges: vec![CoverageRange {
                start_offset: 0,
                end_offset: 50,
                count: 3,
            }],
            is_block_coverage: false,
        };
        assert!(func.was_executed());
        assert_eq!(func.total_count(), 3);
    }

    #[test]
    fn test_function_coverage_not_executed() {
        let func = FunctionCoverage {
            function_name: "unused_fn".to_string(),
            ranges: vec![CoverageRange {
                start_offset: 0,
                end_offset: 50,
                count: 0,
            }],
            is_block_coverage: false,
        };
        assert!(!func.was_executed());
        assert_eq!(func.total_count(), 0);
    }

    #[test]
    fn test_function_byte_range() {
        let func = FunctionCoverage {
            function_name: "test".to_string(),
            ranges: vec![
                CoverageRange {
                    start_offset: 10,
                    end_offset: 30,
                    count: 1,
                },
                CoverageRange {
                    start_offset: 50,
                    end_offset: 100,
                    count: 1,
                },
            ],
            is_block_coverage: false,
        };
        assert_eq!(func.byte_range(), Some((10, 100)));
    }

    #[test]
    fn test_script_coverage() {
        let script = ScriptCoverage {
            script_id: "1".to_string(),
            url: "http://localhost/test.js".to_string(),
            functions: vec![
                FunctionCoverage {
                    function_name: "covered".to_string(),
                    ranges: vec![CoverageRange {
                        start_offset: 0,
                        end_offset: 50,
                        count: 1,
                    }],
                    is_block_coverage: false,
                },
                FunctionCoverage {
                    function_name: "uncovered".to_string(),
                    ranges: vec![CoverageRange {
                        start_offset: 50,
                        end_offset: 100,
                        count: 0,
                    }],
                    is_block_coverage: false,
                },
            ],
        };
        assert_eq!(script.functions_executed(), 1);
        assert_eq!(script.functions_total(), 2);
        assert!((script.coverage_percent() - 50.0).abs() < 0.01);
        assert!(!script.is_wasm());
    }

    #[test]
    fn test_script_is_wasm() {
        let wasm_script = ScriptCoverage {
            script_id: "1".to_string(),
            url: "http://localhost/app.wasm".to_string(),
            functions: vec![],
        };
        assert!(wasm_script.is_wasm());

        let js_script = ScriptCoverage {
            script_id: "2".to_string(),
            url: "http://localhost/app.js".to_string(),
            functions: vec![],
        };
        assert!(!js_script.is_wasm());
    }

    #[test]
    fn test_coverage_report_summary() {
        let mut report = CoverageReport::new();
        report.add_script(ScriptCoverage {
            script_id: "1".to_string(),
            url: "http://localhost/app.wasm".to_string(),
            functions: vec![FunctionCoverage {
                function_name: "main".to_string(),
                ranges: vec![CoverageRange {
                    start_offset: 0,
                    end_offset: 100,
                    count: 1,
                }],
                is_block_coverage: false,
            }],
        });

        assert_eq!(report.functions_covered(), 1);
        assert_eq!(report.functions_total(), 1);
        assert!((report.coverage_percent() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_coverage_report_wasm_only() {
        let mut report = CoverageReport::new();
        report.add_script(ScriptCoverage {
            script_id: "1".to_string(),
            url: "http://localhost/app.wasm".to_string(),
            functions: vec![FunctionCoverage {
                function_name: "wasm_fn".to_string(),
                ranges: vec![CoverageRange {
                    start_offset: 0,
                    end_offset: 100,
                    count: 1,
                }],
                is_block_coverage: false,
            }],
        });
        report.add_script(ScriptCoverage {
            script_id: "2".to_string(),
            url: "http://localhost/app.js".to_string(),
            functions: vec![FunctionCoverage {
                function_name: "js_fn".to_string(),
                ranges: vec![CoverageRange {
                    start_offset: 0,
                    end_offset: 50,
                    count: 1,
                }],
                is_block_coverage: false,
            }],
        });

        let wasm = report.wasm_coverage();
        assert_eq!(wasm.functions_covered, 1);
        assert_eq!(wasm.functions_total, 1);

        let js = report.js_coverage();
        assert_eq!(js.functions_covered, 1);
        assert_eq!(js.functions_total, 1);
    }

    #[test]
    fn test_coverage_report_filter() {
        let mut report = CoverageReport::new();
        report.add_script(ScriptCoverage {
            script_id: "1".to_string(),
            url: "http://localhost/myapp.wasm".to_string(),
            functions: vec![],
        });
        report.add_script(ScriptCoverage {
            script_id: "2".to_string(),
            url: "http://localhost/vendor.js".to_string(),
            functions: vec![],
        });

        let filtered = report.filter_by_url("myapp");
        assert_eq!(filtered.scripts.len(), 1);
        assert!(filtered.scripts[0].url.contains("myapp"));
    }

    #[test]
    fn test_uncovered_functions() {
        let mut report = CoverageReport::new();
        report.add_script(ScriptCoverage {
            script_id: "1".to_string(),
            url: "test.js".to_string(),
            functions: vec![
                FunctionCoverage {
                    function_name: "covered".to_string(),
                    ranges: vec![CoverageRange {
                        start_offset: 0,
                        end_offset: 50,
                        count: 1,
                    }],
                    is_block_coverage: false,
                },
                FunctionCoverage {
                    function_name: "uncovered".to_string(),
                    ranges: vec![CoverageRange {
                        start_offset: 50,
                        end_offset: 100,
                        count: 0,
                    }],
                    is_block_coverage: false,
                },
            ],
        });

        let uncovered = report.uncovered_functions();
        assert_eq!(uncovered.len(), 1);
        assert_eq!(uncovered[0].1, "uncovered");
    }

    #[test]
    fn test_covered_functions() {
        let mut report = CoverageReport::new();
        report.add_script(ScriptCoverage {
            script_id: "1".to_string(),
            url: "test.js".to_string(),
            functions: vec![FunctionCoverage {
                function_name: "my_fn".to_string(),
                ranges: vec![CoverageRange {
                    start_offset: 0,
                    end_offset: 50,
                    count: 5,
                }],
                is_block_coverage: false,
            }],
        });

        let covered = report.covered_functions();
        assert_eq!(covered.len(), 1);
        assert_eq!(covered[0].function_name, "my_fn");
        assert_eq!(covered[0].call_count, 5);
    }

    #[test]
    fn test_line_coverage() {
        let mut lc = LineCoverage::new();
        lc.mark_covered("src/lib.rs", 10, 1);
        lc.mark_covered("src/lib.rs", 10, 2);
        lc.mark_covered("src/lib.rs", 20, 1);

        assert!(lc.is_covered("src/lib.rs", 10));
        assert!(lc.is_covered("src/lib.rs", 20));
        assert!(!lc.is_covered("src/lib.rs", 30));
        assert_eq!(lc.get_count("src/lib.rs", 10), 3);
    }

    #[test]
    fn test_wasm_source_map_lookup() {
        let mut sm = WasmSourceMap::new();
        sm.entries.push(SourceMapEntry {
            wasm_offset: 0,
            source_file: "src/lib.rs".to_string(),
            line: 1,
            column: 0,
        });
        sm.entries.push(SourceMapEntry {
            wasm_offset: 100,
            source_file: "src/lib.rs".to_string(),
            line: 10,
            column: 0,
        });

        let entry = sm.lookup(50).unwrap();
        assert_eq!(entry.line, 1);

        let entry = sm.lookup(150).unwrap();
        assert_eq!(entry.line, 10);
    }

    // =========================================================================
    // Additional tests for 95% coverage
    // =========================================================================

    #[test]
    fn test_function_byte_range_empty() {
        let func = FunctionCoverage {
            function_name: "empty".to_string(),
            ranges: vec![],
            is_block_coverage: false,
        };
        assert!(func.byte_range().is_none());
    }

    #[test]
    fn test_function_byte_range_single_range() {
        let func = FunctionCoverage {
            function_name: "single".to_string(),
            ranges: vec![CoverageRange {
                start_offset: 10,
                end_offset: 50,
                count: 1,
            }],
            is_block_coverage: false,
        };
        assert_eq!(func.byte_range(), Some((10, 50)));
    }

    #[test]
    fn test_script_coverage_percent_empty_functions() {
        let script = ScriptCoverage {
            script_id: "1".to_string(),
            url: "test.js".to_string(),
            functions: vec![],
        };
        // Empty functions should return 100%
        assert!((script.coverage_percent() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_coverage_report_coverage_percent_zero_total() {
        let report = CoverageReport::new();
        // Empty report should return 100%
        assert!((report.coverage_percent() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_wasm_coverage_percent_zero_total() {
        let wasm = WasmCoverage {
            functions_covered: 0,
            functions_total: 0,
            scripts: vec![],
        };
        // Zero total should return 100%
        assert!((wasm.coverage_percent() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_js_coverage_percent_zero_total() {
        let js = JsCoverage {
            functions_covered: 0,
            functions_total: 0,
            scripts: vec![],
        };
        // Zero total should return 100%
        assert!((js.coverage_percent() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_wasm_coverage_percent_partial() {
        let wasm = WasmCoverage {
            functions_covered: 3,
            functions_total: 10,
            scripts: vec![],
        };
        assert!((wasm.coverage_percent() - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_js_coverage_percent_partial() {
        let js = JsCoverage {
            functions_covered: 7,
            functions_total: 10,
            scripts: vec![],
        };
        assert!((js.coverage_percent() - 70.0).abs() < 0.01);
    }

    #[test]
    fn test_line_coverage_covered_lines() {
        let mut lc = LineCoverage::new();
        lc.mark_covered("src/lib.rs", 10, 1);
        lc.mark_covered("src/lib.rs", 20, 2);
        lc.mark_covered("src/lib.rs", 30, 3);

        let lines = lc.covered_lines("src/lib.rs");
        assert_eq!(lines.len(), 3);
        assert!(lines.contains(&10));
        assert!(lines.contains(&20));
        assert!(lines.contains(&30));
    }

    #[test]
    fn test_line_coverage_covered_lines_nonexistent_file() {
        let lc = LineCoverage::new();
        let lines = lc.covered_lines("nonexistent.rs");
        assert!(lines.is_empty());
    }

    #[test]
    fn test_line_coverage_get_count_nonexistent() {
        let lc = LineCoverage::new();
        assert_eq!(lc.get_count("src/lib.rs", 10), 0);
    }

    #[test]
    fn test_line_coverage_is_covered_false() {
        let lc = LineCoverage::new();
        assert!(!lc.is_covered("src/lib.rs", 10));
    }

    #[test]
    fn test_coverage_report_summary_format() {
        let mut report = CoverageReport::new();
        report.add_script(ScriptCoverage {
            script_id: "1".to_string(),
            url: "http://localhost/app.wasm".to_string(),
            functions: vec![
                FunctionCoverage {
                    function_name: "covered".to_string(),
                    ranges: vec![CoverageRange {
                        start_offset: 0,
                        end_offset: 50,
                        count: 1,
                    }],
                    is_block_coverage: false,
                },
                FunctionCoverage {
                    function_name: "uncovered".to_string(),
                    ranges: vec![CoverageRange {
                        start_offset: 50,
                        end_offset: 100,
                        count: 0,
                    }],
                    is_block_coverage: false,
                },
            ],
        });

        let summary = report.summary();
        assert!(summary.contains("Coverage:"));
        assert!(summary.contains("50.0%"));
        assert!(summary.contains("1/2"));
    }

    #[test]
    fn test_wasm_source_map_map_coverage() {
        let mut sm = WasmSourceMap::new();
        sm.entries.push(SourceMapEntry {
            wasm_offset: 0,
            source_file: "src/lib.rs".to_string(),
            line: 1,
            column: 0,
        });
        sm.entries.push(SourceMapEntry {
            wasm_offset: 100,
            source_file: "src/lib.rs".to_string(),
            line: 10,
            column: 0,
        });

        let mut report = CoverageReport::new();
        report.add_script(ScriptCoverage {
            script_id: "1".to_string(),
            url: "http://localhost/app.wasm".to_string(),
            functions: vec![FunctionCoverage {
                function_name: "test_fn".to_string(),
                ranges: vec![CoverageRange {
                    start_offset: 50,
                    end_offset: 150,
                    count: 3,
                }],
                is_block_coverage: false,
            }],
        });

        let line_coverage = sm.map_coverage(&report);
        // Start offset 50 maps to line 1, end offset 150 maps to line 10
        assert!(line_coverage.is_covered("src/lib.rs", 1));
        assert!(line_coverage.is_covered("src/lib.rs", 10));
    }

    #[test]
    fn test_wasm_source_map_map_coverage_skips_js() {
        let sm = WasmSourceMap::new();

        let mut report = CoverageReport::new();
        report.add_script(ScriptCoverage {
            script_id: "1".to_string(),
            url: "http://localhost/app.js".to_string(), // Not WASM
            functions: vec![FunctionCoverage {
                function_name: "js_fn".to_string(),
                ranges: vec![CoverageRange {
                    start_offset: 0,
                    end_offset: 100,
                    count: 1,
                }],
                is_block_coverage: false,
            }],
        });

        let line_coverage = sm.map_coverage(&report);
        // Should not map JS files
        assert!(line_coverage.files.is_empty());
    }

    #[test]
    fn test_wasm_source_map_lookup_no_match() {
        let sm = WasmSourceMap::new();
        // Empty source map should return None
        assert!(sm.lookup(50).is_none());
    }

    #[test]
    fn test_wasm_source_map_lookup_exact_match() {
        let mut sm = WasmSourceMap::new();
        sm.entries.push(SourceMapEntry {
            wasm_offset: 100,
            source_file: "src/lib.rs".to_string(),
            line: 10,
            column: 5,
        });

        let entry = sm.lookup(100).unwrap();
        assert_eq!(entry.wasm_offset, 100);
        assert_eq!(entry.line, 10);
    }

    #[test]
    fn test_script_is_wasm_case_insensitive() {
        let wasm_script = ScriptCoverage {
            script_id: "1".to_string(),
            url: "http://localhost/app.WASM".to_string(),
            functions: vec![],
        };
        assert!(wasm_script.is_wasm());
    }

    #[test]
    fn test_script_is_wasm_url_contains() {
        let wasm_script = ScriptCoverage {
            script_id: "1".to_string(),
            url: "http://localhost/wasm/module".to_string(),
            functions: vec![],
        };
        assert!(wasm_script.is_wasm());
    }

    #[test]
    fn test_coverage_config_new() {
        let config = CoverageConfig::new();
        assert!(config.call_count);
        assert!(config.detailed);
        assert!(!config.allow_triggered_updates);
    }

    #[test]
    fn test_coverage_report_timestamp() {
        let mut report = CoverageReport::new();
        report.timestamp_ms = 1234567890;
        assert_eq!(report.timestamp_ms, 1234567890);
    }

    #[test]
    fn test_coverage_report_filter_by_url_no_match() {
        let mut report = CoverageReport::new();
        report.add_script(ScriptCoverage {
            script_id: "1".to_string(),
            url: "http://localhost/app.js".to_string(),
            functions: vec![],
        });

        let filtered = report.filter_by_url("nonexistent");
        assert!(filtered.scripts.is_empty());
    }

    #[test]
    fn test_uncovered_functions_anonymous() {
        let mut report = CoverageReport::new();
        report.add_script(ScriptCoverage {
            script_id: "1".to_string(),
            url: "test.js".to_string(),
            functions: vec![
                FunctionCoverage {
                    function_name: String::new(), // Anonymous
                    ranges: vec![CoverageRange {
                        start_offset: 0,
                        end_offset: 50,
                        count: 0, // Not executed
                    }],
                    is_block_coverage: false,
                },
                FunctionCoverage {
                    function_name: "named".to_string(),
                    ranges: vec![CoverageRange {
                        start_offset: 50,
                        end_offset: 100,
                        count: 0, // Not executed
                    }],
                    is_block_coverage: false,
                },
            ],
        });

        let uncovered = report.uncovered_functions();
        // Should only include named function
        assert_eq!(uncovered.len(), 1);
        assert_eq!(uncovered[0].1, "named");
    }

    #[test]
    fn test_covered_function_struct() {
        let func = CoveredFunction {
            script_url: "test.js".to_string(),
            function_name: "my_func".to_string(),
            call_count: 5,
        };
        assert_eq!(func.script_url, "test.js");
        assert_eq!(func.function_name, "my_func");
        assert_eq!(func.call_count, 5);
    }

    #[test]
    fn test_source_map_entry_fields() {
        let entry = SourceMapEntry {
            wasm_offset: 100,
            source_file: "src/main.rs".to_string(),
            line: 42,
            column: 8,
        };
        assert_eq!(entry.wasm_offset, 100);
        assert_eq!(entry.source_file, "src/main.rs");
        assert_eq!(entry.line, 42);
        assert_eq!(entry.column, 8);
    }

    #[test]
    fn test_line_coverage_mark_covered_saturating() {
        let mut lc = LineCoverage::new();
        // Test saturating_add doesn't overflow
        lc.mark_covered("src/lib.rs", 10, u32::MAX);
        lc.mark_covered("src/lib.rs", 10, 1);
        assert_eq!(lc.get_count("src/lib.rs", 10), u32::MAX);
    }

    #[test]
    fn test_function_coverage_total_count_multiple_ranges() {
        let func = FunctionCoverage {
            function_name: "multi".to_string(),
            ranges: vec![
                CoverageRange {
                    start_offset: 0,
                    end_offset: 10,
                    count: 5,
                },
                CoverageRange {
                    start_offset: 10,
                    end_offset: 20,
                    count: 3,
                },
                CoverageRange {
                    start_offset: 20,
                    end_offset: 30,
                    count: 2,
                },
            ],
            is_block_coverage: true,
        };
        assert_eq!(func.total_count(), 10);
        assert!(func.is_block_coverage);
    }

    #[test]
    fn test_wasm_source_map_sources_field() {
        let mut sm = WasmSourceMap::new();
        sm.sources
            .insert("src/lib.rs".to_string(), vec!["fn main() {}".to_string()]);

        assert!(sm.sources.contains_key("src/lib.rs"));
        assert_eq!(sm.sources["src/lib.rs"].len(), 1);
    }

    #[test]
    fn test_coverage_report_serialize_deserialize() {
        let mut report = CoverageReport::new();
        report.timestamp_ms = 1000;
        report.add_script(ScriptCoverage {
            script_id: "1".to_string(),
            url: "test.js".to_string(),
            functions: vec![FunctionCoverage {
                function_name: "test".to_string(),
                ranges: vec![CoverageRange {
                    start_offset: 0,
                    end_offset: 50,
                    count: 1,
                }],
                is_block_coverage: false,
            }],
        });

        let json = serde_json::to_string(&report).unwrap();
        let deserialized: CoverageReport = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.timestamp_ms, 1000);
        assert_eq!(deserialized.scripts.len(), 1);
        assert_eq!(deserialized.functions_covered(), 1);
    }

    #[test]
    fn test_line_coverage_default() {
        let lc = LineCoverage::default();
        assert!(lc.files.is_empty());
    }

    #[test]
    fn test_wasm_source_map_default() {
        let sm = WasmSourceMap::default();
        assert!(sm.entries.is_empty());
        assert!(sm.sources.is_empty());
    }
}
