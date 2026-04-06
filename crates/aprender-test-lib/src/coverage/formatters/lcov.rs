//! LCOV Report Formatter (Feature 11)
//!
//! Generates LCOV-format coverage reports for CI integration.
//!
//! ## LCOV Format
//!
//! ```text
//! TN:<test name>
//! SF:<source file>
//! FN:<line>,<function name>
//! FNDA:<execution count>,<function name>
//! FNF:<functions found>
//! FNH:<functions hit>
//! DA:<line>,<execution count>
//! LF:<lines found>
//! LH:<lines hit>
//! end_of_record
//! ```

use crate::coverage::CoverageReport;
use crate::result::ProbarResult;
use std::collections::BTreeMap;
use std::path::Path;

/// LCOV format report generator
#[derive(Debug)]
pub struct LcovFormatter<'a> {
    report: &'a CoverageReport,
    test_name: Option<String>,
}

impl<'a> LcovFormatter<'a> {
    /// Create a new LCOV formatter from coverage data
    #[must_use]
    pub fn new(report: &'a CoverageReport) -> Self {
        Self {
            report,
            test_name: report.session_name().map(String::from),
        }
    }

    /// Set the test name for the report
    #[must_use]
    pub fn with_test_name(mut self, name: impl Into<String>) -> Self {
        self.test_name = Some(name.into());
        self
    }

    /// Generate LCOV format report as a string
    #[must_use]
    pub fn generate(&self) -> String {
        use std::fmt::Write;

        let mut output = String::new();

        // Test name (TN)
        if let Some(ref name) = self.test_name {
            let _ = writeln!(output, "TN:{name}");
        } else {
            output.push_str("TN:\n");
        }

        // Group coverage by source file
        let files = self.group_by_file();

        for (file, blocks) in &files {
            // Source file (SF)
            let _ = writeln!(output, "SF:{file}");

            // Functions (FN, FNDA)
            let functions = Self::extract_functions(blocks);
            let mut functions_hit = 0;

            for (func_name, (line, count)) in &functions {
                let _ = writeln!(output, "FN:{line},{func_name}");
                let _ = writeln!(output, "FNDA:{count},{func_name}");
                if *count > 0 {
                    functions_hit += 1;
                }
            }

            // Functions summary
            let _ = writeln!(output, "FNF:{}", functions.len());
            let _ = writeln!(output, "FNH:{functions_hit}");

            // Line data (DA)
            let lines = Self::extract_lines(blocks);
            let mut lines_hit = 0;

            for (line, count) in &lines {
                let _ = writeln!(output, "DA:{line},{count}");
                if *count > 0 {
                    lines_hit += 1;
                }
            }

            // Lines summary
            let _ = writeln!(output, "LF:{}", lines.len());
            let _ = writeln!(output, "LH:{lines_hit}");

            output.push_str("end_of_record\n");
        }

        output
    }

    /// Save the LCOV report to a file
    ///
    /// # Errors
    ///
    /// Returns error if file write fails
    pub fn save(&self, path: &Path) -> ProbarResult<()> {
        let content = self.generate();
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Group coverage data by source file
    fn group_by_file(&self) -> BTreeMap<String, Vec<(u32, u64, Option<String>)>> {
        let mut files: BTreeMap<String, Vec<(u32, u64, Option<String>)>> = BTreeMap::new();

        for block in self.report.block_coverages() {
            let file = block.source_location.as_ref().map_or_else(
                || "unknown".to_string(),
                |loc| {
                    // Extract file from "file:line" format
                    loc.split(':').next().unwrap_or("unknown").to_string()
                },
            );

            let line = block.source_location.as_ref().map_or(0, |loc| {
                loc.split(':')
                    .nth(1)
                    .and_then(|l| l.parse().ok())
                    .unwrap_or(0)
            });

            files
                .entry(file)
                .or_default()
                .push((line, block.hit_count, block.function_name));
        }

        files
    }

    /// Extract function coverage from blocks
    fn extract_functions(blocks: &[(u32, u64, Option<String>)]) -> BTreeMap<String, (u32, u64)> {
        let mut functions = BTreeMap::new();

        for (line, count, func_name) in blocks {
            if let Some(ref name) = func_name {
                let entry = functions.entry(name.clone()).or_insert((*line, 0));
                entry.1 += count;
            }
        }

        functions
    }

    /// Extract line coverage from blocks
    fn extract_lines(blocks: &[(u32, u64, Option<String>)]) -> BTreeMap<u32, u64> {
        let mut lines = BTreeMap::new();

        for (line, count, _) in blocks {
            if *line > 0 {
                *lines.entry(*line).or_insert(0) += count;
            }
        }

        lines
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::coverage::BlockId;

    fn create_test_report() -> CoverageReport {
        let mut report = CoverageReport::new(5);
        report.set_session_name("test_session");

        // Set up some blocks with coverage
        report.record_hits(BlockId::new(0), 10);
        report.record_hits(BlockId::new(1), 5);
        report.record_hits(BlockId::new(2), 0);
        report.record_hits(BlockId::new(3), 3);
        report.record_hits(BlockId::new(4), 0);

        // Set source locations
        report.set_source_location(BlockId::new(0), "src/game.rs:10");
        report.set_source_location(BlockId::new(1), "src/game.rs:15");
        report.set_source_location(BlockId::new(2), "src/game.rs:20");
        report.set_source_location(BlockId::new(3), "src/player.rs:5");
        report.set_source_location(BlockId::new(4), "src/player.rs:10");

        // Set function names
        report.set_function_name(BlockId::new(0), "main");
        report.set_function_name(BlockId::new(1), "main");
        report.set_function_name(BlockId::new(2), "update");
        report.set_function_name(BlockId::new(3), "move_player");
        report.set_function_name(BlockId::new(4), "move_player");

        report
    }

    #[test]
    fn test_lcov_formatter_new() {
        let report = CoverageReport::new(10);
        let formatter = LcovFormatter::new(&report);
        assert!(formatter.test_name.is_none());
    }

    #[test]
    fn test_lcov_formatter_with_test_name() {
        let report = CoverageReport::new(10);
        let formatter = LcovFormatter::new(&report).with_test_name("my_test");
        assert_eq!(formatter.test_name, Some("my_test".to_string()));
    }

    #[test]
    fn test_generate_empty_report() {
        let report = CoverageReport::new(0);
        let formatter = LcovFormatter::new(&report);
        let output = formatter.generate();

        assert!(output.contains("TN:"));
    }

    #[test]
    fn test_generate_with_test_name() {
        let report = create_test_report();
        let formatter = LcovFormatter::new(&report);
        let output = formatter.generate();

        assert!(output.contains("TN:test_session"));
    }

    #[test]
    fn test_generate_contains_source_files() {
        let report = create_test_report();
        let formatter = LcovFormatter::new(&report);
        let output = formatter.generate();

        assert!(output.contains("SF:src/game.rs"));
        assert!(output.contains("SF:src/player.rs"));
    }

    #[test]
    fn test_generate_contains_functions() {
        let report = create_test_report();
        let formatter = LcovFormatter::new(&report);
        let output = formatter.generate();

        assert!(output.contains("FN:"));
        assert!(output.contains("FNDA:"));
        assert!(output.contains("FNF:"));
        assert!(output.contains("FNH:"));
    }

    #[test]
    fn test_generate_contains_line_data() {
        let report = create_test_report();
        let formatter = LcovFormatter::new(&report);
        let output = formatter.generate();

        assert!(output.contains("DA:"));
        assert!(output.contains("LF:"));
        assert!(output.contains("LH:"));
    }

    #[test]
    fn test_generate_contains_end_of_record() {
        let report = create_test_report();
        let formatter = LcovFormatter::new(&report);
        let output = formatter.generate();

        assert!(output.contains("end_of_record"));
    }

    #[test]
    fn test_generate_line_hit_counts() {
        let report = create_test_report();
        let formatter = LcovFormatter::new(&report);
        let output = formatter.generate();

        // Line 10 in game.rs should have 10 hits
        assert!(output.contains("DA:10,10"));
        // Line 15 in game.rs should have 5 hits
        assert!(output.contains("DA:15,5"));
    }

    #[test]
    fn test_save_creates_file() {
        let report = create_test_report();
        let formatter = LcovFormatter::new(&report);

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("coverage.lcov");

        formatter.save(&path).unwrap();

        assert!(path.exists());
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("TN:"));
    }

    #[test]
    fn test_group_by_file() {
        let report = create_test_report();
        let formatter = LcovFormatter::new(&report);
        let files = formatter.group_by_file();

        assert!(files.contains_key("src/game.rs"));
        assert!(files.contains_key("src/player.rs"));
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_custom_test_name_overrides_session() {
        let report = create_test_report();
        let formatter = LcovFormatter::new(&report).with_test_name("custom_name");
        let output = formatter.generate();

        assert!(output.contains("TN:custom_name"));
        assert!(!output.contains("TN:test_session"));
    }
}
