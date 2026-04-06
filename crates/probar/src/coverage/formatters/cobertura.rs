//! Cobertura XML Coverage Report Formatter (Feature 13)
//!
//! Generates Cobertura XML format coverage reports for CI integration.
//!
//! ## Cobertura XML Format
//!
//! ```xml
//! <?xml version="1.0" ?>
//! <!DOCTYPE coverage SYSTEM "http://cobertura.sourceforge.net/xml/coverage-04.dtd">
//! <coverage line-rate="0.8" branch-rate="0.7" version="1.0">
//!   <packages>
//!     <package name="src" line-rate="0.8" branch-rate="0.7" complexity="0">
//!       <classes>
//!         <class name="Game" filename="src/game.rs" line-rate="0.9">
//!           <lines>
//!             <line number="10" hits="5"/>
//!           </lines>
//!         </class>
//!       </classes>
//!     </package>
//!   </packages>
//! </coverage>
//! ```

use crate::coverage::CoverageReport;
use crate::result::ProbarResult;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::Path;

/// Block coverage data: (line, hit_count, function_name)
type BlockCoverageData = Vec<(u32, u64, Option<String>)>;

/// Files grouped by path
type FileMap = BTreeMap<String, BlockCoverageData>;

/// Packages grouped by directory
type PackageMap = BTreeMap<String, FileMap>;

/// Cobertura XML format report generator
#[derive(Debug)]
pub struct CoberturaFormatter<'a> {
    report: &'a CoverageReport,
    version: String,
}

impl<'a> CoberturaFormatter<'a> {
    /// Create a new Cobertura formatter
    #[must_use]
    pub fn new(report: &'a CoverageReport) -> Self {
        Self {
            report,
            version: "1.0".to_string(),
        }
    }

    /// Set the version string
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Generate Cobertura XML report as a string
    #[must_use]
    pub fn generate(&self) -> String {
        let summary = self.report.summary();
        let files = self.group_by_file();
        let packages = Self::group_by_package(&files);

        let line_rate = summary.coverage_percent / 100.0;

        let mut xml = String::from(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
        xml.push('\n');
        xml.push_str(
            r#"<!DOCTYPE coverage SYSTEM "http://cobertura.sourceforge.net/xml/coverage-04.dtd">"#,
        );
        xml.push('\n');
        let _ = write!(
            xml,
            r#"<coverage line-rate="{:.4}" branch-rate="0" lines-covered="{}" lines-valid="{}" version="{}">"#,
            line_rate, summary.covered_blocks, summary.total_blocks, self.version,
        );
        xml.push('\n');

        xml.push_str("  <packages>\n");

        for (package_name, package_files) in &packages {
            let (pkg_covered, pkg_total) = Self::calculate_package_coverage(package_files);
            let pkg_rate = if pkg_total > 0 {
                pkg_covered as f64 / pkg_total as f64
            } else {
                1.0
            };

            let _ = write!(
                xml,
                r#"    <package name="{}" line-rate="{:.4}" branch-rate="0" complexity="0">"#,
                package_name, pkg_rate
            );
            xml.push('\n');
            xml.push_str("      <classes>\n");

            for (file_path, blocks) in package_files {
                let class_name = Self::extract_class_name(file_path);
                let (file_covered, file_total) = Self::calculate_file_coverage(blocks);
                let file_rate = if file_total > 0 {
                    file_covered as f64 / file_total as f64
                } else {
                    1.0
                };

                let _ = write!(
                    xml,
                    r#"        <class name="{}" filename="{}" line-rate="{:.4}" branch-rate="0" complexity="0">"#,
                    class_name, file_path, file_rate
                );
                xml.push('\n');
                xml.push_str("          <lines>\n");

                let lines = Self::extract_lines(blocks);
                for (line, count) in &lines {
                    let _ = write!(
                        xml,
                        r#"            <line number="{}" hits="{}"/>"#,
                        line, count
                    );
                    xml.push('\n');
                }

                xml.push_str("          </lines>\n");
                xml.push_str("        </class>\n");
            }

            xml.push_str("      </classes>\n");
            xml.push_str("    </package>\n");
        }

        xml.push_str("  </packages>\n");
        xml.push_str("</coverage>\n");

        xml
    }

    /// Save the Cobertura report to a file
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
    fn group_by_file(&self) -> FileMap {
        let mut files: FileMap = BTreeMap::new();

        for block in self.report.block_coverages() {
            let file = block.source_location.as_ref().map_or_else(
                || "unknown".to_string(),
                |loc| loc.split(':').next().unwrap_or("unknown").to_string(),
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

    /// Group files by package (directory)
    fn group_by_package(files: &FileMap) -> PackageMap {
        let mut packages: PackageMap = BTreeMap::new();

        for (file, blocks) in files {
            let package = file
                .rsplit_once('/')
                .map_or_else(|| "default".to_string(), |(dir, _)| dir.to_string());

            let _ = packages
                .entry(package)
                .or_default()
                .insert(file.clone(), blocks.clone());
        }

        packages
    }

    /// Extract class name from file path
    fn extract_class_name(file_path: &str) -> String {
        file_path
            .rsplit_once('/')
            .map_or_else(|| file_path.to_string(), |(_, name)| name.to_string())
            .trim_end_matches(".rs")
            .to_string()
    }

    /// Calculate coverage for a package
    fn calculate_package_coverage(files: &FileMap) -> (usize, usize) {
        let mut covered = 0;
        let mut total = 0;

        for blocks in files.values() {
            for (_, count, _) in blocks {
                total += 1;
                if *count > 0 {
                    covered += 1;
                }
            }
        }

        (covered, total)
    }

    /// Calculate coverage for a file
    fn calculate_file_coverage(blocks: &[(u32, u64, Option<String>)]) -> (usize, usize) {
        let total = blocks.len();
        let covered = blocks.iter().filter(|(_, count, _)| *count > 0).count();
        (covered, total)
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

        report.record_hits(BlockId::new(0), 10);
        report.record_hits(BlockId::new(1), 5);
        report.record_hits(BlockId::new(2), 0);
        report.record_hits(BlockId::new(3), 3);
        report.record_hits(BlockId::new(4), 0);

        report.set_source_location(BlockId::new(0), "src/game.rs:10");
        report.set_source_location(BlockId::new(1), "src/game.rs:15");
        report.set_source_location(BlockId::new(2), "src/game.rs:20");
        report.set_source_location(BlockId::new(3), "src/player.rs:5");
        report.set_source_location(BlockId::new(4), "src/player.rs:10");

        report.set_function_name(BlockId::new(0), "main");
        report.set_function_name(BlockId::new(1), "main");
        report.set_function_name(BlockId::new(2), "update");
        report.set_function_name(BlockId::new(3), "move_player");
        report.set_function_name(BlockId::new(4), "move_player");

        report
    }

    #[test]
    fn test_cobertura_formatter_new() {
        let report = CoverageReport::new(10);
        let formatter = CoberturaFormatter::new(&report);
        assert_eq!(formatter.version, "1.0");
    }

    #[test]
    fn test_cobertura_formatter_with_version() {
        let report = CoverageReport::new(10);
        let formatter = CoberturaFormatter::new(&report).with_version("2.0");
        assert_eq!(formatter.version, "2.0");
    }

    #[test]
    fn test_generate_xml_declaration() {
        let report = CoverageReport::new(0);
        let formatter = CoberturaFormatter::new(&report);
        let output = formatter.generate();

        assert!(output.starts_with(r#"<?xml version="1.0" encoding="UTF-8"?>"#));
    }

    #[test]
    fn test_generate_doctype() {
        let report = CoverageReport::new(0);
        let formatter = CoberturaFormatter::new(&report);
        let output = formatter.generate();

        assert!(output.contains("<!DOCTYPE coverage"));
    }

    #[test]
    fn test_generate_coverage_element() {
        let report = create_test_report();
        let formatter = CoberturaFormatter::new(&report);
        let output = formatter.generate();

        assert!(output.contains("<coverage"));
        assert!(output.contains("line-rate="));
        assert!(output.contains("version=\"1.0\""));
        assert!(output.contains("</coverage>"));
    }

    #[test]
    fn test_generate_packages() {
        let report = create_test_report();
        let formatter = CoberturaFormatter::new(&report);
        let output = formatter.generate();

        assert!(output.contains("<packages>"));
        assert!(output.contains("<package"));
        assert!(output.contains("</packages>"));
    }

    #[test]
    fn test_generate_classes() {
        let report = create_test_report();
        let formatter = CoberturaFormatter::new(&report);
        let output = formatter.generate();

        assert!(output.contains("<classes>"));
        assert!(output.contains("<class"));
        assert!(output.contains("filename="));
        assert!(output.contains("</classes>"));
    }

    #[test]
    fn test_generate_lines() {
        let report = create_test_report();
        let formatter = CoberturaFormatter::new(&report);
        let output = formatter.generate();

        assert!(output.contains("<lines>"));
        assert!(output.contains("<line number="));
        assert!(output.contains("hits="));
        assert!(output.contains("</lines>"));
    }

    #[test]
    fn test_save_creates_file() {
        let report = create_test_report();
        let formatter = CoberturaFormatter::new(&report);

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("coverage.xml");

        formatter.save(&path).unwrap();

        assert!(path.exists());
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("<?xml"));
    }

    #[test]
    fn test_extract_class_name() {
        assert_eq!(
            CoberturaFormatter::extract_class_name("src/game.rs"),
            "game"
        );
        assert_eq!(
            CoberturaFormatter::extract_class_name("src/player/movement.rs"),
            "movement"
        );
        assert_eq!(CoberturaFormatter::extract_class_name("main.rs"), "main");
    }

    #[test]
    fn test_line_rate_calculation() {
        let report = create_test_report();
        let formatter = CoberturaFormatter::new(&report);
        let output = formatter.generate();

        // 3 out of 5 blocks covered = 60% = 0.6
        assert!(output.contains("line-rate=\"0.6"));
    }

    #[test]
    fn test_package_grouping() {
        let report = create_test_report();
        let formatter = CoberturaFormatter::new(&report);
        let files = formatter.group_by_file();
        let packages = CoberturaFormatter::group_by_package(&files);

        assert!(packages.contains_key("src"));
    }

    #[test]
    fn test_calculate_file_coverage() {
        // 2 blocks covered out of 3 in game.rs
        let blocks = vec![
            (10, 10u64, Some("main".to_string())),
            (15, 5u64, Some("main".to_string())),
            (20, 0u64, Some("update".to_string())),
        ];

        let (covered, total) = CoberturaFormatter::calculate_file_coverage(&blocks);
        assert_eq!(covered, 2);
        assert_eq!(total, 3);
    }
}
