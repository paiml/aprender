//! JUnit XML Report Generator
//!
//! Generates JUnit-compatible XML reports for CI/CD integration.
//! Supports standard JUnit XML format for Jenkins, GitHub Actions, etc.

use apr_qa_runner::{Evidence, EvidenceCollector, Outcome};
use std::io::Write;

use crate::error::Result;
use crate::mqs::MqsScore;

/// JUnit XML report generator
#[derive(Debug)]
pub struct JunitReport {
    /// Test suite name
    suite_name: String,
    /// Test class name
    class_name: String,
}

impl JunitReport {
    /// Create a new JUnit report generator
    #[must_use]
    pub fn new(suite_name: impl Into<String>) -> Self {
        let name = suite_name.into();
        Self {
            class_name: name.clone(),
            suite_name: name,
        }
    }

    /// Set the class name for test cases
    #[must_use]
    pub fn with_class_name(mut self, class_name: impl Into<String>) -> Self {
        self.class_name = class_name.into();
        self
    }

    /// Generate JUnit XML from evidence
    ///
    /// # Errors
    ///
    /// Returns an error if XML generation fails.
    pub fn generate(&self, evidence: &EvidenceCollector, score: &MqsScore) -> Result<String> {
        let mut output = Vec::new();
        self.write_xml(&mut output, evidence, score)?;
        Ok(String::from_utf8_lossy(&output).to_string())
    }

    /// Write JUnit XML to a writer
    fn write_xml<W: Write>(
        &self,
        writer: &mut W,
        evidence: &EvidenceCollector,
        score: &MqsScore,
    ) -> Result<()> {
        let all_evidence = evidence.all();
        let tests = all_evidence.len();
        // JUnit spec: failures = assertion failures, errors = crashes/timeouts
        // Timeout and Crashed render as <error>, Falsified renders as <failure>
        let failures = all_evidence
            .iter()
            .filter(|e| e.outcome == Outcome::Falsified)
            .count();
        let errors = all_evidence
            .iter()
            .filter(|e| matches!(e.outcome, Outcome::Crashed | Outcome::Timeout))
            .count();
        let skipped = all_evidence
            .iter()
            .filter(|e| e.outcome == Outcome::Skipped)
            .count();
        let time: f64 = all_evidence
            .iter()
            .map(|e| e.metrics.duration_ms as f64 / 1000.0)
            .sum();

        writeln!(writer, r#"<?xml version="1.0" encoding="UTF-8"?>"#)?;
        writeln!(
            writer,
            r#"<testsuite name="{}" tests="{}" failures="{}" errors="{}" skipped="{}" time="{:.3}">"#,
            Self::escape_xml(&self.suite_name),
            tests,
            failures,
            errors,
            skipped,
            time
        )?;

        // Add properties with MQS score
        writeln!(writer, "  <properties>")?;
        writeln!(
            writer,
            r#"    <property name="mqs.raw_score" value="{}"/>"#,
            score.raw_score
        )?;
        writeln!(
            writer,
            r#"    <property name="mqs.normalized_score" value="{:.2}"/>"#,
            score.normalized_score
        )?;
        writeln!(
            writer,
            r#"    <property name="mqs.grade" value="{}"/>"#,
            Self::escape_xml(&score.grade)
        )?;
        writeln!(
            writer,
            r#"    <property name="mqs.gateways_passed" value="{}"/>"#,
            score.gateways_passed
        )?;
        writeln!(writer, "  </properties>")?;

        // Write test cases
        for e in all_evidence {
            self.write_testcase(writer, e)?;
        }

        writeln!(writer, "</testsuite>")?;
        Ok(())
    }

    /// Write a single test case
    fn write_testcase<W: Write>(&self, writer: &mut W, evidence: &Evidence) -> Result<()> {
        let test_name = format!(
            "{}_{}_{}",
            evidence.scenario.modality, evidence.scenario.backend, evidence.gate_id
        );
        let time = evidence.metrics.duration_ms as f64 / 1000.0;

        writeln!(
            writer,
            r#"  <testcase classname="{}" name="{}" time="{:.3}">"#,
            Self::escape_xml(&self.class_name),
            Self::escape_xml(&test_name),
            time
        )?;

        match evidence.outcome {
            Outcome::Corroborated => {
                // Success - no inner elements needed
            }
            Outcome::Falsified => {
                writeln!(
                    writer,
                    r#"    <failure message="{}" type="AssertionError">"#,
                    Self::escape_xml(&evidence.reason)
                )?;
                writeln!(writer, "Gate: {}", Self::escape_xml(&evidence.gate_id))?;
                writeln!(writer, "Output: {}", Self::escape_xml(&evidence.output))?;
                writeln!(writer, "    </failure>")?;
            }
            Outcome::Crashed => {
                writeln!(
                    writer,
                    r#"    <error message="{}" type="CrashError">"#,
                    Self::escape_xml(&evidence.reason)
                )?;
                if let Some(ref stderr) = evidence.stderr {
                    writeln!(writer, "{}", Self::escape_xml(stderr))?;
                }
                writeln!(writer, "    </error>")?;
            }
            Outcome::Timeout => {
                writeln!(
                    writer,
                    r#"    <error message="Timeout after {}ms" type="TimeoutError"/>"#,
                    evidence.metrics.duration_ms
                )?;
            }
            Outcome::Skipped => {
                writeln!(writer, r#"    <skipped message="Test skipped"/>"#)?;
            }
        }

        writeln!(writer, "  </testcase>")?;
        Ok(())
    }

    /// Escape XML special characters and strip control characters forbidden in XML 1.0.
    ///
    /// XML 1.0 §2.2 allows only: #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD].
    /// All other control characters (0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F) are stripped to
    /// prevent CI parsers (Jenkins, GitHub Actions) from rejecting the entire report.
    fn escape_xml(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '&' => result.push_str("&amp;"),
                '<' => result.push_str("&lt;"),
                '>' => result.push_str("&gt;"),
                '"' => result.push_str("&quot;"),
                '\'' => result.push_str("&apos;"),
                '\t' | '\n' | '\r' => result.push(c),
                c if c < '\u{20}' => {} // strip forbidden control chars
                _ => result.push(c),
            }
        }
        result
    }
}

impl Default for JunitReport {
    fn default() -> Self {
        Self::new("apr-qa-report")
    }
}

#[cfg(test)]
#[path = "junit_tests.rs"]
mod junit_tests;
