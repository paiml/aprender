//! Report command handler

use crate::config::CliConfig;
use crate::{ReportArgs, ReportFormat};
use std::path::Path;

/// Execute the report command
pub fn execute_report(_config: &CliConfig, args: &ReportArgs) {
    use std::fs;
    use std::io::Write;

    println!("Generating report...");
    println!("Format: {:?}", args.format);
    println!("Output: {}", args.output.display());

    if let Some(parent) = args.output.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let report_content = match args.format {
        ReportFormat::Html => generate_html_report(),
        ReportFormat::Json => generate_json_report(),
        ReportFormat::Lcov => generate_lcov_report(),
        ReportFormat::Junit => generate_junit_report(),
        ReportFormat::Cobertura => generate_cobertura_report(),
    };

    match fs::File::create(&args.output) {
        Ok(mut file) => {
            if let Err(e) = file.write_all(report_content.as_bytes()) {
                eprintln!("Failed to write report: {e}");
                return;
            }
            println!("Report generated at: {}", args.output.display());
        }
        Err(e) => {
            eprintln!("Failed to create report file: {e}");
            return;
        }
    }

    if args.open {
        open_in_browser(&args.output);
    }
}

/// Open a file in the system's default browser
pub fn open_in_browser(path: &Path) {
    println!("Opening report in browser...");
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(path).spawn();
    #[cfg(target_os = "linux")]
    let _ = std::process::Command::new("xdg-open").arg(path).spawn();
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("start").arg(path).spawn();
}

/// Generate HTML test report
#[must_use]
pub fn generate_html_report() -> String {
    let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Probar Test Report</title>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .stat {{ background: #f9f9f9; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #4CAF50; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        .timestamp {{ color: #999; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Probar Test Report</h1>
        <p class="timestamp">Generated: {timestamp}</p>
        <div class="summary">
            <div class="stat"><div class="stat-value">0</div><div class="stat-label">Tests Run</div></div>
            <div class="stat"><div class="stat-value">0</div><div class="stat-label">Passed</div></div>
            <div class="stat"><div class="stat-value">0</div><div class="stat-label">Failed</div></div>
            <div class="stat"><div class="stat-value">0ms</div><div class="stat-label">Duration</div></div>
        </div>
        <p>Run <code>probar test</code> to generate test results.</p>
    </div>
</body>
</html>"#
    )
}

/// Generate JSON test report
#[must_use]
pub fn generate_json_report() -> String {
    let timestamp = chrono::Utc::now().to_rfc3339();
    format!(
        r#"{{
  "version": "1.0",
  "timestamp": "{timestamp}",
  "summary": {{
    "total": 0,
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "duration_ms": 0
  }},
  "tests": []
}}"#
    )
}

/// Generate LCOV coverage report
#[must_use]
pub fn generate_lcov_report() -> String {
    "TN:\nSF:src/lib.rs\nDA:1,0\nLF:1\nLH:0\nend_of_record\n".to_string()
}

/// Generate `JUnit` XML report
#[must_use]
pub fn generate_junit_report() -> String {
    let timestamp = chrono::Utc::now().to_rfc3339();
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="probar" tests="0" failures="0" errors="0" time="0" timestamp="{timestamp}">
  <testsuite name="probar" tests="0" failures="0" errors="0" time="0">
  </testsuite>
</testsuites>"#
    )
}

/// Generate Cobertura XML report
#[must_use]
pub fn generate_cobertura_report() -> String {
    let timestamp = chrono::Utc::now().timestamp();
    format!(
        r#"<?xml version="1.0" ?>
<!DOCTYPE coverage SYSTEM "http://cobertura.sourceforge.net/xml/coverage-04.dtd">
<coverage version="1.0" timestamp="{timestamp}" lines-valid="0" lines-covered="0" line-rate="0" branches-valid="0" branches-covered="0" branch-rate="0" complexity="0">
  <packages>
  </packages>
</coverage>"#
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_generate_html_report() {
        let html = generate_html_report();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Probar Test Report"));
        assert!(html.contains("Tests Run"));
        assert!(html.contains("Passed"));
        assert!(html.contains("Failed"));
    }

    #[test]
    fn test_generate_json_report() {
        let json = generate_json_report();
        assert!(json.contains("\"version\": \"1.0\""));
        assert!(json.contains("\"tests\": []"));
        assert!(json.contains("\"timestamp\""));

        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["summary"]["total"], 0);
    }

    #[test]
    fn test_generate_lcov_report() {
        let lcov = generate_lcov_report();
        assert!(lcov.contains("TN:"));
        assert!(lcov.contains("SF:src/lib.rs"));
        assert!(lcov.contains("end_of_record"));
    }

    #[test]
    fn test_generate_junit_report() {
        let junit = generate_junit_report();
        assert!(junit.contains("<?xml"));
        assert!(junit.contains("<testsuites"));
        assert!(junit.contains("tests=\"0\""));
        assert!(junit.contains("failures=\"0\""));
    }

    #[test]
    fn test_generate_cobertura_report() {
        let cobertura = generate_cobertura_report();
        assert!(cobertura.contains("<?xml"));
        assert!(cobertura.contains("<coverage"));
        assert!(cobertura.contains("line-rate=\"0\""));
    }

    #[test]
    fn test_execute_report_html() {
        let temp = TempDir::new().unwrap();
        let output = temp.path().join("report.html");

        let config = CliConfig::default();
        let args = ReportArgs {
            format: ReportFormat::Html,
            output: output.clone(),
            open: false,
        };

        execute_report(&config, &args);

        assert!(output.exists());
        let content = std::fs::read_to_string(&output).unwrap();
        assert!(content.contains("<!DOCTYPE html>"));
    }

    #[test]
    fn test_execute_report_json() {
        let temp = TempDir::new().unwrap();
        let output = temp.path().join("report.json");

        let config = CliConfig::default();
        let args = ReportArgs {
            format: ReportFormat::Json,
            output: output.clone(),
            open: false,
        };

        execute_report(&config, &args);

        assert!(output.exists());
        let content = std::fs::read_to_string(&output).unwrap();
        let _: serde_json::Value = serde_json::from_str(&content).unwrap();
    }

    #[test]
    fn test_execute_report_lcov() {
        let temp = TempDir::new().unwrap();
        let output = temp.path().join("lcov.info");

        let config = CliConfig::default();
        let args = ReportArgs {
            format: ReportFormat::Lcov,
            output: output.clone(),
            open: false,
        };

        execute_report(&config, &args);

        assert!(output.exists());
    }

    #[test]
    fn test_execute_report_junit() {
        let temp = TempDir::new().unwrap();
        let output = temp.path().join("junit.xml");

        let config = CliConfig::default();
        let args = ReportArgs {
            format: ReportFormat::Junit,
            output: output.clone(),
            open: false,
        };

        execute_report(&config, &args);

        assert!(output.exists());
    }

    #[test]
    fn test_execute_report_cobertura() {
        let temp = TempDir::new().unwrap();
        let output = temp.path().join("cobertura.xml");

        let config = CliConfig::default();
        let args = ReportArgs {
            format: ReportFormat::Cobertura,
            output: output.clone(),
            open: false,
        };

        execute_report(&config, &args);

        assert!(output.exists());
    }

    #[test]
    fn test_execute_report_creates_parent_dirs() {
        let temp = TempDir::new().unwrap();
        let output = temp.path().join("nested").join("dir").join("report.html");

        let config = CliConfig::default();
        let args = ReportArgs {
            format: ReportFormat::Html,
            output: output.clone(),
            open: false,
        };

        execute_report(&config, &args);

        assert!(output.exists());
    }
}
