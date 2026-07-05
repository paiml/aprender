// Integration tests for --format html flag (Sprint 22)
#![allow(deprecated)] // suppress assert_cmd::Command::cargo_bin deprecation in tests
                      // Sprint 22: HTML output format for visual trace reports

#[test]
fn test_html_format_flag_accepted() {
    // Test that --format html flag is accepted by CLI
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("html").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    assert!(output.status.success(), "HTML format should be accepted");
}

#[test]
fn test_html_output_basic() {
    // Test that HTML output generates valid HTML document
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("html").arg("--").arg("echo").arg("hello");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should contain HTML document structure
    assert!(stdout.contains("<!DOCTYPE html>") || stdout.contains("<!doctype html>"));
    assert!(stdout.contains("<html"));
    assert!(stdout.contains("</html>"));
    assert!(stdout.contains("<head>"));
    assert!(stdout.contains("<body>"));
}

#[test]
fn test_html_output_contains_syscalls() {
    // Test that HTML output includes syscall traces
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("html").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should contain syscall data in HTML
    assert!(
        stdout.contains("write") || stdout.contains("exit_group"),
        "HTML should contain syscall names"
    );
}

#[test]
fn test_html_output_with_statistics() {
    // Test that HTML output includes statistics when -c flag used
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("html").arg("-c").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should contain statistics section
    assert!(
        stdout.contains("Statistics") || stdout.contains("stats") || stdout.contains("calls"),
        "HTML should contain statistics section"
    );
}

#[test]
fn test_html_output_with_timing() {
    // Test that HTML output includes timing when -T flag used
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("html").arg("-T").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should contain timing/duration information
    assert!(
        stdout.contains("duration") || stdout.contains("time") || stdout.contains("μs"),
        "HTML should contain timing information"
    );
}

#[test]
fn test_html_output_with_filtering() {
    // Test that HTML output works with syscall filtering
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("html").arg("-e").arg("trace=write").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should contain filtered syscalls
    assert!(stdout.contains("write"), "HTML should contain write syscall");
    // Should NOT contain non-filtered syscalls in trace
    // (exit_group might still appear in summary)
}

#[test]
fn test_html_output_standalone() {
    // Test that HTML output is standalone (no external dependencies)
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("html").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should contain embedded CSS (no external stylesheet links)
    assert!(stdout.contains("<style>"), "HTML should have embedded CSS");
    // Should NOT have external CSS links
    assert!(
        !stdout.contains("href=") || stdout.contains("href=\"#"),
        "HTML should not have external dependencies"
    );
}

#[test]
fn test_html_output_escape_special_chars() {
    // Test that HTML output escapes special characters (XSS prevention)
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("html").arg("--").arg("echo").arg("<script>alert('xss')</script>");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Extract only the HTML portion (after <!DOCTYPE html>) to exclude child process output
    let html_part =
        if let Some(idx) = stdout.find("<!DOCTYPE html>") { &stdout[idx..] } else { &stdout[..] };

    // Should escape < and > characters in the HTML report
    assert!(!html_part.contains("<script>alert"), "HTML should escape script tags");
    // Should contain escaped version
    assert!(
        html_part.contains("&lt;") || html_part.contains("&gt;") || !html_part.contains("<script>"),
        "HTML should escape special characters"
    );
}

#[test]
fn test_html_output_has_table_structure() {
    // Test that HTML output uses table structure for traces
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("html").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should contain table elements
    assert!(stdout.contains("<table"), "HTML should contain table element");
    assert!(
        stdout.contains("<tr") || stdout.contains("<th") || stdout.contains("<td"),
        "HTML should contain table row/cell elements"
    );
}

#[test]
fn test_html_output_backward_compatibility() {
    // Test that existing formats still work
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("json").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("{") && stdout.contains("}"), "JSON format should still work");
}
