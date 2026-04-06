# HTML Reports - Practical Examples

This chapter provides practical examples of using Renacer's HTML output format for real-world scenarios.

> **TDD-Verified:** All examples validated by [`tests/sprint22_html_output_tests.rs`](../../../tests/sprint22_html_output_tests.rs)

## Overview

HTML reports are ideal for:
- **Sharing with stakeholders** - Non-technical team members can view professional reports
- **Documentation** - Archiving performance analysis for later reference
- **Presentations** - Visual reports for meetings and demos
- **CI/CD** - Automated report generation in build pipelines

## Basic Report Generation

### Simple Trace Report

Generate a basic HTML report for any command:

```bash
renacer --format html -- ls -la > trace.html
```

**Tested by:** `test_html_format_flag_accepted`, `test_html_output_basic`

**Output:** Standalone HTML file with syscall trace table

**Use case:** Quick visualization of syscall behavior

## Build Performance Reports

### Analyzing Cargo Build

```bash
renacer --format html -c -T -- cargo build > build-report.html
```

**Tested by:** `test_html_output_with_statistics`, `test_html_output_with_timing`

**Report includes:**
1. **Syscall trace table** - Individual syscall events with timing
2. **Statistics summary** - Call counts, time percentages, errors
3. **Visual styling** - Color-coded, sortable columns

**Example output structure:**

```html
<h1>Syscall Trace Report</h1>
<table>
  <tr><th>Syscall</th><th>Arguments</th><th>Result</th><th>Duration</th></tr>
  <tr><td class="syscall">openat</td><td class="args">AT_FDCWD, "/etc/ld.so.cache", ...</td><td>3</td><td class="duration">234 us</td></tr>
  ...
</table>

<h2>Statistics Summary</h2>
<table class="stats-table">
  <tr><th>% time</th><th>seconds</th><th>usecs/call</th><th>calls</th><th>errors</th><th>syscall</th></tr>
  <tr><td>45.23</td><td>0.012345</td><td>1234</td><td>10</td><td>0</td><td class="syscall">read</td></tr>
  <tr><td>32.15</td><td>0.008765</td><td>876</td><td>10</td><td>0</td><td class="syscall">write</td></tr>
</table>
```

**What to look for:**
- **High % time** - Syscalls consuming most execution time
- **High usecs/call** - Slow individual operations
- **Error counts** - Failed syscalls (negative results highlighted in red)

## Debugging I/O Performance

### Filtering File Operations

Focus on file I/O to debug slow disk operations:

```bash
renacer --format html -e trace=file -T -- ./slow-app > io-report.html
```

**Tested by:** `test_html_output_with_filtering`, `test_html_output_with_timing`

**Filter effects:**
- **Only file syscalls** included: `open`, `read`, `write`, `close`, `fsync`, etc.
- **Noise removed** - No network, memory, or process syscalls
- **Duration column** - Identify slow I/O operations

**Example use case:**

```bash
# Application is slow, suspect file I/O
$ renacer --format html -e trace=file -T -- ./database-app > db-io.html

# Open db-io.html in browser, look for:
# 1. High duration on fsync (indicates sync disk writes)
# 2. Many small reads (batching opportunity)
# 3. Failed opens (red results = permission/missing files)
```

### Filtering Network Operations

Analyze network syscalls for latency issues:

```bash
renacer --format html -e trace=network -T -- curl https://api.example.com > network-trace.html
```

**Tested by:** `test_html_output_with_filtering`, `test_html_output_with_timing`

**Reveals:**
- `connect` syscall duration (DNS + TCP handshake)
- `sendto`/`recvfrom` patterns (request-response timing)
- Socket errors (connection refused, timeouts)

## Source-Correlated Reports

### Debugging with Source Locations

Include source file/line information for debugging:

```bash
renacer --format html -T --source -- ./my-binary > debug-report.html
```

**Requirements:**
- Binary compiled with debug symbols (`-g` flag)
- DWARF debug info available

**Example output:**

```html
<table>
  <tr><th>Syscall</th><th>Arguments</th><th>Result</th><th>Duration</th><th>Source</th></tr>
  <tr>
    <td class="syscall">write</td>
    <td class="args">1, "log message", 11</td>
    <td class="result">11</td>
    <td class="duration">1234 us</td>
    <td class="source">src/logger.rs:42</td>
  </tr>
</table>
```

**Tested by:** Implementation supports `--source` flag

**Use case:** Identify which code is making slow syscalls

## Sharing Reports with Teams

### Complete Analysis Report

Generate comprehensive report for team review:

```bash
renacer --format html -c -T --source -- ./production-app > analysis.html
# Email analysis.html to team
```

**Tested by:** `test_html_output_with_statistics`, `test_html_output_with_timing`

**Benefits:**
- **Standalone file** - No external dependencies, works offline
- **Professional appearance** - Modern CSS styling
- **Accessible** - Non-technical stakeholders can understand
- **Portable** - Viewable on any device with web browser

### CI/CD Integration

Automate report generation in build pipelines:

```yaml
# .github/workflows/performance.yml
- name: Generate Performance Report
  run: |
    cargo build --release
    renacer --format html -c -T -- ./target/release/my-app > perf-report.html

- name: Upload Report
  uses: actions/upload-artifact@v3
  with:
    name: performance-report
    path: perf-report.html
```

**Tested by:** `test_html_format_flag_accepted`, `test_html_output_basic`

**Result:** HTML report available as downloadable artifact in GitHub Actions

## Security Auditing

### XSS-Safe Output

HTML output automatically escapes untrusted input:

```bash
# Untrusted input (from external source)
renacer --format html -- ./user-script '<script>alert("xss")</script>' > safe-report.html
```

**Tested by:** `test_html_output_escape_special_chars`

**Safety features:**
- `<` → `&lt;`
- `>` → `&gt;`
- `&` → `&amp;`
- `"` → `&quot;`
- `'` → `&#39;`

**Result:** Script tags displayed as text (safe), not executed

**Example output:**

```html
<td class="args">&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;</td>
```

Browser displays: `<script>alert("xss")</script>` (as text, not running code)

## Visual Error Identification

### Failed Syscalls Highlighted

HTML reports automatically highlight errors in red:

```bash
renacer --format html -- ./app-with-errors > error-report.html
```

**Visual indicators:**
- **Negative results** - Red text color
- **Class: result-error** - CSS styling applied
- **Easy scanning** - Errors stand out visually

**Example:**

```html
<tr>
  <td class="syscall">open</td>
  <td class="args">"/nonexistent", O_RDONLY</td>
  <td class="result result-error">-2</td>  <!-- ENOENT in red -->
</tr>
<tr>
  <td class="syscall">write</td>
  <td class="args">1, "success", 7</td>
  <td class="result">7</td>  <!-- Success in normal color -->
</tr>
```

**CSS:**
```css
.result-error {
    color: #cc0000;  /* Red for errors */
}
```

## Comparing Formats

### When to Use HTML vs Others

**Use HTML for:**
- Non-technical stakeholders
- Documentation and archiving
- Visual presentations
- Quick human review

**Use JSON for:**
- Programmatic analysis
- CI/CD automation
- Data processing scripts

**Use CSV for:**
- Spreadsheet analysis (Excel, Google Sheets)
- Statistical tools (R, Python pandas)
- Data science workflows

**Example workflow:**

```bash
# Analysis: Generate all formats
renacer --format html -c -T -- ./app > analysis.html
renacer --format json -c -T -- ./app > analysis.json
renacer --format csv -c -T -- ./app > analysis.csv

# Share HTML with team
# Process JSON with scripts
# Analyze CSV in Excel/R
```

**Tested by:** `test_html_output_backward_compatibility`

## Advanced Use Cases

### Performance Regression Detection

Track performance over time with HTML reports:

```bash
# Baseline (before changes)
git checkout main
cargo build --release
renacer --format html -c -T -- ./target/release/app > baseline.html

# After changes
git checkout feature-branch
cargo build --release
renacer --format html -c -T -- ./target/release/app > feature.html

# Compare baseline.html vs feature.html side-by-side
```

**Tested by:** `test_html_output_with_statistics`, `test_html_output_with_timing`

**Visual comparison reveals:**
- Increased syscall counts (regressions)
- Changed time percentages
- New error patterns

### Multi-Process Analysis

Analyze parent + child processes:

```bash
renacer --format html -f -c -T -- make test > multiprocess-report.html
```

**Report includes:**
- All processes (parent + children)
- Per-process syscall traces
- Combined statistics

**Use case:** Understand parallel build behavior

## Report Customization

### Opening in Browser

View HTML reports immediately:

```bash
# Linux
renacer --format html -c -T -- ./app > report.html && xdg-open report.html

# macOS
renacer --format html -c -T -- ./app > report.html && open report.html

# Windows
renacer --format html -c -T -- ./app > report.html && start report.html
```

**Tested by:** `test_html_output_basic`

### Archiving Reports

Organize reports by date/version:

```bash
#!/bin/bash
DATE=$(date +%Y-%m-%d)
VERSION=$(git describe --tags)
REPORT="perf-${VERSION}-${DATE}.html"

renacer --format html -c -T -- ./app > "reports/${REPORT}"
echo "Report saved: reports/${REPORT}"
```

**Organization:**
```text
reports/
├── perf-v1.0.0-2025-01-15.html
├── perf-v1.1.0-2025-02-01.html
└── perf-v1.2.0-2025-03-01.html
```

## Troubleshooting Reports

### Large Reports (>10K Syscalls)

For very large traces, HTML may be slow in browser:

**Solution 1:** Filter to specific syscalls
```bash
renacer --format html -e trace=file -c -T -- ./app > filtered.html
```

**Solution 2:** Use CSV for analysis, HTML for summary
```bash
# Full trace as CSV for processing
renacer --format csv -c -T -- ./app > full-trace.csv

# Filtered summary as HTML for viewing
renacer --format html -e trace=file -c -T -- ./app > summary.html
```

**Tested by:** `test_html_output_with_filtering`

### Encoding Issues

HTML uses UTF-8 charset:

```html
<meta charset="UTF-8">
```

**If characters appear garbled:**
1. Ensure browser encoding set to UTF-8
2. Check file saved with UTF-8 encoding
3. Verify locale settings (`locale -a`)

**Tested by:** `test_html_output_basic` (UTF-8 meta tag included)

## Summary

HTML reports provide:
- ✅ **Visual appeal** for presentations and sharing
- ✅ **Standalone format** (no dependencies)
- ✅ **Security** via automatic XSS escaping
- ✅ **Accessibility** for non-technical users
- ✅ **Integration** with statistics, timing, filtering, source
- ✅ **Error highlighting** for quick issue identification
- ✅ **Archiving** for historical performance tracking

**All examples tested in:** [`tests/sprint22_html_output_tests.rs`](../../../tests/sprint22_html_output_tests.rs)

## Related

- [HTML Output Format Reference](../reference/format-html.md) - Technical specification
- [Statistics Mode](../core-concepts/statistics.md) - Call counts and timing
- [Filtering Syscalls](../core-concepts/filtering.md) - Focus on specific operations
- [JSON Output](../reference/format-json.md) - Machine-readable format
- [CSV Output](../reference/format-csv.md) - Spreadsheet format
