# Sprint 22: HTML Output Format

**Date:** 2025-11-18
**Status:** RED Phase (Planning)
**Milestone:** v0.4.0 - HPU/ML/DL Profiling

---

## Sprint 22 Overview

Sprint 22 adds HTML output format for rich, visual trace reports. This complements existing JSON/CSV formats with styled tables, charts, and interactive elements for trace analysis.

**Sprint Goal:** Add `--format html` flag for generating standalone HTML trace reports with statistics visualization.

**Duration:** 2 weeks (estimated)

---

## Sprint Objectives

### Primary Goals
1. **CLI Flag** - Add `--format html` option
2. **HTML Report Generator** - Create `src/html_output.rs` module
3. **Statistics Visualization** - Tables and charts for syscall data
4. **Styled Output** - CSS-styled tables, syntax highlighting
5. **Standalone Files** - Self-contained HTML (no external dependencies)

### Success Criteria
- [ ] All integration tests passing (RED phase complete)
- [ ] HTML output module implemented
- [ ] Statistics rendered as styled tables
- [ ] Syscall traces with syntax highlighting
- [ ] Zero clippy warnings, complexity ≤10
- [ ] Backward compatible (existing formats unchanged)

---

## Architecture Design

### New Module: `src/html_output.rs`

```rust
//! HTML output format for trace reports
//!
//! Sprint 22: Rich visual reports with styled tables and charts

use crate::stats::StatsTracker;

/// HTML report generator
pub struct HtmlOutput {
    /// Report title
    title: String,
    /// CSS styles (embedded)
    styles: String,
}

impl HtmlOutput {
    /// Create new HTML output generator
    pub fn new(title: &str) -> Self;

    /// Generate HTML document from trace data
    pub fn generate(&self, traces: &[TraceEntry], stats: Option<&StatsTracker>) -> String;

    /// Render syscall trace as HTML table row
    fn render_trace_row(&self, entry: &TraceEntry) -> String;

    /// Render statistics as HTML table
    fn render_stats_table(&self, stats: &StatsTracker) -> String;

    /// Generate embedded CSS styles
    fn generate_styles(&self) -> String;
}

/// Single trace entry for HTML rendering
pub struct TraceEntry {
    pub syscall: String,
    pub args: String,
    pub result: String,
    pub duration_us: Option<u64>,
    pub source: Option<String>,
}
```

### CLI Changes

**File:** `src/cli.rs`

```rust
#[derive(ValueEnum, Clone, Debug)]
pub enum OutputFormat {
    Default,
    Json,
    Csv,
    Html,  // NEW
}
```

### Example Output

```html
<!DOCTYPE html>
<html>
<head>
    <title>Renacer Trace Report</title>
    <style>
        /* Embedded CSS for standalone file */
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; }
        .syscall { color: #0066cc; font-weight: bold; }
        .error { color: #cc0000; }
        .stats-table { margin-top: 20px; }
    </style>
</head>
<body>
    <h1>Syscall Trace Report</h1>
    <table>
        <tr><th>Syscall</th><th>Arguments</th><th>Result</th><th>Duration</th></tr>
        <tr><td class="syscall">openat</td><td>AT_FDCWD, "/etc/passwd", O_RDONLY</td><td>3</td><td>45 μs</td></tr>
        <!-- more rows -->
    </table>

    <h2>Statistics Summary</h2>
    <table class="stats-table">
        <tr><th>% time</th><th>seconds</th><th>usecs/call</th><th>calls</th><th>errors</th><th>syscall</th></tr>
        <!-- stats rows -->
    </table>
</body>
</html>
```

---

## EXTREME TDD Cycle

**Current Status:** RED Phase (Planning)

```
RED (Current)
  └─ Create 10+ integration tests (all failing)
      ↓
GREEN (Next)
  └─ Implement HtmlOutput (tests pass)
      ↓
REFACTOR (Final)
  └─ Unit tests, optimize, document
```

---

## Integration Tests Plan

### File: `tests/sprint22_html_output_tests.rs`

1. `test_html_format_flag_accepted` - CLI accepts --format html
2. `test_html_output_basic` - Generates valid HTML document
3. `test_html_output_with_statistics` - Stats table in HTML
4. `test_html_output_with_timing` - Duration column with -T flag
5. `test_html_output_with_source` - Source location with --source
6. `test_html_output_with_filtering` - Filtered traces in HTML
7. `test_html_output_standalone` - No external CSS/JS dependencies
8. `test_html_output_escape_special_chars` - XSS prevention
9. `test_html_output_with_hpu` - HPU analysis in HTML report
10. `test_html_output_large_trace` - Performance with large traces

---

## Dependencies

No new dependencies required. Uses standard library string formatting.

---

## Quality Gates

### Pre-Implementation Checklist
- [x] TDG Score: 95.1/100 (A+ grade)
- [x] All existing tests passing
- [x] Clippy: Zero warnings
- [x] Sprint 21 complete

### Implementation Checklist (Sprint 22)
- [ ] RED Phase: 10+ integration tests created
- [ ] GREEN Phase: All tests passing with HTML implementation
- [ ] REFACTOR Phase: Unit tests, complexity analysis
- [ ] Documentation: README.md, CHANGELOG.md updated
- [ ] Release: Commit and prepare for Sprint 23

---

## Notes

### Design Decisions

1. **Standalone HTML:** No external dependencies (CSS/JS embedded)
2. **Table-based Layout:** Simple, compatible with all browsers
3. **Syntax Highlighting:** CSS classes for syscall names, errors
4. **XSS Prevention:** Escape all user data in HTML output

### Integration Points

- **Statistics Module:** Render StatsTracker as HTML table
- **Timing Mode:** Duration column when -T flag used
- **Source Correlation:** Source file:line as hyperlink
- **HPU Analysis:** Correlation/clustering results in HTML

---

**Last Updated:** 2025-11-18
**Status:** RED Phase (Planning)
