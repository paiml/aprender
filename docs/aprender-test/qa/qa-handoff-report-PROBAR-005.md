# QA Handoff Report: Enhanced Serving & Debugging (PROBAR-005)

**Feature**: Enhanced Serving, Debugging, and Project Scoring
**Ticket**: PROBAR-005
**Spec**: `docs/specifications/enhanced-serving-debugging.md`
**Version**: 1.2.0
**Target Score**: 115 Points

## Overview

This feature set transforms `probar` from a simple runner into a comprehensive development platform. It introduces file visualization, hot reload, linting, load testing, and a unified "Project Score" to quantify testing maturity.

## 115-Point Falsification Checklist

This checklist implements Popperian Falsificationism. We attempt to break the system; if we fail, the system passes.

### A. File Tree Visualization (15 points)

| # | Falsification Attempt | Expected Result | Pass/Fail |
|---|----------------------|-----------------|-----------|
| A01 | Run `probar serve tree` on empty directory | Shows "Empty directory" message, not crash | |
| A02 | Run `probar serve tree` on directory with 10,000+ files | Completes within 5 seconds | |
| A03 | Run `probar serve tree --depth 0` | Shows only root directory, no children | |
| A04 | Run `probar serve tree --depth 1` | Shows exactly 1 level of children | |
| A05 | Run `probar serve tree --filter "*.html"` | Shows only .html files | |
| A06 | Run `probar serve tree` on directory with symlinks | Does not follow symlinks infinitely | |
| A07 | Run `probar serve tree` with non-UTF8 filenames | Handles gracefully, shows placeholder | |
| A08 | Run `probar serve viz` without trueno-viz installed | Shows error with install instructions | |
| A09 | Run `probar serve tree` on non-existent path | Shows clear error message | |
| A10 | Run `probar serve tree` on file (not directory) | Shows error or single file info | |
| A11 | Verify MIME types shown are accurate for .wasm files | Shows "application/wasm" | |
| A12 | Verify MIME types shown are accurate for .js files | Shows "text/javascript" | |
| A13 | Verify file sizes are accurate | Matches `ls -l` output | |
| A14 | Run `probar serve tree` on directory with permission denied files | Shows "[permission denied]" not crash | |
| A15 | Run `probar serve viz` and resize terminal | TUI resizes correctly | |

### B. Content Linting (15 points)

| # | Falsification Attempt | Expected Result | Pass/Fail |
|---|----------------------|-----------------|-----------|
| B01 | Lint valid HTML5 file | Reports 0 errors, 0 warnings | |
| B02 | Lint HTML file with missing `<!DOCTYPE html>` | Reports warning | |
| B03 | Lint HTML file with unclosed tags | Reports error | |
| B04 | Lint HTML file with broken internal links | Reports error with path | |
| B05 | Lint valid CSS3 file | Reports 0 errors | |
| B06 | Lint CSS file with syntax error | Reports error with line number | |
| B07 | Lint CSS file with vendor prefix warning | Reports warning | |
| B08 | Lint JavaScript file with syntax error | Reports error | |
| B09 | Lint JavaScript ES6 module with import errors | Reports unresolved import | |
| B10 | Lint valid WASM file | Reports 0 errors | |
| B11 | Lint corrupted WASM file (invalid magic) | Reports error | |
| B12 | Lint JSON file with syntax error | Reports error with position | |
| B13 | Run `probar serve --lint` on mixed content directory | Lints all supported file types | |
| B14 | Run `probar lint` on binary file (e.g., .png) | Skips gracefully | |
| B15 | Lint file with BOM (byte order mark) | Handles correctly | |

### C. Hot Reload (15 points)

| # | Falsification Attempt | Expected Result | Pass/Fail |
|---|----------------------|-----------------|-----------|
| C01 | Modify HTML file while serving | Browser receives reload notification | |
| C02 | Modify CSS file while serving | Browser receives update (no full reload) | |
| C03 | Create new file while serving | New file immediately serveable | |
| C04 | Delete file while serving | Returns 404 for deleted file | |
| C05 | Rename file while serving | Old path 404, new path works | |
| C06 | Modify file 100 times in 1 second | Debounce prevents 100 notifications | |
| C07 | Modify ignored file (e.g., .git/*) | No reload notification | |
| C08 | Run with `--no-watch` and modify file | No reload notification | |
| C09 | Connect 10 browser tabs simultaneously | All 10 receive notifications | |
| C10 | Disconnect all browsers and modify file | Server does not crash | |
| C11 | Modify .wasm file | Full reload triggered (not CSS inject) | |
| C12 | Verify WebSocket reconnection after server restart | Clients auto-reconnect | |
| C13 | Modify file with non-UTF8 content | Notification sent, no crash | |
| C14 | Create file with very long name (255 chars) | Handled correctly | |
| C15 | Modify file on network mount (slow filesystem) | Debounce handles latency | |

### D. Load Testing (15 points)

| # | Falsification Attempt | Expected Result | Pass/Fail |
|---|----------------------|-----------------|-----------|
| D01 | Run load test with 1 user | Completes successfully | |
| D02 | Run load test with 1000 concurrent users | Completes without crashing probar | |
| D03 | Run load test against non-existent server | Reports connection error | |
| D04 | Run load test with 0 duration | Reports invalid config | |
| D05 | Run load test with ramp 1->100 users | User count increases linearly | |
| D06 | Run load test with assertion latency < 1ms (impossible) | Reports assertion failure | |
| D07 | Interrupt load test with Ctrl+C | Stops gracefully, reports partial results | |
| D08 | Run load test with invalid scenario YAML | Reports parse error | |
| D09 | Verify p50/p95/p99 latency calculations | Match independent calculation | |
| D10 | Run load test against HTTPS endpoint | Works with TLS | |
| D11 | Run load test with custom headers | Headers included in requests | |
| D12 | Run load test outputting JSON | Valid JSON output | |
| D13 | Run load test outputting HTML report | Valid HTML with charts | |
| D14 | Verify throughput calculation (req/s) | Matches request_count / duration | |
| D15 | Run load test with POST requests and body | Body sent correctly | |

### E. WASM/TUI Features (20 points)

| # | Falsification Attempt | Expected Result | Pass/Fail |
|---|----------------------|-----------------|-----------|
| E01 | Record empty session | Creates valid recording file | |
| E02 | Record session with 1000 events | Recording file < 10MB | |
| E03 | Replay recording on different machine | Deterministic behavior | |
| E04 | Replay corrupted recording file | Reports error, not crash | |
| E05 | Profile memory on WASM app | Reports initial, peak, current heap | |
| E06 | Profile memory with leak (no deallocation) | Shows continuous growth | |
| E07 | Set memory threshold 1MB, load 10MB app | Triggers threshold alert | |
| E08 | Validate valid state machine | All assertions pass | |
| E09 | Validate state machine with forbidden transition | Reports violation | |
| E10 | Validate state machine with invariant violation | Reports which invariant failed | |
| E11 | Run cross-browser test on Chrome only | Chrome tests run | |
| E12 | Run cross-browser test with browser not installed | Reports clear error | |
| E13 | Run performance benchmark with no baseline | Creates new baseline | |
| E14 | Run benchmark with 5% regression (under 10% threshold) | Reports OK | |
| E15 | Run benchmark with 15% regression (over 10% threshold) | Reports WARN/FAIL | |
| E16 | Benchmark metric that doesn't exist in baseline | Reports new metric | |
| E17 | Memory profile shows negative allocation | Never shows negative | |
| E18 | State validation timeout after 30s | Reports timeout, not hang | |
| E19 | Cross-browser test with different viewport sizes | All viewports tested | |
| E20 | Performance benchmark outputs JSON | Valid JSON with all metrics | |

### F. Debug Mode (20 points)

| # | Falsification Attempt | Expected Result | Pass/Fail |
|---|----------------------|-----------------|-----------|
| F01 | Run `probar serve --debug` | Shows verbose startup info | |
| F02 | Request file in debug mode | Shows full request/response trace | |
| F03 | Request non-existent file in debug mode | Shows searched paths | |
| F04 | Request file with wrong MIME type expectation | Debug shows actual MIME | |
| F05 | Set breakpoint on state "recording" | Pauses when entering recording | |
| F06 | Set breakpoint on event "wasm_ready" | Pauses when event fires | |
| F07 | Set breakpoint on request pattern "/api/*" | Pauses on matching requests | |
| F08 | Step through playbook state by state | Each step waits for Enter | |
| F09 | Quit step-by-step mode with 'q' | Exits cleanly | |
| F10 | Debug output with verbosity=minimal | Shows only errors | |
| F11 | Debug output with verbosity=trace | Shows internal state changes | |
| F12 | Write debug log to file | File contains all debug output | |
| F13 | Debug mode with 1000 rapid requests | Does not OOM from log buffer | |
| F14 | Debug mode shows CORS header issues | Clear message about CORS | |
| F15 | Debug mode shows COOP/COEP requirements | Clear message about SharedArrayBuffer | |
| F16 | Debug shows directory resolution (index.html) | Shows "Rule: Directory index" | |
| F17 | Debug shows static file resolution | Shows "Rule: Static file" | |
| F18 | Debug shows 404 with suggestions | Suggests similar files | |
| F19 | Debug timestamp precision | Millisecond precision or better | |
| F20 | Debug mode color output in TTY | Colors visible | |

### G. Project Testing Score (15 points)

| # | Falsification Attempt | Expected Result | Pass/Fail |
|---|----------------------|-----------------|-----------|
| G01 | Run `probar serve score` on project with no tests | Returns 0/100, grade F | |
| G02 | Run `probar serve score` on fully-tested project | Returns close to 100/100 | |
| G03 | Run `probar serve score --min 80` on 70-point project | Exit code non-zero | |
| G04 | Run `probar serve score --min 60` on 70-point project | Exit code zero | |
| G05 | Run `probar serve score --format json` | Valid JSON output | |
| G06 | Run `probar serve score --verbose` | Shows all criteria details | |
| G07 | Run `probar serve score --report out.html` | Creates valid HTML report | |
| G08 | Verify playbook scoring (add playbook, score increases) | +5 points for playbook existence | |
| G09 | Verify pixel testing scoring (add snapshot, score increases) | Points reflect snapshot count | |
| G10 | Verify cross-browser scoring (add Firefox, score increases) | +3 points for Firefox | |
| G11 | Recommendations sorted by potential points | Highest impact first | |
| G12 | Score history appends to JSONL file | Valid JSONL format | |
| G13 | Score trend displays ASCII chart | Chart renders correctly | |
| G14 | Grade boundaries correct (89=B, 90=A) | Boundary cases correct | |
| G15 | Empty directory returns valid score (0/100) | Does not crash | |

---

**Implementation Notes:**
- All features must support `NO_COLOR` environment variable.
- All commands must be async/await compatible with `tokio`.
- All file operations must handle OS permissions gracefully.
