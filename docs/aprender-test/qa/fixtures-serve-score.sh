#!/bin/bash
# fixtures-serve-score.sh
# Fixture generator for probar serve score falsification testing
#
# Usage:
#   source fixtures-serve-score.sh
#   create_valid_project /tmp/myproject
#
# See: 100point-falsification-qa-runlist-serve-score.md

set -euo pipefail

# Create completely empty project
create_empty_project() {
    local dir="${1:?Usage: create_empty_project <dir>}"
    mkdir -p "$dir"
    echo "Created empty project: $dir"
}

# Create project with files that LOOK like they should score points
# but contain NO actual test content (for false positive testing)
create_fake_high_score_project() {
    local dir="${1:?Usage: create_fake_high_score_project <dir>}"
    mkdir -p "$dir"/{playbooks,snapshots,tests,recordings}

    # Empty files that match scoring patterns
    touch "$dir/playbooks/fake.yaml"
    touch "$dir/snapshots/home.png"
    touch "$dir/snapshots/mobile-home.png"
    touch "$dir/snapshots/dark-home.png"
    touch "$dir/tests/fake_test.rs"
    touch "$dir/recordings/happy.json"
    touch "$dir/recordings/error.json"
    touch "$dir/recordings/edge.json"
    touch "$dir/browsers.yaml"
    touch "$dir/a11y.yaml"
    touch "$dir/load-test.yaml"
    touch "$dir/chaos.yaml"
    touch "$dir/baseline.json"
    touch "$dir/tests/README.md"
    touch "$dir/README.md"

    # NOTE: No probar-results.json or runtime evidence
    # This should trigger grade capping

    echo "Created fake high-score project (no runtime): $dir"
}

# Create valid project with actual runtime evidence
create_valid_project() {
    local dir="${1:?Usage: create_valid_project <dir>}"
    mkdir -p "$dir"/{playbooks,snapshots,tests,recordings,.probar/results}

    # Valid playbook with actual content
    cat > "$dir/playbooks/app.yaml" << 'YAML'
version: "1.0"
name: Calculator Application Test
machine:
  initial: idle
  states:
    idle:
      invariants:
        - display == "0"
      on:
        DIGIT: inputting
        CLEAR: idle
    inputting:
      invariants:
        - display.length > 0
      on:
        DIGIT: inputting
        OPERATOR: calculating
        CLEAR: idle
    calculating:
      on:
        EQUALS: result
        CLEAR: idle
    result:
      on:
        DIGIT: inputting
        CLEAR: idle
  forbidden:
    - from: result
      to: calculating
      reason: "Cannot chain operations without new input"
performance:
  rtf_target: 60
  max_memory_mb: 50
  p95_latency_ms: 100
YAML

    # Valid test results (proves tests ran)
    cat > "$dir/.probar/results/run-001.json" << 'JSON'
{
  "timestamp": "2025-12-14T10:00:00Z",
  "passed": 15,
  "failed": 0,
  "skipped": 2,
  "duration_ms": 4523
}
JSON

    cat > "$dir/probar-results.json" << 'JSON'
{
  "runs": 1,
  "total_passed": 15,
  "total_failed": 0,
  "coverage": 0.87
}
JSON

    # Valid recordings with content
    cat > "$dir/recordings/happy-path.json" << 'JSON'
{
  "name": "happy-path",
  "events": [
    {"type": "click", "target": "#btn-1", "timestamp": 100},
    {"type": "click", "target": "#btn-plus", "timestamp": 200},
    {"type": "click", "target": "#btn-2", "timestamp": 300},
    {"type": "click", "target": "#btn-equals", "timestamp": 400}
  ],
  "assertions": [
    {"type": "text", "target": "#display", "expected": "3"}
  ],
  "passed": true
}
JSON

    cat > "$dir/recordings/error-handling.json" << 'JSON'
{
  "name": "error-handling",
  "events": [
    {"type": "click", "target": "#btn-divide", "timestamp": 100},
    {"type": "click", "target": "#btn-0", "timestamp": 200},
    {"type": "click", "target": "#btn-equals", "timestamp": 300}
  ],
  "assertions": [
    {"type": "text", "target": "#display", "expected": "Error"}
  ],
  "passed": true
}
JSON

    cat > "$dir/recordings/edge-boundary.json" << 'JSON'
{
  "name": "edge-boundary",
  "events": [
    {"type": "input", "target": "#display", "value": "99999999999999"}
  ],
  "assertions": [
    {"type": "overflow", "expected": true}
  ],
  "passed": true
}
JSON

    # Valid PNG snapshots (with PNG magic bytes)
    printf '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' > "$dir/snapshots/home.png"
    printf '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' > "$dir/snapshots/mobile-home.png"
    printf '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' > "$dir/snapshots/tablet-home.png"
    printf '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' > "$dir/snapshots/dark-home.png"

    # Browser matrix
    cat > "$dir/browsers.yaml" << 'YAML'
browsers:
  desktop:
    - name: chrome
      version: latest
    - name: firefox
      version: latest
    - name: safari
      version: latest
  mobile:
    - name: ios-safari
      device: iPhone 14
    - name: chrome-android
      device: Pixel 7
YAML

    # Accessibility configuration
    cat > "$dir/a11y.yaml" << 'YAML'
wcag_level: AA
checks:
  - color-contrast
  - aria-labels
  - focus-visible
  - keyboard-navigation
  - screen-reader-flow
YAML

    # Load testing configuration
    cat > "$dir/load-test.yaml" << 'YAML'
scenarios:
  - name: boot
    users: 100
    ramp: 30s
    duration: 120s
  - name: sustained
    users: 50
    duration: 300s
assertions:
  p99_latency_ms: 200
  error_rate: 0.01
YAML

    # Load test results
    cat > "$dir/load-test-results.json" << 'JSON'
{
  "scenario": "boot",
  "users": 100,
  "requests": 15234,
  "errors": 12,
  "p50_ms": 45,
  "p95_ms": 120,
  "p99_ms": 185,
  "throughput_rps": 127
}
JSON

    # Chaos/fault injection config
    cat > "$dir/chaos.yaml" << 'YAML'
injections:
  - type: latency
    target: api
    delay_ms: 500
    probability: 0.1
  - type: error
    target: api
    status: 503
    probability: 0.05
YAML

    # Performance baseline
    cat > "$dir/baseline.json" << 'JSON'
{
  "rtf": 62,
  "memory_mb": 48,
  "p95_latency_ms": 95,
  "wasm_size_kb": 245
}
JSON

    # Test files with actual content
    cat > "$dir/tests/gui_test.rs" << 'RUST'
//! GUI interaction tests for calculator

use jugar_probar::prelude::*;

#[test]
fn test_button_click() {
    let page = Page::new();
    page.click("#btn-1").await;
    assert_eq!(page.text("#display"), "1");
}

#[test]
fn test_keyboard_input() {
    let page = Page::new();
    page.press("1");
    page.press("+");
    page.press("2");
    page.press("Enter");
    assert_eq!(page.text("#display"), "3");
}
RUST

    # Keyboard navigation config
    cat > "$dir/keyboard.yaml" << 'YAML'
tab_order:
  - "#btn-clear"
  - "#btn-divide"
  - "#btn-multiply"
  - "#btn-7"
shortcuts:
  Enter: calculate
  Escape: clear
  c: clear
YAML

    # Documentation
    cat > "$dir/tests/README.md" << 'MD'
# Calculator Tests

## Test Structure

- `gui_test.rs` - Button click and keyboard tests
- `../playbooks/app.yaml` - State machine playbook
- `../recordings/` - Deterministic replay files

## Running Tests

```bash
probar test
probar playbook playbooks/app.yaml
```

## Coverage

Target: 95% pixel coverage across all states.
MD

    cat > "$dir/README.md" << 'MD'
# Calculator Application

A WASM-based calculator tested with probar.

## Quick Start

```bash
probar build
probar serve --port 8080
probar test
```
MD

    echo "Created valid project with runtime evidence: $dir"
}

# Create project that should score exactly at a grade boundary
create_boundary_project() {
    local dir="${1:?Usage: create_boundary_project <dir>}"
    local target_grade="${2:-C}"  # A, B, C, D, F

    mkdir -p "$dir"/{playbooks,.probar/results}

    # Minimal runtime evidence
    echo '{"passed": 1}' > "$dir/.probar/results/run.json"
    echo '{"passed": 1}' > "$dir/probar-results.json"

    case "$target_grade" in
        A)
            # Need 90+ points
            mkdir -p "$dir"/{snapshots,tests,recordings}
            touch "$dir/playbooks/app.yaml"
            touch "$dir/snapshots/"{home,mobile-home,dark-home}.png
            touch "$dir/tests/test.rs"
            touch "$dir/recordings/"{happy,error,edge}.json
            touch "$dir/browsers.yaml"
            touch "$dir/a11y.yaml"
            touch "$dir/load-test.yaml"
            touch "$dir/chaos.yaml"
            echo '{}' > "$dir/load-test-results.json"
            echo '{}' > "$dir/baseline.json"
            touch "$dir/keyboard.yaml"
            touch "$dir/gesture.yaml"
            mkdir -p "$dir/tests" && touch "$dir/tests/README.md"
            touch "$dir/README.md"
            ;;
        B)
            # Need 80-89 points
            mkdir -p "$dir"/{snapshots,tests,recordings}
            touch "$dir/playbooks/app.yaml"
            touch "$dir/snapshots/home.png"
            touch "$dir/tests/test.rs"
            touch "$dir/recordings/happy.json"
            touch "$dir/browsers.yaml"
            touch "$dir/a11y.yaml"
            echo '{}' > "$dir/baseline.json"
            ;;
        C)
            # Need 70-79 points
            mkdir -p "$dir"/{snapshots,tests}
            touch "$dir/playbooks/app.yaml"
            touch "$dir/snapshots/home.png"
            touch "$dir/tests/test.rs"
            echo '{}' > "$dir/baseline.json"
            ;;
        D)
            # Need 60-69 points
            touch "$dir/playbooks/app.yaml"
            echo '{}' > "$dir/baseline.json"
            ;;
        F)
            # Need <60 points - minimal content
            touch "$dir/playbooks/app.yaml"
            ;;
    esac

    echo "Created boundary project for grade $target_grade: $dir"
}

# Cleanup function
cleanup_test_fixtures() {
    local root="${1:-/tmp/probar-falsification}"
    if [[ -d "$root" ]]; then
        rm -rf "$root"
        echo "Cleaned up: $root"
    fi
}

# Create project with intentionally broken WASM import (for --live testing)
create_broken_wasm_project() {
    local dir="${1:?Usage: create_broken_wasm_project <dir>}"
    mkdir -p "$dir"

    cat > "$dir/index.html" << 'HTML'
<!DOCTYPE html>
<html>
<head><title>Broken WASM</title></head>
<body>
<script type="module">
import init from './pkg/app.js';
await init();
</script>
</body>
</html>
HTML

    # Note: NO pkg/app.js exists - this should FAIL --live
    echo "Created broken WASM project (missing import): $dir"
}

# Create project that passes static scoring but fails live validation
# This is THE critical falsification test
create_static_pass_live_fail_project() {
    local dir="${1:?Usage: create_static_pass_live_fail_project <dir>}"
    mkdir -p "$dir"/{playbooks,snapshots,tests,recordings,.probar/results}

    # All static scoring artifacts (high score)
    cat > "$dir/playbooks/app.yaml" << 'YAML'
version: "1.0"
name: Fake App
machine:
  initial: idle
  states:
    idle: {}
YAML

    printf '\x89PNG\r\n\x1a\n' > "$dir/snapshots/home.png"
    printf '\x89PNG\r\n\x1a\n' > "$dir/snapshots/mobile-home.png"
    printf '\x89PNG\r\n\x1a\n' > "$dir/snapshots/dark-home.png"

    echo '// fake test' > "$dir/tests/test.rs"
    echo '{"passed": 1}' > "$dir/.probar/results/run.json"
    echo '{"passed": 1}' > "$dir/probar-results.json"
    echo '{"events": []}' > "$dir/recordings/happy.json"
    echo '{"events": []}' > "$dir/recordings/error.json"
    echo '{"events": []}' > "$dir/recordings/edge.json"

    echo 'browsers: [chrome, firefox, safari]' > "$dir/browsers.yaml"
    echo 'wcag_level: AA' > "$dir/a11y.yaml"
    echo 'scenarios: []' > "$dir/load-test.yaml"
    echo '{}' > "$dir/load-test-results.json"
    echo 'injections: []' > "$dir/chaos.yaml"
    echo '{}' > "$dir/baseline.json"
    echo '# Tests' > "$dir/tests/README.md"
    echo '# App' > "$dir/README.md"

    # But the HTML has errors! This should FAIL --live
    cat > "$dir/index.html" << 'HTML'
<!DOCTYPE html>
<html>
<head><title>Fake Working App</title></head>
<body>
<h1>This looks fine...</h1>
<script>
// But there are runtime errors!
console.error("This app is broken!");
throw new Error("Critical failure - app doesn't actually work");
</script>
</body>
</html>
HTML

    echo "Created project: passes static, fails --live: $dir"
    echo "  Static: should get high score (A or B)"
    echo "  Live:   should FAIL with console errors"
}

# Create project with console warnings (not errors) - should pass --live
create_warning_only_project() {
    local dir="${1:?Usage: create_warning_only_project <dir>}"
    mkdir -p "$dir"

    cat > "$dir/index.html" << 'HTML'
<!DOCTYPE html>
<html>
<head><title>Warnings Only</title></head>
<body>
<h1>App with warnings</h1>
<script>
// Warnings should not fail --live
console.warn("This is just a warning");
console.log("Normal log message");
</script>
</body>
</html>
HTML

    echo "Created warning-only project: $dir"
}

# Create project with slow initialization (tests timeout handling)
create_slow_project() {
    local dir="${1:?Usage: create_slow_project <dir>}"
    mkdir -p "$dir"

    cat > "$dir/index.html" << 'HTML'
<!DOCTYPE html>
<html>
<head><title>Slow App</title></head>
<body>
<h1>Slow initialization...</h1>
<script>
// Simulate slow WASM init
const start = Date.now();
while (Date.now() - start < 5000) {
    // Block for 5 seconds
}
console.log("Finally loaded!");
</script>
</body>
</html>
HTML

    echo "Created slow project (5s init): $dir"
}

# Print usage if sourced with --help
if [[ "${1:-}" == "--help" ]]; then
    cat << 'USAGE'
Fixture Generator for probar serve score Falsification Testing

Functions:
  create_empty_project <dir>              - Create empty directory
  create_fake_high_score_project <dir>    - Files without runtime (tests grade cap)
  create_valid_project <dir>              - Full valid project with runtime evidence
  create_boundary_project <dir> [grade]   - Project at grade boundary (A/B/C/D/F)
  cleanup_test_fixtures [root]            - Remove test fixtures

  Live Validation (--live flag):
  create_broken_wasm_project <dir>        - Missing WASM import (FAIL --live)
  create_static_pass_live_fail_project <dir> - High static score, fails --live
  create_warning_only_project <dir>       - Console warnings only (PASS --live)
  create_slow_project <dir>               - 5s init delay (tests timeout)

Example:
  source fixtures-serve-score.sh
  create_valid_project /tmp/test-project
  probar serve score /tmp/test-project --verbose

  # Test --live validation
  create_static_pass_live_fail_project /tmp/critical
  probar serve score /tmp/critical --format json | jq .grade  # High
  probar serve score /tmp/critical --live                      # FAIL
USAGE
fi
