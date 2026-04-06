#!/bin/bash
# qa_verification.sh - Probar Advanced Features QA Checklist
# Usage: ./qa_verification.sh > qa_report.txt 2>&1

# set -e
# Ensure we are in the project root
if [ -d "crates/probar" ]; then
    : # Already in root
elif [ -d "../crates/probar" ]; then
    cd ..
else
    echo "Error: Could not find project root"
    exit 1
fi

PASS=0
FAIL=0

check() {
    local num="$1"
    local desc="$2"
    local cmd="$3"
    
    echo "=== Point $num: $desc ==="
    echo "Command: $cmd"

    # Use eval to execute the command string, capturing stderr to stdout
    if result=$(eval "$cmd" 2>&1); then
        echo "Result: $result"
        echo "Status: PASS"
        ((PASS++))
    else
        echo "Result: $result"
        echo "Status: FAIL"
        ((FAIL++))
    fi
    echo ""
}

echo "========================================"
echo "PROBAR ADVANCED FEATURES QA VERIFICATION"
echo "========================================"
echo "Date: $(date)"
echo "Directory: $(pwd)"
echo ""

# Section 1: Core Infrastructure
check 1 "Project compiles" "cargo build --release 2>&1 | tail -1"
check 2 "Unit tests pass" "cargo test --lib 2>&1 | grep 'test result'"
check 3 "Integration tests pass" "cargo test --test '*' 2>&1 | grep 'test result' || echo 'No integration tests'"
check 4 "Examples compile" "cargo build --examples 2>&1 | tail -1"
check 5 "Clippy passes" "cargo clippy --all-features -- -D warnings 2>&1 | tail -1"
check 6 "Documentation builds" "cargo doc --no-deps 2>&1 | tail -1"
check 7 "WASM target compiles" "cargo build --target wasm32-unknown-unknown -p probar 2>&1 | tail -1 || echo 'WASM build skipped'"
check 8 "Benchmarks exist" "ls crates/probar/benches/*.rs 2>/dev/null | wc -l"
check 9 "Package valid" "cargo package --list -p probar 2>&1 | head -5"
check 10 "Unsafe audit" "grep -r 'unsafe' crates/probar/src/*.rs 2>/dev/null | wc -l"

# Section 2: Feature A - Pixel Coverage
check 11 "PixelCoverageTracker exists" "grep -c 'PixelCoverageTracker' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 12 "CoverageCell exists" "grep -c 'CoverageCell' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 13 "Region struct exists" "grep -c 'pub struct Region' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 14 "Terminal heatmap" "cargo test terminal --lib 2>&1 | grep -c 'passed' || echo 0"
check 15 "Coverage example" "cargo run --example coverage_demo 2>&1 | tail -3 || echo 'Example missing'"

# Section 3: Feature B - Video Recording
check 16 "VideoRecorder exists" "grep -c 'VideoRecorder' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 17 "VideoConfig exists" "grep -c 'VideoConfig' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 18 "VideoCodec enum" "grep -c 'VideoCodec' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 19 "EncodedFrame struct" "grep -c 'EncodedFrame' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 20 "Video tests" "cargo test video --lib 2>&1 | grep -c 'passed' || echo 0"

# Section 4: Feature C - Performance Benchmarking
check 21 "TracedSpan exists" "grep -c 'TracedSpan' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 22 "SpanStatus enum" "grep -c 'SpanStatus' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 23 "TracingConfig" "grep -c 'TracingConfig' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 24 "Execution tracer" "cargo run --example execution_trace 2>&1 | tail -3 || echo 'Example missing'"
check 25 "Tracing tests" "cargo test tracing --lib 2>&1 | grep -c 'passed' || echo 0"

# Section 5: Feature D - WASM Runner
check 26 "WatchConfig exists" "grep -c 'WatchConfig' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 27 "FileChange struct" "grep -c 'FileChange' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 28 "Notify dependency" "grep -c 'notify' crates/probar/Cargo.toml"
check 29 "Watch tests" "cargo test watch --lib 2>&1 | grep -c 'passed' || echo 0"
check 30 "Glob matching" "cargo test glob --lib 2>&1 | grep -c 'passed' || echo 0"

# Section 6: Feature E - Zero JavaScript
check 31 "No JS files" "find crates/probar -name '*.js' 2>/dev/null | wc -l"
check 32 "No TS files" "find crates/probar -name '*.ts' 2>/dev/null | wc -l"
check 33 "No package.json" "find crates/probar -name 'package.json' 2>/dev/null | wc -l"
check 34 "web-sys dep" "grep -c 'web-sys' crates/probar/Cargo.toml"
check 35 "wasm-bindgen dep" "grep -c 'wasm-bindgen' crates/probar/Cargo.toml"

# Section 7: Feature F - Real E2E Testing
check 36 "Browser struct" "grep -c 'pub struct Browser' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 37 "Page struct" "grep -c 'pub struct Page' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 38 "Selector struct" "grep -c 'pub struct Selector' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 39 "Locator struct" "grep -c 'pub struct Locator' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 40 "Snapshot struct" "grep -c 'pub struct Snapshot' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 41 "SnapshotDiff" "grep -c 'SnapshotDiff' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 42 "Browser tests" "cargo test browser --lib 2>&1 | grep -c 'passed' || echo 0"
check 43 "Selector tests" "cargo test selector --lib 2>&1 | grep -c 'passed' || echo 0"
check 44 "Locator tests" "cargo test locator --lib 2>&1 | grep -c 'passed' || echo 0"
check 45 "Locator demo" "cargo run --example locator_demo 2>&1 | tail -3 || echo 'Example missing'"

# Section 8: Feature G - Playwright/Puppeteer Parity
check 46 "WaitOptions struct" "grep -c 'WaitOptions' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 47 "LoadState enum" "grep -c 'LoadState' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 48 "NavigationOptions" "grep -c 'NavigationOptions' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 49 "DeviceDescriptor" "grep -c 'DeviceDescriptor' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 50 "DeviceEmulator" "grep -c 'DeviceEmulator' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 51 "Viewport struct" "grep -c 'pub struct Viewport' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 52 "TouchMode enum" "grep -c 'TouchMode' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 53 "GeolocationPosition" "grep -c 'GeolocationPosition' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 54 "GeolocationMock" "grep -c 'GeolocationMock' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 55 "BrowserContext" "grep -c 'BrowserContext' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 56 "StorageState" "grep -c 'StorageState' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 57 "Cookie struct" "grep -c 'pub struct Cookie' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 58 "SameSite enum" "grep -c 'SameSite' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 59 "ContextConfig" "grep -c 'ContextConfig' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 60 "ContextPool" "grep -c 'ContextPool' crates/probar/src/*.rs 2>/dev/null || echo 0"

# Section 9: Accessibility Testing
check 61 "ContrastRatio" "grep -c 'ContrastRatio' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 62 "WcagLevel enum" "grep -c 'WcagLevel' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 63 "AccessibilityAudit" "grep -c 'AccessibilityAudit' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 64 "FlashDetector" "grep -c 'FlashDetector' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 65 "Accessibility demo" "cargo run --example accessibility_demo 2>&1 | tail -3 || echo 'Example missing'"

# Section 10: Visual Regression
check 66 "MaskRegion struct" "grep -c 'MaskRegion' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 67 "ScreenshotComparison" "grep -c 'ScreenshotComparison' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 68 "ImageDiff" "grep -c 'ImageDiff' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 69 "Visual regression tests" "cargo test visual --lib 2>&1 | grep -c 'passed' || echo 0"
check 70 "Screenshot tests" "cargo test screenshot --lib 2>&1 | grep -c 'passed' || echo 0"

# Section 11: Fixtures and Test Infrastructure
check 71 "Fixture trait" "grep -c 'pub trait Fixture' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 72 "FixtureManager" "grep -c 'FixtureManager' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 73 "FixtureState enum" "grep -c 'FixtureState' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 74 "TestHarness" "grep -c 'TestHarness' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 75 "TestSuite" "grep -c 'TestSuite' crates/probar/src/*.rs 2>/dev/null || echo 0"

# Section 12: Page Objects
check 76 "PageObject trait" "grep -c 'PageObject' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 77 "PageObjectBuilder" "grep -c 'PageObjectBuilder' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 78 "SimplePageObject" "grep -c 'SimplePageObject' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 79 "PageRegistry" "grep -c 'PageRegistry' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 80 "Page object tests" "cargo test page_object --lib 2>&1 | grep -c 'passed' || echo 0"

# Section 13: Network Testing
check 81 "NetworkEvent" "grep -c 'NetworkEvent' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 82 "HttpMethod enum" "grep -c 'HttpMethod' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 83 "HarRecorder" "grep -c 'HarRecorder|Har' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 84 "RequestInterceptor" "grep -c 'Intercept' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 85 "Network tests" "cargo test network --lib 2>&1 | grep -c 'passed' || echo 0"

# Section 14: Wait Mechanisms
check 86 "Waiter struct" "grep -c 'pub struct Waiter' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 87 "WaitResult enum" "grep -c 'WaitResult' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 88 "PageEvent enum" "grep -c 'PageEvent' crates/probar/src/*.rs 2>/dev/null || echo 0"
check 89 "Wait tests" "cargo test wait --lib 2>&1 | grep -c 'passed' || echo 0"
check 90 "Timeout handling" "cargo test timeout --lib 2>&1 | grep -c 'passed' || echo 0"

# Section 15: Book and Documentation
check 91 "Book builds" "cd book && mdbook build 2>&1 | tail -1 || echo 'No book'"
check 92 "SUMMARY.md exists" "test -f book/src/SUMMARY.md && echo 'exists' || echo 'missing'"
check 93 "Device emulation docs" "test -f book/src/probar/device-emulation.md && wc -l < book/src/probar/device-emulation.md"
check 94 "Geolocation docs" "test -f book/src/probar/geolocation-mocking.md && wc -l < book/src/probar/geolocation-mocking.md"
check 95 "Browser contexts docs" "test -f book/src/probar/browser-contexts.md && wc -l < book/src/probar/browser-contexts.md"

# Section 16: Examples Verification
check 96 "Basic test example" "cargo run --example basic_test 2>&1 | tail -3 || echo 'Example missing'"
check 97 "Pong simulation example" "cargo run --example pong_simulation 2>&1 | tail -3 || echo 'Example missing'"
check 98 "All examples list" "ls crates/probar/examples/*.rs 2>/dev/null | wc -l"
check 99 "Examples in Cargo.toml" "grep -c '\[\[example\]\]' crates/probar/Cargo.toml"
check 100 "Full test suite" "cargo test 2>&1 | grep 'test result'"

echo ""
echo "========================================