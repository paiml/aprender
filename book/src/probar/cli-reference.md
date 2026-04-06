# CLI Reference

Command-line interface reference for **probador** - the CLI tool for Probar.

## Installation

```bash
cargo install probador
```

Or build from source:

```bash
cargo build --release -p probador
```

## Commands

### probador test

Run tests with optional coverage and filtering.

```bash
# Run all tests
probador test

# Filter tests by pattern
probador test --filter "game::*"

# Run with coverage
probador test --coverage

# Parallel execution
probador test -j 4

# Fail fast on first error
probador test --fail-fast

# Watch mode (re-run on changes)
probador test --watch

# Custom timeout (ms)
probador test --timeout 60000

# Custom output directory
probador test --output target/my-tests
```

### probador coverage

Generate pixel coverage heatmaps and reports.

```bash
# Generate PNG heatmap
probador coverage --png output.png

# Choose color palette (viridis, magma, heat)
probador coverage --png output.png --palette magma

# Add legend and gap highlighting
probador coverage --png output.png --legend --gaps

# Add title
probador coverage --png output.png --title "My Coverage Report"

# Custom dimensions
probador coverage --png output.png --width 1920 --height 1080

# Export JSON report
probador coverage --json report.json

# Full example
probador coverage --png heatmap.png \
  --palette viridis \
  --legend \
  --gaps \
  --title "Sprint 42 Coverage" \
  --width 800 \
  --height 600
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--png <path>` | Export PNG heatmap | - |
| `--json <path>` | Export JSON report | - |
| `--palette <name>` | Color palette (viridis/magma/heat) | viridis |
| `--legend` | Show color legend | false |
| `--gaps` | Highlight gaps in red | false |
| `--title <text>` | Title text | - |
| `--width <px>` | PNG width | 800 |
| `--height <px>` | PNG height | 600 |
| `-i, --input <path>` | Input coverage data (JSON) | - |

### probador record

Record test execution to media files.

```bash
# Record as GIF (default)
probador record test_login

# Record as PNG screenshots
probador record test_login --format png

# Custom output path
probador record test_login --output recording.gif

# Set frame rate
probador record test_login --fps 30

# Set quality (1-100)
probador record test_login --quality 90
```

**Formats:** `gif`, `png`, `svg`, `mp4`

### probador report

Generate test reports in various formats.

```bash
# HTML report (default)
probador report

# Specific format
probador report --format lcov
probador report --format junit
probador report --format cobertura
probador report --format json

# Custom output directory
probador report --output target/reports

# Open in browser after generation
probador report --open
```

**Formats:** `html`, `junit`, `lcov`, `cobertura`, `json`

### probador init

Initialize a new Probar project.

```bash
# Initialize in current directory
probador init

# Initialize in specific path
probador init ./my-project

# Force overwrite existing files
probador init --force
```

### probador config

View and manage configuration.

```bash
# Show current configuration
probador config --show

# Set a configuration value
probador config --set "parallel=4"

# Reset to defaults
probador config --reset
```

### probador serve

Start a WASM development server with hot reload support.

```bash
# Serve current directory on port 8080
probador serve

# Serve a specific directory
probador serve ./www

# Custom port
probador serve --port 3000

# Enable CORS for cross-origin requests
probador serve --cors

# Open browser automatically
probador serve --open

# Validate module imports before serving
probador serve --validate

# Validate with custom exclusions (node_modules excluded by default)
probador serve --validate --exclude vendor --exclude dist

# Monitor requests in real-time
probador serve --monitor

# Full example
probador serve ./dist --port 8080 --cors --open --validate
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `<directory>` | Directory to serve | `.` |
| `-p, --port <port>` | HTTP port | 8080 |
| `--ws-port <port>` | WebSocket port for hot reload | 8081 |
| `--cors` | Enable CORS | false |
| `--open` | Open browser automatically | false |
| `--validate` | Validate module imports before serving | false |
| `--monitor` | Monitor requests and warn about issues | false |
| `--exclude <dir>` | Exclude directories from validation (repeatable) | node_modules |

**Features:**
- Serves WASM files with correct `application/wasm` MIME type
- WebSocket endpoint at `/ws` for hot reload notifications
- Automatic CORS headers when enabled
- No-cache headers for development
- Module import validation (catches broken imports before serving)
- Request monitoring (shows 404s and MIME mismatches in real-time)

### probador serve tree

Visualize files being served as an ASCII tree.

```bash
# Show file tree
probador serve tree

# Limit depth
probador serve tree --depth 2

# Filter by pattern
probador serve tree --filter "*.wasm"

# Show specific directory
probador serve tree ./www
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `<path>` | Directory to visualize | `.` |
| `--depth <n>` | Maximum depth | unlimited |
| `--filter <glob>` | Filter files by pattern | - |
| `--sizes` | Show file sizes | true |
| `--mime` | Show MIME types | true |

### probador serve score

Generate project testing score (100-point evaluation).

```bash
# Generate score
probador serve score

# Detailed breakdown
probador serve score --verbose

# CI gate (exit non-zero if below threshold)
probador serve score --min 80

# JSON output
probador serve score --format json

# Binary report (view with TUI)
probador serve score --report score.msgpack

# Track history
probador serve score --history scores.jsonl

# Show trend
probador serve score --trend
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `<path>` | Project directory | `.` |
| `--verbose` | Show detailed breakdown | false |
| `--min <score>` | Minimum required score | - |
| `--format <fmt>` | Output format (console/json) | console |
| `--report <path>` | Generate HTML report | - |
| `--history <path>` | Append to JSONL history | - |
| `--trend` | Show score trend chart | false |

### probador lint

Lint HTML, CSS, JavaScript, and WASM files.

```bash
# Lint all files
probador lint

# Lint specific types
probador lint --html --css

# Lint specific directory
probador lint ./www
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `<path>` | Directory to lint | `.` |
| `--html` | Lint HTML files | true |
| `--css` | Lint CSS files | true |
| `--js` | Lint JavaScript files | true |
| `--wasm` | Validate WASM modules | true |

### probador load-test

Run load tests against a server.

```bash
# Basic load test
probador load-test --url http://localhost:8080 --users 100 --duration 30s

# Ramp-up test
probador load-test --url http://localhost:8080 --users 1-100 --ramp 60s --duration 120s

# Scenario-based test
probador load-test --scenario scenarios/boot.yaml

# Output formats
probador load-test --scenario test.yaml --format json
probador load-test --scenario test.yaml --report report.msgpack
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--url <url>` | Target URL | - |
| `--users <n>` | Concurrent users | 10 |
| `--users <n1>-<n2>` | Ramp users | - |
| `--ramp <duration>` | Ramp-up duration | - |
| `--duration <duration>` | Test duration | 30s |
| `--scenario <file>` | Load scenario YAML | - |
| `--format <fmt>` | Output format (console/json/html) | console |
| `--report <path>` | Generate HTML report | - |
| `--timeout <ms>` | Request timeout | 30000 |

### probador build

Build a Rust project to WASM using wasm-pack.

```bash
# Build in development mode
probador build

# Build in release mode
probador build --release

# Specify build target
probador build --target web
probador build --target bundler
probador build --target nodejs

# Custom output directory
probador build --out-dir ./dist

# Enable profiling (adds names section)
probador build --profiling

# Full example
probador build ./my-game --target web --release --out-dir ./www/pkg
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `<path>` | Package directory | `.` |
| `-t, --target <target>` | WASM target (web/bundler/nodejs/no-modules) | web |
| `--release` | Build in release mode | false |
| `-o, --out-dir <path>` | Output directory | pkg |
| `--profiling` | Enable profiling | false |

### probador watch

Watch for file changes and rebuild automatically.

```bash
# Watch current directory
probador watch

# Watch with dev server
probador watch --serve

# Custom port when serving
probador watch --serve --port 3000

# Build in release mode
probador watch --release

# Custom debounce delay
probador watch --debounce 1000

# Full example
probador watch ./my-game --serve --port 8080 --target web
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `<path>` | Directory to watch | `.` |
| `--serve` | Also start dev server | false |
| `-p, --port <port>` | Server port (with --serve) | 8080 |
| `--ws-port <port>` | WebSocket port | 8081 |
| `-t, --target <target>` | WASM target | web |
| `--release` | Build in release mode | false |
| `--debounce <ms>` | Debounce delay | 500 |

**Watched files:** `.rs`, `.toml`

### probador playbook

Run YAML-driven state machine playbook tests with validation and mutation testing.

```bash
# Validate a playbook
probador playbook login.yaml --validate

# Run multiple playbooks
probador playbook login.yaml checkout.yaml profile.yaml

# Export state diagram as SVG
probador playbook login.yaml --export svg --export-output diagram.svg

# Export as DOT (Graphviz)
probador playbook login.yaml --export dot --export-output diagram.dot

# Run mutation testing (M1-M5)
probador playbook login.yaml --mutate

# Run specific mutation classes
probador playbook login.yaml --mutate --mutation-classes M1,M2,M3

# JSON output for CI integration
probador playbook login.yaml --format json

# JUnit XML for test reporting
probador playbook login.yaml --format junit

# Fail fast on first error
probador playbook login.yaml --fail-fast

# Full example
probador playbook tests/*.yaml \
  --validate \
  --mutate \
  --mutation-classes M1,M2,M5 \
  --format json \
  --output results/
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `<files>...` | Playbook YAML file(s) | (required) |
| `--validate` | Validate without executing | false |
| `--export <format>` | Export diagram (dot/svg) | - |
| `--export-output <path>` | Diagram output file | - |
| `--mutate` | Run mutation testing | false |
| `--mutation-classes <M>` | Mutation classes (M1-M5) | all |
| `--fail-fast` | Stop on first error | false |
| `--continue-on-error` | Continue on step failure | false |
| `-f, --format <format>` | Output format (text/json/junit) | text |
| `-o, --output <dir>` | Output directory | target/probar/playbooks |

**Mutation Classes:**

| Class | Description |
|-------|-------------|
| M1 | State removal |
| M2 | Transition removal |
| M3 | Event swap |
| M4 | Target swap |
| M5 | Guard negation |

### probador comply

Check WASM testing compliance with 10-point validation checklist.

```bash
# Basic compliance check
probador comply .

# Detailed output with explanations
probador comply --detailed .

# Strict mode (exit non-zero if any check fails)
probador comply --strict .

# JSON output for CI integration
probador comply --format json .

# Specify custom WASM size limit
probador comply --max-wasm-size 10000000 .

# Run specific checks only
probador comply --checks C001,C002,C005 .

# Fail fast on first failure
probador comply --fail-fast .

# Generate JUnit XML report
probador comply --format junit --report compliance.xml .

# Full example
probador comply --detailed --strict --format text .
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `<path>` | Project directory | `.` |
| `--checks <list>` | Run specific checks (comma-separated) | all |
| `--fail-fast` | Stop on first failure | false |
| `-f, --format <fmt>` | Output format (text/json/junit) | text |
| `--max-wasm-size <bytes>` | WASM binary size limit | 5242880 (5MB) |
| `--strict` | Require WasmStrictMode checks | false |
| `--report <path>` | Output report file | - |
| `--detailed` | Show detailed explanations | false |

**Compliance Checks:**

| Check | Description |
|-------|-------------|
| C001 | Code execution verified (not just DOM presence) |
| C002 | Console errors fail tests |
| C003 | Custom elements registration tested |
| C004 | Threading modes tested (both paths) |
| C005 | Low memory conditions tested |
| C006 | COOP/COEP headers configured |
| C007 | Deterministic replay hash validation |
| C008 | Service worker cache handling |
| C009 | WASM binary size limit |
| C010 | No panic paths (unwrap/expect) |

**Example Output:**

```
══════════════════════════════════════════════════════════════
  PROBAR COMPLY - WASM Compliance Checker
══════════════════════════════════════════════════════════════

Running 10 compliance check(s) on .

  [✓] C001: Code execution verified
  [✓] C002: Console errors fail tests
  [✓] C003: Custom elements tested
  [✓] C004: Threading modes tested
  [✓] C005: Low memory tested
  [✗] C006: COOP/COEP headers (not configured)
  [✓] C007: Replay hash matches
  [✓] C008: Cache handling
  [✓] C009: WASM size limit
  [✓] C010: No panic paths

══════════════════════════════════════════════════════════════
  Result: 9/10 checks passed
══════════════════════════════════════════════════════════════
```

## Global Options

These options work with all commands:

```bash
# Verbose output (-v, -vv, -vvv for more detail)
probador -v test
probador -vvv test

# Quiet mode (suppress non-error output)
probador -q test

# Color output (auto, always, never)
probador --color never test
probador --color always report
```

## Examples

### Basic Test Run

```bash
probador test
```

### Coverage with Heatmap

```bash
# Run tests with coverage
probador test --coverage

# Generate heatmap
probador coverage --png coverage.png --legend --gaps --title "Test Coverage"
```

### CI/CD Pipeline

```bash
# Run tests, fail fast, generate reports
probador test --fail-fast --coverage
probador report --format lcov --output coverage/
probador report --format junit --output test-results/
probador coverage --json coverage/pixel-report.json
```

### Watch Mode Development

```bash
# Run tests on file changes
probador test --watch --filter "unit::*"
```

### WASM Development Workflow

```bash
# Build WASM package
probador build --target web --release

# Start dev server with hot reload
probador serve ./www --port 8080 --cors

# Or combine watch + serve for full development experience
probador watch --serve --port 8080
```

### Playbook State Machine Testing

```bash
# Validate playbook
probador playbook login.yaml --validate

# Export diagram
probador playbook login.yaml --export svg -o login.svg

# Run mutation testing
probador playbook login.yaml --mutate
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Test failure(s) |
| 2 | Configuration error |
| 3 | I/O error |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PROBAR_COLOR` | Color output (auto/always/never) |
| `PROBAR_PARALLEL` | Default parallel jobs |
| `PROBAR_TIMEOUT` | Default test timeout (ms) |

## Library Usage

For programmatic usage in Rust code, use the library crate:

```bash
cargo add jugar-probar --dev
```

```rust
use jugar_probar::prelude::*;
```

See [API Reference](./api-reference.md) for library documentation.
