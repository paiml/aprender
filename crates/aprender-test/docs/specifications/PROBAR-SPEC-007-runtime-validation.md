# PROBAR-SPEC-007: Runtime Validation

## Problem Statement

**Critical Failure:** A project can score 100/100 while having fatal runtime bugs that prevent the application from working at all.

### Root Cause Analysis

The current scoring system measures **test infrastructure presence**, not **application health**:

| What We Score | What We Should Score |
|---------------|----------------------|
| Playbook YAML files exist | Playbooks actually run successfully |
| Snapshot PNG files exist | App renders correctly |
| Config files present | App loads without errors |
| Documentation exists | Module imports resolve |

### Real-World Failure Case

whisper.apr demo scored **100/100** with this bug in production:

```rust
// WRONG: Path doesn't exist when serving from demos/www/
let wasm_url = "/demos/realtime-transcription/pkg/module.js";

// CORRECT: Relative to serve root
let wasm_url = "/realtime-transcription/pkg/module.js";
```

Result:
- Browser received 404 → `text/plain` MIME type
- Module import blocked: "disallowed MIME type"
- **App completely non-functional**
- **Score: 100/100 (A grade)**

This is worse than no score at all. It provides false confidence.

---

## Specification

### 1. Runtime Health Category (NEW - 15 points)

Add a new mandatory category that requires **actual runtime validation**:

```rust
pub struct RuntimeHealthScore {
    /// Module resolution (5 points)
    /// - All JS imports resolve (200 OK)
    /// - All WASM imports resolve (200 OK)
    /// - Correct MIME types returned
    module_resolution: u32,

    /// Application bootstrap (5 points)
    /// - WASM initializes without errors
    /// - No console errors during load
    /// - Main entry point executes
    app_bootstrap: u32,

    /// Critical path validation (5 points)
    /// - Core functionality exercisable
    /// - No runtime exceptions in happy path
    /// - State machine reaches expected states
    critical_path: u32,
}
```

### 2. Serve Command Enhancements

#### 2.1 Startup Validation (`probar serve --validate`)

Before serving, scan HTML files and validate all imports:

```
$ probar serve -d ./www --validate

Validating module imports...
  ✓ /index.html
    ✓ ./pkg/app.js (200 OK, text/javascript)
    ✓ ./pkg/app_bg.wasm (200 OK, application/wasm)
  ✗ /realtime.html
    ✗ /demos/pkg/module.js (404 Not Found)

ERROR: 1 broken import(s) found. Fix before serving.
```

#### 2.2 Request Monitoring (`probar serve --monitor`)

Track all requests and flag issues in real-time:

```
$ probar serve -d ./www --monitor

[INFO] GET /index.html → 200 (text/html)
[INFO] GET /pkg/app.js → 200 (text/javascript)
[WARN] GET /demos/missing.js → 404 (text/plain)
       ^ Module import will fail - check import paths

[ERROR] MIME mismatch: /pkg/data.js served as text/plain
        Expected: text/javascript or application/javascript
```

#### 2.3 Health Endpoint (`GET /__probar__/health`)

Expose runtime health via API:

```json
{
  "status": "unhealthy",
  "checks": {
    "module_resolution": {
      "status": "fail",
      "errors": [
        {
          "path": "/demos/pkg/module.js",
          "expected": "200 OK",
          "actual": "404 Not Found"
        }
      ]
    },
    "mime_types": {
      "status": "pass"
    }
  }
}
```

### 3. Score Command Enhancements

#### 3.1 Runtime Score Requirement

The score command MUST start a server and validate runtime:

```
$ probar score -d ./project

Starting validation server on port 18741...

[1/4] Static Analysis
  ✓ Playbook files: 3 found
  ✓ Snapshot files: 12 found
  ✓ Config files: 5 found

[2/4] Module Resolution
  ✓ Scanning HTML files for imports...
  ✗ /realtime.html imports /demos/pkg/module.js
    → 404 Not Found (should be /pkg/module.js)

[3/4] Application Bootstrap
  ✗ Cannot test - module resolution failed

[4/4] Critical Path
  ✗ Cannot test - bootstrap failed

═══════════════════════════════════════════════════
  SCORE: 72/100 (C)

  BLOCKING ISSUES:
  - Module resolution failed (0/5 points)
  - App bootstrap failed (0/5 points)
  - Critical path untested (0/5 points)
═══════════════════════════════════════════════════

A score above 85 requires passing runtime validation.
```

#### 3.2 Grade Caps

Runtime failures impose grade caps regardless of other scores:

| Runtime Status | Maximum Grade |
|----------------|---------------|
| Module resolution fails | C (max 79) |
| App bootstrap fails | D (max 69) |
| Critical path fails | C (max 79) |
| All runtime checks pass | A (no cap) |

### 4. Implementation Requirements

#### 4.1 Module Resolution Validator

```rust
pub struct ModuleValidator {
    serve_root: PathBuf,
}

impl ModuleValidator {
    /// Scan all HTML files and extract JS/WASM imports
    pub fn scan_imports(&self) -> Vec<ImportRef>;

    /// Validate each import resolves with correct MIME type
    pub fn validate(&self) -> ValidationResult;
}

pub struct ImportRef {
    pub source_file: PathBuf,
    pub import_path: String,
    pub import_type: ImportType, // ES Module, Script, WASM
    pub line_number: u32,
}

pub enum ValidationResult {
    Pass,
    Fail(Vec<ValidationError>),
}

pub struct ValidationError {
    pub import: ImportRef,
    pub expected_status: u16,
    pub actual_status: u16,
    pub expected_mime: String,
    pub actual_mime: String,
}
```

#### 4.2 Bootstrap Validator

```rust
pub struct BootstrapValidator {
    page_url: String,
}

impl BootstrapValidator {
    /// Load page in headless browser and check for errors
    pub async fn validate(&self) -> BootstrapResult;
}

pub struct BootstrapResult {
    pub console_errors: Vec<String>,
    pub uncaught_exceptions: Vec<String>,
    pub wasm_init_success: bool,
    pub load_time_ms: u64,
}
```

#### 4.3 Critical Path Validator

```rust
pub struct CriticalPathValidator {
    playbook: Playbook,
}

impl CriticalPathValidator {
    /// Execute first N steps of playbook and verify no errors
    pub async fn validate(&self, steps: usize) -> CriticalPathResult;
}
```

### 5. Point Redistribution

Current distribution (100 points):
- Playbook Coverage: 15
- Pixel Testing: 13
- GUI Interaction: 13
- Performance Benchmarks: 14
- Load Testing: 10
- Deterministic Replay: 10
- Cross-Browser: 10
- Accessibility: 10
- Documentation: 5

New distribution (100 points):
- **Runtime Health: 15** (NEW - mandatory)
- Playbook Coverage: 12 (-3)
- Pixel Testing: 10 (-3)
- GUI Interaction: 10 (-3)
- Performance Benchmarks: 12 (-2)
- Load Testing: 8 (-2)
- Deterministic Replay: 10
- Cross-Browser: 10
- Accessibility: 8 (-2)
- Documentation: 5

### 6. CLI Changes

```bash
# Validate before serving
probar serve -d ./www --validate

# Validate with exclusions (skip node_modules, vendor, etc.)
probar serve -d ./www --validate --exclude vendor

# Note: node_modules is excluded by default

# Monitor requests while serving
probar serve -d ./www --monitor

# Score with runtime validation (default)
probar score -d ./project

# Score static-only (legacy, discouraged)
probar score -d ./project --static-only

# Validate module resolution only
probar validate -d ./project --modules

# Full validation suite
probar validate -d ./project --all
```

### 7. Success Criteria

1. A project with broken module imports CANNOT score above 79 (C)
2. A project that fails to bootstrap CANNOT score above 69 (D)
3. `probar serve --validate` catches the `/demos/` path bug
4. `probar score` requires runtime validation by default
5. Zero false positives: A 100% score means the app works

### 8. Migration Path

1. Add `--validate` flag to serve (opt-in initially)
2. Add Runtime Health category to score
3. Implement grade caps
4. Make `--validate` default after 1 release cycle
5. Deprecate `--static-only` scoring

---

## Rationale

**Why runtime validation is mandatory:**

The purpose of a testing tool is to prevent bugs from reaching users. A scoring system that gives 100% to broken applications is actively harmful - it provides false confidence and discourages actual testing.

**Core principle:** If it doesn't run, it doesn't pass.

**Corollary:** Points for "having test files" are worthless without points for "tests that pass."

---

## References

- [whisper.apr MIME type bug](../../../whisper.apr/demos/realtime-transcription/src/lib.rs:558)
- Current scoring implementation: `crates/probar-cli/src/score.rs`
- Dev server implementation: `crates/probar-cli/src/dev_server.rs`
