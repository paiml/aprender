# Runtime Validation

Runtime validation ensures your WASM application actually works, not just that test artifacts exist. This prevents false confidence from "100% score" on broken applications.

## The Problem

A project can score 100/100 while having fatal runtime bugs:

```
Score: 100/100 (A grade)
App: BROKEN (404 on WASM module)
```

This happens because traditional scoring measures **test infrastructure presence**, not **application health**.

## Solution: Module Validation

Probar validates that all module imports resolve correctly before serving or scoring.

### Validate Before Serving

```bash
# Validate all imports before starting server
probar serve -d ./www --validate

# Output on success:
# Validating module imports...
#   Scanned: 15 imports
#   Passed:  15
#   Failed:  0
#
# ✓ All module imports validated successfully

# Output on failure:
# Validating module imports...
#   Scanned: 15 imports
#   Passed:  14
#   Failed:  1
#
# Errors:
#   ✗ /index.html:23
#     Import: /assets/missing.js
#     File not found: /srv/www/assets/missing.js
#
# Error: Module validation failed: 1 error(s) found.
```

### Exclude Directories

Skip validation for third-party directories like `node_modules`:

```bash
# node_modules is excluded by default
probar serve -d ./www --validate

# Add custom exclusions
probar serve -d ./www --validate --exclude vendor --exclude dist
```

## What Gets Validated

The validator scans HTML files for:

| Import Type | Pattern | Example |
|-------------|---------|---------|
| ES Module | `import ... from '...'` | `import { foo } from './app.js'` |
| Script src | `<script src="...">` | `<script src="./main.js">` |
| Worker URL | `new Worker('...')` | `new Worker('./worker.js')` |

For each import, the validator checks:
1. File exists at the resolved path
2. MIME type is correct (e.g., `text/javascript` for `.js`)

## Runtime Health Score

Runtime validation is integrated into the project score as a **15-point mandatory category**:

| Criteria | Points | What It Checks |
|----------|--------|----------------|
| Module Resolution | 5 | All imports resolve to existing files |
| Critical Assets | 5 | No 404 errors on required files |
| MIME Types | 5 | Correct content types served |

### Grade Capping

**Key feature:** If Runtime Health fails, the grade is capped at **C** regardless of other scores.

```
Before: 106/115 (92%) → Grade A
After:  106/115 (92%) → Grade C (capped)
        Runtime Health: 7/15 (FAIL)
```

This prevents false confidence from high scores on broken applications.

## Integration with Score Command

Runtime validation runs automatically during `probar score`:

```bash
probar score -d ./project

# Output includes:
# ═══════════════════════════════════════════════════
#  SCORE: 72/100 (C)
#
#  Runtime Health: 7/15 (Partial)
#    ✓ Module imports (3/5)
#    ✓ Critical assets (2/5)
#    ✓ MIME types (2/5)
#
#  GRADE CAPPED: Runtime validation failed
#  Fix: Resolve broken import paths
# ═══════════════════════════════════════════════════
```

## Common Issues

### 1. Wrong Base Path

```html
<!-- WRONG: Path assumes different serve root -->
<script type="module" src="/demos/app/pkg/module.js"></script>

<!-- CORRECT: Path relative to actual serve root -->
<script type="module" src="/pkg/module.js"></script>
```

### 2. Missing WASM File

```
✗ /index.html:15
  Import: ./pkg/app_bg.wasm
  File not found: /srv/www/pkg/app_bg.wasm

Fix: Run wasm-pack build before serving
```

### 3. MIME Type Mismatch

```
✗ /index.html:10
  Import: ./app.js
  MIME mismatch: expected ["text/javascript"], got "text/plain"

Fix: Configure server to serve .js with correct MIME type
```

## API Reference

### ModuleValidator

```rust
use probador::ModuleValidator;

// Create validator
let validator = ModuleValidator::new("./www");

// Add exclusions (node_modules excluded by default)
let validator = validator.with_exclude(vec!["vendor".to_string()]);

// Run validation
let result = validator.validate();

// Check results
if result.is_ok() {
    println!("All {} imports validated", result.passed);
} else {
    for error in &result.errors {
        println!("Error: {}", error.message);
    }
}
```

### ModuleValidationResult

```rust
pub struct ModuleValidationResult {
    pub total_imports: usize,
    pub passed: usize,
    pub errors: Vec<ImportValidationError>,
}

impl ModuleValidationResult {
    /// Returns true if all imports validated successfully
    pub fn is_ok(&self) -> bool;
}
```

## Best Practices

1. **Always validate in CI**: Add `--validate` to your CI pipeline
2. **Fix before deploying**: Never deploy with validation errors
3. **Check after wasm-pack**: Validate after rebuilding WASM
4. **Exclude appropriately**: Skip node_modules but validate your code

## See Also

- [Dev Server](./dev-server.md) - Serving WASM applications
- [Project Score](./project-score.md) - Understanding the scoring system
- [CLI Reference](./cli-reference.md) - Full command documentation
