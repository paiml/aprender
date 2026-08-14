# Zero-JS Validation

Probar provides Zero-JS validation for WASM-first applications, ensuring NO user-generated JavaScript, CSS, or HTML exists in your WASM applications (PROBAR-SPEC-012).

## Overview

Zero-JS validation enforces WASM-first architecture by detecting:

1. **Unauthorized JavaScript files** - Any `.js`, `.ts`, `.jsx`, `.tsx`, `.mjs`, `.cjs` files
2. **Forbidden directories** - `node_modules`, `dist`, `build` containing JS tooling
3. **Inline scripts in HTML** - `<script>` tags and event handlers
4. **Dangerous patterns** - `eval()`, `new Function()`, `innerHTML`, `document.write()`
5. **Tooling files** - `package.json`, `package-lock.json`, etc.

## Quick Start

```rust
use jugar_probar::zero_js::{ZeroJsValidator, ZeroJsConfig};

let validator = ZeroJsValidator::new();
let result = validator.validate_directory("./pkg")?;

assert!(result.is_valid(), "Zero-JS validation failed: {}", result);
```

## Configuration

### Default Configuration

```rust
let validator = ZeroJsValidator::new();
// Allows WASM-generated inline scripts
// Does not require manifest
// Checks dangerous patterns
```

### Strict Mode

```rust
let config = ZeroJsConfig::strict();
let validator = ZeroJsValidator::with_config(config);

// strict() enforces:
// - require_manifest: true
// - check_dangerous_patterns: true
// - allow_wasm_inline_scripts: false
// - forbid_node_modules: true
// - forbid_package_json: true
```

### Custom Configuration

```rust
let config = ZeroJsConfig {
    allow_wasm_inline_scripts: true,
    require_manifest: false,
    check_dangerous_patterns: true,
    forbid_node_modules: true,
    forbid_package_json: true,
    manifest_path: None,
    allowed_js_patterns: vec![],
    ..Default::default()
};
```

## WASM-Generated Scripts

Scripts with the `__PROBAR_WASM_GENERATED__` marker comment are allowed:

```html
<script>
    // __PROBAR_WASM_GENERATED__
    WebAssembly.instantiate(wasmModule).then(instance => {
        instance.exports.main();
    });
</script>
```

## Dangerous Pattern Detection

The validator detects dangerous JavaScript patterns:

| Pattern | Risk |
|---------|------|
| `eval(` | Code injection |
| `new Function(` | Dynamic code execution |
| `innerHTML =` | XSS vulnerability |
| `outerHTML =` | XSS vulnerability |
| `document.write(` | DOM manipulation |
| `insertAdjacentHTML(` | XSS vulnerability |
| `setTimeout(` with string | Code injection |
| `setInterval(` with string | Code injection |

```rust
let violations = validator.validate_js_content(
    "const x = eval('1 + 1');",
    Path::new("test.js")
);
assert!(!violations.is_empty()); // Detected eval()
```

## HTML Validation

Detects inline scripts and event handlers:

```rust
let html = r#"
<button onclick="handleClick()">Click</button>
<script>alert('inline!');</script>
"#;

let violations = validator.validate_html_content(html, Path::new("index.html"));
// Detects: onclick handler, inline script
```

## Validation Result

```rust
let result = validator.validate_directory("./pkg")?;

println!("{}", result);
// Output:
// ══════════════════════════════════════════════
// Zero-JS Validation: PASSED
// ══════════════════════════════════════════════
// Total files scanned: 42
// Verified JS files: 1 (wasm-bindgen generated)
// Violations: 0

// Check specific violations
if !result.is_valid() {
    println!("Unauthorized JS: {:?}", result.unauthorized_js_files);
    println!("Forbidden dirs: {:?}", result.forbidden_directories);
    println!("Dangerous patterns: {:?}", result.dangerous_patterns);
}
```

## Manifest Support

For projects with verified JS files (like wasm-bindgen output):

```json
// .probar-manifest.json
{
    "verified_js_files": [
        "pkg/myapp.js",
        "pkg/myapp_bg.wasm.js"
    ],
    "generator": "wasm-bindgen",
    "version": "0.2.92"
}
```

```rust
let config = ZeroJsConfig::strict()
    .with_manifest_path("./pkg/.probar-manifest.json");
```

## Example

Run the demo:

```bash
cargo run --example zero_js_demo -p jugar-probar
```

## Best Practices

1. **Use strict mode in CI** - Catch violations early
2. **Generate manifests** - Document allowed generated JS
3. **Validate on build** - Add to your build pipeline
4. **Audit regularly** - Check for new violations

## See Also

- [Worker Harness Testing](./worker-harness.md) - Web Worker testing
- [Docker Cross-Browser Testing](./docker-testing.md) - Multi-browser validation
- [Compliance Checking](./compliance.md) - Full compliance validation
