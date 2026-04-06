# Web Validation and Linting

Probar provides Rust-native validation and linting for HTML, CSS, and JavaScript. No external tools like eslint, stylelint, or htmlhint required.

## Overview

The validation system is 100% Rust:

| Validator | Replaces | Checks |
|-----------|----------|--------|
| `WebValidator::validate_html` | htmlhint | Structure, tags, accessibility |
| `WebValidator::lint_css` | stylelint | Syntax, selectors, patterns |
| `WebValidator::lint_js` | eslint | Line limit, security issues |
| `WebValidator::validate_all` | All above | Complete validation |

## HTML Validation

Validate HTML documents for structure and accessibility.

### Basic Usage

```rust
use jugar_probar::web::{HtmlBuilder, WebValidator};

let html = HtmlBuilder::new()
    .title("My App")
    .canvas("game", 800, 600)
    .build()?;

let result = WebValidator::validate_html(&html);

if result.is_valid() {
    println!("HTML validation passed!");
} else {
    for error in &result.errors {
        println!("Error: {}", error);
    }
}
```

### Checks Performed

| Check | Error/Warning |
|-------|---------------|
| Missing DOCTYPE | Error |
| Missing `<html>` tag | Error |
| Missing `<head>` tag | Error |
| Missing `<body>` tag | Error |
| Missing or empty `<title>` | Error |
| Missing charset meta | Warning |
| Missing viewport meta | Warning |
| Missing lang attribute | Warning |

## CSS Linting

Lint CSS stylesheets for syntax and best practices.

### Basic Usage

```rust
use jugar_probar::web::{CssBuilder, WebValidator};

let css = CssBuilder::new()
    .reset()
    .build()?;

let result = WebValidator::lint_css(&css);

if result.is_valid() {
    println!("CSS lint passed!");
}

for warning in &result.warnings {
    println!("Warning: {}", warning);
}
```

### Checks Performed

| Check | Error/Warning |
|-------|---------------|
| Empty selector | Error |
| Empty stylesheet | Warning |
| Use of `!important` | Warning |
| Vendor prefixes | Warning |

## JavaScript Linting

Lint JavaScript for security issues and Zero-JS policy compliance.

### Basic Usage

```rust
use jugar_probar::web::{JsBuilder, WebValidator};

let js = JsBuilder::new("app.wasm", "canvas")
    .build()?;

let result = WebValidator::lint_js(&js);

if result.is_valid() {
    println!("JS lint passed!");
} else {
    for issue in &result.security_issues {
        println!("{:?}: {}", issue.severity, issue.description);
    }
}
```

### Security Checks

| Pattern | Severity | Description |
|---------|----------|-------------|
| `eval(` | Critical | Code injection risk |
| `new Function(` | Critical | Dynamic code execution |
| `innerHTML` | High | XSS vulnerability |
| `document.write` | Medium | Deprecated, security risk |
| `setTimeout("..."` | High | String arg is eval-like |
| `setInterval("..."` | High | String arg is eval-like |

### Line Limit

JavaScript is limited to 20 lines (Zero-JS policy):

```rust
let result = WebValidator::lint_js(&js);

if js.line_count > 20 {
    assert!(!result.is_valid());
    assert!(result.errors.iter().any(|e| e.contains("line limit")));
}
```

## Accessibility Checking

Check HTML for WCAG accessibility issues.

### Basic Usage

```rust
use jugar_probar::web::{HtmlBuilder, WebValidator};

let html = HtmlBuilder::new()
    .title("Game")
    .canvas("game", 800, 600)
    .build()?;

let issues = WebValidator::check_accessibility(&html);

for issue in &issues {
    println!("[{}] {} - {}",
        issue.wcag_ref.as_deref().unwrap_or("N/A"),
        issue.element_id.as_deref().unwrap_or("document"),
        issue.description
    );
}
```

### Accessibility Checks

| Issue | WCAG Ref | Severity |
|-------|----------|----------|
| Canvas missing role | WCAG 4.1.2 | Medium |
| Canvas missing aria-label | WCAG 1.1.1 | Medium |
| Button missing aria-label | WCAG 4.1.2 | Medium |
| Input missing aria-label | WCAG 1.3.1 | Medium |
| Missing lang attribute | WCAG 3.1.1 | High |

## Combined Validation

Validate all assets at once with `validate_all`.

### Basic Usage

```rust
use jugar_probar::web::{HtmlBuilder, CssBuilder, JsBuilder, WebValidator};

let html = HtmlBuilder::new().title("App").canvas("c", 100, 100).build()?;
let css = CssBuilder::new().reset().build()?;
let js = JsBuilder::new("app.wasm", "c").build()?;

let report = WebValidator::validate_all(&html, &css, &js);

println!("Valid: {}", report.is_valid());
println!("Errors: {}", report.error_count());
println!("Warnings: {}", report.warning_count());
```

### ValidationReport Structure

```rust
pub struct ValidationReport {
    pub html: HtmlValidationResult,
    pub css: CssLintResult,
    pub js: JsLintResult,
    pub accessibility: Vec<AccessibilityIssue>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool;
    pub fn error_count(&self) -> usize;
    pub fn warning_count(&self) -> usize;
}
```

## Severity Levels

Security and accessibility issues have severity levels:

```rust
pub enum Severity {
    Low,      // Minor issues
    Medium,   // Should be fixed
    High,     // Security/a11y risk
    Critical, // Must be fixed
}
```

A bundle is invalid if it has any Critical accessibility issues.

## Example

Run the full demo:

```bash
cargo run --example web_validation_demo -p jugar-probar
```

## See Also

- [Web Builders](./web-builders.md) - Asset generation APIs
- [Zero-JS Validation](./zero-js-validation.md) - Unauthorized JS detection
- [Accessibility Testing](./accessibility.md) - WCAG auditing
