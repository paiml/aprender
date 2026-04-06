//! Zero-JS Validation Demo (PROBAR-SPEC-012)
//!
//! Demonstrates WASM-first architecture validation ensuring
//! NO user-generated JavaScript, CSS, or HTML exists in WASM applications.
//!
//! Run with: cargo run --example zero_js_demo -p jugar-probar

use jugar_probar::zero_js::{
    DangerousPatternViolation, ZeroJsConfig, ZeroJsValidationResult, ZeroJsValidator,
};
use std::path::PathBuf;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Zero-JS Validation Demo (PROBAR-SPEC-012)              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // 1. Basic Zero-JS Validation
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  1. Basic Zero-JS Validation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let validator = ZeroJsValidator::new();
    let default_config = ZeroJsConfig::default();
    println!("  Created ZeroJsValidator with default config:");
    println!(
        "    ├─ allow_wasm_inline_scripts: {}",
        default_config.allow_wasm_inline_scripts
    );
    println!(
        "    ├─ require_manifest: {}",
        default_config.require_manifest
    );
    println!(
        "    └─ check_dangerous_patterns: {}",
        default_config.check_dangerous_patterns
    );
    println!();

    // =========================================================================
    // 2. Configuration Presets
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  2. Configuration Presets");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let strict_config = ZeroJsConfig::strict();
    println!("  ZeroJsConfig::strict():");
    println!(
        "    ├─ require_manifest: {}",
        strict_config.require_manifest
    );
    println!(
        "    ├─ check_dangerous_patterns: {}",
        strict_config.check_dangerous_patterns
    );
    println!(
        "    └─ allow_wasm_inline_scripts: {}",
        strict_config.allow_wasm_inline_scripts
    );
    println!();

    let dev_config = ZeroJsConfig::development();
    println!("  ZeroJsConfig::development():");
    println!("    ├─ require_manifest: {}", dev_config.require_manifest);
    println!(
        "    ├─ check_dangerous_patterns: {}",
        dev_config.check_dangerous_patterns
    );
    println!(
        "    └─ allow_wasm_inline_scripts: {}",
        dev_config.allow_wasm_inline_scripts
    );
    println!();

    // =========================================================================
    // 3. Dangerous Pattern Detection
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  3. Dangerous Pattern Detection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let dangerous_js = r#"
        const result = eval('1 + 1');
        element.innerHTML = userInput;
        new Function('return ' + code)();
        document.write('<script>alert(1)</script>');
    "#;

    let violations = validator.validate_js_content(dangerous_js, std::path::Path::new("test.js"));
    println!(
        "  Detected {} dangerous patterns in test.js:",
        violations.len()
    );
    for (i, v) in violations.iter().enumerate() {
        println!("    {}. {}", i + 1, v);
    }
    println!();

    // =========================================================================
    // 4. HTML Inline Script Detection
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  4. HTML Inline Script Detection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let html_with_inline = r#"
<!DOCTYPE html>
<html>
<head>
    <script>alert('inline!');</script>
</head>
<body onclick="handleClick()">
    <button onmouseover="highlight()">Click me</button>
</body>
</html>
    "#;

    let strict_validator = ZeroJsValidator::with_config(ZeroJsConfig {
        allow_wasm_inline_scripts: false,
        ..Default::default()
    });

    let html_violations = strict_validator
        .validate_html_content(html_with_inline, std::path::Path::new("index.html"));
    println!(
        "  Detected {} inline script violations:",
        html_violations.len()
    );
    for (i, v) in html_violations.iter().enumerate() {
        println!("    {}. {}", i + 1, v);
    }
    println!();

    // =========================================================================
    // 5. WASM-Generated Scripts (Allowed)
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  5. WASM-Generated Scripts (Allowed)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let wasm_html = r#"
<!DOCTYPE html>
<html>
<head>
    <script>
        // __PROBAR_WASM_GENERATED__
        WebAssembly.instantiate(wasmModule).then(instance => {
            instance.exports.main();
        });
    </script>
</head>
<body></body>
</html>
    "#;

    let default_validator = ZeroJsValidator::new();
    let wasm_violations =
        default_validator.validate_html_content(wasm_html, std::path::Path::new("wasm.html"));
    println!("  WASM-generated scripts with marker comment:");
    println!(
        "    └─ Violations: {} (allowed by default)",
        wasm_violations.len()
    );
    println!();

    // =========================================================================
    // 6. Validation Result Display
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  6. Validation Result Display");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Create a passing result
    let passing_result = ZeroJsValidationResult {
        valid: true,
        verified_js_files: vec![PathBuf::from("pkg/wasm_bindgen.js")],
        ..Default::default()
    };
    println!("  Passing validation:");
    println!("{}", indent_string(&format!("{}", passing_result), "    "));
    println!();

    // Create a failing result
    let failing_result = ZeroJsValidationResult {
        valid: false,
        unauthorized_js_files: vec![PathBuf::from("src/app.js"), PathBuf::from("src/utils.ts")],
        forbidden_directories: vec![PathBuf::from("node_modules")],
        dangerous_patterns: vec![DangerousPatternViolation {
            file: PathBuf::from("src/app.js"),
            line: 42,
            pattern: "eval(".to_string(),
            context: "const x = eval('code')".to_string(),
        }],
        ..Default::default()
    };
    println!("  Failing validation:");
    println!("{}", indent_string(&format!("{}", failing_result), "    "));
    println!();

    // =========================================================================
    // 7. Integration Example
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  7. Integration Example");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("  Use in your tests:");
    println!();
    println!("    ```rust");
    println!("    use jugar_probar::zero_js::{{ZeroJsValidator, ZeroJsConfig}};");
    println!();
    println!("    #[test]");
    println!("    fn test_wasm_first_compliance() {{");
    println!("        let validator = ZeroJsValidator::with_config(");
    println!("            ZeroJsConfig::strict()");
    println!("        );");
    println!();
    println!("        let result = validator.validate_directory(\"./pkg\").unwrap();");
    println!("        assert!(");
    println!("            result.is_valid(),");
    println!("            \"Zero-JS validation failed: {{}}\",");
    println!("            result");
    println!("        );");
    println!("    }}");
    println!("    ```");
    println!();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              Demo Complete - WASM-First FTW!                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

fn indent_string(s: &str, indent: &str) -> String {
    s.lines()
        .map(|line| format!("{}{}", indent, line))
        .collect::<Vec<_>>()
        .join("\n")
}
