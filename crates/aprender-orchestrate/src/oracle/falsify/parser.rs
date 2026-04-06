//! Specification Parser
//!
//! Parses markdown specifications into structured requirements.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Parsed specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedSpec {
    /// Specification name
    pub name: String,
    /// Module/crate name
    pub module: String,
    /// Extracted requirements
    pub requirements: Vec<ParsedRequirement>,
    /// Types mentioned
    pub types: Vec<String>,
    /// Functions mentioned
    pub functions: Vec<String>,
    /// Numerical tolerances if specified
    pub tolerances: Option<ToleranceSpec>,
}

/// A parsed requirement from the spec
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedRequirement {
    /// Requirement ID (from spec or generated)
    pub id: String,
    /// Description
    pub description: String,
    /// Category hint (boundary, invariant, numerical, etc.)
    pub category_hint: Option<String>,
    /// Input type hint
    pub input_type: Option<String>,
    /// Output type hint
    pub output_type: Option<String>,
    /// Is this a critical requirement?
    pub critical: bool,
}

/// Numerical tolerance specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceSpec {
    /// Absolute tolerance
    pub atol: Option<f64>,
    /// Relative tolerance
    pub rtol: Option<f64>,
}

/// Specification parser
#[derive(Debug)]
pub struct SpecParser {
    // Configuration options could go here
}

impl SpecParser {
    /// Create a new parser
    pub fn new() -> Self {
        Self {}
    }

    /// Parse a specification file
    pub fn parse_file(&self, path: &Path) -> anyhow::Result<ParsedSpec> {
        let content = std::fs::read_to_string(path)?;
        self.parse(&content, path)
    }

    /// Parse specification content
    pub fn parse(&self, content: &str, source: &Path) -> anyhow::Result<ParsedSpec> {
        let name = source
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unnamed".to_string());

        let module = self.extract_module(content).unwrap_or_else(|| name.clone());
        let requirements = self.extract_requirements(content);
        let types = self.extract_types(content);
        let functions = self.extract_functions(content);
        let tolerances = self.extract_tolerances(content);

        Ok(ParsedSpec { name, module, requirements, types, functions, tolerances })
    }

    /// Extract module name from spec
    fn extract_module(&self, content: &str) -> Option<String> {
        // Look for patterns like "module: foo" or "crate: foo"
        for line in content.lines() {
            let lower = line.to_lowercase();
            if lower.starts_with("module:") || lower.starts_with("crate:") {
                return line.split(':').nth(1).map(|s| s.trim().to_string());
            }
        }

        // Try to extract from code blocks
        for line in content.lines() {
            if line.contains("use ") && line.contains("::") {
                if let Some(start) = line.find("use ") {
                    let rest = &line[start + 4..];
                    if let Some(end) = rest.find("::") {
                        return Some(rest[..end].trim().to_string());
                    }
                }
            }
        }

        None
    }

    /// Extract requirements from spec
    fn extract_requirements(&self, content: &str) -> Vec<ParsedRequirement> {
        let mut requirements = Vec::new();
        let mut current_section = String::new();
        let mut req_counter = 0;

        for line in content.lines() {
            // Track sections (headers)
            if line.starts_with('#') {
                current_section = line.trim_start_matches('#').trim().to_lowercase();
                continue;
            }

            // Look for requirement patterns:
            // - Bullet points that look like requirements
            // - "MUST", "SHALL", "SHOULD" keywords
            // - Numbered items
            if let Some(req) = self.parse_requirement_line(line, &current_section, &mut req_counter)
            {
                requirements.push(req);
            }
        }

        requirements
    }

    /// Parse a potential requirement line
    fn parse_requirement_line(
        &self,
        line: &str,
        section: &str,
        counter: &mut u32,
    ) -> Option<ParsedRequirement> {
        let trimmed = line.trim();

        // Skip empty lines and non-requirement lines
        if trimmed.is_empty() || trimmed.starts_with("```") {
            return None;
        }

        // Check for requirement keywords
        let upper = trimmed.to_uppercase();
        let is_requirement = upper.contains("MUST")
            || upper.contains("SHALL")
            || upper.contains("SHOULD")
            || upper.contains("REQUIRE")
            || (trimmed.starts_with("- ") && trimmed.len() > 10);

        if !is_requirement {
            return None;
        }

        *counter += 1;

        // Determine category hint from section name
        let category_hint = self.infer_category(section, trimmed);

        // Check if critical
        let critical =
            upper.contains("CRITICAL") || upper.contains("MUST NOT") || upper.contains("SHALL NOT");

        Some(ParsedRequirement {
            id: format!("REQ-{:03}", counter),
            description: trimmed.trim_start_matches('-').trim_start_matches('*').trim().to_string(),
            category_hint,
            input_type: self.extract_type_hint(trimmed, "input"),
            output_type: self.extract_type_hint(trimmed, "output"),
            critical,
        })
    }

    /// Infer category from section name and content
    fn infer_category(&self, section: &str, content: &str) -> Option<String> {
        let lower_section = section.to_lowercase();
        let lower_content = content.to_lowercase();

        if lower_section.contains("boundary")
            || lower_content.contains("empty")
            || lower_content.contains("null")
            || lower_content.contains("limit")
        {
            return Some("boundary".to_string());
        }

        if lower_section.contains("invariant")
            || lower_content.contains("idempotent")
            || lower_content.contains("commutative")
        {
            return Some("invariant".to_string());
        }

        if lower_section.contains("numeric")
            || lower_content.contains("precision")
            || lower_content.contains("floating")
        {
            return Some("numerical".to_string());
        }

        if lower_section.contains("concurren")
            || lower_content.contains("thread")
            || lower_content.contains("race")
        {
            return Some("concurrency".to_string());
        }

        if lower_section.contains("resource")
            || lower_content.contains("memory")
            || lower_content.contains("exhaust")
        {
            return Some("resource".to_string());
        }

        if lower_section.contains("parity")
            || lower_content.contains("reference")
            || lower_content.contains("match")
        {
            return Some("parity".to_string());
        }

        None
    }

    /// Extract type hint from content
    fn extract_type_hint(&self, content: &str, kind: &str) -> Option<String> {
        // Look for patterns like "input: Vec<f64>" or "returns f64"
        let lower = content.to_lowercase();

        if kind == "input" {
            if lower.contains("vec") || lower.contains("array") || lower.contains("list") {
                return Some("Vec<T>".to_string());
            }
            if lower.contains("string") || lower.contains("str") {
                return Some("String".to_string());
            }
        }

        if kind == "output" && lower.contains("returns") {
            if lower.contains("bool") {
                return Some("bool".to_string());
            }
            if lower.contains("result") || lower.contains("error") {
                return Some("Result<T, E>".to_string());
            }
        }

        None
    }

    /// Extract types mentioned in spec
    fn extract_types(&self, content: &str) -> Vec<String> {
        let mut types = Vec::new();

        // Look for struct/enum definitions
        for line in content.lines() {
            if line.contains("struct ") {
                if let Some(name) = line.split("struct ").nth(1) {
                    if let Some(name) =
                        name.split(|c: char| !c.is_alphanumeric() && c != '_').next()
                    {
                        types.push(name.to_string());
                    }
                }
            }
            if line.contains("enum ") {
                if let Some(name) = line.split("enum ").nth(1) {
                    if let Some(name) =
                        name.split(|c: char| !c.is_alphanumeric() && c != '_').next()
                    {
                        types.push(name.to_string());
                    }
                }
            }
        }

        types.sort();
        types.dedup();
        types
    }

    /// Extract functions mentioned in spec
    fn extract_functions(&self, content: &str) -> Vec<String> {
        let mut functions = Vec::new();

        for line in content.lines() {
            // Look for fn definitions
            if line.contains("fn ") {
                if let Some(name) = line.split("fn ").nth(1) {
                    if let Some(name) = name.split('(').next() {
                        functions.push(name.trim().to_string());
                    }
                }
            }
            // Look for method calls mentioned
            if line.contains("`.") {
                // Markdown code inline
                let parts: Vec<&str> = line.split('`').collect();
                for (i, part) in parts.iter().enumerate() {
                    if i % 2 == 1 && part.contains('.') {
                        if let Some(method) = part.split('.').next_back() {
                            if let Some(name) = method.split('(').next() {
                                functions.push(name.to_string());
                            }
                        }
                    }
                }
            }
        }

        functions.sort();
        functions.dedup();
        functions
    }

    /// Extract tolerance specifications
    fn extract_tolerances(&self, content: &str) -> Option<ToleranceSpec> {
        let mut atol = None;
        let mut rtol = None;

        for line in content.lines() {
            let lower = line.to_lowercase();

            // Look for tolerance specifications
            if lower.contains("atol") || lower.contains("absolute") {
                if let Some(val) = self.extract_number(line) {
                    atol = Some(val);
                }
            }
            if lower.contains("rtol") || lower.contains("relative") {
                if let Some(val) = self.extract_number(line) {
                    rtol = Some(val);
                }
            }
            if lower.contains("tolerance") && lower.contains("1e-") {
                if let Some(val) = self.extract_number(line) {
                    if atol.is_none() {
                        atol = Some(val);
                    }
                }
            }
        }

        if atol.is_some() || rtol.is_some() {
            Some(ToleranceSpec { atol, rtol })
        } else {
            None
        }
    }

    /// Extract a number from a line
    fn extract_number(&self, line: &str) -> Option<f64> {
        // Use regex-like pattern matching for scientific notation
        let mut best_num: Option<f64> = None;

        // Look for patterns like 1e-5, 0.001, 1.5e-10
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if chars[i].is_ascii_digit() {
                // Found start of potential number
                let mut num_str = String::new();
                while i < chars.len() {
                    let c = chars[i];
                    if c.is_ascii_digit() || c == '.' {
                        num_str.push(c);
                        i += 1;
                    } else if (c == 'e' || c == 'E') && i + 1 < chars.len() {
                        num_str.push(c);
                        i += 1;
                        // Handle optional sign after e
                        if i < chars.len() && (chars[i] == '-' || chars[i] == '+') {
                            num_str.push(chars[i]);
                            i += 1;
                        }
                    } else {
                        break;
                    }
                }

                if let Ok(val) = num_str.parse::<f64>() {
                    best_num = Some(val);
                    break; // Return first valid number
                }
            } else {
                i += 1;
            }
        }

        best_num
    }
}

impl Default for SpecParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let parser = SpecParser::new();
        // Verify parser can parse empty content without panic
        let result = parser.parse("", Path::new("test.md"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_simple_spec() {
        let parser = SpecParser::new();
        let content = r#"
# Test Spec

module: test_module

## Requirements

- MUST handle empty input
- SHOULD return error on invalid input
- The function MUST NOT panic
"#;
        let spec = parser.parse(content, Path::new("test-spec.md")).expect("unexpected failure");
        assert_eq!(spec.name, "test-spec");
        assert_eq!(spec.module, "test_module");
        assert!(!spec.requirements.is_empty());
    }

    #[test]
    fn test_extract_tolerances() {
        let parser = SpecParser::new();
        // Use format that matches the parser's expectations
        let content = r#"
Use a tolerance of 1e-5 for comparisons
"#;
        let tolerances = parser.extract_tolerances(content);
        assert!(tolerances.is_some(), "Should extract tolerance from '1e-5'");
        let tol = tolerances.expect("unexpected failure");
        assert!(tol.atol.is_some(), "atol should be extracted");
        assert!((tol.atol.expect("unexpected failure") - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_infer_category() {
        let parser = SpecParser::new();

        assert_eq!(
            parser.infer_category("boundary conditions", "handle empty input"),
            Some("boundary".to_string())
        );

        assert_eq!(
            parser.infer_category("numerical", "floating point precision"),
            Some("numerical".to_string())
        );

        assert_eq!(
            parser.infer_category("other", "thread safety"),
            Some("concurrency".to_string())
        );
    }

    #[test]
    fn test_parser_default() {
        let parser = SpecParser::default();
        let content = "module: test\n- MUST work";
        let spec = parser.parse(content, Path::new("test.md")).expect("unexpected failure");
        assert_eq!(spec.module, "test");
    }

    #[test]
    fn test_extract_module_from_use_statement() {
        let parser = SpecParser::new();
        let content = "```rust\nuse aprender::knn::KnnClassifier;\n```";
        let module = parser.extract_module(content);
        assert_eq!(module, Some("aprender".to_string()));
    }

    #[test]
    fn test_extract_module_none() {
        let parser = SpecParser::new();
        let content = "Just some text without module info";
        assert!(parser.extract_module(content).is_none());
    }

    #[test]
    fn test_extract_types_struct() {
        let parser = SpecParser::new();
        let content = "```rust\nstruct MyType { field: u32 }\nenum MyEnum { A, B }\n```";
        let types = parser.extract_types(content);
        assert!(types.contains(&"MyType".to_string()));
        assert!(types.contains(&"MyEnum".to_string()));
    }

    #[test]
    fn test_extract_types_empty() {
        let parser = SpecParser::new();
        let content = "No types here";
        let types = parser.extract_types(content);
        assert!(types.is_empty());
    }

    #[test]
    fn test_extract_functions() {
        let parser = SpecParser::new();
        let content = "```rust\nfn process_data(input: &str) -> Result<()> {}\n```";
        let functions = parser.extract_functions(content);
        assert!(functions.contains(&"process_data".to_string()));
    }

    #[test]
    fn test_extract_functions_fn_definition() {
        let parser = SpecParser::new();
        let content = "fn process_data(input: &str) -> Result<()>\nfn other() {}";
        let functions = parser.extract_functions(content);
        assert!(functions.contains(&"process_data".to_string()));
        assert!(functions.contains(&"other".to_string()));
    }

    #[test]
    fn test_extract_type_hint_vec() {
        let parser = SpecParser::new();
        let hint = parser.extract_type_hint("accepts a vector of values", "input");
        assert_eq!(hint, Some("Vec<T>".to_string()));
    }

    #[test]
    fn test_extract_type_hint_string() {
        let parser = SpecParser::new();
        let hint = parser.extract_type_hint("input is a string", "input");
        assert_eq!(hint, Some("String".to_string()));
    }

    #[test]
    fn test_extract_type_hint_bool_output() {
        let parser = SpecParser::new();
        let hint = parser.extract_type_hint("returns bool indicating success", "output");
        assert_eq!(hint, Some("bool".to_string()));
    }

    #[test]
    fn test_extract_type_hint_result_output() {
        let parser = SpecParser::new();
        let hint = parser.extract_type_hint("returns result or error", "output");
        assert_eq!(hint, Some("Result<T, E>".to_string()));
    }

    #[test]
    fn test_extract_number_scientific() {
        let parser = SpecParser::new();
        let num = parser.extract_number("tolerance: 1e-10");
        assert!(num.is_some());
        assert!((num.expect("unexpected failure") - 1e-10).abs() < 1e-15);
    }

    #[test]
    fn test_extract_number_decimal() {
        let parser = SpecParser::new();
        let num = parser.extract_number("precision: 0.001");
        assert!(num.is_some());
        assert!((num.expect("unexpected failure") - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_extract_number_with_sign() {
        let parser = SpecParser::new();
        let num = parser.extract_number("value 5E+3");
        assert!(num.is_some());
        assert!((num.expect("unexpected failure") - 5000.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_number_none() {
        let parser = SpecParser::new();
        let num = parser.extract_number("no numbers here");
        assert!(num.is_none());
    }

    #[test]
    fn test_parse_requirement_line_must() {
        let parser = SpecParser::new();
        let mut counter = 0;
        let req = parser.parse_requirement_line(
            "- MUST handle empty input",
            "requirements",
            &mut counter,
        );
        assert!(req.is_some());
        let req = req.expect("unexpected failure");
        assert_eq!(req.id, "REQ-001");
        assert!(req.description.contains("handle empty input"));
    }

    #[test]
    fn test_parse_requirement_line_critical() {
        let parser = SpecParser::new();
        let mut counter = 0;
        let req =
            parser.parse_requirement_line("CRITICAL: MUST NOT panic", "requirements", &mut counter);
        assert!(req.is_some());
        assert!(req.expect("unexpected failure").critical);
    }

    #[test]
    fn test_parse_requirement_line_skip_code_block() {
        let parser = SpecParser::new();
        let mut counter = 0;
        let req = parser.parse_requirement_line("```rust", "code", &mut counter);
        assert!(req.is_none());
    }

    #[test]
    fn test_parse_requirement_line_skip_empty() {
        let parser = SpecParser::new();
        let mut counter = 0;
        let req = parser.parse_requirement_line("   ", "section", &mut counter);
        assert!(req.is_none());
    }

    #[test]
    fn test_infer_category_invariant() {
        let parser = SpecParser::new();
        assert_eq!(
            parser.infer_category("invariants", "must be idempotent"),
            Some("invariant".to_string())
        );
    }

    #[test]
    fn test_infer_category_resource() {
        let parser = SpecParser::new();
        assert_eq!(
            parser.infer_category("resource limits", "memory exhaustion"),
            Some("resource".to_string())
        );
    }

    #[test]
    fn test_infer_category_parity() {
        let parser = SpecParser::new();
        assert_eq!(
            parser.infer_category("parity tests", "must match reference"),
            Some("parity".to_string())
        );
    }

    #[test]
    fn test_infer_category_none() {
        let parser = SpecParser::new();
        assert!(parser.infer_category("overview", "general description").is_none());
    }

    #[test]
    fn test_extract_tolerances_with_rtol() {
        let parser = SpecParser::new();
        let content = "relative tolerance rtol of 1e-6";
        let tol = parser.extract_tolerances(content);
        assert!(tol.is_some());
        assert!(tol.expect("unexpected failure").rtol.is_some());
    }

    #[test]
    fn test_extract_tolerances_none() {
        let parser = SpecParser::new();
        let content = "no tolerance specified";
        let tol = parser.extract_tolerances(content);
        assert!(tol.is_none());
    }

    #[test]
    fn test_parsed_spec_fields() {
        let spec = ParsedSpec {
            name: "test".to_string(),
            module: "mod".to_string(),
            requirements: vec![],
            types: vec!["T".to_string()],
            functions: vec!["f".to_string()],
            tolerances: Some(ToleranceSpec { atol: Some(1e-5), rtol: None }),
        };
        assert_eq!(spec.name, "test");
        assert!(spec.tolerances.is_some());
    }

    #[test]
    fn test_parsed_requirement_fields() {
        let req = ParsedRequirement {
            id: "REQ-001".to_string(),
            description: "test requirement".to_string(),
            category_hint: Some("boundary".to_string()),
            input_type: Some("Vec<T>".to_string()),
            output_type: Some("Result<T, E>".to_string()),
            critical: true,
        };
        assert!(req.critical);
        assert_eq!(req.category_hint, Some("boundary".to_string()));
    }

    // =====================================================================
    // extract_functions: backtick method extraction coverage
    // =====================================================================

    #[test]
    fn test_extract_functions_backtick_method_calls() {
        let parser = SpecParser::new();
        // The backtick extraction path triggers when line contains "`." (backtick + dot).
        // Use `.method()` inline code style which places the dot right after the backtick.
        let content = "Call `.predict()` to get results\nUse `.fit()` for training";
        let functions = parser.extract_functions(content);
        assert!(functions.contains(&"predict".to_string()));
        assert!(functions.contains(&"fit".to_string()));
    }

    #[test]
    fn test_extract_functions_backtick_chained_methods() {
        let parser = SpecParser::new();
        // Backtick extraction splits on '.', so chained `.build().run()` extracts last segment
        let content = "Use `.build().run()` for execution";
        let functions = parser.extract_functions(content);
        // Should extract the last method in the chain
        assert!(functions.contains(&"run".to_string()));
    }

    #[test]
    fn test_extract_functions_backtick_no_method() {
        let parser = SpecParser::new();
        // Backtick content without a dot - should not add anything
        let content = "Use `some_value` directly";
        let functions = parser.extract_functions(content);
        assert!(functions.is_empty());
    }

    #[test]
    fn test_extract_functions_mixed_fn_and_backtick() {
        let parser = SpecParser::new();
        // fn definition on first line, backtick `.method()` on second
        let content = "fn compute(x: f64) -> f64\nCall `.transform()` first";
        let functions = parser.extract_functions(content);
        assert!(functions.contains(&"compute".to_string()));
        assert!(functions.contains(&"transform".to_string()));
    }

    #[test]
    fn test_extract_functions_backtick_with_parens() {
        let parser = SpecParser::new();
        // Backtick extraction requires "`." in the line
        let content = "Method `.validate(input)` must succeed";
        let functions = parser.extract_functions(content);
        assert!(functions.contains(&"validate".to_string()));
    }

    #[test]
    fn test_extract_functions_dedup() {
        let parser = SpecParser::new();
        let content = "fn process()\nfn process()";
        let functions = parser.extract_functions(content);
        // Should be deduplicated
        assert_eq!(
            functions.iter().filter(|f| *f == "process").count(),
            1,
            "Duplicate functions should be deduped"
        );
    }

    // =====================================================================
    // parse_file: filesystem parsing coverage
    // =====================================================================

    #[test]
    fn test_parse_file_valid() {
        let parser = SpecParser::new();
        let temp_dir = std::env::temp_dir().join("batuta_parser_test");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).expect("mkdir failed");

        let spec_path = temp_dir.join("test-spec.md");
        std::fs::write(
            &spec_path,
            "# My Spec\n\nmodule: my_module\n\n## Requirements\n\n- MUST handle edge cases\n",
        )
        .expect("unexpected failure");

        let result = parser.parse_file(&spec_path);
        assert!(result.is_ok());
        let spec = result.expect("operation failed");
        assert_eq!(spec.name, "test-spec");
        assert_eq!(spec.module, "my_module");
        assert!(!spec.requirements.is_empty());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_parse_file_not_found() {
        let parser = SpecParser::new();
        let result = parser.parse_file(Path::new("/nonexistent/spec.md"));
        assert!(result.is_err());
    }

    // =====================================================================
    // extract_module: crate: prefix coverage
    // =====================================================================

    #[test]
    fn test_extract_module_crate_prefix() {
        let parser = SpecParser::new();
        let content = "crate: my_crate\nSome description";
        let module = parser.extract_module(content);
        assert_eq!(module, Some("my_crate".to_string()));
    }

    #[test]
    fn test_extract_module_use_with_nested_path() {
        let parser = SpecParser::new();
        let content = "```rust\nuse trueno::simd::avx2::kernel;\n```";
        let module = parser.extract_module(content);
        assert_eq!(module, Some("trueno".to_string()));
    }

    // =====================================================================
    // extract_tolerances: tolerance + 1e- path + atol.is_none() check
    // =====================================================================

    #[test]
    fn test_extract_tolerances_tolerance_keyword_with_scientific() {
        let parser = SpecParser::new();
        // Triggers the third branch: "tolerance" + "1e-"
        let content = "Use a tolerance of 1e-8 for all comparisons";
        let tol = parser.extract_tolerances(content);
        assert!(tol.is_some());
        let t = tol.expect("unexpected failure");
        assert!(t.atol.is_some());
        assert!((t.atol.expect("unexpected failure") - 1e-8).abs() < 1e-15);
    }

    #[test]
    fn test_extract_tolerances_atol_then_tolerance_keyword() {
        let parser = SpecParser::new();
        // atol set first, then "tolerance 1e-" should NOT overwrite atol
        let content = "atol = 1e-5\ntolerance of 1e-3";
        let tol = parser.extract_tolerances(content);
        assert!(tol.is_some());
        let t = tol.expect("unexpected failure");
        // atol should still be 1e-5 (first match), because atol.is_none() check prevents overwrite
        assert!(t.atol.is_some());
        assert!((t.atol.expect("unexpected failure") - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_extract_tolerances_both_atol_rtol() {
        let parser = SpecParser::new();
        let content = "absolute tolerance atol = 1e-6\nrelative rtol = 1e-4";
        let tol = parser.extract_tolerances(content);
        assert!(tol.is_some());
        let t = tol.expect("unexpected failure");
        assert!(t.atol.is_some());
        assert!(t.rtol.is_some());
        assert!((t.atol.expect("unexpected failure") - 1e-6).abs() < 1e-12);
        assert!((t.rtol.expect("unexpected failure") - 1e-4).abs() < 1e-10);
    }

    // =====================================================================
    // extract_type_hint: additional coverage for non-matching kinds
    // =====================================================================

    #[test]
    fn test_extract_type_hint_input_array() {
        let parser = SpecParser::new();
        let hint = parser.extract_type_hint("takes an array of floats", "input");
        assert_eq!(hint, Some("Vec<T>".to_string()));
    }

    #[test]
    fn test_extract_type_hint_input_list() {
        let parser = SpecParser::new();
        let hint = parser.extract_type_hint("accepts a list of items", "input");
        assert_eq!(hint, Some("Vec<T>".to_string()));
    }

    #[test]
    fn test_extract_type_hint_output_no_returns() {
        let parser = SpecParser::new();
        // "output" kind but no "returns" keyword
        let hint = parser.extract_type_hint("produces a bool value", "output");
        assert!(hint.is_none());
    }

    #[test]
    fn test_extract_type_hint_no_match() {
        let parser = SpecParser::new();
        let hint = parser.extract_type_hint("does something", "input");
        assert!(hint.is_none());
    }

    // =====================================================================
    // parse: unnamed source path coverage
    // =====================================================================

    #[test]
    fn test_parse_with_no_file_stem() {
        let parser = SpecParser::new();
        // Path with no file stem
        let content = "module: test_mod\n- MUST work";
        let spec = parser.parse(content, Path::new("/")).expect("unexpected failure");
        // With "/" as path, file_stem returns None, so name becomes "unnamed"
        // Actually "/" returns None for file_stem in some cases
        assert!(!spec.name.is_empty());
    }

    // =====================================================================
    // infer_category: content-based matching without section match
    // =====================================================================

    #[test]
    fn test_infer_category_content_null() {
        let parser = SpecParser::new();
        // Section doesn't match but content contains "null"
        let cat = parser.infer_category("general", "handle null values");
        assert_eq!(cat, Some("boundary".to_string()));
    }

    #[test]
    fn test_infer_category_content_limit() {
        let parser = SpecParser::new();
        let cat = parser.infer_category("edge cases", "check the limit");
        assert_eq!(cat, Some("boundary".to_string()));
    }

    #[test]
    fn test_infer_category_content_commutative() {
        let parser = SpecParser::new();
        let cat = parser.infer_category("math", "operation must be commutative");
        assert_eq!(cat, Some("invariant".to_string()));
    }

    #[test]
    fn test_infer_category_content_race() {
        let parser = SpecParser::new();
        let cat = parser.infer_category("safety", "avoid race conditions");
        assert_eq!(cat, Some("concurrency".to_string()));
    }

    #[test]
    fn test_infer_category_content_exhaust() {
        let parser = SpecParser::new();
        let cat = parser.infer_category("limits", "may exhaust resources");
        assert_eq!(cat, Some("resource".to_string()));
    }

    // =====================================================================
    // parse_requirement_line: SHALL and REQUIRE keywords
    // =====================================================================

    #[test]
    fn test_parse_requirement_line_shall() {
        let parser = SpecParser::new();
        let mut counter = 0;
        let req = parser.parse_requirement_line(
            "The system SHALL validate input",
            "section",
            &mut counter,
        );
        assert!(req.is_some());
        assert_eq!(req.expect("unexpected failure").id, "REQ-001");
        assert_eq!(counter, 1);
    }

    #[test]
    fn test_parse_requirement_line_require() {
        let parser = SpecParser::new();
        let mut counter = 5;
        let req =
            parser.parse_requirement_line("REQUIRE proper authentication", "section", &mut counter);
        assert!(req.is_some());
        assert_eq!(req.expect("unexpected failure").id, "REQ-006");
        assert_eq!(counter, 6);
    }

    #[test]
    fn test_parse_requirement_line_short_bullet() {
        let parser = SpecParser::new();
        let mut counter = 0;
        // Short bullet (< 10 chars after "- ") without keywords
        let req = parser.parse_requirement_line("- short", "section", &mut counter);
        assert!(req.is_none());
    }

    #[test]
    fn test_parse_requirement_line_shall_not() {
        let parser = SpecParser::new();
        let mut counter = 0;
        let req =
            parser.parse_requirement_line("SHALL NOT expose secrets", "section", &mut counter);
        assert!(req.is_some());
        assert!(req.expect("unexpected failure").critical);
    }

    #[test]
    fn test_parse_requirement_line_star_bullet() {
        let parser = SpecParser::new();
        let mut counter = 0;
        let req = parser.parse_requirement_line(
            "* MUST handle large inputs gracefully",
            "section",
            &mut counter,
        );
        assert!(req.is_some());
        let r = req.expect("unexpected failure");
        assert!(r.description.starts_with("MUST"));
    }
}
