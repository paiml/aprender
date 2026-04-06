//! Test Generator
//!
//! Generates Rust and Python test code from specifications and templates.

use super::parser::ParsedSpec;
use super::template::{FalsificationTemplate, TestSeverity, TestTemplate};
use serde::{Deserialize, Serialize};

/// Target language for generated tests
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetLanguage {
    Rust,
    Python,
}

/// A generated test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedTest {
    /// Test ID (e.g., "BC-001")
    pub id: String,
    /// Test name
    pub name: String,
    /// Category
    pub category: String,
    /// Point value
    pub points: u32,
    /// Severity
    pub severity: TestSeverity,
    /// Generated code
    pub code: String,
}

/// Test generator
#[derive(Debug)]
pub struct FalsifyGenerator {
    /// Module name placeholder
    module_placeholder: String,
    /// Function name placeholder
    function_placeholder: String,
    /// Type placeholder
    type_placeholder: String,
}

impl FalsifyGenerator {
    /// Create a new generator
    pub fn new() -> Self {
        Self {
            module_placeholder: "{{module}}".to_string(),
            function_placeholder: "{{function}}".to_string(),
            type_placeholder: "{{type}}".to_string(),
        }
    }

    /// Generate tests from spec and template
    pub fn generate(
        &self,
        spec: &ParsedSpec,
        template: &FalsificationTemplate,
        language: TargetLanguage,
    ) -> anyhow::Result<Vec<GeneratedTest>> {
        let mut tests = Vec::new();

        for category in &template.categories {
            for test_template in &category.tests {
                let code = self.generate_test_code(spec, test_template, language)?;

                tests.push(GeneratedTest {
                    id: test_template.id.clone(),
                    name: test_template.name.clone(),
                    category: category.name.clone(),
                    points: test_template.points,
                    severity: test_template.severity,
                    code,
                });
            }
        }

        Ok(tests)
    }

    /// Generate code for a single test
    fn generate_test_code(
        &self,
        spec: &ParsedSpec,
        template: &TestTemplate,
        language: TargetLanguage,
    ) -> anyhow::Result<String> {
        let code_template = match language {
            TargetLanguage::Rust => template.rust_template.as_deref().unwrap_or(DEFAULT_RUST),
            TargetLanguage::Python => template.python_template.as_deref().unwrap_or(DEFAULT_PYTHON),
        };

        let code = self.substitute_placeholders(code_template, spec, template);
        Ok(code)
    }

    /// Substitute placeholders in template
    fn substitute_placeholders(
        &self,
        template: &str,
        spec: &ParsedSpec,
        test_template: &TestTemplate,
    ) -> String {
        let mut code = template.to_string();

        // Module substitution
        code = code.replace(&self.module_placeholder, &spec.module);
        code = code.replace("{{module}}", &spec.module);

        // Function substitution (use first function if available)
        let function = spec.functions.first().map(|s| s.as_str()).unwrap_or("function");
        code = code.replace(&self.function_placeholder, function);
        code = code.replace("{{function}}", function);

        // Type substitution (use first type if available)
        let type_name = spec.types.first().map(|s| s.as_str()).unwrap_or("T");
        code = code.replace(&self.type_placeholder, type_name);
        code = code.replace("{{type}}", type_name);

        // ID substitution
        let id_lower = test_template.id.to_lowercase().replace('-', "_");
        code = code.replace("{{id_lower}}", &id_lower);
        code = code.replace("{{id}}", &test_template.id);

        // Max size substitution
        code = code.replace("{{max_size}}", "1_000_000");

        // Strategy substitution for Python hypothesis
        let strategy = match type_name {
            "String" | "str" => "text",
            "Vec" | "list" => "lists(integers())",
            "f32" | "f64" | "float" => "floats(-1e6, 1e6)",
            "i32" | "i64" | "int" => "integers(-1000000, 1000000)",
            _ => "builds(lambda: None)",
        };
        code = code.replace("{{strategy}}", strategy);

        code
    }
}

impl Default for FalsifyGenerator {
    fn default() -> Self {
        Self::new()
    }
}

const DEFAULT_RUST: &str = r#"#[test]
#[ignore = "Stub: implement falsification for {{id}}"]
fn falsify_{{id_lower}}_default() {
    // Placeholder for {{id}} - replace with actual falsification logic
    assert!(false, "Not yet implemented: falsification test for {{id}}");
}"#;

const DEFAULT_PYTHON: &str = r"    # STUB: Test placeholder for {{id}} - replace with actual falsification
    pass
";

#[cfg(test)]
mod tests {
    use super::super::parser::SpecParser;
    use super::*;
    use std::path::Path;

    #[test]
    fn test_generator_creation() {
        let gen = FalsifyGenerator::new();
        assert!(!gen.module_placeholder.is_empty());
    }

    #[test]
    fn test_generate_from_spec() {
        let parser = SpecParser::new();
        let content = r#"
# Test Spec
module: test_module

## Functions
fn test_function(input: &[u8]) -> Result<Vec<u8>, Error>
"#;
        let spec = parser.parse(content, Path::new("test.md")).expect("unexpected failure");
        let template = FalsificationTemplate::default();
        let gen = FalsifyGenerator::new();

        let tests =
            gen.generate(&spec, &template, TargetLanguage::Rust).expect("unexpected failure");
        assert!(!tests.is_empty());

        // Check that we got tests (module substitution may vary based on template)
        for test in &tests {
            // Tests should have valid IDs
            assert!(!test.id.is_empty(), "Test ID should not be empty");
        }
    }

    #[test]
    fn test_placeholder_substitution() {
        let gen = FalsifyGenerator::new();
        let template = "fn test_{{module}}_{{function}}()";

        let spec = ParsedSpec {
            name: "test".to_string(),
            module: "my_module".to_string(),
            requirements: vec![],
            types: vec!["MyType".to_string()],
            functions: vec!["my_func".to_string()],
            tolerances: None,
        };

        let test_template = super::super::template::TestTemplate {
            id: "TEST-001".to_string(),
            name: "Test".to_string(),
            description: "Test".to_string(),
            severity: TestSeverity::Medium,
            points: 5,
            rust_template: Some(template.to_string()),
            python_template: None,
        };

        let result = gen.substitute_placeholders(template, &spec, &test_template);
        assert!(result.contains("my_module"));
        assert!(result.contains("my_func"));
    }

    #[test]
    fn test_generator_default() {
        let gen = FalsifyGenerator::default();
        assert_eq!(gen.module_placeholder, "{{module}}");
    }

    #[test]
    fn test_target_language_variants() {
        assert_ne!(TargetLanguage::Rust, TargetLanguage::Python);
        assert_eq!(TargetLanguage::Rust, TargetLanguage::Rust);
    }

    #[test]
    fn test_generated_test_fields() {
        let test = GeneratedTest {
            id: "BC-001".to_string(),
            name: "Boundary Test".to_string(),
            category: "boundary".to_string(),
            points: 4,
            severity: TestSeverity::High,
            code: "#[test]\nfn test_boundary() {}".to_string(),
        };
        assert_eq!(test.id, "BC-001");
        assert_eq!(test.points, 4);
        assert_eq!(test.severity, TestSeverity::High);
    }

    #[test]
    fn test_generate_python() {
        let parser = SpecParser::new();
        let content = "module: test\n- MUST work";
        let spec = parser.parse(content, Path::new("test.md")).expect("unexpected failure");
        let template = FalsificationTemplate::default();
        let gen = FalsifyGenerator::new();

        let tests =
            gen.generate(&spec, &template, TargetLanguage::Python).expect("unexpected failure");
        assert!(!tests.is_empty());
    }

    #[test]
    fn test_placeholder_id_substitution() {
        let gen = FalsifyGenerator::new();
        let template = "test_{{id_lower}} {{id}}";

        let spec = ParsedSpec {
            name: "test".to_string(),
            module: "mod".to_string(),
            requirements: vec![],
            types: vec![],
            functions: vec![],
            tolerances: None,
        };

        let test_template = super::super::template::TestTemplate {
            id: "BC-001".to_string(),
            name: "Test".to_string(),
            description: "Test".to_string(),
            severity: TestSeverity::Medium,
            points: 5,
            rust_template: Some(template.to_string()),
            python_template: None,
        };

        let result = gen.substitute_placeholders(template, &spec, &test_template);
        assert!(result.contains("bc_001"));
        assert!(result.contains("BC-001"));
    }

    #[test]
    fn test_placeholder_type_substitution() {
        let gen = FalsifyGenerator::new();
        let template = "type: {{type}}";

        let spec = ParsedSpec {
            name: "test".to_string(),
            module: "mod".to_string(),
            requirements: vec![],
            types: vec!["MyStruct".to_string()],
            functions: vec![],
            tolerances: None,
        };

        let test_template = super::super::template::TestTemplate {
            id: "TEST-001".to_string(),
            name: "Test".to_string(),
            description: "Test".to_string(),
            severity: TestSeverity::Low,
            points: 2,
            rust_template: Some(template.to_string()),
            python_template: None,
        };

        let result = gen.substitute_placeholders(template, &spec, &test_template);
        assert!(result.contains("MyStruct"));
    }

    #[test]
    fn test_placeholder_max_size_substitution() {
        let gen = FalsifyGenerator::new();
        let template = "size: {{max_size}}";

        let spec = ParsedSpec {
            name: "test".to_string(),
            module: "mod".to_string(),
            requirements: vec![],
            types: vec![],
            functions: vec![],
            tolerances: None,
        };

        let test_template = super::super::template::TestTemplate {
            id: "TEST-001".to_string(),
            name: "Test".to_string(),
            description: "Test".to_string(),
            severity: TestSeverity::Medium,
            points: 5,
            rust_template: Some(template.to_string()),
            python_template: None,
        };

        let result = gen.substitute_placeholders(template, &spec, &test_template);
        assert!(result.contains("1_000_000"));
    }

    #[test]
    fn test_strategy_substitution_string() {
        let gen = FalsifyGenerator::new();
        let template = "{{strategy}}";

        let spec = ParsedSpec {
            name: "test".to_string(),
            module: "mod".to_string(),
            requirements: vec![],
            types: vec!["String".to_string()],
            functions: vec![],
            tolerances: None,
        };

        let test_template = super::super::template::TestTemplate {
            id: "TEST-001".to_string(),
            name: "Test".to_string(),
            description: "Test".to_string(),
            severity: TestSeverity::Medium,
            points: 5,
            rust_template: Some(template.to_string()),
            python_template: None,
        };

        let result = gen.substitute_placeholders(template, &spec, &test_template);
        assert!(result.contains("text"));
    }

    #[test]
    fn test_strategy_substitution_float() {
        let gen = FalsifyGenerator::new();
        let template = "{{strategy}}";

        let spec = ParsedSpec {
            name: "test".to_string(),
            module: "mod".to_string(),
            requirements: vec![],
            types: vec!["f64".to_string()],
            functions: vec![],
            tolerances: None,
        };

        let test_template = super::super::template::TestTemplate {
            id: "TEST-001".to_string(),
            name: "Test".to_string(),
            description: "Test".to_string(),
            severity: TestSeverity::Medium,
            points: 5,
            rust_template: Some(template.to_string()),
            python_template: None,
        };

        let result = gen.substitute_placeholders(template, &spec, &test_template);
        assert!(result.contains("floats"));
    }

    #[test]
    fn test_fallback_defaults() {
        let gen = FalsifyGenerator::new();
        let template = "{{function}} {{type}}";

        let spec = ParsedSpec {
            name: "test".to_string(),
            module: "mod".to_string(),
            requirements: vec![],
            types: vec![],
            functions: vec![],
            tolerances: None,
        };

        let test_template = super::super::template::TestTemplate {
            id: "TEST-001".to_string(),
            name: "Test".to_string(),
            description: "Test".to_string(),
            severity: TestSeverity::Medium,
            points: 5,
            rust_template: Some(template.to_string()),
            python_template: None,
        };

        let result = gen.substitute_placeholders(template, &spec, &test_template);
        assert!(result.contains("function"));
        assert!(result.contains('T'));
    }
}
