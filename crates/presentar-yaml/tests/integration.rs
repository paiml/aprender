#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! Integration tests for presentar-yaml.
//!
//! These tests verify YAML manifest parsing and expression execution end-to-end.

use presentar_yaml::{DataContext, ExpressionExecutor, ExpressionParser, Manifest, Value};
use std::collections::HashMap;

// =============================================================================
// Manifest Parsing Integration Tests
// =============================================================================

const DASHBOARD_YAML: &str = r#"
presentar: "0.1"
name: Sales Dashboard
version: "1.0.0"
description: Real-time sales analytics

data:
  transactions:
    source: api://transactions
    format: json
    refresh: 30s
  products:
    source: file://data/products.json
    format: json

layout:
  type: dashboard
  columns: 12
  gap: 16
  sections:
    - id: header
      span: [1, 12]
      widgets:
        - type: text
          content: Sales Dashboard
          style: heading
    - id: cards
      span: [1, 12]
      widgets:
        - type: data-card
          data: "{{ transactions | sum(amount) }}"
        - type: data-card
          data: "{{ transactions | count }}"
"#;

#[test]
fn test_parse_dashboard_manifest() {
    let manifest = Manifest::from_yaml(DASHBOARD_YAML).expect("valid YAML");

    assert_eq!(manifest.name, "Sales Dashboard");
    assert_eq!(manifest.version, "1.0.0");
    assert!(!manifest.description.is_empty());
}

#[test]
fn test_manifest_data_sources() {
    let manifest = Manifest::from_yaml(DASHBOARD_YAML).expect("valid YAML");

    assert!(manifest.data.contains_key("transactions"));
    assert!(manifest.data.contains_key("products"));

    let transactions = &manifest.data["transactions"];
    assert_eq!(transactions.source, "api://transactions");
    assert_eq!(transactions.format, "json");
}

#[test]
fn test_manifest_layout_structure() {
    let manifest = Manifest::from_yaml(DASHBOARD_YAML).expect("valid YAML");

    assert_eq!(manifest.layout.layout_type, "dashboard");
    assert_eq!(manifest.layout.columns, 12);
    assert_eq!(manifest.layout.sections.len(), 2);
    assert_eq!(manifest.layout.sections[0].id, "header");
    assert_eq!(manifest.layout.sections[1].id, "cards");
}

#[test]
fn test_manifest_widgets_in_sections() {
    let manifest = Manifest::from_yaml(DASHBOARD_YAML).expect("valid YAML");

    let header_section = &manifest.layout.sections[0];
    assert_eq!(header_section.widgets.len(), 1);
    assert_eq!(header_section.widgets[0].widget_type, "text");

    let cards_section = &manifest.layout.sections[1];
    assert_eq!(cards_section.widgets.len(), 2);
    assert_eq!(cards_section.widgets[0].widget_type, "data-card");
}

#[test]
fn test_manifest_roundtrip() {
    let manifest = Manifest::from_yaml(DASHBOARD_YAML).expect("parse");
    let yaml_output = manifest.to_yaml().expect("serialize");
    let reparsed = Manifest::from_yaml(&yaml_output).expect("reparse");

    assert_eq!(manifest.name, reparsed.name);
    assert_eq!(manifest.version, reparsed.version);
}

// =============================================================================
// Expression Parsing Integration Tests
// =============================================================================

#[test]
fn test_parse_simple_expression() {
    let parser = ExpressionParser::new();
    let expr = parser.parse("data.transactions").expect("valid expression");

    assert_eq!(expr.source, "data.transactions");
    assert!(expr.transforms.is_empty());
}

#[test]
fn test_parse_expression_with_transforms() {
    let parser = ExpressionParser::new();
    let expr = parser
        .parse("{{ data.items | filter(active=true) | sort(date) | limit(10) }}")
        .expect("valid expression");

    assert_eq!(expr.source, "data.items");
    assert_eq!(expr.transforms.len(), 3);
}

#[test]
fn test_parse_expression_whitespace_handling() {
    let parser = ExpressionParser::new();

    // Various whitespace patterns
    let expr1 = parser.parse("  data.items  ").expect("with spaces");
    let expr2 = parser.parse("{{data.items}}").expect("no spaces");
    let expr3 = parser.parse("{{ data.items }}").expect("with spaces");

    assert_eq!(expr1.source, "data.items");
    assert_eq!(expr2.source, "data.items");
    assert_eq!(expr3.source, "data.items");
}

// =============================================================================
// Expression Execution Integration Tests
// =============================================================================

fn sample_context() -> DataContext {
    let mut ctx = DataContext::new();

    // Add sample transactions
    let transactions = Value::Array(vec![
        Value::Object({
            let mut m = HashMap::new();
            m.insert("id".into(), Value::Number(1.0));
            m.insert("amount".into(), Value::Number(100.0));
            m.insert("status".into(), Value::String("completed".into()));
            m
        }),
        Value::Object({
            let mut m = HashMap::new();
            m.insert("id".into(), Value::Number(2.0));
            m.insert("amount".into(), Value::Number(250.0));
            m.insert("status".into(), Value::String("completed".into()));
            m
        }),
        Value::Object({
            let mut m = HashMap::new();
            m.insert("id".into(), Value::Number(3.0));
            m.insert("amount".into(), Value::Number(75.0));
            m.insert("status".into(), Value::String("pending".into()));
            m
        }),
    ]);

    ctx.insert("transactions", transactions);
    ctx
}

#[test]
fn test_execute_count_transform() {
    let parser = ExpressionParser::new();
    let executor = ExpressionExecutor::new();
    let ctx = sample_context();

    let expr = parser.parse("{{ transactions | count }}").expect("parse");
    let result = executor.execute(&expr, &ctx).expect("execute");

    assert_eq!(result.as_number(), Some(3.0));
}

#[test]
fn test_execute_sum_transform() {
    let parser = ExpressionParser::new();
    let executor = ExpressionExecutor::new();
    let ctx = sample_context();

    let expr = parser
        .parse("{{ transactions | sum(amount) }}")
        .expect("parse");
    let result = executor.execute(&expr, &ctx).expect("execute");

    assert_eq!(result.as_number(), Some(425.0)); // 100 + 250 + 75
}

#[test]
fn test_execute_filter_transform() {
    let parser = ExpressionParser::new();
    let executor = ExpressionExecutor::new();
    let ctx = sample_context();

    let expr = parser
        .parse("{{ transactions | filter(status=completed) }}")
        .expect("parse");
    let result = executor.execute(&expr, &ctx).expect("execute");

    let arr = result.as_array().expect("should be array");
    assert_eq!(arr.len(), 2); // Only completed transactions
}

#[test]
fn test_execute_filter_and_sum() {
    let parser = ExpressionParser::new();
    let executor = ExpressionExecutor::new();
    let ctx = sample_context();

    let expr = parser
        .parse("{{ transactions | filter(status=completed) | sum(amount) }}")
        .expect("parse");
    let result = executor.execute(&expr, &ctx).expect("execute");

    assert_eq!(result.as_number(), Some(350.0)); // 100 + 250
}

#[test]
fn test_execute_limit_transform() {
    let parser = ExpressionParser::new();
    let executor = ExpressionExecutor::new();
    let ctx = sample_context();

    let expr = parser
        .parse("{{ transactions | limit(2) }}")
        .expect("parse");
    let result = executor.execute(&expr, &ctx).expect("execute");

    let arr = result.as_array().expect("should be array");
    assert_eq!(arr.len(), 2);
}

#[test]
fn test_execute_mean_transform() {
    let parser = ExpressionParser::new();
    let executor = ExpressionExecutor::new();
    let ctx = sample_context();

    let expr = parser
        .parse("{{ transactions | mean(amount) }}")
        .expect("parse");
    let result = executor.execute(&expr, &ctx).expect("execute");

    let mean = result.as_number().expect("should be number");
    let expected = (100.0 + 250.0 + 75.0) / 3.0;
    assert!((mean - expected).abs() < 0.01);
}

// =============================================================================
// Error Handling Integration Tests
// =============================================================================

#[test]
fn test_invalid_yaml_manifest() {
    let invalid_yaml = "name: Test\n  invalid: indentation";
    let result = Manifest::from_yaml(invalid_yaml);
    assert!(result.is_err());
}

#[test]
fn test_execute_nonexistent_field() {
    let parser = ExpressionParser::new();
    let executor = ExpressionExecutor::new();
    let ctx = DataContext::new();

    let expr = parser.parse("nonexistent").expect("parse");
    let result = executor.execute(&expr, &ctx);

    // Missing data returns an error
    assert!(result.is_err());
}

// =============================================================================
// Complex Workflow Integration Tests
// =============================================================================

#[test]
fn test_analytics_dashboard_workflow() {
    let parser = ExpressionParser::new();
    let executor = ExpressionExecutor::new();
    let ctx = sample_context();

    // Simulate dashboard card computations
    let total_expr = parser.parse("{{ transactions | sum(amount) }}").unwrap();
    let count_expr = parser.parse("{{ transactions | count }}").unwrap();
    let avg_expr = parser.parse("{{ transactions | mean(amount) }}").unwrap();
    let completed_expr = parser
        .parse("{{ transactions | filter(status=completed) | count }}")
        .unwrap();

    let total = executor.execute(&total_expr, &ctx).unwrap();
    let count = executor.execute(&count_expr, &ctx).unwrap();
    let avg = executor.execute(&avg_expr, &ctx).unwrap();
    let completed = executor.execute(&completed_expr, &ctx).unwrap();

    assert_eq!(total.as_number(), Some(425.0));
    assert_eq!(count.as_number(), Some(3.0));
    assert!((avg.as_number().unwrap() - 141.67).abs() < 0.1);
    assert_eq!(completed.as_number(), Some(2.0));
}
