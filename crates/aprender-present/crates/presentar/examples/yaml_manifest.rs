//! YAML Manifest example demonstrating declarative app configuration.
//!
//! Run with: `cargo run --example yaml_manifest`

use presentar_yaml::{DataContext, ExpressionExecutor, ExpressionParser, Manifest, Value};
use std::collections::HashMap;

const APP_MANIFEST: &str = r#"
presentar: "0.1"
name: Analytics Dashboard
version: "1.0.0"
description: Real-time business analytics with Presentar

data:
  sales:
    source: api://analytics/sales
    format: json
    refresh: 30s
  users:
    source: pacha://users.ald
    format: ald

models:
  prediction:
    source: pacha://forecast.apr
    format: apr

layout:
  type: dashboard
  columns: 12
  gap: 16
  sections:
    - id: header
      span: [1, 12]
      widgets:
        - type: text
          content: "Analytics Dashboard"
          style: heading
    - id: metrics
      span: [1, 12]
      widgets:
        - type: data-card
          data: "{{ sales | sum(revenue) }}"
        - type: data-card
          data: "{{ users | count }}"
    - id: chart
      span: [1, 8]
      widgets:
        - type: chart
          chart_type: line
          data: "{{ sales | filter(status=completed) }}"
          x: date
          y: revenue
    - id: table
      span: [9, 12]
      widgets:
        - type: data-table
          data: "{{ sales | sort(date, desc) | limit(10) }}"
          columns: [date, customer, revenue]
"#;

fn main() {
    println!("=== Presentar YAML Manifest Example ===\n");

    // Parse the manifest
    let manifest = Manifest::from_yaml(APP_MANIFEST).expect("valid YAML manifest");

    println!("📋 Manifest Info:");
    println!("   Name: {}", manifest.name);
    println!("   Version: {}", manifest.version);
    println!("   Description: {}", manifest.description);

    println!("\n📊 Data Sources:");
    for (name, source) in &manifest.data {
        println!("   - {} ({}): {}", name, source.format, source.source);
        if let Some(refresh) = &source.refresh {
            println!("     Refresh: {refresh}");
        }
    }

    println!("\n🤖 Models:");
    for (name, model) in &manifest.models {
        println!("   - {} ({}): {}", name, model.format, model.source);
    }

    println!("\n📐 Layout:");
    println!("   Type: {}", manifest.layout.layout_type);
    println!("   Columns: {}", manifest.layout.columns);
    println!("   Gap: {}px", manifest.layout.gap);
    println!("   Sections: {}", manifest.layout.sections.len());

    for section in &manifest.layout.sections {
        println!("   - {} ({} widgets)", section.id, section.widgets.len());
    }

    // Demonstrate expression parsing and execution
    println!("\n🔧 Expression Execution Demo:");

    let parser = ExpressionParser::new();
    let executor = ExpressionExecutor::new();

    // Create sample data context
    let mut ctx = DataContext::new();
    let sales = Value::Array(vec![
        Value::Object({
            let mut m = HashMap::new();
            m.insert("date".into(), Value::String("2024-01-01".into()));
            m.insert("revenue".into(), Value::Number(1500.0));
            m.insert("status".into(), Value::String("completed".into()));
            m
        }),
        Value::Object({
            let mut m = HashMap::new();
            m.insert("date".into(), Value::String("2024-01-02".into()));
            m.insert("revenue".into(), Value::Number(2300.0));
            m.insert("status".into(), Value::String("completed".into()));
            m
        }),
        Value::Object({
            let mut m = HashMap::new();
            m.insert("date".into(), Value::String("2024-01-03".into()));
            m.insert("revenue".into(), Value::Number(1800.0));
            m.insert("status".into(), Value::String("pending".into()));
            m
        }),
    ]);
    ctx.insert("sales", sales);

    // Execute expressions
    let expressions = [
        "{{ sales | count }}",
        "{{ sales | sum(revenue) }}",
        "{{ sales | filter(status=completed) | count }}",
        "{{ sales | mean(revenue) }}",
    ];

    for expr_str in expressions {
        let expr = parser.parse(expr_str).expect("valid expression");
        let result = executor.execute(&expr, &ctx).expect("execute");
        println!("   {expr_str} = {result:?}");
    }

    // Serialize back to YAML
    println!("\n📝 Roundtrip Test:");
    let yaml_output = manifest.to_yaml().expect("serialize");
    let reparsed = Manifest::from_yaml(&yaml_output).expect("reparse");
    println!("   Original name: {}", manifest.name);
    println!("   Reparsed name: {}", reparsed.name);
    println!("   ✓ Roundtrip successful!");

    println!("\n=== YAML Manifest Example Complete ===");
}
