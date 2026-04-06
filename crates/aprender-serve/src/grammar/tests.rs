#[cfg(test)]
mod tests {
    use crate::grammar::*;

    #[test]
    fn test_deep_grcov_json_schema_object_multiple_properties() {
        // Test JsonSchemaType::Object with multiple properties
        let schema = JsonSchemaType::Object(vec![
            ("prop1".to_string(), JsonSchemaType::String, true),
            ("prop2".to_string(), JsonSchemaType::Integer, false),
            ("prop3".to_string(), JsonSchemaType::Boolean, true),
        ]);
        let grammar = grammar_from_json_schema(&schema);

        assert!(grammar.get_rule("root_prop1").is_some());
        assert!(grammar.get_rule("root_prop2").is_some());
        assert!(grammar.get_rule("root_prop3").is_some());
    }

    #[test]
    fn test_deep_grcov_parse_anthropic_multiple_calls() {
        // Test Anthropic format with multiple tool calls
        let tools = vec![
            ToolDefinition::new("tool1", "Tool 1", vec![]),
            ToolDefinition::new("tool2", "Tool 2", vec![]),
        ];
        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Anthropic);

        let text = r#"<tool_use><name>tool1</name><input>{}</input></tool_use>
                      Some text
                      <tool_use><name>tool2</name><input>{"key": "value"}</input></tool_use>"#;
        let calls = parser.parse(text);
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn test_deep_grcov_parse_hermes_multiple_calls() {
        // Test Hermes format with multiple tool calls
        let tools = vec![
            ToolDefinition::new("tool1", "Tool 1", vec![]),
            ToolDefinition::new("tool2", "Tool 2", vec![]),
        ];
        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Hermes);

        let text = r#"<tool_call>{"name": "tool1", "arguments": {}}</tool_call>
                      <tool_call>{"name": "tool2", "arguments": {"x": 1}}</tool_call>"#;
        let calls = parser.parse(text);
        assert_eq!(calls.len(), 2);
    }
include!("tests_grammar_element_alternative.rs");
include!("tests_tool_parameter.rs");
include!("tests_tool_definition_call.rs");
include!("tests_deep_grcov.rs");
}
