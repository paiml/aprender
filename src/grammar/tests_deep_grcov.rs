
    #[test]
    fn test_deep_grcov_grammar_masker_multi_char_token_first_invalid() {
        // Test multi-char token where first char is invalid
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        let mut token_strings = HashMap::new();
        token_strings.insert(0, "xy".to_string()); // First char 'x' is invalid

        let masker = GrammarTokenMasker::new(grammar, token_strings, 99).expect("should create");
        let mask = masker.get_mask();
        assert!(!mask.is_allowed(0)); // Token should not be allowed
    }

    #[test]
    fn test_deep_grcov_grammar_masker_empty_token() {
        // Test with empty token string
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        let mut token_strings = HashMap::new();
        token_strings.insert(0, String::new()); // Empty token

        let masker = GrammarTokenMasker::new(grammar, token_strings, 99).expect("should create");
        let mask = masker.get_mask();
        // Empty token won't have a first char, so it shouldn't be in allowed
        assert!(!mask.is_allowed(0));
    }

    #[test]
    fn test_deep_grcov_find_matching_brace_escaped_quote() {
        // Test find_matching_brace with escaped quotes
        let text = r#"{"key": "value with \" escaped"}"#;
        let result = find_matching_brace(text);
        assert!(result.is_some());
    }

    #[test]
    fn test_deep_grcov_find_matching_brace_escaped_backslash() {
        // Test find_matching_brace with escaped backslash
        let text = r#"{"path": "C:\\Users\\test"}"#;
        let result = find_matching_brace(text);
        assert!(result.is_some());
    }

    #[test]
    fn test_deep_grcov_extract_xml_tag_unclosed() {
        // Test extract_xml_tag with unclosed tag
        let text = "<name>test";
        let result = extract_xml_tag(text, "name");
        assert!(result.is_none());
    }

    #[test]
    fn test_deep_grcov_parse_anthropic_format_unclosed() {
        // Test Anthropic format with unclosed tool_use
        let tools = vec![ToolDefinition::new("test", "Test", vec![])];
        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Anthropic);

        let text = "<tool_use><name>test</name><input>{}</input>"; // No closing tag
        let calls = parser.parse(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_deep_grcov_parse_anthropic_format_missing_name() {
        // Test Anthropic format with missing name
        let tools = vec![ToolDefinition::new("test", "Test", vec![])];
        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Anthropic);

        let text = "<tool_use><input>{}</input></tool_use>";
        let calls = parser.parse(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_deep_grcov_parse_anthropic_format_unknown_tool() {
        // Test Anthropic format with unknown tool
        let tools = vec![ToolDefinition::new("known_tool", "Known", vec![])];
        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Anthropic);

        let text = "<tool_use><name>unknown_tool</name><input>{}</input></tool_use>";
        let calls = parser.parse(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_deep_grcov_parse_hermes_format_unclosed() {
        // Test Hermes format with unclosed tool_call
        let tools = vec![ToolDefinition::new("test", "Test", vec![])];
        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Hermes);

        let text = r#"<tool_call>{"name": "test", "arguments": {}}"#; // No closing tag
        let calls = parser.parse(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_deep_grcov_parse_hermes_format_invalid_json() {
        // Test Hermes format with invalid JSON
        let tools = vec![ToolDefinition::new("test", "Test", vec![])];
        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Hermes);

        let text = "<tool_call>not valid json</tool_call>";
        let calls = parser.parse(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_deep_grcov_parse_hermes_format_string_arguments() {
        // Test Hermes format with string arguments
        let tools = vec![ToolDefinition::new("test", "Test", vec![])];
        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Hermes);

        let text =
            r#"<tool_call>{"name": "test", "arguments": "{\"key\": \"value\"}"}</tool_call>"#;
        let calls = parser.parse(text);
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn test_deep_grcov_parse_hermes_format_unknown_tool() {
        // Test Hermes format with unknown tool
        let tools = vec![ToolDefinition::new("known", "Known", vec![])];
        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Hermes);

        let text = r#"<tool_call>{"name": "unknown", "arguments": {}}</tool_call>"#;
        let calls = parser.parse(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_deep_grcov_generate_tool_grammar_with_all_param_types() {
        // Test generate_tool_grammar with all parameter types
        let tools = vec![ToolDefinition::new(
            "complex_tool",
            "A complex tool",
            vec![
                ToolParameter {
                    name: "str_param".to_string(),
                    description: "String".to_string(),
                    param_type: ToolParameterType::String,
                    required: true,
                    default: None,
                },
                ToolParameter {
                    name: "int_param".to_string(),
                    description: "Integer".to_string(),
                    param_type: ToolParameterType::Integer,
                    required: true,
                    default: None,
                },
                ToolParameter {
                    name: "num_param".to_string(),
                    description: "Number".to_string(),
                    param_type: ToolParameterType::Number,
                    required: true,
                    default: None,
                },
                ToolParameter {
                    name: "bool_param".to_string(),
                    description: "Boolean".to_string(),
                    param_type: ToolParameterType::Boolean,
                    required: true,
                    default: None,
                },
                ToolParameter {
                    name: "enum_param".to_string(),
                    description: "Enum".to_string(),
                    param_type: ToolParameterType::Enum(vec!["a".to_string(), "b".to_string()]),
                    required: true,
                    default: None,
                },
                ToolParameter {
                    name: "array_param".to_string(),
                    description: "Array".to_string(),
                    param_type: ToolParameterType::Array {
                        items: Box::new(ToolParameterType::Integer),
                    },
                    required: true,
                    default: None,
                },
                ToolParameter {
                    name: "object_param".to_string(),
                    description: "Object".to_string(),
                    param_type: ToolParameterType::Object {
                        properties: vec![ToolParameter::required_string("nested", "Nested")],
                    },
                    required: true,
                    default: None,
                },
            ],
        )];

        let grammar = generate_tool_grammar(&tools);
        assert!(grammar.get_rule("root").is_some());
    }

    #[test]
    fn test_deep_grcov_generate_params_grammar_empty() {
        // Test generate_params_grammar with empty params
        let tools = vec![ToolDefinition::new("no_params", "No params", vec![])];

        let grammar = generate_tool_grammar(&tools);
        // Should have created an empty object grammar
        assert!(grammar.get_rule("root").is_some());
    }

    #[test]
    fn test_deep_grcov_tool_call_parser_get_tool() {
        // Test get_tool method
        let tools = vec![
            ToolDefinition::new("tool1", "Tool 1", vec![]),
            ToolDefinition::new("tool2", "Tool 2", vec![]),
        ];

        let parser = ToolCallParser::new(tools);
        assert!(parser.get_tool("tool1").is_some());
        assert!(parser.get_tool("tool2").is_some());
        assert!(parser.get_tool("tool3").is_none());
    }

    #[test]
    fn test_deep_grcov_grammar_rule_names() {
        // Test rule_names iterator
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));
        grammar.add_rule(GrammarRule::single(
            "other",
            vec![GrammarElement::Char('b')],
        ));

        let names: Vec<_> = grammar.rule_names().collect();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&&"root".to_string()));
        assert!(names.contains(&&"other".to_string()));
    }

    #[test]
    fn test_deep_grcov_state_machine_has_valid_continuation_empty() {
        // Test has_valid_continuation behavior
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        let mut sm = GrammarStateMachine::new(grammar).expect("should create");
        assert!(sm.has_valid_continuation()); // Initially has states

        // Advance with invalid char - returns false but preserves states
        let advanced = sm.advance('x'); // Invalid char
        assert!(!advanced); // Should return false
                            // Note: current impl preserves states on failed advance (states not cleared)
                            // This allows retry with different char
        assert!(sm.has_valid_continuation());
    }

    #[test]
    fn test_deep_grcov_grammar_with_root() {
        // Test Grammar::with_root
        let grammar = Grammar::with_root("my_root");
        assert_eq!(grammar.root(), "my_root");
        assert!(grammar.is_empty());
    }

    #[test]
    fn test_deep_grcov_grammar_set_root() {
        // Test Grammar::set_root
        let mut grammar = Grammar::new();
        grammar.set_root("new_root");
        assert_eq!(grammar.root(), "new_root");
    }

    #[test]
    fn test_deep_grcov_state_machine_rule_ref_none_alternatives() {
        // Test RuleRef advancing with no matching alternatives
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![GrammarElement::RuleRef("sub".to_string())],
        ));
        grammar.add_rule(GrammarRule::single("sub", vec![GrammarElement::Char('x')]));

        let mut sm = GrammarStateMachine::new(grammar).expect("should create");
        // Try to advance with a char that doesn't match sub rule
        assert!(!sm.advance('y')); // 'y' doesn't match 'x' in sub
    }

    #[test]
    fn test_deep_grcov_parse_openai_nested_json() {
        // Test OpenAI parser with deeply nested JSON
        let tools = vec![ToolDefinition::new("nested", "Nested", vec![])];
        let mut parser = ToolCallParser::new(tools);

        let text = r#"{"name": "nested", "arguments": {"level1": {"level2": {"level3": "deep"}}}}"#;
        let calls = parser.parse(text);
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn test_deep_grcov_parse_openai_multiple_json_blocks() {
        // Test OpenAI parser skipping non-tool JSON
        let tools = vec![ToolDefinition::new("tool", "Tool", vec![])];
        let mut parser = ToolCallParser::new(tools);

        let text = r#"{"other": "json"} {"name": "tool", "arguments": {}} {"another": "block"}"#;
        let calls = parser.parse(text);
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn test_deep_grcov_grammar_validate_undefined_rule_ref() {
        // Test validation with undefined rule reference
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![
                GrammarElement::Char('a'),
                GrammarElement::RuleRef("undefined".to_string()),
            ],
        ));

        let result = grammar.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_deep_grcov_grammar_element_clone_and_eq() {
        // Test Clone and Eq for GrammarElement
        let elem1 = GrammarElement::CharRange('a', 'z');
        let elem2 = elem1.clone();
        assert_eq!(elem1, elem2);

        let elem3 = GrammarElement::RuleRef("test".to_string());
        let elem4 = elem3.clone();
        assert_eq!(elem3, elem4);
    }

    #[test]
    fn test_deep_grcov_grammar_alternative_clone() {
        // Test Clone for GrammarAlternative
        let alt1 =
            GrammarAlternative::new(vec![GrammarElement::Char('a'), GrammarElement::Char('b')]);
        let alt2 = alt1.clone();
        assert_eq!(alt1, alt2);
    }

    #[test]
    fn test_deep_grcov_grammar_rule_clone() {
        // Test Clone for GrammarRule
        let rule1 = GrammarRule::single("test", vec![GrammarElement::Char('x')]);
        let rule2 = rule1.clone();
        assert_eq!(rule1, rule2);
    }

    #[test]
    fn test_deep_grcov_grammar_state_clone_and_hash() {
        use std::collections::HashSet;

        // Test Clone and Hash for GrammarState
        let state1 = GrammarState::initial("root");
        let state2 = state1.clone();
        assert_eq!(state1, state2);

        // Test that it can be used in HashSet
        let mut set = HashSet::new();
        set.insert(state1.clone());
        assert!(set.contains(&state2));
    }

    #[test]
    fn test_deep_grcov_token_mask_num_allowed() {
        // Test TokenMask::num_allowed
        let allowed: HashSet<u32> = vec![1, 2, 3, 4, 5].into_iter().collect();
        let mask = TokenMask::from_allowed(allowed, true);
        assert_eq!(mask.num_allowed(), 5);
    }

    #[test]
    fn test_deep_grcov_tool_result_default_success() {
        // Test ToolResult deserialization with default success
        let json = r#"{"tool_call_id": "id", "content": "data"}"#;
        let result: ToolResult = serde_json::from_str(json).expect("should parse");
        assert!(result.success); // Default is true
    }

    #[test]
    fn test_deep_grcov_tool_parameter_type_object_serialization() {
        // Test serialization of Object parameter type
        let obj_type = ToolParameterType::Object {
            properties: vec![ToolParameter::required_int("count", "Count value")],
        };

        let json = serde_json::to_string(&obj_type).expect("should serialize");
        let deserialized: ToolParameterType =
            serde_json::from_str(&json).expect("should deserialize");

        match deserialized {
            ToolParameterType::Object { properties } => {
                assert_eq!(properties.len(), 1);
                assert_eq!(properties[0].name, "count");
            },
            _ => panic!("Expected Object type"),
        }
    }

    #[test]
    fn test_deep_grcov_state_machine_advance_any() {
        // Test advance with Any element
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![GrammarElement::Any, GrammarElement::Char('!')],
        ));

        let mut sm = GrammarStateMachine::new(grammar).expect("should create");
        assert!(sm.advance('x')); // Any accepts any char
        assert!(sm.advance('!')); // Then literal
        assert!(sm.is_complete());
    }

    #[test]
    fn test_deep_grcov_state_machine_advance_char_not() {
        // Test advance with CharNot element
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![GrammarElement::CharNot(vec!['x'])],
        ));

        let mut sm = GrammarStateMachine::new(grammar.clone()).expect("should create");
        assert!(sm.advance('a')); // 'a' is not excluded

        let mut sm2 = GrammarStateMachine::new(grammar).expect("should create");
        assert!(!sm2.advance('x')); // 'x' is excluded
    }

    #[test]
    fn test_deep_grcov_json_schema_nested_array() {
        // Test JsonSchemaType::Array with Array items
        let schema = JsonSchemaType::Array(Box::new(JsonSchemaType::Array(Box::new(
            JsonSchemaType::Integer,
        ))));
        let grammar = grammar_from_json_schema(&schema);

        assert!(grammar.get_rule("root").is_some());
        assert!(grammar.get_rule("root_item").is_some());
    }
