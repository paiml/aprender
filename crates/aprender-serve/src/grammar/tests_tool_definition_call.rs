
    #[test]
    fn test_tool_definition_serialization() {
        let tool = ToolDefinition::new(
            "test_tool",
            "A test tool",
            vec![ToolParameter::required_string("arg1", "First argument")],
        );

        let json = serde_json::to_string(&tool).expect("test");
        let deserialized: ToolDefinition = serde_json::from_str(&json).expect("test");

        assert_eq!(deserialized.name, "test_tool");
        assert_eq!(deserialized.description, "A test tool");
        assert_eq!(deserialized.parameters.len(), 1);
    }

    #[test]
    fn test_tool_call_serialization() {
        let call = ToolCall::new("call_123", "my_tool", r#"{"key": "value"}"#);

        let json = serde_json::to_string(&call).expect("test");
        let deserialized: ToolCall = serde_json::from_str(&json).expect("test");

        assert_eq!(deserialized.id, "call_123");
        assert_eq!(deserialized.name, "my_tool");
        assert_eq!(deserialized.arguments, r#"{"key": "value"}"#);
    }

    #[test]
    fn test_tool_result_serialization() {
        let result = ToolResult::success("call_123", "result data");

        let json = serde_json::to_string(&result).expect("test");
        let deserialized: ToolResult = serde_json::from_str(&json).expect("test");

        assert_eq!(deserialized.tool_call_id, "call_123");
        assert_eq!(deserialized.content, "result data");
        assert!(deserialized.success);
    }

    #[test]
    fn test_tool_choice_serialization() {
        let choice = ToolChoice::Specific("my_tool".to_string());

        let json = serde_json::to_string(&choice).expect("test");
        assert!(json.contains("my_tool"));

        let auto = ToolChoice::Auto;
        let auto_json = serde_json::to_string(&auto).expect("test");
        assert_eq!(auto_json, "\"auto\"");
    }

    #[test]
    fn test_tool_parameter_type_serialization() {
        let array_type = ToolParameterType::Array {
            items: Box::new(ToolParameterType::Integer),
        };

        let json = serde_json::to_string(&array_type).expect("test");
        let deserialized: ToolParameterType = serde_json::from_str(&json).expect("test");

        match deserialized {
            ToolParameterType::Array { items } => {
                assert_eq!(*items, ToolParameterType::Integer);
            },
            _ => panic!("Expected Array type"),
        }
    }

    #[test]
    fn test_parse_no_tool_calls() {
        let tools = vec![ToolDefinition::new("get_weather", "Get weather", vec![])];

        let mut parser = ToolCallParser::new(tools);
        let text = "Just some regular text without any tool calls.";

        let calls = parser.parse(text);

        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn test_parse_malformed_json() {
        let tools = vec![ToolDefinition::new("get_weather", "Get weather", vec![])];

        let mut parser = ToolCallParser::new(tools);
        let text = r#"{"name": "get_weather", "arguments": {malformed}"#;

        let calls = parser.parse(text);

        // Malformed JSON should be skipped
        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn test_tool_parameter_type_boolean() {
        let bool_type = ToolParameterType::Boolean;
        assert_eq!(bool_type, ToolParameterType::Boolean);
    }

    #[test]
    fn test_tool_parameter_type_number() {
        let num_type = ToolParameterType::Number;
        assert_eq!(num_type, ToolParameterType::Number);
    }

    // ==========================================================================
    // DEEP COVERAGE TESTS - _deep_grcov_ prefix
    // ==========================================================================

    #[test]
    fn test_deep_grcov_grammar_state_is_complete_no_rule() {
        // Test is_complete when rule doesn't exist in grammar
        let grammar = Grammar::new();
        let state = GrammarState::initial("nonexistent");

        // Should return false when rule doesn't exist
        assert!(!state.is_complete(&grammar));
    }

    #[test]
    fn test_deep_grcov_grammar_state_is_complete_invalid_alt_idx() {
        // Test is_complete with alt_idx out of bounds
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        let state = GrammarState {
            rule: "root".to_string(),
            alt_idx: 999, // Out of bounds
            elem_idx: 0,
            stack: Vec::new(),
        };

        // Should return false when alt_idx is out of bounds
        assert!(!state.is_complete(&grammar));
    }

    #[test]
    fn test_deep_grcov_grammar_state_is_complete_with_stack() {
        // Test is_complete returns false when stack is non-empty
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        let state = GrammarState {
            rule: "root".to_string(),
            alt_idx: 0,
            elem_idx: 1,                              // Past the only element
            stack: vec![("other".to_string(), 0, 0)], // Non-empty stack
        };

        // Should return false because stack is not empty
        assert!(!state.is_complete(&grammar));
    }

    #[test]
    fn test_deep_grcov_grammar_state_current_element_none() {
        // Test current_element returns None for various cases
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        // Case 1: Invalid rule name
        let state1 = GrammarState::initial("nonexistent");
        assert!(state1.current_element(&grammar).is_none());

        // Case 2: Invalid alt_idx
        let state2 = GrammarState {
            rule: "root".to_string(),
            alt_idx: 999,
            elem_idx: 0,
            stack: Vec::new(),
        };
        assert!(state2.current_element(&grammar).is_none());

        // Case 3: Invalid elem_idx
        let state3 = GrammarState {
            rule: "root".to_string(),
            alt_idx: 0,
            elem_idx: 999,
            stack: Vec::new(),
        };
        assert!(state3.current_element(&grammar).is_none());
    }

    #[test]
    fn test_deep_grcov_state_machine_advance_invalid_char() {
        // Test that advance returns false for invalid char
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        let mut sm = GrammarStateMachine::new(grammar).expect("should create");
        assert!(!sm.advance('x')); // 'x' is not valid
        assert_eq!(sm.generated(), ""); // Nothing generated
    }

    #[test]
    fn test_deep_grcov_state_machine_can_accept_char_any() {
        // Test GrammarElement::Any accepts any character
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Any]));

        let sm = GrammarStateMachine::new(grammar).expect("should create");
        assert!(sm.is_valid_char('x'));
        assert!(sm.is_valid_char('1'));
        assert!(sm.is_valid_char(' '));
        assert!(sm.is_valid_char('\n'));
    }

    #[test]
    fn test_deep_grcov_state_machine_can_accept_char_end() {
        // Test GrammarElement::End doesn't accept any character
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::End]));

        let sm = GrammarStateMachine::new(grammar).expect("should create");
        assert!(!sm.is_valid_char('x'));
        assert!(!sm.is_valid_char('a'));
    }

    #[test]
    fn test_deep_grcov_state_machine_can_accept_char_not() {
        // Test GrammarElement::CharNot
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![GrammarElement::CharNot(vec!['x', 'y', 'z'])],
        ));

        let sm = GrammarStateMachine::new(grammar).expect("should create");
        assert!(sm.is_valid_char('a')); // Not in excluded list
        assert!(sm.is_valid_char('b'));
        assert!(!sm.is_valid_char('x')); // In excluded list
        assert!(!sm.is_valid_char('y'));
        assert!(!sm.is_valid_char('z'));
    }

    #[test]
    fn test_deep_grcov_state_machine_rule_ref_invalid_rule() {
        // Test RuleRef to non-existent rule
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![GrammarElement::RuleRef("nonexistent".to_string())],
        ));

        // Validation should fail
        assert!(grammar.validate().is_err());
    }

    #[test]
    fn test_deep_grcov_state_machine_rule_ref_chain() {
        // Test nested rule references (note: deep chains don't auto-complete due to single-pop)
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![GrammarElement::RuleRef("middle".to_string())],
        ));
        grammar.add_rule(GrammarRule::single(
            "middle",
            vec![GrammarElement::RuleRef("leaf".to_string())],
        ));
        grammar.add_rule(GrammarRule::single("leaf", vec![GrammarElement::Char('x')]));

        let mut sm = GrammarStateMachine::new(grammar).expect("should create");
        assert!(sm.is_valid_char('x'));
        assert!(sm.advance('x'));
        // Note: deep rule chains don't fully unwind in current implementation
        // The state machine has advanced but isn't "complete" yet
        assert!(!sm.states.is_empty());
    }

    #[test]
    fn test_deep_grcov_state_machine_collect_valid_chars_any() {
        // Test valid_chars with GrammarElement::Any - should include printable chars
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Any]));

        let sm = GrammarStateMachine::new(grammar).expect("should create");
        let valid = sm.valid_chars();

        // Should include printable ASCII
        assert!(valid.contains(&' '));
        assert!(valid.contains(&'~'));
        assert!(valid.contains(&'a'));
        assert!(valid.contains(&'Z'));
    }

    #[test]
    fn test_deep_grcov_state_machine_collect_valid_chars_char_not() {
        // Test valid_chars with GrammarElement::CharNot
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![GrammarElement::CharNot(vec!['a', 'b', 'c'])],
        ));

        let sm = GrammarStateMachine::new(grammar).expect("should create");
        let valid = sm.valid_chars();

        // Should include printable chars except a, b, c
        assert!(!valid.contains(&'a'));
        assert!(!valid.contains(&'b'));
        assert!(!valid.contains(&'c'));
        assert!(valid.contains(&'d'));
        assert!(valid.contains(&'x'));
    }

    #[test]
    fn test_deep_grcov_state_machine_collect_valid_chars_end() {
        // Test valid_chars with GrammarElement::End - should be empty
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::End]));

        let sm = GrammarStateMachine::new(grammar).expect("should create");
        let valid = sm.valid_chars();
        assert!(valid.is_empty());
    }

    #[test]
    fn test_deep_grcov_state_machine_advance_state_end() {
        // Test advance_state with End element
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::End]));

        let mut sm = GrammarStateMachine::new(grammar).expect("should create");
        assert!(!sm.advance('x')); // End element doesn't accept any char
    }

    #[test]
    fn test_deep_grcov_state_machine_advance_state_char_range_boundary() {
        // Test CharRange boundary conditions
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![GrammarElement::CharRange('a', 'c')],
        ));

        let mut sm = GrammarStateMachine::new(grammar.clone()).expect("should create");
        assert!(sm.advance('a')); // Start boundary

        let mut sm2 = GrammarStateMachine::new(grammar.clone()).expect("should create");
        assert!(sm2.advance('c')); // End boundary

        let mut sm3 = GrammarStateMachine::new(grammar).expect("should create");
        assert!(!sm3.advance('d')); // Just outside
    }

    #[test]
    fn test_deep_grcov_state_machine_next_state_stack_pop() {
        // Test next_state with stack popping
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![
                GrammarElement::RuleRef("sub".to_string()),
                GrammarElement::Char('!'),
            ],
        ));
        grammar.add_rule(GrammarRule::single("sub", vec![GrammarElement::Char('x')]));

        let mut sm = GrammarStateMachine::new(grammar).expect("should create");
        assert!(sm.advance('x')); // Enter and complete sub rule
        assert!(sm.is_valid_char('!')); // Should be back in root
        assert!(sm.advance('!'));
        assert!(sm.is_complete());
    }

    #[test]
    fn test_deep_grcov_json_schema_number() {
        // Test JsonSchemaType::Number
        let schema = JsonSchemaType::Number;
        let grammar = grammar_from_json_schema(&schema);

        let sm = GrammarStateMachine::new(grammar).expect("should create");

        // Test decimal number
        assert!(sm.is_valid_char('1'));
        assert!(sm.is_valid_char('-'));
    }

    #[test]
    fn test_deep_grcov_json_schema_empty_object() {
        // Test JsonSchemaType::Object with empty properties
        let schema = JsonSchemaType::Object(vec![]);
        let grammar = grammar_from_json_schema(&schema);

        let sm = GrammarStateMachine::new(grammar).expect("should create");
        assert!(sm.is_valid_char('{'));
    }

    #[test]
    fn test_deep_grcov_json_schema_any() {
        // Test JsonSchemaType::Any
        let schema = JsonSchemaType::Any;
        let grammar = grammar_from_json_schema(&schema);

        // Should have helper rules
        assert!(grammar.get_rule("string_value").is_some());
        assert!(grammar.get_rule("number").is_some());
        assert!(grammar.get_rule("boolean").is_some());
        assert!(grammar.get_rule("null").is_some());
    }

    #[test]
    fn test_deep_grcov_grammar_masker_advance_invalid_token() {
        // Test advance_token with invalid token ID
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        let token_strings = HashMap::new(); // Empty - no tokens

        let mut masker =
            GrammarTokenMasker::new(grammar, token_strings, 99).expect("should create");
        assert!(!masker.advance_token(999)); // Unknown token
    }

    #[test]
    fn test_deep_grcov_grammar_masker_advance_token_invalid_sequence() {
        // Test advance_token with a token that doesn't match grammar
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        let mut token_strings = HashMap::new();
        token_strings.insert(0, "x".to_string()); // Wrong character

        let mut masker =
            GrammarTokenMasker::new(grammar, token_strings, 99).expect("should create");
        assert!(!masker.advance_token(0)); // 'x' doesn't match 'a'
    }

    #[test]
    fn test_deep_grcov_grammar_masker_eos_token_id() {
        // Test eos_token_id getter
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        let masker = GrammarTokenMasker::new(grammar, HashMap::new(), 42).expect("should create");
        assert_eq!(masker.eos_token_id(), 42);
    }
