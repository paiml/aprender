
    #[test]
    fn test_grammar_element_types() {
        let char_elem = GrammarElement::Char('a');
        let range_elem = GrammarElement::CharRange('a', 'z');
        let rule_ref = GrammarElement::RuleRef("test".to_string());
        let char_not = GrammarElement::CharNot(vec!['x', 'y']);
        let any = GrammarElement::Any;
        let end = GrammarElement::End;

        assert_eq!(char_elem, GrammarElement::Char('a'));
        assert_eq!(range_elem, GrammarElement::CharRange('a', 'z'));
        assert_eq!(rule_ref, GrammarElement::RuleRef("test".to_string()));
        assert_eq!(char_not, GrammarElement::CharNot(vec!['x', 'y']));
        assert_eq!(any, GrammarElement::Any);
        assert_eq!(end, GrammarElement::End);
    }

    #[test]
    fn test_grammar_alternative() {
        let alt =
            GrammarAlternative::new(vec![GrammarElement::Char('a'), GrammarElement::Char('b')]);
        assert_eq!(alt.elements.len(), 2);
        assert!(!alt.is_empty());

        let empty_alt = GrammarAlternative::new(vec![]);
        assert!(empty_alt.is_empty());

        let char_alt = GrammarAlternative::char('x');
        assert_eq!(char_alt.elements.len(), 1);
    }

    #[test]
    fn test_grammar_rule() {
        let rule = GrammarRule::new(
            "test",
            vec![GrammarAlternative::char('a'), GrammarAlternative::char('b')],
        );
        assert_eq!(rule.name, "test");
        assert_eq!(rule.alternatives.len(), 2);

        let single = GrammarRule::single("single", vec![GrammarElement::Char('x')]);
        assert_eq!(single.alternatives.len(), 1);
    }

    #[test]
    fn test_grammar_basic() {
        let mut grammar = Grammar::new();
        assert!(grammar.is_empty());

        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));
        assert!(!grammar.is_empty());
        assert_eq!(grammar.len(), 1);
        assert_eq!(grammar.root(), "root");

        assert!(grammar.get_rule("root").is_some());
        assert!(grammar.get_rule("nonexistent").is_none());
    }

    #[test]
    fn test_grammar_validation() {
        let mut grammar = Grammar::new();

        // Empty grammar fails
        assert!(grammar.validate().is_err());

        // Grammar without root rule fails
        grammar.set_root("missing");
        assert!(grammar.validate().is_err());

        // Valid grammar passes
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));
        grammar.set_root("root");
        assert!(grammar.validate().is_ok());

        // Grammar with invalid rule reference fails
        grammar.add_rule(GrammarRule::single(
            "bad",
            vec![GrammarElement::RuleRef("undefined".to_string())],
        ));
        assert!(grammar.validate().is_err());
    }

    #[test]
    fn test_grammar_state_initial() {
        let state = GrammarState::initial("root");
        assert_eq!(state.rule, "root");
        assert_eq!(state.alt_idx, 0);
        assert_eq!(state.elem_idx, 0);
        assert!(state.stack.is_empty());
    }

    #[test]
    fn test_state_machine_simple() {
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![
                GrammarElement::Char('a'),
                GrammarElement::Char('b'),
                GrammarElement::Char('c'),
            ],
        ));

        let mut sm = GrammarStateMachine::new(grammar).expect("test");

        assert!(sm.is_valid_char('a'));
        assert!(!sm.is_valid_char('b'));
        assert!(!sm.is_valid_char('x'));

        assert!(sm.advance('a'));
        assert!(sm.is_valid_char('b'));

        assert!(sm.advance('b'));
        assert!(sm.is_valid_char('c'));

        assert!(sm.advance('c'));
        assert!(sm.is_complete());
    }

    #[test]
    fn test_state_machine_alternatives() {
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::new(
            "root",
            vec![
                GrammarAlternative::new(vec![GrammarElement::Char('a')]),
                GrammarAlternative::new(vec![GrammarElement::Char('b')]),
            ],
        ));

        let sm = GrammarStateMachine::new(grammar).expect("test");

        // Both 'a' and 'b' should be valid initially
        assert!(sm.is_valid_char('a'));
        assert!(sm.is_valid_char('b'));
        assert!(!sm.is_valid_char('c'));
    }

    #[test]
    fn test_state_machine_char_range() {
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![GrammarElement::CharRange('a', 'z')],
        ));

        let sm = GrammarStateMachine::new(grammar).expect("test");

        assert!(sm.is_valid_char('a'));
        assert!(sm.is_valid_char('m'));
        assert!(sm.is_valid_char('z'));
        assert!(!sm.is_valid_char('A'));
        assert!(!sm.is_valid_char('0'));
    }

    #[test]
    fn test_state_machine_reset() {
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        let mut sm = GrammarStateMachine::new(grammar).expect("test");

        sm.advance('a');
        assert!(sm.is_complete());
        assert_eq!(sm.generated(), "a");

        sm.reset();
        assert!(!sm.is_complete());
        assert_eq!(sm.generated(), "");
        assert!(sm.is_valid_char('a'));
    }

    #[test]
    fn test_json_schema_string() {
        let schema = JsonSchemaType::String;
        let grammar = grammar_from_json_schema(&schema);

        assert!(grammar.get_rule("root").is_some());
        assert!(grammar.get_rule("string_content").is_some());
    }

    #[test]
    fn test_json_schema_integer() {
        let schema = JsonSchemaType::Integer;
        let grammar = grammar_from_json_schema(&schema);

        let sm = GrammarStateMachine::new(grammar).expect("test");
        assert!(sm.is_valid_char('0'));
        assert!(sm.is_valid_char('9'));
        assert!(sm.is_valid_char('-'));
    }

    #[test]
    fn test_json_schema_boolean() {
        let schema = JsonSchemaType::Boolean;
        let grammar = grammar_from_json_schema(&schema);

        let mut sm = GrammarStateMachine::new(grammar).expect("test");

        // Should accept 't' for 'true' or 'f' for 'false'
        assert!(sm.is_valid_char('t'));
        assert!(sm.is_valid_char('f'));
        assert!(!sm.is_valid_char('a'));

        // Test "true"
        assert!(sm.advance('t'));
        assert!(sm.advance('r'));
        assert!(sm.advance('u'));
        assert!(sm.advance('e'));
        assert!(sm.is_complete());
    }

    #[test]
    fn test_json_schema_null() {
        let schema = JsonSchemaType::Null;
        let grammar = grammar_from_json_schema(&schema);

        let mut sm = GrammarStateMachine::new(grammar).expect("test");

        assert!(sm.advance('n'));
        assert!(sm.advance('u'));
        assert!(sm.advance('l'));
        assert!(sm.advance('l'));
        assert!(sm.is_complete());
    }

    #[test]
    fn test_json_schema_enum() {
        let schema = JsonSchemaType::Enum(vec!["red".to_string(), "blue".to_string()]);
        let grammar = grammar_from_json_schema(&schema);

        let mut sm = GrammarStateMachine::new(grammar).expect("test");

        // Should accept '"' to start string
        assert!(sm.is_valid_char('"'));

        // Test "red"
        assert!(sm.advance('"'));
        assert!(sm.advance('r'));
        assert!(sm.advance('e'));
        assert!(sm.advance('d'));
        assert!(sm.advance('"'));
        assert!(sm.is_complete());
    }

    #[test]
    fn test_json_schema_object() {
        let schema = JsonSchemaType::Object(vec![
            ("name".to_string(), JsonSchemaType::String, true),
            ("age".to_string(), JsonSchemaType::Integer, true),
        ]);
        let grammar = grammar_from_json_schema(&schema);

        assert!(grammar.get_rule("root").is_some());
        assert!(grammar.get_rule("root_name").is_some());
        assert!(grammar.get_rule("root_age").is_some());
    }

    #[test]
    fn test_json_schema_array() {
        let schema = JsonSchemaType::Array(Box::new(JsonSchemaType::Integer));
        let grammar = grammar_from_json_schema(&schema);

        assert!(grammar.get_rule("root").is_some());
        assert!(grammar.get_rule("root_item").is_some());
        assert!(grammar.get_rule("root_items").is_some());
    }

    #[test]
    fn test_token_mask_allow_all() {
        let mask = TokenMask::allow_all(100);
        assert_eq!(mask.num_allowed(), 100);
        assert!(mask.is_allowed(0));
        assert!(mask.is_allowed(99));
        assert!(!mask.is_allowed(100));
        assert!(mask.allow_eos);
    }

    #[test]
    fn test_token_mask_from_allowed() {
        let allowed: HashSet<u32> = vec![1, 2, 3].into_iter().collect();
        let mask = TokenMask::from_allowed(allowed, false);

        assert!(mask.is_allowed(1));
        assert!(mask.is_allowed(2));
        assert!(mask.is_allowed(3));
        assert!(!mask.is_allowed(0));
        assert!(!mask.is_allowed(4));
        assert!(!mask.allow_eos);
    }

    #[test]
    fn test_token_mask_apply_to_logits() {
        let allowed: HashSet<u32> = vec![1, 3].into_iter().collect();
        let mask = TokenMask::from_allowed(allowed, true);

        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        mask.apply_to_logits(&mut logits);

        assert_eq!(logits[0], f32::NEG_INFINITY); // 0 not allowed
        assert_eq!(logits[1], 2.0); // 1 allowed
        assert_eq!(logits[2], f32::NEG_INFINITY); // 2 not allowed
        assert_eq!(logits[3], 4.0); // 3 allowed
        assert_eq!(logits[4], f32::NEG_INFINITY); // 4 not allowed
    }

    #[test]
    fn test_grammar_token_masker() {
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![GrammarElement::Char('a'), GrammarElement::Char('b')],
        ));

        let mut token_strings = HashMap::new();
        token_strings.insert(0, "a".to_string());
        token_strings.insert(1, "b".to_string());
        token_strings.insert(2, "c".to_string());
        token_strings.insert(3, "ab".to_string());

        let mut masker = GrammarTokenMasker::new(grammar, token_strings, 99).expect("test");

        // Initially, only 'a' and 'ab' should be allowed
        let mask = masker.get_mask();
        assert!(mask.is_allowed(0)); // 'a'
        assert!(!mask.is_allowed(1)); // 'b' - not valid first char
        assert!(!mask.is_allowed(2)); // 'c' - not valid
        assert!(mask.is_allowed(3)); // 'ab' - valid sequence

        // Advance with 'a'
        assert!(masker.advance_token(0));

        // Now only 'b' should be allowed
        let mask = masker.get_mask();
        assert!(!mask.is_allowed(0)); // 'a' - not valid after 'a'
        assert!(mask.is_allowed(1)); // 'b' - valid
        assert!(!mask.is_allowed(2)); // 'c' - not valid

        // Advance with 'b'
        assert!(masker.advance_token(1));
        assert!(masker.is_complete());
    }

    #[test]
    fn test_grammar_token_masker_reset() {
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('x')]));

        let mut token_strings = HashMap::new();
        token_strings.insert(0, "x".to_string());

        let mut masker = GrammarTokenMasker::new(grammar, token_strings, 99).expect("test");

        masker.advance_token(0);
        assert!(masker.is_complete());

        masker.reset();
        assert!(!masker.is_complete());
    }

    #[test]
    fn test_valid_chars_collection() {
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::new(
            "root",
            vec![
                GrammarAlternative::new(vec![GrammarElement::Char('a')]),
                GrammarAlternative::new(vec![GrammarElement::Char('b')]),
                GrammarAlternative::new(vec![GrammarElement::CharRange('0', '2')]),
            ],
        ));

        let sm = GrammarStateMachine::new(grammar).expect("test");
        let valid = sm.valid_chars();

        assert!(valid.contains(&'a'));
        assert!(valid.contains(&'b'));
        assert!(valid.contains(&'0'));
        assert!(valid.contains(&'1'));
        assert!(valid.contains(&'2'));
        assert!(!valid.contains(&'3'));
        assert!(!valid.contains(&'c'));
    }

    #[test]
    fn test_grammar_serialization() {
        let mut grammar = Grammar::with_root("test");
        grammar.add_rule(GrammarRule::single("test", vec![GrammarElement::Char('x')]));

        let json = serde_json::to_string(&grammar).expect("test");
        let deserialized: Grammar = serde_json::from_str(&json).expect("test");

        assert_eq!(deserialized.root(), "test");
        assert!(deserialized.get_rule("test").is_some());
    }

    // ==================== Tool Calling Tests ====================

    #[test]
    fn test_tool_parameter_type_default() {
        let param_type = ToolParameterType::default();
        assert_eq!(param_type, ToolParameterType::String);
    }

    #[test]
    fn test_tool_parameter_required_string() {
        let param = ToolParameter::required_string("query", "Search query");

        assert_eq!(param.name, "query");
        assert_eq!(param.description, "Search query");
        assert_eq!(param.param_type, ToolParameterType::String);
        assert!(param.required);
        assert!(param.default.is_none());
    }

    #[test]
    fn test_tool_parameter_optional_string() {
        let param = ToolParameter::optional_string("format", "Output format");

        assert_eq!(param.name, "format");
        assert_eq!(param.description, "Output format");
        assert_eq!(param.param_type, ToolParameterType::String);
        assert!(!param.required);
    }

    #[test]
    fn test_tool_parameter_required_int() {
        let param = ToolParameter::required_int("count", "Number of results");

        assert_eq!(param.name, "count");
        assert_eq!(param.param_type, ToolParameterType::Integer);
        assert!(param.required);
    }
