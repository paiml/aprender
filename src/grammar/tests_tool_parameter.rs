
    #[test]
    fn test_tool_parameter_required_enum() {
        let param = ToolParameter::required_enum(
            "format",
            "Output format",
            vec!["json".to_string(), "xml".to_string()],
        );

        match &param.param_type {
            ToolParameterType::Enum(values) => {
                assert_eq!(values.len(), 2);
                assert!(values.contains(&"json".to_string()));
                assert!(values.contains(&"xml".to_string()));
            },
            _ => panic!("Expected Enum type"),
        }
        assert!(param.required);
    }

    #[test]
    fn test_tool_parameter_with_default() {
        let param = ToolParameter::optional_string("format", "Output format").with_default("json");

        assert_eq!(param.default.as_deref(), Some("json"));
    }

    #[test]
    fn test_tool_parameter_type_array() {
        let array_type = ToolParameterType::Array {
            items: Box::new(ToolParameterType::Integer),
        };

        match &array_type {
            ToolParameterType::Array { items } => {
                assert_eq!(**items, ToolParameterType::Integer);
            },
            _ => panic!("Expected Array type"),
        }
    }

    #[test]
    fn test_tool_parameter_type_nested_object() {
        let inner_params = vec![
            ToolParameter::required_string("street", "Street address"),
            ToolParameter::required_string("city", "City name"),
        ];

        let object_type = ToolParameterType::Object {
            properties: inner_params,
        };

        match &object_type {
            ToolParameterType::Object { properties } => {
                assert_eq!(properties.len(), 2);
                assert_eq!(properties[0].name, "street");
                assert_eq!(properties[1].name, "city");
            },
            _ => panic!("Expected Object type"),
        }
    }

    #[test]
    fn test_tool_definition_creation() {
        let tool = ToolDefinition::new(
            "get_weather",
            "Get current weather for a location",
            vec![
                ToolParameter::required_string("location", "City name"),
                ToolParameter::optional_string("unit", "Temperature unit"),
            ],
        );

        assert_eq!(tool.name, "get_weather");
        assert_eq!(tool.description, "Get current weather for a location");
        assert_eq!(tool.parameters.len(), 2);
        assert!(tool.parameters[0].required);
        assert!(!tool.parameters[1].required);
    }

    #[test]
    fn test_tool_definition_required_params() {
        let tool = ToolDefinition::new(
            "test",
            "Test tool",
            vec![
                ToolParameter::required_string("req1", "Required 1"),
                ToolParameter::optional_string("opt1", "Optional 1"),
                ToolParameter::required_string("req2", "Required 2"),
            ],
        );

        let required: Vec<_> = tool.required_params().collect();
        assert_eq!(required.len(), 2);
        assert_eq!(required[0].name, "req1");
        assert_eq!(required[1].name, "req2");
    }

    #[test]
    fn test_tool_definition_optional_params() {
        let tool = ToolDefinition::new(
            "test",
            "Test tool",
            vec![
                ToolParameter::required_string("req1", "Required 1"),
                ToolParameter::optional_string("opt1", "Optional 1"),
                ToolParameter::optional_string("opt2", "Optional 2"),
            ],
        );

        let optional: Vec<_> = tool.optional_params().collect();
        assert_eq!(optional.len(), 2);
        assert_eq!(optional[0].name, "opt1");
        assert_eq!(optional[1].name, "opt2");
    }

    #[test]
    fn test_tool_definition_is_valid_name() {
        // Valid names
        assert!(ToolDefinition::is_valid_name("get_weather"));
        assert!(ToolDefinition::is_valid_name("search"));
        assert!(ToolDefinition::is_valid_name("_private"));
        assert!(ToolDefinition::is_valid_name("tool123"));
        assert!(ToolDefinition::is_valid_name("GetWeather"));

        // Invalid names
        assert!(!ToolDefinition::is_valid_name(""));
        assert!(!ToolDefinition::is_valid_name("invalid name"));
        assert!(!ToolDefinition::is_valid_name("123tool"));
        assert!(!ToolDefinition::is_valid_name("tool!"));
        assert!(!ToolDefinition::is_valid_name("tool-name"));
    }

    #[test]
    fn test_tool_choice_default() {
        let choice = ToolChoice::default();
        assert_eq!(choice, ToolChoice::Auto);
    }

    #[test]
    fn test_tool_choice_variants() {
        let auto = ToolChoice::Auto;
        let required = ToolChoice::Required;
        let none = ToolChoice::None;
        let specific = ToolChoice::Specific("my_tool".to_string());

        assert_eq!(auto, ToolChoice::Auto);
        assert_eq!(required, ToolChoice::Required);
        assert_eq!(none, ToolChoice::None);
        assert_eq!(specific, ToolChoice::Specific("my_tool".to_string()));
    }

    #[test]
    fn test_tool_call_creation() {
        let call = ToolCall::new("call_1", "get_weather", r#"{"location": "NYC"}"#);

        assert_eq!(call.id, "call_1");
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.arguments, r#"{"location": "NYC"}"#);
    }

    #[test]
    fn test_tool_call_parse_arguments() {
        let call = ToolCall::new(
            "call_1",
            "get_weather",
            r#"{"location": "NYC", "unit": "celsius"}"#,
        );

        let args = call.parse_arguments().expect("test");
        assert_eq!(args["location"], "NYC");
        assert_eq!(args["unit"], "celsius");
    }

    #[test]
    fn test_tool_call_parse_arguments_invalid() {
        let call = ToolCall::new("call_1", "get_weather", "not valid json");

        let result = call.parse_arguments();
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success("call_1", r#"{"temp": 72}"#);

        assert_eq!(result.tool_call_id, "call_1");
        assert_eq!(result.content, r#"{"temp": 72}"#);
        assert!(result.success);
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("call_1", "API rate limited");

        assert_eq!(result.tool_call_id, "call_1");
        assert_eq!(result.content, "API rate limited");
        assert!(!result.success);
    }

    #[test]
    fn test_tool_call_parser_creation() {
        let tools = vec![
            ToolDefinition::new("get_weather", "Get weather", vec![]),
            ToolDefinition::new("search", "Search web", vec![]),
        ];

        let parser = ToolCallParser::new(tools);
        let names: Vec<_> = parser.tool_names().collect();

        assert_eq!(names.len(), 2);
        assert!(names.contains(&"get_weather"));
        assert!(names.contains(&"search"));
    }

    #[test]
    fn test_tool_call_parser_generate_id() {
        let tools = vec![];
        let mut parser = ToolCallParser::new(tools);

        assert_eq!(parser.generate_id(), "call_0");
        assert_eq!(parser.generate_id(), "call_1");
        assert_eq!(parser.generate_id(), "call_2");
    }

    #[test]
    fn test_tool_call_format_default() {
        let format = ToolCallFormat::default();
        assert_eq!(format, ToolCallFormat::OpenAI);
    }

    #[test]
    fn test_tool_call_parser_with_format() {
        let tools = vec![];
        let parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Anthropic);

        // The format change is internal, but we can verify it was created
        assert_eq!(parser.tool_names().count(), 0);
    }

    #[test]
    fn test_parse_openai_format() {
        let tools = vec![ToolDefinition::new("get_weather", "Get weather", vec![])];

        let mut parser = ToolCallParser::new(tools);
        let text = r#"I'll get the weather for you.
{"name": "get_weather", "arguments": {"location": "New York"}}
Here's the weather."#;

        let calls = parser.parse(text);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].arguments).expect("test");
        assert_eq!(args["location"], "New York");
    }

    #[test]
    fn test_parse_openai_format_string_arguments() {
        let tools = vec![ToolDefinition::new("search", "Search", vec![])];

        let mut parser = ToolCallParser::new(tools);
        let text = r#"{"name": "search", "arguments": "{\"query\": \"rust programming\"}"}"#;

        let calls = parser.parse(text);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let tools = vec![
            ToolDefinition::new("get_weather", "Get weather", vec![]),
            ToolDefinition::new("search", "Search", vec![]),
        ];

        let mut parser = ToolCallParser::new(tools);
        let text = r#"
{"name": "get_weather", "arguments": {"location": "NYC"}}
Some text
{"name": "search", "arguments": {"query": "restaurants"}}
"#;

        let calls = parser.parse(text);

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[1].name, "search");
    }

    #[test]
    fn test_parse_unknown_tool_ignored() {
        let tools = vec![ToolDefinition::new("get_weather", "Get weather", vec![])];

        let mut parser = ToolCallParser::new(tools);
        let text = r#"{"name": "unknown_tool", "arguments": {}}"#;

        let calls = parser.parse(text);

        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn test_parse_anthropic_format() {
        let tools = vec![ToolDefinition::new("get_weather", "Get weather", vec![])];

        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Anthropic);
        let text = r#"<tool_use>
<name>get_weather</name>
<input>{"location": "Paris"}</input>
</tool_use>"#;

        let calls = parser.parse(text);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].arguments).expect("test");
        assert_eq!(args["location"], "Paris");
    }

    #[test]
    fn test_parse_hermes_format() {
        let tools = vec![ToolDefinition::new("search", "Search web", vec![])];

        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Hermes);
        let text = r#"<tool_call>
{"name": "search", "arguments": {"query": "Rust tutorials"}}
</tool_call>"#;

        let calls = parser.parse(text);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
    }

    #[test]
    fn test_find_matching_brace_simple() {
        let text = r#"{"key": "value"}"#;
        let result = find_matching_brace(text);

        assert_eq!(result, Some(15)); // Position of closing brace
    }

    #[test]
    fn test_find_matching_brace_nested() {
        let text = r#"{"outer": {"inner": "value"}}"#;
        let result = find_matching_brace(text);

        assert_eq!(result, Some(28));
    }

    #[test]
    fn test_find_matching_brace_with_string_braces() {
        let text = r#"{"text": "has { and } inside"}"#;
        let result = find_matching_brace(text);

        assert_eq!(result, Some(29));
    }

    #[test]
    fn test_find_matching_brace_unmatched() {
        let text = r#"{"key": "value""#;
        let result = find_matching_brace(text);

        assert_eq!(result, None);
    }

    #[test]
    fn test_find_matching_brace_not_starting_with_brace() {
        let text = "key: value";
        let result = find_matching_brace(text);

        assert_eq!(result, None);
    }

    #[test]
    fn test_extract_xml_tag() {
        let text = "<name>get_weather</name>";
        let result = extract_xml_tag(text, "name");

        assert_eq!(result, Some("get_weather".to_string()));
    }

    #[test]
    fn test_extract_xml_tag_with_whitespace() {
        let text = r#"<input>
{"location": "NYC"}
</input>"#;
        let result = extract_xml_tag(text, "input");

        // Result includes surrounding newlines - trim for comparison
        assert!(result.is_some());
        let trimmed = result.expect("test").trim().to_string();
        assert_eq!(trimmed, r#"{"location": "NYC"}"#);
    }

    #[test]
    fn test_extract_xml_tag_not_found() {
        let text = "<name>test</name>";
        let result = extract_xml_tag(text, "other");

        assert_eq!(result, None);
    }

    #[test]
    fn test_generate_tool_grammar_single_tool() {
        let tools = vec![ToolDefinition::new(
            "get_weather",
            "Get weather",
            vec![ToolParameter::required_string("location", "City name")],
        )];

        let grammar = generate_tool_grammar(&tools);

        // Grammar must have a root rule
        assert!(grammar.get_rule("root").is_some());
        // Grammar should have rules (at least one per tool or base rules)
        assert!(grammar.rule_names().count() > 0);
    }

    #[test]
    fn test_generate_tool_grammar_multiple_tools() {
        let tools = vec![
            ToolDefinition::new("get_weather", "Get weather", vec![]),
            ToolDefinition::new("search", "Search web", vec![]),
        ];

        let grammar = generate_tool_grammar(&tools);

        // Should have rules for both tools
        assert!(grammar.get_rule("root").is_some());
    }

    #[test]
    fn test_generate_tool_grammar_empty_tools() {
        let tools: Vec<ToolDefinition> = vec![];
        let grammar = generate_tool_grammar(&tools);

        // Should still create a valid grammar
        assert!(grammar.get_rule("root").is_some());
    }
