
/// Generate grammar for tool parameters
fn generate_params_grammar(grammar: &mut Grammar, rule_name: &str, params: &[ToolParameter]) {
    if params.is_empty() {
        // Empty object: {}
        grammar.add_rule(GrammarRule::new(
            rule_name,
            vec![GrammarAlternative::new(vec![
                GrammarElement::Char('{'),
                GrammarElement::RuleRef("ws".to_string()),
                GrammarElement::Char('}'),
            ])],
        ));
        return;
    }

    // Build object with all parameters
    let mut elements = vec![
        GrammarElement::Char('{'),
        GrammarElement::RuleRef("ws".to_string()),
    ];

    for (i, param) in params.iter().enumerate() {
        if i > 0 {
            elements.push(GrammarElement::Char(','));
            elements.push(GrammarElement::RuleRef("ws".to_string()));
        }

        // Property name
        elements.push(GrammarElement::Char('"'));
        for c in param.name.chars() {
            elements.push(GrammarElement::Char(c));
        }
        elements.push(GrammarElement::Char('"'));
        elements.push(GrammarElement::RuleRef("ws".to_string()));
        elements.push(GrammarElement::Char(':'));
        elements.push(GrammarElement::RuleRef("ws".to_string()));

        // Property value based on type
        let value_rule = format!("{rule_name}_{}", param.name);
        generate_param_type_grammar(grammar, &value_rule, &param.param_type);
        elements.push(GrammarElement::RuleRef(value_rule));
    }

    elements.push(GrammarElement::RuleRef("ws".to_string()));
    elements.push(GrammarElement::Char('}'));

    grammar.add_rule(GrammarRule::new(
        rule_name,
        vec![GrammarAlternative::new(elements)],
    ));
}

/// Generate grammar for a parameter type
fn generate_param_type_grammar(
    grammar: &mut Grammar,
    rule_name: &str,
    param_type: &ToolParameterType,
) {
    match param_type {
        ToolParameterType::String => {
            grammar.add_rule(GrammarRule::new(
                rule_name,
                vec![GrammarAlternative::new(vec![GrammarElement::RuleRef(
                    "string".to_string(),
                )])],
            ));
        },
        ToolParameterType::Integer => {
            grammar.add_rule(GrammarRule::new(
                rule_name,
                vec![GrammarAlternative::new(vec![GrammarElement::RuleRef(
                    "integer".to_string(),
                )])],
            ));
        },
        ToolParameterType::Number => {
            grammar.add_rule(GrammarRule::new(
                rule_name,
                vec![GrammarAlternative::new(vec![GrammarElement::RuleRef(
                    "number".to_string(),
                )])],
            ));
        },
        ToolParameterType::Boolean => {
            grammar.add_rule(GrammarRule::new(
                rule_name,
                vec![GrammarAlternative::new(vec![GrammarElement::RuleRef(
                    "boolean".to_string(),
                )])],
            ));
        },
        ToolParameterType::Enum(values) => {
            let alternatives: Vec<_> = values
                .iter()
                .map(|v| {
                    let mut chars = vec![GrammarElement::Char('"')];
                    chars.extend(v.chars().map(GrammarElement::Char));
                    chars.push(GrammarElement::Char('"'));
                    GrammarAlternative::new(chars)
                })
                .collect();
            grammar.add_rule(GrammarRule::new(rule_name, alternatives));
        },
        ToolParameterType::Array { items } => {
            let item_rule = format!("{rule_name}_item");
            generate_param_type_grammar(grammar, &item_rule, items);

            let items_rule = format!("{rule_name}_items");
            grammar.add_rule(GrammarRule::new(
                &items_rule,
                vec![
                    GrammarAlternative::new(vec![]), // Empty
                    GrammarAlternative::new(vec![
                        GrammarElement::Char(','),
                        GrammarElement::RuleRef("ws".to_string()),
                        GrammarElement::RuleRef(item_rule.clone()),
                        GrammarElement::RuleRef(items_rule.clone()),
                    ]),
                ],
            ));

            grammar.add_rule(GrammarRule::new(
                rule_name,
                vec![
                    GrammarAlternative::new(vec![
                        GrammarElement::Char('['),
                        GrammarElement::RuleRef("ws".to_string()),
                        GrammarElement::Char(']'),
                    ]),
                    GrammarAlternative::new(vec![
                        GrammarElement::Char('['),
                        GrammarElement::RuleRef("ws".to_string()),
                        GrammarElement::RuleRef(item_rule),
                        GrammarElement::RuleRef(items_rule),
                        GrammarElement::RuleRef("ws".to_string()),
                        GrammarElement::Char(']'),
                    ]),
                ],
            ));
        },
        ToolParameterType::Object { properties } => {
            generate_params_grammar(grammar, rule_name, properties);
        },
    }
}

/// Add standard JSON whitespace rules to grammar
fn add_json_whitespace_rules(grammar: &mut Grammar) {
    // Whitespace (optional)
    grammar.add_rule(GrammarRule::new(
        "ws",
        vec![
            GrammarAlternative::new(vec![]),
            GrammarAlternative::new(vec![
                GrammarElement::Char(' '),
                GrammarElement::RuleRef("ws".to_string()),
            ]),
            GrammarAlternative::new(vec![
                GrammarElement::Char('\n'),
                GrammarElement::RuleRef("ws".to_string()),
            ]),
            GrammarAlternative::new(vec![
                GrammarElement::Char('\t'),
                GrammarElement::RuleRef("ws".to_string()),
            ]),
        ],
    ));

    // String (simplified - no escape handling)
    grammar.add_rule(GrammarRule::new(
        "string",
        vec![GrammarAlternative::new(vec![
            GrammarElement::Char('"'),
            GrammarElement::RuleRef("string_chars".to_string()),
            GrammarElement::Char('"'),
        ])],
    ));

    grammar.add_rule(GrammarRule::new(
        "string_chars",
        vec![
            GrammarAlternative::new(vec![]),
            GrammarAlternative::new(vec![
                GrammarElement::CharNot(vec!['"', '\\']),
                GrammarElement::RuleRef("string_chars".to_string()),
            ]),
        ],
    ));

    // Integer
    grammar.add_rule(GrammarRule::new(
        "integer",
        vec![
            GrammarAlternative::new(vec![
                GrammarElement::Char('-'),
                GrammarElement::RuleRef("digits".to_string()),
            ]),
            GrammarAlternative::new(vec![GrammarElement::RuleRef("digits".to_string())]),
        ],
    ));

    // Number (with optional decimal)
    grammar.add_rule(GrammarRule::new(
        "number",
        vec![
            GrammarAlternative::new(vec![
                GrammarElement::RuleRef("integer".to_string()),
                GrammarElement::Char('.'),
                GrammarElement::RuleRef("digits".to_string()),
            ]),
            GrammarAlternative::new(vec![GrammarElement::RuleRef("integer".to_string())]),
        ],
    ));

    // Digits
    grammar.add_rule(GrammarRule::new(
        "digits",
        vec![
            GrammarAlternative::new(vec![
                GrammarElement::CharRange('0', '9'),
                GrammarElement::RuleRef("digits".to_string()),
            ]),
            GrammarAlternative::new(vec![GrammarElement::CharRange('0', '9')]),
        ],
    ));

    // Boolean
    grammar.add_rule(GrammarRule::new(
        "boolean",
        vec![
            GrammarAlternative::new(vec![
                GrammarElement::Char('t'),
                GrammarElement::Char('r'),
                GrammarElement::Char('u'),
                GrammarElement::Char('e'),
            ]),
            GrammarAlternative::new(vec![
                GrammarElement::Char('f'),
                GrammarElement::Char('a'),
                GrammarElement::Char('l'),
                GrammarElement::Char('s'),
                GrammarElement::Char('e'),
            ]),
        ],
    ));
}

// =============================================================================
// TESTS
// =============================================================================

// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod grammar_tests;
