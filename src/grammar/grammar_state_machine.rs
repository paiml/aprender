
impl GrammarStateMachine {
    /// Create new state machine from grammar
    ///
    /// # Errors
    ///
    /// Returns an error if the grammar fails validation.
    pub fn new(grammar: Grammar) -> Result<Self> {
        grammar.validate()?;

        let initial = GrammarState::initial(grammar.root());
        let mut states = vec![initial];

        // Expand initial states for all alternatives of root rule
        if let Some(root_rule) = grammar.get_rule(grammar.root()) {
            states.clear();
            for (alt_idx, _) in root_rule.alternatives.iter().enumerate() {
                states.push(GrammarState {
                    rule: grammar.root().to_string(),
                    alt_idx,
                    elem_idx: 0,
                    stack: Vec::new(),
                });
            }
        }

        Ok(Self {
            grammar,
            states,
            generated: String::new(),
        })
    }

    /// Check if a character is valid at current state
    pub fn is_valid_char(&self, c: char) -> bool {
        for state in &self.states {
            if self.can_accept_char(state, c) {
                return true;
            }
        }
        false
    }

    /// Get all valid characters at current state
    pub fn valid_chars(&self) -> HashSet<char> {
        let mut valid = HashSet::new();

        for state in &self.states {
            self.collect_valid_chars(state, &mut valid);
        }

        valid
    }

    /// Advance state machine with a character
    pub fn advance(&mut self, c: char) -> bool {
        let mut new_states = Vec::new();

        for state in &self.states {
            if let Some(next_states) = self.advance_state(state, c) {
                new_states.extend(next_states);
            }
        }

        if new_states.is_empty() {
            return false;
        }

        self.states = new_states;
        self.generated.push(c);
        true
    }

    /// Check if generation is complete (valid end state)
    pub fn is_complete(&self) -> bool {
        self.states.iter().any(|s| s.is_complete(&self.grammar))
    }

    /// Check if any valid continuation exists
    pub fn has_valid_continuation(&self) -> bool {
        !self.states.is_empty()
    }

    /// Get generated string so far
    pub fn generated(&self) -> &str {
        &self.generated
    }

    /// Reset state machine
    pub fn reset(&mut self) {
        let initial = GrammarState::initial(self.grammar.root());
        self.states = vec![initial];

        // Expand for all alternatives
        if let Some(root_rule) = self.grammar.get_rule(self.grammar.root()) {
            self.states.clear();
            for (alt_idx, _) in root_rule.alternatives.iter().enumerate() {
                self.states.push(GrammarState {
                    rule: self.grammar.root().to_string(),
                    alt_idx,
                    elem_idx: 0,
                    stack: Vec::new(),
                });
            }
        }

        self.generated.clear();
    }

    // Internal: Check if state can accept character
    fn can_accept_char(&self, state: &GrammarState, c: char) -> bool {
        let Some(elem) = state.current_element(&self.grammar) else {
            return false;
        };
        match elem {
            GrammarElement::Char(expected) => c == *expected,
            GrammarElement::CharRange(start, end) => c >= *start && c <= *end,
            GrammarElement::CharNot(excluded) => !excluded.contains(&c),
            GrammarElement::Any => true,
            GrammarElement::RuleRef(rule_name) => self.any_alternative_accepts(rule_name, c),
            GrammarElement::End => false,
        }
    }

    // Check if any alternative of the referenced rule accepts character c
    fn any_alternative_accepts(&self, rule_name: &str, c: char) -> bool {
        let Some(rule) = self.grammar.get_rule(rule_name) else {
            return false;
        };
        rule.alternatives.iter().enumerate().any(|(alt_idx, _)| {
            let sub_state = GrammarState {
                rule: rule_name.to_string(),
                alt_idx,
                elem_idx: 0,
                stack: Vec::new(),
            };
            self.can_accept_char(&sub_state, c)
        })
    }

    // Internal: Collect valid characters for a state
    fn collect_valid_chars(&self, state: &GrammarState, valid: &mut HashSet<char>) {
        let Some(elem) = state.current_element(&self.grammar) else {
            return;
        };
        match elem {
            GrammarElement::Char(c) => {
                valid.insert(*c);
            },
            GrammarElement::CharRange(start, end) => {
                for c in *start..=*end {
                    valid.insert(c);
                }
            },
            GrammarElement::CharNot(_) => {
                // For negated sets, check each printable char against the exclusion list
                for c in ' '..='~' {
                    if self.can_accept_char(state, c) {
                        valid.insert(c);
                    }
                }
            },
            GrammarElement::Any => {
                valid.extend(' '..='~');
            },
            GrammarElement::RuleRef(rule_name) => {
                self.collect_chars_from_alternatives(rule_name, valid);
            },
            GrammarElement::End => {},
        }
    }

    // Recurse into all alternatives of a referenced rule to collect valid chars
    fn collect_chars_from_alternatives(&self, rule_name: &str, valid: &mut HashSet<char>) {
        let Some(rule) = self.grammar.get_rule(rule_name) else {
            return;
        };
        for (alt_idx, _) in rule.alternatives.iter().enumerate() {
            let sub_state = GrammarState {
                rule: rule_name.to_string(),
                alt_idx,
                elem_idx: 0,
                stack: Vec::new(),
            };
            self.collect_valid_chars(&sub_state, valid);
        }
    }

    // Internal: Advance state and return new states
    fn advance_state(&self, state: &GrammarState, c: char) -> Option<Vec<GrammarState>> {
        let elem = state.current_element(&self.grammar)?;

        match elem {
            GrammarElement::Char(expected) => {
                if c == *expected {
                    Some(vec![self.next_state(state)])
                } else {
                    None
                }
            },
            GrammarElement::CharRange(start, end) => {
                if c >= *start && c <= *end {
                    Some(vec![self.next_state(state)])
                } else {
                    None
                }
            },
            GrammarElement::CharNot(excluded) => {
                if !excluded.contains(&c) {
                    Some(vec![self.next_state(state)])
                } else {
                    None
                }
            },
            GrammarElement::Any => Some(vec![self.next_state(state)]),
            GrammarElement::RuleRef(rule_name) => {
                // Enter referenced rule
                let rule = self.grammar.get_rule(rule_name)?;
                let mut new_states = Vec::new();

                for (alt_idx, _) in rule.alternatives.iter().enumerate() {
                    let mut sub_state = GrammarState {
                        rule: rule_name.clone(),
                        alt_idx,
                        elem_idx: 0,
                        stack: state.stack.clone(),
                    };
                    // Push return address
                    sub_state
                        .stack
                        .push((state.rule.clone(), state.alt_idx, state.elem_idx + 1));

                    if let Some(advanced) = self.advance_state(&sub_state, c) {
                        new_states.extend(advanced);
                    }
                }

                if new_states.is_empty() {
                    None
                } else {
                    Some(new_states)
                }
            },
            GrammarElement::End => None,
        }
    }

    // Internal: Create next state after consuming element
    fn next_state(&self, state: &GrammarState) -> GrammarState {
        let mut new_state = state.clone();
        new_state.elem_idx += 1;

        // Check if we've completed current alternative
        if let Some(rule) = self.grammar.get_rule(&state.rule) {
            if let Some(alt) = rule.alternatives.get(state.alt_idx) {
                if new_state.elem_idx >= alt.elements.len() {
                    // Pop from stack if there's a return address
                    if let Some((ret_rule, ret_alt, ret_elem)) = new_state.stack.pop() {
                        new_state.rule = ret_rule;
                        new_state.alt_idx = ret_alt;
                        new_state.elem_idx = ret_elem;
                    }
                }
            }
        }

        new_state
    }
}

// =============================================================================
// JSON SCHEMA GRAMMAR BUILDER
// =============================================================================

/// JSON Schema types for grammar generation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum JsonSchemaType {
    /// String type
    String,
    /// Integer type
    Integer,
    /// Number type (float)
    Number,
    /// Boolean type
    Boolean,
    /// Null type
    Null,
    /// Array type with item schema
    Array(Box<JsonSchemaType>),
    /// Object type with properties
    Object(Vec<(String, JsonSchemaType, bool)>), // (name, type, required)
    /// Enum with allowed values
    Enum(Vec<String>),
    /// Any type
    Any,
}

/// Build grammar from JSON schema type
pub fn grammar_from_json_schema(schema: &JsonSchemaType) -> Grammar {
    let mut grammar = Grammar::with_root("root");

    // Add whitespace rule
    grammar.add_rule(GrammarRule::new(
        "ws",
        vec![
            GrammarAlternative::new(vec![]), // Empty (epsilon)
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

    // Add digit rule
    grammar.add_rule(GrammarRule::new(
        "digit",
        vec![GrammarAlternative::new(vec![GrammarElement::CharRange(
            '0', '9',
        )])],
    ));

    // Add digits rule (one or more)
    grammar.add_rule(GrammarRule::new(
        "digits",
        vec![
            GrammarAlternative::new(vec![GrammarElement::RuleRef("digit".to_string())]),
            GrammarAlternative::new(vec![
                GrammarElement::RuleRef("digit".to_string()),
                GrammarElement::RuleRef("digits".to_string()),
            ]),
        ],
    ));

    // Add string character rule
    grammar.add_rule(GrammarRule::new(
        "string_char",
        vec![
            GrammarAlternative::new(vec![GrammarElement::CharNot(vec!['"', '\\', '\n'])]),
            GrammarAlternative::new(vec![GrammarElement::Char('\\'), GrammarElement::Char('"')]),
            GrammarAlternative::new(vec![GrammarElement::Char('\\'), GrammarElement::Char('\\')]),
            GrammarAlternative::new(vec![GrammarElement::Char('\\'), GrammarElement::Char('n')]),
        ],
    ));

    // Add string content rule
    grammar.add_rule(GrammarRule::new(
        "string_content",
        vec![
            GrammarAlternative::new(vec![]), // Empty
            GrammarAlternative::new(vec![
                GrammarElement::RuleRef("string_char".to_string()),
                GrammarElement::RuleRef("string_content".to_string()),
            ]),
        ],
    ));

    // Add base type rules based on schema
    add_schema_rules(&mut grammar, "root", schema);

    grammar
}

/// Add array schema rules
fn add_array_schema_rule(grammar: &mut Grammar, rule_name: &str, item_schema: &JsonSchemaType) {
    let item_rule = format!("{rule_name}_item");
    add_schema_rules(grammar, &item_rule, item_schema);

    let items_rule = format!("{rule_name}_items");
    grammar.add_rule(GrammarRule::new(
        &items_rule,
        vec![
            GrammarAlternative::new(vec![]),
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
}

/// Add "any JSON value" schema rules
fn add_any_schema_rule(grammar: &mut Grammar, rule_name: &str) {
    grammar.add_rule(GrammarRule::new(
        rule_name,
        vec![
            GrammarAlternative::new(vec![GrammarElement::RuleRef("string_value".to_string())]),
            GrammarAlternative::new(vec![GrammarElement::RuleRef("number".to_string())]),
            GrammarAlternative::new(vec![GrammarElement::RuleRef("boolean".to_string())]),
            GrammarAlternative::new(vec![GrammarElement::RuleRef("null".to_string())]),
        ],
    ));

    if grammar.get_rule("string_value").is_none() {
        add_schema_rules(grammar, "string_value", &JsonSchemaType::String);
    }
    if grammar.get_rule("number").is_none() {
        add_schema_rules(grammar, "number", &JsonSchemaType::Number);
    }
    if grammar.get_rule("boolean").is_none() {
        add_schema_rules(grammar, "boolean", &JsonSchemaType::Boolean);
    }
    if grammar.get_rule("null").is_none() {
        add_schema_rules(grammar, "null", &JsonSchemaType::Null);
    }
}
