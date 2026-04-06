
fn add_schema_rules(grammar: &mut Grammar, rule_name: &str, schema: &JsonSchemaType) {
    match schema {
        JsonSchemaType::String => {
            grammar.add_rule(GrammarRule::new(
                rule_name,
                vec![GrammarAlternative::new(vec![
                    GrammarElement::Char('"'),
                    GrammarElement::RuleRef("string_content".to_string()),
                    GrammarElement::Char('"'),
                ])],
            ));
        },
        JsonSchemaType::Integer => {
            grammar.add_rule(GrammarRule::new(
                rule_name,
                vec![
                    GrammarAlternative::new(vec![GrammarElement::RuleRef("digits".to_string())]),
                    GrammarAlternative::new(vec![
                        GrammarElement::Char('-'),
                        GrammarElement::RuleRef("digits".to_string()),
                    ]),
                ],
            ));
        },
        JsonSchemaType::Number => {
            grammar.add_rule(GrammarRule::new(
                rule_name,
                vec![
                    GrammarAlternative::new(vec![GrammarElement::RuleRef("digits".to_string())]),
                    GrammarAlternative::new(vec![
                        GrammarElement::Char('-'),
                        GrammarElement::RuleRef("digits".to_string()),
                    ]),
                    GrammarAlternative::new(vec![
                        GrammarElement::RuleRef("digits".to_string()),
                        GrammarElement::Char('.'),
                        GrammarElement::RuleRef("digits".to_string()),
                    ]),
                    GrammarAlternative::new(vec![
                        GrammarElement::Char('-'),
                        GrammarElement::RuleRef("digits".to_string()),
                        GrammarElement::Char('.'),
                        GrammarElement::RuleRef("digits".to_string()),
                    ]),
                ],
            ));
        },
        JsonSchemaType::Boolean => {
            grammar.add_rule(GrammarRule::new(
                rule_name,
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
        },
        JsonSchemaType::Null => {
            grammar.add_rule(GrammarRule::new(
                rule_name,
                vec![GrammarAlternative::new(vec![
                    GrammarElement::Char('n'),
                    GrammarElement::Char('u'),
                    GrammarElement::Char('l'),
                    GrammarElement::Char('l'),
                ])],
            ));
        },
        JsonSchemaType::Enum(values) => {
            let alternatives: Vec<GrammarAlternative> = values
                .iter()
                .map(|v| {
                    let mut elements = vec![GrammarElement::Char('"')];
                    for c in v.chars() {
                        elements.push(GrammarElement::Char(c));
                    }
                    elements.push(GrammarElement::Char('"'));
                    GrammarAlternative::new(elements)
                })
                .collect();
            grammar.add_rule(GrammarRule::new(rule_name, alternatives));
        },
        JsonSchemaType::Array(item_schema) => {
            add_array_schema_rule(grammar, rule_name, item_schema);
        },
        JsonSchemaType::Object(properties) => {
            add_object_schema_rules(grammar, rule_name, properties);
        },
        JsonSchemaType::Any => {
            add_any_schema_rule(grammar, rule_name);
        },
    }
}

/// Build grammar rules for a JSON object schema with the given properties
fn add_object_schema_rules(
    grammar: &mut Grammar,
    rule_name: &str,
    properties: &[(String, JsonSchemaType, bool)],
) {
    if properties.is_empty() {
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

    let mut elements = vec![
        GrammarElement::Char('{'),
        GrammarElement::RuleRef("ws".to_string()),
    ];

    for (i, (prop_name, prop_type, _required)) in properties.iter().enumerate() {
        if i > 0 {
            elements.push(GrammarElement::Char(','));
            elements.push(GrammarElement::RuleRef("ws".to_string()));
        }

        // Property name
        elements.push(GrammarElement::Char('"'));
        for c in prop_name.chars() {
            elements.push(GrammarElement::Char(c));
        }
        elements.push(GrammarElement::Char('"'));
        elements.push(GrammarElement::RuleRef("ws".to_string()));
        elements.push(GrammarElement::Char(':'));
        elements.push(GrammarElement::RuleRef("ws".to_string()));

        // Property value
        let prop_rule = format!("{rule_name}_{prop_name}");
        add_schema_rules(grammar, &prop_rule, prop_type);
        elements.push(GrammarElement::RuleRef(prop_rule));
    }

    elements.push(GrammarElement::RuleRef("ws".to_string()));
    elements.push(GrammarElement::Char('}'));

    grammar.add_rule(GrammarRule::new(
        rule_name,
        vec![GrammarAlternative::new(elements)],
    ));
}

// =============================================================================
// TOKEN MASKING FOR CONSTRAINED GENERATION
// =============================================================================

/// Token mask for constrained generation
#[derive(Debug, Clone)]
pub struct TokenMask {
    /// Allowed token IDs
    pub allowed: HashSet<u32>,
    /// Whether to allow end-of-sequence
    pub allow_eos: bool,
}

impl TokenMask {
    /// Create mask allowing all tokens
    pub fn allow_all(vocab_size: usize) -> Self {
        Self {
            allowed: (0..vocab_size as u32).collect(),
            allow_eos: true,
        }
    }

    /// Create mask from allowed set
    pub fn from_allowed(allowed: HashSet<u32>, allow_eos: bool) -> Self {
        Self { allowed, allow_eos }
    }

    /// Check if token is allowed
    pub fn is_allowed(&self, token_id: u32) -> bool {
        self.allowed.contains(&token_id)
    }

    /// Apply mask to logits (set disallowed to -inf)
    pub fn apply_to_logits(&self, logits: &mut [f32]) {
        for (i, logit) in logits.iter_mut().enumerate() {
            if !self.allowed.contains(&(i as u32)) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    /// Number of allowed tokens
    pub fn num_allowed(&self) -> usize {
        self.allowed.len()
    }
}

/// Grammar-based token masker
pub struct GrammarTokenMasker {
    /// State machine tracking grammar state
    state_machine: GrammarStateMachine,
    /// Token to string mapping
    token_strings: HashMap<u32, String>,
    /// EOS token ID
    eos_token_id: u32,
}

impl GrammarTokenMasker {
    /// Create new masker from grammar and vocabulary
    ///
    /// # Errors
    ///
    /// Returns an error if the grammar fails validation.
    pub fn new(
        grammar: Grammar,
        token_strings: HashMap<u32, String>,
        eos_token_id: u32,
    ) -> Result<Self> {
        let state_machine = GrammarStateMachine::new(grammar)?;
        Ok(Self {
            state_machine,
            token_strings,
            eos_token_id,
        })
    }

    /// Check if all characters in a multi-char token form a valid sequence
    fn is_token_valid_sequence(&self, token_str: &str) -> bool {
        let mut temp_sm = self.state_machine.clone();
        token_str.chars().all(|c| temp_sm.advance(c))
    }

    /// Get mask for current state
    pub fn get_mask(&self) -> TokenMask {
        let valid_chars = self.state_machine.valid_chars();
        let mut allowed = HashSet::new();

        for (token_id, token_str) in &self.token_strings {
            if let Some(first_char) = token_str.chars().next() {
                if valid_chars.contains(&first_char)
                    && (token_str.len() == 1 || self.is_token_valid_sequence(token_str))
                {
                    allowed.insert(*token_id);
                }
            }
        }

        TokenMask::from_allowed(allowed, self.state_machine.is_complete())
    }

    /// Advance masker with selected token
    pub fn advance_token(&mut self, token_id: u32) -> bool {
        if let Some(token_str) = self.token_strings.get(&token_id) {
            for c in token_str.chars() {
                if !self.state_machine.advance(c) {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }

    /// Check if generation is complete
    pub fn is_complete(&self) -> bool {
        self.state_machine.is_complete()
    }

    /// Reset masker state
    pub fn reset(&mut self) {
        self.state_machine.reset();
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
}

// =============================================================================
// TOOL CALLING / FUNCTION CALLING
// =============================================================================
//
// Implements OpenAI-style tool/function calling for LLM inference.
// Allows models to generate structured function calls that can be executed
// and results fed back into the conversation.
//
// Reference: OpenAI Function Calling API
// - Tool definitions with JSON Schema parameters
// - Tool choice: auto, required, none, or specific tool
// - Tool call parsing from model output
// - Grammar generation for constrained tool output

/// JSON Schema property type for tool parameters
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolParameterType {
    /// String parameter
    #[default]
    String,
    /// Integer parameter
    Integer,
    /// Number parameter (float)
    Number,
    /// Boolean parameter
    Boolean,
    /// Array parameter
    Array {
        /// Type of array items
        items: Box<ToolParameterType>,
    },
    /// Object parameter with properties
    Object {
        /// Properties of the object
        properties: Vec<ToolParameter>,
    },
    /// Enum of allowed string values
    Enum(Vec<String>),
}

/// Tool parameter definition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolParameter {
    /// Parameter name
    pub name: String,
    /// Parameter description
    pub description: String,
    /// Parameter type
    #[serde(rename = "type")]
    pub param_type: ToolParameterType,
    /// Whether the parameter is required
    #[serde(default)]
    pub required: bool,
    /// Default value (JSON string)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
}

impl ToolParameter {
    /// Create new required string parameter
    pub fn required_string(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            param_type: ToolParameterType::String,
            required: true,
            default: None,
        }
    }

    /// Create new optional string parameter
    pub fn optional_string(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            param_type: ToolParameterType::String,
            required: false,
            default: None,
        }
    }

    /// Create new required integer parameter
    pub fn required_int(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            param_type: ToolParameterType::Integer,
            required: true,
            default: None,
        }
    }

    /// Create new enum parameter
    pub fn required_enum(
        name: impl Into<String>,
        description: impl Into<String>,
        values: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            param_type: ToolParameterType::Enum(values),
            required: true,
            default: None,
        }
    }

    /// Set default value
    #[must_use]
    pub fn with_default(mut self, default: impl Into<String>) -> Self {
        self.default = Some(default.into());
        self
    }
}

/// Tool/function definition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name (must be valid identifier: [a-zA-Z_][a-zA-Z0-9_]*)
    pub name: String,
    /// Tool description
    pub description: String,
    /// Tool parameters
    pub parameters: Vec<ToolParameter>,
}
