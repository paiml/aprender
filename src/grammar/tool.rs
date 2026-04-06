
impl ToolDefinition {
    /// Create new tool definition
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Vec<ToolParameter>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    /// Get required parameters
    pub fn required_params(&self) -> impl Iterator<Item = &ToolParameter> {
        self.parameters.iter().filter(|p| p.required)
    }

    /// Get optional parameters
    pub fn optional_params(&self) -> impl Iterator<Item = &ToolParameter> {
        self.parameters.iter().filter(|p| !p.required)
    }

    /// Validate tool name is a valid identifier
    pub fn is_valid_name(name: &str) -> bool {
        if name.is_empty() {
            return false;
        }
        let mut chars = name.chars();
        // SAFETY: name is non-empty (checked above)
        let first = chars.next().expect("name is non-empty");
        if !first.is_ascii_alphabetic() && first != '_' {
            return false;
        }
        chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    }
}

/// Tool choice mode
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// Model decides whether to call a tool
    #[default]
    Auto,
    /// Model must call at least one tool
    Required,
    /// Model must not call any tools
    None,
    /// Model must call the specified tool
    Specific(String),
}

/// Parsed tool call from model output
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique ID for this tool call
    pub id: String,
    /// Tool name
    pub name: String,
    /// Tool arguments as JSON string
    pub arguments: String,
}

impl ToolCall {
    /// Create new tool call
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    /// Parse arguments as JSON value
    ///
    /// # Errors
    ///
    /// Returns `RealizarError::InvalidConfiguration` if the arguments are not valid JSON.
    pub fn parse_arguments(&self) -> Result<serde_json::Value> {
        serde_json::from_str(&self.arguments).map_err(|e| {
            RealizarError::InvalidConfiguration(format!("Failed to parse tool arguments: {e}"))
        })
    }
}

/// Tool call result
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResult {
    /// Tool call ID this result is for
    pub tool_call_id: String,
    /// Result content (JSON or plain text)
    pub content: String,
    /// Whether the tool call succeeded
    #[serde(default = "default_true")]
    pub success: bool,
}

fn default_true() -> bool {
    true
}

impl ToolResult {
    /// Create successful result
    pub fn success(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
            success: true,
        }
    }

    /// Create error result
    pub fn error(tool_call_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: error.into(),
            success: false,
        }
    }
}

/// Tool call parser for extracting tool calls from model output
pub struct ToolCallParser {
    /// Available tools
    tools: Vec<ToolDefinition>,
    /// Tool call format (default: OpenAI-style JSON)
    format: ToolCallFormat,
    /// Next tool call ID
    next_id: u64,
}

/// Format for tool calls in model output
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolCallFormat {
    /// OpenAI-style JSON: {"name": "tool", "arguments": {...}}
    #[default]
    OpenAI,
    /// Anthropic-style XML: <tool_use><name>tool</name><input>{...}</input></tool_use>
    Anthropic,
    /// Hermes format: <tool_call>{"name": "tool", "arguments": {...}}</tool_call>
    Hermes,
}

impl ToolCallParser {
    /// Create new parser with tools
    pub fn new(tools: Vec<ToolDefinition>) -> Self {
        Self {
            tools,
            format: ToolCallFormat::default(),
            next_id: 0,
        }
    }

    /// Set tool call format
    #[must_use]
    pub fn with_format(mut self, format: ToolCallFormat) -> Self {
        self.format = format;
        self
    }

    /// Generate unique tool call ID
    pub fn generate_id(&mut self) -> String {
        let id = format!("call_{}", self.next_id);
        self.next_id += 1;
        id
    }

    /// Get available tool names
    pub fn tool_names(&self) -> impl Iterator<Item = &str> {
        self.tools.iter().map(|t| t.name.as_str())
    }

    /// Get tool by name
    pub fn get_tool(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Parse tool calls from text (OpenAI format)
    ///
    /// Looks for JSON objects with "name" and "arguments" fields.
    pub fn parse(&mut self, text: &str) -> Vec<ToolCall> {
        match self.format {
            ToolCallFormat::OpenAI => self.parse_openai(text),
            ToolCallFormat::Anthropic => self.parse_anthropic(text),
            ToolCallFormat::Hermes => self.parse_hermes(text),
        }
    }

    /// Try to extract a tool call from a parsed JSON value with "name" and "arguments" fields.
    fn try_extract_json_tool_call(&mut self, value: &serde_json::Value) -> Option<ToolCall> {
        let name = value.get("name").and_then(|v| v.as_str())?;
        let args = value.get("arguments")?;
        if !self.tools.iter().any(|t| t.name == name) {
            return None;
        }
        let arguments = if args.is_string() {
            args.as_str().expect("checked is_string above").to_string()
        } else {
            args.to_string()
        };
        Some(ToolCall::new(self.generate_id(), name, arguments))
    }

    fn parse_openai(&mut self, text: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();

        // Try to find JSON objects with tool call structure
        // Look for patterns like {"name": "...", "arguments": {...}}
        let mut start = 0;
        while let Some(pos) = text[start..].find('{') {
            let abs_pos = start + pos;
            if let Some(end) = find_matching_brace(&text[abs_pos..]) {
                let json_str = &text[abs_pos..=(abs_pos + end)];
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if let Some(call) = self.try_extract_json_tool_call(&value) {
                        calls.push(call);
                    }
                }
                start = abs_pos + end + 1;
            } else {
                start = abs_pos + 1;
            }
        }

        calls
    }

    fn parse_anthropic(&mut self, text: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();

        // Look for <tool_use>...</tool_use> blocks
        let mut pos = 0;
        while let Some(start) = text[pos..].find("<tool_use>") {
            let abs_start = pos + start + 10; // Skip "<tool_use>"
            if let Some(end) = text[abs_start..].find("</tool_use>") {
                let content = &text[abs_start..abs_start + end];

                // Extract name
                let name = extract_xml_tag(content, "name");
                let input = extract_xml_tag(content, "input");

                if let (Some(name), Some(input)) = (name, input) {
                    if self.tools.iter().any(|t| t.name == name) {
                        calls.push(ToolCall::new(self.generate_id(), name, input));
                    }
                }

                pos = abs_start + end + 11; // Skip "</tool_use>"
            } else {
                break;
            }
        }

        calls
    }

    fn parse_hermes(&mut self, text: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();

        // Look for <tool_call>...</tool_call> blocks
        let mut pos = 0;
        while let Some(start) = text[pos..].find("<tool_call>") {
            let abs_start = pos + start + 11; // Skip "<tool_call>"
            if let Some(end) = text[abs_start..].find("</tool_call>") {
                let json_str = text[abs_start..abs_start + end].trim();

                if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if let Some(call) = self.try_extract_json_tool_call(&value) {
                        calls.push(call);
                    }
                }

                pos = abs_start + end + 12; // Skip "</tool_call>"
            } else {
                break;
            }
        }

        calls
    }
}

/// Process a single character for brace matching state machine.
/// Returns (depth_delta, toggle_string, set_escape) for the char.
#[inline]
fn process_brace_char(c: char, in_string: bool) -> (i32, bool, bool) {
    match c {
        '\\' if in_string => (0, false, true),
        '"' => (0, true, false),
        '{' if !in_string => (1, false, false),
        '}' if !in_string => (-1, false, false),
        _ => (0, false, false),
    }
}

/// Find matching closing brace, handling nested braces
fn find_matching_brace(s: &str) -> Option<usize> {
    let mut depth = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, c) in s.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }

        let (delta, toggle, escape) = process_brace_char(c, in_string);
        depth += delta;
        if toggle {
            in_string = !in_string;
        }
        escape_next = escape;

        if depth == 0 && delta < 0 {
            return Some(i);
        }
    }

    None
}

/// Extract content of an XML tag
fn extract_xml_tag(content: &str, tag: &str) -> Option<String> {
    let open_tag = format!("<{tag}>");
    let close_tag = format!("</{tag}>");

    if let Some(start) = content.find(&open_tag) {
        let value_start = start + open_tag.len();
        if let Some(end) = content[value_start..].find(&close_tag) {
            return Some(content[value_start..value_start + end].to_string());
        }
    }
    None
}

/// Generate grammar for tool calling output
///
/// Creates a grammar that constrains model output to valid tool calls.
pub fn generate_tool_grammar(tools: &[ToolDefinition]) -> Grammar {
    let mut grammar = Grammar::default();

    // Add basic rules
    add_json_whitespace_rules(&mut grammar);

    // Generate alternatives for each tool
    let mut tool_alternatives = Vec::new();

    for tool in tools {
        let tool_rule = format!("tool_{}", tool.name);

        // Generate parameter object grammar
        let params_rule = format!("{tool_rule}_params");
        generate_params_grammar(&mut grammar, &params_rule, &tool.parameters);

        // Tool call rule: {"name": "tool_name", "arguments": {...}}
        let mut elements = vec![
            GrammarElement::Char('{'),
            GrammarElement::RuleRef("ws".to_string()),
            GrammarElement::Char('"'),
            GrammarElement::Char('n'),
            GrammarElement::Char('a'),
            GrammarElement::Char('m'),
            GrammarElement::Char('e'),
            GrammarElement::Char('"'),
            GrammarElement::RuleRef("ws".to_string()),
            GrammarElement::Char(':'),
            GrammarElement::RuleRef("ws".to_string()),
            GrammarElement::Char('"'),
        ];

        // Tool name literal
        for c in tool.name.chars() {
            elements.push(GrammarElement::Char(c));
        }

        elements.extend([
            GrammarElement::Char('"'),
            GrammarElement::RuleRef("ws".to_string()),
            GrammarElement::Char(','),
            GrammarElement::RuleRef("ws".to_string()),
            GrammarElement::Char('"'),
            GrammarElement::Char('a'),
            GrammarElement::Char('r'),
            GrammarElement::Char('g'),
            GrammarElement::Char('u'),
            GrammarElement::Char('m'),
            GrammarElement::Char('e'),
            GrammarElement::Char('n'),
            GrammarElement::Char('t'),
            GrammarElement::Char('s'),
            GrammarElement::Char('"'),
            GrammarElement::RuleRef("ws".to_string()),
            GrammarElement::Char(':'),
            GrammarElement::RuleRef("ws".to_string()),
            GrammarElement::RuleRef(params_rule),
            GrammarElement::RuleRef("ws".to_string()),
            GrammarElement::Char('}'),
        ]);

        grammar.add_rule(GrammarRule::new(
            &tool_rule,
            vec![GrammarAlternative::new(elements)],
        ));

        tool_alternatives.push(GrammarAlternative::new(vec![GrammarElement::RuleRef(
            tool_rule,
        )]));
    }

    // Root rule: one of the tools
    grammar.add_rule(GrammarRule::new("root", tool_alternatives));

    grammar
}
