
impl Default for ChatMLTemplate {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatTemplateEngine for ChatMLTemplate {
    fn format_message(&self, role: &str, content: &str) -> Result<String, AprenderError> {
        // Sanitize user content to prevent prompt injection (GH-204)
        let safe_content = if role == "user" {
            sanitize_user_content(content)
        } else {
            content.to_string()
        };
        Ok(format!("<|im_start|>{role}\n{safe_content}<|im_end|>\n"))
    }

    fn format_conversation(&self, messages: &[ChatMessage]) -> Result<String, AprenderError> {
        use std::fmt::Write;
        let mut result = String::new();

        for msg in messages {
            // Sanitize user content to prevent prompt injection (GH-204)
            // System messages are trusted; user messages are untrusted
            let safe_content = if msg.role == "user" {
                sanitize_user_content(&msg.content)
            } else {
                msg.content.clone()
            };
            let _ = write!(
                result,
                "<|im_start|>{}\n{}<|im_end|>\n",
                msg.role, safe_content
            );
        }

        // Add generation prompt
        result.push_str("<|im_start|>assistant\n");

        Ok(result)
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn format(&self) -> TemplateFormat {
        TemplateFormat::ChatML
    }

    fn supports_system_prompt(&self) -> bool {
        true
    }
}

/// LLaMA 2 Template (TinyLlama, Vicuna, LLaMA 2)
///
/// Format: `<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]`
///
/// # Example
///
/// ```
/// use aprender::text::chat_template::{ChatMessage, Llama2Template, ChatTemplateEngine};
///
/// let template = Llama2Template::new();
/// let messages = vec![ChatMessage::user("Hello!")];
/// let output = template.format_conversation(&messages).expect("format conversation should succeed");
/// assert!(output.contains("[INST]"));
/// ```
#[derive(Debug, Clone)]
pub struct Llama2Template {
    special_tokens: SpecialTokens,
}

impl Llama2Template {
    /// Create a new LLaMA 2 template
    #[must_use]
    pub fn new() -> Self {
        Self {
            special_tokens: SpecialTokens {
                bos_token: Some("<s>".to_string()),
                eos_token: Some("</s>".to_string()),
                inst_start: Some("[INST]".to_string()),
                inst_end: Some("[/INST]".to_string()),
                sys_start: Some("<<SYS>>".to_string()),
                sys_end: Some("<</SYS>>".to_string()),
                ..Default::default()
            },
        }
    }
}

impl Default for Llama2Template {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatTemplateEngine for Llama2Template {
    fn format_message(&self, role: &str, content: &str) -> Result<String, AprenderError> {
        // Sanitize user content to prevent prompt injection (GH-204)
        let safe_content = if role == "user" {
            sanitize_user_content(content)
        } else {
            content.to_string()
        };
        match role {
            "system" => Ok(format!("<<SYS>>\n{safe_content}\n<</SYS>>\n\n")),
            "user" => Ok(format!("[INST] {safe_content} [/INST]")),
            "assistant" => Ok(format!(" {safe_content}</s>")),
            _ => Ok(safe_content),
        }
    }

    fn format_conversation(&self, messages: &[ChatMessage]) -> Result<String, AprenderError> {
        let mut result = String::from("<s>");
        let mut system_prompt: Option<String> = None;
        let mut in_user_turn = false;

        for (i, msg) in messages.iter().enumerate() {
            match msg.role.as_str() {
                "system" => {
                    system_prompt = Some(msg.content.clone());
                }
                "user" => {
                    // Sanitize user content to prevent prompt injection (GH-204)
                    let safe_content = sanitize_user_content(&msg.content);
                    if i > 0 && !in_user_turn {
                        result.push_str("<s>");
                    }
                    result.push_str("[INST] ");

                    // Include system prompt with first user message
                    if let Some(sys) = system_prompt.take() {
                        result.push_str("<<SYS>>\n");
                        result.push_str(&sys);
                        result.push_str("\n<</SYS>>\n\n");
                    }

                    result.push_str(&safe_content);
                    result.push_str(" [/INST]");
                    in_user_turn = true;
                }
                "assistant" => {
                    result.push(' ');
                    result.push_str(&msg.content);
                    result.push_str("</s>");
                    in_user_turn = false;
                }
                _ => {}
            }
        }

        Ok(result)
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn format(&self) -> TemplateFormat {
        TemplateFormat::Llama2
    }

    fn supports_system_prompt(&self) -> bool {
        true
    }
}

/// Mistral Template (Mistral, Mixtral)
///
/// Format: `<s>[INST] {user} [/INST]` (no system prompt support)
///
/// # Example
///
/// ```
/// use aprender::text::chat_template::{ChatMessage, MistralTemplate, ChatTemplateEngine};
///
/// let template = MistralTemplate::new();
/// assert!(!template.supports_system_prompt());
/// ```
#[derive(Debug, Clone)]
pub struct MistralTemplate {
    special_tokens: SpecialTokens,
}

impl MistralTemplate {
    /// Create a new Mistral template
    #[must_use]
    pub fn new() -> Self {
        Self {
            special_tokens: SpecialTokens {
                bos_token: Some("<s>".to_string()),
                eos_token: Some("</s>".to_string()),
                inst_start: Some("[INST]".to_string()),
                inst_end: Some("[/INST]".to_string()),
                ..Default::default()
            },
        }
    }
}

impl Default for MistralTemplate {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatTemplateEngine for MistralTemplate {
    fn format_message(&self, role: &str, content: &str) -> Result<String, AprenderError> {
        // Sanitize user content to prevent prompt injection (GH-204)
        let safe_content = if role == "user" {
            sanitize_user_content(content)
        } else {
            content.to_string()
        };
        match role {
            "user" => Ok(format!("[INST] {safe_content} [/INST]")),
            "assistant" => Ok(format!(" {safe_content}</s>")),
            "system" => {
                // Mistral doesn't support system prompts, prepend to first user message
                Ok(format!("{safe_content}\n\n"))
            }
            _ => Ok(safe_content),
        }
    }

    fn format_conversation(&self, messages: &[ChatMessage]) -> Result<String, AprenderError> {
        let mut result = String::from("<s>");

        for msg in messages {
            match msg.role.as_str() {
                "user" => {
                    // Sanitize user content to prevent prompt injection (GH-204)
                    let safe_content = sanitize_user_content(&msg.content);
                    result.push_str("[INST] ");
                    result.push_str(&safe_content);
                    result.push_str(" [/INST]");
                }
                "assistant" => {
                    result.push(' ');
                    result.push_str(&msg.content);
                    result.push_str("</s>");
                }
                // Mistral doesn't support system prompts - ignored
                _ => {}
            }
        }

        Ok(result)
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn format(&self) -> TemplateFormat {
        TemplateFormat::Mistral
    }

    fn supports_system_prompt(&self) -> bool {
        false // Mistral doesn't support system prompts
    }
}

/// Phi Template (Phi-2, Phi-3)
///
/// Format: `Instruct: {content}\nOutput:`
///
/// # Example
///
/// ```
/// use aprender::text::chat_template::{ChatMessage, PhiTemplate, ChatTemplateEngine};
///
/// let template = PhiTemplate::new();
/// let messages = vec![ChatMessage::user("Hello!")];
/// let output = template.format_conversation(&messages).expect("format conversation should succeed");
/// assert!(output.contains("Instruct:"));
/// ```
#[derive(Debug, Clone)]
pub struct PhiTemplate {
    special_tokens: SpecialTokens,
}

impl PhiTemplate {
    /// Create a new Phi template
    #[must_use]
    pub fn new() -> Self {
        Self {
            special_tokens: SpecialTokens::default(),
        }
    }
}

impl Default for PhiTemplate {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatTemplateEngine for PhiTemplate {
    fn format_message(&self, role: &str, content: &str) -> Result<String, AprenderError> {
        // Sanitize user content to prevent prompt injection (GH-204)
        let safe_content = if role == "user" {
            sanitize_user_content(content)
        } else {
            content.to_string()
        };
        match role {
            "user" => Ok(format!("Instruct: {safe_content}\n")),
            "assistant" => Ok(format!("Output: {safe_content}\n")),
            "system" => Ok(format!("{safe_content}\n")),
            _ => Ok(safe_content),
        }
    }

    fn format_conversation(&self, messages: &[ChatMessage]) -> Result<String, AprenderError> {
        let mut result = String::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    result.push_str(&msg.content);
                    result.push('\n');
                }
                "user" => {
                    // Sanitize user content to prevent prompt injection (GH-204)
                    let safe_content = sanitize_user_content(&msg.content);
                    result.push_str("Instruct: ");
                    result.push_str(&safe_content);
                    result.push('\n');
                }
                "assistant" => {
                    result.push_str("Output: ");
                    result.push_str(&msg.content);
                    result.push('\n');
                }
                _ => {}
            }
        }

        // Add generation prompt
        result.push_str("Output:");

        Ok(result)
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn format(&self) -> TemplateFormat {
        TemplateFormat::Phi
    }

    fn supports_system_prompt(&self) -> bool {
        true
    }
}

/// Alpaca Template
///
/// Format: `### Instruction:\n{content}\n\n### Response:`
///
/// # Example
///
/// ```
/// use aprender::text::chat_template::{ChatMessage, AlpacaTemplate, ChatTemplateEngine};
///
/// let template = AlpacaTemplate::new();
/// let messages = vec![ChatMessage::user("Hello!")];
/// let output = template.format_conversation(&messages).expect("format conversation should succeed");
/// assert!(output.contains("### Instruction:"));
/// ```
#[derive(Debug, Clone)]
pub struct AlpacaTemplate {
    special_tokens: SpecialTokens,
}

impl AlpacaTemplate {
    /// Create a new Alpaca template
    #[must_use]
    pub fn new() -> Self {
        Self {
            special_tokens: SpecialTokens::default(),
        }
    }
}

impl Default for AlpacaTemplate {
    fn default() -> Self {
        Self::new()
    }
}
