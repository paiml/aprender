
/// LLaMA 2 Template (TinyLlama, Vicuna, LLaMA 2)
///
/// Format: `<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]`
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
    fn format_message(&self, role: &str, content: &str) -> Result<String, RealizarError> {
        // Sanitize content to prevent prompt injection (F-SEC-220)
        let safe_content = sanitize_special_tokens(content);
        match role {
            "system" => Ok(format!("<<SYS>>\n{safe_content}\n<</SYS>>\n\n")),
            "user" => Ok(format!("[INST] {safe_content} [/INST]")),
            "assistant" => Ok(format!(" {safe_content}</s>")),
            _ => Ok(safe_content),
        }
    }

    fn format_conversation(&self, messages: &[ChatMessage]) -> Result<String, RealizarError> {
        let mut result = String::from("<s>");
        let mut system_prompt: Option<String> = None;
        let mut in_user_turn = false;
        // PMAT-762: have we emitted any [INST] turn yet? A new Llama-2 round opens with <s>,
        // but only AFTER a completed assistant turn — not after a leading system message.
        let mut turn_started = false;

        for msg in messages.iter() {
            // Sanitize content to prevent prompt injection (F-SEC-220)
            let safe_content = sanitize_special_tokens(&msg.content);

            match msg.role.as_str() {
                "system" => {
                    // PMAT-762: concatenate multiple system messages instead of overwriting.
                    // The previous `system_prompt = Some(safe_content)` silently dropped all
                    // but the LAST system message.
                    match system_prompt {
                        Some(ref mut prev) => {
                            prev.push_str("\n\n");
                            prev.push_str(&safe_content);
                        },
                        None => system_prompt = Some(safe_content),
                    }
                },
                "user" => {
                    // Open a new round with <s> only AFTER a completed assistant turn. The
                    // previous `i > 0 && !in_user_turn` also fired when the only prior message
                    // was a system one, producing a spurious DOUBLE <s> (double-BOS) for the
                    // common [system, user] case.
                    if turn_started && !in_user_turn {
                        result.push_str("<s>");
                    }
                    result.push_str("[INST] ");

                    if let Some(sys) = system_prompt.take() {
                        result.push_str("<<SYS>>\n");
                        result.push_str(&sys);
                        result.push_str("\n<</SYS>>\n\n");
                    }

                    result.push_str(&safe_content);
                    result.push_str(" [/INST]");
                    in_user_turn = true;
                    turn_started = true;
                },
                "assistant" => {
                    result.push(' ');
                    result.push_str(&safe_content);
                    result.push_str("</s>");
                    in_user_turn = false;
                },
                _ => {},
            }
        }

        // PMAT-762: flush a system prompt never consumed by a following user message (e.g. a
        // system-only request, or one ending on a system turn). Previously it was silently
        // DROPPED — the entire system context was lost. Emit it as a system block inside an
        // (empty-user) INST turn so the prompt stays well-formed Llama-2.
        if let Some(sys) = system_prompt.take() {
            if !in_user_turn {
                result.push_str("[INST] ");
            }
            result.push_str("<<SYS>>\n");
            result.push_str(&sys);
            result.push_str("\n<</SYS>>\n\n [/INST]");
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
    fn format_message(&self, role: &str, content: &str) -> Result<String, RealizarError> {
        // Sanitize content to prevent prompt injection (F-SEC-220)
        let safe_content = sanitize_special_tokens(content);
        match role {
            "user" => Ok(format!("[INST] {safe_content} [/INST]")),
            "assistant" => Ok(format!(" {safe_content}</s>")),
            "system" => Ok(format!("{safe_content}\n\n")),
            _ => Ok(safe_content),
        }
    }

    fn format_conversation(&self, messages: &[ChatMessage]) -> Result<String, RealizarError> {
        let mut result = String::from("<s>");

        for msg in messages {
            // Sanitize content to prevent prompt injection (F-SEC-220)
            let safe_content = sanitize_special_tokens(&msg.content);

            match msg.role.as_str() {
                "user" => {
                    result.push_str("[INST] ");
                    result.push_str(&safe_content);
                    result.push_str(" [/INST]");
                },
                "assistant" => {
                    result.push(' ');
                    result.push_str(&safe_content);
                    result.push_str("</s>");
                },
                _ => {},
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
        false
    }
}

/// Zephyr Template (TinyLlama, Zephyr, StableLM)
///
/// Format: `<|user|>\n{content}</s>\n<|assistant|>\n`
///
/// This is the correct template for TinyLlama-1.1B-Chat-v1.0.
/// Note: TinyLlama is NOT Llama2 format despite the name!
#[derive(Debug, Clone)]
pub struct ZephyrTemplate {
    special_tokens: SpecialTokens,
}

impl ZephyrTemplate {
    /// Create a new Zephyr template
    #[must_use]
    pub fn new() -> Self {
        Self {
            special_tokens: SpecialTokens {
                bos_token: Some("<s>".to_string()),
                eos_token: Some("</s>".to_string()),
                ..Default::default()
            },
        }
    }
}

impl Default for ZephyrTemplate {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatTemplateEngine for ZephyrTemplate {
    fn format_message(&self, role: &str, content: &str) -> Result<String, RealizarError> {
        // Sanitize content to prevent prompt injection (F-SEC-220)
        let safe_content = sanitize_special_tokens(content);
        match role {
            "system" => Ok(format!("<|system|>\n{safe_content}</s>\n")),
            "user" => Ok(format!("<|user|>\n{safe_content}</s>\n")),
            "assistant" => Ok(format!("<|assistant|>\n{safe_content}</s>\n")),
            _ => Ok(safe_content),
        }
    }

    fn format_conversation(&self, messages: &[ChatMessage]) -> Result<String, RealizarError> {
        let mut result = String::new();

        for msg in messages {
            // Sanitize content to prevent prompt injection (F-SEC-220)
            let safe_content = sanitize_special_tokens(&msg.content);

            match msg.role.as_str() {
                "system" => {
                    result.push_str("<|system|>\n");
                    result.push_str(&safe_content);
                    result.push_str("</s>\n");
                },
                "user" => {
                    result.push_str("<|user|>\n");
                    result.push_str(&safe_content);
                    result.push_str("</s>\n");
                },
                "assistant" => {
                    result.push_str("<|assistant|>\n");
                    result.push_str(&safe_content);
                    result.push_str("</s>\n");
                },
                _ => {},
            }
        }

        // Add generation prompt
        result.push_str("<|assistant|>\n");

        Ok(result)
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn format(&self) -> TemplateFormat {
        TemplateFormat::Zephyr
    }

    fn supports_system_prompt(&self) -> bool {
        true
    }
}

/// Phi Template (Phi-2, Phi-3)
///
/// Format: `Instruct: {content}\nOutput:`
#[derive(Debug, Clone, Default)]
pub struct PhiTemplate {
    special_tokens: SpecialTokens,
}

impl PhiTemplate {
    /// Create a new Phi template
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl ChatTemplateEngine for PhiTemplate {
    fn format_message(&self, role: &str, content: &str) -> Result<String, RealizarError> {
        // Sanitize content to prevent prompt injection (F-SEC-220)
        let safe_content = sanitize_special_tokens(content);
        match role {
            "user" => Ok(format!("Instruct: {safe_content}\n")),
            "assistant" => Ok(format!("Output: {safe_content}\n")),
            "system" => Ok(format!("{safe_content}\n")),
            _ => Ok(safe_content),
        }
    }

    fn format_conversation(&self, messages: &[ChatMessage]) -> Result<String, RealizarError> {
        let mut result = String::new();

        for msg in messages {
            // Sanitize content to prevent prompt injection (F-SEC-220)
            let safe_content = sanitize_special_tokens(&msg.content);

            match msg.role.as_str() {
                "system" => {
                    result.push_str(&safe_content);
                    result.push('\n');
                },
                "user" => {
                    result.push_str("Instruct: ");
                    result.push_str(&safe_content);
                    result.push('\n');
                },
                "assistant" => {
                    result.push_str("Output: ");
                    result.push_str(&safe_content);
                    result.push('\n');
                },
                _ => {},
            }
        }

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
#[derive(Debug, Clone, Default)]
pub struct AlpacaTemplate {
    special_tokens: SpecialTokens,
}

impl AlpacaTemplate {
    /// Create a new Alpaca template
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl ChatTemplateEngine for AlpacaTemplate {
    fn format_message(&self, role: &str, content: &str) -> Result<String, RealizarError> {
        // Sanitize content to prevent prompt injection (F-SEC-220)
        let safe_content = sanitize_special_tokens(content);
        match role {
            "system" => Ok(format!("{safe_content}\n\n")),
            "user" => Ok(format!("### Instruction:\n{safe_content}\n\n")),
            "assistant" => Ok(format!("### Response:\n{safe_content}\n\n")),
            _ => Ok(safe_content),
        }
    }

    fn format_conversation(&self, messages: &[ChatMessage]) -> Result<String, RealizarError> {
        let mut result = String::new();

        for msg in messages {
            // Sanitize content to prevent prompt injection (F-SEC-220)
            let safe_content = sanitize_special_tokens(&msg.content);

            match msg.role.as_str() {
                "system" => {
                    result.push_str(&safe_content);
                    result.push_str("\n\n");
                },
                "user" => {
                    result.push_str("### Instruction:\n");
                    result.push_str(&safe_content);
                    result.push_str("\n\n");
                },
                "assistant" => {
                    result.push_str("### Response:\n");
                    result.push_str(&safe_content);
                    result.push_str("\n\n");
                },
                _ => {},
            }
        }

        result.push_str("### Response:\n");

        Ok(result)
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn format(&self) -> TemplateFormat {
        TemplateFormat::Alpaca
    }

    fn supports_system_prompt(&self) -> bool {
        true
    }
}

/// Raw Template (Fallback - no formatting)
#[derive(Debug, Clone, Default)]
pub struct RawTemplate {
    special_tokens: SpecialTokens,
}

impl RawTemplate {
    /// Create a new raw template
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}
