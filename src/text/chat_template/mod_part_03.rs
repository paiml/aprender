
impl ChatTemplateEngine for AlpacaTemplate {
    fn format_message(&self, role: &str, content: &str) -> Result<String, AprenderError> {
        // Sanitize user content to prevent prompt injection (GH-204)
        let safe_content = if role == "user" {
            sanitize_user_content(content)
        } else {
            content.to_string()
        };
        match role {
            "system" => Ok(format!("{safe_content}\n\n")),
            "user" => Ok(format!("### Instruction:\n{safe_content}\n\n")),
            "assistant" => Ok(format!("### Response:\n{safe_content}\n\n")),
            _ => Ok(safe_content),
        }
    }

    fn format_conversation(&self, messages: &[ChatMessage]) -> Result<String, AprenderError> {
        let mut result = String::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    result.push_str(&msg.content);
                    result.push_str("\n\n");
                }
                "user" => {
                    // Sanitize user content to prevent prompt injection (GH-204)
                    let safe_content = sanitize_user_content(&msg.content);
                    result.push_str("### Instruction:\n");
                    result.push_str(&safe_content);
                    result.push_str("\n\n");
                }
                "assistant" => {
                    result.push_str("### Response:\n");
                    result.push_str(&msg.content);
                    result.push_str("\n\n");
                }
                _ => {}
            }
        }

        // Add generation prompt
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
///
/// Simply concatenates messages without any template.
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

impl ChatTemplateEngine for RawTemplate {
    fn format_message(&self, _role: &str, content: &str) -> Result<String, AprenderError> {
        Ok(content.to_string())
    }

    fn format_conversation(&self, messages: &[ChatMessage]) -> Result<String, AprenderError> {
        let result: String = messages.iter().map(|m| m.content.as_str()).collect();
        Ok(result)
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn format(&self) -> TemplateFormat {
        TemplateFormat::Raw
    }

    fn supports_system_prompt(&self) -> bool {
        true
    }
}

// ============================================================================
// Auto-Detection (CTA-01 to CTA-10)
// ============================================================================

/// Auto-detect template format from model name or path
///
/// # Arguments
/// * `model_name` - Model name or path (e.g., "TinyLlama/TinyLlama-1.1B-Chat")
///
/// # Returns
/// Detected `TemplateFormat`
///
/// # Example
///
/// ```
/// use aprender::text::chat_template::{detect_format_from_name, TemplateFormat};
///
/// assert_eq!(detect_format_from_name("TinyLlama-1.1B-Chat"), TemplateFormat::Llama2);
/// assert_eq!(detect_format_from_name("Qwen2-0.5B-Instruct"), TemplateFormat::ChatML);
/// assert_eq!(detect_format_from_name("mistral-7b-instruct"), TemplateFormat::Mistral);
/// ```
#[must_use]
pub fn detect_format_from_name(model_name: &str) -> TemplateFormat {
    let name_lower = model_name.to_lowercase();

    // ChatML models
    if name_lower.contains("qwen")
        || name_lower.contains("openhermes")
        || name_lower.contains("yi-")
    {
        return TemplateFormat::ChatML;
    }

    // Mistral (check before LLaMA since Mistral uses similar tokens)
    if name_lower.contains("mistral") || name_lower.contains("mixtral") {
        return TemplateFormat::Mistral;
    }

    // LLaMA 2 / TinyLlama / Vicuna
    if name_lower.contains("llama")
        || name_lower.contains("vicuna")
        || name_lower.contains("tinyllama")
    {
        return TemplateFormat::Llama2;
    }

    // Phi
    if name_lower.contains("phi-") || name_lower.contains("phi2") || name_lower.contains("phi3") {
        return TemplateFormat::Phi;
    }

    // Alpaca
    if name_lower.contains("alpaca") {
        return TemplateFormat::Alpaca;
    }

    // Default to Raw
    TemplateFormat::Raw
}

/// Auto-detect template format from special tokens
///
/// # Arguments
/// * `special_tokens` - Special tokens from tokenizer
///
/// # Returns
/// Detected `TemplateFormat`
#[must_use]
pub fn detect_format_from_tokens(special_tokens: &SpecialTokens) -> TemplateFormat {
    // ChatML detection
    if special_tokens.im_start_token.is_some() || special_tokens.im_end_token.is_some() {
        return TemplateFormat::ChatML;
    }

    // LLaMA2/Mistral detection
    if special_tokens.inst_start.is_some() || special_tokens.inst_end.is_some() {
        return TemplateFormat::Llama2; // Could be Mistral, refine with model name
    }

    TemplateFormat::Raw
}

/// Create a template engine for a given format
///
/// # Arguments
/// * `format` - Template format
///
/// # Returns
/// Boxed `ChatTemplateEngine` implementation
#[must_use]
pub fn create_template(format: TemplateFormat) -> Box<dyn ChatTemplateEngine + Send + Sync> {
    match format {
        TemplateFormat::ChatML => Box::new(ChatMLTemplate::new()),
        TemplateFormat::Llama2 => Box::new(Llama2Template::new()),
        TemplateFormat::Mistral => Box::new(MistralTemplate::new()),
        TemplateFormat::Phi => Box::new(PhiTemplate::new()),
        TemplateFormat::Alpaca => Box::new(AlpacaTemplate::new()),
        TemplateFormat::Custom | TemplateFormat::Raw => Box::new(RawTemplate::new()),
    }
}

/// Auto-detect and create template from model name
///
/// # Arguments
/// * `model_name` - Model name or path
///
/// # Returns
/// Boxed `ChatTemplateEngine` implementation
#[must_use]
pub fn auto_detect_template(model_name: &str) -> Box<dyn ChatTemplateEngine + Send + Sync> {
    let format = detect_format_from_name(model_name);
    create_template(format)
}

#[cfg(test)]
mod tests;
