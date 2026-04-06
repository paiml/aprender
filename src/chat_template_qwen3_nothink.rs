/// PMAT-181: ChatML template variant for Qwen3 with thinking mode disabled.
///
/// Qwen3 enters thinking mode by default, emitting `<think>...</think>` blocks
/// before answering. For tool-calling tasks this wastes tokens and can loop
/// on `</think>` tags. This variant pre-fills an empty thinking block to skip
/// directly to the response.
pub struct Qwen3NoThinkTemplate {
    inner: ChatMLTemplate,
}

impl Qwen3NoThinkTemplate {
    /// Create a new Qwen3 no-thinking template (PMAT-181).
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: ChatMLTemplate::new(),
        }
    }
}

impl Default for Qwen3NoThinkTemplate {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatTemplateEngine for Qwen3NoThinkTemplate {
    fn format_message(&self, role: &str, content: &str) -> Result<String, RealizarError> {
        self.inner.format_message(role, content)
    }

    fn format_conversation(&self, messages: &[ChatMessage]) -> Result<String, RealizarError> {
        use std::fmt::Write;
        let mut result = String::new();
        for msg in messages {
            let safe_content = sanitize_special_tokens(&msg.content);
            let _ = write!(
                result,
                "<|im_start|>{}\n{}<|im_end|>\n",
                msg.role, safe_content
            );
        }
        // Pre-fill empty thinking block → model skips directly to response
        result.push_str("<|im_start|>assistant\n<think>\n</think>\n");
        Ok(result)
    }

    fn special_tokens(&self) -> &SpecialTokens {
        self.inner.special_tokens()
    }

    fn format(&self) -> TemplateFormat {
        TemplateFormat::Qwen3NoThink
    }

    fn supports_system_prompt(&self) -> bool {
        true
    }
}
