
impl ChatTemplateEngine for RawTemplate {
    fn format_message(&self, _role: &str, content: &str) -> Result<String, RealizarError> {
        // Sanitize content to prevent prompt injection (F-SEC-220)
        // Even raw templates should sanitize to prevent special token attacks
        Ok(sanitize_special_tokens(content))
    }

    fn format_conversation(&self, messages: &[ChatMessage]) -> Result<String, RealizarError> {
        // Sanitize content to prevent prompt injection (F-SEC-220).
        // PMAT-763: newline-separate messages. The previous `.collect::<String>()`
        // concatenated content with NO separators, so a multi-turn chat sent to an
        // unknown / "default"-named model (RawTemplate is the fallback selected by
        // detect_format_from_name) became e.g. "HelloWorld" — a prompt the model can't parse
        // into turns. `join("\n")` separates BETWEEN turns while leaving a single message
        // verbatim (no spurious trailing newline).
        let result = messages
            .iter()
            .map(|m| sanitize_special_tokens(&m.content))
            .collect::<Vec<_>>()
            .join("\n");
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
// Auto-Detection
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
/// use realizar::chat_template::{detect_format_from_name, TemplateFormat};
///
/// assert_eq!(detect_format_from_name("TinyLlama-1.1B-Chat"), TemplateFormat::Zephyr);
/// assert_eq!(detect_format_from_name("Qwen2-0.5B-Instruct"), TemplateFormat::ChatML);
/// ```
#[must_use]
pub fn detect_format_from_name(model_name: &str) -> TemplateFormat {
    let name_lower = model_name.to_lowercase();

    // Pattern rules ordered by specificity (more specific patterns first)
    // Format: (patterns, format) - check patterns before formats that share prefixes
    //
    // This is the FALLBACK for a file with no `tokenizer.chat_template`. A file that
    // ships one gets it rendered (`EmbeddedChatTemplate`, #3755): its thinking switch
    // included, so no Qwen3 no-think scaffold is typed here any more. The one that was,
    // `<think>\n</think>\n`, was not the model's (`<think>\n\n</think>\n\n`), and
    // Qwen3.5-0.8B answered 2+2 = "2" with it.

    let rules: &[(&[&str], TemplateFormat)] = &[
        // ChatML: Qwen (2.x), OpenHermes, Yi
        (&["qwen", "openhermes", "yi-"], TemplateFormat::ChatML),
        // Zephyr: TinyLlama, Zephyr, StableLM (check BEFORE llama!)
        (&["tinyllama", "zephyr", "stablelm"], TemplateFormat::Zephyr),
        // Mistral/Mixtral (check before LLaMA since both use [INST])
        (&["mistral", "mixtral"], TemplateFormat::Mistral),
        // LLaMA 2 / Vicuna
        (&["llama", "vicuna"], TemplateFormat::Llama2),
        // Phi variants
        (&["phi-", "phi2", "phi3"], TemplateFormat::Phi),
        // Alpaca
        (&["alpaca"], TemplateFormat::Alpaca),
    ];

    for (patterns, format) in rules {
        if patterns.iter().any(|p| name_lower.contains(p)) {
            return *format;
        }
    }

    TemplateFormat::Raw
}

/// Auto-detect template format from special tokens
#[must_use]
pub fn detect_format_from_tokens(special_tokens: &SpecialTokens) -> TemplateFormat {
    if special_tokens.im_start_token.is_some() || special_tokens.im_end_token.is_some() {
        return TemplateFormat::ChatML;
    }

    if special_tokens.inst_start.is_some() || special_tokens.inst_end.is_some() {
        return TemplateFormat::Llama2;
    }

    TemplateFormat::Raw
}

/// Create a template engine for a given format
#[must_use]
pub fn create_template(format: TemplateFormat) -> Box<dyn ChatTemplateEngine> {
    match format {
        TemplateFormat::ChatML => Box::new(ChatMLTemplate::new()),
        // Legacy variant (#3755): no hand-typed scaffold remains; plain ChatML.
        TemplateFormat::Qwen3NoThink => Box::new(ChatMLTemplate::new()),
        TemplateFormat::Llama2 => Box::new(Llama2Template::new()),
        TemplateFormat::Zephyr => Box::new(ZephyrTemplate::new()),
        TemplateFormat::Mistral => Box::new(MistralTemplate::new()),
        TemplateFormat::Phi => Box::new(PhiTemplate::new()),
        TemplateFormat::Alpaca => Box::new(AlpacaTemplate::new()),
        TemplateFormat::Custom | TemplateFormat::Raw => Box::new(RawTemplate::new()),
    }
}

/// Auto-detect and create template from model name
#[must_use]
pub fn auto_detect_template(model_name: &str) -> Box<dyn ChatTemplateEngine> {
    let format = detect_format_from_name(model_name);
    create_template(format)
}

/// Format chat messages using auto-detected template
///
/// This is the main entry point for the API. It replaces the naive
/// "System: ...\nUser: ...\nAssistant: " format with proper model-specific
/// templates.
///
/// # Arguments
/// * `messages` - The chat messages to format
/// * `model_name` - Optional model name for auto-detection (defaults to Raw)
///
/// # Returns
/// Formatted prompt string ready for tokenization
///
/// # Example
///
/// ```
/// use realizar::chat_template::{ChatMessage, format_messages};
///
/// let messages = vec![
///     ChatMessage::system("You are helpful."),
///     ChatMessage::user("Hello!"),
/// ];
///
/// // With model name - uses ChatML format
/// let prompt = format_messages(&messages, Some("Qwen2-0.5B")).expect("prompt");
/// assert!(prompt.contains("<|im_start|>"));
///
/// // Without model name - uses Raw format
/// let prompt = format_messages(&messages, None).expect("prompt");
/// assert!(prompt.contains("You are helpful."));
/// ```
pub fn format_messages(
    messages: &[ChatMessage],
    model_name: Option<&str>,
) -> Result<String, RealizarError> {
    let template = model_name.map_or_else(
        || Box::new(RawTemplate::new()) as Box<dyn ChatTemplateEngine>,
        auto_detect_template,
    );
    template.format_conversation(messages)
}
