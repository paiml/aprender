
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
    // M32d Step 6 (companion claude-code-parity-apr-poc.md § "M32d FAST
    // PATH"): Qwen3-Coder / Qwen3-MoE-arch models do NOT have thinking
    // mode — Qwen3MoeForCausalLM was trained without `<think>` blocks.
    // Pre-injecting empty `<think>\n</think>\n` confuses the model and
    // causes it to emit `<|endoftext|>` immediately. Use plain ChatML
    // for qwen3_moe; keep Qwen3NoThink for dense Qwen3.
    if name_lower.contains("qwen3_moe") || name_lower.contains("qwen3moe") {
        return TemplateFormat::ChatML;
    }
    // PMAT-181: Qwen3 gets special no-think template (before generic "qwen" match)
    if name_lower.contains("qwen3") {
        return TemplateFormat::Qwen3NoThink;
    }

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
        TemplateFormat::Qwen3NoThink => Box::new(Qwen3NoThinkTemplate::new()),
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

/// #3723: set the thinking mode of an already RENDERED prompt, for `--thinking on|off`.
///
/// `None` and `Some(false)` return the rendering unchanged: OFF is what production renders
/// today (a Qwen3/Qwen3.5 model is routed to [`Qwen3NoThinkTemplate`], whose rendering ends
/// in an EMPTY `<think>` block that pre-closes the model's reasoning).
///
/// `Some(true)` removes that empty block, so the model opens its own. It is the derivation
/// `apr qa`'s golden ON leg has always used (`without_thinking_prefill`, #3724), applied to the
/// rendering rather than to a template name, so a replacement template is followed rather than
/// bypassed. The prompt is never pre-rendered by a caller and re-sent as user text: realizar
/// would zero-width-escape its special tokens (#3743).
///
/// # Errors
///
/// `--thinking on` on a rendering with no empty `<think>` prefill is REFUSED by name: that
/// template has no thinking mode to enable, and silently answering in OFF mode is the defect
/// #3723 exists to remove.
pub fn apply_thinking_mode(rendered: &str, thinking: Option<bool>) -> Result<String, RealizarError> {
    if thinking != Some(true) {
        return Ok(rendered.to_string());
    }
    let refuse = || {
        RealizarError::InferenceError(
            "--thinking on: this model's rendered prompt carries no empty <think></think> \
             prefill, so its chat template has no thinking mode to enable (#3723). Refused, \
             not ignored: re-run without --thinking, or with --thinking off."
                .to_string(),
        )
    };
    let start = rendered.rfind("<think>").ok_or_else(refuse)?;
    let tail = &rendered[start + "<think>".len()..];
    let close = tail.find("</think>").ok_or_else(refuse)?;
    // Only an EMPTY block at the very END is the suppression prefill. A block with reasoning
    // in it is conversation content, and cutting it would change what was asked.
    if !tail[..close].trim().is_empty() || !tail[close + "</think>".len()..].trim().is_empty() {
        return Err(refuse());
    }
    Ok(rendered[..start].to_string())
}

#[cfg(test)]
mod thinking_mode_tests {
    use super::*;

    fn qwen35(q: &str) -> String {
        format_messages(&[ChatMessage::user(q)], Some("Qwen3.5-0.8B-Q4_K_M.gguf")).expect("render")
    }

    /// #3723 must-RED: ON renders the thinking template, NOT the no-think one. A mutant mapping ON
    /// to OFF leaves the empty prefill in place and fails here.
    #[test]
    fn thinking_on_removes_the_empty_prefill() {
        let off = qwen35("What is 2+2?");
        assert!(off.ends_with("<|im_start|>assistant\n<think>\n</think>\n"), "{off:?}");
        let on = apply_thinking_mode(&off, Some(true)).expect("a Qwen3.5 template has a thinking mode");
        assert_ne!(on, off, "--thinking on rendered the no-think prompt");
        assert!(on.ends_with("<|im_start|>assistant\n"), "{on:?}");
        assert!(!on.contains("<think>"), "{on:?}");
        // the conversation itself is untouched
        assert_eq!(format!("{on}<think>\n</think>\n"), off);
    }

    /// #3723 must-RED: OFF (and no flag) still render no-think, byte for byte.
    #[test]
    fn thinking_off_and_absent_keep_the_no_think_rendering() {
        let off = qwen35("What is 2+2?");
        assert_eq!(apply_thinking_mode(&off, Some(false)).expect("off"), off);
        assert_eq!(apply_thinking_mode(&off, None).expect("absent"), off);
    }

    /// ON on a template with no thinking mode is refused by name, never silently OFF.
    #[test]
    fn thinking_on_without_a_thinking_template_is_refused() {
        let chatml = format_messages(&[ChatMessage::user("hi")], Some("Qwen2-0.5B-Instruct")).expect("render");
        let err = apply_thinking_mode(&chatml, Some(true)).expect_err("ChatML has no thinking mode");
        assert!(err.to_string().contains("no thinking mode to enable (#3723)"), "{err}");
        assert_eq!(apply_thinking_mode(&chatml, Some(false)).expect("off"), chatml);
    }

    /// A think block WITH content, or text after an empty one, is conversation, not a prefill.
    #[test]
    fn a_non_empty_or_non_trailing_block_is_not_a_prefill() {
        for r in [
            "<|im_start|>assistant\n<think>\nreasoning\n</think>\n",
            "<|im_start|>assistant\n<think>\n</think>\nanswer",
            "<|im_start|>assistant\n<think>\n",
        ] {
            assert!(apply_thinking_mode(r, Some(true)).is_err(), "{r:?}");
        }
        assert_eq!(
            apply_thinking_mode("u<think>  \n\t</think>\n", Some(true)).expect("whitespace-only is empty"),
            "u"
        );
    }
}
