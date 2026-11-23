// Qwen3.5-family thinking chat template (also the Qwen3.8-Flash-Next / qwen4_exp template).
// Source: Qwen/Qwen3.8-Flash-Next chat_template.jinja (text content, no tools, no vision).

/// Reasoning effort accepted by the Qwen3.5-family template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    /// Default: long, careful reasoning.
    XHigh,
    /// No instruction text is injected.
    Medium,
    /// Brief reasoning.
    Low,
}

/// One turn for the thinking template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinkingMessage {
    /// Role: system, user or assistant.
    pub role: String,
    /// Visible content.
    pub content: String,
    /// Assistant reasoning (the `<think>` body), if any.
    pub reasoning_content: Option<String>,
}

/// Render options, mirroring the template variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35ThinkingOptions {
    /// `enable_thinking` (template default: on).
    pub enable_thinking: bool,
    /// `reasoning_effort` (template default: xhigh).
    pub reasoning_effort: ReasoningEffort,
    /// `preserve_thinking` (template default: on).
    pub preserve_thinking: bool,
    /// `add_generation_prompt`.
    pub add_generation_prompt: bool,
}

impl Default for Qwen35ThinkingOptions {
    fn default() -> Self {
        Self {
            enable_thinking: true,
            reasoning_effort: ReasoningEffort::XHigh,
            preserve_thinking: true,
            add_generation_prompt: true,
        }
    }
}

/// Qwen3.5-family chat template with thinking tags (no tools, text content).
#[derive(Debug, Default, Clone, Copy)]
pub struct Qwen35ThinkingTemplate;

const INSTR_XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const INSTR_LOW: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";

fn qwen35_err(msg: &str) -> AprenderError {
    AprenderError::ValidationError {
        message: msg.to_string(),
    }
}

fn qwen35_instruction(opts: &Qwen35ThinkingOptions) -> &'static str {
    if !opts.enable_thinking {
        return "";
    }
    match opts.reasoning_effort {
        ReasoningEffort::XHigh => INSTR_XHIGH,
        ReasoningEffort::Low => INSTR_LOW,
        ReasoningEffort::Medium => "",
    }
}

fn qwen35_system_block(first: &ThinkingMessage, instr: &str) -> String {
    let c = if first.role == "system" {
        first.content.trim()
    } else {
        ""
    };
    match (c.is_empty(), instr.is_empty()) {
        (true, true) => String::new(),
        (true, false) => format!("<|im_start|>system\n{instr}<|im_end|>\n"),
        (false, true) => format!("<|im_start|>system\n{c}<|im_end|>\n"),
        (false, false) => format!("<|im_start|>system\n{instr}\n\n{c}<|im_end|>\n"),
    }
}

fn qwen35_is_query(m: &ThinkingMessage) -> bool {
    let c = m.content.trim();
    m.role == "user" && !(c.starts_with("<tool_response>") && c.ends_with("</tool_response>"))
}

fn qwen35_turn(
    out: &mut String,
    i: usize,
    m: &ThinkingMessage,
    keep_think: bool,
) -> Result<(), AprenderError> {
    let content = m.content.trim();
    match m.role.as_str() {
        "system" if i != 0 => return Err(qwen35_err("System message must be at the beginning.")),
        "system" => {}
        "user" => out.push_str(&format!("<|im_start|>user\n{content}<|im_end|>\n")),
        "assistant" if keep_think => {
            let r = m.reasoning_content.as_deref().unwrap_or("").trim();
            out.push_str(&format!(
                "<|im_start|>assistant\n<think>\n{r}\n</think>\n\n{content}<|im_end|>\n"
            ));
        }
        "assistant" => out.push_str(&format!("<|im_start|>assistant\n{content}<|im_end|>\n")),
        _ => return Err(qwen35_err("Unexpected message role.")),
    }
    Ok(())
}

impl Qwen35ThinkingTemplate {
    /// Render the conversation exactly as the published jinja does for text-only,
    /// tool-free input. `Err` mirrors the template's `raise_exception` calls.
    pub fn render(
        &self,
        messages: &[ThinkingMessage],
        opts: &Qwen35ThinkingOptions,
    ) -> Result<String, AprenderError> {
        let first = messages
            .first()
            .ok_or_else(|| qwen35_err("No messages provided."))?;
        let mut out = qwen35_system_block(first, qwen35_instruction(opts));
        let last_query = messages
            .iter()
            .rposition(qwen35_is_query)
            .ok_or_else(|| qwen35_err("No user query found in messages."))?;
        for (i, m) in messages.iter().enumerate() {
            qwen35_turn(&mut out, i, m, opts.preserve_thinking || i > last_query)?;
        }
        if opts.add_generation_prompt {
            out.push_str("<|im_start|>assistant\n");
            out.push_str(if opts.enable_thinking {
                "<think>\n"
            } else {
                "<think>\n\n</think>\n\n"
            });
        }
        Ok(out)
    }
}

/// Split generated text into (reasoning, answer) at the first `</think>`.
/// A leading `<think>` (present when the whole assistant turn is passed, absent when only
/// the continuation of the generation prompt is) is accepted. No `</think>`: no reasoning.
#[must_use]
pub fn split_thinking(text: &str) -> (Option<String>, String) {
    match text.split_once("</think>") {
        Some((before, after)) => {
            let r = before.trim().trim_start_matches("<think>").trim();
            let reasoning = if r.is_empty() {
                None
            } else {
                Some(r.to_string())
            };
            (reasoning, after.trim().to_string())
        }
        None => (None, text.to_string()),
    }
}
