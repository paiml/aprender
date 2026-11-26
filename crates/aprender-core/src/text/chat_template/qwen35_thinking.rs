// Qwen3.5-family thinking chat template (also the Qwen3.8-Flash-Next / qwen4_exp template).
// Source: Qwen/Qwen3.8-Flash-Next chat_template.jinja, checked in verbatim at
// golden/qwen4_exp/chat_template.jinja (sha256 af16f654...9f6375). Every branch is ported:
// text, thinking, tools (jinja lines 57-75, 123-147, 151-162) and vision content parts
// (lines 3-41). `tests_qwen4_exp.rs` renders the published jinja itself and compares (#7336).

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

/// One item of a list-valued `content` (jinja lines 6-35).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThinkingContentPart {
    /// `{"type": "text", "text": ...}`.
    Text(String),
    /// An image item: rendered as `<|vision_start|><|image_pad|><|vision_end|>`.
    Image,
    /// A video item: rendered as `<|vision_start|><|video_pad|><|vision_end|>`.
    Video,
}

/// One assistant tool call (jinja lines 124-146). `arguments` keeps the caller's order;
/// a string value is emitted as is, any other value as JSON (line 140).
#[derive(Debug, Clone, PartialEq)]
pub struct ThinkingToolCall {
    /// Function name.
    pub name: String,
    /// Parameters, in emission order.
    pub arguments: Vec<(String, serde_json::Value)>,
}

/// One turn for the thinking template.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ThinkingMessage {
    /// Role: system, user, assistant or tool.
    pub role: String,
    /// Visible content (used when `parts` is empty).
    pub content: String,
    /// Assistant reasoning (the `<think>` body), if any.
    pub reasoning_content: Option<String>,
    /// List-valued content. When non-empty it replaces `content`.
    pub parts: Vec<ThinkingContentPart>,
    /// Assistant tool calls.
    pub tool_calls: Vec<ThinkingToolCall>,
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
    /// `add_vision_id` (template default: off): prefix each image / video with its number.
    pub add_vision_id: bool,
}

impl Default for Qwen35ThinkingOptions {
    fn default() -> Self {
        Self {
            enable_thinking: true,
            reasoning_effort: ReasoningEffort::XHigh,
            preserve_thinking: true,
            add_generation_prompt: true,
            add_vision_id: false,
        }
    }
}

/// Qwen3.5-family chat template with thinking tags, tools and vision content parts.
#[derive(Debug, Default, Clone, Copy)]
pub struct Qwen35ThinkingTemplate;

const INSTR_XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const INSTR_LOW: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
/// Jinja line 62.
const QWEN35_TOOLS_HEAD: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
/// Jinja line 68, verbatim.
const QWEN35_TOOLS_FORMAT: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

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

fn qwen35_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn qwen35_json_write(v: &serde_json::Value, out: &mut String) {
    match v {
        serde_json::Value::String(s) => qwen35_json_string(s, out),
        serde_json::Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                qwen35_json_write(x, out);
            }
            out.push(']');
        }
        serde_json::Value::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                qwen35_json_string(k, out);
                out.push_str(": ");
                qwen35_json_write(x, out);
            }
            out.push('}');
        }
        scalar => out.push_str(&scalar.to_string()),
    }
}

/// The `tojson` the template is rendered with by `transformers`
/// (`json.dumps(x, ensure_ascii=False)`): `", "` and `": "` separators, non-ASCII text
/// emitted raw. Object keys come out in the order the `serde_json::Value` holds them
/// (sorted unless serde_json's `preserve_order` is on). Numbers use serde_json's spelling,
/// which equals Python's for integers and for floats written without an exponent.
#[must_use]
pub fn qwen35_tojson(v: &serde_json::Value) -> String {
    let mut out = String::new();
    qwen35_json_write(v, &mut out);
    out
}

/// Image / video counters (jinja lines 1-2).
#[derive(Default)]
struct Qwen35VisionCount {
    image: usize,
    video: usize,
}

/// The `render_content` macro (jinja lines 3-41), untrimmed. `count` is false where the
/// template passes `do_vision_count=false`: the numbers printed are then the current ones.
fn qwen35_content(
    m: &ThinkingMessage,
    seen: &mut Qwen35VisionCount,
    count: bool,
    add_vision_id: bool,
    is_system: bool,
) -> Result<String, AprenderError> {
    if m.parts.is_empty() {
        return Ok(m.content.clone());
    }
    let mut out = String::new();
    for part in &m.parts {
        let (n, label, pad, refusal) = match part {
            ThinkingContentPart::Text(t) => {
                out.push_str(t);
                continue;
            }
            ThinkingContentPart::Image => (
                &mut seen.image,
                "Picture",
                "<|image_pad|>",
                "System message cannot contain images.",
            ),
            ThinkingContentPart::Video => (
                &mut seen.video,
                "Video",
                "<|video_pad|>",
                "System message cannot contain videos.",
            ),
        };
        if is_system {
            return Err(qwen35_err(refusal));
        }
        if count {
            *n += 1;
        }
        if add_vision_id {
            out.push_str(&format!("{label} {n}: "));
        }
        out.push_str(&format!("<|vision_start|>{pad}<|vision_end|>"));
    }
    Ok(out)
}

/// Jinja lines 57-87: the system block, with the tool list when `tools` is non-empty.
fn qwen35_system_block(
    first: &ThinkingMessage,
    instr: &str,
    tools: &[serde_json::Value],
    add_vision_id: bool,
) -> Result<String, AprenderError> {
    let c = if first.role == "system" {
        let mut zero = Qwen35VisionCount::default();
        qwen35_content(first, &mut zero, false, add_vision_id, true)?
    } else {
        String::new()
    };
    let c = c.trim();
    if !tools.is_empty() {
        let mut s = String::from("<|im_start|>system\n");
        if !instr.is_empty() {
            s.push_str(instr);
            s.push_str("\n\n");
        }
        s.push_str(QWEN35_TOOLS_HEAD);
        for tool in tools {
            s.push('\n');
            s.push_str(&qwen35_tojson(tool));
        }
        s.push_str("\n</tools>");
        s.push_str(QWEN35_TOOLS_FORMAT);
        if !c.is_empty() {
            s.push_str("\n\n");
            s.push_str(c);
        }
        s.push_str("<|im_end|>\n");
        return Ok(s);
    }
    Ok(match (c.is_empty(), instr.is_empty()) {
        (true, true) => String::new(),
        (true, false) => format!("<|im_start|>system\n{instr}<|im_end|>\n"),
        (false, true) => format!("<|im_start|>system\n{c}<|im_end|>\n"),
        (false, false) => format!("<|im_start|>system\n{instr}\n\n{c}<|im_end|>\n"),
    })
}

/// Jinja lines 91-96: a user turn that is not a wrapped tool response.
fn qwen35_is_query(m: &ThinkingMessage, add_vision_id: bool) -> Result<bool, AprenderError> {
    if m.role != "user" {
        return Ok(false);
    }
    let mut zero = Qwen35VisionCount::default();
    let c = qwen35_content(m, &mut zero, false, add_vision_id, false)?;
    let c = c.trim();
    Ok(!(c.starts_with("<tool_response>") && c.ends_with("</tool_response>")))
}

/// Jinja lines 88-101: index of the last real user query.
fn qwen35_last_query(
    messages: &[ThinkingMessage],
    add_vision_id: bool,
) -> Result<usize, AprenderError> {
    for (i, m) in messages.iter().enumerate().rev() {
        if qwen35_is_query(m, add_vision_id)? {
            return Ok(i);
        }
    }
    Err(qwen35_err("No user query found in messages."))
}

/// Jinja lines 123-147.
fn qwen35_tool_calls(out: &mut String, calls: &[ThinkingToolCall], has_content: bool) {
    for (k, call) in calls.iter().enumerate() {
        out.push_str(match (k, has_content) {
            (0, true) => "\n\n",
            (0, false) => "",
            _ => "\n",
        });
        out.push_str(&format!("<tool_call>\n<function={}>\n", call.name));
        for (name, value) in &call.arguments {
            let text = match value {
                serde_json::Value::String(s) => s.clone(),
                other => qwen35_tojson(other),
            };
            out.push_str(&format!("<parameter={name}>\n{text}\n</parameter>\n"));
        }
        out.push_str("</function>\n</tool_call>");
    }
}

/// Jinja lines 151-162: consecutive tool turns share one `user` block.
fn qwen35_tool_turn(out: &mut String, i: usize, messages: &[ThinkingMessage], content: &str) {
    if i > 0 && messages[i - 1].role != "tool" {
        out.push_str("<|im_start|>user");
    }
    out.push_str(&format!("\n<tool_response>\n{content}\n</tool_response>"));
    if messages.get(i + 1).map_or(true, |next| next.role != "tool") {
        out.push_str("<|im_end|>\n");
    }
}

fn qwen35_turn(
    out: &mut String,
    i: usize,
    messages: &[ThinkingMessage],
    keep_think: bool,
    seen: &mut Qwen35VisionCount,
    add_vision_id: bool,
) -> Result<(), AprenderError> {
    let m = &messages[i];
    let content = qwen35_content(m, seen, true, add_vision_id, false)?;
    let content = content.trim();
    match m.role.as_str() {
        "system" if i != 0 => return Err(qwen35_err("System message must be at the beginning.")),
        "system" => {}
        "user" => out.push_str(&format!("<|im_start|>user\n{content}<|im_end|>\n")),
        "assistant" => {
            out.push_str("<|im_start|>assistant\n");
            if keep_think {
                let r = m.reasoning_content.as_deref().unwrap_or("").trim();
                out.push_str(&format!("<think>\n{r}\n</think>\n\n"));
            }
            out.push_str(content);
            qwen35_tool_calls(out, &m.tool_calls, !content.is_empty());
            out.push_str("<|im_end|>\n");
        }
        "tool" => qwen35_tool_turn(out, i, messages, content),
        _ => return Err(qwen35_err("Unexpected message role.")),
    }
    Ok(())
}

impl Qwen35ThinkingTemplate {
    /// Render the conversation exactly as the published jinja does with no `tools`.
    /// `Err` mirrors the template's `raise_exception` calls.
    pub fn render(
        &self,
        messages: &[ThinkingMessage],
        opts: &Qwen35ThinkingOptions,
    ) -> Result<String, AprenderError> {
        self.render_with_tools(messages, &[], opts)
    }

    /// Render the conversation exactly as the published jinja does, `tools` being the
    /// template's `tools` variable (an empty slice takes the tool-free branch, line 76).
    pub fn render_with_tools(
        &self,
        messages: &[ThinkingMessage],
        tools: &[serde_json::Value],
        opts: &Qwen35ThinkingOptions,
    ) -> Result<String, AprenderError> {
        let first = messages
            .first()
            .ok_or_else(|| qwen35_err("No messages provided."))?;
        let vid = opts.add_vision_id;
        let mut out = qwen35_system_block(first, qwen35_instruction(opts), tools, vid)?;
        let last_query = qwen35_last_query(messages, vid)?;
        let mut seen = Qwen35VisionCount::default();
        for i in 0..messages.len() {
            let keep = opts.preserve_thinking || i > last_query;
            qwen35_turn(&mut out, i, messages, keep, &mut seen, vid)?;
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
