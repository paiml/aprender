// ============================================================================
// The model's OWN chat template (`tokenizer.chat_template`), rendered as HF does
// ============================================================================
//
// #3755: apr used to RETYPE model templates by hand. The Qwen3 no-think scaffold it
// typed, `<think>\n</think>\n`, is not the model's (`<think>\n\n</think>\n\n`), and
// with it Qwen3.5-0.8B answers 2+2 = "2" (llama.cpp does too, given apr's string).
// A GGUF ships its template, so apr renders THAT, with the environment HF's
// `apply_chat_template` uses: trim_blocks + lstrip_blocks, Python string methods,
// loop controls, `raise_exception`, `strftime_now`, and a `tojson` that writes what
// Python's `json.dumps` writes. The byte-equality oracle is
// `chat_template_embedded_oracle` (fixtures rendered by transformers).

/// Which thinking modes a model's own chat template offers (#3723).
///
/// Derived by RENDERING the template with thinking on and off, never by searching it
/// for markers: Qwen3-30B-A3B-Instruct-2507 carries `<think>` in its template (to strip
/// reasoning from history) and has no thinking mode (measured by aprender-62, #3723).
/// `apr run/chat/code --thinking` and the release gate (#3712) call this one function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingModes {
    /// `enable_thinking` changes the prompt: thinking can be switched on and off.
    Both,
    /// The generation prompt always opens a think block: thinking is always on.
    OnOnly,
    /// No thinking mode: thinking is always off.
    OffOnly,
}

impl ThinkingModes {
    /// Whether a request for thinking `on` (true) or off (false) can be honoured.
    #[must_use]
    pub fn allows(self, on: bool) -> bool {
        match self {
            Self::Both => true,
            Self::OnOnly => on,
            Self::OffOnly => !on,
        }
    }

    /// The mode a request that does not choose gets: OFF wherever the model allows it
    /// (today's production behaviour), and the model's only mode otherwise.
    #[must_use]
    pub fn default_on(self) -> bool {
        self == Self::OnOnly
    }

    /// Resolve a request (`None` = no choice) to a mode, or refuse it by name.
    ///
    /// # Errors
    ///
    /// [`RealizarError::ThinkingModeUnsupported`] when the template cannot honour it.
    pub fn resolve(self, requested: Option<bool>) -> Result<bool, RealizarError> {
        match requested {
            None => Ok(self.default_on()),
            Some(on) if self.allows(on) => Ok(on),
            Some(on) => Err(RealizarError::ThinkingModeUnsupported {
                requested: if on { "on" } else { "off" }.to_string(),
                supported: self.to_string(),
            }),
        }
    }
}

impl std::fmt::Display for ThinkingModes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Both => "on and off",
            Self::OnOnly => "on only",
            Self::OffOnly => "off only",
        })
    }
}

/// Render steps one template render may take. A GGUF's template is untrusted input
/// (models are downloaded from the internet), so a looping template is REFUSED by name
/// rather than hanging the server. The inventory's largest template (Qwen3.5, 7.8 KB)
/// renders a three-turn conversation well inside this.
pub const EMBEDDED_TEMPLATE_FUEL: u64 = 5_000_000;

/// Bytes one rendered prompt may reach (about a million tokens, past any context
/// window). Enforced while writing, so output built up by a loop is refused before it
/// grows further. Residual, as in jinja2's sandbox: one expression such as
/// `'x' * 10**12` allocates eagerly before anything is written.
pub const EMBEDDED_TEMPLATE_MAX_OUTPUT: usize = 4 * 1024 * 1024;

/// A model's own chat template, rendered the way HF `apply_chat_template` renders it.
pub struct EmbeddedChatTemplate {
    env: Environment<'static>,
    thinking: ThinkingModes,
}

impl std::fmt::Debug for EmbeddedChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddedChatTemplate")
            .field("thinking", &self.thinking)
            .finish_non_exhaustive()
    }
}

impl EmbeddedChatTemplate {
    /// Compile a template and derive its thinking modes.
    ///
    /// # Errors
    ///
    /// [`RealizarError::FormatError`] when the template does not compile, or cannot
    /// render a one-message conversation (so it fails at load, not per request).
    pub fn new(source: impl Into<String>) -> Result<Self, RealizarError> {
        let mut env = Environment::new();
        env.set_recursion_limit(MAX_RECURSION_DEPTH);
        env.set_fuel(Some(EMBEDDED_TEMPLATE_FUEL));
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        env.add_filter("tojson", python_tojson);
        env.add_template_owned("chat", source.into())
            .map_err(|e| RealizarError::FormatError {
                reason: format!("embedded chat template does not compile: {e}"),
            })?;
        let mut template = Self {
            env,
            thinking: ThinkingModes::OffOnly,
        };
        let probe = [ChatMessage::user("What is 2+2?")];
        let on = template.render(&probe, true, Some(true))?;
        let off = template.render(&probe, true, Some(false))?;
        template.thinking = if on != off {
            ThinkingModes::Both
        } else if opens_think_block(&on) {
            ThinkingModes::OnOnly
        } else {
            ThinkingModes::OffOnly
        };
        Ok(template)
    }

    /// The model's template from a GGUF's `tokenizer.chat_template`, if it has one.
    #[must_use]
    pub fn from_gguf(model: &crate::gguf::GGUFModel) -> Option<Result<Self, RealizarError>> {
        match model.metadata.get("tokenizer.chat_template") {
            Some(crate::gguf::GGUFValue::String(s)) if !s.is_empty() => Some(Self::new(s.clone())),
            _ => None,
        }
    }

    /// The thinking modes this template offers.
    #[must_use]
    pub fn thinking_modes(&self) -> ThinkingModes {
        self.thinking
    }

    /// Render a conversation.
    ///
    /// `enable_thinking: None` leaves the variable undefined, which is what HF does when
    /// the caller passes no such kwarg; production always passes the resolved mode.
    /// Message content is sanitized first (F-SEC-220), as in every apr template.
    ///
    /// # Errors
    ///
    /// [`RealizarError::FormatError`] when rendering fails (e.g. `raise_exception`).
    pub fn render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> Result<String, RealizarError> {
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| RealizarError::FormatError {
                reason: format!("embedded chat template: {e}"),
            })?;
        let messages: Vec<ChatMessage> = messages
            .iter()
            .map(|m| ChatMessage::new(&m.role, sanitize_special_tokens(&m.content)))
            .collect();
        let ctx = match enable_thinking {
            Some(on) => context!(
                messages => messages,
                add_generation_prompt => add_generation_prompt,
                enable_thinking => on
            ),
            None => context!(
                messages => messages,
                add_generation_prompt => add_generation_prompt
            ),
        };
        let mut out = CappedOutput::default();
        if let Err(e) = tmpl.render_captured_to(ctx, &mut out) {
            let reason = if out.overflowed {
                format!(
                    "embedded chat template refused: its output passed the \
                     {EMBEDDED_TEMPLATE_MAX_OUTPUT}-byte limit"
                )
            } else if e.kind() == minijinja::ErrorKind::OutOfFuel {
                format!(
                    "embedded chat template refused: it ran past the \
                     {EMBEDDED_TEMPLATE_FUEL}-step render limit"
                )
            } else {
                format!("embedded chat template render: {e}")
            };
            return Err(RealizarError::FormatError { reason });
        }
        String::from_utf8(out.bytes).map_err(|e| RealizarError::FormatError {
            reason: format!("embedded chat template rendered invalid UTF-8: {e}"),
        })
    }
}

/// A render sink that refuses to grow past [`EMBEDDED_TEMPLATE_MAX_OUTPUT`].
#[derive(Default)]
struct CappedOutput {
    bytes: Vec<u8>,
    overflowed: bool,
}

impl std::io::Write for CappedOutput {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.bytes.len() + buf.len() > EMBEDDED_TEMPLATE_MAX_OUTPUT {
            self.overflowed = true;
            return Err(std::io::Error::other("rendered prompt too large"));
        }
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Whether a rendered prompt ends inside an open `<think>` block.
fn opens_think_block(prompt: &str) -> bool {
    prompt
        .rfind("<think>")
        .is_some_and(|open| !prompt[open..].contains("</think>"))
}

/// HF's `raise_exception(message)`: fail the render with the template's message.
fn raise_exception(message: String) -> Result<minijinja::Value, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        message,
    ))
}

/// HF's `strftime_now(format)`: local time, formatted.
fn strftime_now(format: String) -> String {
    chrono::Local::now().format(&format).to_string()
}

/// HF's `tojson`: Python `json.dumps(x, ensure_ascii=False, indent=indent)`, i.e.
/// `", "` / `": "` separators on one line, `","` + newlines when indented, keys in
/// insertion order, non-ASCII kept. minijinja's own `tojson` writes compact JSON.
fn python_tojson(
    value: minijinja::Value,
    kwargs: minijinja::value::Kwargs,
) -> Result<String, minijinja::Error> {
    let indent: Option<usize> = kwargs.get("indent")?;
    kwargs.assert_all_used()?;
    let mut out = String::new();
    write_py_json(&value, indent, 0, &mut out)?;
    Ok(out)
}

fn write_py_json(
    value: &minijinja::Value,
    indent: Option<usize>,
    depth: usize,
    out: &mut String,
) -> Result<(), minijinja::Error> {
    use minijinja::value::ValueKind;
    use std::fmt::Write;
    let newline = |out: &mut String, depth: usize| {
        if let Some(n) = indent {
            out.push('\n');
            out.push_str(&" ".repeat(n * depth));
        }
    };
    let item_sep = if indent.is_some() { "," } else { ", " };
    match value.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if value.is_true() { "true" } else { "false" }),
        ValueKind::Number if value.is_integer() => {
            let _ = write!(out, "{value}");
        }
        ValueKind::Number => {
            // Python's float repr keeps a ".0" on integral floats (1.0, not 1).
            let f = f64::try_from(value.clone()).unwrap_or(f64::NAN);
            if f.is_finite() && f.fract() == 0.0 && f.abs() < 1e16 {
                let _ = write!(out, "{f:.1}");
            } else {
                let _ = write!(out, "{f}");
            }
        }
        ValueKind::String => write_py_json_str(value.as_str().unwrap_or_default(), out),
        ValueKind::Map => {
            let keys: Vec<minijinja::Value> = value.try_iter()?.collect();
            if keys.is_empty() {
                out.push_str("{}");
                return Ok(());
            }
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                newline(out, depth + 1);
                write_py_json_str(&key.to_string(), out);
                out.push_str(": ");
                write_py_json(&value.get_item(key)?, indent, depth + 1, out)?;
            }
            newline(out, depth);
            out.push('}');
        }
        _ => {
            let items: Vec<minijinja::Value> = value.try_iter()?.collect();
            if items.is_empty() {
                out.push_str("[]");
                return Ok(());
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                newline(out, depth + 1);
                write_py_json(item, indent, depth + 1, out)?;
            }
            newline(out, depth);
            out.push(']');
        }
    }
    Ok(())
}

/// A JSON string as Python's `json.dumps(..., ensure_ascii=False)` writes it.
fn write_py_json_str(s: &str, out: &mut String) {
    use std::fmt::Write;
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A prompt as production sends it, and the thinking mode it was built for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatPrompt {
    /// The formatted prompt text.
    pub text: String,
    /// Whether the model was asked to think.
    pub thinking: bool,
}

/// THE prompt for a conversation, as every apr verb sends it (#3755, #3723).
///
/// A model that ships a chat template gets that template, rendered with the thinking
/// mode resolved against what the template offers: `thinking: None` is OFF wherever the
/// model allows it, and a mode the template cannot honour is REFUSED. A file with no
/// template falls back to apr's per-family template (`format_messages`), which has no
/// thinking switch, so asking it for thinking ON is refused.
///
/// # Errors
///
/// [`RealizarError::ThinkingModeUnsupported`] for a mode the model cannot honour, or
/// the template's render error.
pub fn format_chat_prompt(
    embedded: Option<&EmbeddedChatTemplate>,
    model_name: Option<&str>,
    messages: &[ChatMessage],
    thinking: Option<bool>,
) -> Result<ChatPrompt, RealizarError> {
    if let Some(template) = embedded {
        let on = template.thinking_modes().resolve(thinking)?;
        return Ok(ChatPrompt {
            text: template.render(messages, true, Some(on))?,
            thinking: on,
        });
    }
    let on = ThinkingModes::OffOnly.resolve(thinking)?;
    Ok(ChatPrompt {
        text: format_messages(messages, model_name)?,
        thinking: on,
    })
}

/// A completion split into the model's reasoning and its answer (#3723).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitCompletion {
    /// The think block's content, trimmed, when the completion reasoned.
    pub reasoning: Option<String>,
    /// Everything after the think block, trimmed.
    pub answer: String,
}

impl ChatPrompt {
    /// Whether this prompt itself leaves a `<think>` block open (Qwen3.5 thinking ON),
    /// so the completion starts inside the reasoning.
    #[must_use]
    pub fn opens_think_block(&self) -> bool {
        opens_think_block(&self.text)
    }

    /// Split a completion of this prompt into reasoning and answer.
    ///
    /// The reasoning runs to `</think>`, from the block this prompt opened or from a
    /// `<think>` the completion itself starts with (Qwen3). A block still open at the
    /// end is an ERROR naming the `budget`, never an empty answer (#3720, #3723). A
    /// completion with no reasoning is all answer.
    ///
    /// # Errors
    ///
    /// [`RealizarError::InferenceError`] when a think block is still open at the end.
    pub fn split(&self, completion: &str, budget: usize) -> Result<SplitCompletion, RealizarError> {
        let reasoning = if self.opens_think_block() {
            Some(completion)
        } else {
            completion.trim_start().strip_prefix("<think>")
        };
        let Some(reasoning) = reasoning else {
            return Ok(SplitCompletion {
                reasoning: None,
                answer: completion.trim().to_string(),
            });
        };
        match reasoning.find("</think>") {
            Some(close) => Ok(SplitCompletion {
                reasoning: Some(reasoning[..close].trim().to_string()),
                answer: reasoning[close + "</think>".len()..].trim().to_string(),
            }),
            None => Err(RealizarError::InferenceError(format!(
                "think block unclosed within the {budget}-token budget: the model was still \
                 reasoning when generation stopped"
            ))),
        }
    }
}
