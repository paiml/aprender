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
        let rendered = match enable_thinking {
            Some(on) => tmpl.render(context!(
                messages => messages,
                add_generation_prompt => add_generation_prompt,
                enable_thinking => on
            )),
            None => tmpl.render(context!(
                messages => messages,
                add_generation_prompt => add_generation_prompt
            )),
        };
        rendered.map_err(|e| RealizarError::FormatError {
            reason: format!("embedded chat template render: {e}"),
        })
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
