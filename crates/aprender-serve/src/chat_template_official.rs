// #3990: render the MODEL'S OWN chat template -- the GGUF's `tokenizer.chat_template` --
// instead of a hand-coded approximation chosen by name.
//
// The hand-coded path was measured wrong in two ways that change model behaviour:
//   * Qwen2.5 dropped the template's default system prompt ("You are Qwen, created by
//     Alibaba Cloud. You are a helpful assistant.") on a system-less request: 26 prompt
//     tokens against llama.cpp's 47, and apr 1/18 vs llama.cpp 10/18 on the same GGUF.
//   * Qwen3.5 thinking OFF prefilled `<think>\n</think>\n` where the template gives
//     `<think>\n\n</think>\n\n`, and thinking ON ended `assistant\n` where the template
//     OPENS the block, `assistant\n<think>\n`.
// Every thinking-ON measurement taken on the old prompt ran on a prompt the model was
// never trained on.
//
// FIDELITY, not approximation, is the contract: the rendered prompt's token ids must EQUAL
// llama.cpp's /apply-template ids. Three environment settings exist only for that:
//
//   trim_blocks + lstrip_blocks = true. HuggingFace renders with Jinja2(trim_blocks=True,
//     lstrip_blocks=True), and so does llama.cpp's template engine. minijinja defaults both
//     to FALSE, and every `{% ... %}` line then leaks a newline into the prompt.
//   Python string methods. Qwen3/3.5 templates call .startswith .endswith .split .strip
//     .lstrip .rstrip -- measured from the three templates, not guessed -- which minijinja
//     does not provide. minijinja-contrib's pycompat would, but it is neither in the lock
//     nor the offline cache, so exactly these six are implemented below.
//   raise_exception. Templates call it on invalid input; it must be an error, not a
//     silently rendered empty string.
//
// bos_token/eos_token are passed because templates reference them (TinyLlama appends
// eos_token to every turn). minijinja renders an UNDEFINED variable as an empty string
// with no error, which would be exactly this ticket's bug class, silently.
// enable_thinking: None leaves the variable UNDEFINED, so the template's own default
// applies -- as llama.cpp does when it is not passed.

/// The six Python `str` methods the Qwen2.5 / Qwen3 / Qwen3.5 templates call (#3990).
/// Anything else is an error rather than a guess.
fn official_template_str_method(
    _state: &minijinja::State,
    value: &minijinja::Value,
    method: &str,
    args: &[minijinja::Value],
) -> Result<minijinja::Value, minijinja::Error> {
    use minijinja::{Error, ErrorKind, Value};
    let Some(s) = value.as_str() else {
        return Err(Error::new(
            ErrorKind::UnknownMethod,
            format!("{method} is only provided on strings here"),
        ));
    };
    let chars_of = |i: usize| str_arg(args, i).map(|c| c.chars().collect::<Vec<char>>());
    match method {
        "startswith" => str_affix_any(method, args, &|p| s.starts_with(p)),
        "endswith" => str_affix_any(method, args, &|p| s.ends_with(p)),
        "strip" => Ok(Value::from(match chars_of(0) {
            Some(cs) => s.trim_matches(|c| cs.contains(&c)).to_string(),
            None => s.trim().to_string(),
        })),
        "lstrip" => Ok(Value::from(match chars_of(0) {
            Some(cs) => s.trim_start_matches(|c| cs.contains(&c)).to_string(),
            None => s.trim_start().to_string(),
        })),
        "rstrip" => Ok(Value::from(match chars_of(0) {
            Some(cs) => s.trim_end_matches(|c| cs.contains(&c)).to_string(),
            None => s.trim_end().to_string(),
        })),
        "split" => str_split(s, args),
        _ => Err(Error::new(
            ErrorKind::UnknownMethod,
            format!("str.{method} is not provided (#3990 implements only what the templates call)"),
        )),
    }
}

/// Positional argument `i` as a string; `None`/undefined count as absent (Python's default).
fn str_arg(args: &[minijinja::Value], i: usize) -> Option<String> {
    args.get(i)
        .filter(|v| !v.is_none() && !v.is_undefined())
        .and_then(|v| v.as_str().map(str::to_string))
}

/// `str.startswith` / `str.endswith`: Python accepts a str or a tuple of strs.
fn str_affix_any(
    method: &str,
    args: &[minijinja::Value],
    f: &dyn Fn(&str) -> bool,
) -> Result<minijinja::Value, minijinja::Error> {
    use minijinja::{Error, ErrorKind, Value};
    let a = args
        .first()
        .ok_or_else(|| Error::new(ErrorKind::MissingArgument, method.to_string()))?;
    if let Some(p) = a.as_str() {
        return Ok(Value::from(f(p)));
    }
    let mut any = false;
    for item in a.try_iter()? {
        if let Some(p) = item.as_str() {
            any |= f(p);
        }
    }
    Ok(Value::from(any))
}

/// `str.split(sep=None, maxsplit=-1)`.
fn str_split(s: &str, args: &[minijinja::Value]) -> Result<minijinja::Value, minijinja::Error> {
    use minijinja::{Error, ErrorKind, Value};
    let maxsplit = args
        .get(1)
        .and_then(|v| i64::try_from(v.clone()).ok())
        .unwrap_or(-1);
    let parts: Vec<Value> = match str_arg(args, 0) {
        // Python: no separator splits on runs of whitespace and drops empties.
        None => s.split_whitespace().map(Value::from).collect(),
        Some(sep) if sep.is_empty() => {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "split: empty separator",
            ));
        },
        Some(sep) if maxsplit >= 0 => s
            .splitn(usize::try_from(maxsplit).unwrap_or(0) + 1, sep.as_str())
            .map(Value::from)
            .collect(),
        Some(sep) => s.split(sep.as_str()).map(Value::from).collect(),
    };
    Ok(Value::from(parts))
}

/// `json.dumps` separators -- `", "` and `": "` -- for [`py_tojson`] (#4650).
struct PyJsonFormatter;

impl serde_json::ser::Formatter for PyJsonFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }
}

/// The `tojson` filter as HuggingFace defines it for chat templates (#4650):
/// `json.dumps(x, ensure_ascii=False)`. minijinja's own `tojson` writes compact separators
/// and HTML-escapes `<`, `>`, `&` and `'` to `<`-style escapes, so a tool schema that
/// says "a < b" reached the model as text it was never trained on. The `tools` block and
/// replayed tool-call arguments are the only places the Qwen templates call it.
///
/// `indent` is taken positionally or as a keyword, as HF and minijinja's built-in take it.
/// Llama 3.x templates render each tool with `tojson(indent=4)`; a filter that refused the
/// argument failed the whole render, and the request fell back to apr's built-in template.
/// With an indent the output is `json.dumps(x, indent=N)`: one item per line, `","`
/// between items and `": "` after a key.
fn py_tojson(
    value: &minijinja::Value,
    indent: Option<minijinja::Value>,
    kwargs: minijinja::value::Kwargs,
) -> Result<minijinja::Value, minijinja::Error> {
    use serde::Serialize;
    let indent = match indent {
        Some(i) => Some(i),
        None => kwargs.get::<Option<minijinja::Value>>("indent")?,
    };
    kwargs.assert_all_used()?;
    let bad = |e: serde_json::Error| {
        minijinja::Error::new(
            minijinja::ErrorKind::BadSerialization,
            format!("tojson: {e}"),
        )
    };
    let mut out = Vec::new();
    match indent.filter(|i| !i.is_none()) {
        None => {
            let mut ser = serde_json::Serializer::with_formatter(&mut out, PyJsonFormatter);
            value.serialize(&mut ser).map_err(bad)?;
        },
        Some(i) => {
            let n = i64::try_from(i).map_err(|_| {
                minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    "tojson: indent must be an integer",
                )
            })?;
            // json.dumps treats a negative indent as 0: newlines, no padding.
            let pad = " ".repeat(usize::try_from(n).unwrap_or(0));
            let fmt = serde_json::ser::PrettyFormatter::with_indent(pad.as_bytes());
            let mut ser = serde_json::Serializer::with_formatter(&mut out, fmt);
            value.serialize(&mut ser).map_err(bad)?;
        },
    }
    let s = String::from_utf8(out).map_err(|e| {
        minijinja::Error::new(minijinja::ErrorKind::BadSerialization, e.to_string())
    })?;
    Ok(minijinja::Value::from_safe_string(s))
}

/// Render a model's own jinja chat template, with llama.cpp/HuggingFace semantics (#3990).
///
/// # Errors
/// On a template that does not parse, a render error, or a `raise_exception` call.
pub fn render_official(
    chat_template: &str,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    enable_thinking: Option<bool>,
) -> Result<String, RealizarError> {
    render_official_with_tools(
        chat_template,
        bos_token,
        eos_token,
        messages,
        add_generation_prompt,
        enable_thinking,
        None,
    )
}

/// [`render_official`] with the request's OpenAI `tools` array (#4650).
///
/// Qwen2.5, Qwen3 and Qwen3.5 templates gate their whole `# Tools` system block on
/// `{% if tools %}`. The context used to carry no `tools` key, and minijinja renders an
/// UNDEFINED variable as false with no error -- so every tool a client sent was dropped
/// from the prompt, silently, and the model was never told a tool existed. `tools` is the
/// array as the client sent it (`[{"type":"function","function":{...}}]`), the shape
/// HuggingFace's `apply_chat_template(tools=...)` passes; `None` leaves it undefined.
///
/// # Errors
/// See [`render_official`].
pub fn render_official_with_tools(
    chat_template: &str,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    enable_thinking: Option<bool>,
    tools: Option<&serde_json::Value>,
) -> Result<String, RealizarError> {
    let mut env = Environment::new();
    env.set_recursion_limit(MAX_RECURSION_DEPTH);
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_unknown_method_callback(official_template_str_method);
    env.add_filter("tojson", py_tojson);
    env.add_function(
        "raise_exception",
        |msg: String| -> Result<minijinja::Value, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                format!("template raised: {msg}"),
            ))
        },
    );
    env.add_template("chat", chat_template)
        .map_err(|e| RealizarError::FormatError {
            reason: format!("model chat_template does not parse: {e}"),
        })?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| RealizarError::FormatError {
            reason: format!("model chat_template: {e}"),
        })?;
    let mut ctx = std::collections::BTreeMap::<&str, minijinja::Value>::new();
    ctx.insert("messages", minijinja::Value::from_serialize(messages));
    ctx.insert(
        "add_generation_prompt",
        minijinja::Value::from(add_generation_prompt),
    );
    if let Some(b) = bos_token {
        ctx.insert("bos_token", minijinja::Value::from(b));
    }
    if let Some(e) = eos_token {
        ctx.insert("eos_token", minijinja::Value::from(e));
    }
    if let Some(t) = enable_thinking {
        ctx.insert("enable_thinking", minijinja::Value::from(t));
    }
    if let Some(t) = tools {
        ctx.insert("tools", minijinja::Value::from_serialize(t));
    }
    tmpl.render(minijinja::Value::from(ctx))
        .map_err(|e| RealizarError::FormatError {
            reason: format!("model chat_template failed to render: {e}"),
        })
}

/// Render the GGUF's own `tokenizer.chat_template` for `messages`, with the generation
/// prompt appended (#3990). bos/eos are looked up as STRINGS through the vocabulary.
///
/// # Errors
/// If the GGUF carries no `tokenizer.chat_template` -- named, never a silent fallback to a
/// hand-coded template -- or if rendering fails.
pub fn render_official_for_model(
    gguf: &crate::gguf::GGUFModel,
    messages: &[ChatMessage],
    enable_thinking: Option<bool>,
) -> Result<String, RealizarError> {
    render_official_for_model_with_tools(gguf, messages, enable_thinking, None)
}

/// [`render_official_for_model`] with the request's `tools` (#4650); see
/// [`render_official_with_tools`].
///
/// # Errors
/// See [`render_official_for_model`].
pub fn render_official_for_model_with_tools(
    gguf: &crate::gguf::GGUFModel,
    messages: &[ChatMessage],
    enable_thinking: Option<bool>,
    tools: Option<&serde_json::Value>,
) -> Result<String, RealizarError> {
    let Some(crate::gguf::GGUFValue::String(tpl)) = gguf.metadata.get("tokenizer.chat_template")
    else {
        return Err(RealizarError::FormatError {
            reason: "this GGUF carries no tokenizer.chat_template; the official renderer has nothing to render (#3990)".to_string(),
        });
    };
    let vocab = gguf.vocabulary();
    let piece = |id: Option<u32>| -> Option<String> {
        let (v, i) = (vocab.as_ref()?, id?);
        v.get(usize::try_from(i).ok()?).cloned()
    };
    let (bos, eos) = (piece(gguf.bos_token_id()), piece(gguf.eos_token_id()));
    render_official_with_tools(
        tpl,
        bos.as_deref(),
        eos.as_deref(),
        messages,
        true,
        enable_thinking,
        tools,
    )
}

/// A `tokenizer_config.json` special token: a bare string, or an AddedToken object
/// (`{"content": "<s>", "lstrip": false, ...}`) -- both forms ship in the wild.
fn tokenizer_config_token(v: Option<&serde_json::Value>) -> Option<String> {
    match v? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(o) => o.get("content")?.as_str().map(str::to_string),
        _ => None,
    }
}

/// Render a SafeTensors model's OWN chat template, from its HuggingFace
/// `tokenizer_config.json`, with the generation prompt appended (#3990).
///
/// `chat_template` may be one string or a list of `{name, template}`; the list form takes
/// the entry named `default`, as transformers does.
///
/// # Errors
/// If the JSON does not parse, carries no usable `chat_template` (named, never a silent
/// fallback), or the template fails to render.
pub fn render_official_from_tokenizer_config(
    tokenizer_config_json: &str,
    messages: &[ChatMessage],
    enable_thinking: Option<bool>,
) -> Result<String, RealizarError> {
    let cfg: serde_json::Value =
        serde_json::from_str(tokenizer_config_json).map_err(|e| RealizarError::FormatError {
            reason: format!("tokenizer_config.json does not parse: {e}"),
        })?;
    let tpl = match cfg.get("chat_template") {
        Some(serde_json::Value::String(s)) => Some(s.as_str()),
        Some(serde_json::Value::Array(list)) => list
            .iter()
            .find(|t| t.get("name").and_then(serde_json::Value::as_str) == Some("default"))
            .and_then(|t| t.get("template")?.as_str()),
        _ => None,
    }
    .ok_or_else(|| RealizarError::FormatError {
        reason: "tokenizer_config.json carries no usable chat_template; the official renderer has nothing to render (#3990)".to_string(),
    })?;
    let bos = tokenizer_config_token(cfg.get("bos_token"));
    let eos = tokenizer_config_token(cfg.get("eos_token"));
    render_official(
        tpl,
        bos.as_deref(),
        eos.as_deref(),
        messages,
        true,
        enable_thinking,
    )
}
