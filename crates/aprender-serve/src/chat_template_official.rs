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
    let arg_str = |i: usize| -> Option<String> {
        args.get(i).filter(|v| !v.is_none() && !v.is_undefined()).and_then(|v| v.as_str().map(str::to_string))
    };
    let affix_any = |f: &dyn Fn(&str) -> bool| -> Result<Value, Error> {
        // Python accepts a str or a tuple of strs.
        let a = args.first().ok_or_else(|| Error::new(ErrorKind::MissingArgument, method.to_string()))?;
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
    };
    let chars_of = |i: usize| arg_str(i).map(|c| c.chars().collect::<Vec<char>>());
    match method {
        "startswith" => affix_any(&|p| s.starts_with(p)),
        "endswith" => affix_any(&|p| s.ends_with(p)),
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
        "split" => {
            let maxsplit = args.get(1).and_then(|v| i64::try_from(v.clone()).ok()).unwrap_or(-1);
            let parts: Vec<Value> = match arg_str(0) {
                // Python: no separator splits on runs of whitespace and drops empties.
                None => s.split_whitespace().map(Value::from).collect(),
                Some(sep) if sep.is_empty() => {
                    return Err(Error::new(ErrorKind::InvalidOperation, "split: empty separator"));
                },
                Some(sep) if maxsplit >= 0 => s
                    .splitn(usize::try_from(maxsplit).unwrap_or(0) + 1, sep.as_str())
                    .map(Value::from)
                    .collect(),
                Some(sep) => s.split(sep.as_str()).map(Value::from).collect(),
            };
            Ok(Value::from(parts))
        },
        _ => Err(Error::new(
            ErrorKind::UnknownMethod,
            format!("str.{method} is not provided (#3990 implements only what the templates call)"),
        )),
    }
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
    let mut env = Environment::new();
    env.set_recursion_limit(MAX_RECURSION_DEPTH);
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_unknown_method_callback(official_template_str_method);
    env.add_function("raise_exception", |msg: String| -> Result<minijinja::Value, minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, format!("template raised: {msg}")))
    });
    env.add_template("chat", chat_template).map_err(|e| RealizarError::FormatError {
        reason: format!("model chat_template does not parse: {e}"),
    })?;
    let tmpl = env.get_template("chat").map_err(|e| RealizarError::FormatError {
        reason: format!("model chat_template: {e}"),
    })?;
    let mut ctx = std::collections::BTreeMap::<&str, minijinja::Value>::new();
    ctx.insert("messages", minijinja::Value::from_serialize(messages));
    ctx.insert("add_generation_prompt", minijinja::Value::from(add_generation_prompt));
    if let Some(b) = bos_token {
        ctx.insert("bos_token", minijinja::Value::from(b));
    }
    if let Some(e) = eos_token {
        ctx.insert("eos_token", minijinja::Value::from(e));
    }
    if let Some(t) = enable_thinking {
        ctx.insert("enable_thinking", minijinja::Value::from(t));
    }
    tmpl.render(minijinja::Value::from(ctx)).map_err(|e| RealizarError::FormatError {
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
    let Some(crate::gguf::GGUFValue::String(tpl)) = gguf.metadata.get("tokenizer.chat_template") else {
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
    render_official(tpl, bos.as_deref(), eos.as_deref(), messages, true, enable_thinking)
}
