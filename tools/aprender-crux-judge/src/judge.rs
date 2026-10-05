//! The judge core of `crux_inference_judge.py`: the engine-output parsers,
//! one engine's entry for a cell, the cell verdict, and the deterministic
//! (tok/tmpl) and greedy row reports. `collect` and `render_md` live in
//! `collect.rs`.
//!
//! Every record the receipt carries is built as a `Val::Dict`, in the key
//! order the Python literal writes, so `json.dump` order is preserved.

use std::collections::HashMap;

use crate::oracles;
use crate::pyerr::{PyErr, PyResult};
use crate::pyio::{decode_lossy, load_bytes, path_arg, read_text, read_text_strict};
use crate::pyjson::{dumps, loads, raw_decode};
use crate::pyre::{
    ansi, assistant_turn, debug_escape, encoded_ids, formatted_prompt, leading_num, ollama_stat,
    rate_line, think_either, you_prompt,
};
use crate::pyval::{
    concat_str, dict, is_py_space, py_eq, py_lower, py_lstrip, py_ne, py_repr, py_sorted,
    py_sorted_by, py_str, py_strip, repr_str, slice_chars, Dict, PyInt, Val,
};

pub const COMPARATORS: [&str; 5] = ["llama.cpp", "ollama", "hf", "llamafile", "vllm"];
pub const PLUGIN_ENGINES: [&str; 3] = ["hf", "llamafile", "vllm"];
pub const ENGINES: [&str; 6] = ["apr", "llama.cpp", "ollama", "hf", "llamafile", "vllm"];

/// Text an engine's `why` may carry -> its NOT_RAN reason; the first match wins.
const NOT_RAN_PATTERNS: [(&str, &str); 15] = [
    ("not requested", "not_requested"),
    ("command not found", "not_on_PATH"),
    ("not found on path", "not_on_PATH"),
    ("not on path", "not_on_PATH"),
    ("no such file or directory", "binary_not_found_at_path"),
    ("does not exist", "binary_not_found_at_path"),
    ("not installed", "not_installed"),
    ("model not pulled", "model_not_pulled"),
    ("no such model", "model_not_pulled"),
    ("refused", "refused"),
    ("timed out", "timed_out"),
    ("timeout", "timed_out"),
    ("killed", "crashed"),
    ("crashed", "crashed"),
    ("exited", "crashed"),
];

const RATE_UNIT: &str = "tokens per second, as the engine reported it";

/// #3962 B2: llama-cli's thinking markers -> the tags the oracle reads.
const LLAMA_CLI_THINK: [(&str, &str); 2] = [
    ("[Start thinking]", "<think>"),
    ("[End thinking]", "</think>"),
];

/// realizar logs `formatted_prompt` from its first 200 bytes; a cut floored to
/// a char boundary can be as short as 197.
const FORMATTED_PROMPT_WHOLE_BELOW: usize = 197;

pub const V1_WHY: &str =
    "v1 prompt: `expect_any` is a substring oracle, not a constrained <answer> -- \
\"not Paris\" contains \"Paris\" (#3957 Q3); certify a v2 prompt (#3962)";

pub const CELL_QUORUM_FLOOR: i64 = 2;

const THINK_UNKNOWN_WHY: &str = "thinking ON, and nothing shows whether the prompt opened a think block \
(apr's rendering is not visible and no reference tmpl row covers it), while the reply carries no think tag -- \
the reasoning cannot be told from the answer (#3962 B2)";

// ---------------------------------------------------------------- helpers --

fn is(v: &Val, lit: &str) -> bool {
    v.as_str() == Some(lit)
}

fn one_of(v: &Val, set: &[&str]) -> bool {
    v.as_str().is_some_and(|x| set.contains(&x))
}

fn empty() -> Val {
    Val::Dict(Dict::new())
}

fn opt_idx(i: Option<usize>) -> Val {
    i.map_or(Val::None, |i| Val::int(i as i64))
}

fn family(engine: &str) -> Option<&'static str> {
    match engine {
        "llama.cpp" | "ollama" | "llamafile" => Some("ggml"),
        "hf" | "vllm" => Some("bf16"),
        _ => None,
    }
}

/// `SAME_REP.get(fmt)`: TypeError for an unhashable `fmt`, as dict.get raises.
fn same_rep(fmt: &Val) -> PyResult<Option<&'static str>> {
    match fmt {
        Val::List(_) | Val::Dict(_) => Err(PyErr::type_err(format!(
            "unhashable type: '{}'",
            fmt.type_name()
        ))),
        Val::Str(s) if s == "gguf" => Ok(Some("ggml")),
        Val::Str(s) if s == "safetensors" => Ok(Some("bf16")),
        _ => Ok(None),
    }
}

/// `float(s)` for a string the judge matched as `[0-9.]+`.
fn py_float(s: &str) -> PyResult<f64> {
    s.parse::<f64>().map_err(|_| {
        PyErr::value(format!(
            "could not convert string to float: {}",
            repr_str(s)
        ))
    })
}

/// `x + suffix` where `x` should be a str: CPython's TypeError otherwise.
pub fn add_str(x: &Val, suffix: &str) -> PyResult<String> {
    match x {
        Val::Str(s) => Ok(format!("{s}{suffix}")),
        Val::List(_) | Val::Tuple(_) => Err(PyErr::type_err(format!(
            "can only concatenate {0} (not \"str\") to {0}",
            x.type_name()
        ))),
        other => Err(PyErr::type_err(format!(
            "unsupported operand type(s) for +: '{}' and 'str'",
            other.type_name()
        ))),
    }
}

// -------------------------------------------------------------- not-ran --

/// `classify_not_ran(why)`.
pub fn classify_not_ran(why: &Val) -> &'static str {
    if !why.truthy() {
        return "unclassified";
    }
    let low = py_lower(&py_str(why));
    NOT_RAN_PATTERNS
        .iter()
        .find(|(needle, _)| low.contains(needle))
        .map_or("unclassified", |(_, reason)| reason)
}

/// `engine_versions(meta)`, in the Python dict's key order.
pub fn engine_versions(meta: &Val) -> PyResult<Dict> {
    let ol = meta.get("ollama")?.or(empty());
    let lc = meta.get("llama_cpp")?.or(empty());
    let mut out = Dict::new();
    out.put("apr", meta.get("apr")?.or(empty()).get("version_line")?);
    out.put("ollama", ol.get("server_version")?);
    out.put("llama.cpp", lc.get("build")?);
    for eng in PLUGIN_ENGINES {
        let m = meta.get(eng)?.or(empty());
        let v = m.get("version")?;
        let v = if v.truthy() { v } else { m.get("probe")? };
        out.put(eng, v);
    }
    Ok(out)
}

// ---------------------------------------------------------------- parsers --
// Each returns {"answer", "why", "reported", ...}: answer None means "did not
// answer" and `why` says what the judge saw instead.

fn parser_out() -> Dict {
    Dict::from_pairs(vec![
        ("answer", Val::None),
        ("why", Val::None),
        ("reported", empty()),
    ])
}

/// `parse_apr(stdout, stderr)`.
pub fn parse_apr(stdout: &str, stderr: &str) -> PyResult<Dict> {
    let mut out = Dict::from_pairs(vec![
        ("answer", Val::None),
        ("why", Val::None),
        ("reported", empty()),
        ("backend", Val::None),
        ("prompt_ids", Val::None),
        ("prompt_token_count", Val::None),
        ("rendered_prompt", Val::None),
    ]);
    if let Some(c) = formatted_prompt().captures(stderr) {
        out.put("rendered_prompt", Val::str(&c[1]));
    }
    if let Some(c) = encoded_ids().captures(stderr) {
        out.put("prompt_token_count", Val::Int(PyInt::parse_digits(&c[1])?));
        let ids = c[2]
            .replace(' ', "")
            .split(',')
            .filter(|x| !x.is_empty())
            .map(|x| PyInt::parse_digits(x).map(Val::Int))
            .collect::<PyResult<Vec<_>>>()?;
        out.put("prompt_ids", Val::List(ids));
    }
    let Some(start) = stdout.find('{') else {
        out.put("why", Val::str("no JSON object on stdout"));
        return Ok(out);
    };
    let doc = match raw_decode(&stdout[start..]) {
        Ok((doc, _)) => doc,
        Err(e) if e.is_value() => {
            out.put("why", Val::Str(format!("stdout JSON unparseable: {e}")));
            return Ok(out);
        }
        Err(e) => return Err(e),
    };
    out.put("backend", doc.get("backend")?);
    let reported = dict(vec![
        ("reported_by", Val::str("apr")),
        (
            "prompt_tokens",
            out.get_str("prompt_token_count")
                .cloned()
                .unwrap_or(Val::None),
        ),
        ("completion_tokens", doc.get("tokens_generated")?),
        ("decode_rate", doc.get("tok_per_sec")?),
        ("rate_unit", Val::str(RATE_UNIT)),
        ("inference_ms", doc.get("inference_time_ms")?),
    ]);
    out.put("reported", reported);
    match doc.get("text")? {
        Val::Str(text) => out.put("answer", Val::Str(text)),
        _ => out.put("why", Val::str("stdout JSON has no text field")),
    }
    Ok(out)
}

/// `parse_llamacpp_cli(stdout, prompt_text)`.
pub fn parse_llamacpp_cli(stdout: &str, prompt_text: &Val) -> PyResult<Dict> {
    let mut out = parser_out();
    let text = ansi().replace_all(stdout, "");
    let echo = concat_str("> ", prompt_text)?;
    let Some(at) = text.rfind(&echo) else {
        out.put("why", Val::str("the echoed prompt was not found in stdout"));
        return Ok(out);
    };
    let rest = &text[at + echo.len()..];
    let Some(m) = rate_line().captures(rest) else {
        out.put(
            "why",
            Val::str("no end-of-turn timing line after the echoed prompt"),
        );
        return Ok(out);
    };
    let whole = m.get(0).expect("group 0");
    let mut ans = py_strip(&rest[..whole.start()]).to_string();
    for (marker, tag) in LLAMA_CLI_THINK {
        ans = ans.replace(marker, tag);
    }
    out.put("answer", Val::Str(ans));
    let prompt_rate = py_float(&m[1])?;
    let decode_rate = py_float(&m[2])?;
    out.put(
        "reported",
        dict(vec![
            ("reported_by", Val::str("llama.cpp")),
            ("prompt_rate", Val::Float(prompt_rate)),
            ("decode_rate", Val::Float(decode_rate)),
            ("rate_unit", Val::str(RATE_UNIT)),
            ("prompt_tokens", Val::None),
            ("completion_tokens", Val::None),
        ]),
    );
    Ok(out)
}

/// `parse_ollama(stdout, stderr)`.
pub fn parse_ollama(stdout: &str, stderr: &str) -> PyResult<Dict> {
    let mut out = parser_out();
    let clean = ansi().replace_all(stderr, "");
    let mut stats: HashMap<String, String> = HashMap::new();
    for c in ollama_stat().captures_iter(&clean) {
        stats.insert(c[1].to_string(), c[2].to_string());
    }
    let num = |key: &str| -> PyResult<Option<f64>> {
        let v = stats.get(key).map_or("", String::as_str);
        match leading_num().captures(v) {
            Some(c) => Ok(Some(py_float(&c[1])?)),
            None => Ok(None),
        }
    };
    let stat = |key: &str| stats.get(key).map_or(Val::None, |v| Val::str(v.as_str()));
    let opt_float = |f: Option<f64>| f.map_or(Val::None, Val::Float);
    let as_int = |f: Option<f64>| -> PyResult<Val> {
        Ok(match f {
            Some(f) => Val::Int(PyInt::from_f64_trunc(f)?),
            None => Val::None,
        })
    };
    let pe = num("prompt eval count")?;
    let ev = num("eval count")?;
    let prompt_tokens = as_int(pe)?;
    let completion_tokens = as_int(ev)?;
    let prompt_rate = opt_float(num("prompt eval rate")?);
    let decode_rate = opt_float(num("eval rate")?);
    out.put(
        "reported",
        dict(vec![
            ("reported_by", Val::str("ollama")),
            ("prompt_tokens", prompt_tokens),
            ("completion_tokens", completion_tokens),
            ("prompt_rate", prompt_rate),
            ("decode_rate", decode_rate),
            ("rate_unit", Val::str(RATE_UNIT)),
            ("load_duration", stat("load duration")),
            ("total_duration", stat("total duration")),
        ]),
    );
    let answer = ansi().replace_all(stdout, "");
    let answer = py_strip(&answer);
    if answer.is_empty() {
        out.put("why", Val::str("empty stdout"));
        return Ok(out);
    }
    out.put("answer", Val::str(answer));
    Ok(out)
}

/// `parse_apr_chat(stdout)`: every `Assistant: ` reply up to the next `You:`.
pub fn parse_apr_chat(stdout: &str) -> Dict {
    let mut out = Dict::from_pairs(vec![
        ("answer", Val::None),
        ("why", Val::None),
        ("reported", dict(vec![("reported_by", Val::str("apr"))])),
        ("turns", Val::List(vec![])),
    ]);
    let text = ansi().replace_all(stdout, "");
    let turns: Vec<Val> = assistant_turn()
        .split(&text)
        .skip(1)
        .map(|seg| {
            let body = you_prompt().find(seg).map_or(seg, |m| &seg[..m.start()]);
            Val::str(py_strip(body))
        })
        .collect();
    let Some(last) = turns.last().cloned() else {
        out.put("why", Val::str("no `Assistant:` turn in the transcript"));
        return out;
    };
    out.put("turns", Val::List(turns));
    out.put("answer", last);
    out
}

/// `parse_engine_json(stdout)`: the row-contract v1 JSON of a plugin driver,
/// the OpenAI client, or the pty helper.
pub fn parse_engine_json(stdout: &str) -> PyResult<Dict> {
    let mut out = parser_out();
    let mut doc = match loads(stdout) {
        Ok(d) => d,
        Err(e) if e.is_value() => {
            out.put(
                "why",
                Val::Str(format!("stdout is not the contract's JSON: {e}")),
            );
            return Ok(out);
        }
        Err(e) => return Err(e),
    };
    if let Val::Dict(d) = &doc {
        if let Some(pf) = d.get_str("protocol_fault").filter(|v| v.truthy()) {
            out.put("why", Val::Str(format!("protocol fault: {}", py_str(pf))));
            return Ok(out);
        }
    }
    if let Val::Dict(d) = &mut doc {
        if let Some(Val::Str(reasoning)) = d.get_str("reasoning").cloned() {
            if !reasoning.is_empty() {
                let t = d
                    .get_str("text")
                    .cloned()
                    .unwrap_or(Val::None)
                    .or(Val::str(""));
                let tail = if t.truthy() {
                    concat_str("</think>", &t)?
                } else {
                    String::new()
                };
                d.put("text", Val::Str(format!("<think>{reasoning}{tail}")));
            }
        }
    }
    let Val::Dict(d) = &doc else {
        out.put("why", Val::str("stdout JSON has no text field"));
        return Ok(out);
    };
    let Some(Val::Str(text)) = d.get_str("text") else {
        out.put("why", Val::str("stdout JSON has no text field"));
        return Ok(out);
    };
    let rep = match d.get_str("reported") {
        Some(r @ Val::Dict(_)) => r.clone(),
        _ => empty(),
    };
    out.put("reported", rep);
    out.put("answer", Val::str(text.as_str()));
    if let Some(t @ Val::List(_)) = d.get_str("turns") {
        out.put("turns", t.clone());
    }
    Ok(out)
}

// ------------------------------------------------------ think detection --

/// `undebug(s)`: undo Rust's `{:?}` escaping, as code points (a `\u{d800}`
/// escape yields a lone surrogate, which a Rust `String` cannot hold).
fn undebug(s: &str) -> PyResult<Vec<u32>> {
    let mut out: Vec<u32> = Vec::with_capacity(s.len());
    let mut last = 0;
    for c in debug_escape().captures_iter(s) {
        let m = c.get(0).expect("group 0");
        out.extend(s[last..m.start()].chars().map(u32::from));
        last = m.end();
        if let Some(hex) = c.get(2) {
            let cp = u32::from_str_radix(hex.as_str(), 16).expect("1-6 hex digits");
            if cp > 0x10FFFF {
                return Err(PyErr::value("chr() arg not in range(0x110000)"));
            }
            out.push(cp);
            continue;
        }
        let rep = match &c[1] {
            "n" => "\n",
            "t" => "\t",
            "r" => "\r",
            "0" => "\0",
            "\\" => "\\",
            "\"" => "\"",
            "'" => "'",
            _ => m.as_str(),
        };
        out.extend(rep.chars().map(u32::from));
    }
    out.extend(s[last..].chars().map(u32::from));
    Ok(out)
}

fn is_surrogate(cp: u32) -> bool {
    (0xD800..=0xDFFF).contains(&cp)
}

/// `str.encode("utf-8")`'s byte length, or the UnicodeEncodeError CPython
/// raises at the first run of surrogates.
fn utf8_len(cps: &[u32]) -> PyResult<usize> {
    let mut n = 0;
    let mut i = 0;
    while i < cps.len() {
        let cp = cps[i];
        if is_surrogate(cp) {
            let mut end = i + 1;
            while end < cps.len() && is_surrogate(cps[end]) {
                end += 1;
            }
            let msg = if end == i + 1 {
                format!(
                    "'utf-8' codec can't encode character '\\u{cp:04x}' in position {i}: surrogates not allowed"
                )
            } else {
                format!(
                    "'utf-8' codec can't encode characters in position {i}-{}: surrogates not allowed",
                    end - 1
                )
            };
            return Err(PyErr::new("UnicodeEncodeError", msg));
        }
        n += match cp {
            0..=0x7F => 1,
            0x80..=0x7FF => 2,
            0x800..=0xFFFF => 3,
            _ => 4,
        };
        i += 1;
    }
    Ok(n)
}

/// `rendered_opens_think(rendered)` -> True / False / None.
pub fn rendered_opens_think(rendered: &Val) -> PyResult<Val> {
    let Val::Str(s) = rendered else {
        return Ok(Val::None);
    };
    let raw = undebug(s)?;
    let tail: Vec<u32> = "<think>\n".chars().map(u32::from).collect();
    if raw.ends_with(&tail) {
        return Ok(Val::Bool(true));
    }
    Ok(if utf8_len(&raw)? < FORMATTED_PROMPT_WHOLE_BELOW {
        Val::Bool(false)
    } else {
        Val::None
    })
}

/// Whether `prompt_opens_think` reads row `r`: a thinking-on tmpl row, not
/// from apr, not refused.
fn opens_think_candidate(r: &Val) -> PyResult<bool> {
    Ok(is(&r.get("kind")?, "tmpl")
        && !is(&r.get("engine")?, "apr")
        && is(&r.get("thinking")?, "on")
        && !r.get("refused")?.truthy())
}

/// Row `r`'s rendered prompt: `None` when it cannot be read (no key, a
/// wrong type, an OS error), which the source skips.
fn rendered_bytes(r: &Val) -> PyResult<Option<Vec<u8>>> {
    match r.item("rendered").and_then(|p| load_bytes(&p)) {
        Ok(b) => Ok(Some(b)),
        Err(e) if e.is_os() || e.is_key() || e.is_type() => Ok(None),
        Err(e) => Err(e),
    }
}

/// `prompt_opens_think(rows)`: {(model_sha256, prompt_id): bool}, keyed by a
/// tuple.
pub fn prompt_opens_think(rows: &[Val]) -> PyResult<Dict> {
    let mut out = Dict::new();
    for r in rows {
        if !opens_think_candidate(r)? {
            continue;
        }
        let Some(bytes) = rendered_bytes(r)? else {
            continue;
        };
        let opens = decode_lossy(&bytes).ends_with("<think>\n");
        let key = Val::Tuple(vec![r.get("model_sha256")?, r.get("prompt_id")?]);
        out.set(key, Val::Bool(opens))?;
    }
    Ok(out)
}

// ------------------------------------------------------------- degenerate --

/// `degenerate(text)`: one non-space character >= 90% of at least 8, or a
/// multi-byte token loop.
pub fn degenerate(text: &str) -> bool {
    let chars: Vec<char> = text.chars().filter(|c| !is_py_space(*c)).collect();
    if chars.len() < 8 {
        return false;
    }
    let mut counts: HashMap<char, usize> = HashMap::new();
    for c in &chars {
        *counts.entry(*c).or_insert(0) += 1;
    }
    let top = counts.values().copied().max().unwrap_or(0);
    top as f64 >= 0.9 * chars.len() as f64 || token_loop(text).is_some()
}

/// `bytes.strip()`'s whitespace set.
fn is_bytes_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

/// Whether the `fl`-byte fragment at `i` repeats three times back to back
/// and is neither one byte repeated nor all whitespace.
fn loops_at(b: &[u8], i: usize, fl: usize) -> bool {
    let f = &b[i..i + fl];
    if f.iter().all(|x| *x == f[0]) || f.iter().all(|x| is_bytes_space(*x)) {
        return false;
    }
    &b[i + fl..i + 2 * fl] == f && &b[i + 2 * fl..i + 3 * fl] == f
}

/// `token_loop(text)`: a 4..16-byte fragment repeated three times back to
/// back, ignoring one-byte and all-whitespace fragments.
pub fn token_loop(text: &str) -> Option<String> {
    let b = text.as_bytes();
    if b.len() < 12 {
        return None;
    }
    for fl in 4..=16.min(b.len() / 3) {
        if let Some(i) = (0..=(b.len() - fl * 3)).find(|&i| loops_at(b, i, fl)) {
            return Some(decode_lossy(&b[i..i + fl]));
        }
    }
    None
}

// ---------------------------------------------------------------- oracle --

/// `model_format(name)`.
pub fn model_format(name: &Val) -> PyResult<Val> {
    let f = match name {
        n if !n.truthy() => String::new(),
        Val::Str(s) => py_lower(s),
        other => {
            return Err(PyErr::attr(format!(
                "'{}' object has no attribute 'lower'",
                other.type_name()
            )))
        }
    };
    Ok(if f.ends_with(".gguf") {
        Val::str("gguf")
    } else if f.ends_with(".apr") {
        Val::str("apr")
    } else if f.ends_with(".safetensors") || f.contains("safetensors") {
        Val::str("safetensors")
    } else {
        Val::None
    })
}

/// `spoke(entry)`: answered, or produced a degenerate completion.
pub fn spoke(entry: &Val) -> PyResult<bool> {
    if entry.get("answered")?.truthy() {
        return Ok(true);
    }
    Ok(py_str(&entry.get("why")?.or(Val::str(""))).starts_with("degenerate"))
}

/// `oracle_eval(prompt, entry)` -> {"correct", "why", "extracted"} (the
/// not-answered forms put `extracted` before `why`, as the source does).
pub fn oracle_eval(prompt: &Val, entry: &Val) -> PyResult<Dict> {
    if !entry.get("answered")?.truthy() {
        return Ok(Dict::from_pairs(vec![
            ("correct", Val::Bool(false)),
            ("extracted", Val::None),
            ("why", entry.get("why")?.or(Val::str("did not answer"))),
        ]));
    }
    if !prompt.contains_str("oracle")? {
        return Ok(Dict::from_pairs(vec![
            ("correct", Val::Bool(false)),
            ("extracted", Val::None),
            ("why", Val::str(V1_WHY)),
        ]));
    }
    let turns = entry.get("turns")?.or(Val::None);
    let text = entry.get("answer")?;
    let turns_arg = if turns.is_none() { None } else { Some(&turns) };
    let v = oracles::evaluate(prompt, &text, turns_arg)?;
    let judged = if turns.truthy() && is(&prompt.item("oracle")?.get("type")?, "state_recall") {
        turns.last()?
    } else {
        text
    };
    let why = v.why.map_or(Val::None, Val::Str);
    let extracted = oracles::extract(prompt, &judged)?;
    Ok(Dict::from_pairs(vec![
        ("correct", Val::Bool(v.correct)),
        ("why", why),
        ("extracted", extracted),
    ]))
}

fn pget(p: &Dict, k: &str) -> Val {
    p.get_str(k).cloned().unwrap_or(Val::None)
}

/// What one engine run wrote.
struct RunOutput {
    stdout: String,
    stderr: String,
}

/// `why = "unknown engine ..."`: no parser fits the cell.
fn unknown_engine(e: &mut Dict, engine: &Val) {
    e.put(
        "why",
        Val::Str(format!("unknown engine {}", py_repr(engine))),
    );
}

/// The `serve run` / `serve stream` verbs: `None` for an unknown engine.
fn parse_serve(e: &mut Dict, engine: &Val, stdout: &str) -> PyResult<Option<Dict>> {
    let p = parse_engine_json(stdout)?;
    if is(engine, "apr") {
        e.put("backend_verified", Val::Bool(false));
    } else if !one_of(engine, &COMPARATORS) {
        unknown_engine(e, engine);
        return Ok(None);
    }
    Ok(Some(p))
}

/// The `chat` verb: `None` for an unknown engine.
fn parse_chat(e: &mut Dict, engine: &Val, stdout: &str) -> PyResult<Option<Dict>> {
    let p = if is(engine, "apr") {
        let p = parse_apr_chat(stdout);
        e.put("backend_verified", Val::Bool(false));
        p
    } else if one_of(engine, &COMPARATORS) {
        parse_engine_json(stdout)?
    } else {
        unknown_engine(e, engine);
        return Ok(None);
    };
    let turns = pget(&p, "turns");
    e.put("turns", turns.or(Val::List(vec![])));
    Ok(Some(p))
}

/// The one-shot verbs, by engine: `None` for an unknown engine.
fn parse_cli(
    e: &mut Dict,
    row: &Val,
    engine: &Val,
    out: &RunOutput,
    content: &Val,
) -> PyResult<Option<Dict>> {
    if is(engine, "apr") {
        let p = parse_apr(&out.stdout, &out.stderr)?;
        for k in [
            "backend",
            "prompt_ids",
            "prompt_token_count",
            "rendered_prompt",
        ] {
            e.put(k, pget(&p, k));
        }
        Ok(Some(p))
    } else if is(engine, "llama.cpp") {
        parse_llamacpp_cli(&out.stdout, content).map(Some)
    } else if is(engine, "ollama") {
        parse_ollama(&out.stdout, &out.stderr).map(Some)
    } else if one_of(engine, &PLUGIN_ENGINES) {
        let p = parse_engine_json(&out.stdout)?;
        let source = row.get("source")?;
        if source.truthy() {
            e.put("source", source);
        }
        Ok(Some(p))
    } else {
        unknown_engine(e, engine);
        Ok(None)
    }
}

/// Parse a run's output by verb, then engine: `None` for an unknown engine.
fn parse_output(
    e: &mut Dict,
    row: &Val,
    engine: &Val,
    out: &RunOutput,
    content: &Val,
) -> PyResult<Option<Dict>> {
    let verb = row.get("verb")?;
    if one_of(&verb, &["serve run", "serve stream"]) {
        parse_serve(e, engine, &out.stdout)
    } else if is(&verb, "chat") {
        parse_chat(e, engine, &out.stdout)
    } else if is(&verb, "code") && one_of(engine, &COMPARATORS) {
        parse_engine_json(&out.stdout).map(Some)
    } else {
        parse_cli(e, row, engine, out, content)
    }
}

/// #3962 B2: thinking ON prefills `<think>\n` in the prompt, so apr's reply
/// starts inside the block; re-attach the opener and let the oracle decide.
/// `Ok(true)` when the entry is final: the opener is unknown and the answer
/// carries no think tag.
fn reattach_think(
    row: &Val,
    engine: &Val,
    prompt_opened: &Val,
    p: &mut Dict,
    e: &mut Dict,
) -> PyResult<bool> {
    if !(is(engine, "apr")
        && is(&row.get("thinking")?, "on")
        && one_of(&row.get("verb")?, &["run", "chat"]))
    {
        return Ok(false);
    }
    let Val::Str(answer) = pget(p, "answer") else {
        return Ok(false);
    };
    let mut opened = rendered_opens_think(&pget(p, "rendered_prompt"))?;
    if opened.is_none() {
        opened = prompt_opened.clone();
    }
    e.put("prompt_opened_think", opened.clone());
    if matches!(opened, Val::Bool(true)) && !py_lower(py_lstrip(&answer)).starts_with("<think>") {
        p.put("answer", Val::Str(format!("<think>\n{answer}")));
    } else if opened.is_none() && !think_either().is_match(&answer) {
        e.put("reported", pget(p, "reported"));
        e.put("answer", pget(p, "answer"));
        e.put("why", Val::str(THINK_UNKNOWN_WHY));
        return Ok(true);
    }
    Ok(false)
}

/// apr's backend check: the lane asked for one backend and another ran, or
/// it fell back.
fn backend_why(row: &Val, engine: &Val, p: &Dict) -> PyResult<Option<String>> {
    let be = if is(engine, "apr") {
        pget(p, "backend")
    } else {
        Val::None
    };
    if !be.truthy() {
        return Ok(None);
    }
    let fell_back = be.get("fell_back")?;
    let mismatch = fell_back.truthy() || {
        let lane = row.get("backend")?;
        lane.truthy() && py_ne(&be.get("ran")?, &row.get("backend")?)
    };
    if !mismatch {
        return Ok(None);
    }
    Ok(Some(format!(
        "backend: asked {}, ran {} (fell_back={})",
        py_str(&row.get("backend")?),
        py_str(&be.get("ran")?),
        py_str(&be.get("fell_back")?)
    )))
}

/// A plugin engine's device check: it must report a device, on the lane's
/// side of cpu/gpu.
fn device_why(row: &Val, engine: &Val, p: &Dict) -> PyResult<Option<String>> {
    if !one_of(engine, &PLUGIN_ENGINES) {
        return Ok(None);
    }
    let rep = pget(p, "reported").or(empty());
    let dev = py_str(&rep.get("device")?.or(Val::str("")));
    let lane = row.get("backend")?;
    if dev.is_empty() {
        return Ok(Some(format!(
            "no reported.device: the {} lane cannot be verified",
            py_str(&lane)
        )));
    }
    if is(&lane, "cpu") != py_lower(&dev).starts_with("cpu") {
        return Ok(Some(format!(
            "device {} is not the {} lane",
            repr_str(&dev),
            py_str(&lane)
        )));
    }
    Ok(None)
}

/// `engine_entry(row, prompt, prompt_opened)`: one engine's answer to one cell.
pub fn engine_entry(row: &Val, prompt: &Val, prompt_opened: &Val) -> PyResult<Dict> {
    let mut e = Dict::from_pairs(vec![
        ("answered", Val::Bool(false)),
        ("rc", row.get("rc")?),
        ("why", Val::None),
        ("answer", Val::None),
        ("reported", empty()),
    ]);
    if row.contains_str("ollama_unloaded")? {
        e.put("vram_released", row.item("ollama_unloaded")?);
    }
    let refused = row.get("refused")?;
    if refused.truthy() {
        e.put("why", Val::Str(concat_str("refused: ", &refused)?));
        return Ok(e);
    }
    let out = RunOutput {
        stdout: read_text(&row.get("stdout")?)?,
        stderr: read_text(&row.get("stderr")?)?,
    };
    let content = prompt.item("messages")?.last()?.item("content")?;
    let engine = row.item("engine")?;
    let Some(mut p) = parse_output(&mut e, row, &engine, &out, &content)? else {
        return Ok(e);
    };
    if reattach_think(row, &engine, prompt_opened, &mut p, &mut e)? {
        return Ok(e);
    }
    e.put("reported", pget(&p, "reported"));
    let answer = pget(&p, "answer");
    e.put("answer", answer.clone());
    let rc = row.get("rc")?;
    if py_ne(&rc, &Val::int(0)) {
        e.put("why", Val::Str(format!("exit {}", py_str(&rc))));
        return Ok(e);
    }
    let Val::Str(answer) = answer else {
        // only None reaches here: every parser's answer is a str or None
        e.put("why", pget(&p, "why"));
        return Ok(e);
    };
    let why = if degenerate(&answer) {
        Some(format!(
            "degenerate output (one character is >=90% of it): {}",
            repr_str(slice_chars(&answer, 0, 24))
        ))
    } else if let Some(why) = backend_why(row, &engine, &p)? {
        Some(why)
    } else {
        device_why(row, &engine, &p)?
    };
    match why {
        Some(why) => e.put("why", Val::Str(why)),
        None => e.put("answered", Val::Bool(true)),
    }
    Ok(e)
}

/// `token_parity(apr_entry, tok_row)`.
pub fn token_parity(apr_entry: &Val, tok_row: Option<&Val>) -> PyResult<Val> {
    let Some(tok) = tok_row else {
        return Ok(dict(vec![
            ("measured", Val::Bool(false)),
            ("why", Val::str("no llama.cpp tokenization row")),
        ]));
    };
    let loaded = tok
        .item("ids")
        .and_then(|p| path_arg(&p))
        .and_then(|p| read_text_strict(&p))
        .and_then(|t| loads(&t));
    let doc = match loaded {
        Ok(doc) => doc,
        Err(e) if e.unreadable() => {
            return Ok(dict(vec![
                ("measured", Val::Bool(false)),
                ("why", Val::Str(format!("llama.cpp ids unreadable: {e}"))),
            ]))
        }
        Err(e) => return Err(e),
    };
    // `.get` on a non-dict document is an AttributeError the except does not catch
    let Val::List(reference) = doc.get("tokens")? else {
        return Ok(dict(vec![
            ("measured", Val::Bool(false)),
            ("why", Val::str("llama.cpp ids missing")),
        ]));
    };
    let ids = apr_entry.get("prompt_ids")?;
    let n = apr_entry.get("prompt_token_count")?;
    if ids.is_none() || n.is_none() {
        return Ok(dict(vec![
            ("measured", Val::Bool(false)),
            ("why", Val::str("apr printed no prompt ids")),
            ("llama_cpp_count", Val::int(reference.len() as i64)),
        ]));
    }
    let ids = ids.iter()?;
    let mut first = ids.iter().zip(&reference).position(|(a, b)| py_ne(a, b));
    if first.is_none() && ids.len() > reference.len() {
        first = Some(reference.len());
    }
    let n_ref = Val::int(reference.len() as i64);
    Ok(dict(vec![
        ("measured", Val::Bool(true)),
        ("apr_count", n.clone()),
        ("llama_cpp_count", n_ref.clone()),
        ("apr_ids_compared", Val::int(ids.len() as i64)),
        (
            "apr_ids_complete",
            Val::Bool(py_eq(&Val::int(ids.len() as i64), &n)),
        ),
        ("first_divergence", opt_idx(first)),
        ("parity", Val::Bool(py_eq(&n, &n_ref) && first.is_none())),
    ]))
}

// ------------------------------------------- deterministic rows (tok/tmpl) --

/// `_load_json(path, field)`: the list at `field`, or ValueError.
fn load_json(path: &Val, field: &str) -> PyResult<Vec<Val>> {
    let text = read_text_strict(&path_arg(path)?)?;
    match loads(&text)?.get(field)? {
        Val::List(v) => Ok(v),
        _ => Err(PyErr::value(format!(
            "{} has no list {}",
            py_str(path),
            repr_str(field)
        ))),
    }
}

/// What a deterministic row produced: tok ids or a tmpl rendering's bytes.
#[derive(Clone)]
enum Produced {
    Ids(Vec<Val>),
    Bytes(Vec<u8>),
}

fn first_diff_by<T>(a: &[T], b: &[T], ne: impl Fn(&T, &T) -> bool) -> Option<usize> {
    let i = a.iter().zip(b).position(|(x, y)| ne(x, y));
    if i.is_none() && a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    i
}

fn first_diff(a: &Produced, b: &Produced) -> Option<usize> {
    match (a, b) {
        (Produced::Ids(x), Produced::Ids(y)) => first_diff_by(x, y, py_ne),
        (Produced::Bytes(x), Produced::Bytes(y)) => first_diff_by(x, y, |p, q| p != q),
        _ => unreachable!("one kind per group"),
    }
}

/// `_det_side(row, loader, field)` -> (value, why).
fn det_side(row: Option<&Val>, tok: bool) -> PyResult<(Option<Produced>, Val)> {
    let Some(row) = row else {
        return Ok((None, Val::str("missing: no row for this engine")));
    };
    let refused = row.get("refused")?;
    if refused.truthy() {
        return Ok((None, Val::Str(concat_str("refused: ", &refused)?)));
    }
    let loaded = if tok {
        row.item("ids")
            .and_then(|p| load_json(&p, "tokens"))
            .map(Produced::Ids)
    } else {
        row.item("rendered")
            .and_then(|p| load_bytes(&p))
            .map(Produced::Bytes)
    };
    match loaded {
        Ok(v) => Ok((Some(v), Val::None)),
        Err(e) if e.unreadable() => Ok((None, Val::Str(format!("unreadable: {e}")))),
        Err(e) => Err(e),
    }
}

/// `groups.setdefault(key, {})[engine] = row`.
fn group_insert(groups: &mut Dict, key: Val, engine: Val, row: &Val) -> PyResult<()> {
    if groups.get(&key)?.is_none() {
        groups.set(key.clone(), empty())?;
    }
    match groups.get_mut(&key)? {
        Some(Val::Dict(by)) => by.set(engine, row.clone()),
        _ => unreachable!("groups hold dicts"),
    }
}

/// `sorted(d.items())` for a dict whose keys are distinct, so the tuple
/// comparison never reaches the values.
fn sorted_items(d: &Dict) -> PyResult<Vec<(Val, Val)>> {
    let items: Vec<(Val, Val)> = d.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    py_sorted_by(&items, &|kv: &(Val, Val)| kv.0.clone())
}

fn zip_key(names: &[&str], key: &Val) -> PyResult<Val> {
    let parts = key.iter()?;
    Ok(Val::Dict(Dict::from_pairs(
        names.iter().copied().zip(parts).collect(),
    )))
}

/// Row `r`'s group key for `judge_deterministic`: `None` when the row is
/// not of that kind (a tok row also needs an `input`).
fn det_key(r: &Val, tok: bool) -> PyResult<Option<Val>> {
    let keep = if tok {
        is(&r.get("kind")?, "tok") && r.contains_str("input")?
    } else {
        is(&r.get("kind")?, "tmpl")
    };
    if !keep {
        return Ok(None);
    }
    let mut key = vec![
        r.item("model_sha256")?,
        r.item("host")?,
        r.item("prompt_id")?,
    ];
    if !tok {
        key.push(r.get_or("thinking", Val::str("unset"))?);
    }
    Ok(Some(Val::Tuple(key)))
}

/// Each reference engine's side of one group against apr's: the
/// `references` dict, and (engine, equal) for every engine that produced.
fn det_refs(
    by: &Dict,
    apr_val: Option<&Produced>,
    tok: bool,
) -> PyResult<(Dict, Vec<(Val, bool)>)> {
    let mut refs = Dict::new();
    let mut produced = Vec::new();
    for (eng, r) in sorted_items(by)? {
        if is(&eng, "apr") {
            continue;
        }
        let (val, why) = det_side(Some(&r), tok)?;
        let mut ent = Dict::from_pairs(vec![("produced", Val::Bool(val.is_some())), ("why", why)]);
        let mut equal = true;
        if let (Some(v), Some(a)) = (&val, apr_val) {
            let d = first_diff(a, v);
            equal = d.is_none();
            ent.put("equal", Val::Bool(equal));
            ent.put("first_difference", opt_idx(d));
        }
        if val.is_some() {
            produced.push((eng.clone(), equal));
        }
        refs.set(eng, Val::Dict(ent))?;
    }
    Ok((refs, produced))
}

/// One group's `judge_deterministic` row.
fn det_row(kind: &str, names: &[&str], key: &Val, by: &Dict) -> PyResult<Val> {
    let tok = kind == "tok";
    let (apr_val, apr_why) = det_side(by.get_str("apr"), tok)?;
    let (refs, produced) = det_refs(by, apr_val.as_ref(), tok)?;
    let produced_names: Vec<Val> = produced.iter().map(|(e, _)| e.clone()).collect();
    let coverage = dict(vec![
        ("field", Val::str(if tok { "ids" } else { "rendered" })),
        ("produced_by", Val::List(py_sorted(&produced_names)?)),
        ("references_producing", Val::int(produced.len() as i64)),
        ("apr_produced", Val::Bool(apr_val.is_some())),
    ]);
    let verdict = if produced.is_empty() || apr_val.is_none() || produced.iter().any(|(_, eq)| !eq)
    {
        "RED"
    } else {
        "GREEN"
    };
    Ok(dict(vec![
        ("kind", Val::str(kind)),
        ("key", zip_key(names, key)?),
        ("verdict", Val::str(verdict)),
        ("coverage", coverage),
        (
            "apr",
            dict(vec![
                ("produced", Val::Bool(apr_val.is_some())),
                ("why", apr_why),
            ]),
        ),
        ("references", Val::Dict(refs)),
    ]))
}

/// `judge_deterministic(rows, kind)`: tok and tmpl are byte-equal or RED.
pub fn judge_deterministic(rows: &[Val], kind: &str) -> PyResult<Vec<Val>> {
    let tok = kind == "tok";
    let mut groups = Dict::new();
    for r in rows {
        let Some(key) = det_key(r, tok)? else {
            continue;
        };
        let engine = r.item("engine")?;
        group_insert(&mut groups, key, engine, r)?;
    }
    let names: &[&str] = if tok {
        &["model_sha256", "host", "prompt_id"]
    } else {
        &["model_sha256", "host", "prompt_id", "thinking"]
    };
    let keys: Vec<Val> = groups.keys().cloned().collect();
    let mut out = Vec::new();
    for key in py_sorted(&keys)? {
        let by = match groups.get(&key)? {
            Some(Val::Dict(by)) => by.clone(),
            _ => unreachable!("groups hold dicts"),
        };
        out.push(det_row(kind, names, &key, &by)?);
    }
    Ok(out)
}

// ------------------------------------------------------------- greedy rows --

/// A minimal `ast.literal_eval` for the npy header dict: str keys and values,
/// ints, True/False/None, tuples and lists. Divergence (README): anything
/// else is a SyntaxError here, where Python may accept it.
struct Literal<'a> {
    s: &'a [char],
    i: usize,
}

fn syntax_err() -> PyErr {
    PyErr::new(
        "SyntaxError",
        "the .npy header is not a literal this port parses",
    )
}

impl Literal<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], ' ' | '\t' | '\n' | '\r' | '\x0c') {
            self.i += 1;
        }
    }
    fn eat(&mut self, c: char) -> bool {
        self.ws();
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    /// `items, close`: comma-separated values up to `close`, trailing comma ok.
    fn seq(&mut self, close: char) -> PyResult<(Vec<Val>, bool)> {
        let mut items = Vec::new();
        let mut trailing_comma = false;
        loop {
            if self.eat(close) {
                return Ok((items, trailing_comma));
            }
            items.push(self.value()?);
            trailing_comma = self.eat(',');
            if !trailing_comma {
                if self.eat(close) {
                    return Ok((items, false));
                }
                return Err(syntax_err());
            }
        }
    }
    fn value(&mut self) -> PyResult<Val> {
        self.ws();
        let Some(&c) = self.s.get(self.i) else {
            return Err(syntax_err());
        };
        match c {
            '{' => {
                self.i += 1;
                self.dict_body()
            }
            '(' => {
                self.i += 1;
                let (items, trailing) = self.seq(')')?;
                if items.len() == 1 && !trailing {
                    return Ok(items.into_iter().next().expect("one item"));
                }
                Ok(Val::Tuple(items))
            }
            '[' => {
                self.i += 1;
                Ok(Val::List(self.seq(']')?.0))
            }
            '\'' | '"' => {
                self.i += 1;
                self.str_body(c)
            }
            '0'..='9' | '-' | '+' => self.int(),
            _ => self.word(),
        }
    }
    /// A dict after its `{`.
    fn dict_body(&mut self) -> PyResult<Val> {
        let mut d = Dict::new();
        loop {
            if self.eat('}') {
                return Ok(Val::Dict(d));
            }
            let k = self.value()?;
            if !self.eat(':') {
                return Err(syntax_err());
            }
            let v = self.value()?;
            d.set(k, v)?;
            if !self.eat(',') {
                if self.eat('}') {
                    return Ok(Val::Dict(d));
                }
                return Err(syntax_err());
            }
        }
    }
    /// A str after its opening `quote`: no escapes, no newlines.
    fn str_body(&mut self, quote: char) -> PyResult<Val> {
        let start = self.i;
        while self.i < self.s.len() && self.s[self.i] != quote {
            if matches!(self.s[self.i], '\\' | '\n') {
                return Err(syntax_err());
            }
            self.i += 1;
        }
        if self.i >= self.s.len() {
            return Err(syntax_err());
        }
        let body: String = self.s[start..self.i].iter().collect();
        self.i += 1;
        Ok(Val::Str(body))
    }
    /// An int: an optional sign, then digits.
    fn int(&mut self) -> PyResult<Val> {
        let start = self.i;
        self.i += 1;
        while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
            self.i += 1;
        }
        let lit: String = self.s[start..self.i].iter().collect();
        PyInt::parse_digits(&lit)
            .map(Val::Int)
            .map_err(|_| syntax_err())
    }
    /// True, False or None.
    fn word(&mut self) -> PyResult<Val> {
        let start = self.i;
        while self.i < self.s.len() && self.s[self.i].is_ascii_alphanumeric() {
            self.i += 1;
        }
        let word: String = self.s[start..self.i].iter().collect();
        match word.as_str() {
            "True" => Ok(Val::Bool(true)),
            "False" => Ok(Val::Bool(false)),
            "None" => Ok(Val::None),
            _ => Err(syntax_err()),
        }
    }
}

fn literal_eval(text: &str) -> PyResult<Val> {
    let chars: Vec<char> = text.trim_start_matches([' ', '\t']).chars().collect();
    let mut p = Literal { s: &chars, i: 0 };
    let v = p.value()?;
    p.ws();
    if p.i != chars.len() {
        return Err(syntax_err());
    }
    Ok(v)
}

/// `struct.error`.
fn struct_err(msg: impl Into<String>) -> PyErr {
    PyErr::new("error", msg)
}

/// `_npy_rows(path)`: a float32 C-order 2-D .npy as rows of floats.
fn npy_rows(path: &Val) -> PyResult<Vec<Vec<f64>>> {
    let data = load_bytes(path)?;
    if data.len() < 6 || &data[..6] != b"\x93NUMPY" {
        return Err(PyErr::value("not an .npy file"));
    }
    let Some(&major) = data.get(6) else {
        return Err(PyErr::index("index out of range"));
    };
    let (hstart, width) = if major == 1 {
        (10usize, 2usize)
    } else {
        (12, 4)
    };
    let field = data
        .get(8..hstart)
        .filter(|f| f.len() == width)
        .ok_or_else(|| struct_err(format!("unpack requires a buffer of {width} bytes")))?;
    let hlen = if width == 2 {
        usize::from(u16::from_le_bytes([field[0], field[1]]))
    } else {
        u32::from_le_bytes([field[0], field[1], field[2], field[3]]) as usize
    };
    let start = hstart + hlen;
    let header: String = data[hstart.min(data.len())..start.min(data.len())]
        .iter()
        .map(|b| char::from(*b))
        .collect();
    let hdr = literal_eval(&header)?;
    let shape = hdr.get_or("shape", Val::Tuple(vec![]))?;
    if !is(&hdr.get("descr")?, "<f4") || hdr.get("fortran_order")?.truthy() || shape.len()? != 2 {
        return Err(PyErr::value(format!(
            "expected little-endian float32 C-order 2-D, got {}",
            py_repr(&hdr)
        )));
    }
    let dims = hdr.item("shape")?.iter()?;
    let dim = |v: &Val| -> PyResult<i128> {
        match v {
            Val::Int(i) => i
                .to_i64()
                .map(i128::from)
                .ok_or_else(|| struct_err("total struct size too long")),
            Val::Bool(b) => Ok(i128::from(*b)),
            other => Err(PyErr::type_err(format!(
                "unsupported operand type(s) for *: '{}' and '{}'",
                other.type_name(),
                other.type_name()
            ))),
        }
    };
    let (n, m) = (dim(&dims[0])?, dim(&dims[1])?);
    let count = n * m;
    if count < 0 {
        return Err(struct_err("bad char in struct format"));
    }
    let want = usize::try_from(4 * count).map_err(|_| struct_err("total struct size too long"))?;
    let body = &data[start.min(data.len())..start.saturating_add(want).min(data.len())];
    if body.len() != want {
        return Err(struct_err(format!(
            "unpack requires a buffer of {want} bytes"
        )));
    }
    let flat: Vec<f64> = body
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f64::from(f32::from_le_bytes(*c)))
        .collect();
    let m = m.max(0) as usize;
    Ok((0..n.max(0) as usize)
        .map(|i| flat[(i * m).min(flat.len())..((i + 1) * m).min(flat.len())].to_vec())
        .collect())
}

/// `sum(xs)` from int 0, as CPython 3.12+ computes it for floats: None for
/// an empty sequence (int 0), else the first item added to 0, then Neumaier
/// compensation, the compensation applied only when non-zero and finite.
fn py_sum(xs: impl Iterator<Item = f64>) -> Option<f64> {
    let mut it = xs;
    let first = it.next()?;
    let mut f = 0.0 + first;
    let mut c = 0.0;
    for x in it {
        let t = f + x;
        if f.abs() >= x.abs() {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    if c != 0.0 && c.is_finite() {
        f += c;
    }
    Some(f)
}

/// `_cosine(a, b)`.
fn cosine(a: &[f64], b: &[f64]) -> Val {
    let num = py_sum(a.iter().zip(b).map(|(x, y)| x * y)).unwrap_or(0.0);
    let sa = py_sum(a.iter().map(|x| x * x)).unwrap_or(0.0);
    let sb = py_sum(b.iter().map(|y| y * y)).unwrap_or(0.0);
    let den = sa.sqrt() * sb.sqrt();
    if den == 0.0 {
        return Val::None;
    }
    Val::Float(num / den)
}

/// `_greedy_raw(r)` -> {"raw": {...}} | {"refused": why} | {"why": ...}.
fn greedy_raw(r: &Val) -> PyResult<Dict> {
    if r.get("refused")?.truthy() {
        return Ok(Dict::from_pairs(vec![("refused", r.item("refused")?)]));
    }
    let loaded = r
        .item("tokens")
        .and_then(|p| path_arg(&p))
        .and_then(|p| read_text_strict(&p))
        .and_then(|t| loads(&t));
    Ok(match loaded {
        Ok(raw @ Val::Dict(_)) => Dict::from_pairs(vec![("raw", raw)]),
        Ok(_) => Dict::from_pairs(vec![(
            "why",
            Val::str("raw greedy record is not an object"),
        )]),
        Err(e) if e.unreadable() => Dict::from_pairs(vec![(
            "why",
            Val::Str(format!("raw greedy record unreadable: {e}")),
        )]),
        Err(e) => return Err(e),
    })
}

/// `rep.setdefault(eng, {}).update(item)`.
fn merge_into(rep: &mut Dict, eng: &Val, item: impl FnOnce() -> PyResult<Dict>) -> PyResult<()> {
    if rep.get(eng)?.is_none() {
        rep.set(eng.clone(), empty())?;
    }
    match rep.get(eng)? {
        Some(Val::Dict(_)) => {}
        Some(other) => {
            return Err(PyErr::attr(format!(
                "'{}' object has no attribute 'update'",
                other.type_name()
            )))
        }
        None => unreachable!("set above"),
    }
    let item = item()?;
    if let Some(Val::Dict(d)) = rep.get_mut(eng)? {
        d.update(&item);
    }
    Ok(())
}

/// One engine's comparison against apr's greedy row.
fn greedy_compare(apr: Option<&Val>, eng: &str, r: &Val) -> PyResult<Dict> {
    let apr = match apr {
        Some(a) if !a.get("refused")?.truthy() && !r.get("refused")?.truthy() => a,
        _ => return Err(PyErr::value(format!("apr or {eng} has no greedy row"))),
    };
    let a = load_json(&apr.item("tokens")?, "generated_ids")?;
    let b = load_json(&r.item("tokens")?, "generated_ids")?;
    let d = first_diff_by(&a, &b, py_ne);
    let mut item = Dict::from_pairs(vec![
        ("first_divergence", opt_idx(d)),
        ("steps_compared", Val::int(a.len().min(b.len()) as i64)),
    ]);
    if let Some(d) = d {
        if apr.get("logits")?.truthy() && r.get("logits")?.truthy() {
            let la = npy_rows(&apr.item("logits")?)?;
            let lb = npy_rows(&r.item("logits")?)?;
            if d < la.len() && d < lb.len() {
                item.put("logit_cosine_at_divergence", cosine(&la[d], &lb[d]));
            }
        }
    }
    Ok(item)
}

/// Row `r`'s greedy group key and engine name (`@official` for an official
/// prompt): `None` when it is not a greedy row.
fn greedy_key(r: &Val) -> PyResult<Option<(Val, Val)>> {
    if !is(&r.get("kind")?, "greedy") {
        return Ok(None);
    }
    let engine = r.item("engine")?;
    let official = is(&r.get("prompt_source")?, "official");
    let eng = add_str(&engine, if official { "@official" } else { "" })?;
    let key = Val::Tuple(vec![
        r.item("model_sha256")?,
        r.item("host")?,
        r.item("prompt_id")?,
        r.get_or("thinking", Val::str("unset"))?,
    ]);
    Ok(Some((key, Val::Str(eng))))
}

/// One reference engine against apr; a comparison that cannot read its
/// inputs is reported as `not compared`.
fn greedy_item(apr: Option<&Val>, name: &str, r: &Val) -> PyResult<Dict> {
    match greedy_compare(apr, name, r) {
        Ok(i) => Ok(i),
        Err(e) if e.unreadable() => Ok(Dict::from_pairs(vec![(
            "why",
            Val::Str(format!("not compared: {e}")),
        )])),
        Err(e) => Err(e),
    }
}

/// One group's `report_greedy` row.
fn greedy_report(names: &[&str], key: &Val, by: &Dict) -> PyResult<Val> {
    let engines: Vec<Val> = by.keys().cloned().collect();
    let mut rep = Dict::from_pairs(vec![
        ("key", zip_key(names, key)?),
        ("engines", Val::List(py_sorted(&engines)?)),
    ]);
    let items = sorted_items(by)?;
    for (eng, r) in &items {
        merge_into(&mut rep, eng, || greedy_raw(r))?;
    }
    let apr = by.get_str("apr");
    for (eng, r) in &items {
        let Val::Str(name) = eng else {
            unreachable!("greedy engine names are str")
        };
        if name == "apr" {
            continue;
        }
        let item = greedy_item(apr, name, r)?;
        merge_into(&mut rep, eng, || Ok(item))?;
    }
    Ok(Val::Dict(rep))
}

/// `report_greedy(rows)`: REPORTED, never judged.
pub fn report_greedy(rows: &[Val]) -> PyResult<Vec<Val>> {
    let mut groups = Dict::new();
    for r in rows {
        if let Some((key, eng)) = greedy_key(r)? {
            group_insert(&mut groups, key, eng, r)?;
        }
    }
    let names = ["model_sha256", "host", "prompt_id", "thinking"];
    let keys: Vec<Val> = groups.keys().cloned().collect();
    let mut out = Vec::new();
    for key in py_sorted(&keys)? {
        let by = match groups.get(&key)? {
            Some(Val::Dict(by)) => by.clone(),
            _ => unreachable!("groups hold dicts"),
        };
        out.push(greedy_report(&names, &key, &by)?);
    }
    Ok(out)
}

// ------------------------------------------------------------------ cells --

/// `cell_quorum(entries)`.
pub fn cell_quorum(entries: &Dict) -> PyResult<Val> {
    let mut names = Vec::new();
    for e in ENGINES {
        let entry = entries.get_str(e).cloned().unwrap_or(empty());
        if entry.get("answered")?.truthy() {
            names.push(e);
        }
    }
    names.sort_unstable();
    let comparators: Vec<Val> = names
        .iter()
        .filter(|e| COMPARATORS.contains(e))
        .map(|e| Val::str(*e))
        .collect();
    Ok(dict(vec![
        ("floor", Val::int(CELL_QUORUM_FLOOR)),
        ("engines_answered", Val::int(names.len() as i64)),
        (
            "answered",
            Val::List(names.iter().map(|e| Val::str(*e)).collect()),
        ),
        ("comparators_answered", Val::List(comparators)),
        ("met", Val::Bool(names.len() as i64 >= CELL_QUORUM_FLOOR)),
    ]))
}

/// The verdict of one cell: (GREEN|RED, {engine: correct}, reasons,
/// {engine: extracted}).
pub struct CellVerdict {
    pub verdict: &'static str,
    pub ok: Dict,
    pub why: Vec<String>,
    pub ext: Dict,
}

fn entry_of<'a>(entries: &'a Dict, e: &str) -> PyResult<&'a Val> {
    entries.get_str(e).ok_or_else(|| PyErr::key(repr_str(e)))
}

/// `ev[e][k]`: KeyError when `e` has no evaluation.
fn ev_field(ev: &Dict, e: &Val, k: &str) -> PyResult<Val> {
    match ev.get(e)? {
        Some(v) => v.item(k),
        None => Err(PyErr::key(py_repr(e))),
    }
}

/// apr's own reason: it did not answer, or it answered wrong.
fn apr_why(entries: &Dict, ev: &Dict, ok: &Dict) -> PyResult<Option<String>> {
    let a = entry_of(entries, "apr")?;
    if !a.get("answered")?.truthy() {
        return Ok(Some(format!(
            "apr did not answer: {}",
            py_str(&a.get("why")?.or(Val::str("no row")))
        )));
    }
    if !ok.get_str("apr").is_some_and(Val::truthy) {
        return Ok(Some(format!(
            "apr is wrong: {}",
            py_str(&ev_field(ev, &Val::str("apr"), "why")?)
        )));
    }
    Ok(None)
}

/// The `fam`-family comparators that spoke, and their extracted values
/// (`<degenerate>` for one that spoke without answering).
fn family_values(entries: &Dict, ext: &Dict, fam: &str) -> PyResult<(Vec<&'static str>, Dict)> {
    let mut same = Vec::new();
    for e in COMPARATORS {
        if family(e) == Some(fam) && spoke(entry_of(entries, e)?)? {
            same.push(e);
        }
    }
    let mut vals = Dict::new();
    for e in &same {
        let v = if entry_of(entries, e)?.get("answered")?.truthy() {
            ext.get_str(e).cloned().unwrap_or(Val::None)
        } else {
            Val::str("<degenerate>")
        };
        vals.put(e, v);
    }
    Ok((same, vals))
}

/// Whether the values hold more than one distinct value.
fn more_than_one(values: &[Val]) -> bool {
    let mut distinct: Vec<&Val> = Vec::new();
    for v in values {
        if !distinct.iter().any(|d| py_eq(d, v)) {
            distinct.push(v);
        }
    }
    distinct.len() > 1
}

/// The same-representation oracle: the engines that read apr's file format
/// must agree with each other and with apr.
fn same_rep_why(entries: &Dict, ext: &Dict, fmt: &Val) -> PyResult<Option<String>> {
    let Some(fam) = same_rep(fmt)? else {
        return Ok(Some(format!(
            "no same-representation oracle: no engine but apr reads a {} file -- a .apr is proven only \
             through its chain to its source (#3957 F8)",
            py_str(&fmt.clone().or(Val::str("format-unknown")))
        )));
    };
    let (same, vals) = family_values(entries, ext, fam)?;
    let values: Vec<Val> = vals.values().cloned().collect();
    let first = values.first().cloned().unwrap_or(Val::None);
    let apr_ext = ext.get_str("apr").cloned().unwrap_or(Val::None);
    if same.is_empty() {
        Ok(Some(format!(
            "no same-representation oracle: no {fam}-family engine answered on the identical weights \
             (#3957 Q2)"
        )))
    } else if more_than_one(&values) || values.iter().any(Val::is_none) {
        Ok(Some(format!(
            "same-representation SPLIT {} -- a split is RED, never a prompt swap (#3957 Q1)",
            dumps(&Val::Dict(vals), false)?
        )))
    } else if entry_of(entries, "apr")?.get("answered")?.truthy() && py_ne(&apr_ext, &first) {
        Ok(Some(format!(
            "apr differs from the {fam} family on the identical weights: apr {} vs {} (#3957 Q2)",
            py_repr(&apr_ext),
            py_repr(&first)
        )))
    } else {
        Ok(None)
    }
}

/// The ground-truth control: a bf16 engine must speak, and every one that
/// spoke must be correct.
fn control_why(entries: &Dict, ev: &Dict, ok: &Dict) -> PyResult<Option<String>> {
    let mut ctl = Vec::new();
    for e in COMPARATORS {
        if family(e) == Some("bf16") && spoke(entry_of(entries, e)?)? {
            ctl.push(e);
        }
    }
    if ctl.is_empty() {
        return Ok(Some(
            "no ground-truth control: neither hf nor vllm answered, so nothing shows this prompt is \
             answerable (#3957 Q2)"
                .to_string(),
        ));
    }
    let mut bad = Vec::new();
    for e in &ctl {
        if !ok.get_str(e).is_some_and(Val::truthy) {
            bad.push(format!(
                "{e}: {}",
                py_str(&ev_field(ev, &Val::str(*e), "why")?)
            ));
        }
    }
    if bad.is_empty() {
        return Ok(None);
    }
    Ok(Some(format!(
        "ground-truth control FAILED ({}) -- a control that cannot answer vouches for nothing; \
         the prompt or the control engine is broken (#3957 Q2, #3971)",
        bad.join("; ")
    )))
}

/// `judge_cell(entries, prompt, fmt)`.
pub fn judge_cell(entries: &Dict, prompt: &Val, fmt: &Val) -> PyResult<CellVerdict> {
    let mut ev = Dict::new();
    for (e, v) in entries.iter() {
        ev.set(e.clone(), Val::Dict(oracle_eval(prompt, v)?))?;
    }
    let mut ok = Dict::new();
    let mut ext = Dict::new();
    for (e, v) in entries.iter() {
        ok.set(e.clone(), ev_field(&ev, e, "correct")?)?;
        if v.get("answered")?.truthy() {
            ext.set(e.clone(), ev_field(&ev, e, "extracted")?)?;
        }
    }
    let why: Vec<String> = [
        apr_why(entries, &ev, &ok)?,
        same_rep_why(entries, &ext, fmt)?,
        control_why(entries, &ev, &ok)?,
    ]
    .into_iter()
    .flatten()
    .collect();
    Ok(CellVerdict {
        verdict: if why.is_empty() { "GREEN" } else { "RED" },
        ok,
        why,
        ext,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_ran_classes() {
        assert_eq!(classify_not_ran(&Val::None), "unclassified");
        assert_eq!(
            classify_not_ran(&Val::str("zsh: command not found: x")),
            "not_on_PATH"
        );
        assert_eq!(
            classify_not_ran(&Val::str("Not Found on PATH")),
            "not_on_PATH"
        );
        assert_eq!(classify_not_ran(&Val::str("connection REFUSED")), "refused");
        assert_eq!(classify_not_ran(&Val::str("weird")), "unclassified");
    }

    #[test]
    fn undebug_and_rendered_think() {
        let cps = undebug(r#"a\n\u{263a}\"\q"#).expect("the debug string decodes");
        let s: String = cps
            .iter()
            .map(|c| char::from_u32(*c).expect("a decoded code point is a char"))
            .collect();
        assert_eq!(s, "a\n\u{263a}\"\\q");
        assert_eq!(
            undebug(r"\u{110000}").unwrap_err().msg,
            "chr() arg not in range(0x110000)"
        );
        assert!(matches!(
            rendered_opens_think(&Val::str(r"x<think>\n")).expect("a str template renders"),
            Val::Bool(true)
        ));
        assert!(matches!(
            rendered_opens_think(&Val::str("short")).expect("a short str template renders"),
            Val::Bool(false)
        ));
        assert!(rendered_opens_think(&Val::str("x".repeat(197)))
            .expect("a 197-char template renders")
            .is_none());
        assert!(rendered_opens_think(&Val::int(1))
            .expect("a non-str template is None")
            .is_none());
        let e = rendered_opens_think(&Val::str(r"a\u{d800}\u{dc00}b")).unwrap_err();
        assert_eq!(
            e.msg,
            "'utf-8' codec can't encode characters in position 1-2: surrogates not allowed"
        );
        let e = rendered_opens_think(&Val::str(r"\u{dfff}")).unwrap_err();
        assert_eq!(
            e.msg,
            "'utf-8' codec can't encode character '\\udfff' in position 0: surrogates not allowed"
        );
    }

    #[test]
    fn degenerate_and_loops() {
        assert!(degenerate("!!!!!!!!!!"));
        assert!(!degenerate("The capital of France is Paris."));
        assert!(degenerate("NavControllerNavControllerNavController"));
        assert_eq!(token_loop("abcdabcdabcd").as_deref(), Some("abcd"));
        assert_eq!(token_loop("    \n    \n    \n"), None);
        assert_eq!(token_loop("short"), None);
    }

    #[test]
    fn parsers() {
        let p = parse_llamacpp_cli(
            "junk\n> hi\n[Start thinking]x[End thinking] 4 \n[ Prompt: 1.5 t/s | Generation: 2. t/s ]",
            &Val::str("hi"),
        )
        .expect("the llama.cpp CLI output parses");
        assert_eq!(
            py_str(p.get_str("answer").expect("the parse has an answer")),
            "<think>x</think> 4"
        );
        let p = parse_ollama(
            "  Paris \n",
            "eval count: 12 token(s)\neval rate: 3.5 tokens/s\n",
        )
        .expect("the ollama output parses");
        let rep = p.get_str("reported").expect("the parse has reported");
        assert_eq!(
            py_repr(
                &rep.get("completion_tokens")
                    .expect("reported has completion_tokens")
            ),
            "12"
        );
        assert_eq!(
            py_repr(&rep.get("decode_rate").expect("reported has decode_rate")),
            "3.5"
        );
        let p = parse_apr_chat("Assistant: one\nYou: q\nAssistant: two\n");
        assert_eq!(
            py_repr(p.get_str("turns").expect("the parse has turns")),
            "['one', 'two']"
        );
        let p = parse_engine_json(r#"{"reasoning": "r", "text": ""}"#)
            .expect("engine JSON with reasoning parses");
        assert_eq!(
            py_str(p.get_str("answer").expect("the parse has an answer")),
            "<think>r"
        );
        let p = parse_engine_json("nope").expect("non-JSON engine output is a parse result");
        assert_eq!(
            py_str(p.get_str("why").expect("a failed parse says why")),
            "stdout is not the contract's JSON: Expecting value: line 1 column 1 (char 0)"
        );
    }

    #[test]
    fn sum_and_cosine() {
        assert_eq!(py_sum([0.1, 0.2, 0.3].into_iter()), Some(0.6));
        assert_eq!(py_sum(std::iter::empty()), None);
        assert!(cosine(&[], &[]).is_none());
        assert_eq!(py_repr(&cosine(&[1.0, 0.0], &[1.0, 0.0])), "1.0");
    }

    #[test]
    fn npy_header_literal() {
        let v = literal_eval("{'descr': '<f4', 'fortran_order': False, 'shape': (3, 5), }   \n")
            .expect("the npy header dict evaluates");
        assert_eq!(
            py_repr(&v),
            "{'descr': '<f4', 'fortran_order': False, 'shape': (3, 5)}"
        );
        assert_eq!(
            py_repr(&literal_eval("(7,)").expect("a one-tuple evaluates")),
            "(7,)"
        );
        assert_eq!(
            py_repr(&literal_eval("(7)").expect("a parenthesised int evaluates")),
            "7"
        );
    }
}
