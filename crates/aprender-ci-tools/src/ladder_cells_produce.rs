//! `ladder-cells-produce`: the PRODUCER of the model ladder receipt's `cells[]` (#3712 row B),
//! ported from `scripts/lib/model_ladder_cells_produce.py` (#5225), which stays while
//! `scripts/model_ladder.sh` calls it.
//!
//! * `gpumem` prints `{"gpu_mem_total_bytes": T, "gpu_mem_free_bytes": F}`: nvidia-smi, else
//!   `/proc/meminfo` (a unified-memory GPU), else nulls.
//! * `enrich` adds the terms the judge derives the owed set from (arch, context length, the
//!   chat template's thinking evidence, the memory terms) to every inventory row, read from
//!   the FILE HEADER via `apr inspect --json`, never from the file name.
//! * `measure` RUNS every owed (model, verb, thinking, rung) cell on the accelerator and
//!   writes one row per cell. The owed set is the judge's ([`crate::ladder_cells_judge`]).
//!
//! A row claims only what was measured: backend and fallback come from apr's own envelope
//! (`backend.ran`/`backend.fell_back` for run, chat and code, `used_gpu` in a serve body), and
//! a document without them leaves both null (#4986). A pass needs the answer to contain a
//! NEEDLE planted at the start of the prompt.
//!
//! Output is byte-identical with the original (`json.dumps` separators, insertion order,
//! `ensure_ascii`, Python float repr). Known gaps, beyond those of [`crate::pyjson`] and
//! [`crate::ladder_cells_io`]: the ladder YAML is read as YAML 1.2 (`yes`/`no` stay strings);
//! a Python crash (a traceback) is one `Err` line here, at the same exit code 1; a chat
//! template, answer text or `prompt_tokens` of a type the original only half-handles (a list
//! template, a non-str answer, a str count) is refused here; a value past `i128` (a bigint in
//! Python) is refused; option abbreviations are clap's `infer_long_args`, close to argparse's.

use std::fmt::Write as _;
use std::io::Write as _;

use clap::{Args, Subcommand};
use regex::Regex;
use sha2::{Digest, Sha256};
use std::sync::LazyLock;

use crate::ladder_cells_io::{free_port, run_with_timeout, Serve};
use crate::ladder_cells_judge::{
    declaimed, expected_modes, find_declaimed, load_rungs, or_list, owed_rungs,
};
use crate::llama_fit_verdict::PyInt;
use crate::pyjson::{
    dict_get, dumps, dumps_indent1, from_yaml, loads, or_empty, py_contains, py_eq, py_int,
    py_repr_value, py_str, subscript, LoadErr, Py,
};
use crate::pystr::{is_py_space, py_repr, py_splitlines, py_strip, py_text_lines};

const MIB: i128 = 1024 * 1024;
pub const NEEDLE_WORD: &str = "TANGERINE-4417";
const QUESTION: &str = "Question: what is the passphrase stated in the very first line of this \
                        message? Reply with the passphrase only.";
const SOURCE: &str = "apr run, identical prompt text (this verb prints no count)";

/// Python's `\s` in a `str` pattern also matches the separators U+001C..U+001F.
static CAPACITY_RE: LazyLock<Regex> = LazyLock::new(|| {
    let s = r"[\s\x1c-\x1f]*";
    let re = format!(r"={s}([0-9.]+){s}MiB,{s}against{s}([0-9.]+){s}MiB free of{s}([0-9.]+){s}MiB");
    Regex::new(&re).expect("static regex")
});

#[derive(Subcommand, Debug)]
pub enum Sub {
    /// The GPU's total and free bytes, one JSON line.
    Gpumem,
    /// inventory.jsonl + models (file|path) -> enriched inventory.jsonl (same rows, order).
    #[command(args_override_self = true, infer_long_args = true)]
    Enrich(Common),
    /// Run every owed cell and write the rows as a JSON array.
    #[command(args_override_self = true, infer_long_args = true)]
    Measure(Measure),
}

#[derive(Args, Debug)]
pub struct Common {
    #[arg(long)]
    pub apr: String,
    #[arg(long)]
    pub inventory: String,
    /// file|path lines, the ladder's INVENTORY.
    #[arg(long)]
    pub models: String,
    #[arg(long)]
    pub out: String,
}

#[derive(Args, Debug)]
pub struct Measure {
    #[command(flatten)]
    pub common: Common,
    #[arg(long)]
    pub ladder: String,
    #[arg(long)]
    pub rungs: String,
    #[arg(long)]
    pub work: String,
    #[arg(long)]
    pub lock: Option<String>,
    /// This host's ladder id: a cells.declaimed arch on it is not measured.
    #[arg(long)]
    pub host: Option<String>,
    #[arg(long, default_value = "1800", value_parser = py_int_arg, allow_negative_numbers = true)]
    pub lock_wait: i64,
    /// Per apr call, seconds.
    #[arg(long, default_value = "3600", value_parser = py_int_arg, allow_negative_numbers = true)]
    pub timeout: i64,
    #[arg(long, default_value = "600", value_parser = py_int_arg, allow_negative_numbers = true)]
    pub serve_ceiling: i64,
    #[arg(long, default_value = "64", value_parser = py_int_arg, allow_negative_numbers = true)]
    pub max_tokens: i64,
    /// A think block must have room to CLOSE.
    #[arg(long, default_value = "1024", value_parser = py_int_arg, allow_negative_numbers = true)]
    pub max_tokens_thinking: i64,
    #[arg(
        long,
        default_value = "4.5",
        value_parser = py_float_arg,
        allow_negative_numbers = true
    )]
    pub chars_per_token: f64,
    #[arg(long)]
    pub only: Option<String>,
}

/// argparse's `type=int`.
fn py_int_arg(s: &str) -> Result<i64, String> {
    PyInt::parse(s)
        .and_then(|n| n.to_string().parse::<i64>().ok())
        .ok_or_else(|| format!("invalid int value: {}", py_repr(s)))
}

/// argparse's `type=float`: surrounding whitespace, `_` between digits, `nan`/`inf`.
fn py_float_arg(s: &str) -> Result<f64, String> {
    let bad = || format!("invalid float value: {}", py_repr(s));
    let t = py_strip(s);
    let chars: Vec<char> = t.chars().collect();
    let digit_at = |i: Option<usize>| {
        i.and_then(|i| chars.get(i))
            .is_some_and(char::is_ascii_digit)
    };
    for (i, c) in chars.iter().enumerate() {
        if *c == '_' && !(digit_at(i.checked_sub(1)) && digit_at(Some(i + 1))) {
            return Err(bad());
        }
    }
    t.replace('_', "").parse::<f64>().map_err(|_| bad())
}

/// `int(<str>)` as an `i128`, `None` where Python raises (or past `i128`, a known gap).
fn int_of(s: &str) -> Option<i128> {
    PyInt::parse(s).and_then(|n| n.to_string().parse::<i128>().ok())
}

fn kv(k: &str, v: Py) -> (String, Py) {
    (k.to_owned(), v)
}

fn opt_int(v: Option<i128>) -> Py {
    v.map_or(Py::None, Py::Int)
}

// ------------------------------------------------------------------------------------ enrich

/// `_num`: `int(str(md.get(key)).strip())`, or `None`.
fn num(md: &Py, key: &str) -> Result<Option<i128>, String> {
    Ok(int_of(py_strip(&py_str(dict_get(md, key)?))))
}

/// `-> (markers, generation_opens_think)`, from the template text itself (#3723).
pub fn thinking_evidence(tmpl: &str) -> (Vec<&'static str>, bool) {
    let markers = ["enable_thinking", "<think>", "</think>"]
        .into_iter()
        .filter(|m| tmpl.contains(m))
        .collect();
    let after_gen = tmpl
        .split_once("add_generation_prompt")
        .map_or("", |(_, after)| after);
    // The generation prompt OPENS a think block unconditionally when it emits <think> and no
    // enable_thinking switch exists anywhere to turn it off.
    let opens = after_gen.contains("<think>") && !tmpl.contains("enable_thinking");
    (markers, opens)
}

/// Python's `a // b` (floor division); `None` on a zero divisor or an overflow.
fn floordiv(a: i128, b: i128) -> Option<i128> {
    let q = a.checked_div(b)?;
    // Rust's remainder takes the dividend's sign; Python rounds down where it and b's differ.
    let r = a % b;
    if r != 0 && r.signum() != b.signum() {
        q.checked_sub(1)
    } else {
        Some(q)
    }
}

fn overflow() -> String {
    "OverflowError: a kv-cache term exceeds 128 bits".to_owned()
}

/// capacity.rs's term: 2 (K,V) x kv heads x head dim x 4 B, over the layers that HAVE a KV
/// cache (a hybrid's full-attention layers only). `None` when a number is absent.
pub fn kv_bytes_per_token_f32(md: &Py, arch: &Py) -> Result<Option<i128>, String> {
    let p = format!("{}.", py_str(arch));
    let n = |k: &str| num(md, &format!("{p}{k}"));
    let layers = n("block_count")?;
    let kvh = n("attention.head_count_kv")?;
    let heads = n("attention.head_count")?;
    let emb = n("embedding_length")?;
    let kl = match n("attention.key_length")? {
        Some(k) if k != 0 => Some(k),
        _ => match (emb, heads) {
            (Some(e), Some(h)) if e != 0 && h != 0 => Some(floordiv(e, h).ok_or_else(overflow)?),
            _ => None,
        },
    };
    let vl = match n("attention.value_length")? {
        Some(v) if v != 0 => Some(v),
        _ => kl,
    };
    let interval = n("full_attention_interval")?;
    let (Some(layers), Some(kvh), Some(kl), Some(vl)) = (layers, kvh, kl, vl) else {
        return Ok(None);
    };
    let attn_layers = match interval {
        Some(i) if i != 0 => floordiv(layers, i).ok_or_else(overflow)?,
        _ => layers,
    };
    kl.checked_add(vl)
        .and_then(|d| attn_layers.checked_mul(kvh)?.checked_mul(d)?.checked_mul(4))
        .map(Some)
        .ok_or_else(overflow)
}

/// The fields the judge derives the owed set from, from `apr inspect --json`.
pub fn inventory_terms(inspect: &Py, file_bytes: Py) -> Result<Py, String> {
    let ins = or_empty(inspect);
    let md = or_empty(dict_get(ins, "metadata")?);
    let mut arch = dict_get(md, "general.architecture")?.clone();
    if !arch.truthy() {
        let unset = Py::List(vec![Py::None, Py::str(""), Py::str("unknown")]);
        if !py_contains(&unset, dict_get(ins, "architecture")?)? {
            arch = subscript(ins, "architecture")?.clone();
        }
    }
    let tmpl = dict_get(md, "tokenizer.chat_template")?;
    let (markers, opens, sha) = match tmpl {
        Py::None => (Py::None, false, Py::None),
        Py::Str(t) => {
            let (m, o) = thinking_evidence(t);
            let mut hex = String::with_capacity(64);
            for b in Sha256::digest(t.as_bytes()) {
                let _ = write!(hex, "{b:02x}");
            }
            (
                Py::List(m.into_iter().map(Py::str).collect()),
                o,
                Py::Str(hex),
            )
        }
        other => {
            return Err(format!(
                "TypeError: a chat template of type '{}' is not text",
                other.type_name()
            ))
        }
    };
    let modes = expected_modes(&markers, opens)?;
    let has_arch = arch.truthy();
    let context_length = if has_arch {
        opt_int(num(md, &format!("{}.context_length", py_str(&arch)))?)
    } else {
        Py::None
    };
    let kv_bytes = if has_arch {
        opt_int(kv_bytes_per_token_f32(md, &arch)?)
    } else {
        Py::None
    };
    Ok(Py::Dict(vec![
        kv("arch", if has_arch { arch.clone() } else { Py::None }),
        kv("quant", dict_get(md, "general.file_type")?.clone()),
        kv("context_length", context_length),
        kv("thinking_markers", markers),
        kv("generation_opens_think", Py::Bool(opens)),
        kv(
            "thinking_modes",
            match modes {
                Some(m) if !m.is_empty() => Py::List(m.into_iter().map(Py::str).collect()),
                _ => Py::None,
            },
        ),
        kv("chat_template_sha256", sha),
        kv("weights_bytes", file_bytes),
        kv("kv_bytes_per_token", kv_bytes),
        kv("kv_dtype", Py::str("f32")),
        // NOT measured: apr computes its workspace inside the load path and prints it only in
        // a refusal. Null keeps every cell owed in full (the judge's rule).
        kv("workspace_bytes", Py::None),
    ]))
}

/// The original's nvidia-smi query.
const NVIDIA_SMI: [&str; 3] = [
    "nvidia-smi",
    "--query-gpu=memory.total,memory.free",
    "--format=csv,noheader,nounits",
];

/// `(total, free)` bytes from the first line of `argv`'s `total, free` MiB output; `None`
/// where it cannot run or prints anything else (`[N/A]` on a unified-memory GPU).
fn gpu_mem(argv: &[&str]) -> Option<(i128, i128)> {
    let argv: Vec<String> = argv.iter().map(|s| (*s).to_owned()).collect();
    let ran = run_with_timeout(&argv, None, 30).ok()?;
    let first = py_splitlines(&ran.stdout).first().copied()?.to_owned();
    let parts: Vec<&str> = first.split(',').collect();
    let [t, f] = parts.as_slice() else {
        return None;
    };
    let total = int_of(t)?.checked_mul(MIB)?;
    let free = int_of(f)?.checked_mul(MIB)?;
    Some((total, free))
}

/// `/proc/meminfo`'s MemTotal and MemAvailable, as text; `Err` where the original crashes
/// (a key with an empty value).
pub fn meminfo(text: &str) -> Result<(Option<i128>, Option<i128>), String> {
    let mut mi: Vec<(String, String)> = Vec::new();
    for line in text.split_inclusive('\n') {
        let Some((k, v)) = line.split_once(':') else {
            return Ok((None, None));
        };
        mi.push((k.to_owned(), v.to_owned()));
    }
    let kb = |key: &str| -> Result<Option<i128>, String> {
        let Some((_, v)) = mi.iter().rev().find(|(k, _)| k == key) else {
            return Ok(None);
        };
        let first = v
            .split(is_py_space)
            .find(|w| !w.is_empty())
            .ok_or("IndexError: list index out of range")?;
        Ok(int_of(first).and_then(|n| n.checked_mul(1024)))
    };
    // A missing key or a bad number stops at the first: (None, None).
    let Some(total) = kb("MemTotal")? else {
        return Ok((None, None));
    };
    let Some(avail) = kb("MemAvailable")? else {
        return Ok((None, None));
    };
    Ok((Some(total), Some(avail)))
}

/// `-> (total_bytes, free_bytes)`: a discrete GPU per nvidia-smi; a unified-memory GPU
/// (where nvidia-smi says [N/A]) per /proc/meminfo. The query and the meminfo path are
/// [`NVIDIA_SMI`] and `/proc/meminfo` outside the tests.
fn gpu_memory(smi: &[&str], meminfo_path: &str) -> Result<(Option<i128>, Option<i128>), String> {
    if let Some((t, f)) = gpu_mem(smi) {
        return Ok((Some(t), Some(f)));
    }
    let Ok(b) = std::fs::read(meminfo_path) else {
        return Ok((None, None));
    };
    meminfo(&crate::pystr::py_read_text(&b))
}

// ------------------------------------------------------------------------------------ measure

/// A prompt of at least `tokens` tokens at the given density: the needle FIRST, then
/// filler, then the question LAST, so the answer is only reachable across the whole context.
pub fn build_prompt(tokens: i128, chars_per_token: f64) -> Result<String, String> {
    let head = format!("The passphrase is {NEEDLE_WORD}. Remember it.\n");
    let w = tokens as f64 * chars_per_token * 1.01;
    if !w.is_finite() {
        return Err(format!(
            "ValueError: cannot convert float {} to integer",
            crate::pyjson::float_repr(w)
        ));
    }
    let want = w.trunc() as i128;
    let mut out = head.clone();
    let mut size = (head.len() + QUESTION.len()) as i128;
    let mut n: u64 = 0;
    while size < want {
        let line = format!(
            "Log entry {n:06}: the ferry left the harbour on time and the weather stayed calm.\n"
        );
        size += line.len() as i128;
        out.push_str(&line);
        n += 1;
    }
    out.push_str(QUESTION);
    Ok(out)
}

/// `-> (answer, think_closed)`; `think_closed` is `None` when no think block was opened.
pub fn split_thinking(text: &str) -> (String, Option<bool>) {
    if let Some(i) = text.rfind("</think>") {
        return (
            py_strip(&text[i + "</think>".len()..]).to_owned(),
            Some(true),
        );
    }
    if text.contains("<think>") {
        return (String::new(), Some(false));
    }
    (py_strip(text).to_owned(), None)
}

/// `int(float(x) * MIB)`.
fn mib_bytes(x: &str) -> Result<Option<i128>, String> {
    let f: f64 = x.parse().map_err(|_| {
        format!(
            "ValueError: could not convert string to float: {}",
            py_repr(x)
        )
    })?;
    let v = f * MIB as f64;
    if !v.is_finite() || v.abs() >= 1.0e38 {
        return Err(format!(
            "OverflowError: cannot convert float {} to a 128-bit integer",
            crate::pyjson::float_repr(v)
        ));
    }
    Ok(Some(v.trunc() as i128))
}

/// apr's own pre-load arithmetic, if this was a capacity refusal: `(required, available,
/// total)`, each `None` when the numbers are not in the message.
pub type Capacity = (Option<i128>, Option<i128>, Option<i128>);

pub fn refusal(stderr: &str) -> Result<Option<Capacity>, String> {
    if !stderr.contains("capacity refused") {
        return Ok(None);
    }
    let Some(m) = CAPACITY_RE.captures(stderr) else {
        return Ok(Some((None, None, None)));
    };
    let g = |i: usize| m.get(i).map_or("", |x| x.as_str());
    Ok(Some((mib_bytes(g(1))?, mib_bytes(g(2))?, mib_bytes(g(3))?)))
}

/// `stderr.strip().splitlines()[-1]`, if the stripped text is not empty.
fn last_line(stderr: &str) -> Option<&str> {
    py_splitlines(py_strip(stderr)).last().copied()
}

pub fn base_row(item: &Py, verb: &Py, mode: &str, rid: &Py, max_tokens: i64) -> Result<Py, String> {
    Ok(Py::Dict(vec![
        kv("sha256", subscript(item, "sha256")?.clone()),
        kv("file", subscript(item, "file")?.clone()),
        kv("verb", verb.clone()),
        kv("thinking", Py::str(mode)),
        kv("context", rid.clone()),
        kv("prompt_tokens", Py::None),
        kv("max_tokens", Py::Int(i128::from(max_tokens))),
        kv("think_closed", Py::None),
        kv("answer_chars", Py::Int(0)),
        kv("ttft_ms", Py::None),
        kv("required_bytes", Py::None),
        kv("available_bytes", Py::None),
        kv("verdict", Py::str("fail")),
        kv("backend", Py::None),
        kv("fallback", Py::None),
        kv("rc", Py::None),
        kv("reason", Py::str("")),
    ]))
}

/// The shared verdict: refusal, failure, or a pass that must contain the needle.
pub fn finish(mut row: Py, rc: i64, text: &str, stderr: &str) -> Result<Py, String> {
    row.set("rc", Py::Int(i128::from(rc)));
    let refused = refusal(stderr)?;
    if let (true, Some((required, available, _))) = (rc != 0, refused) {
        row.set("verdict", Py::str("refused"));
        row.set("required_bytes", opt_int(required));
        row.set("available_bytes", opt_int(available));
        row.set(
            "reason",
            Py::str(last_line(stderr).unwrap_or("capacity refused")),
        );
        return Ok(row);
    }
    let (answer, closed) = split_thinking(text);
    let chars = answer.chars().count();
    row.set("answer_chars", Py::Int(chars as i128));
    row.set("think_closed", closed.map_or(Py::None, Py::Bool));
    let reason = if rc != 0 {
        format!("rc {rc}: {}", last_line(stderr).unwrap_or("no stderr"))
    } else if closed == Some(false) {
        format!(
            "thinking never closed within max_tokens {}: no </think> in the output",
            py_str(row.at("max_tokens"))
        )
    } else if !answer.contains(NEEDLE_WORD) {
        let head: String = answer.chars().take(120).collect();
        let more = if chars > 120 {
            format!(" ... and {} more chars", chars - 120)
        } else {
            String::new()
        };
        format!(
            "answer does not contain the needle planted at token 0: {}{more}",
            py_repr(&head)
        )
    } else {
        row.set("verdict", Py::str("pass"));
        "ok".to_owned()
    };
    row.set("reason", Py::Str(reason));
    Ok(row)
}

/// `{"gpu": "cuda", "cpu": "cpu"}.get(ran)`.
fn backend_name(ran: &Py) -> Result<Py, String> {
    match ran {
        Py::Str(s) if s == "gpu" => Ok(Py::str("cuda")),
        Py::Str(s) if s == "cpu" => Ok(Py::str("cpu")),
        Py::List(_) | Py::Dict(_) => {
            Err(format!("TypeError: unhashable type: '{}'", ran.type_name()))
        }
        _ => Ok(Py::None),
    }
}

/// `backend`/`fallback` from an envelope's `backend` object (#4986): null when absent.
fn set_backend(row: &mut Py, env: &Py) -> Result<(), String> {
    let be = or_empty(dict_get(env, "backend")?);
    let ran = backend_name(dict_get(be, "ran")?)?;
    let fell_back = dict_get(be, "fell_back")?.clone();
    row.set("backend", ran);
    row.set("fallback", fell_back);
    Ok(())
}

/// `text or ""` for a value the original goes on to search as text.
fn text_of(v: Option<&Py>) -> Result<String, String> {
    match v {
        None => Ok(String::new()),
        Some(Py::Str(s)) => Ok(s.clone()),
        Some(v) if !v.truthy() => Ok(String::new()),
        Some(v) => Err(format!(
            "TypeError: an answer of type '{}' is not text",
            v.type_name()
        )),
    }
}

/// `json.loads(out) if rc == 0 else {}`; `Ok(None)` is the caught `JSONDecodeError`.
fn envelope(rc: i64, out: &str) -> Result<Option<Py>, String> {
    if rc != 0 {
        return Ok(Some(Py::dict()));
    }
    match loads(out) {
        Ok(v) => Ok(Some(v)),
        Err(LoadErr::Decode(_)) => Ok(None),
        Err(LoadErr::Fatal(m)) => Err(m),
    }
}

/// The row of one `apr run --format json` call.
pub fn run_row(mut row: Py, rc: i64, out: &str, err: &str) -> Result<Py, String> {
    let (env, rc, err) = match envelope(rc, out)? {
        Some(env) => (env, rc, err.to_owned()),
        None => (
            Py::dict(),
            3,
            format!("{err} | stdout was not the --format json envelope"),
        ),
    };
    let be = or_empty(dict_get(&env, "backend")?);
    row.set("prompt_tokens", dict_get(&env, "prompt_tokens")?.clone());
    row.set("backend", backend_name(dict_get(be, "ran")?)?);
    row.set("fallback", dict_get(be, "fell_back")?.clone());
    let text = text_of(env.get("text"))?;
    finish(row, rc, &text, &err)
}

/// The row of one `apr chat --json` call: the line carrying `backend` is the envelope.
pub fn chat_row(mut row: Py, rc: i64, out: &str, err: &str) -> Result<Py, String> {
    let mut text: Vec<&str> = Vec::new();
    let mut env = Py::dict();
    for line in py_splitlines(out) {
        match loads(line) {
            Ok(d @ Py::Dict(_)) if d.get("backend").is_some() => {
                env = d;
                continue;
            }
            Ok(_) | Err(LoadErr::Decode(_)) => {}
            Err(LoadErr::Fatal(m)) => return Err(m),
        }
        text.push(line);
    }
    set_backend(&mut row, &env)?;
    finish(row, rc, &text.join("\n"), err)
}

/// The row of one `apr code --output-format json` call. `apr code` reports its serve child's
/// device in `backend` (#3719); a document without it leaves backend and fallback null,
/// which the judge refuses as not_measured (#4986).
pub fn code_row(mut row: Py, rc: i64, out: &str, err: &str) -> Result<Py, String> {
    let (doc, rc) = match envelope(rc, out)? {
        Some(doc) => (doc, rc),
        None => (Py::dict(), 3),
    };
    set_backend(&mut row, &doc)?;
    let text = text_of(doc.get("result"))?;
    finish(row, rc, &text, err)
}

/// The row of one serve answer: `(status or 1, body, error text)`.
pub fn serve_row(mut row: Py, rc: i64, body: &Py, err: &str) -> Result<Py, String> {
    let choices = dict_get(body, "choices")?;
    let first = if choices.truthy() {
        match choices {
            Py::List(items) => items.first().cloned().unwrap_or(Py::None),
            Py::Str(s) => Py::Str(s.chars().take(1).collect()),
            Py::Dict(_) => return Err("KeyError: 0".to_owned()),
            other => {
                return Err(format!(
                    "TypeError: '{}' object is not subscriptable",
                    other.type_name()
                ))
            }
        }
    } else {
        Py::dict()
    };
    let msg = or_empty(dict_get(&first, "message")?).clone();
    let content = dict_get(&msg, "content")?;
    let reasoning = dict_get(&msg, "reasoning_content")?;
    let text = if reasoning.truthy() {
        let answer = if content.truthy() {
            py_str(content)
        } else {
            String::new()
        };
        format!("<think>{}</think>{answer}", py_str(reasoning))
    } else {
        text_of(Some(content))?
    };
    let usage = or_empty(dict_get(body, "usage")?);
    row.set("prompt_tokens", dict_get(usage, "prompt_tokens")?.clone());
    let ug = dict_get(body, "used_gpu")?;
    let (backend, fallback) = if *ug == Py::None {
        (Py::None, Py::None)
    } else {
        let gpu = ug.truthy();
        (Py::str(if gpu { "cuda" } else { "cpu" }), Py::Bool(!gpu))
    };
    row.set("backend", backend);
    row.set("fallback", fallback);
    finish(row, rc, &text, err)
}

/// Runs apr under the fleet GPU lock: `flock -E 75 -w WAIT LOCK choom -n 1000 -- apr ...`.
pub struct Runner<'a> {
    pub apr: &'a str,
    pub lock: &'a str,
    pub wait: i64,
    pub timeout: i64,
}

impl Runner<'_> {
    pub fn argv(&self, args: &[&str]) -> Vec<String> {
        let mut v: Vec<String> = Vec::new();
        if !self.lock.is_empty() {
            let wait = self.wait.to_string();
            let wait = wait.as_str();
            let pre = [
                "flock", "-E", "75", "-w", wait, self.lock, "choom", "-n", "1000", "--",
            ];
            v.extend(pre.iter().map(|s| (*s).to_owned()));
        }
        v.push(self.apr.to_owned());
        v.extend(args.iter().map(|s| (*s).to_owned()));
        v
    }

    /// `(returncode, stdout, stderr)`; a timeout is `(124, "", "timed out after T s")`.
    pub fn call(
        &self,
        args: &[&str],
        stdin: Option<&str>,
    ) -> Result<(i64, String, String), String> {
        let r = run_with_timeout(&self.argv(args), stdin, self.timeout)?;
        Ok((r.rc, r.stdout, r.stderr))
    }
}

fn measure_run(
    r: &Runner,
    path: &str,
    pf: &str,
    mode: &str,
    budget: i64,
    row: Py,
) -> Result<Py, String> {
    let mt = budget.to_string();
    let mt = mt.as_str();
    let args = [
        "run",
        path,
        "--input",
        pf,
        "--thinking",
        mode,
        "--format",
        "json",
        "--max-tokens",
        mt,
        "--gpu",
    ];
    let (rc, out, err) = r.call(&args, None)?;
    run_row(row, rc, &out, &err)
}

fn measure_chat(
    r: &Runner,
    path: &str,
    prompt: &str,
    mode: &str,
    budget: i64,
    row: Py,
) -> Result<Py, String> {
    let mt = budget.to_string();
    let mt = mt.as_str();
    let args = [
        "chat",
        path,
        "--json",
        "--max-tokens",
        mt,
        "--thinking",
        mode,
        "--gpu",
    ];
    // chat reads ONE turn per line
    let stdin = format!("{}\n/exit\n", prompt.replace('\n', " "));
    let (rc, out, err) = r.call(&args, Some(&stdin))?;
    chat_row(row, rc, &out, &err)
}

fn measure_code(
    r: &Runner,
    path: &str,
    prompt: &str,
    mode: &str,
    budget: i64,
    row: Py,
) -> Result<Py, String> {
    let mt = budget.to_string();
    let mt = mt.as_str();
    let args = [
        "code",
        "--model",
        path,
        "-p",
        "--output-format",
        "json",
        "--thinking",
        mode,
        "--max-tokens",
        mt,
        "--gpu",
    ];
    let (rc, out, err) = r.call(&args, Some(prompt))?;
    code_row(row, rc, &out, &err)
}

/// The chat-completions request body `Serve.ask` posts.
pub fn ask_body(prompt: &str, mode: &str, max_tokens: i64) -> String {
    let message = Py::Dict(vec![
        kv("role", Py::str("user")),
        kv("content", Py::str(prompt)),
    ]);
    dumps(&Py::Dict(vec![
        kv("model", Py::str("apr")),
        kv("messages", Py::List(vec![message])),
        kv("max_tokens", Py::Int(i128::from(max_tokens))),
        kv("stream", Py::Bool(false)),
        kv(
            "chat_template_kwargs",
            Py::Dict(vec![kv("enable_thinking", Py::Bool(mode == "on"))]),
        ),
    ]))
}

/// `Serve.ask`: `(0, body, "")`, `(status, {}, body text)` on an HTTP error, `(1, {}, why)`
/// on a connection or decode error.
fn ask(
    s: &Serve,
    prompt: &str,
    mode: &str,
    max_tokens: i64,
    timeout: i64,
) -> Result<(i64, Py, String), String> {
    let body = ask_body(prompt, mode, max_tokens);
    let resp = match s.post("/v1/chat/completions", body.as_bytes(), timeout) {
        Ok(resp) => resp,
        Err(why) => return Ok((1, Py::dict(), why)),
    };
    if !(200..300).contains(&resp.status) {
        let text = String::from_utf8_lossy(&resp.body).into_owned();
        return Ok((i64::from(resp.status), Py::dict(), text));
    }
    let raw = resp
        .body
        .strip_prefix(b"\xef\xbb\xbf")
        .unwrap_or(resp.body.as_slice());
    let text = std::str::from_utf8(raw).map_err(|e| format!("UnicodeDecodeError: {e}"))?;
    match loads(text) {
        Ok(v) => Ok((0, v, String::new())),
        Err(LoadErr::Decode(m)) => Ok((1, Py::dict(), m)),
        Err(LoadErr::Fatal(m)) => Err(m),
    }
}

fn measure_serve(
    s: &Serve,
    prompt: &str,
    mode: &str,
    budget: i64,
    mut row: Py,
    timeout: i64,
) -> Result<Py, String> {
    if !s.up {
        row.set("rc", Py::Int(1));
        row.set("reason", Py::str("apr serve never answered /health"));
        return Ok(row);
    }
    let (rc, body, err) = ask(s, prompt, mode, budget, timeout)?;
    serve_row(row, rc, &body, &err)
}

/// One owed cell: `(rung id, tokens, thinking mode, verb)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Cell {
    pub rid: Py,
    pub tok: Option<i128>,
    pub mode: &'static str,
    pub verb: Py,
}

/// The cells this item owes -- the judge's own enumeration. `Err` (the original's
/// `SystemExit`) when the rungs file fails the judge.
pub fn cells_for(item: &Py, ladder: &Py, rungs_doc: &Py) -> Result<Vec<Cell>, String> {
    let c = or_empty(dict_get(ladder, "cells")?);
    let mut why = Vec::new();
    let (rungs, consumer_max, rc) = load_rungs(rungs_doc, &mut why)?;
    if rc != 0 {
        // Measuring an empty or partial owed set against a failed rungs file would print
        // "0 row(s)" as if nothing were owed (#3715).
        return Err(format!("cells: refused -- {}", why.join("; ")));
    }
    let markers = dict_get(item, "thinking_markers")?;
    let opens = dict_get(item, "generation_opens_think")?.truthy();
    let modes = expected_modes(markers, opens)?.unwrap_or_else(|| vec!["off", "on"]);
    let long_for = or_empty(dict_get(c, "long_rungs_for")?);
    let mut out = Vec::new();
    for (rid, tok) in owed_rungs(item, &rungs, long_for, consumer_max)? {
        for verb in or_list(dict_get(c, "verbs")?)? {
            for &mode in &modes {
                out.push(Cell {
                    rid: rid.clone(),
                    tok,
                    mode,
                    verb: verb.clone(),
                });
            }
        }
    }
    Ok(out)
}

/// `os.path.join(a, b)`.
fn path_join(a: &str, b: &str) -> String {
    if b.starts_with('/') {
        b.to_owned()
    } else if a.is_empty() || a.ends_with('/') {
        format!("{a}{b}")
    } else {
        format!("{a}/{b}")
    }
}

fn write_text(path: &str, text: &str) -> Result<(), String> {
    std::fs::write(path, text).map_err(|e| format!("OSError: {path}: {e}"))
}

/// A numeric `prompt_tokens` as a float, for `len(prompt) / pt`.
fn as_float(v: &Py) -> Result<f64, String> {
    match v {
        Py::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Py::Int(i) => Ok(*i as f64),
        Py::Float(f) => Ok(*f),
        other => Err(format!(
            "TypeError: unsupported operand type(s) for /: 'int' and '{}'",
            other.type_name()
        )),
    }
}

/// `pt >= target`.
fn reaches(pt: &Py, target: i128) -> Result<bool, String> {
    match pt {
        Py::Bool(b) => Ok(i128::from(*b) >= target),
        Py::Int(i) => Ok(*i >= target),
        Py::Float(f) => Ok(*f >= target as f64),
        other => Err(format!(
            "TypeError: '>=' not supported between instances of '{}' and 'int'",
            other.type_name()
        )),
    }
}

/// A row whose verb prints no count carries the count `apr run` measured for the same text.
fn fill_measured(row: &mut Py, measured: Option<&Py>) {
    if let Some(m) = measured {
        if *row.at("prompt_tokens") == Py::None {
            row.set("prompt_tokens", m.clone());
            row.set("prompt_tokens_source", Py::str(SOURCE));
        }
    }
}

/// A serve cell, measured after every other cell of its model.
struct Deferred {
    idx: usize,
    prompt: String,
    mode: &'static str,
    budget: i64,
    measured: Option<Py>,
}

type Groups = Vec<((Py, Option<i128>), Vec<(&'static str, Py)>)>;

/// The owed cells grouped by rung, in first-seen order, then stably sorted by size.
fn group_by_rung(owed: Vec<Cell>) -> Groups {
    let mut groups: Groups = Vec::new();
    for c in owed {
        let at = groups
            .iter()
            .position(|((rid, tok), _)| py_eq(rid, &c.rid) && *tok == c.tok);
        match at {
            Some(i) => groups[i].1.push((c.mode, c.verb)),
            None => groups.push(((c.rid, c.tok), vec![(c.mode, c.verb)])),
        }
    }
    groups.sort_by_key(|((_, tok), _)| tok.unwrap_or(0));
    groups
}

/// One model being measured: the runner, its inventory row, its path and the options.
struct Model<'a> {
    r: &'a Runner<'a>,
    item: &'a Py,
    path: &'a str,
    a: &'a Measure,
}

/// One (rung, mode) slot of a model: the prompt file every verb of the slot reads.
struct Slot<'a> {
    rid: &'a Py,
    mode: &'static str,
    budget: i64,
    target: i128,
    pf: String,
}

/// What a model's cells accumulate: its rows, its deferred serve cells, the density.
struct Acc {
    rows: Vec<Py>,
    deferred: Vec<Deferred>,
    density: f64,
}

/// The `run` cell: the first rung's density is a guess, so up to three attempts, each
/// correcting it from apr's own count.
fn count_run(m: &Model, s: &Slot, prompt: &mut String, density: &mut f64) -> Result<Py, String> {
    let verb = Py::str("run");
    let mut row = base_row(m.item, &verb, s.mode, s.rid, s.budget)?;
    for _attempt in 0..3 {
        let fresh = base_row(m.item, &verb, s.mode, s.rid, s.budget)?;
        row = measure_run(m.r, m.path, &s.pf, s.mode, s.budget, fresh)?;
        let pt = row.at("prompt_tokens");
        if !pt.truthy() {
            break;
        }
        let d = prompt.chars().count() as f64 / as_float(pt)?;
        // `max(1.0, d)`: a NaN density is 1.0 there too.
        *density = d.max(1.0);
        if reaches(pt, s.target)? {
            break;
        }
        *prompt = build_prompt(s.target, *density)?;
        write_text(&s.pf, prompt)?;
    }
    Ok(row)
}

/// The verbs `jobs` owes in `mode`, `run` first: it counts.
fn slot_verbs<'a>(jobs: &'a [(&'static str, Py)], mode: &str) -> Vec<&'a Py> {
    let mut verbs: Vec<&Py> = jobs
        .iter()
        .filter(|(m, _)| *m == mode)
        .map(|(_, v)| v)
        .collect();
    verbs.sort_by_key(|v| v.as_str() != Some("run"));
    verbs
}

/// A prompt of `tok` tokens, less when the model's context must leave room for a `budget`
/// answer: `min(tok, ctx - budget - 1)`.
fn prompt_target(tok: i128, ctx: &Py, budget: i64) -> Result<i128, String> {
    if !ctx.truthy() {
        return Ok(tok);
    }
    let room = py_int(ctx)?
        .checked_sub(i128::from(budget) + 1)
        .ok_or_else(overflow)?;
    Ok(tok.min(room))
}

/// Every verb of one (rung, mode) slot, `run` first: it counts.
fn measure_slot(
    m: &Model,
    acc: &mut Acc,
    s: &Slot,
    jobs: &[(&'static str, Py)],
) -> Result<(), String> {
    let mut prompt = build_prompt(s.target, acc.density)?;
    write_text(&s.pf, &prompt)?;
    let mut measured: Option<Py> = None;
    for verb in slot_verbs(jobs, s.mode) {
        let row = base_row(m.item, verb, s.mode, s.rid, s.budget)?;
        let mut row = match verb.as_str() {
            Some("run") => {
                let row = count_run(m, s, &mut prompt, &mut acc.density)?;
                measured = Some(row.at("prompt_tokens").clone()).filter(Py::truthy);
                row
            }
            Some("serve") => {
                acc.rows.push(row);
                acc.deferred.push(Deferred {
                    idx: acc.rows.len() - 1,
                    prompt: prompt.clone(),
                    mode: s.mode,
                    budget: s.budget,
                    measured: measured.clone(),
                });
                continue;
            }
            Some("chat") => measure_chat(m.r, m.path, &prompt, s.mode, s.budget, row)?,
            _ => measure_code(m.r, m.path, &prompt, s.mode, s.budget, row)?,
        };
        fill_measured(&mut row, measured.as_ref());
        acc.rows.push(row);
    }
    Ok(())
}

/// One rung of a model: a row per job when the rung has no size, else each mode's slot.
fn measure_rung(
    m: &Model,
    acc: &mut Acc,
    rid: &Py,
    tok: Option<i128>,
    jobs: &[(&'static str, Py)],
) -> Result<(), String> {
    let Some(tok) = tok else {
        for (mode, verb) in jobs {
            let mut row = base_row(m.item, verb, mode, rid, m.a.max_tokens)?;
            row.set("reason", Py::str("rung has no token count"));
            acc.rows.push(row);
        }
        return Ok(());
    };
    let ctx = dict_get(m.item, "context_length")?;
    let mut modes: Vec<&'static str> = jobs.iter().map(|(mode, _)| *mode).collect();
    modes.sort_unstable();
    modes.dedup();
    for mode in modes {
        // Each mode's prompt leaves room for THAT mode's answer budget, no more (#3715).
        let budget = if mode == "on" {
            m.a.max_tokens_thinking
        } else {
            m.a.max_tokens
        };
        let target = prompt_target(tok, ctx, budget)?;
        let pf = path_join(&m.a.work, &format!("prompt-{}-{mode}.txt", py_str(rid)));
        let slot = Slot {
            rid,
            mode,
            budget,
            target,
            pf,
        };
        measure_slot(m, acc, &slot, jobs)?;
    }
    Ok(())
}

/// The deferred serve cells of one model, through ONE `apr serve`, started lazily. Dropping
/// it kills the serve child by its recorded pid's group.
fn measure_deferred(m: &Model, rows: &mut [Py], deferred: Vec<Deferred>) -> Result<(), String> {
    let mut serve: Option<Serve> = None;
    for d in deferred {
        if serve.is_none() {
            let port = free_port()?;
            let port_s = port.to_string();
            let port_s = port_s.as_str();
            let argv =
                m.r.argv(&["serve", "run", m.path, "--port", port_s, "--gpu"]);
            serve = Some(Serve::start(&argv, port, m.a.serve_ceiling)?);
        }
        let Some(s) = serve.as_ref() else {
            continue;
        };
        let row = rows[d.idx].clone();
        let mut row = measure_serve(s, &d.prompt, d.mode, d.budget, row, m.r.timeout)?;
        fill_measured(&mut row, d.measured.as_ref());
        rows[d.idx] = row;
    }
    Ok(())
}

/// The cells of one model. Its serve cells run LAST: `apr serve` keeps the GPU lock for its
/// whole life, so one serve per model, started after every run/chat/code cell, never waits
/// on ourselves (#3715).
pub fn measure_item(
    r: &Runner,
    item: &Py,
    path: &str,
    ladder: &Py,
    rungs_doc: &Py,
    a: &Measure,
) -> Result<Vec<Py>, String> {
    let m = Model { r, item, path, a };
    let mut acc = Acc {
        rows: Vec::new(),
        deferred: Vec::new(),
        density: a.chars_per_token,
    };
    for ((rid, tok), jobs) in group_by_rung(cells_for(item, ladder, rungs_doc)?) {
        measure_rung(&m, &mut acc, &rid, tok, &jobs)?;
    }
    let Acc {
        mut rows, deferred, ..
    } = acc;
    measure_deferred(&m, &mut rows, deferred)?;
    Ok(rows)
}

// --------------------------------------------------------------------------------------- main

fn read_lines(path: &str) -> Result<Vec<String>, String> {
    std::fs::read(path)
        .map(|b| py_text_lines(&b))
        .map_err(|e| format!("FileNotFoundError: {path}: {e}"))
}

/// `dict(l.rstrip("\n").split("|", 1) for l in open(models) if "|" in l)`, later lines win.
fn model_paths(path: &str) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    for line in read_lines(path)? {
        if let Some((k, v)) = line.trim_end_matches('\n').split_once('|') {
            out.push((k.to_owned(), v.to_owned()));
        }
    }
    Ok(out)
}

fn path_for<'a>(paths: &'a [(String, String)], file: &Py) -> Option<&'a str> {
    let f = file.as_str()?;
    paths
        .iter()
        .rev()
        .find(|(k, _)| k == f)
        .map(|(_, v)| v.as_str())
}

/// The non-blank lines of a JSONL file, each `json.loads`ed.
fn jsonl(path: &str) -> Result<Vec<Py>, String> {
    let mut rows = Vec::new();
    for line in read_lines(path)? {
        if !py_strip(&line).is_empty() {
            rows.push(loads(&line).map_err(LoadErr::into_msg)?);
        }
    }
    Ok(rows)
}

/// inventory.jsonl + models (file|path) -> enriched inventory.jsonl (same rows, same order).
fn cmd_enrich(a: &Common) -> Result<String, String> {
    let paths = model_paths(&a.models)?;
    // inspect reads a header: no GPU, no lock
    let r = Runner {
        apr: &a.apr,
        lock: "",
        wait: 0,
        timeout: 120,
    };
    let rows = jsonl(&a.inventory)?;
    let f = std::fs::File::create(&a.out).map_err(|e| format!("OSError: {}: {e}", a.out))?;
    let mut f = std::io::BufWriter::new(f);
    for mut it in rows {
        let file = subscript(&it, "file")?;
        let path = path_for(&paths, file).unwrap_or("").to_owned();
        let (rc, out, _) = r.call(&["inspect", path.as_str(), "--json"], None)?;
        let ins = envelope(rc, &out)?.unwrap_or_else(Py::dict);
        let terms = inventory_terms(&ins, dict_get(&it, "bytes")?.clone())?;
        let Py::Dict(pairs) = terms else {
            return Err("inventory_terms returned a non-dict".to_owned());
        };
        if !matches!(it, Py::Dict(_)) {
            return Err(format!(
                "AttributeError: '{}' object has no attribute 'update'",
                it.type_name()
            ));
        }
        for (k, v) in pairs {
            it.set(&k, v);
        }
        f.write_all(format!("{}\n", dumps(&it)).as_bytes())
            .map_err(|e| format!("OSError: {}: {e}", a.out))?;
    }
    f.flush().map_err(|e| format!("OSError: {}: {e}", a.out))?;
    Ok(String::new())
}

/// What `measure` ends in: its stdout, or a refusal with its exit code.
enum Outcome {
    Out(String),
    Refuse(u8, String),
}

fn load_ladder(path: &str) -> Result<Py, String> {
    let raw = std::fs::read(path).map_err(|e| format!("FileNotFoundError: {path}: {e}"))?;
    let text = String::from_utf8_lossy(&raw);
    let v: serde_yaml::Value =
        serde_yaml::from_str(&text).map_err(|e| format!("yaml.YAMLError: {path}: {e}"))?;
    Ok(subscript(&from_yaml(&v), "ladder")?.clone())
}

/// `--only` keeps one inventory model, named by its file or as `inv:<file>`; true = skip `file`.
fn only_skips(only: &str, file: &Py) -> Result<bool, String> {
    if only.is_empty() {
        return Ok(false);
    }
    let f = file.as_str().ok_or_else(|| {
        format!(
            "TypeError: can only concatenate str (not \"{}\") to str",
            file.type_name()
        )
    })?;
    Ok(only != f && only != format!("inv:{f}"))
}

/// The judge owes nothing for a declaimed model (#4590): hours of cells would measure a claim
/// the release does not make.
fn note_declaimed(file: &Py, host: &Py, d: &Py) -> Result<String, String> {
    Ok(format!(
        "cells: {} DECLAIMED on {} (arch {}, #{}) -- not measured",
        py_str(file),
        py_str(host),
        py_str(subscript(d, "arch")?),
        py_str(subscript(d, "issue")?)
    ))
}

fn cmd_measure(a: &Measure) -> Result<Outcome, String> {
    let ladder = load_ladder(&a.ladder)?;
    let rungs_raw =
        std::fs::read(&a.rungs).map_err(|e| format!("FileNotFoundError: {}: {e}", a.rungs))?;
    let rungs_doc = loads(&crate::pystr::py_read_text(&rungs_raw)).map_err(LoadErr::into_msg)?;
    let paths = model_paths(&a.common.models)?;
    let lock = a.lock.as_deref().unwrap_or("");
    let host = Py::str(a.host.as_deref().unwrap_or(""));
    let only = a.only.as_deref().unwrap_or("");
    let r = Runner {
        apr: &a.common.apr,
        lock,
        wait: a.lock_wait,
        timeout: a.timeout,
    };
    let mut hosts = Vec::new();
    for h in or_list(dict_get(&ladder, "hosts")?)? {
        hosts.push(dict_get(&h, "id")?.clone());
    }
    let mut why = Vec::new();
    let (dec, drc) = declaimed(or_empty(dict_get(&ladder, "cells")?), &hosts, &mut why)?;
    if drc != 0 {
        let lines: Vec<String> = why
            .iter()
            .map(|w| format!("cells: refused -- {w}"))
            .collect();
        return Ok(Outcome::Refuse(1, lines.join("\n")));
    }
    let mut rows: Vec<Py> = Vec::new();
    let mut matched = false;
    // Line by line, as the original: a bad line is met only after the models before it.
    for line in read_lines(&a.common.inventory)? {
        if py_strip(&line).is_empty() {
            continue;
        }
        let it = loads(&line).map_err(LoadErr::into_msg)?;
        let file = subscript(&it, "file")?;
        if only_skips(only, file)? {
            continue;
        }
        matched = true;
        let d = find_declaimed(&dec, &host, dict_get(&it, "arch")?).filter(|d| d.truthy());
        if let Some(d) = d {
            eprintln!("{}", note_declaimed(file, &host, d)?);
            continue;
        }
        let path =
            path_for(&paths, file).ok_or_else(|| format!("KeyError: {}", py_repr_value(file)))?;
        rows.extend(measure_item(&r, &it, path, &ladder, &rungs_doc, a)?);
    }
    if !only.is_empty() && !matched {
        // A rung id selects no inventory model; zero rows at rc 0 would read as "this host
        // owes no cells" (#3715).
        return Ok(Outcome::Refuse(
            2,
            format!(
                "cells: refused -- --only {} names no inventory model; cells are measured per \
                 model",
                py_repr(only)
            ),
        ));
    }
    let n = rows.len();
    let pass = rows
        .iter()
        .filter(|r| py_eq(r.at("verdict"), &Py::str("pass")))
        .count();
    let refused = rows
        .iter()
        .filter(|r| py_eq(r.at("verdict"), &Py::str("refused")))
        .count();
    write_text(&a.common.out, &dumps_indent1(&Py::List(rows)))?;
    Ok(Outcome::Out(format!(
        "cells: {n} row(s), {pass} pass, {refused} refused\n"
    )))
}

/// Run one subcommand: its stdout, or `(exit code, stderr text)`.
pub fn run(cmd: &Sub) -> Result<String, (u8, String)> {
    match cmd {
        Sub::Gpumem => {
            let (t, f) = gpu_memory(&NVIDIA_SMI, "/proc/meminfo").map_err(|e| (1, e))?;
            let doc = Py::Dict(vec![
                kv("gpu_mem_total_bytes", opt_int(t)),
                kv("gpu_mem_free_bytes", opt_int(f)),
            ]);
            Ok(format!("{}\n", dumps(&doc)))
        }
        Sub::Enrich(a) => cmd_enrich(a).map_err(|e| (1, e)),
        Sub::Measure(a) => match cmd_measure(a).map_err(|e| (1, e))? {
            Outcome::Out(s) => Ok(s),
            Outcome::Refuse(code, why) => Err((code, why)),
        },
    }
}

#[cfg(test)]
#[path = "ladder_cells_produce_e2e.rs"]
mod e2e;

#[cfg(test)]
mod tests {
    use super::*;

    fn j(s: &str) -> Py {
        loads(s).expect("valid json")
    }

    fn row() -> Py {
        let item = j(r#"{"sha256": "ab", "file": "m.gguf"}"#);
        base_row(&item, &Py::str("code"), "off", &Py::str("4k"), 64).expect("a row")
    }

    #[test]
    fn thinking_evidence_reads_the_template() {
        assert_eq!(thinking_evidence("plain"), (vec![], false));
        let t = "{% if enable_thinking %}<think></think>{% endif %}add_generation_prompt<think>";
        assert_eq!(
            thinking_evidence(t),
            (vec!["enable_thinking", "<think>", "</think>"], false)
        );
        assert_eq!(
            thinking_evidence("x add_generation_prompt <think>\n"),
            (vec!["<think>"], true)
        );
        assert_eq!(
            thinking_evidence("<think> add_generation_prompt"),
            (vec!["<think>"], false)
        );
    }

    #[test]
    fn kv_bytes_follow_capacity_rs() {
        let md = j(r#"{"q.block_count": 28, "q.attention.head_count_kv": "2",
                      "q.attention.head_count": 12, "q.embedding_length": 1536}"#);
        // 28 layers x 2 kv heads x (128 + 128) x 4 B.
        assert_eq!(kv_bytes_per_token_f32(&md, &Py::str("q")), Ok(Some(57_344)));
        let hybrid = j(r#"{"h.block_count": 48, "h.attention.head_count_kv": 4,
                          "h.attention.key_length": 0, "h.attention.head_count": 16,
                          "h.embedding_length": 2048, "h.attention.value_length": 64,
                          "h.full_attention_interval": 4}"#);
        // key_length 0 falls through to 2048 // 16 = 128; 12 attention layers.
        assert_eq!(
            kv_bytes_per_token_f32(&hybrid, &Py::str("h")),
            Ok(Some(36_864))
        );
        let partial = j(r#"{"q.block_count": 28}"#);
        assert_eq!(kv_bytes_per_token_f32(&partial, &Py::str("q")), Ok(None));
        // A nonzero key_length wins over emb // heads; a zero value_length or interval is absent.
        let own = j(r#"{"k.block_count": 2, "k.attention.head_count_kv": 1,
                       "k.attention.key_length": 64, "k.attention.value_length": 0,
                       "k.attention.head_count": 4, "k.embedding_length": 1024,
                       "k.full_attention_interval": 0}"#);
        assert_eq!(kv_bytes_per_token_f32(&own, &Py::str("k")), Ok(Some(1024)));
        for (emb, heads) in [(1024, 0), (0, 4)] {
            let md = j(&format!(
                r#"{{"k.block_count": 2, "k.attention.head_count_kv": 1,
                    "k.attention.head_count": {heads}, "k.embedding_length": {emb}}}"#
            ));
            let got = kv_bytes_per_token_f32(&md, &Py::str("k"));
            assert_eq!(got, Ok(None), "emb {emb}, heads {heads}");
        }
        let huge = j(r#"{"k.block_count": "100000000000000000000",
                        "k.attention.head_count_kv": "100000000000000000000",
                        "k.attention.key_length": 1}"#);
        assert_eq!(
            kv_bytes_per_token_f32(&huge, &Py::str("k")),
            Err("OverflowError: a kv-cache term exceeds 128 bits".to_owned())
        );
    }

    #[test]
    fn floordiv_rounds_down_as_python_does() {
        let cases = [
            (7, 2, 3),
            (-7, 2, -4),
            (7, -2, -4),
            (-7, -2, 3),
            (6, 3, 2),
            (6, -3, -2),
            (-6, 3, -2),
            (0, -3, 0),
            (-1, 5, -1),
        ];
        for (a, b, q) in cases {
            assert_eq!(floordiv(a, b), Some(q), "{a} // {b}");
        }
        assert_eq!(floordiv(1, 0), None);
        assert_eq!(floordiv(i128::MIN, -1), None);
    }

    #[test]
    fn gpu_mem_reads_the_first_line_in_mib() {
        let first = gpu_mem(&["sh", "-c", "echo '24564, 1000'; echo '1, 2'"]);
        assert_eq!(first, Some((24_564 * MIB, 1_000 * MIB)));
        assert_eq!(gpu_mem(&["sh", "-c", "echo '[N/A], [N/A]'"]), None);
        assert_eq!(gpu_mem(&["sh", "-c", "echo 1, 2, 3"]), None);
        assert_eq!(gpu_mem(&["sh", "-c", "true"]), None);
        assert_eq!(gpu_mem(&["/nonexistent/nvidia-smi-5225"]), None);
    }

    #[test]
    fn inventory_terms_from_the_header() {
        let ins = j(
            r#"{"metadata": {"general.architecture": "qwen2", "general.file_type": "15",
            "qwen2.context_length": "32768", "qwen2.block_count": 28,
            "qwen2.attention.head_count": 12, "qwen2.attention.head_count_kv": 2,
            "qwen2.embedding_length": 1536,
            "tokenizer.chat_template": "enable_thinking add_generation_prompt"}}"#,
        );
        let t = inventory_terms(&ins, Py::Int(1000)).expect("terms");
        assert_eq!(
            dumps(&t),
            "{\"arch\": \"qwen2\", \"quant\": \"15\", \"context_length\": 32768, \
             \"thinking_markers\": [\"enable_thinking\"], \"generation_opens_think\": false, \
             \"thinking_modes\": [\"off\", \"on\"], \"chat_template_sha256\": \
             \"8e6a83e0c2b0d1e5b9ac1b3cb4f43d8b0de15fdb0f8a52aa62d1e8c0c2ed1a46\", \
             \"weights_bytes\": 1000, \"kv_bytes_per_token\": 57344, \"kv_dtype\": \"f32\", \
             \"workspace_bytes\": null}"
                .replace(
                    "8e6a83e0c2b0d1e5b9ac1b3cb4f43d8b0de15fdb0f8a52aa62d1e8c0c2ed1a46",
                    t.at("chat_template_sha256").as_str().unwrap_or("")
                )
        );
        assert_eq!(
            t.at("chat_template_sha256").as_str().map(str::len),
            Some(64)
        );
        let empty = inventory_terms(&Py::dict(), Py::None).expect("terms");
        assert_eq!(*empty.at("arch"), Py::None);
        assert_eq!(*empty.at("thinking_modes"), Py::None);
        assert_eq!(*empty.at("chat_template_sha256"), Py::None);
        let top = inventory_terms(&j(r#"{"architecture": "llama"}"#), Py::None).expect("t");
        assert_eq!(*top.at("arch"), Py::str("llama"));
        let unknown = inventory_terms(&j(r#"{"architecture": "unknown"}"#), Py::None).expect("t");
        assert_eq!(*unknown.at("arch"), Py::None);
    }

    #[test]
    fn sha256_of_the_template_is_the_hex_digest() {
        let t = inventory_terms(
            &j(r#"{"metadata": {"tokenizer.chat_template": "abc"}}"#),
            Py::None,
        )
        .expect("terms");
        assert_eq!(
            t.at("chat_template_sha256").as_str(),
            Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
    }

    #[test]
    fn meminfo_reads_total_and_available() {
        let text = "MemTotal:       1000 kB\nMemFree:  5 kB\nMemAvailable:    600 kB\n";
        assert_eq!(meminfo(text), Ok((Some(1_024_000), Some(614_400))));
        assert_eq!(meminfo("MemTotal: 1 kB\nnocolon\n"), Ok((None, None)));
        assert_eq!(meminfo("MemTotal: 1 kB\n"), Ok((None, None)));
        assert!(meminfo("MemTotal:\nMemAvailable: 1 kB\n").is_err());
    }

    #[test]
    fn build_prompt_plants_the_needle_first_and_asks_last() {
        let p = build_prompt(100, 4.5).expect("a prompt");
        assert!(p.starts_with("The passphrase is TANGERINE-4417. Remember it.\n"));
        assert!(p.ends_with(QUESTION));
        assert!(p.contains("Log entry 000000: the ferry"));
        // int(100 * 4.5 * 1.01) = 454 chars at least, and one filler line past it at most.
        assert!(p.len() >= 454 && p.len() < 454 + 82, "{}", p.len());
        let tiny = build_prompt(1, 4.5).expect("a prompt");
        assert_eq!(
            tiny,
            format!("The passphrase is {NEEDLE_WORD}. Remember it.\n{QUESTION}")
        );
        assert!(build_prompt(10, f64::INFINITY).is_err());
        // Filler stops on the line that reaches the want exactly: int(1 * c * 1.01) = exact.
        let line = "Log entry 000000: the ferry left the harbour on time and the weather stayed \
                    calm.\n"
            .len();
        let exact = tiny.len() + 2 * line;
        let p = build_prompt(1, (exact as f64 + 0.5) / 1.01).expect("a prompt");
        assert_eq!(p.len(), exact);
        // Each filler line is numbered in turn: the second one is 000001, never 000000 again.
        assert!(p.contains("Log entry 000001: the ferry"));
        assert!(!p.contains("Log entry 000002"));
    }

    #[test]
    fn a_slot_runs_only_its_own_modes_verbs_run_first() {
        let jobs = [
            ("off", Py::str("chat")),
            ("on", Py::str("code")),
            ("off", Py::str("run")),
        ];
        assert_eq!(
            slot_verbs(&jobs, "off"),
            vec![&Py::str("run"), &Py::str("chat")]
        );
        assert_eq!(slot_verbs(&jobs, "on"), vec![&Py::str("code")]);
    }

    #[test]
    fn a_prompt_leaves_room_for_the_answer_and_one_token() {
        // int(ctx) - budget - 1 when that is under the rung, else the rung.
        assert_eq!(prompt_target(9000, &Py::Int(8192), 64), Ok(8127));
        assert_eq!(prompt_target(4096, &Py::Int(8192), 64), Ok(4096));
        assert_eq!(prompt_target(9000, &Py::None, 64), Ok(9000));
        assert_eq!(prompt_target(9000, &Py::Int(0), 64), Ok(9000));
    }

    #[test]
    fn an_answer_of_exactly_120_chars_has_no_more_clause() {
        let r = finish(row(), 0, &"x".repeat(120), "").expect("row");
        let want = format!(
            "answer does not contain the needle planted at token 0: '{}'",
            "x".repeat(120)
        );
        assert_eq!(*r.at("reason"), Py::Str(want));
    }

    #[test]
    fn a_container_ran_is_unhashable_as_in_python() {
        assert_eq!(
            backend_name(&Py::List(vec![])),
            Err("TypeError: unhashable type: 'list'".to_owned())
        );
        assert_eq!(
            backend_name(&Py::dict()),
            Err("TypeError: unhashable type: 'dict'".to_owned())
        );
        assert_eq!(backend_name(&Py::Int(1)), Ok(Py::None));
        assert_eq!(backend_name(&Py::str("gpu")), Ok(Py::str("cuda")));
        assert_eq!(backend_name(&Py::str("cpu")), Ok(Py::str("cpu")));
        assert_eq!(backend_name(&Py::str("tpu")), Ok(Py::None));
    }

    #[test]
    fn text_of_a_truthy_non_string_is_a_type_error() {
        assert_eq!(text_of(Some(&Py::Int(0))), Ok(String::new()));
        assert_eq!(
            text_of(Some(&Py::Int(5))),
            Err("TypeError: an answer of type 'int' is not text".to_owned())
        );
    }

    #[test]
    fn chat_keeps_a_dict_line_without_backend_as_text() {
        let out =
            "{\"a\": 1}\nTANGERINE-4417\n{\"backend\": {\"ran\": \"gpu\", \"fell_back\": false}}\n";
        let r = chat_row(row(), 0, out, "").expect("row");
        assert_eq!(*r.at("answer_chars"), Py::Int(23));
        assert_eq!(*r.at("backend"), Py::str("cuda"));
    }

    #[test]
    fn mib_bytes_refuses_what_has_no_integer() {
        assert_eq!(mib_bytes("2"), Ok(Some(2 * MIB)));
        // The regex only captures digits and dots, but the fn takes any text: NaN is refused,
        // never truncated to 0.
        assert!(mib_bytes("nan").is_err());
        assert!(mib_bytes("inf").is_err());
    }

    #[test]
    fn gpu_memory_falls_back_to_meminfo() {
        let smi = ["sh", "-c", "echo '100, 50'"];
        let mib = Some(100 * MIB);
        assert_eq!(gpu_memory(&smi, "/nonexistent"), Ok((mib, Some(50 * MIB))));
        let dir = std::env::temp_dir().join(format!("lcp-meminfo-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a dir");
        let path = dir.join("meminfo");
        std::fs::write(&path, "MemTotal: 10 kB\nMemAvailable: 5 kB\n").expect("written");
        let path = path.to_str().expect("utf-8");
        let na = ["sh", "-c", "echo '[N/A], [N/A]'"];
        assert_eq!(gpu_memory(&na, path), Ok((Some(10_240), Some(5_120))));
        assert_eq!(gpu_memory(&na, "/nonexistent"), Ok((None, None)));
        std::fs::remove_dir_all(&dir).expect("removed");
    }

    #[test]
    fn split_thinking_takes_the_text_after_the_last_close() {
        assert_eq!(
            split_thinking("<think>a</think>b</think>  ans \n"),
            ("ans".to_owned(), Some(true))
        );
        assert_eq!(
            split_thinking("<think>still going"),
            (String::new(), Some(false))
        );
        assert_eq!(split_thinking("  plain \n"), ("plain".to_owned(), None));
    }

    #[test]
    fn refusal_reads_apr_arithmetic() {
        let e = "error: GPU capacity refused: = 29612 MiB, against 24000 MiB free of 24564 MiB";
        assert_eq!(
            refusal(e),
            Ok(Some((
                Some(29_612 * MIB),
                Some(24_000 * MIB),
                Some(24_564 * MIB)
            )))
        );
        assert_eq!(
            refusal("capacity refused, no numbers"),
            Ok(Some((None, None, None)))
        );
        assert_eq!(refusal("some other error"), Ok(None));
        let half = "capacity refused: = 1.5 MiB, against 0.5 MiB free of 2 MiB";
        assert_eq!(
            refusal(half),
            Ok(Some((Some(1_572_864), Some(524_288), Some(2 * MIB))))
        );
        assert!(refusal("capacity refused: = 1.2.3 MiB, against 1 MiB free of 2 MiB").is_err());
    }

    #[test]
    fn finish_judges_refusal_failure_and_needle() {
        let ok = finish(row(), 0, "the passphrase is TANGERINE-4417", "").expect("row");
        assert_eq!(*ok.at("verdict"), Py::str("pass"));
        assert_eq!(*ok.at("reason"), Py::str("ok"));
        assert_eq!(*ok.at("answer_chars"), Py::Int(32));
        let refused = finish(
            row(),
            1,
            "",
            "load\nGPU capacity refused: = 2 MiB, against 1 MiB free of 3 MiB\n",
        )
        .expect("row");
        assert_eq!(*refused.at("verdict"), Py::str("refused"));
        assert_eq!(*refused.at("required_bytes"), Py::Int(2 * MIB));
        assert_eq!(*refused.at("available_bytes"), Py::Int(MIB));
        assert_eq!(*refused.at("answer_chars"), Py::Int(0));
        let failed = finish(row(), 2, "", "a\nboom\n").expect("row");
        assert_eq!(*failed.at("reason"), Py::str("rc 2: boom"));
        assert_eq!(*failed.at("verdict"), Py::str("fail"));
        let quiet = finish(row(), 2, "", " \n").expect("row");
        assert_eq!(*quiet.at("reason"), Py::str("rc 2: no stderr"));
        let open = finish(row(), 0, "<think>hmm", "").expect("row");
        assert_eq!(
            *open.at("reason"),
            Py::str("thinking never closed within max_tokens 64: no </think> in the output")
        );
        assert_eq!(*open.at("think_closed"), Py::Bool(false));
        let wrong = finish(row(), 0, "Paris", "").expect("row");
        assert_eq!(
            *wrong.at("reason"),
            Py::str("answer does not contain the needle planted at token 0: 'Paris'")
        );
        let long = finish(row(), 0, &"x".repeat(125), "").expect("row");
        let want = format!(
            "answer does not contain the needle planted at token 0: '{}' ... and 5 more chars",
            "x".repeat(120)
        );
        assert_eq!(*long.at("reason"), Py::Str(want));
    }

    #[test]
    fn code_row_backend_comes_from_the_document_4986() {
        let gpu = r#"{"result": "TANGERINE-4417", "backend": {"ran": "gpu", "fell_back": false}}"#;
        let r = code_row(row(), 0, gpu, "").expect("row");
        assert_eq!(*r.at("backend"), Py::str("cuda"));
        assert_eq!(*r.at("fallback"), Py::Bool(false));
        assert_eq!(*r.at("verdict"), Py::str("pass"));
        let cpu = r#"{"result": "TANGERINE-4417", "backend": {"ran": "cpu", "fell_back": true}}"#;
        let r = code_row(row(), 0, cpu, "").expect("row");
        assert_eq!(*r.at("backend"), Py::str("cpu"));
        assert_eq!(*r.at("fallback"), Py::Bool(true));
        assert!(r.get("reason_backend").is_none());
        // No backend in the document: both stay null, never a guess, and no reason is added.
        let r = code_row(row(), 0, r#"{"result": "TANGERINE-4417"}"#, "").expect("row");
        assert_eq!(*r.at("backend"), Py::None);
        assert_eq!(*r.at("fallback"), Py::None);
        assert!(r.get("reason_backend").is_none());
        // Not JSON: rc 3, nothing claimed.
        let r = code_row(row(), 0, "not json", "").expect("row");
        assert_eq!(*r.at("rc"), Py::Int(3));
        assert_eq!(*r.at("backend"), Py::None);
        assert_eq!(*r.at("reason"), Py::str("rc 3: no stderr"));
    }

    #[test]
    fn run_and_chat_rows_read_their_envelopes() {
        let out = r#"{"text": "TANGERINE-4417", "prompt_tokens": 4100,
                      "backend": {"ran": "gpu", "fell_back": false}}"#;
        let r = run_row(row(), 0, out, "").expect("row");
        assert_eq!(*r.at("prompt_tokens"), Py::Int(4100));
        assert_eq!(*r.at("backend"), Py::str("cuda"));
        let bad = run_row(row(), 0, "{", "e").expect("row");
        assert_eq!(
            *bad.at("reason"),
            Py::str("rc 3: e | stdout was not the --format json envelope")
        );
        let chat = "TANGERINE-4417\n{\"backend\": {\"ran\": \"cpu\", \"fell_back\": true}}\n";
        let c = chat_row(row(), 0, chat, "").expect("row");
        assert_eq!(*c.at("backend"), Py::str("cpu"));
        assert_eq!(*c.at("verdict"), Py::str("pass"));
    }

    #[test]
    fn serve_row_reads_used_gpu() {
        let body = j(r#"{"choices": [{"message": {"content": "TANGERINE-4417",
                        "reasoning_content": "thought"}}], "usage": {"prompt_tokens": 9},
                        "used_gpu": true}"#);
        let r = serve_row(row(), 0, &body, "").expect("row");
        assert_eq!(*r.at("backend"), Py::str("cuda"));
        assert_eq!(*r.at("fallback"), Py::Bool(false));
        assert_eq!(*r.at("think_closed"), Py::Bool(true));
        assert_eq!(*r.at("prompt_tokens"), Py::Int(9));
        let r = serve_row(row(), 503, &Py::dict(), "GPU busy").expect("row");
        assert_eq!(*r.at("backend"), Py::None);
        assert_eq!(*r.at("reason"), Py::str("rc 503: GPU busy"));
    }

    #[test]
    fn ask_body_matches_json_dumps() {
        assert_eq!(
            ask_body("hi", "on", 8),
            "{\"model\": \"apr\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \
             \"max_tokens\": 8, \"stream\": false, \"chat_template_kwargs\": \
             {\"enable_thinking\": true}}"
        );
    }

    #[test]
    fn cells_for_is_the_judges_enumeration() {
        let ladder = j(r#"{"cells": {"verbs": ["run", "serve"],
                          "long_rungs_for": {"families": ["qwen2"], "representatives": {}}}}"#);
        let rungs = j(r#"{"schema": "apr-release-context-rungs/v1",
                         "consumers": [{"name": "x", "max_prompt_tokens": 7000}],
                         "rungs": [{"id": "4k", "tokens": 4096}, {"id": "8k", "tokens": 8192}]}"#);
        let item = j(r#"{"arch": "qwen2", "context_length": 32768,
                        "thinking_markers": ["enable_thinking"]}"#);
        let cells = cells_for(&item, &ladder, &rungs).expect("cells");
        let got: Vec<(String, Option<i128>, &str, String)> = cells
            .iter()
            .map(|c| (py_str(&c.rid), c.tok, c.mode, py_str(&c.verb)))
            .collect();
        let want = [
            ("4k", "off", "run"),
            ("4k", "on", "run"),
            ("4k", "off", "serve"),
            ("4k", "on", "serve"),
            ("8k", "off", "run"),
            ("8k", "on", "run"),
            ("8k", "off", "serve"),
            ("8k", "on", "serve"),
        ];
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert_eq!((g.0.as_str(), g.2, g.3.as_str()), w);
        }
        assert_eq!(cells[0].tok, Some(4096));
        let off_only = j(r#"{"arch": "qwen2", "thinking_markers": []}"#);
        let cells = cells_for(&off_only, &ladder, &rungs).expect("cells");
        assert!(cells.iter().all(|c| c.mode == "off"));
        let bad = j(r#"{"schema": "other"}"#);
        let err = cells_for(&item, &ladder, &bad).expect_err("a refusal");
        assert!(err.starts_with("cells: refused -- FAIL  "), "{err}");
    }

    #[test]
    fn groups_keep_first_seen_order_then_sort_by_size() {
        let cell = |rid: &str, tok: Option<i128>, mode: &'static str| Cell {
            rid: Py::str(rid),
            tok,
            mode,
            verb: Py::str("run"),
        };
        let g = group_by_rung(vec![
            cell("8k", Some(8192), "off"),
            cell("x", None, "off"),
            cell("4k", Some(4096), "off"),
            cell("8k", Some(8192), "on"),
        ]);
        let order: Vec<String> = g.iter().map(|((rid, _), _)| py_str(rid)).collect();
        assert_eq!(order, ["x", "4k", "8k"]);
        assert_eq!(g[2].1.len(), 2);
    }

    #[test]
    fn helpers_match_python() {
        assert_eq!(path_join("w", "p.txt"), "w/p.txt");
        assert_eq!(path_join("w/", "p.txt"), "w/p.txt");
        assert_eq!(path_join("", "p.txt"), "p.txt");
        assert_eq!(path_join("w", "/abs"), "/abs");
        assert_eq!(py_int_arg(" 1_0 "), Ok(10));
        assert_eq!(py_int_arg("-5"), Ok(-5));
        assert!(py_int_arg("0x10").is_err());
        assert_eq!(py_float_arg(" 4.5\n"), Ok(4.5));
        assert_eq!(py_float_arg("1_0.5"), Ok(10.5));
        assert!(py_float_arg("1__0").is_err());
        assert!(py_float_arg("_1").is_err());
        assert!(py_float_arg("1_").is_err());
        assert!(py_float_arg("nan").is_ok_and(f64::is_nan));
        let r = Runner {
            apr: "apr",
            lock: "/l",
            wait: 5,
            timeout: 1,
        };
        assert_eq!(
            r.argv(&["run"]).join(" "),
            "flock -E 75 -w 5 /l choom -n 1000 -- apr run"
        );
    }

    #[test]
    fn runner_call_returns_rc_stdout_stderr() {
        let r = Runner {
            apr: "sh",
            lock: "",
            wait: 0,
            timeout: 10,
        };
        let got = r.call(&["-c", "cat; echo err >&2; exit 3"], Some("in"));
        assert_eq!(got, Ok((3, "in".to_owned(), "err\n".to_owned())));
    }

    #[test]
    fn model_paths_split_at_the_first_bar_and_later_lines_win() {
        let dir = std::env::temp_dir().join(format!("lcp-paths-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let f = dir.join("models.txt");
        std::fs::write(&f, "a|/x\nno bar\nb|/y|z\na|/w\n\n").expect("models");
        let f = f.display().to_string();
        let paths = model_paths(&f).expect("read");
        assert_eq!(
            paths,
            vec![
                ("a".to_owned(), "/x".to_owned()),
                ("b".to_owned(), "/y|z".to_owned()),
                ("a".to_owned(), "/w".to_owned()),
            ]
        );
        assert_eq!(path_for(&paths, &Py::str("a")), Some("/w"));
        assert_eq!(path_for(&paths, &Py::str("b")), Some("/y|z"));
        assert_eq!(path_for(&paths, &Py::str("c")), None);
        assert_eq!(path_for(&paths, &Py::Int(1)), None);
        let rows_f = dir.join("rows.jsonl");
        std::fs::write(&rows_f, "{\"a\": 1}\n  \n\n[2]\n").expect("jsonl");
        let rows = jsonl(&rows_f.display().to_string()).expect("rows");
        assert_eq!(rows, vec![j(r#"{"a": 1}"#), j("[2]")]);
        assert_eq!(
            read_lines(&f).expect("lines"),
            vec!["a|/x\n", "no bar\n", "b|/y|z\n", "a|/w\n", "\n"]
        );
        let _ = std::fs::remove_dir_all(&dir);
        assert!(read_lines(&f).is_err_and(|e| e.starts_with("FileNotFoundError: ")));
    }

    #[test]
    fn only_skips_every_other_model() {
        let f = Py::str("m.gguf");
        assert_eq!(only_skips("", &f), Ok(false));
        assert_eq!(only_skips("m.gguf", &f), Ok(false));
        assert_eq!(only_skips("inv:m.gguf", &f), Ok(false));
        assert_eq!(only_skips("n.gguf", &f), Ok(true));
        assert_eq!(only_skips("inv:n.gguf", &f), Ok(true));
        assert_eq!(only_skips("", &Py::Int(1)), Ok(false));
        assert!(only_skips("x", &Py::Int(1)).is_err_and(|e| e.contains("\"int\"")));
    }

    #[test]
    fn note_declaimed_names_the_arch_and_issue() {
        let d = j(r#"{"arch": "qwen2", "issue": 4590}"#);
        assert_eq!(
            note_declaimed(&Py::str("m.gguf"), &Py::str("t"), &d),
            Ok("cells: m.gguf DECLAIMED on t (arch qwen2, #4590) -- not measured".to_owned())
        );
        assert!(note_declaimed(&Py::str("m"), &Py::str("t"), &j("{}")).is_err());
    }

    #[test]
    fn counts_compare_as_python_numbers() {
        assert_eq!(as_float(&Py::Bool(true)), Ok(1.0));
        assert_eq!(as_float(&Py::Bool(false)), Ok(0.0));
        assert_eq!(as_float(&Py::Int(7)), Ok(7.0));
        assert_eq!(as_float(&Py::Float(2.5)), Ok(2.5));
        assert!(as_float(&Py::str("7")).is_err());
        for (pt, want) in [
            (Py::Bool(true), [true, true, false]),
            (Py::Int(1), [true, true, false]),
            (Py::Float(1.0), [true, true, false]),
        ] {
            let got = [0, 1, 2].map(|t| reaches(&pt, t).expect("a number"));
            assert_eq!(got, want, "{pt:?}");
        }
        assert!(reaches(&Py::None, 0).is_err());
    }

    #[test]
    fn fill_measured_fills_only_a_missing_count() {
        let mut r = row();
        fill_measured(&mut r, None);
        assert_eq!(r, row());
        fill_measured(&mut r, Some(&Py::Int(9)));
        assert_eq!(r.at("prompt_tokens"), &Py::Int(9));
        assert_eq!(r.at("prompt_tokens_source"), &Py::str(SOURCE));
        let mut r = row();
        r.set("prompt_tokens", Py::Int(3));
        let before = r.clone();
        fill_measured(&mut r, Some(&Py::Int(9)));
        assert_eq!(r, before);
    }

    #[test]
    fn rungs_group_by_id_and_size_then_sort_by_size() {
        let cell = |rid: &str, tok: Option<i128>, mode: &'static str| Cell {
            rid: Py::str(rid),
            tok,
            mode,
            verb: Py::str("run"),
        };
        let groups = group_by_rung(vec![
            cell("8k", Some(8192), "off"),
            cell("a", Some(4096), "off"),
            cell("b", Some(4096), "off"),
            cell("8k", Some(8192), "on"),
            cell("8k", Some(1), "on"),
            cell("x", None, "off"),
        ]);
        let keys: Vec<(String, Option<i128>, usize)> = groups
            .iter()
            .map(|((rid, tok), jobs)| (py_str(rid), *tok, jobs.len()))
            .collect();
        assert_eq!(
            keys,
            vec![
                ("x".to_owned(), None, 1),
                ("8k".to_owned(), Some(1), 1),
                ("a".to_owned(), Some(4096), 1),
                ("b".to_owned(), Some(4096), 1),
                ("8k".to_owned(), Some(8192), 2),
            ]
        );
    }

    #[test]
    fn write_text_writes_or_names_the_path() {
        let dir = std::env::temp_dir().join(format!("lcp-write-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let f = dir.join("t.txt").display().to_string();
        assert_eq!(write_text(&f, "hé"), Ok(()));
        assert_eq!(std::fs::read_to_string(&f).expect("read"), "hé");
        let _ = std::fs::remove_dir_all(&dir);
        assert!(write_text(&f, "x").is_err_and(|e| e.starts_with("OSError: ")));
    }
}
