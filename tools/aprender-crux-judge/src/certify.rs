//! The port of `crux_prompt_certify.py`: `certify`, which writes the offline
//! certification receipt of the prompt set (#3962 Q1), and `check`, which
//! holds a receipt against the prompt set it names, with the judge's
//! `certification_ok` wrapper around it (#3962 J2).
//!
//! A prompt is admitted for a (model, quant) only when every leg is correct
//! under every thinking mode the model has:
//!
//! ```text
//! ggml@bf16    llama.cpp / ollama / llamafile on the model's BF16 GGUF
//! hf@bf16      transformers on the pinned source weights
//! vllm@bf16    vLLM on the same pinned source weights
//! ggml@quant   the ggml family on the IDENTICAL quantized GGUF apr is gated on
//! ```
//!
//! A leg with no row, a refused row, or any wrong row rejects the prompt, and
//! the reason names the first such cell. Each engine's output is read through
//! the judge's own parsers (`judge::engine_entry`), so certification and the
//! gate cannot disagree about what an engine said.

use sha2::{Digest, Sha256};

use crate::judge::engine_entry;
use crate::oracles::{evaluate, strip_think, validate_set};
use crate::pyerr::{PyErr, PyResult};
use crate::pyio::{os_err, path_arg, read_bytes, read_text_strict};
use crate::pyjson::{dump_indent, loads};
use crate::pyre::think_either;
use crate::pyval::{
    concat_str, dict, no_attr, py_eq, py_join, py_ne, py_repr, py_sorted, py_sorted_by,
    py_splitlines, py_str, py_strip, repr_str, slice_chars, Dict, Val,
};

pub const SCHEMA: &str = "crux-prompt-certification/v1";

/// `sha256_path(p)`: the hex digest of the file's bytes.
pub fn sha256_path(path: &str) -> PyResult<String> {
    let digest = Sha256::digest(read_bytes(path)?);
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}

/// `check(a)` with its stdout captured -> (rc, what it printed).
pub fn check(prompts: &str, receipt_path: &str) -> PyResult<(i32, String)> {
    let mut out = String::new();
    let receipt = loads(&read_text_strict(receipt_path)?)?;
    let schema = receipt.get("schema")?;
    if py_ne(&schema, &Val::str(SCHEMA)) {
        out.push_str(&format!(
            "refused: receipt schema {}, want {}\n",
            py_repr(&schema),
            repr_str(SCHEMA)
        ));
        return Ok((1, out));
    }
    let have_sha = sha256_path(prompts)?;
    let receipt_sha = py_str(&receipt.get("prompts_sha256")?);
    if py_ne(
        &receipt.get("prompts_sha256")?,
        &Val::str(have_sha.as_str()),
    ) {
        out.push_str(&format!(
            "refused: {prompts} is sha256 {}, the certification covers {} - an edited prompt set is \
             uncertified until it is certified again\n",
            &have_sha[..12],
            slice_chars(&receipt_sha, 0, 12)
        ));
        return Ok((1, out));
    }
    for u in receipt
        .get("uncontrolled_detail")?
        .or(Val::List(vec![]))
        .iter()?
    {
        let model = py_str(&u.item("model")?);
        let thinking = py_str(&u.item("thinking")?);
        let verbs = py_join(", ", &u.item("verbs")?)?;
        out.push_str(&format!(
            "note: {model} thinking={thinking} has no certified control for {verbs}\n"
        ));
    }
    let admitted = receipt.item("admitted")?;
    let Val::Dict(by_sha) = &admitted else {
        return Err(no_attr(&admitted, "values"));
    };
    let mut total = 0usize;
    for v in by_sha.values() {
        total += v.len()?;
    }
    out.push_str(&format!("certified: {total} (model, prompt) admissions\n"));
    Ok((0, out))
}

/// `certification_ok(prompts_path, receipt_path)` -> True, or the reason the
/// certifier refused.
pub fn certification_ok(prompts: &str, receipt: Option<&str>) -> PyResult<Val> {
    let Some(receipt) = receipt.filter(|r| !r.is_empty()) else {
        return Ok(Val::str("no --certification receipt was given"));
    };
    match check(prompts, receipt) {
        Ok((0, _)) => Ok(Val::Bool(true)),
        Ok((rc, printed)) => {
            let said = py_strip(&printed);
            Ok(Val::Str(if said.is_empty() {
                format!("refused (rc {rc})")
            } else {
                said.to_string()
            }))
        }
        Err(e) if e.declines() => Ok(Val::Str(format!("certification unreadable: {e}"))),
        Err(e) => Err(e),
    }
}

// ------------------------------------------------------------ the generator

const GGML: [&str; 3] = ["llama.cpp", "ollama", "llamafile"];
const LEGS: [&str; 4] = ["ggml@bf16", "hf@bf16", "vllm@bf16", "ggml@quant"];
const PRE_3990: &str =
    "pre-#3990 driver row: the template's think opener is unknown, so the reply cannot be split";

/// `v in (a, b, ...)` for a tuple of str constants.
fn is_in(v: &Val, set: &[&str]) -> bool {
    set.iter().any(|s| py_eq(v, &Val::str(*s)))
}

/// `model.get("thinking") or ["off"]`.
fn modes_of(model: &Val) -> PyResult<Vec<Val>> {
    model
        .get("thinking")?
        .or(Val::List(vec![Val::str("off")]))
        .iter()
}

/// `sorted(model["quants"].items())`. Keys are unique, so the key decides.
fn sorted_quants(model: &Val) -> PyResult<Vec<(Val, Val)>> {
    let quants = model.item("quants")?;
    let Val::Dict(q) = &quants else {
        return Err(no_attr(&quants, "items"));
    };
    let items: Vec<(Val, Val)> = q.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    py_sorted_by(&items, &|(k, _): &(Val, Val)| k.clone())
}

/// `str(Path(p))` on POSIX: repeated and trailing slashes and `.` parts go;
/// exactly two leading slashes stay.
fn pathlib_str(p: &str) -> String {
    let root = if p.starts_with("//") && !p.starts_with("///") {
        "//"
    } else if p.starts_with('/') {
        "/"
    } else {
        ""
    };
    let body: Vec<&str> = p
        .split('/')
        .filter(|s| !s.is_empty() && *s != ".")
        .collect();
    if root.is_empty() && body.is_empty() {
        return ".".to_string();
    }
    format!("{root}{}", body.join("/"))
}

/// `read_rows(paths)`: the `kind == "gen"` rows of every manifest, each
/// tagged `_at` = `path#L<n>`.
fn read_rows(paths: &[String]) -> PyResult<Vec<Val>> {
    let mut rows = Vec::new();
    for p in paths {
        let text = read_text_strict(p)?;
        for (n, line) in py_splitlines(&text).into_iter().enumerate() {
            if py_strip(line).is_empty() {
                continue;
            }
            let mut row = loads(line)?;
            let tn = row.type_name();
            let Val::Dict(d) = &mut row else {
                return Err(PyErr::type_err(if tn == "list" {
                    "list indices must be integers or slices, not str".to_string()
                } else {
                    format!("'{tn}' object does not support item assignment")
                }));
            };
            d.put("_at", Val::Str(format!("{p}#L{}", n + 1)));
            rows.push(row);
        }
    }
    let mut gen = Vec::with_capacity(rows.len());
    for r in rows {
        if py_eq(&r.get("kind")?, &Val::str("gen")) {
            gen.push(r);
        }
    }
    Ok(gen)
}

/// `leg_of(row, model, quant_sha)`: which leg a row is evidence for, if any.
fn leg_of(row: &Val, model: &Val, quant_sha: &Val) -> PyResult<Option<String>> {
    let eng = row.get("engine")?;
    if is_in(&eng, &GGML) {
        if py_eq(&row.get("model_sha256")?, &model.item("bf16_gguf")?) {
            return Ok(Some("ggml@bf16".to_string()));
        }
        if py_eq(&row.get("model_sha256")?, quant_sha) {
            return Ok(Some("ggml@quant".to_string()));
        }
        return Ok(None);
    }
    if is_in(&eng, &["hf", "vllm"])
        && py_eq(
            &row.get("source")?.or(dict(vec![])).get("repo")?,
            &model.item("source")?.item("repo")?,
        )
    {
        if py_ne(
            &row.get("source")?.or(dict(vec![])).get("revision")?,
            &model.item("source")?.item("revision")?,
        ) {
            return Ok(None); // a moving or different revision is not the pinned source
        }
        return Ok(Some(format!("{}@bf16", py_str(&eng))));
    }
    Ok(None)
}

/// A row's reply for the oracles, or why there is none.
enum Reply {
    /// `(text, turns)`; turns `None` when the engine reported none.
    Text(Val, Option<Val>),
    Why(Val),
}

/// `driver_raw(row)`: hf and vLLM split the think block off before writing
/// `text`; rebuild the RAW reply so the oracle sees the shape ggml sends it.
///
/// - fixed drivers (#3990) record `raw_text` and
///   `reported.prompt_opens_think`; the opener is restored before judging.
/// - a thinking-ON row from an older driver carries neither. Its `text` may
///   be a whole unclosed reasoning block, so the row is refused, never judged.
/// - older thinking-OFF rows: `reasoning` + `text` (empty text = unclosed).
fn driver_raw(row: &Val) -> PyResult<Option<Reply>> {
    let read = || -> PyResult<Val> { loads(&read_text_strict(&path_arg(&row.item("stdout")?)?)?) };
    let doc = match read() {
        Ok(doc) => doc,
        Err(e) if e.unreadable() => return Ok(None),
        Err(e) => return Err(e),
    };
    let reported = doc.get("reported")?.or(dict(vec![]));
    let turns = doc.get("turns")?.or(Val::None);
    let turns = (!turns.is_none()).then_some(turns);
    if matches!(doc.get("raw_text")?, Val::Str(_)) && reported.contains_str("prompt_opens_think")? {
        let opener = if reported.item("prompt_opens_think")?.truthy() {
            "<think>"
        } else {
            ""
        };
        let raw = concat_str(opener, &doc.item("raw_text")?)?;
        return Ok(Some(Reply::Text(Val::Str(raw), turns)));
    }
    if py_eq(&row.get("thinking")?, &Val::str("on")) {
        return Ok(Some(Reply::Why(Val::str(PRE_3990))));
    }
    let reasoning = doc.get("reasoning")?;
    if !reasoning.truthy() {
        return Ok(None);
    }
    let head = concat_str("<think>", &reasoning)?;
    let text = doc.get("text")?.or(Val::str(""));
    let tail = if text.truthy() {
        concat_str("</think>", &text)?
    } else {
        String::new()
    };
    Ok(Some(Reply::Text(Val::Str(head + &tail), turns)))
}

/// `think_state(raw)`: whether a reply's think block closed.
fn think_state(raw: Option<&Val>) -> &'static str {
    let Some(Val::Str(raw)) = raw else {
        return "none";
    };
    if !think_either().is_match(raw) {
        return "none";
    }
    if strip_think(raw).is_none() {
        "unclosed"
    } else {
        "closed"
    }
}

/// `reply_of(row, prompt)`. vLLM writes the same JSON as hf (row contract v1).
fn reply_of(row: &Val, prompt: &Val) -> PyResult<Reply> {
    let r = match row {
        Val::Dict(d) if py_eq(&row.get("engine")?, &Val::str("vllm")) => {
            let mut d = d.clone();
            d.put("engine", Val::str("hf"));
            Val::Dict(d)
        }
        _ => row.clone(),
    };
    let e = engine_entry(&r, prompt, &Val::None)?;
    let rc = row.get("rc")?;
    if py_ne(&rc, &Val::int(0)) {
        return Ok(Reply::Why(Val::Str(format!("exit {}", py_str(&rc)))));
    }
    if is_in(&row.get("engine")?, &["hf", "vllm"]) {
        if let Some(got) = driver_raw(row)? {
            return Ok(got);
        }
    }
    let field = |k: &str| e.get_str(k).cloned().unwrap_or(Val::None);
    let answer = field("answer");
    if answer.is_none() {
        return Ok(Reply::Why(field("why").or(Val::str("no answer"))));
    }
    // An engine that reports no turns gives [] through the judge's parser:
    // that is "not reported", not zero.
    let turns = field("turns");
    Ok(Reply::Text(answer, turns.truthy().then_some(turns)))
}

/// What `certify_one` returns: the first failure (None = admitted in every
/// mode), the cells, and {thinking: admitted in that mode}.
struct One {
    first_bad: Option<String>,
    cells: Vec<Dict>,
    by_mode: Dict,
}

/// The rows of one (prompt, thinking, leg). Any verb counts: certification
/// asks whether the PROMPT is answerable, not whether an interface works
/// (the gate's job).
fn leg_rows<'a>(
    rows: &'a [Val],
    prompt: &Val,
    model: &Val,
    quant_sha: &Val,
    thinking: &Val,
    leg: &str,
) -> PyResult<Vec<&'a Val>> {
    let mut mine = Vec::new();
    for r in rows {
        if py_eq(&r.get("prompt_id")?, &prompt.item("id")?)
            && py_eq(&r.get("thinking")?, thinking)
            && leg_of(r, model, quant_sha)?.as_deref() == Some(leg)
        {
            mine.push(r);
        }
    }
    Ok(mine)
}

/// One row's cell, and its failure line when it is not correct.
fn row_cell(r: &Val, prompt: &Val, leg: &str, thinking: &Val) -> PyResult<(Dict, Option<String>)> {
    let (correct, why, extracted, think) = match reply_of(r, prompt)? {
        Reply::Text(text, turns) => {
            let v = evaluate(prompt, &text, turns.as_ref())?;
            let think = think_state(Some(&text));
            (
                v.correct,
                v.why.map_or(Val::None, Val::Str),
                v.extracted,
                think,
            )
        }
        Reply::Why(why) => (false, why, Val::None, "none"),
    };
    let engine = r.item("engine")?;
    let verb = r.item("verb")?;
    let host = r.get("host")?;
    let bad = (!correct).then(|| {
        format!(
            "{leg} {} {} thinking={} on {}: {}",
            py_str(&engine),
            py_str(&verb),
            py_str(thinking),
            py_str(&host),
            py_str(&why)
        )
    });
    let cell = Dict::from_pairs(vec![
        ("leg", Val::str(leg)),
        ("thinking", thinking.clone()),
        ("engine", engine),
        ("verb", verb),
        ("host", host),
        ("correct", Val::Bool(correct)),
        ("why", why),
        ("extracted", extracted),
        ("think", Val::str(think)),
        ("max_tokens", r.get("max_tokens")?),
        ("row", r.item("_at")?),
    ]);
    Ok((cell, bad))
}

fn certify_one(prompt: &Val, model: &Val, quant_sha: &Val, rows: &[Val]) -> PyResult<One> {
    let mut cells: Vec<Dict> = Vec::new();
    let mut first_bad: Option<String> = None;
    let mut by_mode = Dict::new();
    for thinking in modes_of(model)? {
        let mode_bad = first_bad.clone();
        for leg in LEGS {
            let mine = leg_rows(rows, prompt, model, quant_sha, &thinking, leg)?;
            if mine.is_empty() {
                first_bad
                    .get_or_insert_with(|| format!("{leg} thinking={}: no row", py_str(&thinking)));
                cells.push(Dict::from_pairs(vec![
                    ("leg", Val::str(leg)),
                    ("thinking", thinking.clone()),
                    ("correct", Val::Bool(false)),
                    ("why", Val::str("no row")),
                    ("row", Val::None),
                ]));
                continue;
            }
            for r in mine {
                let (cell, bad) = row_cell(r, prompt, leg, &thinking)?;
                if let Some(bad) = bad {
                    first_bad.get_or_insert(bad);
                }
                cells.push(cell);
            }
        }
        let mode_wrong = cells.iter().any(|c| {
            c.get_str("thinking").is_some_and(|t| py_eq(t, &thinking))
                && !c.get_str("correct").is_some_and(Val::truthy)
        });
        by_mode.set(thinking, Val::Bool(first_bad == mode_bad && !mode_wrong))?;
    }
    Ok(One {
        first_bad,
        cells,
        by_mode,
    })
}

/// `closure(cells)`: per "model|prompt", each engine's thinking-ON think state.
fn closure(cells: &[Dict]) -> Dict {
    let mut out = Dict::new();
    for c in cells {
        let field = |k: &str| c.get_str(k).cloned().unwrap_or(Val::None);
        if py_ne(&field("thinking"), &Val::str("on")) || !field("engine").truthy() {
            continue;
        }
        let key = format!(
            "{}|{}",
            py_str(&field("model")),
            py_str(&field("prompt_id"))
        );
        let mut inner = match out.get_str(&key) {
            Some(Val::Dict(d)) => d.clone(),
            _ => Dict::new(),
        };
        inner.put(
            &format!("{}:{}", py_str(&field("leg")), py_str(&field("engine"))),
            field("think"),
        );
        out.put(&key, Val::Dict(inner));
    }
    out
}

/// `d[k].append(v)` for a list value.
fn append_at(d: &mut Dict, k: &Val, v: Val) -> PyResult<()> {
    match d.get_mut(k)? {
        Some(Val::List(l)) => {
            l.push(v);
            Ok(())
        }
        Some(other) => Err(no_attr(other, "append")),
        None => Err(PyErr::key(py_repr(k))),
    }
}

/// `d[k]` for a dict value.
fn dict_at<'a>(d: &'a mut Dict, k: &Val) -> PyResult<&'a mut Dict> {
    match d.get_mut(k)? {
        Some(Val::Dict(inner)) => Ok(inner),
        Some(other) => Err(PyErr::type_err(format!(
            "'{}' object does not support item assignment",
            other.type_name()
        ))),
        None => Err(PyErr::key(py_repr(k))),
    }
}

/// The `certify` subcommand's arguments.
pub struct CertifyArgs {
    pub prompts: String,
    pub inventory: String,
    pub apr_commit: String,
    pub out: String,
    pub manifests: Vec<String>,
}

/// `certify(a)`: write the receipt to `a.out` and print one admission line
/// per (model, quant). Ok(2) when the prompt set fails its lint.
pub fn certify(a: &CertifyArgs) -> PyResult<u8> {
    let doc = loads(&read_text_strict(&a.prompts)?)?;
    let errs = validate_set(&doc)?;
    if !errs.is_empty() {
        eprintln!(
            "crux_prompt_certify: the prompt set is invalid:\n  {}",
            errs.join("\n  ")
        );
        return Ok(2);
    }
    let inventory = loads(&read_text_strict(&a.inventory)?)?;
    let rows = read_rows(&a.manifests)?;
    let models = inventory.iter()?;
    let prompts = doc.item("prompts")?.iter()?;
    let Admissions {
        admitted,
        rejected,
        mut by_thinking,
        cells,
    } = admit_all(&models, &prompts, &rows)?;
    let detail = lane_gaps(&models, &prompts, &mut by_thinking)?;
    let mut uncontrolled: Vec<String> = detail
        .iter()
        .map(|u| py_str(u.get_str("model").unwrap_or(&Val::None)))
        .collect();
    uncontrolled.sort();
    uncontrolled.dedup();
    let mut manifests = Dict::new();
    for m in &a.manifests {
        manifests.put(m, Val::Str(sha256_path(m)?));
    }
    let by_sha = keyed_by_sha(&models, &admitted)?;
    let closure = closure(&cells);
    let receipt = dict(vec![
        ("schema", Val::str(SCHEMA)),
        ("prompts", Val::Str(pathlib_str(&a.prompts))),
        ("prompts_sha256", Val::Str(sha256_path(&a.prompts)?)),
        ("apr_commit", Val::str(a.apr_commit.as_str())),
        ("inventory_sha256", Val::Str(sha256_path(&a.inventory)?)),
        ("manifests", Val::Dict(manifests)),
        ("admitted", Val::Dict(admitted.clone())),
        ("admitted_by_sha", Val::Dict(by_sha)),
        // Admission per thinking mode, keyed like admitted_by_sha.
        ("admitted_by_sha_thinking", Val::Dict(by_thinking.clone())),
        ("rejected", Val::Dict(rejected)),
        (
            "uncontrolled",
            Val::List(uncontrolled.into_iter().map(Val::Str).collect()),
        ),
        // Which (model, thinking, verb) lanes have no certified control.
        // Those lanes DECLINE at the judge.
        (
            "uncontrolled_detail",
            Val::List(detail.iter().cloned().map(Val::Dict).collect()),
        ),
        // Per (model, prompt): did each engine's thinking-ON reply close its
        // think block? The judge joins this against apr's cell.
        ("think_closure", Val::Dict(closure)),
        (
            "cells",
            Val::List(cells.into_iter().map(Val::Dict).collect()),
        ),
    ]);
    let body = dump_indent(&receipt, 1)? + "\n";
    std::fs::write(&a.out, body).map_err(|e| os_err(&e, &a.out))?;
    print_summary(&models, &mut by_thinking, &detail, &admitted)?;
    Ok(0)
}

/// What `admit_all` builds: the admissions per model/quant, the rejections
/// with their first failure, admissions per quant sha and thinking mode, and
/// every cell.
struct Admissions {
    admitted: Dict,
    rejected: Dict,
    by_thinking: Dict,
    cells: Vec<Dict>,
}

fn admit_all(models: &[Val], prompts: &[Val], rows: &[Val]) -> PyResult<Admissions> {
    let mut acc = Admissions {
        admitted: Dict::new(),
        rejected: Dict::new(),
        by_thinking: Dict::new(),
        cells: Vec::new(),
    };
    for model in models {
        for (quant, qsha) in sorted_quants(model)? {
            let key = format!("{}/{}", py_str(&model.item("model")?), py_str(&quant));
            let kval = Val::str(key.as_str());
            acc.admitted.put(&key, Val::List(vec![]));
            acc.rejected.put(&key, Val::Dict(Dict::new()));
            let mut modes = Dict::new();
            for t in modes_of(model)? {
                modes.set(t, Val::List(vec![]))?;
            }
            acc.by_thinking.set(qsha.clone(), Val::Dict(modes))?;
            for p in prompts {
                let one = certify_one(p, model, &qsha, rows)?;
                acc.record(p, one, &kval, &qsha)?;
            }
        }
    }
    Ok(acc)
}

impl Admissions {
    /// File one prompt's `certify_one` result under its model/quant key.
    fn record(&mut self, p: &Val, one: One, kval: &Val, qsha: &Val) -> PyResult<()> {
        for (t, m_ok) in one.by_mode.iter() {
            if m_ok.truthy() {
                append_at(dict_at(&mut self.by_thinking, qsha)?, t, p.item("id")?)?;
            }
        }
        for mut c in one.cells {
            c.put("prompt_id", p.item("id")?);
            c.put("model", kval.clone());
            self.cells.push(c);
        }
        match one.first_bad {
            None => append_at(&mut self.admitted, kval, p.item("id")?),
            Some(why) => dict_at(&mut self.rejected, kval)?.set(p.item("id")?, Val::Str(why)),
        }
    }
}

/// Every verb any prompt serves, sorted.
fn all_verbs(prompts: &[Val]) -> PyResult<Vec<Val>> {
    let mut verb_set: Vec<Val> = Vec::new();
    for p in prompts {
        for v in p.item("verb")?.iter()? {
            if !verb_set.iter().any(|x| py_eq(x, &v)) {
                verb_set.push(v);
            }
        }
    }
    py_sorted(&verb_set)
}

/// Is `v` served by a certified control among `ids`?
fn served(prompts: &[Val], ids: &Val, v: &Val) -> PyResult<bool> {
    for p in prompts {
        if p.get("control")?.truthy()
            && ids.contains(&p.item("id")?)?
            && p.item("verb")?.contains(v)?
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// A lane needs ONE certified positive control serving its verb, per
/// thinking mode. A second control that fails does not un-control a lane
/// another control covers. One entry per (model, thinking) with bare verbs.
fn lane_gaps(models: &[Val], prompts: &[Val], by_thinking: &mut Dict) -> PyResult<Vec<Dict>> {
    let verbs = all_verbs(prompts)?;
    let mut detail: Vec<Dict> = Vec::new();
    for model in models {
        for (quant, qsha) in sorted_quants(model)? {
            let modes = dict_at(by_thinking, &qsha)?.clone();
            for (t, ids) in modes.iter() {
                let mut bare = Vec::new();
                for v in &verbs {
                    if !served(prompts, ids, v)? {
                        bare.push(v.clone());
                    }
                }
                if bare.is_empty() {
                    continue;
                }
                detail.push(Dict::from_pairs(vec![
                    (
                        "model",
                        Val::Str(format!(
                            "{}/{}",
                            py_str(&model.item("model")?),
                            py_str(&quant)
                        )),
                    ),
                    ("sha256", qsha.clone()),
                    ("thinking", t.clone()),
                    ("verbs", Val::List(bare)),
                ]));
            }
        }
    }
    Ok(detail)
}

/// The same admissions keyed by the quantized GGUF's sha256, so the judge
/// can check each cell's (model_sha256, prompt_id) directly.
fn keyed_by_sha(models: &[Val], admitted: &Dict) -> PyResult<Dict> {
    let mut by_sha = Dict::new();
    for m in models {
        for (k, v) in admitted.iter() {
            let ks = py_str(k);
            let Some((head, tail)) = ks.split_once('/') else {
                return Err(PyErr::index("list index out of range"));
            };
            if py_eq(&Val::str(head), &m.item("model")?) {
                by_sha.set(m.item("quants")?.item(tail)?, v.clone())?;
            }
        }
    }
    Ok(by_sha)
}

/// One stdout line per (model, quant): admissions per mode, all-modes count,
/// and the lanes with no certified control.
fn print_summary(
    models: &[Val],
    by_thinking: &mut Dict,
    detail: &[Dict],
    admitted: &Dict,
) -> PyResult<()> {
    for model in models {
        for (quant, qsha) in sorted_quants(model)? {
            let k = format!("{}/{}", py_str(&model.item("model")?), py_str(&quant));
            let mut modes = Vec::new();
            for (t, v) in dict_at(by_thinking, &qsha)?.iter() {
                modes.push(format!("{} {}", py_str(t), v.len()?));
            }
            let gaps = gaps_of(detail, &k)?;
            let n_all = admitted.get_str(&k).map_or(Ok(0), Val::len)?;
            let gaps = if gaps.is_empty() {
                String::new()
            } else {
                format!("  [no certified control: {}]", gaps.join("; "))
            };
            println!(
                "{k}: admitted {} (all modes {n_all}){gaps}",
                modes.join(", ")
            );
        }
    }
    Ok(())
}

/// `thinking:verb/verb` for each uncontrolled lane of model/quant `k`.
fn gaps_of(detail: &[Dict], k: &str) -> PyResult<Vec<String>> {
    let mut gaps = Vec::new();
    for u in detail {
        if u.get_str("model").and_then(Val::as_str) == Some(k) {
            let t = py_str(u.get_str("thinking").unwrap_or(&Val::None));
            let vs = py_join("/", u.get_str("verbs").unwrap_or(&Val::None))?;
            gaps.push(format!("{t}:{vs}"));
        }
    }
    Ok(gaps)
}

// ------------------------------------------------------------------- CLI

/// One subcommand's argparse shape.
struct Sub {
    prog: &'static str,
    usage: &'static str,
    /// Long options, all required; `--out` also answers to `-o`.
    flags: &'static [&'static str],
    /// The required `nargs="+"` positional, if the subcommand takes one.
    positional: Option<&'static str>,
}

const CERTIFY: Sub = Sub {
    prog: "aprender-crux-judge certify",
    usage: "usage: aprender-crux-judge certify [-h] --prompts PROMPTS --inventory INVENTORY \
            --apr-commit APR_COMMIT -o OUT manifests [manifests ...]",
    flags: &["--prompts", "--inventory", "--apr-commit", "--out"],
    positional: Some("manifests"),
};

const CHECK: Sub = Sub {
    prog: "aprender-crux-judge check",
    usage: "usage: aprender-crux-judge check [-h] --prompts PROMPTS --receipt RECEIPT",
    flags: &["--prompts", "--receipt"],
    positional: None,
};

/// argparse hands a subparser's leftovers back to the top-level parser, so
/// unrecognized arguments are reported with the top-level usage, and only
/// after the subcommand's required arguments are all present.
fn top_error(msg: &str) -> u8 {
    eprintln!(
        "usage: aprender-crux-judge [-h] {{certify,check}} ...\naprender-crux-judge: error: {msg}"
    );
    2
}

impl Sub {
    fn error(&self, msg: &str) -> u8 {
        eprintln!("{}\n{}: error: {msg}", self.usage, self.prog);
        2
    }
    fn label(&self, idx: usize) -> String {
        match self.flags[idx] {
            "--out" => "-o/--out".to_string(),
            f => f.to_string(),
        }
    }
    /// A long option by exact name or unique prefix: `Ok(None)` is `--help`,
    /// `Err(None)` an unknown option (a leftover), `Err(Some(rc))` an error.
    fn resolve(&self, opt: &str) -> Result<Option<usize>, Option<u8>> {
        if let Some(i) = self.flags.iter().position(|f| *f == opt) {
            return Ok(Some(i));
        }
        let mut hits: Vec<&str> = self
            .flags
            .iter()
            .copied()
            .filter(|f| f.starts_with(opt))
            .collect();
        if "--help".starts_with(opt) {
            hits.push("--help");
        }
        match hits.as_slice() {
            [] => Err(None),
            ["--help"] => Ok(None),
            [one] => Ok(self.flags.iter().position(|f| f == one)),
            many => Err(Some(self.error(&format!(
                "ambiguous option: {opt} could match {}",
                many.join(", ")
            )))),
        }
    }
    /// argparse, for the shapes these subcommands take: `--long[=v]`,
    /// `-o v` / `-ov`, `-h`, `--` ending options. Returns the option values
    /// (all required) and the positionals; Err is the exit code.
    fn parse(&self, args: &[String]) -> Result<(Vec<String>, Vec<String>), u8> {
        let mut vals: Vec<Option<String>> = vec![None; self.flags.len()];
        let (mut pos, mut extras) = (Vec::new(), Vec::new());
        let (mut i, mut only_pos) = (0, false);
        while i < args.len() {
            let a = &args[i];
            i += 1;
            if only_pos || !a.starts_with('-') || a == "-" {
                pos.push(a.clone());
                continue;
            }
            if a == "--" {
                only_pos = true;
                continue;
            }
            match self.option_of(a)? {
                None => extras.push(a.clone()),
                Some((idx, inline)) => vals[idx] = Some(self.value(args, &mut i, inline, idx)?),
            }
        }
        self.finish(vals, pos, extras)
    }
    /// One option argument: its flag index and any inline value, `None` for a
    /// leftover; Err(0) after printing help, else an error's exit code.
    fn option_of(&self, a: &str) -> Result<Option<(usize, Option<String>)>, u8> {
        if a == "-h" {
            println!("{}", self.usage);
            return Err(0);
        }
        if let Some(rest) = a.strip_prefix("--") {
            let (opt, inline) = match rest.split_once('=') {
                Some((o, v)) => (&a[..o.len() + 2], Some(v.to_string())),
                None => (a, None),
            };
            return match self.resolve(opt) {
                Ok(Some(idx)) => Ok(Some((idx, inline))),
                Ok(None) => {
                    println!("{}", self.usage);
                    Err(0)
                }
                Err(None) => Ok(None),
                Err(Some(rc)) => Err(rc),
            };
        }
        match (
            a.strip_prefix("-o"),
            self.flags.iter().position(|f| *f == "--out"),
        ) {
            (Some(rest), Some(idx)) => {
                Ok(Some((idx, (!rest.is_empty()).then(|| rest.to_string()))))
            }
            _ => Ok(None),
        }
    }
    /// An option's value: the inline one, else the next argument unless it
    /// looks like an option.
    fn value(
        &self,
        args: &[String],
        i: &mut usize,
        inline: Option<String>,
        idx: usize,
    ) -> Result<String, u8> {
        if let Some(v) = inline {
            return Ok(v);
        }
        match args.get(*i) {
            Some(v) if !(v.starts_with('-') && v.len() > 1) => {
                *i += 1;
                Ok(v.clone())
            }
            _ => Err(self.error(&format!(
                "argument {}: expected one argument",
                self.label(idx)
            ))),
        }
    }
    /// argparse's end-of-parse checks: required arguments first, leftovers
    /// after.
    fn finish(
        &self,
        vals: Vec<Option<String>>,
        mut pos: Vec<String>,
        mut extras: Vec<String>,
    ) -> Result<(Vec<String>, Vec<String>), u8> {
        let mut missing: Vec<String> = (0..self.flags.len())
            .filter(|&j| vals[j].is_none())
            .map(|j| self.label(j))
            .collect();
        match self.positional {
            Some(name) if pos.is_empty() => missing.push(name.to_string()),
            Some(_) => {}
            None => extras.append(&mut pos),
        }
        if !missing.is_empty() {
            return Err(self.error(&format!(
                "the following arguments are required: {}",
                missing.join(", ")
            )));
        }
        if !extras.is_empty() {
            return Err(top_error(&format!(
                "unrecognized arguments: {}",
                extras.join(" ")
            )));
        }
        Ok((vals.into_iter().flatten().collect(), pos))
    }
}

/// `certify ...` / `check ...`: exit 0 ok, 1 refused (check) or a crash,
/// 2 usage or an invalid prompt set (certify).
pub fn cli(argv: &[String]) -> u8 {
    let run = |argv: &[String]| -> Result<PyResult<u8>, u8> {
        match argv.first().map(String::as_str) {
            Some("certify") => {
                let (v, manifests) = CERTIFY.parse(&argv[1..])?;
                let [prompts, inventory, apr_commit, out] = <[String; 4]>::try_from(v)
                    .map_err(|_| CERTIFY.error("internal: option count"))?;
                Ok(certify(&CertifyArgs {
                    prompts,
                    inventory,
                    apr_commit,
                    out,
                    manifests,
                }))
            }
            Some("check") => {
                let (v, _) = CHECK.parse(&argv[1..])?;
                let [prompts, receipt] = <[String; 2]>::try_from(v)
                    .map_err(|_| CHECK.error("internal: option count"))?;
                Ok(check(&prompts, &receipt).map(|(rc, out)| {
                    print!("{out}");
                    u8::try_from(rc).unwrap_or(1)
                }))
            }
            _ => Err(2),
        }
    };
    match run(argv) {
        Err(rc) | Ok(Ok(rc)) => rc,
        Ok(Err(e)) => {
            eprintln!("crash: {}: {e}", e.kind);
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn pathlib_str_like_posix_path() {
        for (p, want) in [
            ("", "."),
            (".", "."),
            ("./a//b/./c/", "a/b/c"),
            ("/a/b", "/a/b"),
            ("//a", "//a"),
            ("///a", "/a"),
            ("//", "//"),
            ("/", "/"),
            ("a/..", "a/.."),
        ] {
            assert_eq!(pathlib_str(p), want, "{p:?}");
        }
    }

    #[test]
    fn parse_like_argparse() {
        let (v, pos) = CERTIFY
            .parse(&args("--pro p --inv i --apr-commit=c -oOUT m1 -- -m2"))
            .unwrap();
        assert_eq!(v, args("p i c OUT"));
        assert_eq!(pos, args("m1 -m2"));
        let (v, _) = CHECK.parse(&args("--receipt r --prompts p")).unwrap();
        assert_eq!(v, args("p r"));
        // Missing manifests is a required-argument error (2), like a flag.
        assert_eq!(CERTIFY.parse(&args("--prompts p")), Err(2));
        assert_eq!(CHECK.parse(&args("--prompts p --receipt r x")), Err(2));
        assert_eq!(CHECK.parse(&args("--prompts p --receipt r --nope")), Err(2));
        assert_eq!(CHECK.parse(&args("--prompts")), Err(2));
        assert_eq!(CHECK.parse(&args("-h")), Err(0));
        assert_eq!(CHECK.parse(&args("--he")), Err(0));
        assert_eq!(cli(&args("nope")), 2);
    }

    #[test]
    fn join_and_digest() {
        assert_eq!(
            py_join(", ", &Val::List(vec![Val::str("run"), Val::str("chat")])).unwrap(),
            "run, chat"
        );
        assert_eq!(py_join(", ", &Val::str("ab")).unwrap(), "a, b");
        assert_eq!(
            py_join(", ", &Val::List(vec![Val::int(1)]))
                .unwrap_err()
                .msg,
            "sequence item 0: expected str instance, int found"
        );
        assert_eq!(
            py_join(", ", &Val::int(1)).unwrap_err().msg,
            "can only join an iterable"
        );
        assert_eq!(
            certification_ok("x", None).unwrap().as_str(),
            Some("no --certification receipt was given")
        );
    }
}
