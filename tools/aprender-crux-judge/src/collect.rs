//! `collect` and `render_md` of `crux_inference_judge.py`: one sweep's
//! manifest judged into the receipt JSON and its markdown table.

use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::certify::certification_ok;
use crate::judge::{
    add_str, cell_quorum, classify_not_ran, engine_entry, engine_versions, judge_cell,
    judge_deterministic, model_format, prompt_opens_think, rendered_opens_think, report_greedy,
    token_parity, COMPARATORS, ENGINES,
};
use crate::pyerr::{PyErr, PyResult};
use crate::pyio::{for_each_nonblank_line, os_err, read_text, read_text_strict};
use crate::pyjson::{dump_indent2, dumps, loads};
use crate::pyre::formatted_prompt;
use crate::pyval::{
    char_len, concat_str, dict, fmt_d, no_attr, norm_ws, py_eq, py_join, py_ne, py_repr, py_sorted,
    py_str, slice_chars, Dict, Val,
};
use crate::serve_routes::oracle_route;

/// The `collect` subcommand's arguments.
pub struct Args {
    pub manifest: String,
    pub prompts: String,
    pub meta: String,
    pub out_json: String,
    pub out_md: String,
    pub certification: Option<String>,
}

/// The cell key's first six parts, by name.
const KEY_NAMES: [&str; 6] = [
    "model_sha256",
    "host",
    "verb",
    "thinking",
    "rung",
    "prompt_id",
];

const PLANTED: &str = "<answer>__crux_negative_control__</answer>";

fn tup(items: Vec<Val>) -> Val {
    Val::Tuple(items)
}

/// `d[k]`: KeyError with the key's repr.
fn dget<'a>(d: &'a Dict, k: &Val) -> PyResult<&'a Val> {
    d.get(k)?.ok_or_else(|| PyErr::key(py_repr(k)))
}

/// `dict(v)` of a value the judge built as a dict.
fn dict_copy(v: &Val) -> PyResult<Dict> {
    match v {
        Val::Dict(d) => Ok(d.clone()),
        other => Err(PyErr::type_err(format!(
            "'{}' object is not iterable",
            other.type_name()
        ))),
    }
}

/// `by_key[k]`: every key was inserted with a dict of rows.
fn rows_of<'a>(by_key: &'a Dict, k: &Val) -> &'a Dict {
    match by_key.get(k) {
        Ok(Some(Val::Dict(d))) => d,
        _ => panic!("by_key holds a dict for every key it lists"),
    }
}

fn rows_of_mut<'a>(by_key: &'a mut Dict, k: &Val) -> &'a mut Dict {
    match by_key.get_mut(k) {
        Ok(Some(Val::Dict(d))) => d,
        _ => panic!("by_key holds a dict for every key it lists"),
    }
}

/// `entries[eng]`: every engine's entry is a dict.
fn entry_mut<'a>(entries: &'a mut Dict, eng: &str) -> &'a mut Dict {
    match entries.get_mut(&Val::str(eng)) {
        Ok(Some(Val::Dict(d))) => d,
        _ => panic!("entries holds a dict for every engine"),
    }
}

/// `k[i]` of a key tuple.
fn part(k: &Val, i: usize) -> &Val {
    match k {
        Val::Tuple(t) => &t[i],
        _ => panic!("a cell key is a tuple"),
    }
}

fn parts(k: &Val, n: usize) -> Vec<Val> {
    match k {
        Val::Tuple(t) => t[..n].to_vec(),
        _ => panic!("a cell key is a tuple"),
    }
}

/// `norm(text)`: `" ".join(text.split())`.
fn norm(text: &Val) -> PyResult<String> {
    match text {
        Val::Str(s) => Ok(norm_ws(s)),
        other => Err(no_attr(other, "split")),
    }
}

/// `short(text, n)`.
fn short(text: &Val, n: usize) -> PyResult<String> {
    let t = norm(&text.clone().or(Val::str("")))?;
    Ok(if char_len(&t) > n {
        format!("{}…", slice_chars(&t, 0, n - 1))
    } else {
        t
    })
}

/// Days since 1970-01-01 -> (year, month, day), proleptic Gregorian.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

/// `datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")`.
fn judged_at(secs: i64) -> String {
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    let rem = secs.rem_euclid(86_400);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60
    )
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
}

fn write_file(path: &str, text: &str) -> PyResult<()> {
    let mut fh = std::fs::File::create(path).map_err(|e| os_err(&e, path))?;
    fh.write_all(text.as_bytes()).map_err(|e| os_err(&e, path))
}

/// What the summary reads back of a cell.
struct Cell {
    k: Val,
    verdict: &'static str,
    all_wrong: bool,
    control: bool,
    fmt: Val,
    engines: Dict,
}

/// The run's inputs: the prompt set, its prompts by id, the meta and the
/// manifest rows.
struct Inputs {
    pdoc: Val,
    prompts: Dict,
    meta: Val,
    rows: Vec<Val>,
}

fn load_inputs(args: &Args) -> PyResult<Inputs> {
    let pdoc = loads(&read_text_strict(&args.prompts)?)?;
    let mut prompts = Dict::new();
    for p in pdoc.item("prompts")?.iter()? {
        prompts.set(p.item("id")?, p)?;
    }
    let meta = loads(&read_text_strict(&args.meta)?)?;
    let mut rows: Vec<Val> = Vec::new();
    for_each_nonblank_line(&args.manifest, &mut |line: &str| {
        rows.push(loads(line)?);
        Ok(())
    })?;
    Ok(Inputs {
        pdoc,
        prompts,
        meta,
        rows,
    })
}

fn gen_rows(rows: &[Val]) -> PyResult<Vec<Val>> {
    let mut gens: Vec<Val> = Vec::new();
    for r in rows {
        if py_eq(&r.get("kind")?, &Val::str("gen")) {
            gens.push(r.clone());
        }
    }
    Ok(gens)
}

/// An apr `run` row with thinking on that was not refused.
fn apr_thinking_run(r: &Val) -> PyResult<bool> {
    Ok(py_eq(&r.get("engine")?, &Val::str("apr"))
        && py_eq(&r.get("verb")?, &Val::str("run"))
        && py_eq(&r.get("thinking")?, &Val::str("on"))
        && !r.get("refused")?.truthy())
}

/// #3962 B2: the reference tmpl rows first, else apr's own `run` rendering.
fn opened_by_prompt(rows: &[Val], gens: &[Val]) -> PyResult<Dict> {
    let mut opened_by = prompt_opens_think(rows)?;
    for r in gens {
        if !apr_thinking_run(r)? {
            continue;
        }
        let stderr = read_text(&r.get("stderr")?)?;
        let rendered = formatted_prompt()
            .captures(&stderr)
            .and_then(|m| m.get(1))
            .map_or(Val::None, |g| Val::str(g.as_str()));
        let o = rendered_opens_think(&rendered)?;
        if o.is_none() {
            continue;
        }
        let key = tup(vec![r.get("model_sha256")?, r.get("prompt_id")?]);
        if opened_by.get(&key)?.is_none() {
            opened_by.set(key, o)?;
        }
    }
    Ok(opened_by)
}

/// llama.cpp's `tok` rows with no `input`, by (model, prompt).
fn tok_rows(rows: &[Val]) -> PyResult<Dict> {
    let mut toks = Dict::new();
    for r in rows {
        if py_eq(&r.get("kind")?, &Val::str("tok"))
            && py_eq(&r.get("engine")?, &Val::str("llama.cpp"))
            && !r.contains_str("input")?
        {
            toks.set(
                tup(vec![r.item("model_sha256")?, r.item("prompt_id")?]),
                r.clone(),
            )?;
        }
    }
    Ok(toks)
}

/// The models under development, and each model's format, by sha.
fn model_maps(meta: &Val) -> PyResult<(Dict, Dict)> {
    let models = meta.get("models")?.or(Val::List(vec![]));
    let mut subject_dev = Dict::new();
    for m in models.iter()? {
        if m.get("under_development")?.truthy() {
            subject_dev.set(m.get("sha256")?, m.get("under_development")?)?;
        }
    }
    let mut fmt_of_model = Dict::new();
    for m in models.iter()? {
        let sha = m.get("sha256")?;
        let fmt = match m.get("format")? {
            f if f.truthy() => f,
            _ => model_format(&m.get("name")?)?,
        };
        fmt_of_model.set(sha, fmt)?;
    }
    Ok((subject_dev, fmt_of_model))
}

/// A gen row's cell key: (model, host, verb, thinking, rung, prompt, mode,
/// route).
fn cell_key(r: &Val, prompts: &Dict) -> PyResult<Val> {
    let mode = match r.get("mode")? {
        m if m.truthy() => m,
        _ => Val::str(if py_eq(&r.item("verb")?, &Val::str("serve run")) {
            "nonstream"
        } else {
            ""
        }),
    };
    let pr = dget(prompts, &r.item("prompt_id")?)?;
    let rung = pr.get("rung")?.or(pr.get("tier")?).or(Val::str("v2"));
    Ok(tup(vec![
        r.item("model_sha256")?,
        r.item("host")?,
        r.item("verb")?,
        r.item("thinking")?,
        rung,
        r.item("prompt_id")?,
        mode,
        r.get("route")?.or(Val::str("")),
    ]))
}

/// The gen rows by cell key, keys in first-seen order: {key: {engine: row}}.
fn group_gens(gens: &[Val], prompts: &Dict) -> PyResult<(Vec<Val>, Dict)> {
    let mut keys: Vec<Val> = Vec::new();
    let mut by_key = Dict::new();
    for r in gens {
        let k = cell_key(r, prompts)?;
        if by_key.get(&k)?.is_none() {
            keys.push(k.clone());
            by_key.set(k.clone(), Val::Dict(Dict::new()))?;
        }
        rows_of_mut(&mut by_key, &k).set(r.item("engine")?, r.clone())?;
    }
    Ok((keys, by_key))
}

/// Where route cell `k` looks for a comparator row: the oracle route, the
/// route-less row of its mode, the row with neither mode nor route.
fn route_sources(k: &Val, orc: &str) -> [Val; 3] {
    let mut s1 = parts(k, 7);
    s1.push(Val::str(orc));
    let mut s2 = parts(k, 7);
    s2.push(Val::str(""));
    let mut s3 = parts(k, 6);
    s3.extend([Val::str(""), Val::str("")]);
    [tup(s1), tup(s2), tup(s3)]
}

/// The first source with a row for `eng` lends a copy of it to cell `k`.
fn borrow_row(
    by_key: &mut Dict,
    lent: &mut Dict,
    k: &Val,
    eng: &str,
    sources: &[Val; 3],
) -> PyResult<()> {
    for src in sources {
        if !py_ne(src, k) {
            continue;
        }
        let found = match by_key.get(src)? {
            Some(Val::Dict(d)) => d.get_str(eng).cloned(),
            _ => None,
        };
        let Some(row) = found else {
            continue;
        };
        let mut row = dict_copy(&row)?;
        row.put(
            "borrowed_from_route",
            part(src, 7).clone().or(Val::str("(route-less plugin row)")),
        );
        rows_of_mut(by_key, k).put(eng, Val::Dict(row));
        lent.set(tup(vec![src.clone(), Val::str(eng)]), Val::None)?;
        return Ok(());
    }
    Ok(())
}

/// #3962 B4: an apr route cell with no comparator row on its own route
/// borrows one per engine, from the oracle route, else the route-less row.
/// Returns each route cell's oracle route and the (source, engine) pairs lent.
fn borrow_routes(keys: &[Val], by_key: &mut Dict) -> PyResult<(Dict, Dict)> {
    let mut oracle_of = Dict::new();
    let mut lent = Dict::new();
    for k in keys {
        if !part(k, 7).truthy() || !rows_of(by_key, k).has("apr") {
            continue;
        }
        let orc = oracle_route(part(k, 7));
        oracle_of.set(k.clone(), orc.map_or(Val::None, Val::str))?;
        let Some(orc) = orc else {
            continue;
        };
        let sources = route_sources(k, orc);
        for eng in COMPARATORS {
            if !rows_of(by_key, k).has(eng) {
                borrow_row(by_key, &mut lent, k, eng, &sources)?;
            }
        }
    }
    Ok((oracle_of, lent))
}

/// Every row of cell `k` was lent to a route cell.
fn all_lent(k: &Val, rows_k: &Dict, lent: &Dict) -> PyResult<bool> {
    for e in rows_k.keys() {
        if lent.get(&tup(vec![k.clone(), e.clone()]))?.is_none() {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The cells to judge: a cell with no apr row whose every row was lent is
/// dropped.
fn kept_keys(keys: Vec<Val>, by_key: &Dict, lent: &Dict) -> PyResult<Vec<Val>> {
    let mut kept = Vec::with_capacity(keys.len());
    for k in keys {
        let rows_k = rows_of(by_key, &k);
        let lent_all = all_lent(&k, rows_k, lent)?;
        if rows_k.has("apr") || rows_k.is_empty() || !lent_all {
            kept.push(k);
        }
    }
    Ok(kept)
}

/// #3962 J2: admission per model (quant sha), per thinking mode when the
/// receipt carries it.
#[derive(Default)]
struct Admission {
    admitted: Option<Dict>,
    admitted_mode: Option<Dict>,
}

fn admission(certification: Option<&str>, v2: bool) -> PyResult<Admission> {
    let Some(cert) = certification.filter(|c| v2 && !c.is_empty()) else {
        return Ok(Admission::default());
    };
    let (a, am) = match read_text_strict(cert).and_then(|t| loads(&t)) {
        Ok(cdoc) => (
            cdoc.get("admitted_by_sha")?,
            cdoc.get("admitted_by_sha_thinking")?,
        ),
        Err(e) if e.is_os() || e.is_value() => (Val::None, Val::None),
        Err(e) => return Err(e),
    };
    Ok(Admission {
        admitted: Some(match a {
            Val::Dict(d) => d,
            _ => Dict::new(),
        }),
        admitted_mode: match am {
            Val::Dict(d) => Some(d),
            _ => None,
        },
    })
}

/// Why the certification does not admit cell `k`'s prompt, when it does not.
fn admission_reason(adm: &Admission, k: &Val) -> PyResult<Option<String>> {
    if let Some(am) = &adm.admitted_mode {
        let by_mode = match am.get(part(k, 0))? {
            Some(Val::Dict(d)) => d.clone(),
            _ => Dict::new(),
        };
        let allowed = by_mode
            .get(part(k, 3))?
            .cloned()
            .unwrap_or(Val::None)
            .or(Val::Tuple(vec![]));
        if allowed.contains(part(k, 5))? {
            return Ok(None);
        }
        return Ok(Some(format!(
            "prompt {} is not admitted for this model with thinking {} by the certification \
             (admitted_by_sha_thinking) -- never shown answerable here (#3962 J2)",
            py_str(part(k, 5)),
            py_str(part(k, 3))
        )));
    }
    let Some(ad) = &adm.admitted else {
        return Ok(None);
    };
    let allowed = ad.get(part(k, 0))?.cloned().unwrap_or(Val::Tuple(vec![]));
    if allowed.contains(part(k, 5))? {
        return Ok(None);
    }
    Ok(Some(format!(
        "prompt {} is not admitted for this model by the certification (admitted_by_sha) \
         -- never shown answerable here (#3962 J2)",
        py_str(part(k, 5))
    )))
}

/// What every cell is judged against.
struct Ctx<'a> {
    prompts: &'a Dict,
    by_key: Dict,
    opened_by: Dict,
    toks: Dict,
    requested: Val,
    versions: Dict,
    subject_dev: Dict,
    fmt_of_model: Dict,
    oracle_of: Dict,
    admission: Admission,
}

/// The entry of an engine with no row in the cell.
fn missing_entry(requested: &Val, eng: &str) -> PyResult<Val> {
    let why = if requested.contains_str(eng)? {
        "missing: no row for this engine"
    } else {
        "not requested"
    };
    Ok(dict(vec![
        ("answered", Val::Bool(false)),
        ("missing", Val::Bool(true)),
        ("why", Val::str(why)),
    ]))
}

/// An engine's entry from its row, and whether it is a comparator that
/// answered with no recorded version.
fn row_entry(ctx: &Ctx, row: &Val, prompt: &Val, eng: &str) -> PyResult<(Dict, bool)> {
    let opened_key = tup(vec![row.get("model_sha256")?, row.get("prompt_id")?]);
    let opened = ctx
        .opened_by
        .get(&opened_key)?
        .cloned()
        .unwrap_or(Val::None);
    let mut entry = engine_entry(row, prompt, &opened)?;
    if row.get("borrowed_from_route")?.truthy() {
        entry.put("borrowed_from_route", row.item("borrowed_from_route")?);
    }
    // #3952: a comparator with no recorded version cannot vouch.
    let unpinned = COMPARATORS.contains(&eng)
        && entry.get_str("answered").is_some_and(Val::truthy)
        && !ctx.versions.get_str(eng).is_some_and(Val::truthy);
    if unpinned {
        entry.put("answered", Val::Bool(false));
        entry.put(
            "why",
            Val::Str(format!(
                "unpinned: the run's meta records no version for {eng}, so a \
                 verdict it vouched for could not name what produced it"
            )),
        );
    }
    Ok((entry, unpinned))
}

/// Each engine's entry for cell `k`, and the comparators that answered with
/// no recorded version.
fn engine_entries(ctx: &Ctx, k: &Val, prompt: &Val) -> PyResult<(Dict, Vec<Val>)> {
    let mut entries = Dict::new();
    let mut unpinned: Vec<Val> = Vec::new();
    for eng in ENGINES {
        let Some(row) = rows_of(&ctx.by_key, k).get_str(eng) else {
            entries.put(eng, missing_entry(&ctx.requested, eng)?);
            continue;
        };
        let (entry, no_version) = row_entry(ctx, row, prompt, eng)?;
        if no_version {
            unpinned.push(Val::str(eng));
        }
        entries.put(eng, Val::Dict(entry));
    }
    Ok((entries, unpinned))
}

/// The cell's format: the first row that records one, else its model's.
fn cell_format(ctx: &Ctx, k: &Val) -> PyResult<Val> {
    let mut fmt = Val::None;
    for row in rows_of(&ctx.by_key, k).values() {
        let f = row.get("format")?;
        if f.truthy() {
            fmt = f;
            break;
        }
    }
    Ok(fmt.or(ctx
        .fmt_of_model
        .get(part(k, 0))?
        .cloned()
        .unwrap_or(Val::None)))
}

/// The reasons past the engines' answers that turn cell `k` RED.
fn cell_reasons(ctx: &Ctx, k: &Val, unpinned: Vec<Val>) -> PyResult<Vec<String>> {
    let mut reasons: Vec<String> = Vec::new();
    if matches!(ctx.oracle_of.get(k)?, Some(Val::None)) {
        reasons.push(format!(
            "no oracle route mapped for {}: no comparator route asks its question in the same \
             representation, so nothing can vouch for it -- map its kind in \
             crux_serve_routes.ORACLE_ROUTE_BY_KIND (#3962 B4)",
            py_str(part(k, 7))
        ));
    }
    reasons.extend(admission_reason(&ctx.admission, k)?);
    if !unpinned.is_empty() {
        reasons.push(format!(
            "oracle unpinned: {} answered with no recorded version, so this cell is not proven \
             (this is not a finding that apr was wrong)",
            py_join(", ", &Val::List(unpinned))?
        ));
    }
    Ok(reasons)
}

/// Each entry gets its correctness, its engine's version and, when it did
/// not answer, why not.
fn mark_entries(entries: &mut Dict, ok: &Dict, versions: &Dict) -> PyResult<()> {
    for eng in ENGINES {
        let correct = dget(ok, &Val::str(eng))?.clone();
        let version = versions.get_str(eng).cloned().unwrap_or(Val::None);
        let entry = entry_mut(entries, eng);
        entry.put("correct", correct);
        entry.put("version", version);
        if !entry.get_str("answered").is_some_and(Val::truthy) {
            let why = entry.get_str("why").cloned().unwrap_or(Val::None);
            entry.put("not_ran_reason", Val::str(classify_not_ran(&why)));
        }
    }
    Ok(())
}

/// What each engine that answered said, normalized.
fn said_by(entries: &Dict) -> PyResult<Dict> {
    let mut said = Dict::new();
    for (e, v) in entries.iter() {
        if v.get("answered")?.truthy() {
            said.set(e.clone(), Val::Str(norm(&v.item("answer")?)?))?;
        }
    }
    Ok(said)
}

/// The cell's key, verdict, oracle route, reasons, format and extracted
/// answers.
fn cell_head(
    ctx: &Ctx,
    k: &Val,
    verdict: &'static str,
    reasons: Vec<String>,
    fmt: &Val,
    ext: Dict,
) -> PyResult<Dict> {
    let mut key = Dict::new();
    for (i, name) in KEY_NAMES.iter().enumerate() {
        key.put(name, part(k, i).clone());
    }
    if part(k, 6).truthy() {
        key.put("mode", part(k, 6).clone());
    }
    if part(k, 7).truthy() {
        key.put("route", part(k, 7).clone());
    }
    let mut cell = Dict::new();
    cell.put("key", Val::Dict(key));
    cell.put("verdict", Val::str(verdict));
    if let Some(orc) = ctx.oracle_of.get(k)? {
        cell.put("oracle_route", orc.clone());
    }
    cell.put(
        "reasons",
        Val::List(reasons.into_iter().map(Val::Str).collect()),
    );
    cell.put("format", fmt.clone());
    cell.put("extracted", Val::Dict(ext));
    Ok(cell)
}

fn any_answered(entries: &Dict) -> PyResult<bool> {
    for v in entries.values() {
        if v.get("answered")?.truthy() {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The model under test: its sha, apr's version, and any development note.
fn subject(ctx: &Ctx, k: &Val) -> PyResult<Val> {
    let dev = ctx
        .subject_dev
        .get(part(k, 0))?
        .cloned()
        .unwrap_or(Val::None);
    Ok(dict(vec![
        ("model_sha256", part(k, 0).clone()),
        (
            "apr_version",
            ctx.versions.get_str("apr").cloned().unwrap_or(Val::None),
        ),
        ("under_development", Val::Bool(dev.truthy())),
        ("development_note", dev.or(Val::None)),
    ]))
}

/// Who answered, whether they all said the same, and who agrees with apr.
fn agreement(said: &Dict) -> PyResult<Val> {
    let answered = py_sorted(&said.keys().cloned().collect::<Vec<_>>())?;
    let mut distinct: Vec<&Val> = Vec::new();
    for v in said.values() {
        if !distinct.iter().any(|d| py_eq(d, v)) {
            distinct.push(v);
        }
    }
    let apr_said = said.get_str("apr").cloned();
    let mut matches: Vec<Val> = Vec::new();
    for (e, v) in said.iter() {
        if py_ne(e, &Val::str("apr")) && apr_said.as_ref().is_some_and(|a| py_eq(v, a)) {
            matches.push(e.clone());
        }
    }
    Ok(dict(vec![
        ("answered", Val::List(answered)),
        (
            "all_identical",
            Val::Bool(said.len() >= 2 && distinct.len() == 1),
        ),
        ("apr_matches", Val::List(py_sorted(&matches)?)),
    ]))
}

/// apr's token count against llama.cpp's, for the `run` verb only.
fn cell_token_parity(toks: &Dict, k: &Val, entries: &Dict) -> PyResult<Val> {
    if !py_eq(part(k, 2), &Val::str("run")) {
        return Ok(dict(vec![
            ("measured", Val::Bool(false)),
            (
                "why",
                Val::Str(format!("not measured for the {} verb", py_str(part(k, 2)))),
            ),
        ]));
    }
    let apr_entry = dget(entries, &Val::str("apr"))?;
    let tok = toks.get(&tup(vec![part(k, 0).clone(), part(k, 5).clone()]))?;
    token_parity(apr_entry, tok)
}

/// Cell `k` judged: what the summary reads back of it, and its receipt
/// entry.
fn judge_one(ctx: &Ctx, k: &Val) -> PyResult<(Cell, Val)> {
    let prompt = dget(ctx.prompts, part(k, 5))?.clone();
    let (mut entries, unpinned) = engine_entries(ctx, k, &prompt)?;
    let fmt = cell_format(ctx, k)?;
    let cv = judge_cell(&entries, &prompt, &fmt)?;
    let mut verdict = cv.verdict;
    let ok = cv.ok;
    let mut reasons = cv.why;
    let extra = cell_reasons(ctx, k, unpinned)?;
    if !extra.is_empty() {
        verdict = "RED";
    }
    reasons.extend(extra);
    mark_entries(&mut entries, &ok, &ctx.versions)?;
    let said = said_by(&entries)?;
    let mut cell = cell_head(ctx, k, verdict, reasons, &fmt, cv.ext)?;
    let any_ok = ENGINES
        .iter()
        .any(|e| ok.get_str(e).is_some_and(Val::truthy));
    let answered = any_answered(&entries)?;
    let all_wrong = verdict == "RED" && !any_ok && answered;
    cell.put("all_wrong", Val::Bool(all_wrong));
    cell.put("quorum", cell_quorum(&entries)?);
    cell.put("subject", subject(ctx, k)?);
    let control = prompt.get("control")?.truthy();
    cell.put("positive_control", Val::Bool(control));
    cell.put("oracle", prompt.get("oracle")?);
    if prompt.contains_str("expect_any")? {
        cell.put("expect_any", prompt.item("expect_any")?);
    }
    cell.put("engines", Val::Dict(entries.clone()));
    cell.put("agreement", agreement(&said)?);
    cell.put("token_parity", cell_token_parity(&ctx.toks, k, &entries)?);
    let summary = Cell {
        k: k.clone(),
        verdict,
        all_wrong,
        control,
        fmt,
        engines: entries,
    };
    Ok((summary, Val::Dict(cell)))
}

/// How many ALL_WRONG cells each model has.
fn all_wrong_by_model(cells: &[Cell]) -> PyResult<Dict> {
    let mut by_model = Dict::new();
    for c in cells.iter().filter(|c| c.all_wrong) {
        let sha = part(&c.k, 0);
        let n = match by_model.get(sha)? {
            Some(Val::Int(i)) => i.to_i64().unwrap_or(0),
            _ => 0,
        };
        by_model.set(sha.clone(), Val::int(n + 1))?;
    }
    Ok(by_model)
}

/// The ids of the positive-control prompts.
fn control_ids(prompts: &Dict) -> PyResult<Vec<Val>> {
    let mut controls: Vec<Val> = Vec::new();
    for (pid, p) in prompts.iter() {
        if p.get("control")?.truthy() {
            controls.push(pid.clone());
        }
    }
    Ok(controls)
}

/// #3957 F6: one positive control per (model, host, verb, thinking); the
/// lanes with none.
fn uncontrolled_lanes(cells: &[Cell]) -> PyResult<Vec<String>> {
    let mut lane_set = Dict::new();
    for c in cells {
        lane_set.set(tup(parts(&c.k, 4)), Val::None)?;
    }
    let lanes = py_sorted(&lane_set.keys().cloned().collect::<Vec<_>>())?;
    let mut uncontrolled: Vec<String> = Vec::new();
    for lane in &lanes {
        let covered = cells
            .iter()
            .any(|c| py_eq(&tup(parts(&c.k, 4)), lane) && c.control);
        if !covered {
            uncontrolled.push(format!(
                "{}/{}/{}/{}",
                py_str(&part(lane, 0).slice_to(12)?),
                py_str(part(lane, 1)),
                py_str(part(lane, 2)),
                py_str(part(lane, 3))
            ));
        }
    }
    Ok(uncontrolled)
}

/// The control's apr turns with the last one replaced by the planted answer.
fn planted_turns(old_turns: &Val, planted: &Val) -> PyResult<Val> {
    if !old_turns.truthy() {
        return Ok(Val::None);
    }
    match old_turns {
        Val::List(l) => {
            let mut t = l[..l.len() - 1].to_vec();
            t.push(planted.clone());
            Ok(Val::List(t))
        }
        Val::Str(_) | Val::Tuple(_) => Err(PyErr::type_err(format!(
            "can only concatenate {0} (not \"list\") to {0}",
            old_turns.type_name()
        ))),
        other => Err(PyErr::type_err(format!(
            "'{}' object is not subscriptable",
            other.type_name()
        ))),
    }
}

/// A GREEN positive control judged again with a planted wrong apr answer:
/// its prompt id, the planted answer and the verdict.
fn negative_control(ctl: &Cell, prompts: &Dict) -> PyResult<(Val, Val, &'static str)> {
    let pid = part(&ctl.k, 5);
    let ctl_prompt = dget(prompts, pid)?;
    let planted = ctl_prompt.get("negative")?.or(Val::str(PLANTED));
    let mut ents = Dict::new();
    for (e, v) in ctl.engines.iter() {
        ents.set(e.clone(), Val::Dict(dict_copy(v)?))?;
    }
    let mut apr = dict_copy(dget(&ents, &Val::str("apr"))?)?;
    let old_turns = apr.get_str("turns").cloned().unwrap_or(Val::None);
    let turns = planted_turns(&old_turns, &planted)?;
    apr.put("answered", Val::Bool(true));
    apr.put("answer", planted.clone());
    apr.put("why", Val::None);
    apr.put("turns", turns);
    ents.put("apr", Val::Dict(apr));
    let nv = judge_cell(&ents, ctl_prompt, &ctl.fmt)?.verdict;
    Ok((pid.clone(), planted, nv))
}

/// #3957 F6 / J3: a negative control per verb, and the verbs that judged a
/// planted wrong answer other than RED.
fn negative_controls(cells: &[Cell], prompts: &Dict) -> PyResult<(Dict, Vec<Val>)> {
    let mut verb_set = Dict::new();
    for c in cells {
        verb_set.set(part(&c.k, 2).clone(), Val::None)?;
    }
    let mut negative = Dict::new();
    let mut blind: Vec<Val> = Vec::new();
    for verb in py_sorted(&verb_set.keys().cloned().collect::<Vec<_>>())? {
        let Some(ctl) = cells
            .iter()
            .find(|c| py_eq(part(&c.k, 2), &verb) && c.control && c.verdict == "GREEN")
        else {
            continue;
        };
        let (pid, planted, nv) = negative_control(ctl, prompts)?;
        if nv != "RED" {
            blind.push(verb.clone());
        }
        negative.set(
            verb,
            dict(vec![
                ("prompt_id", pid),
                ("planted", planted),
                ("verdict", Val::str(nv)),
            ]),
        )?;
    }
    Ok((negative, py_sorted(&blind)?))
}

/// Why the run cannot PASS, when it cannot.
fn declined_because(
    controls: &[Val],
    cells: &[Cell],
    uncontrolled: &[String],
    blind: Vec<Val>,
    certified: &Val,
) -> PyResult<Option<String>> {
    Ok(if controls.is_empty() {
        Some("the prompt set declares no positive control (\"control\": true)".into())
    } else if cells.is_empty() {
        Some("no cell was measured".into())
    } else if !uncontrolled.is_empty() {
        Some(format!(
            "no positive-control cell for (model/host/verb/thinking) {}",
            uncontrolled.join(", ")
        ))
    } else if !blind.is_empty() {
        Some(format!(
            "negative control: a planted wrong apr answer was NOT judged RED for verb(s) {} -- \
             the lane cannot see a wrong answer",
            py_join(", ", &Val::List(blind))?
        ))
    } else if !certified.is_none() && !matches!(certified, Val::Bool(true)) {
        Some(format!(
            "the v2 prompt set is not certified: {} (#3962 J2)",
            py_str(certified)
        ))
    } else {
        None
    })
}

/// How many deterministic rows have this verdict.
fn det_count(det: &[Val], verdict: &str) -> PyResult<i64> {
    let mut n = 0i64;
    for d in det {
        if py_eq(&d.item("verdict")?, &Val::str(verdict)) {
            n += 1;
        }
    }
    Ok(n)
}

/// The receipt's summary block, and the exit code.
fn summarize(
    args: &Args,
    prompts: &Dict,
    cells: &[Cell],
    det: &[Val],
    v2: bool,
) -> PyResult<(Val, i32)> {
    let red = cells.iter().filter(|c| c.verdict == "RED").count();
    let green = cells.iter().filter(|c| c.verdict == "GREEN").count();
    let all_wrong_n = cells.iter().filter(|c| c.all_wrong).count();
    let by_model = all_wrong_by_model(cells)?;
    let controls = control_ids(prompts)?;
    let uncontrolled = uncontrolled_lanes(cells)?;
    let (negative, blind) = negative_controls(cells, prompts)?;
    // #3962 J2: a v2 prompt set is used only under the receipt that covers it.
    let certified = if v2 {
        certification_ok(&args.prompts, args.certification.as_deref())?
    } else {
        Val::None
    };
    let declined = declined_because(&controls, cells, &uncontrolled, blind, &certified)?;
    let det_red = det_count(det, "RED")?;
    let det_green = det_count(det, "GREEN")?;
    let (verdict, rc) = if red > 0 || det_red > 0 {
        ("RED", 1)
    } else if declined.is_some() {
        ("DECLINE", 2)
    } else {
        ("PASS", 0)
    };
    let count = |n: usize| Val::int(i64::try_from(n).unwrap_or(i64::MAX));
    let summary = dict(vec![
        ("RED", count(red)),
        ("GREEN", count(green)),
        ("ALL_WRONG", count(all_wrong_n)),
        ("cells", count(cells.len())),
        ("judged", count(red + green)),
        ("verdict", Val::str(verdict)),
        (
            "deterministic",
            dict(vec![
                ("RED", Val::int(det_red)),
                ("GREEN", Val::int(det_green)),
            ]),
        ),
        ("all_wrong_by_model", Val::Dict(by_model)),
        ("controls", Val::List(controls)),
        ("negative_controls", Val::Dict(negative)),
        ("certified", certified),
        (
            "declined_because",
            match declined {
                Some(d) if verdict == "DECLINE" => Val::Str(d),
                _ => Val::None,
            },
        ),
    ]);
    Ok((summary, rc))
}

/// The receipt to `--out-json`, its table to `--out-md` and stdout.
fn write_outputs(args: &Args, receipt: &Val) -> PyResult<()> {
    let mut fh = std::fs::File::create(&args.out_json).map_err(|e| os_err(&e, &args.out_json))?;
    let json = dump_indent2(receipt)?;
    fh.write_all(json.as_bytes())
        .and_then(|()| fh.write_all(b"\n"))
        .map_err(|e| os_err(&e, &args.out_json))?;
    drop(fh);
    let md = render_md(receipt)?;
    write_file(&args.out_md, &md)?;
    let mut out = std::io::stdout().lock();
    out.write_all(md.as_bytes())
        .and_then(|()| out.flush())
        .map_err(|e| os_err(&e, "<stdout>"))
}

/// `collect(args)` -> the exit code; the receipt and the table are written.
pub fn collect(args: &Args) -> PyResult<i32> {
    let Inputs {
        pdoc,
        prompts,
        meta,
        rows,
    } = load_inputs(args)?;
    let gens = gen_rows(&rows)?;
    let opened_by = opened_by_prompt(&rows, &gens)?;
    let toks = tok_rows(&rows)?;
    let mut det = judge_deterministic(&rows, "tok")?;
    det.extend(judge_deterministic(&rows, "tmpl")?);
    let greedy = report_greedy(&rows)?;
    let requested = meta.get_or(
        "engines",
        Val::List(ENGINES.iter().map(|e| Val::str(*e)).collect()),
    )?;
    let versions = engine_versions(&meta)?;
    let (subject_dev, fmt_of_model) = model_maps(&meta)?;
    let (keys, mut by_key) = group_gens(&gens, &prompts)?;
    let (oracle_of, lent) = borrow_routes(&keys, &mut by_key)?;
    let keys = kept_keys(keys, &by_key, &lent)?;
    let v2 = py_eq(&pdoc.get("schema")?, &Val::str("crux-inference-prompts/v2"));
    let admission = admission(args.certification.as_deref(), v2)?;
    let ctx = Ctx {
        prompts: &prompts,
        by_key,
        opened_by,
        toks,
        requested,
        versions,
        subject_dev,
        fmt_of_model,
        oracle_of,
        admission,
    };
    let mut cells: Vec<Cell> = Vec::new();
    let mut cell_vals: Vec<Val> = Vec::new();
    for k in &keys {
        let (cell, val) = judge_one(&ctx, k)?;
        cells.push(cell);
        cell_vals.push(val);
    }
    let (summary, rc) = summarize(args, &prompts, &cells, &det, v2)?;
    let mut receipt = dict_copy(&meta)?;
    receipt.update(&Dict::from_pairs(vec![
        ("schema", Val::str("crux-inference-receipt/v1")),
        ("judged_at", Val::Str(judged_at(now_secs()))),
        ("cells", Val::List(cell_vals)),
        ("deterministic", Val::List(det)),
        ("greedy", Val::List(greedy)),
        ("summary", summary),
    ]));
    write_outputs(args, &Val::Dict(receipt))?;
    Ok(rc)
}

/// `names.get(sha, sha[:12])`: the model's display name.
fn model_name(names: &Dict, sha: &Val) -> PyResult<String> {
    let default = sha.slice_to(12)?;
    Ok(py_str(names.get(sha)?.unwrap_or(&default)))
}

/// The title, the engine versions, the verdict line and the table head.
fn md_header(r: &Val, s: &Val) -> PyResult<Vec<String>> {
    let sub = |key: &str, field: &str| -> PyResult<String> {
        Ok(py_str(&r.get_or(key, Val::Dict(Dict::new()))?.get(field)?))
    };
    Ok(vec![
        format!(
            "# CRUX inference dogfood: {} on {} ({} lane)",
            py_str(&r.get("version")?),
            py_str(&r.get("host")?),
            py_str(&r.get("backend")?)
        ),
        String::new(),
        format!(
            "apr `{}` · llama.cpp `{}` · ollama `{}` · hf `{}` · llamafile `{}` · vllm `{}` · judged {}",
            sub("apr", "version_line")?,
            sub("llama_cpp", "build")?,
            sub("ollama", "server_version")?,
            sub("hf", "probe")?,
            sub("llamafile", "probe")?,
            sub("vllm", "probe")?,
            py_str(&r.item("judged_at")?)
        ),
        String::new(),
        format!(
            "**{}**: {} cells, {} RED ({} of them ALL_WRONG), {} GREEN.",
            py_str(&s.item("verdict")?),
            fmt_d(&s.item("cells")?)?,
            fmt_d(&s.item("RED")?)?,
            fmt_d(&s.item("ALL_WRONG")?)?,
            fmt_d(&s.item("GREEN")?)?
        ),
        String::new(),
        format!(
            "| model | verb | thinking | prompt | verdict | {} | token parity |",
            ENGINES.join(" | ")
        ),
        format!("|---|---|---|---|---|{}---|", "---|".repeat(ENGINES.len())),
    ])
}

/// Each model's display name by sha: its name, else the sha's first 12.
fn model_names(r: &Val) -> PyResult<Dict> {
    let mut names = Dict::new();
    for m in r.get_or("models", Val::List(vec![]))?.iter()? {
        let sha = m.item("sha256")?;
        let default = sha.slice_to(12)?;
        let name = m.get_or("name", default)?;
        names.set(sha, name)?;
    }
    Ok(names)
}

/// One engine's column: its mark, then its answer or why it gave none.
fn engine_col(v: &Val) -> PyResult<String> {
    let mark = if v.item("correct")?.truthy() {
        "✅"
    } else if v.get("answered")?.truthy() {
        "❌"
    } else {
        "⛔"
    };
    let text = if v.get("answered")?.truthy() {
        short(&v.item("answer")?, 48)?
    } else {
        short(&v.get("why")?, 60)?
    };
    Ok(format!("{mark} {text}"))
}

fn parity_col(tp: &Val) -> PyResult<String> {
    if !tp.get("measured")?.truthy() {
        return concat_str("unmeasured: ", &tp.get_or("why", Val::str(""))?);
    }
    Ok(format!(
        "{} (apr {} vs {}; first diff {})",
        if tp.item("parity")?.truthy() {
            "="
        } else {
            "≠"
        },
        fmt_d(&tp.item("apr_count")?)?,
        fmt_d(&tp.item("llama_cpp_count")?)?,
        py_str(&tp.item("first_divergence")?)
    ))
}

/// A cell's table row.
fn cell_row(c: &Val, names: &Dict) -> PyResult<String> {
    let k = c.item("key")?;
    let mut cols: Vec<String> = Vec::new();
    for e in ENGINES {
        cols.push(engine_col(&c.item("engines")?.item(e)?)?);
    }
    let tps = parity_col(&c.item("token_parity")?)?;
    let name = model_name(names, &k.item("model_sha256")?)?;
    let pid = k.item("prompt_id")?;
    let at = if k.get("mode")?.truthy() {
        concat_str("@", &k.item("mode")?)?
    } else {
        String::new()
    };
    let prompt_col = add_str(&pid, &at)?;
    Ok(format!(
        "| {} | {} | {} | {} | **{}** | {} | {} |",
        name,
        py_str(&k.item("verb")?),
        py_str(&k.item("thinking")?),
        prompt_col,
        py_str(&c.item("verdict")?),
        cols.join(" | "),
        tps
    ))
}

/// A reference's column text in a deterministic row.
fn det_ref(e: &Val, v: &Val) -> PyResult<String> {
    let text = if v.item("produced")?.truthy() {
        if v.get("equal")?.truthy() {
            "= ".to_string()
        } else {
            format!("≠ at {}", py_str(&v.get("first_difference")?))
        }
    } else {
        short(&v.get("why")?, 50)?
    };
    Ok(format!("{}: {}", py_str(e), text))
}

/// A deterministic row's table row.
fn det_row_md(d: &Val, names: &Dict) -> PyResult<String> {
    let k = d.item("key")?;
    let references = d.item("references")?;
    let Val::Dict(refs_d) = &references else {
        return Err(no_attr(&references, "items"));
    };
    let mut refs: Vec<String> = Vec::new();
    for (e, v) in refs_d.iter() {
        refs.push(det_ref(e, v)?);
    }
    let kind = py_str(&d.item("kind")?);
    let name = model_name(names, &k.item("model_sha256")?)?;
    let pid = py_str(&k.item("prompt_id")?);
    let thinking = py_str(&k.get_or("thinking", Val::str(""))?);
    let verdict = py_str(&d.item("verdict")?);
    let apr = d.item("apr")?;
    let apr_col = if apr.item("produced")?.truthy() {
        "produced".to_string()
    } else {
        short(&apr.item("why")?, 50)?
    };
    Ok(format!(
        "| {kind} | {name} | {pid} | {thinking} | **{verdict}** | {apr_col} | {} |",
        refs.join("; ")
    ))
}

/// The deterministic rows' table.
fn det_section(r: &Val, s: &Val, names: &Dict) -> PyResult<Vec<String>> {
    let mut lines: Vec<String> = vec![
        String::new(),
        format!(
            "Deterministic rows (byte-equal or RED): {}",
            py_str(&s.get("deterministic")?)
        ),
        String::new(),
        "| kind | model | prompt | thinking | verdict | apr | references |".into(),
        "|---|---|---|---|---|---|---|".into(),
    ];
    for d in r.item("deterministic")?.iter()? {
        lines.push(det_row_md(&d, names)?);
    }
    Ok(lines)
}

/// `render_md(r)`: the receipt as a markdown table.
pub fn render_md(r: &Val) -> PyResult<String> {
    let s = r.item("summary")?;
    let mut lines = md_header(r, &s)?;
    let names = model_names(r)?;
    for c in r.item("cells")?.iter()? {
        lines.push(cell_row(&c, &names)?);
    }
    if r.get("deterministic")?.truthy() {
        lines.extend(det_section(r, &s, &names)?);
    }
    if r.get("greedy")?.truthy() {
        let g = dumps(&r.item("greedy")?, true)?;
        lines.extend([
            String::new(),
            format!(
                "Greedy divergence (REPORTED, not judged): {}",
                slice_chars(&g, 0, 600)
            ),
        ]);
    }
    lines.extend([
        String::new(),
        "✅ correct · ❌ answered, wrong · ⛔ did not answer (reason shown).".into(),
        "Rates and token counts are in the JSON, as each engine reported them, and are not judged."
            .into(),
        String::new(),
    ]);
    let nc = r.get("not_covered")?;
    if nc.truthy() {
        lines.extend([
            format!(
                "Not covered by this run (the issue requires them): {}.",
                py_join("; ", &nc)?
            ),
            String::new(),
        ]);
    }
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(59), (1970, 3, 1));
        assert_eq!(civil_from_days(11_016), (2000, 2, 29));
        assert_eq!(civil_from_days(11_017), (2000, 3, 1));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
        assert_eq!(
            judged_at(951_782_400 + 3600 * 13 + 60 * 7 + 59),
            "2000-02-29T13:07Z"
        );
    }

    #[test]
    fn short_cuts_by_chars() {
        assert_eq!(short(&Val::str("  a \n b "), 48).unwrap(), "a b");
        assert_eq!(short(&Val::None, 48).unwrap(), "");
        assert_eq!(short(&Val::str("é".repeat(5)), 3).unwrap(), "éé…");
        assert_eq!(
            short(&Val::int(3), 48).unwrap_err().msg,
            "'int' object has no attribute 'split'"
        );
    }
}
