//! The owed-set half of `scripts/lib/model_ladder_cells.py` (the cells JUDGE), ported for
//! `ladder-cells-produce` (#5225): `expected_modes`, `owes_long`, `_tokens`, `owed_rungs`,
//! `load_rungs` and `declaimed`. The producer must measure exactly the set the judge demands,
//! so these follow the judge line by line; the judge itself stays in Python for now.
//!
//! Known gap: a `context_length` that is not an int is read with `int()` at the `declared`
//! rung, where the judge would carry the raw value (and usually crash on comparing it).

use crate::pyjson::{
    dict_get, or_empty, py_contains, py_eq, py_int, py_iter, py_repr_value, py_str, Py, PY_NONE,
};
use crate::pystr::py_strip;

pub const RUNGS_SCHEMA: &str = "apr-release-context-rungs/v1";

/// `_tokens`: a rung's size in tokens, or `None` (owed but unsizable).
pub fn tokens(rung: &Py, ctx: &Py, consumer_max: Option<i128>) -> Result<Option<i128>, String> {
    let t = dict_get(rung, "tokens")?;
    if *t != Py::None {
        return py_int(t).map(Some);
    }
    let id = rung.at("id");
    if py_eq(id, &Py::str("consumer-max")) {
        return Ok(consumer_max);
    }
    if py_eq(id, &Py::str("declared")) && *ctx != Py::None {
        return py_int(ctx).map(Some);
    }
    Ok(None)
}

/// `expected_modes`: the thinking modes a model owes, sorted, from its template evidence.
/// `None` when no evidence was measured.
pub fn expected_modes(
    markers: &Py,
    gen_opens_think: bool,
) -> Result<Option<Vec<&'static str>>, String> {
    if *markers == Py::None {
        return Ok(None);
    }
    if py_contains(markers, &Py::str("enable_thinking"))? {
        return Ok(Some(vec!["off", "on"]));
    }
    Ok(Some(vec![if gen_opens_think { "on" } else { "off" }]))
}

/// `owes_long(item, long_for)[0]`: does this item owe the long rungs?
pub fn owes_long(item: &Py, long_for: &Py) -> Result<bool, String> {
    let arch = dict_get(item, "arch")?;
    if *arch == Py::None {
        return Ok(true);
    }
    let families = dict_get(long_for, "families")?;
    if families.truthy() && py_contains(families, arch)? {
        return Ok(true);
    }
    let reps = or_empty(dict_get(long_for, "representatives")?);
    // `.get(arch)` needs a dict whatever arch is; a non-str arch names no key read here.
    let found = dict_get(reps, arch.as_str().unwrap_or_default())?;
    let rep = if arch.as_str().is_some() {
        found
    } else {
        &PY_NONE
    };
    Ok(py_eq(rep, item.at("file")))
}

/// `owed_rungs`: `(rung id, tokens)` for every rung the item owes; tokens `None` = owed but
/// unsizable. A long rung is skipped unless [`owes_long`]; a rung above the model's own
/// declared context is not owed.
pub fn owed_rungs(
    item: &Py,
    rungs: &[Py],
    long_for: &Py,
    consumer_max: Option<i128>,
) -> Result<Vec<(Py, Option<i128>)>, String> {
    let mut long_ids = Vec::new();
    for r in rungs {
        if dict_get(r, "long")?.truthy() {
            long_ids.push(r.at("id").clone());
        }
    }
    let lo = owes_long(item, long_for)?;
    let ctx = item.at("context_length");
    let mut out = Vec::new();
    for rung in rungs {
        let rid = rung.at("id");
        if !lo && long_ids.iter().any(|l| py_eq(l, rid)) {
            continue;
        }
        let tok = tokens(rung, ctx, consumer_max)?;
        if let Some(t) = tok {
            if *ctx != Py::None && t > py_int(ctx)? {
                continue; // above the model's own declared context: not owed
            }
        }
        out.push((rid.clone(), tok));
    }
    Ok(out)
}

/// Iterating `v or []`.
pub fn or_list(v: &Py) -> Result<Vec<Py>, String> {
    if v.truthy() {
        py_iter(v)
    } else {
        Ok(Vec::new())
    }
}

/// `isinstance(n, int) and n > 0`, as the int (a bool counts, as it does in Python).
fn positive_int(n: &Py) -> Option<i128> {
    match n {
        Py::Int(i) if *i > 0 => Some(*i),
        Py::Bool(true) => Some(1),
        _ => None,
    }
}

/// `load_rungs`: `(rungs, consumer_max, rc)`; each FAIL line goes to `out`. A file that
/// sizes nothing is a named FAIL, never an empty universe. `Err` where the judge crashes.
pub fn load_rungs(doc: &Py, out: &mut Vec<String>) -> Result<(Vec<Py>, Option<i128>, i32), String> {
    if *doc == Py::None {
        out.push(
            "FAIL  evidence/release/context-rungs.json is missing or unreadable -- no context \
             rung is sized, so no cell can be judged (#3715)"
                .to_owned(),
        );
        return Ok((Vec::new(), None, 1));
    }
    let schema = dict_get(doc, "schema")?;
    if !py_eq(schema, &Py::str(RUNGS_SCHEMA)) {
        out.push(format!(
            "FAIL  context rungs schema {} is not {RUNGS_SCHEMA}",
            py_repr_value(schema)
        ));
        return Ok((Vec::new(), None, 1));
    }
    let mut rc = 0;
    let mut rungs = Vec::new();
    for r in or_list(dict_get(doc, "rungs")?)? {
        if dict_get(&r, "id")?.truthy() {
            rungs.push(r);
        }
    }
    if rungs.is_empty() {
        out.push(
            "FAIL  the context rungs file names no rung -- it owes nothing, which is not a pass"
                .to_owned(),
        );
        rc = 1;
    }
    let mut maxima = Vec::new();
    for c in or_list(dict_get(doc, "consumers")?)? {
        if let Some(n) = positive_int(dict_get(&c, "max_prompt_tokens")?) {
            maxima.push(n);
        } else {
            out.push(format!(
                "FAIL  consumer {} has no max_prompt_tokens -- consumer-max cannot be derived \
                 without it",
                py_repr_value(c.at("name"))
            ));
            rc = 1;
        }
    }
    let consumer_max = if rc == 0 {
        maxima.into_iter().max()
    } else {
        None
    };
    Ok((rungs, consumer_max, rc))
}

/// One `cells.declaimed` entry: `(host, arch, entry)`. A later entry for the same pair wins,
/// as the judge's dict assignment does; look up with [`find_declaimed`].
pub type Declaimed = Vec<(Py, Py, Py)>;

/// `str(d.get(k) or "").strip()` is empty.
fn blank(v: &Py) -> bool {
    !v.truthy() || py_strip(&py_str(v)).is_empty()
}

/// `declaimed`: the de-claimed `(host, arch)` pairs and rc; a bare entry is a FAIL line in
/// `out`, never a quiet exemption.
pub fn declaimed(c: &Py, hosts: &[Py], out: &mut Vec<String>) -> Result<(Declaimed, i32), String> {
    let mut got = Vec::new();
    let mut rc = 0;
    for d in or_list(dict_get(c, "declaimed")?)? {
        let d = or_empty(&d).clone();
        let mut bad: Vec<String> = Vec::new();
        for k in ["arch", "host", "until", "why"] {
            if blank(dict_get(&d, k)?) {
                bad.push(k.to_owned());
            }
        }
        let issue_ok = match d.at("issue") {
            Py::Int(i) => *i > 0,
            Py::Bool(b) => *b,
            _ => false,
        };
        if !issue_ok {
            bad.push("issue".to_owned());
        }
        let host = d.at("host");
        if host.truthy() && !hosts.iter().any(|h| py_eq(h, host)) {
            bad.push(format!(
                "host {} (not a declared host)",
                py_repr_value(host)
            ));
        }
        if !bad.is_empty() {
            out.push(format!(
                "FAIL  cells.declaimed entry {} lacks {} -- a de-claim names its arch, host, \
                 issue, restoring release and reason",
                py_repr_value(&d),
                bad.join(", ")
            ));
            rc = 1;
            continue;
        }
        got.push((host.clone(), d.at("arch").clone(), d.clone()));
    }
    Ok((got, rc))
}

/// `dec.get((host, arch))`.
pub fn find_declaimed<'a>(dec: &'a Declaimed, host: &Py, arch: &Py) -> Option<&'a Py> {
    dec.iter()
        .rev()
        .find(|(h, a, _)| py_eq(h, host) && py_eq(a, arch))
        .map(|(_, _, d)| d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyjson::loads;

    fn j(s: &str) -> Py {
        loads(s).expect("valid json")
    }

    const RUNGS: &str = r#"{"schema": "apr-release-context-rungs/v1",
        "consumers": [{"name": "c", "max_prompt_tokens": 7000}],
        "rungs": [{"id": "4k", "tokens": 4096}, {"id": "consumer-max"},
                  {"id": "long", "tokens": 131072, "long": true}, {"id": "declared"},
                  {"id": "big", "tokens": 65536}]}"#;

    #[test]
    fn modes_follow_the_template_evidence() {
        assert_eq!(expected_modes(&Py::None, true), Ok(None));
        let both = j(r#"["enable_thinking", "<think>"]"#);
        assert_eq!(expected_modes(&both, false), Ok(Some(vec!["off", "on"])));
        assert_eq!(
            expected_modes(&j(r#"["<think>"]"#), true),
            Ok(Some(vec!["on"]))
        );
        assert_eq!(expected_modes(&j("[]"), false), Ok(Some(vec!["off"])));
        assert!(expected_modes(&Py::Int(1), false).is_err());
    }

    #[test]
    fn owed_rungs_skip_long_and_oversized() {
        let mut why = Vec::new();
        let (rungs, cmax, rc) = load_rungs(&j(RUNGS), &mut why).expect("loads");
        assert_eq!((cmax, rc), (Some(7000), 0));
        let item = j(r#"{"file": "m.gguf", "arch": "qwen2", "context_length": 32768}"#);
        let no_long = j(r#"{"families": [], "representatives": {}}"#);
        let owed = owed_rungs(&item, &rungs, &no_long, cmax).expect("owed");
        let want = vec![
            (Py::str("4k"), Some(4096)),
            (Py::str("consumer-max"), Some(7000)),
            (Py::str("declared"), Some(32768)),
        ];
        assert_eq!(owed, want);
        let rep = j(r#"{"representatives": {"qwen2": "m.gguf"}}"#);
        let owed = owed_rungs(&item, &rungs, &rep, cmax).expect("owed");
        assert_eq!(owed.len(), 3, "131072 is above the declared 32768");
        let no_size = j(r#"{"file": "m.gguf", "arch": null}"#);
        let owed = owed_rungs(&no_size, &rungs, &no_long, None).expect("owed");
        assert_eq!(owed.len(), 5);
        assert_eq!(owed[1], (Py::str("consumer-max"), None));
        assert_eq!(owed[3], (Py::str("declared"), None));
    }

    #[test]
    fn owes_long_by_family_or_representative() {
        let item = j(r#"{"file": "m.gguf", "arch": "qwen2"}"#);
        let owes = |long_for: &str| owes_long(&item, &j(long_for));
        assert_eq!(
            owes(r#"{"families": [], "representatives": {}}"#),
            Ok(false)
        );
        assert_eq!(owes(r#"{"families": ["qwen2"]}"#), Ok(true));
        assert_eq!(
            owes(r#"{"representatives": {"qwen2": "m.gguf"}}"#),
            Ok(true)
        );
        assert_eq!(
            owes(r#"{"representatives": {"qwen2": "n.gguf"}}"#),
            Ok(false)
        );
        let no_arch = j(r#"{"file": "m.gguf", "arch": null}"#);
        assert_eq!(owes_long(&no_arch, &j("{}")), Ok(true));
    }

    #[test]
    fn positive_int_counts_a_bool() {
        assert_eq!(positive_int(&Py::Bool(true)), Some(1));
        assert_eq!(positive_int(&Py::Bool(false)), None);
        assert_eq!(positive_int(&Py::Int(3)), Some(3));
        assert_eq!(positive_int(&Py::Int(0)), None);
        assert_eq!(positive_int(&Py::str("3")), None);
    }

    #[test]
    fn load_rungs_names_each_fail() {
        let mut why = Vec::new();
        assert_eq!(load_rungs(&Py::None, &mut why), Ok((vec![], None, 1)));
        let bad = j(r#"{"schema": "x", "rungs": []}"#);
        assert_eq!(load_rungs(&bad, &mut why).map(|r| r.2), Ok(1));
        assert_eq!(
            why[1],
            "FAIL  context rungs schema 'x' is not apr-release-context-rungs/v1"
        );
        why.clear();
        let doc = j(
            r#"{"schema": "apr-release-context-rungs/v1", "rungs": [{"id": ""}],
               "consumers": [{"name": "c", "max_prompt_tokens": 0}]}"#,
        );
        assert_eq!(load_rungs(&doc, &mut why), Ok((vec![], None, 1)));
        assert_eq!(why.len(), 2);
        assert!(why[1].starts_with("FAIL  consumer 'c' has no max_prompt_tokens"));
    }

    #[test]
    fn declaimed_needs_every_field() {
        let hosts = [Py::str("t")];
        let mut why = Vec::new();
        let c = j(
            r#"{"declaimed": [{"arch": "qwen3", "host": "t", "issue": 4590,
                                      "until": "0.72", "why": "slow"}]}"#,
        );
        let (dec, rc) = declaimed(&c, &hosts, &mut why).expect("declaimed");
        assert_eq!(rc, 0);
        assert!(find_declaimed(&dec, &Py::str("t"), &Py::str("qwen3")).is_some());
        assert!(find_declaimed(&dec, &Py::str("u"), &Py::str("qwen3")).is_none());
        let c = j(r#"{"declaimed": [{"arch": "qwen3", "host": "u", "issue": 0}]}"#);
        let (dec, rc) = declaimed(&c, &hosts, &mut why).expect("declaimed");
        assert_eq!((dec.len(), rc), (0, 1));
        let c = j(
            r#"{"declaimed": [{"arch": "qwen3", "host": "t", "issue": true,
                                      "until": "0.72", "why": "slow"}]}"#,
        );
        let (dec, rc) = declaimed(&c, &hosts, &mut Vec::new()).expect("declaimed");
        assert_eq!(
            (dec.len(), rc),
            (1, 0),
            "a bool issue is an int, as in Python"
        );
        assert_eq!(
            why[0],
            "FAIL  cells.declaimed entry {'arch': 'qwen3', 'host': 'u', 'issue': 0} lacks \
             until, why, issue, host 'u' (not a declared host) -- a de-claim names its arch, \
             host, issue, restoring release and reason"
        );
    }
}
