//! EXT-24 (aprender#4406): the CRUX bind receipt is complete.
//!
//! `evidence/ext-001/EXT-24/crux-bind-receipt.json` pins every competitor arm
//! of EXT-001 §10 (C1..C7). Done-when: every arm is pinned or carries a
//! `Refused`/`NotRun` reason, the matrix lists each EXT surface exactly as
//! `dogfood-model-lifecycle-v1.yaml` §`crux_surfaces` does, and the existing
//! inference contracts that C2 reuses are named by a path that parses.
//!
//! The arm list is the spec's table, copied here as `SPEC_ARMS`: a receipt
//! that drops an arm, or adds one the spec does not name, is a violation.
//! `metadata.category` is not modeled by `Contract` (serde drops it), so the
//! category enumeration reads it from the raw YAML of each file that the
//! loader accepts.

use crate::schema::parse_contract_str;
use serde_json::Value as Json;
use serde_yaml::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const RECEIPT: &str = include_str!("../../../../evidence/ext-001/EXT-24/crux-bind-receipt.json");
const DOGFOOD: &str = include_str!("../../../../contracts/dogfood-model-lifecycle-v1.yaml");

/// EXT-001 §10, rows C1..C7.
const EXT_SURFACES: [&str; 7] = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"];

/// EXT-001 §10 (spec sha256 a47a6c0386da), one entry per competitor arm.
const SPEC_ARMS: [(&str, &str); 24] = [
    ("C1", "unsloth-q4_k_m"),
    ("C1", "bartowski-q4_k_m"),
    ("C1", "ollama-qwen3.5-4b"),
    ("C1", "unsloth-ud-q4_k_xl"),
    ("C1", "reference-bf16"),
    ("C1", "comparator-llama.cpp"),
    ("C2", "llama.cpp"),
    ("C2", "ollama"),
    ("C2", "mistral.rs"),
    ("C3", "peft-trl-reference"),
    ("C3", "unsloth"),
    ("C4", "llama-quantize+imatrix"),
    ("C5", "mlflow"),
    ("C5", "dvc"),
    ("C5", "wandb"),
    ("C6", "qwen-official"),
    ("C6", "unsloth"),
    ("C6", "bartowski"),
    ("C6", "ollama-library"),
    ("C7", "upstream-stock-4b"),
    ("C7", "qwen3.5-9b-control"),
    ("C7", "haiku-4.5"),
    ("C7", "agy-lane"),
    ("C7", "typesafe-jev"),
];

const STATUSES: [&str; 3] = ["Pinned", "Refused", "NotRun"];

fn is_hex(s: &str, n: usize) -> bool {
    s.len() == n
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Checks every hash-shaped key under `v`; returns how many were valid and
/// pushes one violation per malformed value.
fn check_hashes(v: &Json, at: &str, bad: &mut Vec<String>) -> usize {
    let mut ok = 0;
    match v {
        Json::Object(m) => {
            for (k, x) in m {
                let want = match k.as_str() {
                    "sha256" | "manifest_sha256" | "wheel_sha256" => Some("64 hex"),
                    "commit" | "revision" | "runner_commit" => Some("40 hex"),
                    "digest" | "model_layer" => Some("sha256:<64 hex>"),
                    _ => None,
                };
                match (want, x.as_str()) {
                    (Some(w), Some(s)) => {
                        let good = match w {
                            "64 hex" => is_hex(s, 64),
                            "40 hex" => is_hex(s, 40),
                            _ => s.strip_prefix("sha256:").is_some_and(|h| is_hex(h, 64)),
                        };
                        if good {
                            ok += 1;
                        } else {
                            bad.push(format!("{at}: {k} = {s:?} is not {w}"));
                        }
                    }
                    (Some(w), None) => bad.push(format!("{at}: {k} is not a string ({w})")),
                    (None, _) => ok += check_hashes(x, at, bad),
                }
            }
        }
        Json::Array(a) => ok += a.iter().map(|x| check_hashes(x, at, bad)).sum::<usize>(),
        _ => {}
    }
    ok
}

fn strs(v: Option<&Json>) -> Vec<String> {
    v.and_then(Json::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Json::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn yaml_strs(v: &Value, key: &str) -> Vec<String> {
    v.get(key)
        .and_then(Value::as_sequence)
        .map(|s| {
            s.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Every reason the receipt fails EXT-24's done-when. `read(path)` returns the
/// text of a repo-relative path, or `None` if there is no such file.
fn bind_violations(
    receipt: &str,
    dogfood: &str,
    read: &dyn Fn(&str) -> Option<String>,
) -> Vec<String> {
    let mut bad = Vec::new();
    let r: Json = match serde_json::from_str(receipt) {
        Ok(r) => r,
        Err(e) => return vec![format!("receipt does not parse: {e}")],
    };
    let arms = r
        .get("arms")
        .and_then(Json::as_array)
        .cloned()
        .unwrap_or_default();

    // every spec arm exactly once, and nothing the spec does not name
    let mut seen: BTreeMap<(String, String), usize> = BTreeMap::new();
    for a in &arms {
        let s = a.get("surface").and_then(Json::as_str).unwrap_or("?");
        let n = a.get("arm").and_then(Json::as_str).unwrap_or("?");
        *seen.entry((s.to_string(), n.to_string())).or_default() += 1;
    }
    for (s, n) in SPEC_ARMS {
        match seen.get(&(s.to_string(), n.to_string())) {
            None => bad.push(format!("{s}/{n}: spec arm missing from the receipt")),
            Some(&c) if c > 1 => bad.push(format!("{s}/{n}: arm listed {c} times")),
            _ => {}
        }
    }
    for (s, n) in seen.keys() {
        if !SPEC_ARMS.contains(&(s.as_str(), n.as_str())) {
            bad.push(format!("{s}/{n}: arm is not in EXT-001 §10"));
        }
    }

    // each arm is pinned by a real hash, or says why not
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for a in &arms {
        let at = format!(
            "{}/{}",
            a.get("surface").and_then(Json::as_str).unwrap_or("?"),
            a.get("arm").and_then(Json::as_str).unwrap_or("?")
        );
        let status = a.get("status").and_then(Json::as_str).unwrap_or("");
        let Some(st) = STATUSES.iter().find(|x| **x == status) else {
            bad.push(format!(
                "{at}: status {status:?} is not Pinned/Refused/NotRun"
            ));
            continue;
        };
        *counts.entry(st).or_default() += 1;
        let valid = a.get("pin").map_or(0, |p| check_hashes(p, &at, &mut bad));
        let reason = a.get("reason").and_then(Json::as_str).unwrap_or("").trim();
        if *st == "Pinned" && valid == 0 {
            bad.push(format!("{at}: Pinned with no sha256, commit or digest"));
        }
        if *st != "Pinned" && reason.is_empty() {
            bad.push(format!("{at}: {st} with no reason"));
        }
        let unrunnable =
            a.get("comparator").and_then(Json::as_str) == Some("unrunnable-hermetically");
        if unrunnable && *st == "Pinned" {
            bad.push(format!(
                "{at}: unrunnable-hermetically cannot be Pinned (S-12)"
            ));
        }
    }

    // the verdict states the counts the arms produce
    let want = format!(
        "BOUND: {} arms, {} Pinned, {} NotRun, {} Refused;",
        arms.len(),
        counts.get("Pinned").copied().unwrap_or(0),
        counts.get("NotRun").copied().unwrap_or(0),
        counts.get("Refused").copied().unwrap_or(0)
    );
    let verdict = r.get("verdict").and_then(Json::as_str).unwrap_or("");
    if !verdict.starts_with(&want) {
        bad.push(format!("verdict {verdict:?} does not start with {want:?}"));
    }

    // the matrix is the dogfood contract's crux_surfaces, surface for surface
    let doc: Value = match serde_yaml::from_str(dogfood) {
        Ok(d) => d,
        Err(e) => return [bad, vec![format!("dogfood contract does not parse: {e}")]].concat(),
    };
    let surfaces = doc
        .get("crux_surfaces")
        .and_then(Value::as_sequence)
        .cloned()
        .unwrap_or_default();
    let matrix = r
        .get("matrix")
        .and_then(Json::as_array)
        .cloned()
        .unwrap_or_default();
    let row = |id: &str| {
        matrix
            .iter()
            .find(|m| m.get("surface").and_then(Json::as_str) == Some(id))
    };
    for id in EXT_SURFACES {
        let Some(m) = row(id) else {
            bad.push(format!("{id}: EXT surface missing from the matrix"));
            continue;
        };
        let Some(s) = surfaces
            .iter()
            .find(|s| s.get("id").and_then(Value::as_str) == Some(id))
        else {
            bad.push(format!(
                "{id}: no crux_surfaces entry in the dogfood contract"
            ));
            continue;
        };
        for key in ["contracts", "extends"] {
            if strs(m.get(key)) != yaml_strs(s, key) {
                bad.push(format!("{id}: matrix {key} differs from crux_surfaces"));
            }
        }
    }

    // C2's reused inference contracts are named by path, exist and parse
    let reused = strs(r.get("c2_reused_contracts"));
    if reused.is_empty() {
        bad.push("C2: no reused inference contracts named".to_string());
    }
    for ext in row("C2")
        .map(|m| strs(m.get("extends")))
        .unwrap_or_default()
    {
        if !reused.contains(&format!("contracts/{ext}")) {
            bad.push(format!(
                "C2: extended contract {ext} is not in c2_reused_contracts"
            ));
        }
    }
    for p in &reused {
        match read(p) {
            None => bad.push(format!("C2: reused contract {p} does not exist")),
            Some(t) => {
                if let Err(e) = parse_contract_str(&t) {
                    bad.push(format!("C2: reused contract {p} does not parse: {e}"));
                }
            }
        }
    }
    bad
}

fn repo_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_live(path: &str) -> Option<String> {
    std::fs::read_to_string(repo_root().join(path)).ok()
}

/// letter -> the distinct `metadata.category` names, over every live
/// `contracts/crux-*.yaml` the loader accepts.
fn live_categories() -> BTreeMap<String, BTreeSet<String>> {
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let dir = repo_root().join("contracts");
    for e in std::fs::read_dir(&dir)
        .expect("contracts/ is readable")
        .filter_map(Result::ok)
    {
        let name = e.file_name().to_string_lossy().into_owned();
        if !(name.starts_with("crux-") && name.ends_with(".yaml")) {
            continue;
        }
        let text = std::fs::read_to_string(e.path()).expect("crux contract is readable");
        if parse_contract_str(&text).is_err() {
            continue;
        }
        let doc: Value = serde_yaml::from_str(&text).expect("a loaded contract is YAML");
        if let Some(c) = doc
            .get("metadata")
            .and_then(|m| m.get("category"))
            .and_then(Value::as_str)
        {
            let letter = c
                .split([' ', '—', '-'])
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            out.entry(letter).or_default().insert(c.trim().to_string());
        }
    }
    out
}

/// EXT-24 done-when: the live receipt is complete, and each planted gap is RED.
#[test]
fn ext_24_crux_bind_receipt_is_complete() {
    let live = bind_violations(RECEIPT, DOGFOOD, &read_live);
    assert!(
        live.is_empty(),
        "EXT-24: the live receipt fails:\n{}",
        live.join("\n")
    );

    let plant = |from: &str, to: &str| {
        assert!(
            RECEIPT.contains(from),
            "plant anchor {from:?} is not in the receipt"
        );
        bind_violations(&RECEIPT.replacen(from, to, 1), DOGFOOD, &read_live)
    };
    let cases: [(&str, &str, &str, &str); 8] = [
        (
            "drop an arm",
            r#""arm": "wandb""#,
            r#""arm": "wandb-gone""#,
            "wandb: spec arm missing",
        ),
        (
            "short sha256",
            "00fe7986ff5f6b463e",
            "00fe7986ff5f6b463",
            "is not 64 hex",
        ),
        (
            "Pinned without a pin",
            r#""status": "Pinned", "pin": {"repo": "ggml-org/llama.cpp", "commit": "d1d3c3396aa13a5f239109a822666c4870490ad5", "tools""#,
            r#""status": "Pinned", "nopin": {"repo": "x", "tools""#,
            "Pinned with no sha256",
        ),
        (
            "NotRun without a reason",
            r#""reason": "S-12: SaaS"#,
            r#""why": "S-12: SaaS"#,
            "NotRun with no reason",
        ),
        (
            "unrunnable Pinned",
            r#""arm": "wandb", "status": "NotRun""#,
            r#""arm": "wandb", "status": "Pinned""#,
            "cannot be Pinned",
        ),
        (
            "stale verdict",
            "BOUND: 24 arms, 18 Pinned",
            "BOUND: 24 arms, 19 Pinned",
            "verdict",
        ),
        (
            "matrix drifts",
            r#""contracts": ["crux-P-04-v1.yaml"]"#,
            r#""contracts": ["crux-P-09-v1.yaml"]"#,
            "C5: matrix contracts differs",
        ),
        (
            "C2 reuse drops an extension",
            r#""contracts/crux-C-27-v1.yaml","#,
            "",
            "crux-C-27-v1.yaml is not in c2_reused",
        ),
    ];
    for (what, from, to, want) in cases {
        let got = plant(from, to);
        assert!(
            got.iter().any(|v| v.contains(want)),
            "EXT-24 planted `{what}` should report {want:?}, got {got:?}"
        );
    }
    let gone = bind_violations(RECEIPT, DOGFOOD, &|p| {
        if p.ends_with("pp-llama-001-perf-gate-v1.yaml") {
            None
        } else {
            read_live(p)
        }
    });
    assert!(
        gone.iter().any(|v| v.contains("does not exist")),
        "a missing reused contract is RED: {gone:?}"
    );
}

/// EXT-24: `metadata.category` is not a closed enum. The receipt says so, and
/// the live tree shows why: letters carry more than one spelling.
#[test]
fn ext_24_crux_category_is_open_and_the_receipt_says_so() {
    let r: Json = serde_json::from_str(RECEIPT).expect("receipt parses");
    let closed = r.pointer("/category_enum/closed").and_then(Json::as_bool);
    assert_eq!(
        closed,
        Some(false),
        "EXT-24: the receipt must state category_enum.closed = false"
    );
    let cats = live_categories();
    assert!(
        cats.len() >= 15,
        "EXT-24: expected the A..P letters, the loader saw {cats:?}"
    );
    let drift: Vec<_> = cats.iter().filter(|(_, n)| n.len() > 1).collect();
    assert!(
        !drift.is_empty(),
        "EXT-24: no letter has two spellings any more; the category is now consistent and \
         category_enum in the receipt is stale: {cats:?}"
    );
}
