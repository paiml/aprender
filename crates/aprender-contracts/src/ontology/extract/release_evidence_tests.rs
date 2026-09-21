//! aprender#3715 — `extract:release-evidence` in isolation: the universe, the join, and what is computed. The
//! shapes' verdicts over the same graphs are `tests/fixtures/ont/release-*` on the CLI
//! (`crates/aprender-contracts-cli/tests/ont_release_readiness.rs`).

use super::*;
use crate::ontology::extract::release_inputs::{derive_rungs, Consumer};

const MC: &str = "1111111111111111111111111111111111111111";
const BUMP: &str = "2222222222222222222222222222222222222222";
const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

/// A throwaway repo: a ladder contract with two required hosts, a context-rung file, and whatever receipts
/// the case writes. Returns (tempdir, contract dir).
fn repo(ladder_rungs: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let t = tempfile::tempdir().expect("tempdir");
    let c = t.path().join("contracts");
    std::fs::create_dir_all(&c).expect("contracts dir");
    std::fs::write(
        c.join("ladder.yaml"),
        format!(
            "name: ladder\nentity: {{type: gguf}}\nladder:\n  hosts:\n    - {{id: lambda, cc: sm_89, required: true}}\n    - {{id: gx10, cc: sm_121, required: true}}\n    - {{id: yoga, cc: sm_86, required: false}}\n  rungs:\n{ladder_rungs}"
        ),
    )
    .expect("ladder");
    let ev = t.path().join("evidence/release");
    std::fs::create_dir_all(&ev).expect("evidence dir");
    std::fs::write(
        ev.join("context-rungs.json"),
        r#"{"schema":"apr-release-context-rungs/v1",
            "consumers":[{"consumer":"rah","max_prompt_tokens":1000,"basis":"measured","source":"u1"}],
            "rungs":[{"id":"golden","tokens":0,"derived_from":["g"]},{"id":"consumer-max"}]}"#,
    )
    .expect("rungs");
    (t, c)
}

fn write_receipt(root: &Path, host: &str, body: &str) {
    let d = root.join("evidence/dogfood/models/0.69.1");
    std::fs::create_dir_all(&d).expect("receipt dir");
    std::fs::write(d.join(format!("{host}.json")), body).expect("receipt");
}

fn receipt(host: &str, apr_sha: &str, inventory: &str, cells: &str) -> String {
    format!(
        r#"{{"schema":"apr-model-ladder-receipt/v2","host":"{host}","version":"0.69.1","sha":"x",
            "apr_sha":"{apr_sha}","cc":"8.9","gpu":"g","inventory":[{inventory}],"cells":[{cells}],"rungs":[]}}"#
    )
}

fn subject() -> Subject {
    Subject::new("0.69.1", MC).expect("subject")
}

/// A generic surface (#3745 S2): one command that GENERATES from a model and a prompt. Not an apr verb, so the
/// hand-list guard (S3) has nothing to find here.
const SURFACE: &str = r#"{"schema":"apr-cli-surface/v1.1","binary":{"version":"0.69.1","git_sha":"t"},"global_args":[],
"commands":[{"path":["gen"],"key":"gen","leaf":true,"generates":true,"args":[
 {"id":"model","positional":true,"required":true,"value_type":"path","role":"model"},
 {"id":"prompt","long":"prompt","value_type":"text","role":"prompt"}]}]}"#;

/// `subject()` with the generic surface written into the repo.
fn subject_with_surface(root: &Path) -> Subject {
    let path = root.join("surface.json");
    std::fs::write(&path, SURFACE).expect("surface");
    let mut s = subject();
    s.surface = Some(path);
    s
}

/// The matrix cell id of `gen` for (host, file, thinking, rung).
fn gen_id(host: &str, file: &str, thinking: &str, rung: &str) -> String {
    format!("gen/{host}/{file}/think-{thinking}/{rung}/--prompt/defaults")
}

/// A passing row for `id`.
fn row(id: &str) -> String {
    format!(
        r#"{{"cell_id":"{id}","prompt_tokens":2000,"max_tokens":512,"answer_chars":12,
             "verdict":"pass","backend":"cuda","fallback":false,"rc":0}}"#
    )
}

#[test]
fn a_subject_refuses_a_short_or_empty_sha_and_an_empty_version() {
    assert!(
        Subject::new("0.69.1", "225b2a9ab").is_err(),
        "a prefix is refused, never matched"
    );
    assert!(Subject::new("", MC).is_err());
    assert!(subject().with_receipts_commit("abc").is_err());
    let s = subject().with_receipts_commit(BUMP).expect("40 hex");
    assert_eq!(s.measured_commit(), BUMP);
    assert_eq!(subject().measured_commit(), MC);
}

#[test]
fn consumer_max_is_the_max_over_the_records_and_names_a_consumer_with_no_number() {
    let consumers = vec![
        Consumer {
            name: "rah".into(),
            max_prompt_tokens: Some(148_000),
            basis: "measured".into(),
            source: "u1".into(),
        },
        Consumer {
            name: "rmedia".into(),
            max_prompt_tokens: Some(128_000),
            basis: "plan".into(),
            source: "u2".into(),
        },
        Consumer {
            name: "arbiter".into(),
            max_prompt_tokens: None,
            basis: String::new(),
            source: "u3".into(),
        },
    ];
    let rung = ContextRung {
        id: "consumer-max".into(),
        tokens: Some(7),
        long: true,
        derived_from: vec![],
        unmeasured_consumers: vec![],
    };
    let out = derive_rungs(vec![rung], &consumers);
    assert_eq!(
        out[0].tokens,
        Some(148_000),
        "a declared number is ignored: the rung is derived"
    );
    assert_eq!(out[0].derived_from.len(), 3);
    assert_eq!(out[0].unmeasured_consumers, vec!["arbiter".to_string()]);
}

#[test]
fn the_universe_is_the_inventory_plus_the_cuda_rungs_listed_for_the_host_and_every_cell_gets_a_node(
) {
    // rung a: cuda, every host · rung b: cpu only (not a cuda obligation) · rung c: cuda, gx10 only
    let (t, c) = repo(&format!(
        "    - {{id: a, sha256: {SHA_A}, arch: qwen2, gguf: a.gguf, backends: [cpu, cuda], required: true}}\n    - {{id: b, sha256: {}, arch: qwen2, gguf: b.gguf, backends: [cpu], required: true}}\n    - {{id: c, sha256: {}, arch: qwen3, gguf: c.gguf, backends: [cuda], hosts: [gx10], required: true}}\n",
        "c".repeat(64),
        "d".repeat(64)
    ));
    // lambda holds e (inventory) and one file it never hashed
    let inv = format!(r#"{{"file":"e.gguf","sha256":"{SHA_B}"}},{{"file":"nohash.gguf"}}"#);
    let id = gen_id("lambda", "a.gguf", "off", "golden");
    write_receipt(t.path(), "lambda", &receipt("lambda", MC, &inv, &row(&id)));
    let mut g = Graph::new();
    let st = extract(&mut g, &c, &subject_with_surface(t.path())).expect("extracts");
    assert_eq!(st.required_hosts, 2, "yoga is not required");
    // lambda: a (rung) + e (inventory) · gx10: a + c (no receipt, but the rungs listed for it stay)
    assert_eq!(st.models, 4);
    let files: BTreeSet<(&str, &str)> = st
        .derived
        .iter()
        .filter_map(|c| Some((c.host.as_str(), c.model_file.as_deref()?)))
        .collect();
    assert!(
        !files.iter().any(|(_, f)| *f == "b.gguf"),
        "a cpu-only rung owes no cuda cell"
    );
    assert!(files.contains(&("gx10", "c.gguf")) && !files.contains(&("lambda", "c.gguf")));
    // no measured length or modes: both modes × (golden, consumer-max, declared), pairwise with the one shape
    let a_lambda = st
        .derived
        .iter()
        .filter(|c| {
            c.host == "lambda"
                && c.model_file.as_deref() == Some("a.gguf")
                && c.kind == CellKind::Matrix
        })
        .count();
    assert_eq!(a_lambda, 6, "every (thinking, rung) pair is a cell");
    assert_eq!(st.cells, st.derived.len(), "every derived cell is a node");
    assert_eq!(st.cells_with_row, 1);
    let lambda = iri_path("release-host", &["0.69.1", "lambda"]);
    assert_eq!(
        g.objects(&lambda, &rel("unmeasuredModel")).len(),
        1,
        "an inventory row with no hash is named, never dropped"
    );
    let gx10 = iri_path("release-host", &["0.69.1", "gx10"]);
    assert!(
        g.objects(&gx10, &rel("hostReceipt")).is_empty(),
        "no receipt is a missing edge"
    );
    assert!(!g.to_ntriples().contains("_:"), "no blank nodes (R-15)");
}

#[test]
fn a_row_keys_by_its_cell_id_alone_and_carries_fresh_and_context_met() {
    let (t, c) = repo(&format!(
        "    - {{id: a, sha256: {SHA_A}, arch: qwen2, gguf: a.gguf, backends: [cuda], hosts: [lambda], required: true}}\n"
    ));
    let id = gen_id("lambda", "a.gguf", "off", "consumer-max");
    let rows = [
        row(&id),
        row(&gen_id("lambda", "a.gguf", "off", "no-such-rung")),
        row(&id).replace(&format!(r#""cell_id":"{id}","#), ""),
    ]
    .join(",");
    let inv = format!(r#"{{"file":"a.gguf","sha256":"{SHA_A}"}}"#);
    write_receipt(t.path(), "lambda", &receipt("lambda", BUMP, &inv, &rows));
    let mut g = Graph::new();
    let st = extract(&mut g, &c, &subject_with_surface(t.path())).expect("extracts");
    assert_eq!(st.cells_with_row, 1);
    assert_eq!(
        st.orphan_rows, 2,
        "an id the surface does not derive, and a row with no id"
    );
    let mut segs = vec!["0.69.1"];
    segs.extend(id.split('/'));
    let cellnode = iri_path("release-cell", &segs);
    let row_n = g.objects(&cellnode, &rel("row"))[0]
        .as_iri()
        .expect("iri")
        .to_string();
    let lit = |g: &Graph, p: &str| {
        g.objects(&row_n, &rel(p))[0]
            .as_literal()
            .map(|(v, _)| v.to_string())
            .expect("literal")
    };
    assert_eq!(
        lit(&g, "fresh"),
        "false",
        "measured at BUMP, released at MC, no --receipts-commit"
    );
    assert_eq!(
        lit(&g, "contextMet"),
        "true",
        "2000 prompt tokens ≥ consumer-max 1000"
    );
    let mut g2 = Graph::new();
    let s2 = subject_with_surface(t.path())
        .with_receipts_commit(BUMP)
        .expect("sha");
    extract(&mut g2, &c, &s2).expect("extracts");
    assert_eq!(
        lit(&g2, "fresh"),
        "true",
        "T-4: R7 proved BUMP ≡ MC and said so"
    );
}

#[test]
fn a_kernel_on_one_hosts_dispatch_path_is_an_obligation_on_every_required_host() {
    let (t, c) = repo(&format!(
        "    - {{id: a, sha256: {SHA_A}, arch: qwen2, gguf: a.gguf, backends: [cuda], required: true}}\n"
    ));
    let kd = t.path().join("evidence/dogfood/kernels/0.69.1");
    std::fs::create_dir_all(&kd).expect("kernel dir");
    std::fs::write(
        kd.join("gx10.json"),
        format!(
            r#"{{"schema":"apr-kernel-diff-receipt/v1","host":"gx10","version":"0.69.1","apr_sha":"{MC}","sm":"sm_121",
                "dispatch":[{{"sha256":"{SHA_A}","file":"a.gguf","kernels":[{{"kernel":"q4k_gemv","quant":"q4_k"}}]}}],
                "rows":[{{"kernel":"q4k_gemv","quant":"q4_k","reference":"cpu","max_err":0.001,"bound":0.004,"bound_source":"pair","verdict":"pass"}},
                        {{"kernel":"q4k_gemv","quant":"q6_k","reference":"cpu","max_err":0.001,"bound":0.004,"bound_source":"pair","verdict":"pass"}}]}}"#
        ),
    )
    .expect("kernel receipt");
    let mut g = Graph::new();
    let st = extract(&mut g, &c, &subject()).expect("extracts");
    assert_eq!(st.kernel_cells, 2, "one kernel × two required hosts");
    let lambda = iri_path("release-kernel", &["0.69.1", "lambda", "q4k_gemv", "q4_k"]);
    let gx10 = iri_path("release-kernel", &["0.69.1", "gx10", "q4k_gemv", "q4_k"]);
    assert!(
        g.objects(&lambda, &rel("row")).is_empty(),
        "measured on gx10 only: lambda's row is missing"
    );
    assert_eq!(
        g.objects(&gx10, &rel("row")).len(),
        1,
        "the q6_k row is a different cell (#3712, eb)"
    );
    let host = iri_path("release-host", &["0.69.1", "lambda"]);
    assert_eq!(
        g.objects(&host, &rel("modelWithoutDispatch")).len(),
        1,
        "lambda holds a.gguf and has no dispatch list for it"
    );
}

#[test]
fn a_foreign_context_or_kernel_schema_is_refused_by_name() {
    let (t, c) = repo("");
    std::fs::write(
        t.path().join("evidence/release/context-rungs.json"),
        r#"{"schema":"something/v9"}"#,
    )
    .expect("rewrite");
    let e = extract(&mut Graph::new(), &c, &subject()).expect_err("refused");
    assert!(e.to_string().contains("something/v9"), "{e}");
}

#[test]
fn a_measured_length_and_thinking_mode_shape_the_cells_and_a_think_on_row_must_close_and_answer() {
    let (t, c) = repo("");
    // e.gguf: 1500-token context (golden 0 and consumer-max 1000 fit), off only
    // f.gguf: 800-token context (consumer-max 1000 does NOT fit), modes unmeasured → both
    let inv = format!(
        r#"{{"file":"e.gguf","sha256":"{SHA_A}","context_length":1500,"thinking_modes":["off"],"thinking_markers":[]}},
           {{"file":"f.gguf","sha256":"{SHA_B}","context_length":800}}"#
    );
    let on = gen_id("lambda", "f.gguf", "on", "golden");
    let unclosed = row(&on).replace(
        r#""answer_chars":12"#,
        r#""answer_chars":0,"think_closed":false"#,
    );
    let declared = gen_id("lambda", "e.gguf", "off", "declared"); // 2000 + 512 ≥ 1500
    let rows = [unclosed, row(&declared)].join(",");
    write_receipt(t.path(), "lambda", &receipt("lambda", MC, &inv, &rows));
    let mut g = Graph::new();
    let st = extract(&mut g, &c, &subject_with_surface(t.path())).expect("extracts");
    let m = |f: &str| {
        st.derived
            .iter()
            .filter(|c| c.model_file.as_deref() == Some(f) && c.kind == CellKind::Matrix)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        m("e.gguf").len(),
        3,
        "off only × (golden, consumer-max, declared)"
    );
    assert!(m("e.gguf")
        .iter()
        .all(|c| c.thinking.as_deref() == Some("off")));
    assert_eq!(
        m("f.gguf").len(),
        4,
        "both modes × (golden, declared): consumer-max does not fit"
    );
    assert!(!m("f.gguf")
        .iter()
        .any(|c| c.rung.as_deref() == Some("consumer-max")));
    let node = |id: &str| {
        let mut segs = vec!["0.69.1"];
        segs.extend(id.split('/'));
        iri_path("release-cell", &segs)
    };
    let lit = |id: &str, p: &str| {
        let r = g.objects(&node(id), &rel("row"))[0]
            .as_iri()
            .expect("iri")
            .to_string();
        g.objects(&r, &rel(p))[0]
            .as_literal()
            .map(|(v, _)| v.to_string())
    };
    assert_eq!(
        lit(&on, "thinkOk").as_deref(),
        Some("false"),
        "an unclosed think block"
    );
    assert_eq!(
        lit(&on, "answered").as_deref(),
        Some("false"),
        "an empty answer"
    );
    assert_eq!(
        lit(&declared, "contextMet").as_deref(),
        Some("true"),
        "prompt + budget fill the declared length"
    );
}

#[test]
fn a_rung_file_that_redeclares_the_per_model_rung_is_refused() {
    let (t, c) = repo("");
    std::fs::write(
        t.path().join("evidence/release/context-rungs.json"),
        r#"{"schema":"apr-release-context-rungs/v1","consumers":[],"rungs":[{"id":"declared","tokens":4096}]}"#,
    )
    .expect("rewrite");
    let e = extract(&mut Graph::new(), &c, &subject()).expect_err("refused");
    assert!(e.to_string().contains("declared"), "{e}");
}

#[test]
fn long_rungs_are_owed_per_the_ladders_long_rungs_for_and_an_echo_that_disagrees_is_named() {
    // qwen35 is a family that owes everything; for qwen2 only the representative file does
    let (t, c) = repo(
        "  cells:\n    long_rungs_for: {families: [qwen35], representatives: {qwen2: rep.gguf}}\n",
    );
    std::fs::write(
        t.path().join("evidence/release/context-rungs.json"),
        r#"{"schema":"apr-release-context-rungs/v1",
            "consumers":[{"consumer":"rah","max_prompt_tokens":1000,"basis":"measured","source":"u1"}],
            "rungs":[{"id":"4k","tokens":40,"derived_from":["g"]},{"id":"consumer-max","long":true}]}"#,
    )
    .expect("rungs");
    let inv = format!(
        r#"{{"file":"big.gguf","sha256":"{SHA_A}","arch":"qwen35","context_length":262144,"thinking_modes":["on","off"],"thinking_markers":["<think>","enable_thinking"]}},
           {{"file":"small.gguf","sha256":"{SHA_B}","arch":"qwen2","context_length":32768,"thinking_modes":["off"],"thinking_markers":["<think>","enable_thinking"],"owes_long_rungs":true}}"#
    );
    write_receipt(t.path(), "lambda", &receipt("lambda", MC, &inv, ""));
    let mut g = Graph::new();
    let st = extract(&mut g, &c, &subject_with_surface(t.path())).expect("extracts");
    let rungs = |f: &str| {
        st.derived
            .iter()
            .filter(|c| c.model_file.as_deref() == Some(f) && c.kind == CellKind::Matrix)
            .filter_map(|c| c.rung.clone())
            .collect::<BTreeSet<String>>()
    };
    let all: BTreeSet<String> = ["4k", "consumer-max", "declared"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    assert_eq!(rungs("big.gguf"), all, "a qwen35 model owes every rung");
    assert_eq!(
        rungs("small.gguf"),
        BTreeSet::from(["4k".to_string()]),
        "qwen2, not the representative"
    );
    let small = iri("model", SHA_B);
    let lit = |p: &str| {
        g.objects(&small, &rel(p))
            .first()
            .and_then(|t| t.as_literal())
            .map(|(v, _)| v.to_string())
    };
    assert_eq!(lit("owesLongRungs").as_deref(), Some("false"));
    assert!(
        lit("longRungsMismatch").is_some_and(|m| m.contains("small.gguf")),
        "the echo said true"
    );
    assert!(
        lit("thinkingContradiction").is_some_and(|m| m.contains("<think>")),
        "off only, though the template carries the enable_thinking switch"
    );
    let big = iri("model", SHA_A);
    assert!(g.objects(&big, &rel("thinkingContradiction")).is_empty());
    assert!(
        g.objects(&big, &rel("longRungsMismatch")).is_empty(),
        "no echo, nothing to disagree with"
    );
}

#[test]
fn a_discrete_kernel_is_within_bound_only_on_zero_mismatches_or_near_ties() {
    use crate::ontology::extract::release_inputs::KernelRow;
    let row = |mism: Option<u64>, tie: Option<f64>, err: Option<f64>| KernelRow {
        kernel: "topk".into(),
        quant: "f32".into(),
        reference: "cpu".into(),
        max_err: err,
        index_mismatch: mism,
        tie_margin: tie,
        bound: Some(0.01),
        bound_source: "pair".into(),
        verdict: "pass".into(),
        reason: String::new(),
    };
    assert!(row(Some(0), None, None).within_bound(), "same expert sets");
    assert!(
        row(Some(3), Some(0.002), None).within_bound(),
        "three near-ties"
    );
    assert!(
        !row(Some(1), Some(0.5), Some(0.0)).within_bound(),
        "a real divergence; max_err is moot"
    );
    assert!(
        !row(Some(1), None, None).within_bound(),
        "a mismatch with no margin measured"
    );
    assert!(
        !row(None, None, None).within_bound(),
        "nothing measured is outside every bound"
    );
}

#[test]
fn the_positive_control_fires_without_any_release_subject_or_file() {
    // PMAT-3704 R-3: drawn on every gate run — a sample cell with one fresh row, and a planted cell with none
    // that must still be a node (the mutant that emits only measured cells turns this false; measured)
    assert!(positive_control());
}
