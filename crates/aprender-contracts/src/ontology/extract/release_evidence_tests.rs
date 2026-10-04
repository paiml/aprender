//! aprender#3715 — `extract:release-evidence` in isolation: the universe, the join, and what is computed. The
//! shapes' verdicts over the same graphs are `tests/fixtures/ont/release-*` on the CLI
//! (`crates/aprender-contracts-cli/tests/ont_release_readiness.rs`).

use super::*;
use crate::ontology::extract::kernel_cells;
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
    assert_eq!(
        (st.kernel_receipts, st.model_receipts, st.context_rungs),
        (1, 0, 2),
        "the stats count what was read"
    );
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

/// `SURFACE` plus one arg no role type claims, so both ratchet counts are 1 (#4587: kills the emit_surface
/// guard, `!generates_declared` and ratchet_rows mutants).
fn surface_with_unknown(generates: bool) -> cli_surface::Surface {
    let mut text = SURFACE.replace(
        r#"{"id":"prompt","long":"prompt","value_type":"text","role":"prompt"}"#,
        r#"{"id":"prompt","long":"prompt","value_type":"text","role":"prompt"},
 {"id":"x","long":"x","value_type":"text","role":"unknown"}"#,
    );
    if !generates {
        text = text.replace(r#""generates":true,"#, "");
    }
    cli_surface::parse("s.json", &text).expect("parses")
}

/// Every string literal `rel(p)` puts on the release-subject node.
fn subject_lits(g: &Graph, p: &str) -> Vec<String> {
    let n = iri_path("release-subject", &["0.69.1"]);
    let mut v: Vec<String> = g
        .objects(&n, &rel(p))
        .iter()
        .map(|t| t.as_literal().map(|(v, _)| v.to_string()).expect("literal"))
        .collect();
    v.sort();
    v
}

#[test]
fn ratchet_rows_names_both_counts_with_their_values_and_ceilings() {
    let st = cli_surface::SurfaceStats {
        unknown_args: 3,
        stdin_undeclared: 5,
        ..Default::default()
    };
    let r = crate::ontology::extract::release_inputs::SurfaceRatchet {
        unknown_args: 7,
        stdin_undeclared: 11,
    };
    assert_eq!(
        ratchet_rows(&st, Some(r)),
        [
            ("unknown_args", 3, Some(7)),
            ("stdin_undeclared", 5, Some(11))
        ]
    );
    assert_eq!(
        ratchet_rows(&st, None),
        [("unknown_args", 3, None), ("stdin_undeclared", 5, None)]
    );
}

#[test]
fn emit_surface_links_the_surface_and_names_an_undeclared_generate() {
    let s = surface_with_unknown(true);
    let mut g = Graph::new();
    let mut stats = ReleaseStats::default();
    let r = crate::ontology::extract::release_inputs::SurfaceRatchet {
        unknown_args: 5,
        stdin_undeclared: 5,
    };
    emit_surface(&mut g, &subject(), &s, Some(r), &mut stats);
    let sn = iri_path("cli-surface", &["t"]);
    let rel_node = iri_path("release-subject", &["0.69.1"]);
    assert_eq!(
        g.objects(&rel_node, &rel("surface"))[0].as_iri(),
        Some(sn.as_str())
    );
    let lit = |p: &str| {
        g.objects(&sn, &cli_surface::cli(p))[0]
            .as_literal()
            .map(|(v, _)| v.to_string())
            .expect("literal")
    };
    assert_eq!(lit("version"), "0.69.1");
    assert_eq!(lit("file"), "s.json");
    let st = stats.surface.as_ref().expect("stats recorded");
    assert_eq!((st.unknown_args, st.stdin_undeclared), (1, 1));
    assert!(subject_lits(&g, "generatesUndeclared").is_empty());
    assert!(
        subject_lits(&g, "ratchetGrew").is_empty(),
        "below the ceiling"
    );
    assert!(subject_lits(&g, "ratchetBaselineMissing").is_empty());

    let mut g2 = Graph::new();
    let mut stats2 = ReleaseStats::default();
    emit_surface(
        &mut g2,
        &subject(),
        &surface_with_unknown(false),
        Some(r),
        &mut stats2,
    );
    assert_eq!(
        subject_lits(&g2, "generatesUndeclared").len(),
        1,
        "a model command that does not say whether it generates is named"
    );
}

#[test]
fn a_count_above_its_ceiling_grows_the_ratchet_and_one_at_it_does_not() {
    let s = surface_with_unknown(true);
    let mut g = Graph::new();
    let mut stats = ReleaseStats::default();
    // unknown_args 1 == ceiling 1 (holds); stdin_undeclared 1 > ceiling 0 (grew)
    let r = crate::ontology::extract::release_inputs::SurfaceRatchet {
        unknown_args: 1,
        stdin_undeclared: 0,
    };
    emit_surface(&mut g, &subject(), &s, Some(r), &mut stats);
    assert_eq!(
        subject_lits(&g, "ratchetGrew"),
        vec!["stdin_undeclared 1 > ceiling 0 (shrink-only)".to_string()]
    );
    assert!(subject_lits(&g, "ratchetBaselineMissing").is_empty());

    let mut g2 = Graph::new();
    emit_surface(&mut g2, &subject(), &s, None, &mut ReleaseStats::default());
    let missing = subject_lits(&g2, "ratchetBaselineMissing");
    assert_eq!(missing.len(), 2, "{missing:?}");
    assert!(missing[0].starts_with("stdin_undeclared: no committed ceiling in "));
    assert!(missing[1].starts_with("unknown_args: no committed ceiling in "));
    assert!(subject_lits(&g2, "ratchetGrew").is_empty());
}

#[test]
fn the_projection_costs_each_cell_the_median_wall_time_of_its_class() {
    let (t, c) = repo(&format!(
        "    - {{id: a, sha256: {SHA_A}, arch: qwen2, gguf: a.gguf, backends: [cuda], hosts: [lambda], required: true}}\n"
    ));
    // four samples of the (gen, golden) class: the median is the upper middle one, 7000 ms
    let rows = [1000, 9000, 2000, 7000]
        .iter()
        .map(|ms| {
            row(&gen_id("lambda", "a.gguf", "off", "golden"))
                .replace(r#""rc":0"#, &format!(r#""rc":0,"wall_ms":{ms}"#))
        })
        .collect::<Vec<_>>()
        .join(",");
    let inv = format!(r#"{{"file":"a.gguf","sha256":"{SHA_A}"}}"#);
    write_receipt(t.path(), "lambda", &receipt("lambda", MC, &inv, &rows));
    let mut g = Graph::new();
    let st = extract(&mut g, &c, &subject_with_surface(t.path())).expect("extracts");
    assert_eq!(st.model_receipts, 1);
    let p = &st.projection["lambda"];
    let golden = st
        .derived
        .iter()
        .filter(|c| c.host == "lambda" && c.rung.as_deref() == Some("golden"))
        .count();
    let all = st.derived.iter().filter(|c| c.host == "lambda").count();
    assert!(
        golden >= 2 && all > golden,
        "the case needs measured and unmeasured cells"
    );
    assert_eq!(p.cells, all);
    assert_eq!(p.measured_cells, golden);
    assert_eq!(p.unmeasured_cells, all - golden);
    assert_eq!(
        p.projected_secs,
        7 * golden as u64,
        "7000 ms is 7 s per golden cell"
    );
}

#[test]
fn the_dominating_class_is_the_largest_total_and_a_tie_keeps_the_earlier() {
    let (t, c) = repo(&format!(
        "    - {{id: a, sha256: {SHA_A}, arch: qwen2, gguf: a.gguf, backends: [cuda], hosts: [lambda], required: true}}\n"
    ));
    let inv = format!(r#"{{"file":"a.gguf","sha256":"{SHA_A}"}}"#);
    let base = extract(&mut Graph::new(), &c, &subject_with_surface(t.path())).expect("extracts");
    let mine: Vec<&CellSpec> = base.derived.iter().filter(|c| c.host == "lambda").collect();
    // a class's total is its median times EVERY cell in it, measured row or not
    let n = |r: &str| mine.iter().filter(|c| c.rung.as_deref() == Some(r)).count() as u64;
    let has_matrix = |r: &str| {
        mine.iter()
            .any(|c| c.kind == CellKind::Matrix && c.rung.as_deref() == Some(r))
    };
    let rungs: Vec<&str> = mine
        .iter()
        .filter_map(|c| c.rung.as_deref())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    assert!(
        rungs.len() == 3 && rungs.iter().all(|r| has_matrix(r)),
        "the case needs three classes with matrix cells: {rungs:?}"
    );
    // the first class costs 1 s a cell; the second and third both total f·n1·n2 s,
    // which outgrows the first — so the second displaces it and the third only ties
    let f = n(rungs[0]) + 1;
    let wall = |r: &str| match rungs.iter().position(|x| *x == r) {
        Some(0) => 1000,
        Some(1) => 1000 * f * n(rungs[2]),
        _ => 1000 * f * n(rungs[1]),
    };
    let rows = mine
        .iter()
        .filter(|c| c.kind == CellKind::Matrix)
        .map(|c| {
            let ms = wall(c.rung.as_deref().expect("a matrix cell has a rung"));
            row(&c.id).replace(r#""rc":0"#, &format!(r#""rc":0,"wall_ms":{ms}"#))
        })
        .collect::<Vec<_>>()
        .join(",");
    write_receipt(t.path(), "lambda", &receipt("lambda", MC, &inv, &rows));
    let st = extract(&mut Graph::new(), &c, &subject_with_surface(t.path())).expect("extracts");
    assert_eq!(
        st.projection["lambda"].dominating_class,
        format!("gen @ {} ({} s)", rungs[1], f * n(rungs[1]) * n(rungs[2])),
        "a larger class displaces the first; a tie keeps the one already chosen"
    );
}

// ---- #4587 k4: kill tests for the class_of / emit_derived / count_class / context_met / emit_row /
// emit_sampling / sampling_check / emit_effects survivors ----

fn spec_of(kind: CellKind, id: &str) -> CellSpec {
    CellSpec {
        id: id.into(),
        host: "h".into(),
        command: "gen".into(),
        generates: true,
        model_sha256: None,
        model_file: None,
        args: Vec::new(),
        shape: None,
        thinking: None,
        rung: None,
        rung_tokens: None,
        kind,
    }
}

/// The one `CellRow` of a v2 receipt from `apr_sha` whose single row is `{row_body}`.
fn parsed_row(apr_sha: &str, row_body: &str) -> (Receipt, CellRow) {
    let text = format!(
        r#"{{"schema":"apr-model-ladder-receipt/v2","host":"h","version":"0.69.1","sha":"x","apr_sha":"{apr_sha}","inventory":[],"cells":[{{{row_body}}}],"rungs":[]}}"#
    );
    let r = receipts::parse("dir/h.json", &text).expect("parses");
    let c = r.cells[0].clone();
    (r, c)
}

fn lit_of(g: &Graph, s: &str, p: &str) -> Vec<String> {
    g.objects(s, &rel(p))
        .iter()
        .map(|t| t.as_literal().map(|(v, _)| v.to_string()).expect("literal"))
        .collect()
}

fn model_info(mem: Option<(u64, u64, u64)>) -> ModelInfo {
    ModelInfo {
        file: "m.gguf".into(),
        arch: None,
        quant: None,
        context_length: None,
        thinking_modes: None,
        thinking_markers: None,
        owes_long_echo: None,
        owes_long: false,
        mem,
        kv_dtype: None,
    }
}

#[test]
fn class_of_grades_a_non_generating_matrix_cell_as_a_model_cell_and_a_misfit_as_a_refusal() {
    let mut s = spec_of(CellKind::Matrix, "a");
    s.generates = false;
    assert_eq!(class_of(&s, None, None), "ModelCell");
    s.generates = true;
    assert_eq!(class_of(&s, None, None), "Cell");
    s.rung_tokens = Some(1000);
    let m = model_info(Some((100, 1, 0)));
    assert_eq!(class_of(&s, Some(&m), Some(500)), "RefusalCell");
    assert_eq!(class_of(&s, Some(&m), Some(5000)), "Cell");
    assert_eq!(
        class_of(&spec_of(CellKind::Probe, "p"), None, None),
        "ProbeCell"
    );
    assert_eq!(
        class_of(&spec_of(CellKind::Base, "b"), None, None),
        "EffectCell"
    );
}

#[test]
fn count_class_counts_each_class_into_its_own_counter_and_every_cell_into_cells() {
    let mut st = ReleaseStats::default();
    count_class(&mut st, "RefusalCell");
    assert_eq!(
        (st.cells, st.refusal_cells, st.probe_cells, st.effect_cells),
        (1, 1, 0, 0)
    );
    count_class(&mut st, "ProbeCell");
    count_class(&mut st, "ProbeCell");
    assert_eq!(
        (st.cells, st.refusal_cells, st.probe_cells, st.effect_cells),
        (3, 1, 2, 0)
    );
    count_class(&mut st, "EffectCell");
    count_class(&mut st, "EffectCell");
    count_class(&mut st, "EffectCell");
    count_class(&mut st, "Cell");
    assert_eq!(
        (st.cells, st.refusal_cells, st.probe_cells, st.effect_cells),
        (7, 1, 2, 3)
    );
}

#[test]
fn emit_derived_owes_a_rung_only_on_the_hosts_where_it_fits() {
    let sha = SHA_A;
    let decl_small = HostDecl {
        id: "small".into(),
        cc: "sm_0".into(),
    };
    let decl_big = HostDecl {
        id: "big".into(),
        cc: "sm_0".into(),
    };
    let s = subject();
    let view = |d: &'static HostDecl, gpu: u64| HostView {
        decl: d,
        node: iri_path("release-host", &["0.69.1", &d.id]),
        receipts: Vec::new(),
        kernels: Vec::new(),
        models: BTreeMap::from([(sha.to_string(), model_info(Some((100, 1, 0))))]),
        gpu_mem: Some(gpu),
    };
    let decl_small: &'static HostDecl = Box::leak(Box::new(decl_small));
    let decl_big: &'static HostDecl = Box::leak(Box::new(decl_big));
    let views = [view(decl_small, 500), view(decl_big, 5000)];
    let cell = |host: &str| {
        let mut c = spec_of(CellKind::Matrix, &format!("gen/{host}/m.gguf/r"));
        c.host = host.into();
        c.model_sha256 = Some(sha.into());
        c.model_file = Some("m.gguf".into());
        c.rung = Some("r".into());
        c.rung_tokens = Some(1000);
        c
    };
    let cells = [cell("small"), cell("big")];
    let mut g = Graph::new();
    let mut st = ReleaseStats::default();
    emit_derived(&mut g, &s, &views, &cells, &mut st);
    assert_eq!((st.cells, st.refusal_cells), (2, 1));
    let cov = iri_path("release-coverage", &["0.69.1", "m.gguf", "r"]);
    let owed: Vec<String> = g
        .objects(&cov, &rel("owedOn"))
        .iter()
        .map(|t| t.as_iri().expect("iri").to_string())
        .collect();
    assert_eq!(owed, vec![iri_path("release-host", &["0.69.1", "big"])]);
}

#[test]
fn context_met_is_the_prompt_for_a_fixed_rung_and_prompt_plus_budget_for_declared() {
    let row = |p: Option<u64>, o: Option<u64>| {
        let mut b = String::from(r#""verdict":"pass","backend":"cuda""#);
        if let Some(p) = p {
            b.push_str(&format!(r#","prompt_tokens":{p}"#));
        }
        if let Some(o) = o {
            b.push_str(&format!(r#","max_tokens":{o}"#));
        }
        parsed_row(MC, &b).1
    };
    let mut fixed = spec_of(CellKind::Matrix, "a");
    fixed.rung = Some("r".into());
    fixed.rung_tokens = Some(100);
    assert!(
        context_met(&row(Some(100), None), &fixed),
        "p == t fills a fixed rung"
    );
    assert!(context_met(&row(Some(150), Some(0)), &fixed));
    assert!(
        !context_met(&row(Some(99), Some(500)), &fixed),
        "the budget is not counted"
    );
    assert!(
        !context_met(&row(None, Some(500)), &fixed),
        "an unknown prompt met nothing"
    );
    let mut declared = fixed.clone();
    declared.rung = Some(DECLARED.into());
    assert!(context_met(&row(Some(60), Some(40)), &declared));
    assert!(!context_met(&row(Some(60), Some(39)), &declared));
    assert!(
        !context_met(&row(Some(200), None), &declared),
        "no budget → unknown"
    );
    assert!(!context_met(&row(None, Some(200)), &declared));
    let mut none = spec_of(CellKind::Probe, "p");
    assert!(
        context_met(&row(None, None), &none),
        "no rung: nothing to fill"
    );
    none.rung = Some("r".into());
    assert!(
        !context_met(&row(Some(5), None), &none),
        "a rung with unknown size met nothing"
    );
}

#[test]
fn emit_row_grades_think_answered_and_f2_from_the_row() {
    let s = subject();
    let emit = |apr: &str, body: &str, thinking: Option<&str>| {
        let (r, c) = parsed_row(apr, &format!(r#""verdict":"Pass","backend":"cuda",{body}"#));
        let mut sp = spec_of(CellKind::Matrix, "a");
        sp.thinking = thinking.map(str::to_string);
        let mut g = Graph::new();
        let n = emit_row(&mut g, &s, &r, 0, &c, &sp, None);
        let get = |p: &str| lit_of(&g, &n, p);
        (
            get("thinkOk"),
            get("answered"),
            get("f2Measured"),
            get("verdict"),
        )
    };
    let t = |b: bool| vec![b.to_string()];
    let (think, ans, f2, verdict) = emit(MC, r#""think_closed":true,"answer_chars":3"#, Some("on"));
    assert_eq!((think, ans, f2), (t(true), t(true), t(false)));
    assert_eq!(verdict, vec!["pass".to_string()]);
    let (think, ans, _, _) = emit(MC, r#""answer_chars":0"#, Some("on"));
    assert_eq!(
        (think, ans),
        (t(false), t(false)),
        "on without a closed block, empty answer"
    );
    let (think, _, _, _) = emit(MC, r#""think_closed":false"#, Some("on"));
    assert_eq!(think, t(false));
    let (think, _, _, _) = emit(MC, r#""answer_chars":1"#, Some("off"));
    assert_eq!(think, t(true), "thinking off owes no closed block");
    let (think, _, _, _) = emit(MC, r#""answer_chars":1"#, None);
    assert_eq!(think, t(true));
    let (_, _, f2, _) = emit(MC, r#""f2_source":"fresh""#, None);
    assert_eq!(f2, t(true));
    let (_, _, f2, _) = emit(
        MC,
        &format!(r#""f2_source":"receipt","f2_receipt_binary_sha":"{MC}""#),
        None,
    );
    assert_eq!(f2, t(true), "a receipt of THIS binary");
    let (_, _, f2, _) = emit(
        MC,
        &format!(r#""f2_source":"receipt","f2_receipt_binary_sha":"{BUMP}""#),
        None,
    );
    assert_eq!(f2, t(false), "a receipt of another binary");
    let (_, _, f2, _) = emit(MC, r#""f2_source":"receipt""#, None);
    assert_eq!(f2, t(false), "a receipt naming no binary");
    let (_, _, f2, _) = emit(
        MC,
        &format!(r#""f2_source":"other","f2_receipt_binary_sha":"{MC}""#),
        None,
    );
    assert_eq!(f2, t(false));
}

#[test]
fn a_receipt_with_no_apr_sha_and_a_receipt_row_naming_none_is_not_an_f2_measurement() {
    let s = subject();
    let text = r#"{"schema":"apr-model-ladder-receipt/v2","host":"h","version":"0.69.1","sha":"x","inventory":[],"cells":[{"verdict":"pass","backend":"cuda","f2_source":"receipt"}],"rungs":[]}"#;
    let r = receipts::parse("dir/h.json", text).expect("parses");
    let mut g = Graph::new();
    let n = emit_row(
        &mut g,
        &s,
        &r,
        0,
        &r.cells[0],
        &spec_of(CellKind::Matrix, "a"),
        None,
    );
    assert_eq!(
        lit_of(&g, &n, "f2Measured"),
        vec!["false".to_string()],
        "None == None is no proof"
    );
}

#[test]
fn emit_row_detail_carries_what_the_row_measured() {
    let (r, c) = parsed_row(
        MC,
        r#""verdict":"pass","backend":"cuda","prompt_tokens":7,"max_tokens":9,"ttft_ms":1.5,"wall_ms":42,"rc":-3,"reason":"why","output_sha256":"ab""#,
    );
    let mut g = Graph::new();
    emit_row_detail(&mut g, "urn:n", &r, &c);
    let get = |p: &str| lit_of(&g, "urn:n", p);
    assert_eq!(get("promptTokens"), vec!["7".to_string()]);
    assert_eq!(get("maxTokens"), vec!["9".to_string()]);
    assert_eq!(get("wallMs"), vec!["42".to_string()]);
    assert_eq!(get("rc"), vec!["-3".to_string()]);
    assert_eq!(get("reason"), vec!["why".to_string()]);
    assert_eq!(get("aprSha"), vec![MC.to_string()]);
    assert_eq!(get("receiptFile"), vec!["dir/h.json".to_string()]);
    assert_eq!(get("ttftMs").len(), 1);
}

fn sampling_cells(controls: &[&str]) -> Vec<CellSpec> {
    controls
        .iter()
        .map(|c| {
            let mut s = spec_of(
                CellKind::Sampling {
                    control: (*c).to_string(),
                },
                &format!("gen/h/m.gguf/{c}"),
            );
            s.model_file = Some("m.gguf".into());
            s
        })
        .collect()
}

fn sampling_lits(cells: &[CellSpec], out: &[(&str, &str)], p: &str) -> (Vec<String>, ReleaseStats) {
    let mut outputs: BTreeMap<(&str, &str), Option<String>> = BTreeMap::new();
    for (c, sha) in out {
        let spec = cells
            .iter()
            .find(|s| matches!(&s.kind, CellKind::Sampling { control } if control == c))
            .expect("spec");
        outputs.insert(("h", spec.id.as_str()), Some((*sha).to_string()));
    }
    let mut g = Graph::new();
    let mut st = ReleaseStats::default();
    emit_sampling(&mut g, &subject(), cells, &outputs, &mut st);
    let n = iri_path("release-sampling", &["0.69.1", "h", "gen", "m.gguf"]);
    (lit_of(&g, &n, p), st)
}

fn all<'a>(
    t0: &'a str,
    tk: &'a str,
    a: &'a str,
    a2: &'a str,
    b: &'a str,
) -> [(&'static str, &'a str); 5] {
    [
        ("t0", t0),
        ("topk1", tk),
        ("seed-a", a),
        ("seed-a-again", a2),
        ("seed-b", b),
    ]
}

#[test]
fn emit_sampling_judges_each_control_from_the_output_digests() {
    let cells = sampling_cells(&["t0", "topk1", "seed-a", "seed-a-again", "seed-b"]);
    let t = |b: bool| vec![b.to_string()];
    let ps = [
        "sampledDiffers",
        "seedRepeatable",
        "seedsDiffer",
        "greedyAgrees",
    ];
    let run = |o: [(&str, &str); 5]| -> Vec<Vec<String>> {
        ps.iter().map(|p| sampling_lits(&cells, &o, p).0).collect()
    };
    assert_eq!(
        run(all("z", "z", "a", "a", "b")),
        vec![t(true), t(true), t(true), t(true)]
    );
    // seed-a differs from greedy, seed-b equals it
    assert_eq!(run(all("z", "y", "a", "c", "z"))[0], t(true));
    // seed-a equals greedy, seed-b differs
    assert_eq!(run(all("z", "y", "z", "c", "b"))[0], t(true));
    // both equal greedy: sampling changed nothing
    let none = run(all("z", "y", "z", "c", "z"));
    assert_eq!(none, vec![t(false), t(false), t(false), t(false)]);
    let (_, st) = sampling_lits(&cells, &all("z", "z", "a", "a", "b"), "seedsDiffer");
    assert_eq!(st.sampling_checks, 1);
    let (u, _) = sampling_lits(&cells, &all("z", "z", "a", "a", "b"), "underivable");
    assert!(
        u.is_empty(),
        "all five controls present: nothing underivable"
    );
}

#[test]
fn emit_sampling_names_a_check_whose_controls_the_command_cannot_express() {
    let cells = sampling_cells(&["t0", "seed-a"]);
    let (mut u, st) = sampling_lits(&cells, &[("t0", "z"), ("seed-a", "a")], "underivable");
    u.sort();
    assert_eq!(st.sampling_checks, 1);
    assert_eq!(
        u,
        vec![
            "greedyAgrees: no typed knob for topk1".to_string(),
            "sampledDiffers: no typed knob for seed-b".to_string(),
            "seedRepeatable: no typed knob for seed-a-again".to_string(),
            "seedsDiffer: no typed knob for seed-b".to_string(),
        ]
    );
    let (d, _) = sampling_lits(&cells, &[("t0", "z"), ("seed-a", "a")], "sampledDiffers");
    assert!(d.is_empty(), "an underivable check carries no verdict");
}

#[test]
fn sampling_check_needs_every_control_measured_and_applies_its_own_comparison() {
    let s = |x: &str| Some(x.to_string());
    assert!(!sampling_check("seedsDiffer", &[s("a"), None]));
    assert!(!sampling_check("seedRepeatable", &[None, s("a")]));
    assert!(sampling_check("sampledDiffers", &[s("z"), s("a"), s("z")]));
    assert!(sampling_check("sampledDiffers", &[s("z"), s("z"), s("a")]));
    assert!(!sampling_check("sampledDiffers", &[s("z"), s("z"), s("z")]));
    assert!(sampling_check("seedRepeatable", &[s("a"), s("a")]));
    assert!(!sampling_check("seedRepeatable", &[s("a"), s("b")]));
    assert!(sampling_check("greedyAgrees", &[s("a"), s("a")]));
    assert!(!sampling_check("greedyAgrees", &[s("a"), s("b")]));
    assert!(sampling_check("seedsDiffer", &[s("a"), s("b")]));
    assert!(!sampling_check("seedsDiffer", &[s("a"), s("a")]));
}

#[test]
fn emit_effects_observes_a_mode_only_where_the_effect_output_differs_from_its_base() {
    let eff = |id: &str, host: &str| {
        let mut c = spec_of(
            CellKind::Effect {
                arg: "--mode".into(),
                level: "on".into(),
                base: format!("base-{host}"),
            },
            id,
        );
        c.host = host.into();
        c
    };
    let cells = [eff("e1", "h1"), eff("e2", "h2")];
    let mut outputs: BTreeMap<(&str, &str), Option<String>> = BTreeMap::new();
    outputs.insert(("h1", "e1"), Some("x".into()));
    outputs.insert(("h1", "base-h1"), Some("y".into()));
    outputs.insert(("h2", "e2"), Some("x".into()));
    outputs.insert(("h2", "base-h2"), Some("x".into()));
    let mut g = Graph::new();
    let mut st = ReleaseStats::default();
    let s = subject();
    emit_effects(&mut g, &s, &cells, &outputs, &mut st);
    assert_eq!(
        st.mode_effects, 1,
        "one (command, arg=level) across both hosts"
    );
    let n = iri_path("release-effect", &["0.69.1", "gen", "--mode=on"]);
    assert_eq!(lit_of(&g, &n, "setting"), vec!["--mode=on".to_string()]);
    let seen: Vec<String> = g
        .objects(&n, &rel("observedIn"))
        .iter()
        .map(|t| t.as_iri().expect("iri").to_string())
        .collect();
    assert_eq!(
        seen,
        vec![cell_iri(&s, &cells[0])],
        "only h1's output differs from its base"
    );
}

/// KTEST-08: `--v2-evidence` adds the #3715 v2 kernel cells to the same graph, over the same hosts and
/// models; without it the graph has none.
#[test]
fn v2_cells_join_the_release_graph_only_when_asked() {
    let (t, c) = repo(&format!(
        "    - {{id: a, sha256: {SHA_A}, arch: qwen2, gguf: a.gguf, backends: [cuda], required: true}}\n"
    ));
    let inv = format!(r#"{{"file":"a.gguf","sha256":"{SHA_A}","tensor_types":[12]}}"#);
    write_receipt(
        t.path(),
        "lambda",
        &receipt(
            "lambda",
            MC,
            &inv,
            &row(&gen_id("lambda", "a.gguf", "off", "golden")),
        ),
    );
    let reg = t.path().join(kernel_cells::REGISTRY_PATH);
    std::fs::create_dir_all(reg.parent().expect("registry dir")).expect("mkdir");
    std::fs::write(
        &reg,
        r#"{"kernels":[{"kernel_id":"cuda.gemv.q4_k","backend":"cuda","ggml_type":12,"layout":"row_major","arch":"any"}],"ops":[]}"#,
    )
    .expect("registry");
    let h = "e".repeat(64);
    let v2 = t.path().join("evidence/release-v2");
    std::fs::create_dir_all(v2.join("lambda/parity")).expect("parity dir");
    std::fs::write(
        v2.join("input-sets.json"),
        format!(
            r#"{{"schema":"{}","build_identity":"{MC}","reuse":{{"fresh":1,"total":1}},
               "input_sets":{{"cuda.gemv.q4_k":{{"receipt":"k.json","input_set_hash":"{h}","stale":[]}}}}}}"#,
            kernel_cells::INPUT_SETS_SCHEMA
        ),
    )
    .expect("input sets");
    std::fs::write(
        v2.join("lambda/parity/k.json"),
        format!(
            r#"{{"schema":"kernel-parity-receipt/v1","kernel_id":"cuda.gemv.q4_k","sm":"sm_89","input_set_hash":"{h}",
               "oracle_independent":true,"served":{{"max_abs_err":2.7e-6,"max_rel_err":3.2e-7}},"tolerance_rel":7e-7}}"#
        ),
    )
    .expect("parity receipt");
    std::fs::create_dir_all(v2.join("lambda/smoke")).expect("smoke dir");
    std::fs::write(
        v2.join("lambda/smoke/a.json"),
        format!(
            r#"{{"schema":"{}","host":"lambda","model_sha256":"{SHA_A}","apr_sha":"{MC}","verdict":"pass",
               "kernel_path":{{"source":"kreg","entries":[{{"op":"gemv","kernel_id":"cuda.gemv.q4_k","qtype":"q4_k",
               "layout":"row_major","arch":"sm_89","shape_class":"m1","precision":"f32"}}]}}}}"#,
            kernel_cells::SMOKE_SCHEMA
        ),
    )
    .expect("smoke receipt");
    std::fs::create_dir_all(v2.join("lambda/sanitizer")).expect("sanitizer dir");
    let row = |tool: &str, filter: &str| {
        format!(
            r#"{{"tool":"{tool}","verdict":"CLEAN","filter":"{filter}","covers":["cuda.gemv.q4_k"]}}"#
        )
    };
    std::fs::write(
        v2.join("lambda/sanitizer/r.json"),
        format!(
            r#"{{"schema":"{}","host":"lambda","utc":"2026-09-28T12:00:00Z",
               "kernel_path":{{"source":"kreg","entries":[{{"kernel_id":"cuda.gemv.q4_k"}}]}},"tools":[{},{},{},{}]}}"#,
            kernel_cells::SANITIZER_SCHEMA_V2,
            row("memcheck", "none"),
            row("racecheck", "regex=gemv"),
            row("initcheck", "none"),
            row("synccheck", "none"),
        ),
    )
    .expect("sanitizer run");

    let mut off = Graph::new();
    extract(&mut off, &c, &subject()).expect("extracts");
    assert!(
        !off.to_ntriples().contains(&rel("ModelCell")),
        "v2 is opt-in: no dir, no v2 cells"
    );

    let mut s = subject();
    s.v2_dir = Some(v2.clone());
    let mut g = Graph::new();
    extract(&mut g, &c, &s).expect("extracts with v2");
    let lambda_kc = kernel_cells::kernel_cell("lambda", "cuda.gemv.q4_k");
    let uses = g.objects(
        &kernel_cells::model_cell("lambda", SHA_A),
        &rel("usesKernel"),
    );
    assert_eq!(
        uses,
        vec![&Term::iri(lambda_kc.clone())],
        "types from the inventory"
    );
    assert_eq!(
        g.objects(&lambda_kc, &rel("fresh")),
        vec![&Term::boolean(true)],
        "judged against input-sets.json at the release commit"
    );
    assert_eq!(
        g.objects(&lambda_kc, &rel("archMatch")),
        vec![&Term::boolean(true)]
    );
    let smoke = kernel_cells::smoke_cell("lambda", SHA_A);
    assert_eq!(
        g.objects(&smoke, &rel("kernelPathKnown")),
        vec![&Term::boolean(true)],
        "the smoke read from <dir>/lambda/smoke"
    );
    assert!(
        g.objects(
            &kernel_cells::model_cell("lambda", SHA_A),
            &rel("unpredictedKernel")
        )
        .is_empty(),
        "it dispatched only the predicted kernel"
    );
    assert_eq!(
        g.objects(&lambda_kc, &rel("sanitizerClean")),
        vec![&Term::boolean(true)],
        "the run read from <dir>/lambda/sanitizer, attributed by its kernel_path"
    );
    assert_eq!(
        g.objects(&lambda_kc, &rel("sanitizerFresh")),
        vec![&Term::boolean(false)],
        "no --gate-utc: the extractor reads no clock, so the run is stale"
    );
    let mut dated = s.clone();
    dated.v2_gate_utc = Some("2026-09-29T00:00:00Z".to_string());
    let mut gd = Graph::new();
    extract(&mut gd, &c, &dated).expect("extracts with a gate time");
    assert_eq!(
        gd.objects(&lambda_kc, &rel("sanitizerFresh")),
        vec![&Term::boolean(true)]
    );
    dated.v2_gate_utc = Some("yesterday".to_string());
    let err = extract(&mut Graph::new(), &c, &dated).expect_err("a malformed gate time");
    assert!(
        matches!(&err, ReleaseError::Input { file, .. } if file == "--gate-utc"),
        "{err:?}"
    );
    let gx10_kc = kernel_cells::kernel_cell("gx10", "cuda.gemv.q4_k");
    assert!(
        g.objects(&gx10_kc, &rel("verdict")).is_empty(),
        "gx10 has no parity dir: its kernel cell has no evidence"
    );

    // The registry is required once v2 is asked for.
    std::fs::remove_file(&reg).expect("rm registry");
    let err = extract(&mut Graph::new(), &c, &s).expect_err("no registry");
    assert!(
        matches!(&err, ReleaseError::Input { file, .. } if file == kernel_cells::REGISTRY_PATH),
        "{err:?}"
    );
}
