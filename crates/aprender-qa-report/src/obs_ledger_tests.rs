use super::*;
use serde_json::json;

// The spec's §2.1 list, written out: iterating the module's own IDENTITY would let a
// field dropped from it drop out of this test too.
const SPEC_IDENTITY: [&str; 20] = [
    "schema",
    "ts",
    "host",
    "apr_version",
    "apr_tag",
    "crate_tarball_sha256",
    "binary_sha256",
    "build_identity",
    "comparator",
    "model_id",
    "model_sha256",
    "backend",
    "quiescence_proof",
    "tier",
    "band",
    "order",
    "epoch_id",
    "lease_id",
    "prev_row_sha256",
    "request_id",
];

fn quiet() -> Value {
    json!({"lease_id": "L-1", "foreign_gpu_procs": [], "load1_pre": 0.4,
           "cpu_governor": "performance", "gpu_sm_clock_mhz": [2520, 2520],
           "gpu_throttle_reasons": [[], []], "train_active": false})
}

// Shaped like an rc.1 engine-tier row on gx10 CUDA. No recorder exists yet (§1 G4), so
// this is a specimen of the §2.1 shape, not a measured night.
fn rc1(host: &str, backend: &str) -> Value {
    json!({
        "schema": "apr-perf-ledger-v1", "ts": "2026-10-01T03:12:00Z",
        "host": host, "apr_version": "0.71.0-rc.1", "apr_tag": "v0.71.0-rc.1",
        "crate_tarball_sha256": "c71", "binary_sha256": "b71",
        "build_identity": {"rustc": "rustc 1.93.0", "driver": "580.1", "features": ["cuda"]},
        "comparator": {"name": "llama-bench", "build_commit": "b100", "binary_sha256": "lb", "flags": {"p": 512, "n": 128}},
        "model_id": "qwen3.5-4b-q4_k_m", "model_sha256": "m1",
        "backend": backend,
        "gpu_proof": if backend == "cpu" { Value::Null } else { json!("trace: sm_121 q4k_gemv") },
        "quiescence_proof": quiet(),
        "kernel_path": null, "kernel_path_reason": "OBS-09 not merged",
        "tier": "engine", "band": {"c": 1, "ctx": 4096, "workload_id": "W1"},
        "order": {"design": "ABBA", "seed": 7, "blocks": ["ABBA", "BAAB", "ABBA"]},
        "epoch_id": "e1", "lease_id": "L-1",
        "prev_row_sha256": "0".repeat(64), "request_id": "0192-0000",
        "L_n": 0.02,
    })
}

fn line(v: &Value) -> String {
    canonical(v)
}

fn night() -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 10, 1).expect("date")
}

fn series(host: &str, backend: &str) -> Series {
    Series {
        host: host.into(),
        backend: backend.into(),
        tier: "engine".into(),
    }
}

// Positive: a well-formed rc.1 night over every declared series is admitted and GREEN.
#[test]
fn an_rc1_night_is_admitted_and_green() {
    let rows = [line(&rc1("gx10", "cuda")), line(&rc1("lambda", "cpu"))];
    let lines: Vec<&str> = rows.iter().map(String::as_str).collect();
    let v = judge_night(
        &lines,
        &[series("gx10", "cuda"), series("lambda", "cpu")],
        night(),
    );
    assert!(!v.red(), "{v:?}");
    assert!(v.refused.is_empty() && v.skipped.is_empty());
}

// FALSIFY-OBS-PERF-001: a declared series without a row is RED and named.
#[test]
fn a_declared_series_without_a_row_is_red_and_named() {
    let only = line(&rc1("gx10", "cuda"));
    let v = judge_night(
        &[only.as_str()],
        &[series("gx10", "cuda"), series("lambda", "cpu")],
        night(),
    );
    assert!(v.red());
    assert_eq!(v.missing, vec![series("lambda", "cpu")]);

    // A refused row is absent, not partial.
    let mut bad = rc1("lambda", "cpu");
    bad.as_object_mut().expect("obj").remove("binary_sha256");
    let bad = line(&bad);
    let v = judge_night(
        &[only.as_str(), bad.as_str()],
        &[series("lambda", "cpu")],
        night(),
    );
    assert_eq!(v.missing, vec![series("lambda", "cpu")]);
    assert_eq!(v.refused.len(), 1);

    // A row from another night does not cover tonight.
    let mut old = rc1("lambda", "cpu");
    old["ts"] = json!("2026-09-30T03:12:00Z");
    let old = line(&old);
    assert!(judge_night(&[old.as_str()], &[series("lambda", "cpu")], night()).red());

    // A skip is written, not a gap: reported, not missing.
    let skip = line(
        &json!({"state": "skipped", "ts": "2026-10-01T04:00:00Z", "host": "lambda",
                             "backend": "cpu", "tier": "engine", "reason": "train-active",
                             "lease_holder": "entrenar", "blocks_valid": 1}),
    );
    let v = judge_night(&[skip.as_str()], &[series("lambda", "cpu")], night());
    assert!(!v.red() && v.skipped == vec![series("lambda", "cpu")]);
}

// FALSIFY-OBS-PERF-002: an empty ledger is RED, never 0/0.
#[test]
fn an_empty_ledger_is_red() {
    assert!(judge_night(&[], &[], night()).red());
    assert!(judge_night(&["", "  "], &[], night()).red());
}

// FALSIFY-OBS-PERF-003 (and FALSIFY-OBS-ID-001): every identity field, version and sha included.
#[test]
fn a_row_missing_any_identity_field_is_refused_and_named() {
    for f in SPEC_IDENTITY {
        for bad in [
            None,
            Some(Value::Null),
            Some(json!("")),
            Some(json!("unknown")),
        ] {
            let mut r = rc1("gx10", "cuda");
            match &bad {
                None => {
                    r.as_object_mut().expect("obj").remove(f);
                }
                Some(b) => r[f] = b.clone(),
            }
            match admit_row(&r) {
                Err(Refusal::Inadmissible(m)) => assert!(m.iter().any(|x| x == f), "{f}: {m:?}"),
                other => panic!("{f} = {bad:?}: {other:?}"),
            }
        }
    }
    for key in ["gpu_proof", "kernel_path"] {
        let mut r = rc1("lambda", "cpu");
        r.as_object_mut().expect("obj").remove(key);
        assert_eq!(admit_row(&r), Err(Refusal::Inadmissible(vec![key.into()])));
    }
    let mut r = rc1("gx10", "cuda");
    r["schema"] = json!("apr-perf-backfill-v1");
    assert!(matches!(admit_row(&r), Err(Refusal::WrongSchema(_))));
    let mut r = rc1("gx10", "cuda");
    r.as_object_mut().expect("obj").remove("kernel_path_reason");
    assert_eq!(admit_row(&r), Err(Refusal::KernelPathUnexplained));
}

// FALSIFY-OBS-PERF-004 (and FALSIFY-OBS-ID-002): a GPU claim needs gpu_proof.
#[test]
fn a_gpu_row_without_gpu_proof_is_backend_unproven() {
    for b in ["cuda", "wgpu", "metal"] {
        let mut r = rc1("gx10", b);
        r["gpu_proof"] = Value::Null;
        assert_eq!(admit_row(&r), Err(Refusal::BackendUnproven), "{b}");
    }
    let mut cpu = rc1("lambda", "cpu");
    cpu["gpu_proof"] = json!("CUDA_VISIBLE_DEVICES=0");
    assert_eq!(admit_row(&cpu), Err(Refusal::CpuWithGpuProof));
    assert_eq!(admit_row(&rc1("gx10", "cuda")), Ok(()));
}

// FALSIFY-OBS-PERF-005 (and FALSIFY-OBS-ID-003): no ratio across mismatched identity.
#[test]
fn a_ratio_across_mismatched_identity_is_fatal() {
    let a = rc1("gx10", "cuda");
    for (f, v) in [
        ("host", json!("lambda")),
        ("backend", json!("wgpu")),
        ("tier", json!("serve")),
        ("band", json!({"c": 16, "ctx": 4096, "workload_id": "W1"})),
        ("model_sha256", json!("m2")),
    ] {
        let mut b = a.clone();
        b[f] = v;
        assert_eq!(comparable(&a, &b), Err(vec![f]), "{f}");
    }
    let mut b = a.clone();
    b["crate_tarball_sha256"] = json!("c72");
    b["binary_sha256"] = json!("b72");
    assert_eq!(
        comparable(&a, &b),
        Ok(vec!["crate_tarball_sha256", "binary_sha256"]),
        "a different apr build is the signal: annotated, not fatal"
    );
    assert_eq!(comparable(&a, &a), Ok(vec![]));
    assert!(
        comparable(&json!({}), &json!({})).is_err(),
        "absent identity is not equal identity"
    );
}

// FALSIFY-OBS-PERF-006: a blocked (non-ABBA) order is refused.
#[test]
fn a_blocked_order_row_is_refused() {
    for order in [
        json!({"design": "AABB", "seed": 7, "blocks": ["ABBA", "BAAB", "ABBA"]}),
        json!({"design": "ABBA", "seed": 7, "blocks": ["AABB", "BAAB", "ABBA"]}),
        json!({"design": "ABBA", "seed": 7, "blocks": ["ABBA", "BAAB"]}),
        json!({"design": "ABBA", "seed": 7, "blocks": ["ABBA", "BAAB", "ABBA", "BAAB"]}),
        json!({"design": "ABBA", "blocks": ["ABBA", "BAAB", "ABBA"]}),
        json!({"design": "ABBA", "seed": 7}),
    ] {
        let mut r = rc1("gx10", "cuda");
        r["order"] = order.clone();
        assert!(matches!(admit_row(&r), Err(Refusal::NotAbba(_))), "{order}");
    }
}

// FALSIFY-OBS-PERF-007: a row without a passing quiescence proof is host_unproven.
#[test]
fn a_row_without_a_passing_quiescence_proof_is_host_unproven() {
    let cases: [(&str, fn(&mut Value), &str); 6] = [
        ("no lease", |q| q["lease_id"] = Value::Null, "no_lease"),
        (
            "foreign proc",
            |q| q["foreign_gpu_procs"] = json!([{"pid": 9, "process_name": "python"}]),
            "foreign_gpu_procs",
        ),
        (
            "procs unread",
            |q| {
                q.as_object_mut().expect("obj").remove("foreign_gpu_procs");
            },
            "foreign_gpu_procs_unread",
        ),
        (
            "training",
            |q| q["train_active"] = json!(true),
            "train_active",
        ),
        (
            "training unread",
            |q| q["train_active"] = json!("no"),
            "train_active_unread",
        ),
        ("not an object", |q| *q = json!("ok"), "no_proof"),
    ];
    for (name, mutate, want) in cases {
        let mut r = rc1("gx10", "cuda");
        mutate(&mut r["quiescence_proof"]);
        match admit_row(&r) {
            Err(Refusal::HostUnproven(w)) => assert!(w.contains(&want), "{name}: {w:?}"),
            other => panic!("{name}: {other:?}"),
        }
    }
}

// FALSIFY-OBS-PERF-008 (E12): a ledger the subject can write is refused.
#[test]
fn a_ledger_writable_by_the_subject_is_refused() {
    let recorder = 1201;
    let apr = 1000;
    assert!(writer_check(recorder, 0o100640, &[apr]).is_ok());
    assert!(
        writer_check(apr, 0o100640, &[apr]).is_err(),
        "owned by the subject"
    );
    assert!(
        writer_check(recorder, 0o100660, &[apr]).is_err(),
        "group writable"
    );
    assert!(
        writer_check(recorder, 0o100642, &[apr]).is_err(),
        "other writable"
    );

    let d = tempfile::tempdir().expect("tmp");
    let p = d.path().join("perf.jsonl");
    std::fs::write(&p, "").expect("w");
    let me = std::os::unix::fs::MetadataExt::uid(&std::fs::metadata(&p).expect("m"));
    assert!(
        ledger_writer_check(&p, &[me]).is_err(),
        "a file this uid owns, with this uid as subject"
    );
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o640)).expect("chmod");
    assert!(ledger_writer_check(&p, &[me.wrapping_add(1)]).is_ok());
    assert!(ledger_writer_check(&d.path().join("absent"), &[]).is_err());
}

fn chained(n: usize) -> (String, Vec<String>) {
    let h0 = genesis("apr-perf-ledger-v1", "gx10", "cuda", "2026-10-01");
    let mut prev = h0.clone();
    let mut out = Vec::new();
    for i in 0..n {
        let mut r = rc1("gx10", "cuda");
        r["ts"] = json!(format!("2026-10-{:02}T03:12:00Z", i + 1));
        r["prev_row_sha256"] = json!(prev);
        prev = row_hash(&r);
        out.push(line(&r));
    }
    (h0, out)
}

// FALSIFY-OBS-PERF-009 (T5): an edit, deletion or reorder before the head breaks the chain.
#[test]
fn a_mid_file_edit_breaks_the_chain() {
    let (h0, rows) = chained(5);
    let lines: Vec<&str> = rows.iter().map(String::as_str).collect();
    let head = verify_chain(&lines, &h0, None).expect("clean chain");
    assert_eq!(head.n_rows, 5);

    let edited = rows[1].replace("\"L_n\":0.02", "\"L_n\":0.09");
    assert_ne!(edited, rows[1]);
    let mut l = lines.clone();
    l[1] = &edited;
    assert_eq!(verify_chain(&l, &h0, None), Err(ChainError::Break(2)));

    let mut l = lines.clone();
    l.remove(2);
    assert_eq!(verify_chain(&l, &h0, None), Err(ChainError::Break(2)));

    let mut l = lines.clone();
    l.swap(0, 1);
    assert_eq!(verify_chain(&l, &h0, None), Err(ChainError::Break(0)));

    let other = genesis("apr-perf-ledger-v1", "gx10", "wgpu", "2026-10-01");
    assert_eq!(
        verify_chain(&lines, &other, None),
        Err(ChainError::Break(0))
    );
    assert_eq!(
        verify_chain(&["{"], &h0, None),
        Err(ChainError::Unparsable(0))
    );
}

// FALSIFY-OBS-PERF-010 (T5): the chain cannot see tail truncation; the anchor can.
#[test]
fn tail_truncation_below_the_anchor_is_caught_only_by_the_anchor() {
    let (h0, rows) = chained(5);
    let lines: Vec<&str> = rows.iter().map(String::as_str).collect();
    let anchor = verify_chain(&lines, &h0, None).expect("anchor");

    assert!(
        verify_chain(&lines[..3], &h0, None).is_ok(),
        "the chain alone is blind to it"
    );
    assert_eq!(
        verify_chain(&lines[..3], &h0, Some(&anchor)),
        Err(ChainError::BelowAnchor {
            rows: 3,
            anchored: 5
        })
    );
    assert!(verify_chain(&lines, &h0, Some(&anchor)).is_ok());

    // Rows appended after the anchor are fine; the anchored prefix must still hash to h_n.
    let (_, longer) = chained(6);
    let longer: Vec<&str> = longer.iter().map(String::as_str).collect();
    assert!(verify_chain(&longer, &h0, Some(&anchor)).is_ok());
    let forged = Anchor {
        n_rows: 5,
        h_n: "f".repeat(64),
    };
    assert_eq!(
        verify_chain(&lines, &h0, Some(&forged)),
        Err(ChainError::AnchorMismatch)
    );
    let genesis_anchor = Anchor {
        n_rows: 0,
        h_n: h0.clone(),
    };
    assert!(verify_chain(&[], &h0, Some(&genesis_anchor)).is_ok());
}

fn block(start_s: u64, order: Order, ratio: f64) -> Block {
    // apr faster than the comparator by `ratio` in every pair.
    let (a, b) = (100.0 * ratio, 100.0);
    let x = match order {
        Order::Abba => [a, b, b, a],
        Order::Baab => [b, a, a, b],
    };
    Block {
        start_s,
        order,
        x,
        quiet_start: true,
        quiet_end: true,
    }
}

// T1: under a shared log-linear drift both orders return exactly α − β.
#[test]
fn abba_and_baab_cancel_linear_drift() {
    let (alpha, beta, s, t, w) = (5.0_f64, 4.8_f64, 0.013_f64, 10.0_f64, 85.0_f64);
    let xa = |u: f64| (alpha + s * u).exp();
    let xb = |u: f64| (beta + s * u).exp();
    let slots = [t, t + w, t + 2.0 * w, t + 3.0 * w];
    let abba = Block {
        start_s: 0,
        order: Order::Abba,
        x: [xa(slots[0]), xb(slots[1]), xb(slots[2]), xa(slots[3])],
        quiet_start: true,
        quiet_end: true,
    };
    let baab = Block {
        order: Order::Baab,
        x: [xb(slots[0]), xa(slots[1]), xa(slots[2]), xb(slots[3])],
        ..abba.clone()
    };
    for b in [abba, baab] {
        assert!(
            (block_stat(&b) - (alpha - beta)).abs() < 1e-9,
            "{:?}",
            b.order
        );
    }
}

// FALSIFY-OBS-PERF-011 (§2.8): a voided block never enters E2.
#[test]
fn a_voided_block_never_enters_e2() {
    let good = [
        block(0, Order::Abba, 1.02),
        block(400, Order::Baab, 1.03),
        block(800, Order::Abba, 1.04),
    ];
    let clean = nightly_stat(&good).expect("3 valid");
    assert!((clean - 1.03f64.ln()).abs() < 1e-12);

    for (start, end) in [(false, true), (true, false)] {
        // train-active toggled mid-block: a huge outlier that would move the median if counted.
        let voided = Block {
            quiet_start: start,
            quiet_end: end,
            ..block(200, Order::Baab, 9.0)
        };
        let mut blocks = good.to_vec();
        blocks.insert(1, voided);
        assert_eq!(nightly_stat(&blocks), Ok(clean), "start={start} end={end}");
    }
    let voided = Block {
        quiet_end: false,
        ..good[2].clone()
    };
    assert_eq!(
        nightly_stat(&[good[0].clone(), good[1].clone(), voided]),
        Err(NightError::Skipped { blocks_valid: 2 })
    );
    let mut bad = good.to_vec();
    bad[1].x[2] = 0.0;
    assert_eq!(nightly_stat(&bad), Err(NightError::BadSample(1)));
}

// Positive: a night assembled from 3 non-contiguous valid blocks, out of line order.
#[test]
fn a_night_from_three_non_contiguous_valid_blocks_is_measured() {
    let preempted = Block {
        quiet_start: false,
        ..block(1_000, Order::Abba, 0.5)
    };
    let blocks = [
        // A 4th valid block, after the band is complete, listed FIRST: selection is by
        // time, never by input order.
        block(12_000, Order::Abba, 0.10),
        block(9_000, Order::Baab, 1.10),
        block(0, Order::Abba, 1.01),
        preempted,
        block(4_200, Order::Abba, 1.05),
    ];
    let l = nightly_stat(&blocks).expect("3 valid blocks");
    assert!((l - 1.05f64.ln()).abs() < 1e-12, "{l}");
}

#[test]
fn oversize_or_unparsable_lines_are_malformed() {
    let mut r = rc1("gx10", "cuda");
    r["build_identity"]["uname"] = json!("x".repeat(MAX_LINE_BYTES));
    assert!(matches!(admit_line(&line(&r)), Err(Refusal::Malformed(_))));
    assert!(matches!(admit_line("not json"), Err(Refusal::Malformed(_))));
    assert!(matches!(admit_row(&json!([1])), Err(Refusal::Malformed(_))));
    assert!(admit_line(&line(&rc1("gx10", "cuda"))).is_ok());
}
