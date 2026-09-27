use super::*;
use crate::obs_ledger::admit_row;
use crate::obs_ledger::tests::rc1;
use serde_json::json;

// The spec's §2.10 entry fields and GPU backends, written out: iterating the module's own
// constants would let a dropped field drop out of the test too.
const SPEC_ENTRY: [&str; 7] = [
    "op",
    "kernel_id",
    "qtype",
    "layout",
    "arch",
    "shape_class",
    "precision",
];
const SPEC_GPU: [&str; 3] = ["cuda", "wgpu", "metal"];

fn d(s: &str) -> NaiveDate {
    NaiveDate::parse_from_str(s, "%Y-%m-%d").expect("date")
}

fn entry(op: &str, kernel_id: &str) -> Value {
    json!({"op": op, "kernel_id": kernel_id, "qtype": "q4_k", "layout": "row_major",
           "arch": "sm_121", "shape_class": "gemv_4096", "precision": "f32acc"})
}

fn gx10_row() -> Value {
    let mut r = rc1("gx10", "cuda");
    r["kernel_path"] = json!({"source": "trace", "entries": [
        entry("attn_qkv", "q4k_gemv_dp4a"),
        entry("ffn_gate_up", "fused_q4k_gate_up"),
        entry("lm_head", "q6k_gemv"),
    ]});
    r.as_object_mut().expect("obj").remove("kernel_path_reason");
    r
}

// FALSIFY-OBS-KP-001
#[test]
fn a_gpu_row_after_the_trace_cut_without_kernel_path_is_refused() {
    let cut = d("2026-10-01");
    for backend in SPEC_GPU {
        let mut r = rc1("gx10", backend);
        assert_eq!(
            check_kernel_path(&r, Some(cut)),
            Err(KernelPathError::NullAfterTrace),
            "{backend} on the cut night"
        );
        r["ts"] = json!("2026-10-05T03:12:00Z");
        assert_eq!(
            check_kernel_path(&r, Some(cut)),
            Err(KernelPathError::NullAfterTrace)
        );
        r.as_object_mut().expect("obj").remove("kernel_path");
        assert_eq!(
            check_kernel_path(&r, Some(cut)),
            Err(KernelPathError::NullAfterTrace)
        );
        r["ts"] = json!("2026-09-30T03:12:00Z");
        r["kernel_path"] = Value::Null;
        assert_eq!(
            check_kernel_path(&r, Some(cut)),
            Ok(()),
            "{backend} before the cut"
        );
        assert_eq!(
            check_kernel_path(&r, None),
            Ok(()),
            "{backend} before OBS-09 merges"
        );
        r["ts"] = json!("garbage");
        assert_eq!(
            check_kernel_path(&r, Some(cut)),
            Err(KernelPathError::NullAfterTrace)
        );
    }
    // 100% of GPU rows, not CPU rows: a cpu row names no GPU kernel.
    assert_eq!(check_kernel_path(&rc1("lambda", "cpu"), Some(cut)), Ok(()));
}

// FALSIFY-OBS-KP-002
#[test]
fn a_kreg_entry_without_a_host_receipt_of_its_arch_is_refused() {
    let e = json!({"arch": "aarch64", "isa_features": ["neon", "dotprod"],
                   "backend": "cpu", "kernel_id": "q4k_q8k_sdot"});
    let ok = json!({"host": "mini", "arch": "aarch64", "backend": "cpu",
                    "kernel_id": "q4k_q8k_sdot", "parity": "pass"});
    assert_eq!(admit_kreg_entry(&e, &[ok.clone()]), Ok(()));
    assert_eq!(admit_kreg_entry(&e, &[]), Err(vec!["no_host_receipt"]));
    for (field, value) in [
        ("arch", json!("x86_64")),
        ("backend", json!("wgpu")),
        ("kernel_id", json!("q4k_gemv")),
        ("parity", json!("fail")),
        ("host", json!("")),
    ] {
        let mut r = ok.clone();
        r[field] = value;
        assert_eq!(
            admit_kreg_entry(&e, &[r]),
            Err(vec!["no_host_receipt"]),
            "receipt with a different {field}"
        );
    }
    for field in ["arch", "isa_features", "backend", "kernel_id"] {
        let mut bad = e.clone();
        bad.as_object_mut().expect("obj").remove(field);
        assert_eq!(admit_kreg_entry(&bad, &[ok.clone()]), Err(vec![field]));
    }
}

// FALSIFY-OBS-KP-003
#[test]
fn a_malformed_kernel_path_is_refused_and_named() {
    let with = |p: Value| {
        let mut r = gx10_row();
        r["kernel_path"] = p;
        check_kernel_path(&r, None)
    };
    assert_eq!(with(json!("q4k")), Err(KernelPathError::NotAnObject));
    assert_eq!(
        with(json!({"source": "guess", "entries": [entry("a", "k")]})),
        Err(KernelPathError::BadSource("guess".into()))
    );
    assert_eq!(
        with(json!({"source": "kreg"})),
        Err(KernelPathError::NoEntries)
    );
    assert_eq!(
        with(json!({"source": "trace", "entries": []})),
        Err(KernelPathError::NoEntries)
    );
    for field in SPEC_ENTRY {
        for blank in [Value::Null, json!(""), json!("unknown")] {
            let mut e = entry("lm_head", "q6k_gemv");
            e[field] = blank;
            assert_eq!(
                with(json!({"source": "kreg", "entries": [entry("a", "k"), e]})),
                Err(KernelPathError::Entry {
                    index: 1,
                    missing: vec![field]
                }),
                "{field}"
            );
        }
    }
}

// FALSIFY-OBS-KP-004
#[test]
fn a_kernel_change_is_attributed_night_over_night() {
    let prev = gx10_row();
    assert_eq!(kernel_diff(&prev, &prev), Some(vec![]));

    let mut cur = gx10_row();
    cur["kernel_path"]["entries"][2]["kernel_id"] = json!("q6k_gemv_scalar_fallback");
    cur["kernel_path"]["entries"]
        .as_array_mut()
        .expect("entries")
        .push(entry("rope", "rope_f32"));
    let changes = kernel_diff(&prev, &cur).expect("both rows carry a path");
    let slots: Vec<&str> = changes.iter().map(|c| c.slot.0.as_str()).collect();
    assert_eq!(slots, ["lm_head", "rope"]);
    assert_eq!(
        changes[0].after.as_ref().map(|e| &e["kernel_id"]),
        Some(&json!("q6k_gemv_scalar_fallback"))
    );
    assert!(changes[1].before.is_none());

    // A row without a path is unattributable, never "no kernel changed".
    assert_eq!(kernel_diff(&rc1("gx10", "cuda"), &cur), None);
    assert_eq!(kernel_diff(&cur, &rc1("gx10", "cuda")), None);
}

// FALSIFY-OBS-KP-P01
#[test]
fn a_gx10_row_naming_its_kernels_is_admitted() {
    let r = gx10_row();
    assert_eq!(admit_row(&r), Ok(()));
    assert_eq!(check_kernel_path(&r, Some(d("2026-10-01"))), Ok(()));
    assert_eq!(check_kernel_path(&r, None), Ok(()));
}
