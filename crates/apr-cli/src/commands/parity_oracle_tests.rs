//! aprender#4444 — `apr parity-oracle` case table. The planted-divergence
//! fixture must turn the verdict RED; every refusal must refuse BEFORE a receipt
//! is written.

use super::*;
use std::path::PathBuf;

const N_POS: usize = 64;
const N_VOCAB: usize = 32;
const BASIS: &str = "fixture: synthetic rows, no model";

/// Deterministic, non-degenerate logits (an LCG, so every row differs).
fn fixture() -> RawLogits {
    let mut state: u32 = 0x4444_1234;
    let logits = (0..N_POS * N_VOCAB)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (f64::from(state >> 8) / f64::from(1u32 << 24) * 8.0 - 4.0) as f32
        })
        .collect();
    RawLogits {
        n_vocab: N_VOCAB,
        token_ids: (0..N_POS as i32).map(|i| 1000 + i).collect(),
        logits,
    }
}

/// Reverse one row: same norm, same values, different function.
fn plant_divergence(raw: &mut RawLogits, pos: usize) {
    raw.logits[pos * N_VOCAB..(pos + 1) * N_VOCAB].reverse();
}

struct Case {
    dir: tempfile::TempDir,
}

impl Case {
    fn new(reference: &RawLogits, subject: &RawLogits) -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("ref.bin"), encode_raw_logits(reference))
            .expect("write ref");
        std::fs::write(dir.path().join("sub.bin"), encode_raw_logits(subject)).expect("write sub");
        Self { dir }
    }
    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }
    fn run(&self, threshold: f64, basis: &str) -> Result<()> {
        run(
            &self.path("ref.bin"),
            &self.path("sub.bin"),
            threshold,
            basis,
            DEFAULT_MIN_POSITIONS,
            &self.path("receipt.json"),
            true,
        )
    }
    fn receipt(&self) -> serde_json::Value {
        let bytes = std::fs::read(self.path("receipt.json")).expect("receipt written");
        serde_json::from_slice(&bytes).expect("receipt is JSON")
    }
}

fn exit(r: Result<()>) -> u8 {
    r.expect_err("must fail").exit_code_value()
}

#[test]
fn identical_logits_are_green_and_the_receipt_carries_the_admission_fields() {
    let f = fixture();
    let c = Case::new(&f, &f);
    c.run(0.999, BASIS).expect("identical logits must be GREEN");
    let r = c.receipt();
    assert_eq!(r["verdict"], "GREEN");
    assert_eq!(r["oracle"], ORACLE);
    assert_eq!(r["schema"], SCHEMA);
    assert_eq!(r["threshold_basis"], BASIS);
    assert_eq!(r["n_positions"], N_POS);
    assert_eq!(r["per_position"].as_array().expect("rows").len(), N_POS);
    assert!((r["cosine"].as_f64().expect("cosine") - 1.0).abs() < 1e-12);
    assert_eq!(r["argmax_mismatches"].as_array().expect("list").len(), 0);
    // Both inputs are hashed; the hash is of the file bytes.
    let ref_bytes = std::fs::read(c.path("ref.bin")).expect("ref");
    assert_eq!(r["reference"]["sha256"], sha256_hex(&ref_bytes));
}

#[test]
fn planted_divergence_at_one_position_turns_it_red_and_names_the_position() {
    let f = fixture();
    let mut s = f.clone();
    plant_divergence(&mut s, 17);
    let c = Case::new(&f, &s);
    assert_eq!(exit(c.run(0.99, BASIS)), 13, "RED is ParityFailed (13)");
    // A RED run still writes its receipt: the RED is the evidence.
    let r = c.receipt();
    assert_eq!(r["verdict"], "RED");
    assert_eq!(r["min_cosine_pos"], 17);
    assert_eq!(r["positions_below_threshold"], serde_json::json!([17]));
    assert!(r["cosine"].as_f64().expect("cosine") < 0.99);
}

#[test]
fn small_noise_stays_green_so_the_red_is_not_trivial() {
    let f = fixture();
    let mut s = f.clone();
    for (i, v) in s.logits.iter_mut().enumerate() {
        *v += if i % 2 == 0 { 1e-3 } else { -1e-3 };
    }
    let c = Case::new(&f, &s);
    c.run(0.9999, BASIS)
        .expect("1e-3 noise on |logit|~2 is far above 0.9999");
}

#[test]
fn threshold_equal_to_the_min_cosine_is_green_the_gate_is_inclusive() {
    let f = fixture();
    let mut s = f.clone();
    plant_divergence(&mut s, 3);
    let min = judge(&f, &s, 0.5, BASIS, (rec(), rec())).cosine;
    let at = judge(&f, &s, min, BASIS, (rec(), rec()));
    assert_eq!(at.verdict, "GREEN");
    let above = judge(&f, &s, min + 1e-9, BASIS, (rec(), rec()));
    assert_eq!(above.verdict, "RED");
}

fn rec() -> InputRecord {
    InputRecord {
        path: "x".into(),
        sha256: "0".into(),
    }
}

#[test]
fn a_zero_norm_row_is_red_never_a_silent_nan_pass() {
    let f = fixture();
    let mut s = f.clone();
    s.logits[5 * N_VOCAB..6 * N_VOCAB].fill(0.0);
    let r = judge(&f, &s, 0.0, BASIS, (rec(), rec()));
    assert_eq!(r.verdict, "RED");
    assert_eq!(r.min_cosine_pos, 5);
    assert_eq!(r.positions_below_threshold, vec![5]);
}

#[test]
fn token_id_mismatch_is_refused_before_any_receipt() {
    let f = fixture();
    let mut s = f.clone();
    s.token_ids[9] += 1;
    let c = Case::new(&f, &s);
    let err = c
        .run(0.99, BASIS)
        .expect_err("different input measures nothing");
    assert_eq!(err.exit_code_value(), 4);
    assert!(err.to_string().contains("position 9"), "{err}");
    assert!(!c.path("receipt.json").exists());
}

#[test]
fn shape_mismatch_and_non_finite_logits_are_refused() {
    let f = fixture();
    let mut short = f.clone();
    short.n_vocab = N_VOCAB / 2;
    short.logits.truncate(N_POS * N_VOCAB / 2);
    assert_eq!(exit(Case::new(&f, &short).run(0.99, BASIS)), 4);

    let mut nan = f.clone();
    nan.logits[7] = f32::NAN;
    assert_eq!(exit(Case::new(&f, &nan).run(0.99, BASIS)), 4);
}

#[test]
fn too_few_positions_is_refused() {
    let mut f = fixture();
    f.token_ids.truncate(8);
    f.logits.truncate(8 * N_VOCAB);
    let c = Case::new(&f, &f);
    assert_eq!(exit(c.run(0.99, BASIS)), 5);
    assert!(!c.path("receipt.json").exists());
}

#[test]
fn a_threshold_without_a_basis_or_outside_the_cosine_domain_is_refused() {
    let f = fixture();
    let c = Case::new(&f, &f);
    assert_eq!(exit(c.run(0.99, "  ")), 5);
    assert!(
        c.run(f64::NAN, BASIS).is_err(),
        "NaN makes every comparison false"
    );
    assert!(c.run(1.5, BASIS).is_err());
    assert!(!c.path("receipt.json").exists());
}

#[test]
fn aprrawlg_round_trips_and_malformed_files_are_refused() {
    let f = fixture();
    let bytes = encode_raw_logits(&f);
    assert_eq!(parse_raw_logits(&bytes, "t").expect("parse"), f);
    assert!(
        parse_raw_logits(&bytes[..bytes.len() - 1], "t").is_err(),
        "truncated"
    );
    let mut magic = bytes.clone();
    magic[0] = b'X';
    assert!(parse_raw_logits(&magic, "t").is_err(), "bad magic");
    let mut version = bytes;
    version[8] = 2;
    assert!(parse_raw_logits(&version, "t").is_err(), "version 2");
    assert!(parse_raw_logits(b"APRRAWLG", "t").is_err(), "short header");
}

/// The oracle string a cell declares must name the comparator pin of record.
#[test]
fn oracle_commit_is_the_llama_pin_build_commit() {
    let pin = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/llama_pin.toml");
    let text = std::fs::read_to_string(&pin).expect("scripts/llama_pin.toml");
    let commit = text
        .lines()
        .find_map(|l| l.strip_prefix("build_commit = \""))
        .and_then(|l| l.strip_suffix('"'))
        .expect("build_commit line");
    assert_eq!(ORACLE, format!("llama.cpp@{commit}"));
}
