#![allow(clippy::disallowed_methods)] // serde_json::json! expands to unwrap
use super::*;
use serde_json::json;

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, day).expect("date")
}

// §2.4: three gated bands every night, in priority order, then exactly one rotating band.
#[test]
fn every_night_runs_the_three_gated_bands_first_then_one_rotating() {
    for night in [d(2026, 10, 1), d(2026, 10, 2)] {
        let b = bands_for_night(night);
        assert_eq!(b.len(), 4);
        let spec: [(u32, u32, &str); 3] = [(1, 4096, "W1"), (16, 4096, "W1"), (1, 32_768, "W5")];
        for (band, (c, ctx, w)) in b.iter().zip(spec) {
            assert!(band.gated);
            assert_eq!((band.c, band.ctx, band.workload.id()), (c, ctx, w));
        }
        assert!(!b[3].gated);
    }
}

// §2.4: the rotating band alternates every night, including across month and year ends.
#[test]
fn rotating_band_alternates_across_month_and_year_ends() {
    let nights = [
        d(2026, 10, 30),
        d(2026, 10, 31),
        d(2026, 11, 1),
        d(2026, 12, 31),
        d(2027, 1, 1),
    ];
    let cs: Vec<u32> = nights.iter().map(|n| rotating(*n).c).collect();
    for w in cs.windows(2).take(2) {
        assert_ne!(w[0], w[1], "{cs:?}");
    }
    assert_ne!(cs[3], cs[4], "year end repeats: {cs:?}");
    assert!(cs.iter().all(|c| *c == 4 || *c == 8));
}

// PP-24: a band above either server's ceiling is NA, never silently run lower.
#[test]
fn a_band_above_either_servers_slots_is_refused_with_the_budget() {
    let b16 = GATED[1];
    assert_eq!(ladder(&b16, 16, 16), Ok(()));
    assert_eq!(
        ladder(&b16, 8, 32),
        Err(BandRefusal::AboveCeiling {
            c: 16,
            slots_apr: 8,
            slots_llama: 32
        })
    );
    assert!(ladder(&b16, 32, 15).is_err());
}

// §2.4: -np c, -c c·n_ctx_slot; the slot must hold the longest prompt plus 128 tokens.
#[test]
fn llama_server_args_multiply_the_slot_and_refuse_a_short_slot() {
    let b16 = GATED[1];
    assert_eq!(
        llama_server_args(&b16, 648).expect("args"),
        ["-np", "16", "-c", "10368"]
    );
    // 640 meets the §2.4 floor but truncates a 520-token W1 prompt.
    assert_eq!(
        llama_server_args(&b16, 640),
        Err(BandRefusal::SlotTooSmall {
            n_ctx_slot: 640,
            needed: 648
        })
    );
    let long = GATED[2];
    assert_eq!(needed_ctx_slot(&long), 30_720 + 128);
    assert!(llama_server_args(&long, 32_768).is_ok());
}

// §2.4: W5 is W1 in file order, cycled, truncated; deterministic; empty W1 is refused.
#[test]
fn w5_is_w1_concatenated_cycled_and_truncated() {
    let w1 = vec![vec![1, 2, 3], vec![4, 5]];
    let w5 = w5_prompt(&w1).expect("w5");
    assert_eq!(w5.len(), W5_PROMPT_TOKENS);
    assert_eq!(&w5[..7], &[1, 2, 3, 4, 5, 1, 2]);
    assert_eq!(w5_prompt(&w1), Some(w5));
    assert_eq!(w5_prompt(&[]), None);
    assert_eq!(w5_prompt(&[vec![], vec![]]), None);
}

// PP-28: any retained sample short of 128 tokens makes the band fatal, and is named.
#[test]
fn a_short_completion_is_named() {
    assert!(short_completions(&[128, 128, 128]).is_empty());
    assert_eq!(short_completions(&[128, 127, 128, 129]), vec![1, 3]);
}

// E6 L6: a serial server has c·η(c) = 1 for every c.
#[test]
fn a_serial_server_has_c_eta_one() {
    for c in [1u32, 4, 8, 16] {
        let e = eta(37.0, 37.0, c).expect("eta");
        assert!((f64::from(c) * e - 1.0).abs() < 1e-12);
    }
    // G12 [C]: c·η flat at ≈ 1.2 is the serial-admission signature, not scaling.
    let e16 = eta(0.075 * 16.0 * 10.0, 10.0, 16).expect("eta");
    assert!((e16 - 0.075).abs() < 1e-12);
    assert_eq!(eta(1.0, 0.0, 4), None);
    assert_eq!(eta(1.0, 1.0, 0), None);
    assert_eq!(eta(f64::NAN, 1.0, 4), None);
    assert_eq!(varsigma(2.0, 0.5), Some(4.0));
    assert_eq!(varsigma(-1.0, 0.5), None);
}

fn row(tier: &str, c: u64, l: f64) -> Value {
    json!({
        "tier": tier, "band": {"c": c, "ctx": 4096, "workload_id": "W1"},
        "host": "gx10", "backend": "cuda", "binary_sha256": "b71", "model_sha256": "m1",
        "epoch_id": "e1", "ts": "2026-10-01T03:12:00Z", "L_n": l,
    })
}

// E5: ln ω = L_serve − L_engine for a matched pair; the identity ln ρ_s = ln ρ_e + ln ω holds.
#[test]
fn ln_omega_is_the_difference_of_matched_rows() {
    let (ls, le) = (-1.4_f64, -0.2_f64);
    let w = ln_omega(&row("serve", 1, ls), &row("engine", 1, le)).expect("omega");
    assert!((w - (ls - le)).abs() < 1e-12);
    assert!((le + w - ls).abs() < 1e-12);
}

// E5 / §1 join prohibition: any identity, night or tier mismatch is refused, never computed.
#[test]
fn ln_omega_refuses_every_unmatched_pair() {
    let s = row("serve", 1, -1.0);
    let e = row("engine", 1, -0.1);
    for f in OMEGA_IDENTITY {
        let mut e2 = e.clone();
        e2[f] = json!("other");
        assert_eq!(
            ln_omega(&s, &e2),
            Err(OmegaRefusal::Identity(vec![f])),
            "{f}"
        );
        let mut s2 = s.clone();
        s2.as_object_mut().expect("obj").remove(f);
        let mut e3 = e.clone();
        e3.as_object_mut().expect("obj").remove(f);
        assert_eq!(
            ln_omega(&s2, &e3),
            Err(OmegaRefusal::Identity(vec![f])),
            "absent {f}"
        );
    }
    let mut e4 = e.clone();
    e4["ts"] = json!("2026-10-02T03:12:00Z");
    assert_eq!(ln_omega(&s, &e4), Err(OmegaRefusal::DifferentNight));
    assert!(matches!(
        ln_omega(&row("serve", 16, -1.0), &e),
        Err(OmegaRefusal::WrongRow(_))
    ));
    assert!(matches!(ln_omega(&e, &e), Err(OmegaRefusal::WrongRow(_))));
    assert!(matches!(ln_omega(&s, &s), Err(OmegaRefusal::WrongRow(_))));
    let mut e5 = e.clone();
    e5["L_n"] = Value::Null;
    assert_eq!(ln_omega(&s, &e5), Err(OmegaRefusal::NoStatistic("engine")));
}

// §2.8: lease = 1.25 × measured p95, and nothing before a p95 exists.
#[test]
fn lease_is_one_and_a_quarter_measured_p95_and_absent_before_it() {
    assert_eq!(PLANNED_BLOCK_S, 340);
    assert_eq!(lease_len_s(Some(340.0)), Some(425));
    assert_eq!(lease_len_s(Some(341.0)), Some(427));
    assert_eq!(lease_len_s(None), None);
    assert_eq!(lease_len_s(Some(0.0)), None);
}

// §2.8: fewer than 3 valid blocks writes a skip row, never a gap; 3 writes none.
#[test]
fn a_band_short_of_three_blocks_writes_a_skip_row() {
    let r = skip_row(&GATED[1], "train-active", "aprender-36", 2).expect("skip");
    assert_eq!(r["state"], "skipped");
    assert_eq!(r["blocks_valid"], 2);
    assert_eq!(
        r["band"],
        json!({"c": 16, "ctx": 4096, "workload_id": "W1"})
    );
    assert_eq!(skip_row(&GATED[1], "x", "y", 3), None);
}
