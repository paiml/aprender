// `json!` expands to an `unwrap` of an infallible `to_value`.
#![allow(clippy::disallowed_methods)]

use super::*;

use std::fs;
use std::path::PathBuf;

/// A throwaway traces root under the target dir's tmp; removed on drop.
struct Root(PathBuf);

impl Root {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("tdc-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).expect("mkdir root");
        Root(p)
    }
}

impl Drop for Root {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Synthetic rows only: no real transcript or trace ever enters a fixture.
const SECRET_TEXT: &str = "SENTINEL-FINDING-TEXT-never-in-a-card";
const SECRET_DIFF: &str = "0badc0de0badc0de0badc0de0badc0de0badc0de0badc0de0badc0de0badc0de";

fn row(id: &str, lane: &str, provider: &str, tier: &str, split: &str) -> String {
    serde_json::json!({
        "schema": "agent-trace-v1",
        "trace_id": id,
        "quorum_id": "q1",
        "round": 1,
        "lane": lane,
        "counted": provider != "local",
        "provider": provider,
        "model_id": format!("{lane}-2026-09-01"),
        "access_channel": if provider == "local" { "apr-serve" } else { "claude-code-cli" },
        "repo": "paiml/aprender",
        "pr": 4400,
        "head_sha": "a".repeat(40),
        "diff_sha256": SECRET_DIFF,
        "input_sha": "b".repeat(64),
        "output_sha": "c".repeat(64),
        "prompt_version": "1.0.0",
        "lane_state": "Verdict",
        "verdict": "approve",
        "findings": [{"file": "src/a.rs", "line_start": 1, "line_end": 2,
                      "severity": "low", "category": "style", "text": SECRET_TEXT}],
        "parse_status": "ok",
        "label_tier": tier,
        "outcome": "merged",
        "split_guard": {"group_key": "paiml/aprender#4400", "split": split},
        "secret_scan": {"status": "clean", "hits": 0},
        "trace_status": "ok",
        "agreed": true,
        "lane_disagreement": 0,
        "unique_diff": true,
        "dedup_cluster": "c0",
    })
    .to_string()
}

fn day(root: &Path, ymd: (&str, &str, &str), rows: &[String]) {
    let d = root.join(INDEX_DIR).join(ymd.0).join(ymd.1);
    fs::create_dir_all(&d).expect("mkdir day");
    let mut body = rows.join("\n");
    body.push('\n');
    fs::write(d.join(format!("{}.jsonl", ymd.2)), body).expect("write day");
}

fn fixture(tag: &str) -> Root {
    let r = Root::new(tag);
    day(
        &r.0,
        ("2026", "09", "24"),
        &[
            row("t1", "sonnet", "anthropic", "gold", "train"),
            row("t2", "agy", "google", "gold", "train"),
        ],
    );
    day(
        &r.0,
        ("2026", "09", "25"),
        &[
            row("t3", "qwen-shadow", "local", "gold", "val"),
            row("t4", "qwen-shadow", "local", "silver", "train"),
            row("t5", "haiku", "anthropic", "pending", "train"),
        ],
    );
    r
}

fn card_of(root: &Path) -> (Snapshot, Value) {
    let s = Snapshot::read(root).expect("snapshot");
    let c = croissant(&s, &CardMeta::default());
    (s, c)
}

#[test]
fn falsify_tdc_001_the_generated_card_passes_croissant_validation() {
    let r = fixture("v");
    let (_, c) = card_of(&r.0);
    assert_eq!(validate(&c), Vec::<String>::new(), "{c:#}");

    // The validator is not vacuous: each plant is refused, and names why.
    let plants: [(&str, fn(&mut Value)); 9] = [
        ("conformsTo", |c| {
            c.as_object_mut().expect("obj").remove("conformsTo");
        }),
        ("name", |c| c["name"] = json!("")),
        ("@type", |c| c["@type"] = json!("sc:CreativeWork")),
        ("sha256", |c| {
            c["distribution"][0]
                .as_object_mut()
                .expect("obj")
                .remove("sha256");
        }),
        ("unknown resource", |c| {
            c["recordSet"][0]["field"][0]["source"]["fileSet"]["@id"] = json!("nope")
        }),
        ("duplicate @id", |c| {
            let f = c["recordSet"][0]["field"][0].clone();
            c["recordSet"][0]["field"]
                .as_array_mut()
                .expect("arr")
                .push(f);
        }),
        ("dataType", |c| {
            c["recordSet"][0]["field"][0]["dataType"] = json!("sc:Banana")
        }),
        ("rai:dataLimitations", |c| {
            c.as_object_mut()
                .expect("obj")
                .remove("rai:dataLimitations");
        }),
        ("@context", |c| {
            c["@context"].as_object_mut().expect("obj").remove("rai");
        }),
    ];
    for (why, plant) in plants {
        let mut bad = c.clone();
        plant(&mut bad);
        let errs = validate(&bad);
        assert!(
            errs.iter().any(|e| e.contains(why)),
            "plant {why:?} not refused: {errs:?}"
        );
    }
}

#[test]
fn falsify_tdc_002_a_snapshot_is_deterministic_and_bound_to_the_index_bytes() {
    let r = fixture("det");
    let (s1, c1) = card_of(&r.0);
    let (s2, c2) = card_of(&r.0);
    assert_eq!(s1.id, s2.id);
    assert_eq!(render(&c1), render(&c2), "same index, same bytes");
    assert_eq!(
        datasheet(&s1, &CardMeta::default()),
        datasheet(&s2, &CardMeta::default())
    );

    // The root digest is over every index file's bytes.
    let manifest = s1.manifest();
    assert_eq!(manifest.lines().count(), 2, "{manifest}");
    for line in manifest.lines() {
        let (sha, rel) = line.split_once("  ").expect("sha  path");
        let bytes = fs::read(r.0.join(rel)).expect("read");
        assert_eq!(sha, sha256_hex(&bytes), "{rel}");
    }
    assert_eq!(
        c1["distribution"][0]["sha256"],
        json!(sha256_hex(manifest.as_bytes()))
    );

    // One appended row is a new snapshot.
    day(
        &r.0,
        ("2026", "09", "26"),
        &[row("t6", "sonnet", "anthropic", "silver", "test")],
    );
    let (s3, c3) = card_of(&r.0);
    assert_ne!(s1.id, s3.id);
    assert_ne!(
        c1["distribution"][0]["sha256"],
        c3["distribution"][0]["sha256"]
    );
    assert_eq!(s3.rows, 6);
}

#[test]
fn falsify_tdc_003_only_gold_local_rows_are_public_eligible() {
    let r = fixture("pub");
    let (s, c) = card_of(&r.0);
    // t1 gold anthropic, t2 gold google: out. t3 gold local: in. t4 silver local,
    // t5 pending anthropic: out.
    assert_eq!(s.public_eligible, 1);
    assert_eq!(s.count("provider", "anthropic"), 2);
    assert_eq!(s.count("provider", "google"), 1);
    assert_eq!(s.count("provider", "local"), 2);
    assert_eq!(s.count("label_tier", "gold"), 3);
    assert_eq!(s.count("split", "val"), 1);
    let sheet = datasheet(&s, &CardMeta::default());
    assert!(
        sheet.contains("public-eligible (gold, provider local): 1"),
        "{sheet}"
    );
    assert!(c["rai:personalSensitiveInformation"]
        .as_str()
        .expect("str")
        .contains("quarantine"));

    // A secret-hit gold local row is never eligible.
    let mut hit: Value =
        serde_json::from_str(&row("t7", "qwen-shadow", "local", "gold", "train")).expect("json");
    hit["secret_scan"] = json!({"status": "hit", "hits": 1});
    day(&r.0, ("2026", "09", "27"), &[hit.to_string()]);
    let s = Snapshot::read(&r.0).expect("snapshot");
    assert_eq!(s.public_eligible, 1);
}

#[test]
fn falsify_tdc_004_a_hand_edited_card_is_refused_and_regeneration_is_idempotent() {
    let r = fixture("edit");
    let out = write_snapshot(&r.0, &CardMeta::default()).expect("first write");
    assert!(out.join("croissant.json").is_file());
    assert!(out.join("datasheet.md").is_file());
    assert!(out.join("manifest.sha256").is_file());
    assert!(out.join("weekly.json").is_file());
    assert_eq!(out.parent().expect("parent"), r.0.join(DATACARD_DIR));

    // Regenerating the same snapshot is a no-op, not an error.
    assert_eq!(
        write_snapshot(&r.0, &CardMeta::default()).expect("again"),
        out
    );

    // A hand edit is refused on the next regeneration, never overwritten.
    let p = out.join("croissant.json");
    let edited = fs::read_to_string(&p)
        .expect("read")
        .replace("agent-trace", "agent-trice");
    fs::write(&p, &edited).expect("edit");
    let e = write_snapshot(&r.0, &CardMeta::default()).expect_err("hand edit refused");
    assert!(e.contains("hand-edited"), "{e}");
    assert_eq!(
        fs::read_to_string(&p).expect("read"),
        edited,
        "never overwritten"
    );
}

#[test]
fn falsify_tdc_005_no_row_content_reaches_the_card() {
    let r = fixture("priv");
    let (s, c) = card_of(&r.0);
    let text = format!("{}\n{}", render(&c), datasheet(&s, &CardMeta::default()));
    assert!(
        !text.contains(SECRET_TEXT),
        "a finding's text leaked into the card"
    );
    assert!(
        !text.contains(SECRET_DIFF),
        "a diff sha leaked into the card"
    );
    assert!(
        !text.contains(&"c".repeat(64)),
        "an output sha leaked into the card"
    );
}

#[test]
fn falsify_tdc_006_a_malformed_index_is_refused_not_skipped() {
    let r = fixture("bad");
    day(&r.0, ("2026", "09", "28"), &["{not json".to_string()]);
    let e = Snapshot::read(&r.0).expect_err("bad json refused");
    assert!(
        e.contains("2026/09/28.jsonl:1") && e.contains("bad json"),
        "{e}"
    );

    let r = fixture("schema");
    let foreign = row("t9", "sonnet", "anthropic", "gold", "train")
        .replace("agent-trace-v1", "agent-trace-v2");
    day(&r.0, ("2026", "09", "28"), &[foreign]);
    let e = Snapshot::read(&r.0).expect_err("unknown major refused");
    assert!(e.contains("agent-trace-v2"), "{e}");

    let empty = Root::new("empty");
    let e = Snapshot::read(&empty.0).expect_err("an empty index is not a dataset");
    assert!(e.contains("no index"), "{e}");
}

fn rowq(id: &str, quorum: &str) -> String {
    let mut v: Value =
        serde_json::from_str(&row(id, "sonnet", "anthropic", "silver", "train")).expect("json");
    v["quorum_id"] = json!(quorum);
    v.to_string()
}

/// A zstd frame header declaring `raw` bytes (single segment, 4-byte FCS).
fn blob(root: &Path, sha: &str, head: &[u8]) {
    let d = root.join("blobs/sha256").join(&sha[..2]).join(&sha[2..4]);
    fs::create_dir_all(&d).expect("mkdir blob");
    let mut b = head.to_vec();
    b.extend_from_slice(&[0u8; 10]);
    fs::write(d.join(format!("{sha}.zst")), b).expect("write blob");
}

const ZSTD: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

fn declared(raw: u32) -> Vec<u8> {
    let mut h = ZSTD.to_vec();
    h.push(0xA0);
    h.extend_from_slice(&raw.to_le_bytes());
    h
}

#[test]
fn falsify_tdc_007_the_weekly_receipt_measures_g16_or_says_unmeasured() {
    let r = Root::new("weekly");
    let (b, c) = ("b".repeat(64), "c".repeat(64));
    day(
        &r.0,
        ("2026", "09", "24"),
        &[rowq("t1", "q1"), rowq("t2", "q1"), rowq("t3", "q2")],
    );
    day(&r.0, ("2026", "09", "25"), &[rowq("t4", "q3")]);
    day(&r.0, ("2026", "09", "28"), &[rowq("t5", "q4")]);
    blob(&r.0, &b, &declared(1000));

    let s = Snapshot::read(&r.0).expect("snapshot");
    let w = weekly(&s, &r.0).expect("weekly");
    assert_eq!(w["schema"], json!(WEEKLY_SCHEME));
    let w39 = &w["weeks"][0];
    assert_eq!(w39["week"], json!("2026-W39"));
    assert_eq!(w39["days_with_rows"], json!(2));
    assert_eq!(w39["rows"], json!(4));
    assert_eq!(w39["quorums"], json!(3));
    assert_eq!(w39["quorums_per_day_min"], json!(1));
    assert_eq!(w39["quorums_per_day_max"], json!(2));
    let idx: u64 = s.files[..2].iter().map(|f| f.bytes).sum();
    assert_eq!(w39["index_bytes"], json!(idx));
    assert_eq!(w39["blobs"], json!(2));
    assert_eq!(w39["blobs_missing"], json!(1));
    assert_eq!(w39["blob_bytes_stored"], json!(19));
    assert_eq!(w39["blob_bytes_raw"], json!(1000));
    assert_eq!(
        w39["g16"],
        json!("unmeasured"),
        "a missing blob is not a measurement"
    );
    assert_eq!(w["weeks"][1]["week"], json!("2026-W40"));

    // A blob whose header does not declare its size is still unmeasured.
    blob(&r.0, &c, &[ZSTD[0], ZSTD[1], ZSTD[2], ZSTD[3], 0x00, 0x00]);
    let w = weekly(&s, &r.0).expect("weekly");
    assert_eq!(w["weeks"][0]["blobs_raw_undeclared"], json!(1));
    assert_eq!(w["weeks"][0]["g16"], json!("unmeasured"));

    // Every blob measured: 1–2 quorums/day is outside the [O] 50–200 band.
    blob(&r.0, &c, &declared(500));
    let w = weekly(&s, &r.0).expect("weekly");
    assert_eq!(w["weeks"][0]["blob_bytes_raw"], json!(1500));
    assert_eq!(w["weeks"][0]["g16"], json!("outside"));
    let text = render(&w);
    assert!(
        !text.contains(&b) && !text.contains(SECRET_TEXT),
        "no sha or row text"
    );

    // 50 quorums on one day, every blob measured and tiny: within.
    let r = Root::new("weekly-in");
    let rows: Vec<String> = (0..50)
        .map(|i| rowq(&format!("t{i}"), &format!("q{i}")))
        .collect();
    day(&r.0, ("2026", "09", "24"), &rows);
    blob(&r.0, &b, &declared(1000));
    blob(&r.0, &c, &declared(1000));
    let s = Snapshot::read(&r.0).expect("snapshot");
    assert_eq!(
        weekly(&s, &r.0).expect("weekly")["weeks"][0]["g16"],
        json!("within")
    );

    // In the quorum band but ~4.29 GB/day raw ≈ 1.57 TB/yr: outside the 0.45 TB cap.
    blob(&r.0, &c, &declared(u32::MAX));
    let w = weekly(&s, &r.0).expect("weekly");
    assert!(w["weeks"][0]["raw_tb_per_year"].as_f64().expect("f64") > G16_RAW_TB_PER_YEAR);
    assert_eq!(w["weeks"][0]["g16"], json!("outside"));
}

#[test]
fn falsify_tdc_008_iso_weeks_and_zstd_sizes_match_their_standards() {
    for (day, want) in [
        ("2026-09-24", "2026-W39"),
        ("2026-09-28", "2026-W40"),
        ("2026-01-01", "2026-W01"),
        ("2027-01-01", "2026-W53"),
        ("2021-01-03", "2020-W53"),
        ("2024-12-30", "2025-W01"),
        ("2020-02-29", "2020-W09"),
    ] {
        assert_eq!(iso_week(day).expect("day"), want, "{day}");
    }
    assert!(iso_week("2026-9").is_err());

    let z = |tail: &[u8]| [&ZSTD[..], tail].concat();
    assert_eq!(
        zstd_content_size(&z(&[0x20, 5])),
        Some(5),
        "single segment, 1-byte FCS"
    );
    assert_eq!(
        zstd_content_size(&z(&[0x40, 0x58, 0, 1])),
        Some(512),
        "2-byte FCS is +256"
    );
    assert_eq!(
        zstd_content_size(&z(&[0xA1, 7, 0x10, 0, 0, 0])),
        Some(16),
        "dict id skipped"
    );
    assert_eq!(
        zstd_content_size(&z(&[0x00, 0x58])),
        None,
        "no FCS declared"
    );
    assert_eq!(zstd_content_size(&[0, 1, 2, 3, 0x20, 5]), None, "not zstd");
    assert_eq!(zstd_content_size(&z(&[0x80, 1, 2])), None, "truncated");
}
