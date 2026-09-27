use super::*;
use crate::corpus::{hunk_fingerprints, Class, Sealed};

const IN_TREE: &str = include_str!("../../../evidence/verbs/refusals.json");

fn ledger(rows: &str) -> String {
    format!(r#"{{"schema":"apr-refusal-ledger/v1","refusals":[{rows}]}}"#)
}

fn refusal(verb: &str, removed_by: &str) -> String {
    format!(
        r#"{{"verb":"{verb}","reason":"refuses qwen35","exit_code":8,"removed_by":"{removed_by}"}}"#
    )
}

#[test]
fn falsify_rxb_001_no_b2_row_is_ready_without_an_acceptance_receipt() {
    // The in-tree ledger at HEAD: every B2 row is NotRun, none silently Ready.
    let l = parse_ledger(IN_TREE).expect("in-tree ledger parses");
    let rows = status(&l, &BTreeMap::new()).expect("status");
    assert_eq!(rows.len(), 4);
    assert!(rows.iter().all(|r| !matches!(r.state, State::Ready { .. })));

    let l = parse_ledger(&ledger(&refusal("distill", "v0.75"))).expect("parses");
    let acc = BTreeMap::from([("finetune".to_string(), "sha-ft".to_string())]);
    let rows = status(&l, &acc).expect("status");
    let by: BTreeMap<&str, &State> = rows.iter().map(|r| (r.tier.as_str(), &r.state)).collect();
    assert_eq!(
        by["B2a"],
        &State::VerbRefused {
            removed_by: "v0.75".into(),
            exit_code: 8
        }
    );
    assert_eq!(
        by["B2b"],
        &State::Ready {
            acceptance: "sha-ft".into()
        }
    );
    assert_eq!(
        by["B2c"],
        &State::Unledgered,
        "no refusal row is not acceptance"
    );

    let both = BTreeMap::from([("distill".to_string(), "sha".to_string())]);
    assert!(status(&l, &both).is_err(), "accepted and refused at once");
}

#[test]
fn falsify_rxb_002_the_ledger_is_refusal_receipt_v1_or_rejected() {
    for good in ["v0.75", "v1.0", "never", "unscheduled"] {
        assert!(valid_removed_by(good), "{good}");
    }
    for bad in [
        "tbd", "soon", "0.70", "v0.70.1", "v0.", "v.7", "vx.y", "", "9f3c2a1",
    ] {
        assert!(!valid_removed_by(bad), "{bad}");
        assert!(
            parse_ledger(&ledger(&refusal("merge", bad))).is_err(),
            "{bad}"
        );
    }
    let dup = format!(
        "{},{}",
        refusal("merge", "never"),
        refusal("merge", "v0.75")
    );
    assert!(
        parse_ledger(&ledger(&dup)).is_err(),
        "two rows for one verb"
    );
    assert!(parse_ledger(&ledger(&refusal("merge", "v0.75")).replace(":8", ":0")).is_err());
    assert!(parse_ledger(&ledger("").replace("ledger/v1", "ledger/v2")).is_err());
}

fn diff(body: &str) -> String {
    format!("--- a/src/a.rs\n+++ b/src/a.rs\n@@ -10,5 +10,5 @@\n fn f(x: u32) -> bool {{\n{body} }}\n \n")
}

fn fixture() -> (Vec<Item>, Index, String) {
    let mk = |id: &str, split: Split, d: &str| {
        let mut i = Item::new(id.into(), Class::R, "src".into(), d);
        i.split = split;
        i
    };
    let test_diff = diff("-    z * 2\n+    z * 3\n");
    let items = vec![
        mk("D01", Split::Dev, &diff("-    a\n+    b\n")),
        mk("D02", Split::Dev, &diff("-    c\n+    d\n")),
        mk("T01", Split::Test, &test_diff),
    ];
    let ix = Index::new(&[Sealed {
        id: "T01".into(),
        diff_sha256: sha256_hex(test_diff.as_bytes()),
        hunks: hunk_fingerprints(&test_diff),
    }]);
    (items, ix, test_diff)
}

fn row(id: &str, k: usize) -> String {
    let pos: Vec<(u32, f32)> = (0..k).map(|t| (t as u32, -(t as f32))).collect();
    serde_json::to_string(&TeacherRow {
        item_id: id.into(),
        top_k: vec![pos.clone(), pos],
    })
    .expect("serializes")
}

#[test]
fn falsify_rxb_003_the_teacher_dataset_is_receipted_with_zero_test_hashes() {
    let (items, ix, test_diff) = fixture();
    let good = format!("{}\n{}\n", row("D01", 4), row("D02", 4));
    let r = teacher_receipt(&good, "w27b", 4, &items, &ix).expect("receipted");
    assert_eq!((r.count, r.m, r.k, r.test_hits), (2, 1, 4, 0));
    assert_eq!(r.sha256, sha256_hex(good.as_bytes()));
    assert_eq!(r.sealed_items_indexed, 1);

    let refused = |text: &str, why: &str| {
        let e = teacher_receipt(text, "w27b", 4, &items, &ix).expect_err(why);
        assert!(e.iter().any(|x| x.contains(why)), "{why}: {e:?}");
    };
    refused(
        &format!("{good}{}\n", row("T01", 4)),
        "T01: a sealed test item",
    );
    refused(&format!("{good}{}\n", row("D01", 4)), "more than one row");
    refused(&format!("{good}{}\n", row("X9", 4)), "not a corpus item");
    refused(&format!("{}\n", row("D01", 3)), "not 4 finite logits");
    refused(&row("D01", 4).replace("-1.0", "NaN"), "line 1");
    // A sealed diff carried in any string field is a hit.
    let smuggled = format!(
        "{good}{}\n",
        format!(
            r#"{{"item_id":"D02","top_k":[],"note":{}}}"#,
            serde_json::to_string(&test_diff).expect("serializes")
        )
    );
    refused(&smuggled, "sealed test leak");
    let leak = format!(r#"{good}"{}""#, sha256_hex(test_diff.as_bytes()));
    refused(&leak, "sealed test leak");
    refused("", "no teacher rows");
    let e = teacher_receipt(&good, "w27b", 4, &items, &Index::default()).expect_err("empty");
    assert!(e[0].contains("prove nothing"), "{e:?}");
}
