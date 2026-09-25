use super::*;
use crate::corpus::{sha256_hex, Class, Loc};
use crate::receipt::tests::{row, EXPECT};

fn s(id: &str, defect: bool, verdict: Verdict, localized: bool) -> Scored {
    Scored {
        id: id.into(),
        defect,
        verdict,
        localized,
        output_sha: verdict.executed().then(|| format!("sha-{id}")),
    }
}

/// Hand-computed fixture: 6 defects, 4 good.
fn fixture() -> Vec<Scored> {
    use Verdict::{Fail, Pass, Unparsed};
    vec![
        s("d1", true, Fail, true),
        s("d2", true, Fail, false),
        s("d3", true, Fail, true),
        s("d4", true, Pass, false),
        s("d5", true, Unparsed, false),
        s("d6", true, Verdict::NotRun(NotRun::ContextOverflow), false),
        s("g1", false, Pass, false),
        s("g2", false, Fail, false),
        s("g3", false, Unparsed, false),
        s(
            "g4",
            false,
            Verdict::NotRun(NotRun::NoDeclaredExecutor),
            false,
        ),
    ]
}

fn kn(r: Ratio) -> (u64, u64) {
    (r.k, r.n)
}

#[test]
fn metrics_match_hand_computation() {
    let l = score(&fixture());
    assert_eq!(l.items, 10);
    assert_eq!(kn(l.parse_rate), (6, 8), "parsed ÷ executed");
    assert_eq!(
        kn(l.recall),
        (3, 6),
        "NotRun defect stays in the denominator"
    );
    assert_eq!(kn(l.precision), (3, 4));
    assert_eq!(kn(l.false_refute), (1, 4));
    assert_eq!(kn(l.localization), (2, 3));
    assert_eq!(kn(l.correct), (4, 10));
    assert_eq!(l.unparsed, 2);
    assert_eq!(l.not_run.get("ContextOverflow"), Some(&1));
    assert_eq!(l.not_run.get("NoDeclaredExecutor"), Some(&1));
    // Wilson 3/6 by hand: centre 0.5, half-width 0.31238.
    let (lo, hi) = l.recall.ci.expect("ci");
    assert!(
        (lo - 0.187_62).abs() < 1e-4 && (hi - 0.812_38).abs() < 1e-4,
        "{lo} {hi}"
    );
    assert_eq!(Ratio::new(0, 0).value(), None);
}

#[test]
fn falsify_rxr_002_not_run_is_never_correct() {
    let why = [
        NotRun::NoDeclaredExecutor,
        NotRun::ContextOverflow,
        NotRun::Refused,
        NotRun::ServeError,
        NotRun::Inadmissible,
    ];
    let rows: Vec<Scored> = why
        .iter()
        .enumerate()
        .flat_map(|(i, w)| {
            [
                s(&format!("d{i}"), true, Verdict::NotRun(*w), false),
                s(&format!("g{i}"), false, Verdict::NotRun(*w), false),
            ]
        })
        .collect();
    let l = score(&rows);
    assert_eq!(kn(l.correct), (0, 10), "NotRun counted as correct");
    assert_eq!(kn(l.recall), (0, 5));
    assert_eq!(kn(l.false_refute), (0, 5));
    assert_eq!(kn(l.parse_rate), (0, 0), "NotRun is not attempted");
    // A missing receipt becomes NotRun{Inadmissible}, in the denominator.
    let item = crate::corpus::Item::new("R-pr1".into(), Class::R, "t".into(), "--- a\n+++ b\n");
    let (got, _) = collect("", EXPECT, &[&item], |_| true, |_| None);
    assert_eq!(got[0].verdict, Verdict::NotRun(NotRun::Inadmissible));
    assert_eq!(kn(score(&got).recall), (0, 1));
}

#[test]
fn falsify_rxr_003_unparsed_is_never_a_pass() {
    let rows = vec![
        s("g1", false, Verdict::Unparsed, false),
        s("g2", false, Verdict::Unparsed, false),
        s("d1", true, Verdict::Unparsed, false),
    ];
    let l = score(&rows);
    assert_eq!(
        kn(l.correct),
        (0, 3),
        "Unparsed counted as PASS on a good item"
    );
    assert_eq!(kn(l.parse_rate), (0, 3));
    assert_eq!(kn(l.recall), (0, 1), "Unparsed is not a FAIL either");
    assert_eq!(kn(l.false_refute), (0, 2));
    let (pairs, dropped) = paired_parsed(&rows, &rows);
    assert!(
        pairs.is_empty() && dropped == 3,
        "Unparsed never enters a paired rate"
    );
}

/// Items, receipts and raw outputs for [`collect`].
fn world() -> (Vec<crate::corpus::Item>, BTreeMap<String, String>) {
    let d = "--- a/crates/x/src/api/router.rs\n+++ b/crates/x/src/api/router.rs\n@@ -1,1 +1,1 @@\n-a\n+b\n";
    let r = crate::corpus::Item::new("R-pr1".into(), Class::R, "t".into(), d);
    let g = crate::corpus::Item::new("G-pr2".into(), Class::G, "t".into(), "--- a/g\n+++ b/g\n");
    let raw = BTreeMap::from([
        (
            "raw/R-pr1.txt".to_string(),
            "VERDICT: FAIL\n- api/router.rs: off by one".to_string(),
        ),
        ("raw/G-pr2.txt".to_string(), "VERDICT: PASS".to_string()),
    ]);
    (vec![r, g], raw)
}

fn receipt_for(
    item: &crate::corpus::Item,
    verdict: Verdict,
    raw: &BTreeMap<String, String>,
) -> Receipt {
    let mut r = row(&item.id, item.class, verdict);
    r.item_sha256.clone_from(&item.diff_sha256);
    let path = format!("raw/{}.txt", item.id);
    let o = r.output.as_mut().expect("output");
    o.sha256 = sha256_hex(raw[&path].as_bytes());
    o.path = path;
    r
}

fn lines(rs: &[Receipt]) -> String {
    rs.iter()
        .map(|r| serde_json::to_string(r).expect("json") + "\n")
        .collect()
}

#[test]
fn collect_verifies_raw_output_and_reparses_the_verdict() {
    let (items, raw) = world();
    let exp: Vec<&crate::corpus::Item> = items.iter().collect();
    let read = |p: &str| raw.get(p).cloned();
    let good = [
        receipt_for(&items[0], Verdict::Fail, &raw),
        receipt_for(&items[1], Verdict::Pass, &raw),
    ];
    let (got, rej) = collect(&lines(&good), EXPECT, &exp, |_| true, read);
    assert!(rej.is_empty(), "{rej:?}");
    assert_eq!(got[0].verdict, Verdict::Fail);
    assert!(got[0].localized, "api/router.rs names the defect file");
    assert_eq!(kn(score(&got).correct), (2, 2));

    // Each corruption turns exactly the corrupted item into NotRun{Inadmissible}.
    let corruptions: Vec<(&str, Box<dyn Fn(&mut Receipt)>)> = vec![
        (
            "raw sha",
            Box::new(|r| r.output.as_mut().expect("o").sha256 = "f".repeat(64)),
        ),
        (
            "raw missing",
            Box::new(|r| r.output.as_mut().expect("o").path = "raw/nope.txt".into()),
        ),
        (
            "verdict disagrees with raw",
            Box::new(|r| r.verdict = Verdict::Pass),
        ),
        ("item sha", Box::new(|r| r.item_sha256 = "e".repeat(64))),
        (
            "unexpected item",
            Box::new(|r| r.item_id = "R-pr999".into()),
        ),
    ];
    for (name, bad) in corruptions {
        let mut rs = good.clone();
        bad(&mut rs[0]);
        let (got, rej) = collect(&lines(&rs), EXPECT, &exp, |_| true, read);
        assert_eq!(rej.len(), 1, "{name}");
        assert_eq!(
            got[0].verdict,
            Verdict::NotRun(NotRun::Inadmissible),
            "{name}"
        );
        assert_eq!(
            got[1].verdict,
            Verdict::Pass,
            "{name}: the other item is untouched"
        );
    }

    let dup = [good[0].clone(), good[0].clone(), good[1].clone()];
    let (got, _) = collect(&lines(&dup), EXPECT, &exp, |_| true, read);
    assert_eq!(
        got[0].verdict,
        Verdict::NotRun(NotRun::Inadmissible),
        "duplicates void the item"
    );

    let (got, rej) = collect(&lines(&good), EXPECT, &exp, |r| r.cell == "C9", read);
    assert!(
        rej.is_empty()
            && got
                .iter()
                .all(|g| g.verdict == Verdict::NotRun(NotRun::Inadmissible))
    );
    let _ = Loc {
        file: String::new(),
        line: 0,
    };
}

#[test]
fn divergence_pairs_and_mcnemar_by_hand() {
    let a = fixture();
    let mut b = fixture();
    b[0].output_sha = Some("other".into()); // bytes differ
    b[3].verdict = Verdict::Fail; // verdict differs (and output sha stays)
    assert_eq!(
        divergence(&a, &b),
        (2, 2),
        "d6/g4 did not run: missing, not agreeing"
    );

    let (pairs, dropped) = paired_parsed(&a, &b);
    assert_eq!((pairs.len(), dropped), (6, 4));

    // Challenger right / champion wrong on 5, the reverse on 1:
    // p = P(X ≥ 5 | n = 6, ½) = (6 + 1) / 64.
    let champ: Vec<Scored> = (0..8)
        .map(|i| {
            s(
                &format!("d{i}"),
                true,
                if i == 5 { Verdict::Fail } else { Verdict::Pass },
                false,
            )
        })
        .collect();
    let chall: Vec<Scored> = (0..8)
        .map(|i| {
            s(
                &format!("d{i}"),
                true,
                if i < 5 { Verdict::Fail } else { Verdict::Pass },
                false,
            )
        })
        .collect();
    assert!((h3_p(&chall, &champ) - 7.0 / 64.0).abs() < 1e-12);
    assert_eq!(
        correctness_pairs(&[], &champ[..1]),
        vec![(false, false)],
        "missing challenger is wrong"
    );
}
