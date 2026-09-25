use super::*;

/// Segments `from..to` of a synthetic diff; each segment's literals are its own.
fn segs(from: u32, to: u32) -> String {
    (from..to)
        .flat_map(|k| {
            (0..4).map(move |j| format!("+    let v = w.get({k}).map(|x| x * {k}{j}).unwrap_or({j});\n"))
        })
        .collect()
}

/// The same change re-indented and renamed.
fn perturb(d: &str) -> String {
    d.replace("    ", "  ").replace("w.get", "table.get")
}

fn plan() -> Plan {
    Plan { val_from: "2026-09-01".into(), test_from: "2026-10-01".into() }
}

fn unit<'a>(id: &'a str, pr: u64, at: &'a str, diff: &'a str) -> Unit<'a> {
    Unit { id, repo: "paiml/aprender", pr, at, diff }
}

fn splits(g: &[Guard]) -> Vec<Option<Split>> {
    g.iter().map(|g| g.split).collect()
}

#[test]
fn falsify_tsg_001_a_pr_never_straddles_splits() {
    let (a, b) = (segs(1, 6), segs(20, 25));
    // PR 1 opened in the train window; its second round lands in test time.
    let u = [
        unit("a", 1, "2026-08-01T00:00:00Z", &a),
        unit("b", 1, "2026-10-05T00:00:00Z", &b),
    ];
    let g = assign(&u, &[], &plan()).expect("assigns");
    assert_eq!(splits(&g), [Some(Split::Train), Some(Split::Train)]);
    assert_eq!(audit(&g), Audit::default());
    let mut bad = g.clone();
    bad[1].split = Some(Split::Test);
    assert_eq!(audit(&bad).straddling_groups, ["paiml/aprender#1"]);
}

#[test]
fn falsify_tsg_002_a_near_dup_cluster_closes_to_its_earliest_members_split() {
    let (a, b, c) = (segs(1, 6), segs(20, 25), segs(40, 45));
    let (a2, b2) = (perturb(&a), perturb(&b));
    let u = [
        // A chain: PR 1 {a} ~ PR 2 {a', b} ~ PR 3 {b'}; PR 4 unrelated.
        unit("b2", 3, "2026-10-09T00:00:00Z", &b2),
        unit("a", 1, "2026-08-01T00:00:00Z", &a),
        unit("a2", 2, "2026-10-05T00:00:00Z", &a2),
        unit("b", 2, "2026-10-05T00:00:00Z", &b),
        unit("c", 4, "2026-10-06T00:00:00Z", &c),
    ];
    let g = assign(&u, &[], &plan()).expect("assigns");
    let train = Some(Split::Train);
    assert_eq!(splits(&g), [train, train, train, train, Some(Split::Test)]);
    assert_eq!(g[2].dedup_cluster, "a");
    assert_eq!(audit(&g), Audit::default());
    // A per-row time split leaks the copy into test, and the audit sees it.
    let mut naive = g.clone();
    naive[2].split = Some(Split::Test);
    naive[2].group_key = "paiml/aprender#9".into();
    assert_eq!(audit(&naive).cross_split_clusters, ["a"]);
}

#[test]
fn falsify_tsg_003_the_embargo_is_at_least_14_days() {
    let d: Vec<String> = (0..5).map(|k| segs(10 * k + 1, 10 * k + 6)).collect();
    let u = [
        unit("t", 1, "2026-08-17T23:59:59Z", &d[0]), // 15 d before val: train
        unit("e1", 2, "2026-08-18T00:00:00Z", &d[1]), // 14 d before val: embargoed
        unit("v", 3, "2026-09-01T00:00:00Z", &d[2]),
        unit("e2", 4, "2026-09-30T00:00:00Z", &d[3]),
        unit("x", 5, "2026-10-01T00:00:00Z", &d[4]),
    ];
    let g = assign(&u, &[], &plan()).expect("assigns");
    assert_eq!(
        splits(&g),
        [Some(Split::Train), None, Some(Split::Val), None, Some(Split::Test)]
    );
    // An embargoed member never makes a straddle.
    let mut one = g.clone();
    one[1].group_key = one[0].group_key.clone();
    assert_eq!(audit(&one), Audit::default());
    // A val window no longer than the embargo is refused, as is a bad date.
    let short = Plan { val_from: "2026-09-01".into(), test_from: "2026-09-15".into() };
    assert!(assign(&u, &[], &short).is_err());
    let bad = [unit("t", 1, "yesterday", &d[0])];
    assert!(assign(&bad, &[], &plan()).is_err());
}

#[test]
fn falsify_tsg_004_a_perturbed_sealed_item_is_refused_with_its_whole_pr() {
    let (s, b) = (segs(1, 6), segs(20, 25));
    let leak = perturb(&s);
    let c = segs(40, 45);
    let u = [
        unit("leak", 1, "2026-08-01T00:00:00Z", &leak),
        unit("b", 1, "2026-08-02T00:00:00Z", &b),
        unit("c", 2, "2026-08-03T00:00:00Z", &c),
    ];
    let g = assign(&u, &[("sealed-7", &s)], &plan()).expect("assigns");
    let sealed = Some(Split::Sealed);
    assert_eq!(splits(&g), [sealed, sealed, Some(Split::Train)]);
    assert_eq!(g[0].dedup_cluster, "sealed-7");
}

#[test]
fn day_matches_the_civil_calendar() {
    let cases = [
        ("1970-01-01", Some(0)),
        ("2000-03-01", Some(11_017)),
        ("2026-09-25T12:00:00Z", Some(20_721)),
        ("2024-02-29", Some(19_782)),
        ("2026-13-01", None),
        ("26-09-25", None),
    ];
    for (s, want) in cases {
        assert_eq!(day(s), want, "{s}");
    }
}
