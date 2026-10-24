//! The `--selftest` rows of the old `roadmap_fragments.py`, printed `"%-4s %-52s %s"`.

use super::fragments::{aggregate, aggregate_with, census, roadmap_path, split};
use super::{split_entries, Entry};

const BASE: &str = "roadmap_version: '1.0'\nroadmap:\n\
- id: PMAT-100\n  title: a\n\
- id: PMAT-300\n  title: c\n\
- id: LEGACY-THING\n  title: legacy\n";

struct Rows {
    bad: usize,
    n: usize,
}

impl Rows {
    fn row(&mut self, name: &str, ok: bool, detail: &str) {
        self.n += 1;
        if !ok {
            self.bad += 1;
        }
        println!(
            "{:<4} {:<52} {}",
            if ok { "ok" } else { "FAIL" },
            name,
            detail
        );
    }
}

fn ids(text: &str) -> Vec<String> {
    split_entries(text).1.into_iter().map(|e| e.0).collect()
}

fn frag(id: &str, body: &str) -> Entry {
    (id.to_string(), format!("- id: {id}\n  {body}\n"))
}

fn show(v: &[String]) -> String {
    let q: Vec<String> = v.iter().map(|s| format!("'{s}'")).collect();
    format!("[{}]", q.join(", "))
}

fn ok(base: &str, f: &[Entry]) -> String {
    aggregate(base, f).unwrap_or_default()
}

fn placement_rows(r: &mut Rows) {
    let out = ok(BASE, &[]);
    r.row(
        "no fragments -> base is byte-identical",
        out == BASE,
        if out == BASE { "" } else { "differs" },
    );
    let i = ids(&ok(BASE, &[frag("PMAT-200", "title: b")]));
    r.row(
        "numeral between two -> sorted slot",
        i == ["PMAT-100", "PMAT-200", "PMAT-300", "LEGACY-THING"],
        &show(&i),
    );
    let i = ids(&ok(BASE, &[frag("PMAT-400", "title: d")]));
    r.row(
        "numeral after all of its prefix -> appended",
        i.last().is_some_and(|s| s == "PMAT-400"),
        &show(&i),
    );
    let i = ids(&ok(BASE, &[frag("APEX-1", "title: new prefix")]));
    r.row(
        "unseen prefix -> appended",
        i.last().is_some_and(|s| s == "APEX-1"),
        &show(&i),
    );
    let i = ids(&ok(BASE, &[frag("FREEFORM", "title: legacy")]));
    r.row(
        "legacy id (no numeral) -> appended",
        i.last().is_some_and(|s| s == "FREEFORM"),
        &show(&i),
    );
    let i = ids(&ok(BASE, &[frag("PMAT-90", "title: ninety")]));
    r.row(
        "numeral compared NUMERICALLY (PMAT-90 before PMAT-100)",
        i.first().is_some_and(|s| s == "PMAT-90"),
        &show(&i),
    );
    let out = ok(BASE, &[frag("PMAT-100", "title: superseded")]);
    let i = ids(&out);
    r.row(
        "fragment supersedes a base entry of the same id",
        i == ["PMAT-100", "PMAT-300", "LEGACY-THING"] && out.contains("superseded"),
        &show(&i),
    );
}

fn property_rows(r: &mut Rows) {
    match aggregate(BASE, &[frag("PMAT-9", "t: a"), frag("PMAT-9", "t: b")]) {
        Ok(_) => r.row(
            "duplicate id AMONG fragments refused",
            false,
            "no error raised",
        ),
        Err(e) => {
            let d: String = e.chars().take(44).collect();
            r.row(
                "duplicate id AMONG fragments refused",
                e.contains("duplicate id among fragments"),
                &d,
            );
        }
    }
    let f = [frag("PMAT-200", "title: b")];
    let once = ok(BASE, &f);
    let twice = ok(&once, &f);
    let detail = format!(
        "second pass changed {} bytes",
        twice.len().abs_diff(once.len())
    );
    r.row(
        "IDEMPOTENT: aggregate(aggregate(x)) == aggregate(x)",
        once == twice,
        if once == twice { "" } else { &detail },
    );
    let two = [frag("PMAT-150", "t: x"), frag("PMAT-250", "t: y")];
    let rev = [two[1].clone(), two[0].clone()];
    r.row(
        "DETERMINISTIC: fragment input order does not change the bytes",
        ok(BASE, &two) == ok(BASE, &rev),
        "",
    );
    let i = ids(&ok(
        BASE,
        &[frag("PMAT-150", "title: x"), frag("PMAT-250", "title: y")],
    ));
    r.row(
        "two fragments -> both at their own slots",
        i == [
            "PMAT-100",
            "PMAT-150",
            "PMAT-250",
            "PMAT-300",
            "LEGACY-THING",
        ],
        &show(&i),
    );
}

fn mutation_row(r: &mut Rows) {
    let append_only: fn(&[Entry], &str) -> usize = |e, _| e.len();
    let m = aggregate_with(BASE, &[frag("PMAT-200", "title: b")], append_only).unwrap_or_default();
    let caught = ids(&m) != ["PMAT-100", "PMAT-200", "PMAT-300", "LEGACY-THING"];
    r.row(
        "mutation: append-only placement is CAUGHT",
        caught,
        if caught {
            ""
        } else {
            "mutant survived -- row 2 proves nothing"
        },
    );
}

fn census_rows(r: &mut Rows) {
    let path = roadmap_path();
    let Ok(text) = std::fs::read_to_string(&path) else {
        r.row(
            "census SKIPPED: no roadmap.yaml here",
            true,
            "[U] not measured",
        );
        return;
    };
    let c = census(&text);
    let n = |k: (&'static str, &'static str)| c.get(&k).map_or(0, Vec::len);
    let total: usize = c.values().map(Vec::len).sum();
    let unsafe_n = n(("unsafe", "prefixN")) + n(("unsafe", "legacy"));
    let detail = format!(
        "fragmentable={} safe-legacy={} NOT-filename-safe={} total={}",
        n(("safe", "prefixN")),
        n(("safe", "legacy")),
        unsafe_n,
        total
    );
    r.row(
        "census: every entry is accounted for by id shape",
        total > 0,
        &detail,
    );
    r.row(
        "census: a real id contains a path separator (the regex must reject it)",
        c.values().flatten().any(|e| e.contains('/')),
        "filename-safety is load-bearing, not defensive",
    );
}

/// RMFR-F-003: split() proves each entry parses alone; a duplicated id is refused.
fn split_rows(r: &mut Rows) {
    let dir = std::path::Path::new("/nonexistent-split-selftest");
    let got = split(BASE, dir, false).map(|f| f.len());
    r.row(
        "split yields one fragment per entry",
        got == Ok(3),
        &format!("{got:?}"),
    );
    let dup = format!("{BASE}- id: PMAT-100\n  title: again\n");
    let refused = split(&dup, dir, false);
    r.row(
        "split refuses a duplicated id",
        refused.is_err(),
        &format!("{:?}", refused.map(|f| f.len())),
    );
}

/// Run every row; 0 green, 1 any red.
#[must_use]
pub fn selftest() -> i32 {
    let mut r = Rows { bad: 0, n: 0 };
    placement_rows(&mut r);
    property_rows(&mut r);
    mutation_row(&mut r);
    census_rows(&mut r);
    split_rows(&mut r);
    println!("\n{} row(s), {} red", r.n, r.bad);
    i32::from(r.bad > 0)
}
