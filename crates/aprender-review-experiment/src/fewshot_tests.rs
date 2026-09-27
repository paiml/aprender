use super::*;
use crate::corpus::{hunk_fingerprints, sha256_hex, Class, Loc, Sealed};

fn diff(body: &str) -> String {
    format!("--- a/src/a.rs\n+++ b/src/a.rs\n@@ -10,5 +10,5 @@\n fn f(x: u32) -> bool {{\n{body} }}\n \n")
}

fn item(id: &str, class: Class, split: Split, d: &str) -> (Item, String) {
    let mut i = Item::new(id.into(), class, "src".into(), d);
    i.split = split;
    if class.is_defect() {
        i.defect = vec![Loc {
            file: "src/a.rs".into(),
            line: 11,
        }];
    }
    (i, d.to_string())
}

fn sealed(test: &[&(Item, String)]) -> Index {
    Index::new(
        &test
            .iter()
            .map(|(i, d)| Sealed {
                id: i.id.clone(),
                diff_sha256: sha256_hex(d.as_bytes()),
                hunks: hunk_fingerprints(d),
            })
            .collect::<Vec<_>>(),
    )
}

fn pool() -> Vec<(Item, String)> {
    vec![
        item(
            "P01",
            Class::P,
            Split::Dev,
            &diff("-    x > 1\n+    x >= 1\n"),
        ),
        item(
            "G01",
            Class::G,
            Split::Dev,
            &diff("-    let y = 2;\n+    let y = 3;\n"),
        ),
        item(
            "R01",
            Class::R,
            Split::Dev,
            &diff("-    x > 1 && x < 9\n+    x >= 1 && x < 9\n"),
        ),
    ]
}

#[test]
fn falsify_rxf_001_a_test_item_in_the_pool_is_an_error() {
    let mut p = pool();
    p.push(item(
        "T01",
        Class::R,
        Split::Test,
        &diff("-    a\n+    b\n"),
    ));
    let (q, qd) = item(
        "Q",
        Class::P,
        Split::Test,
        &diff("-    x > 2\n+    x >= 2\n"),
    );
    let e = select(&q, &qd, &p, 2).expect_err("sealed item in pool");
    assert!(e.contains("T01") && e.contains("R-2"), "{e}");
}

#[test]
fn falsify_rxf_002_retrieval_ranks_by_similarity_and_never_returns_the_query() {
    let p = pool();
    let (q, qd) = item(
        "Q",
        Class::P,
        Split::Test,
        &diff("-    x > 1 && x < 9 || y\n+    x >= 1 && x < 9 || y\n"),
    );
    let got: Vec<&str> = select(&q, &qd, &p, 2)
        .expect("selects")
        .iter()
        .map(|(i, _)| i.id.as_str())
        .collect();
    assert_eq!(got, ["R01", "P01"]);
    // A dev query retrieves neither itself nor a byte-identical copy.
    let self_copy = select(&p[2].0, &p[2].1, &p, 3).expect("selects");
    assert!(self_copy.iter().all(|(i, _)| i.id != "R01"));
    assert_eq!(similarity("", &p[0].1), 0.0);
}

#[test]
fn falsify_rxf_003_a_rendered_prompt_carrying_a_test_hunk_is_refused() {
    let p = pool();
    let test = item(
        "T07",
        Class::R,
        Split::Test,
        &diff("-    z * 2\n+    z * 3\n"),
    );
    let ix = sealed(&[&test]);
    let ok = render("BASE", &[&p[0], &p[1]]);
    assert!(ok.contains("VERDICT: FAIL\n- defect at src/a.rs:11"));
    assert!(ok.contains("VERDICT: PASS"));
    guard(&ok, &ix, "prompt").expect("clean prompt");
    // The same prompt with the sealed item smuggled in as an example.
    let leaked = render("BASE", &[&p[0], &test]);
    // The gold line after the fence joins the hunk in running text: the
    // whole-prompt scan alone is blind here, which is why guard scans blocks.
    assert!(ix.scan(&leaked, "prompt").is_empty());
    let e = guard(&leaked, &ix, "prompt").expect_err("leak");
    assert!(e.contains("T07"), "{e}");
    assert!(
        guard(&ok, &Index::default(), "prompt").is_err(),
        "an empty index proves nothing"
    );
}
