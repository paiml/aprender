//! #3560 R1: the example walk over a planted workspace, the family scan, and the repo's own corpus.

use super::*;
use std::fs;

fn write(root: &Path, rel: &str, text: &str) {
    let p = root.join(rel);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, text).unwrap();
}

/// A workspace with one member (two example forms plus a helper module), one excluded crate and a target dir.
fn planted() -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    let r = d.path();
    write(
        r,
        "Cargo.toml",
        "[workspace]\nmembers = [\"crates/*\"]\nexclude = [\"crates/old\"]\n",
    );
    write(r, "crates/a/Cargo.toml", "[package]\nname = \"a\"\n");
    write(
        r,
        "crates/a/examples/qwen.rs",
        "// run Qwen3.5-4B-Instruct\nfn main() {}\n",
    );
    write(r, "crates/a/examples/plain.rs", "fn main() {}\n");
    write(
        r,
        "crates/a/examples/multi/main.rs",
        "// llama.cpp parity, then TinyLlama\nfn main() {}\n",
    );
    write(
        r,
        "crates/a/examples/multi/helper.rs",
        "// qwen2 is not a target here\n",
    );
    write(r, "crates/a/examples/README.md", "qwen3.5\n");
    write(r, "crates/old/Cargo.toml", "[package]\nname = \"old\"\n");
    write(r, "crates/old/examples/x.rs", "fn main() {}\n");
    write(r, "target/debug/Cargo.toml", "[package]\nname = \"t\"\n");
    write(r, "target/debug/examples/y.rs", "fn main() {}\n");
    d
}

#[test]
fn a_planted_workspace_yields_exactly_its_member_targets() {
    let d = planted();
    let (ex, stats) = walk(d.path());
    let files: Vec<&str> = ex.iter().map(|e| e.file.as_str()).collect();
    assert_eq!(
        files,
        [
            "crates/a/examples/multi/main.rs",
            "crates/a/examples/plain.rs",
            "crates/a/examples/qwen.rs"
        ]
    );
    assert_eq!(ex[0].name, "multi");
    assert!(ex.iter().all(|e| e.krate == "a"));
    assert_eq!(stats.non_member_examples, 1, "crates/old is excluded");
    assert_eq!((stats.packages, stats.examples), (1, 3));
    assert_eq!((stats.naming_a_model, stats.naming_qwen35), (2, 1));
    assert_eq!(
        ex[0].families.iter().copied().collect::<Vec<_>>(),
        ["tinyllama"]
    );
    assert!(stats.errors.is_empty(), "{:?}", stats.errors);
}

#[test]
fn a_workspace_root_with_no_examples_is_an_error_not_a_green() {
    let d = tempfile::tempdir().unwrap();
    write(d.path(), "Cargo.toml", "[workspace]\nmembers = [\"x\"]\n");
    write(d.path(), "x/Cargo.toml", "[package]\nname = \"x\"\n");
    let (_, stats) = walk(d.path());
    assert_eq!(stats.examples, 0);
    assert_eq!(stats.errors.len(), 1, "{:?}", stats.errors);
    // Without a workspace manifest the corpus is not measured, and that is not an error.
    let bare = tempfile::tempdir().unwrap();
    assert!(walk(bare.path()).1.errors.is_empty());
}

#[test]
fn every_example_is_one_well_formed_node() {
    let d = planted();
    let mut g = Graph::new();
    for e in walk(d.path()).0 {
        emit(&mut g, &e);
    }
    let nodes: BTreeSet<String> = g
        .iter()
        .filter(|t| t.predicate == RDF_TYPE && t.object == Term::iri(ont("Example")))
        .map(|t| t.subject.clone())
        .collect();
    assert_eq!(nodes.len(), 3);
    for n in &nodes {
        for p in ["file", "crate", "name", "namesQwen35"] {
            let k = g.objects(n, &ex(p)).len();
            assert_eq!(k, 1, "{n} has {k} {p}");
        }
    }
    assert!(nodes.contains(&iri_path(
        "example",
        &["crates", "a", "examples", "qwen.rs"]
    )));
}

#[test]
fn the_family_scan_separates_models_from_look_alikes() {
    let cases: &[(&str, &[&str])] = &[
        ("Qwen3.5-9B", &["qwen3.5"]),
        ("qwen3_5_moe", &["qwen3.5"]),
        ("Qwen3-8B", &["qwen3"]),
        ("Qwen2.5-Coder-1.5B", &["qwen2.5"]),
        ("qwen2-0.5b", &["qwen2"]),
        ("TinyLlama-1.1B", &["tinyllama"]),
        ("Llama-3.2-1B", &["llama"]),
        ("llama.cpp server", &[]),
        ("graphics and phishing", &[]),
        ("phi-3-mini", &["phi"]),
        ("Phi", &["phi"]),
        ("gpt2 and GPT-2", &["gpt2"]),
        ("no model here", &[]),
    ];
    for (text, want) in cases {
        let got: Vec<&str> = families_in(text).into_iter().collect();
        assert_eq!(&got, want, "{text}");
    }
    assert!(positive_control());
}

/// The repo's own workspace: examples are read, all belong to a member, and the walk reports no error.
#[test]
fn the_repo_examples_are_extracted() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (ex, stats) = walk(&root);
    assert!(stats.errors.is_empty(), "{:?}", stats.errors);
    assert!(
        stats.examples > 500,
        "only {} examples read",
        stats.examples
    );
    assert_eq!(ex.len(), stats.examples);
    assert!(stats.packages > 20, "{} packages", stats.packages);
}
