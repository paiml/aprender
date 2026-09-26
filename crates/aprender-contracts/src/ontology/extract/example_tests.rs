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
        "crates/a/examples/parity.rs",
        "//! ont:model-pinned: GGUF parity fixture is Qwen2-0.5B\n// qwen2\nfn main() {}\n",
    );
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
            "crates/a/examples/parity.rs",
            "crates/a/examples/plain.rs",
            "crates/a/examples/qwen.rs"
        ]
    );
    assert_eq!(ex[0].name, "multi");
    assert!(ex.iter().all(|e| e.krate == "a"));
    assert_eq!(stats.non_member_examples, 1, "crates/old is excluded");
    assert_eq!((stats.packages, stats.examples), (1, 4));
    assert_eq!((stats.naming_a_model, stats.naming_qwen35), (3, 1));
    // multi names TinyLlama only (stale); parity names qwen2 but is pinned; qwen names the current family.
    assert_eq!((stats.stale, stats.pinned), (1, 1));
    assert_eq!(
        ex[0].families.iter().copied().collect::<Vec<_>>(),
        ["tinyllama"]
    );
    assert!(stats.errors.is_empty(), "{:?}", stats.errors);
}

#[test]
fn a_workspace_root_that_read_nothing_is_an_error_and_one_with_no_examples_is_not() {
    // A member with no examples/: zero is the measurement, not an error (the ONT-3b code-bound fixture's shape).
    let d = tempfile::tempdir().unwrap();
    write(d.path(), "Cargo.toml", "[workspace]\nmembers = [\"x\"]\n");
    write(d.path(), "x/Cargo.toml", "[package]\nname = \"x\"\n");
    let (_, stats) = walk(d.path());
    assert_eq!((stats.examples, stats.members), (0, 1));
    assert!(stats.errors.is_empty(), "{:?}", stats.errors);
    // Targets on disk, none admitted: the membership reading lost the corpus.
    write(d.path(), "y/Cargo.toml", "[package]\nname = \"y\"\n");
    write(d.path(), "y/examples/a.rs", "fn main() {}\n");
    let (_, stats) = walk(d.path());
    assert_eq!((stats.examples, stats.non_member_examples), (0, 1));
    assert_eq!(stats.errors.len(), 1, "{:?}", stats.errors);
    // A workspace that admits no member read nothing.
    let none = tempfile::tempdir().unwrap();
    write(
        none.path(),
        "Cargo.toml",
        "[workspace]\nmembers = [\"gone\"]\n",
    );
    assert_eq!(walk(none.path()).1.errors.len(), 1);
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
    assert_eq!(nodes.len(), 4);
    for n in &nodes {
        for p in ["file", "crate", "name", "namesQwen35", "modelCurrent"] {
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

#[test]
fn a_pin_needs_the_marker_in_a_comment_and_a_reason() {
    assert_eq!(
        pin_reason("//! ont:model-pinned: parity fixture\n").as_deref(),
        Some("parity fixture")
    );
    assert_eq!(
        pin_reason("    // ont:model-pinned: tiny\n").as_deref(),
        Some("tiny")
    );
    assert_eq!(
        pin_reason("// ont:model-pinned:   \n"),
        None,
        "no reason, no pin"
    );
    assert_eq!(
        pin_reason("let s = \"ont:model-pinned: x\";\n"),
        None,
        "not a comment"
    );
    let stale = Example {
        file: "f".into(),
        krate: "c".into(),
        name: "n".into(),
        families: ["qwen2.5"].into_iter().collect(),
        pinned: None,
    };
    assert!(!stale.model_current());
    assert!(Example {
        pinned: Some("why".into()),
        ..stale.clone()
    }
    .model_current());
    assert!(Example {
        families: BTreeSet::new(),
        ..stale
    }
    .model_current());
}

/// R4's drift gate: the stale examples on this tree are pinned shrink-only. A new example on an old model raises the
/// count and fails here; migrating or pinning one lowers it, and then the pin must be lowered with it.
#[test]
fn the_repo_stale_examples_are_pinned_shrink_only() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (ex, stats) = walk(&root);
    let stale: Vec<&str> = ex
        .iter()
        .filter(|e| !e.model_current())
        .map(|e| e.file.as_str())
        .collect();
    assert!(
        stats.stale <= STALE_EXAMPLES_PINNED,
        "{} examples name a model other than {CURRENT_FAMILY} without `// {PIN_MARKER} <reason>` (pinned {STALE_EXAMPLES_PINNED}); \
         migrate or pin the new one(s). Stale: {stale:#?}",
        stats.stale
    );
    assert_eq!(
        stats.stale, STALE_EXAMPLES_PINNED,
        "the stale count fell to {}: lower STALE_EXAMPLES_PINNED to it (shrink-only)",
        stats.stale
    );
}
