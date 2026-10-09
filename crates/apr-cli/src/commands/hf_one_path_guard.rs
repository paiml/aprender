//! #4961: `apr` has ONE Hugging Face upload path, aprender-core `hf_hub`, whose
//! `repo_api` holds the Hub write endpoints. This scan is RED when a write
//! endpoint is named anywhere in `crates/apr-cli/` or the root `src/`, the code
//! that builds `apr`. It reads the string literals on the non-comment lines of
//! every `.rs` file there:
//!
//! - an upload-only endpoint (`info/lfs/objects/batch`, `preupload/`,
//!   `xet-write-token`, `api/repos/create`) is a hit by itself;
//! - a commit, branch, tag or upload endpoint (`commit/`, `branch/`, `tag/`,
//!   `/upload/`) is a hit in a file that also names the Hub (`api/models`,
//!   `api/datasets`, `api/spaces`, `huggingface.co`, `hf.co/`).
//!
//! The one file it skips is this one: it names every endpoint as data, and the
//! last test checks that it holds no HTTP client.

use std::path::{Path, PathBuf};
use tempfile::TempDir;

const UPLOAD_ONLY: [&str; 4] = [
    "info/lfs/objects/batch",
    "preupload/",
    "xet-write-token",
    "api/repos/create",
];
const HUB: [&str; 5] = [
    "api/models",
    "api/datasets",
    "api/spaces",
    "huggingface.co",
    "hf.co/",
];
const WRITE: [&str; 4] = ["commit/", "branch/", "tag/", "/upload/"];
const SCOPES: [&str; 2] = ["crates/apr-cli", "src"];
const SELF: &str = "crates/apr-cli/src/commands/hf_one_path_guard.rs";

/// The string literals on the non-comment lines of `src`, with line numbers.
fn literals(src: &str) -> impl Iterator<Item = (usize, &str)> {
    src.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim_start().starts_with("//"))
        .flat_map(|(i, l)| l.split('"').skip(1).step_by(2).map(move |s| (i + 1, s)))
}

fn names_any(s: &str, set: &[&str]) -> bool {
    set.iter().any(|n| s.contains(n))
}

/// The hits in one file's text, as `line:literal`.
fn hits(src: &str) -> Vec<String> {
    let at = |set: &'static [&'static str]| {
        literals(src)
            .filter(move |(_, s)| names_any(s, set))
            .map(|(n, s)| format!("{n}:{s}"))
    };
    let upload: Vec<String> = at(&UPLOAD_ONLY).collect();
    if !upload.is_empty() || at(&HUB).next().is_none() {
        return upload;
    }
    at(&WRITE).collect()
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|x| x == "rs") {
            out.push(path);
        }
    }
}

/// Files scanned per scope, and one `path line:literal…` row per file that hits.
fn scan(root: &Path) -> (Vec<usize>, Vec<String>) {
    let skip = root.join(SELF);
    let mut counts = Vec::new();
    let mut rows = Vec::new();
    for scope in SCOPES {
        let mut files = Vec::new();
        rust_files(&root.join(scope), &mut files);
        files.sort();
        counts.push(files.len());
        for f in files.iter().filter(|f| **f != skip) {
            let src = std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
            let h = hits(&src);
            if !h.is_empty() {
                let rel = f.strip_prefix(root).unwrap_or(f);
                rows.push(format!("{} {}", rel.display(), h.join(" ")));
            }
        }
    }
    (counts, rows)
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A must-hit / must-not-hit table, one file text per row.
#[test]
fn the_case_table() {
    let must_hit = [
        r#"let u = format!("{}/{}.git/info/lfs/objects/batch", base, repo);"#,
        r#"let u = format!("{base}/api/models/{repo}/preupload/{rev}");"#,
        r#"agent.post(&format!("{base}/api/repos/create"))"#,
        r#"let t = format!("{base}/api/models/{repo}/xet-write-token/main");"#,
        r#"let u = format!("{b}/api/models/{r}/commit/{rev}");"#,
        "const BASE: &str = \"https://huggingface.co\";\nlet u = format!(\"{BASE}/x/{r}/branch/{b}\");",
        "const API: &str = \"/api/spaces/\";\nfn t(r: &str) -> String { format!(\"{r}/tag/v1\") }",
        r#"agent.post(&format!("{}/api/datasets/{}/upload/main", b, r))"#,
        r#"let u = "https://hf.co/x/commit/main";"#,
    ];
    let must_not_hit = [
        r#"// let u = format!("{b}/info/lfs/objects/batch");"#,
        r#"    /// POSTs "{base}/api/models/{repo}/commit/{rev}""#,
        r#".map_err(|e| hub_err("preupload", e))"#,
        r#"let u = format!("https://api.github.com/repos/{o}/{r}/git/commit/{sha}");"#,
        r#"let u = format!("https://huggingface.co/{repo}/resolve/main/{file}");"#,
        "let h = \"https://huggingface.co\";\nlet t = format!(\"refs/tags/{t}\");",
        "let c = \"commit/\"; // an anchor in a comment: huggingface.co",
        "",
    ];
    for src in must_hit {
        assert!(!hits(src).is_empty(), "must hit: {src}");
    }
    for src in must_not_hit {
        assert_eq!(hits(src), Vec::<String>::new(), "must not hit: {src}");
    }
    assert_eq!(
        hits("let a = \"x/preupload/y\";\nlet b = \"z/info/lfs/objects/batch\";"),
        ["1:x/preupload/y", "2:z/info/lfs/objects/batch"]
    );
}

/// The tree as built: no HF write endpoint outside aprender-core `hf_hub`.
#[test]
fn apr_has_one_hf_upload_path() {
    let (counts, rows) = scan(&workspace_root());
    assert!(
        counts[0] > 100 && counts[1] > 0,
        "the scan saw too few files: {counts:?}"
    );
    assert!(
        rows.is_empty(),
        "a second HF upload path (#4961): move it onto aprender-core hf_hub::repo_api\n{}",
        rows.join("\n")
    );
}

/// Planted: the one path's own code copied into apr-cli or the root `src/` is a
/// second path, RED; in aprender-core it is out of scope.
#[test]
fn a_planted_second_path_is_red() {
    let repo_api = include_str!("../../../aprender-core/src/hf_hub/repo_api.rs");
    let xet = include_str!("../../../aprender-core/src/hf_hub/xet.rs");
    let plant = |files: &[(&str, &str)]| {
        let t = TempDir::new().expect("tmp");
        for scope in SCOPES {
            std::fs::create_dir_all(t.path().join(scope)).expect("mkdir");
        }
        for (rel, body) in files {
            let p = t.path().join(rel);
            std::fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
            std::fs::write(p, body).expect("write");
        }
        scan(t.path()).1
    };
    let core = [
        ("crates/aprender-core/src/hf_hub/repo_api.rs", repo_api),
        ("crates/aprender-core/src/hf_hub/xet.rs", xet),
        ("crates/apr-cli/src/lib.rs", "pub mod commands;"),
    ];
    assert_eq!(plant(&core), Vec::<String>::new());
    for (rel, body) in [
        ("crates/apr-cli/src/commands/second_path.rs", repo_api),
        ("crates/apr-cli/tests/xet_upload.rs", xet),
        ("src/bin/upload.rs", repo_api),
    ] {
        let rows = plant(&[core[0], core[1], (rel, body)]);
        assert_eq!(rows.len(), 1, "{rel}: {rows:?}");
        assert!(rows[0].starts_with(&format!("{rel} ")), "{rows:?}");
    }
}

/// The skipped file is this guard, and it holds no HTTP client.
#[test]
fn the_one_skip_is_this_guard() {
    let me = workspace_root().join(SELF);
    let src = std::fs::read_to_string(&me).unwrap_or_else(|e| panic!("{}: {e}", me.display()));
    assert!(src.starts_with("//! #4961: `apr` has ONE Hugging Face upload path"));
    assert!(!hits(&src).is_empty(), "the skip must be load-bearing");
    for client in [
        concat!("ur", "eq"),
        concat!("req", "west"),
        concat!("hyp", "er::"),
        concat!("std::", "net"),
        concat!("Repo", "Api"),
        concat!("HfHub", "Client"),
    ] {
        assert!(!src.contains(client), "the guard holds {client}");
    }
}
