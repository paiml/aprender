//! #4961: `apr` has ONE Hugging Face upload path, aprender-core `hf_hub`, whose
//! `repo_api` holds the Hub write endpoints. This scan is RED when a write
//! endpoint is named anywhere in `crates/apr-cli/` or the root `src/`, the code
//! that builds `apr`. It reads the string literals (plain, byte and raw, across
//! lines, a `\` line continuation joined) outside the comments of every `.rs`
//! file there:
//!
//! - an upload-only endpoint (`info/lfs/objects/batch`, `preupload/`,
//!   `xet-write-token`, `api/repos/create`) is a hit by itself;
//! - a commit, branch, tag or upload endpoint (`commit/`, `branch/`, `tag/`,
//!   `/upload/`) is a hit unless the same literal's host is a known non-Hub one
//!   (`github.com`, `gitlab.com`, or a subdomain) and it names no Hub API path
//!   (`api/models`, `api/datasets`, `api/spaces`). Any other host, or none in
//!   sight, is taken as the Hub: its base usually comes from a constant or
//!   another module, and `HF_ENDPOINT` may name a mirror.
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
const NOT_HUB: [&str; 2] = ["github.com", "gitlab.com"];
const HUB_PATHS: [&str; 3] = ["api/models", "api/datasets", "api/spaces"];
const WRITE: [&str; 4] = ["commit/", "branch/", "tag/", "/upload/"];
const SCOPES: [&str; 2] = ["crates/apr-cli", "src"];
const SELF: &str = "crates/apr-cli/src/commands/hf_one_path_guard.rs";

/// A Rust lexer as far as this scan needs: where each string literal starts
/// and ends, and what is a comment, a char literal or a lifetime.
struct Lex<'a> {
    src: &'a str,
    cs: Vec<(usize, char)>,
    line: usize,
}

impl Lex<'_> {
    fn at(&self, j: usize) -> Option<char> {
        self.cs.get(j).map(|p| p.1)
    }

    /// The source text from char `a` up to char `b`.
    fn text(&self, a: usize, b: usize) -> String {
        let off = |k: usize| self.cs.get(k).map_or(self.src.len(), |p| p.0);
        self.src[off(a)..off(b)].to_string()
    }

    /// The token at `j`: its text if it is a string literal, and the index past it.
    fn token(&mut self, j: usize) -> (Option<String>, usize) {
        match (self.at(j), self.at(j + 1)) {
            (Some('\n'), _) => {
                self.line += 1;
                (None, j + 1)
            }
            (Some('/'), Some('/')) => (None, self.eol(j)),
            (Some('/'), Some('*')) => (None, self.block_comment(j)),
            (Some('\''), _) => (None, self.quote(j)),
            (Some('"'), _) => {
                let (s, k) = self.plain(j);
                (Some(s), k)
            }
            (Some('r'), Some('"' | '#')) => self.raw(j),
            _ => (None, j + 1),
        }
    }

    /// The index of the newline that ends the line comment at `j`.
    fn eol(&self, j: usize) -> usize {
        (j..self.cs.len())
            .find(|&k| self.at(k) == Some('\n'))
            .unwrap_or(self.cs.len())
    }

    /// Past the `/* */` comment at `j`, nested ones included.
    fn block_comment(&mut self, mut j: usize) -> usize {
        let mut depth = 0;
        while let Some(c) = self.at(j) {
            j += match (c, self.at(j + 1)) {
                ('/', Some('*')) => {
                    depth += 1;
                    2
                }
                ('*', Some('/')) if depth == 1 => return j + 2,
                ('*', Some('/')) => {
                    depth -= 1;
                    2
                }
                _ => {
                    self.line += usize::from(c == '\n');
                    1
                }
            };
        }
        j
    }

    /// Past the char or byte literal at `j` (`'"'`, `'\''`, `b'\\'`), or past
    /// the `'` of a lifetime or label.
    fn quote(&self, j: usize) -> usize {
        match (self.at(j + 1), self.at(j + 2)) {
            (Some('\\'), _) => (j + 3..j + 12)
                .find(|&k| self.at(k) == Some('\''))
                .map_or(j + 1, |k| k + 1),
            (Some(c), Some('\'')) if c != '\n' => j + 3,
            _ => j + 1,
        }
    }

    /// The plain or byte string at `j` (its `"`) as far as the scan needs: a
    /// `\` line continuation drops the newline and the next line's indent, as
    /// Rust does; other escapes stay as written. An unclosed one runs to the end.
    fn plain(&mut self, j: usize) -> (String, usize) {
        let (mut s, mut k) = (String::new(), j + 1);
        while let Some(c) = self.at(k) {
            match (c, self.at(k + 1)) {
                ('"', _) => return (s, k + 1),
                ('\\', Some('\n' | '\r')) => k = self.indent_end(k + 1),
                ('\\', Some(e)) => {
                    s.push(c);
                    s.push(e);
                    k += 2;
                }
                _ => {
                    self.line += usize::from(c == '\n');
                    s.push(c);
                    k += 1;
                }
            }
        }
        (s, k)
    }

    /// Past the line break at `k` and the whitespace after it.
    fn indent_end(&mut self, mut k: usize) -> usize {
        while let Some(c) = self.at(k).filter(|c| c.is_whitespace()) {
            self.line += usize::from(c == '\n');
            k += 1;
        }
        k
    }

    /// The raw string at `j` (the `r` of `r"…"`, `br#"…"#`), or nothing and
    /// `j + 1` for a raw identifier (`r#type`).
    fn raw(&mut self, j: usize) -> (Option<String>, usize) {
        let hashes = (j + 1..self.cs.len())
            .take_while(|&k| self.at(k) == Some('#'))
            .count();
        let open = j + 1 + hashes;
        if self.at(open) != Some('"') {
            return (None, j + 1);
        }
        let closes = |k: &usize| {
            self.at(*k) == Some('"') && (1..=hashes).all(|h| self.at(k + h) == Some('#'))
        };
        let close = (open + 1..self.cs.len())
            .find(closes)
            .unwrap_or(self.cs.len());
        let s = self.text(open + 1, close);
        self.line += s.matches('\n').count();
        (Some(s), (close + 1 + hashes).min(self.cs.len()))
    }
}

/// The string literals of `src` (plain, byte and raw, which may run across
/// lines), each with the line it opens on; comments are stepped over.
fn literals(src: &str) -> Vec<(usize, String)> {
    let mut lx = Lex {
        src,
        cs: src.char_indices().collect(),
        line: 1,
    };
    let (mut out, mut j) = (Vec::new(), 0);
    while j < lx.cs.len() {
        let line = lx.line;
        let (lit, next) = lx.token(j);
        out.extend(lit.map(|s| (line, s)));
        j = next;
    }
    out
}

fn names_any(s: &str, set: &[&str]) -> bool {
    set.iter().any(|n| s.contains(n))
}

/// A literal whose host is a known non-Hub one, or a subdomain of one, and
/// that names no Hub API path.
fn other_host(s: &str) -> bool {
    let host = s
        .split("://")
        .nth(1)
        .and_then(|r| r.split(['/', '{', ':']).next());
    let known = |h: &str| {
        NOT_HUB
            .iter()
            .any(|n| h == *n || h.strip_suffix(n).is_some_and(|p| p.ends_with('.')))
    };
    host.is_some_and(known) && !names_any(s, &HUB_PATHS)
}

/// The hits in one file's text, as `line:literal`.
fn hits(src: &str) -> Vec<String> {
    literals(src)
        .into_iter()
        .filter(|(_, s)| names_any(s, &UPLOAD_ONLY) || (names_any(s, &WRITE) && !other_host(s)))
        .map(|(n, s)| format!("{n}:{s}"))
        .collect()
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
        r#"let u = format!("{b}/x/commit/{rev}");"#,
        r#"let u = format!("{}/{}/commit/{}", DEFAULT_HF_ENDPOINT, repo, rev);"#,
        r#"let u = format!("https://hub.example.org/api/models/{r}/commit/main");"#,
        r#"let b = "say \"hi"; let u = format!("{b}/x/preupload/{r}");"#,
        r#"let q = '"'; let u = format!("{b}/x/commit/{r}");"#,
        r#"let q = '\"'; let u = format!("{b}/x/preupload/{r}");"#,
        r#"let u = format!("https://hf-mirror.com/{r}/commit/{rev}");"#,
        "let c = \"commit/\"; // a Hub name in a comment does not matter",
        "let u = format!(\"{base}/api/models/{repo}/\\\n    commit/{rev}\");",
        "let u = format!(\"{b}/x/info/lfs/\\\n    objects/batch\");",
        r##"let u = format!(r#"a " {b}/x/commit/{r}"#);"##,
        "let p = r\"C:\\\";\nlet u = format!(\"{b}/x/commit/{r}\");",
        "/* it's \"odd */ let u = format!(\"{b}/x/commit/{r}\");",
        "fn f<'a>(b: &'a str) -> String { format!(\"{b}/x/commit/{r}\") }",
        r#"let u = format!("https://github.com@hub.example/{r}/commit/{rev}");"#,
        r#"let u = format!("https://notgithub.com/{r}/commit/{rev}");"#,
    ];
    let must_not_hit = [
        r#"// let u = format!("{b}/info/lfs/objects/batch");"#,
        r#"    /// POSTs "{base}/api/models/{repo}/commit/{rev}""#,
        r#".map_err(|e| hub_err("preupload", e))"#,
        r#"let u = format!("https://api.github.com/repos/{o}/{r}/git/commit/{sha}");"#,
        r#"let u = format!("https://uploads.github.com/repos/{o}/{r}/releases/{id}/upload/");"#,
        r#"let u = format!("https://huggingface.co/{repo}/resolve/main/{file}");"#,
        "let h = \"https://huggingface.co\";\nlet t = format!(\"refs/tags/{t}\");",
        r#"let s = "a \" b"; // commit/ is not in a literal"#,
        "let x = 1; // format!(\"{b}/x/commit/{r}\")",
        "/* /* */ \"{b}/x/commit/{r}\" */",
        r#"let u = r"https://api.github.com/repos/{o}/{r}/git/commit/{sha}";"#,
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
    assert_eq!(
        hits("let u = format!(\"{b}/x/\\\n    preupload/{r}\");\nlet v = \"a\nb/tag/c\";"),
        ["1:{b}/x/preupload/{r}", "3:a\nb/tag/c"]
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
