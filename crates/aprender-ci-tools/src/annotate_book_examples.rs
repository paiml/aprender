//! `annotate-book-examples [ROOT]` (was `scripts/annotate-book-examples.py`, #4822): insert
//! an `<!-- example-cost: ... -->` line above every bash/rust fence in
//! `ROOT/book/src/{cli,lib}/*.md` that has none, and rewrite each chapter.
//!
//! As the original: every listed file is rewritten even when nothing changed (text-mode
//! read and write, so `\r\n` and a lone `\r` become `\n`, and every other break
//! `str.splitlines` knows does too); the first unreadable, non-UTF-8 or unwritable file
//! stops the run with exit 1 after the lines already printed.

use crate::pystr::{is_py_space, py_splitlines, py_strip};
use std::path::Path;

/// The model a `model-required` fence names when its command line names none.
pub const DEFAULT_MODEL: &str = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

/// Inference-style commands needing a model in `~/models/`.
const MODEL_REQUIRED_CMDS: &[&str] = &[
    "run",
    "chat",
    "serve",
    "qa",
    "bench",
    "eval",
    "inspect",
    "validate",
    "lint",
    "tensors",
    "trace",
    "debug",
    "diff",
    "explain",
    "flow",
    "hex",
    "profile",
    "tokenize",
    "tree",
    "check",
    "canary",
    "qualify",
    "tune",
    "finetune",
    "distill",
    "quantize",
    "prune",
    "compile",
    "merge",
    "compare-hf",
    "parity",
    "import",
    "convert",
    "export",
    "ptx-map",
    "pull",
];
/// Mutating / destructive commands (CI substitutes `--dry-run`).
const DESTRUCTIVE_CMDS: &[&str] = &["publish", "encrypt", "decrypt", "rm", "upload", "stamp"];
/// Interactive REPLs / TUIs that cannot run non-interactively.
const INTERACTIVE_CMDS: &[&str] = &["tui", "cbtop", "monitor", "rosetta", "showcase", "code"];
/// CUDA-specific tooling.
const GPU_CMDS: &[&str] = &["gpu", "ptx"];

/// `FENCE_RE = ^```(bash|rust)\s*$`: the fence's language, if `line` opens one.
fn fence_lang(line: &str) -> Option<&'static str> {
    ["bash", "rust"].into_iter().find(|lang| {
        line.strip_prefix("```")
            .and_then(|r| r.strip_prefix(lang))
            .is_some_and(|rest| rest.chars().all(is_py_space))
    })
}

/// `COST_RE = ^<!--\s*example-cost:\s*([^>]+?)\s*-->\s*$`, matched against a stripped line.
/// With the trailing `\s*` empty, it holds iff the text between `example-cost:` and the
/// final `-->` is non-empty and holds no `>` (`[^>]` also takes the spaces either side).
fn is_cost_line(s: &str) -> bool {
    s.strip_prefix("<!--")
        .map(|r| r.trim_start_matches(is_py_space))
        .and_then(|r| r.strip_prefix("example-cost:"))
        .and_then(|r| r.strip_suffix("-->"))
        .is_some_and(|between| !between.is_empty() && !between.contains('>'))
}

/// `^apr\s+([a-z][a-z0-9-]*)` on a first line: the subcommand, if it is one.
fn apr_subcommand(first_line: &str) -> Option<&str> {
    let after = first_line
        .strip_prefix("apr")
        .filter(|r| r.starts_with(is_py_space))?
        .trim_start_matches(is_py_space);
    if !after.starts_with(|c: char| c.is_ascii_lowercase()) {
        return None;
    }
    let end = after
        .find(|c: char| !(c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'))
        .unwrap_or(after.len());
    Some(&after[..end])
}

/// `first_line.split()[2:]` (skip `apr <cmd>`): the first positional that looks like a model
/// path or HF id, else the default.
fn model_named(first_line: &str) -> &str {
    first_line
        .split(is_py_space)
        .filter(|t| !t.is_empty())
        .skip(2)
        .filter(|t| !t.starts_with('-'))
        .find(|t| {
            [".gguf", ".apr", ".safetensors"]
                .iter()
                .any(|e| t.ends_with(e))
                || ["qwen", "Qwen", "hf://"].iter().any(|p| t.starts_with(p))
        })
        .unwrap_or(DEFAULT_MODEL)
}

/// `classify_apr_command`: the cost class of a bash fence's body, and the model a
/// `model-required` one needs.
///
/// The original's `apr help <command>` check is left out: a first line it would call
/// trivial starts `apr<space>help`, so the subcommand read below starts with `help`, which
/// no list names, and that is trivial too.
pub fn classify_bash(code: &str) -> (&'static str, Option<String>) {
    let stripped = py_strip(code);
    let stripped = stripped.strip_prefix("$ ").unwrap_or(stripped);
    let first_line = py_strip(stripped.split('\n').next().unwrap_or_default());
    if first_line.contains("--help") || first_line.contains("--version") {
        return ("trivial", None);
    }
    let Some(cmd) = apr_subcommand(first_line) else {
        return ("trivial", None);
    };
    if GPU_CMDS.contains(&cmd) {
        ("gpu", None)
    } else if INTERACTIVE_CMDS.contains(&cmd) {
        ("interactive", None)
    } else if DESTRUCTIVE_CMDS.contains(&cmd) {
        ("destructive", None)
    } else if MODEL_REQUIRED_CMDS.contains(&cmd) {
        ("model-required", Some(model_named(first_line).to_owned()))
    } else {
        ("trivial", None)
    }
}

/// Whether the last non-blank line of the (up to) two already written is a cost line.
fn already_annotated(out: &[String]) -> bool {
    for back in 1..=2 {
        let Some(prev) = out.len().checked_sub(back).map(|j| py_strip(&out[j])) else {
            return false;
        };
        if !prev.is_empty() {
            return is_cost_line(prev);
        }
    }
    false
}

/// The cost line for the fence at `lines[i]`; a bash fence is classified by its body (up to
/// the next line starting with three backticks, or the end).
fn annotation(lang: &str, lines: &[&str], i: usize) -> String {
    let (cost, model) = if lang == "bash" {
        let body: Vec<&str> = lines[i + 1..]
            .iter()
            .take_while(|l| !l.starts_with("```"))
            .copied()
            .collect();
        classify_bash(&body.join("\n"))
    } else {
        ("trivial", None)
    };
    match model {
        Some(m) => format!("<!-- example-cost: {cost} model: {m} -->"),
        None => format!("<!-- example-cost: {cost} -->"),
    }
}

/// `annotate_file` on text already read in text mode: `(new text, annotated, skipped)`.
pub fn annotate_text(text: &str) -> (String, usize, usize) {
    let lines = py_splitlines(text);
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let (mut annotated, mut skipped) = (0, 0);
    for (i, line) in lines.iter().enumerate() {
        match fence_lang(line) {
            None => {}
            Some(_) if already_annotated(&out) => skipped += 1,
            Some(lang) => {
                if out.last().is_some_and(|l| !py_strip(l).is_empty()) {
                    out.push(String::new());
                }
                out.push(annotation(lang, &lines, i));
                annotated += 1;
            }
        }
        out.push((*line).to_owned());
    }
    let mut new_text = out.join("\n");
    if text.ends_with('\n') && !new_text.ends_with('\n') {
        new_text.push('\n');
    }
    (new_text, annotated, skipped)
}

/// `sorted(d.glob("*.md"))` for `ROOT/book/src/<sub>` when it is a directory: every entry
/// whose name ends `.md` (dot-files, directories and dangling links included), in code
/// point order. An unlistable directory lists nothing, as pathlib's glob swallows the error.
fn chapters(root: &Path, sub: &str) -> Result<Vec<String>, String> {
    let d = root.join("book").join("src").join(sub);
    if !d.is_dir() {
        return Ok(Vec::new());
    }
    let Ok(entries) = std::fs::read_dir(&d) else {
        return Ok(Vec::new());
    };
    let mut names = Vec::new();
    for e in entries.flatten() {
        let name = e.file_name();
        let Some(name) = name.to_str() else {
            return Err(format!(
                "{}: a file name that is not UTF-8: {}",
                d.display(),
                name.to_string_lossy()
            ));
        };
        if name.ends_with(".md") {
            names.push(format!("book/src/{sub}/{name}"));
        }
    }
    names.sort();
    Ok(names)
}

/// Annotate one chapter in place: `(annotated, skipped)`.
fn annotate_file(path: &Path) -> Result<(usize, usize), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let text = String::from_utf8(bytes)
        .map_err(|e| format!("{}: not UTF-8: {e}", path.display()))?
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    let (new_text, annotated, skipped) = annotate_text(&text);
    std::fs::write(path, new_text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((annotated, skipped))
}

/// The whole run: stdout, or `(already printed, reason)`.
pub fn run(root: &Path) -> Result<String, (String, String)> {
    let mut files = chapters(root, "cli").map_err(|r| (String::new(), r))?;
    files.extend(chapters(root, "lib").map_err(|r| (String::new(), r))?);
    let mut printed = String::new();
    let (mut total_annotated, mut total_skipped) = (0, 0);
    for rel in &files {
        let (a, s) = annotate_file(&root.join(rel)).map_err(|r| (printed.clone(), r))?;
        total_annotated += a;
        total_skipped += s;
        if a > 0 {
            printed.push_str(&format!("  {rel}: +{a} annotation(s)\n"));
        }
    }
    printed.push_str(&format!(
        "\nAnnotated {total_annotated} fence(s) across {} chapter(s); skipped {total_skipped} already-annotated fence(s).\n",
        files.len()
    ));
    Ok(printed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch tree, removed on drop. Tests run in parallel, so each names its own.
    struct Tree(std::path::PathBuf);
    impl Tree {
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for Tree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tree(tag: &str, files: &[(&str, &[u8])]) -> Tree {
        let t = Tree(
            std::env::temp_dir().join(format!("ci-tools-annotate-{tag}-{}", std::process::id())),
        );
        let _ = std::fs::remove_dir_all(t.path());
        std::fs::create_dir_all(t.path()).expect("mkdir");
        for (rel, body) in files {
            let p = t.path().join(rel);
            std::fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
            std::fs::write(p, body).expect("write");
        }
        t
    }

    #[test]
    fn classifies_by_subcommand() {
        for (code, cost, model) in [
            ("apr gpu status", "gpu", None),
            ("apr ptx-map m.gguf", "model-required", Some("m.gguf")),
            ("apr ptx x", "gpu", None),
            ("apr code", "interactive", None),
            ("apr tui", "interactive", None),
            ("apr publish x", "destructive", None),
            ("apr rm m", "destructive", None),
            ("apr run --help", "trivial", None),
            ("apr serve --version", "trivial", None),
            ("apr help run", "trivial", None),
            ("apr helpx", "trivial", None),
            ("apr frobnicate", "trivial", None),
            ("ls -la", "trivial", None),
            ("aprun x", "trivial", None),
            ("apr", "trivial", None),
            ("apr Run m.gguf", "trivial", None),
            ("apr 9run", "trivial", None),
            ("", "trivial", None),
            ("apr run", "model-required", Some(DEFAULT_MODEL)),
            ("apr run -m x.gguf", "model-required", Some("x.gguf")),
            ("apr run --x a.apr b.gguf", "model-required", Some("a.apr")),
            (
                "apr qa m.safetensors",
                "model-required",
                Some("m.safetensors"),
            ),
            (
                "apr pull hf://Qwen/x",
                "model-required",
                Some("hf://Qwen/x"),
            ),
            ("apr pull qwen2", "model-required", Some("qwen2")),
            ("apr pull Qwen2", "model-required", Some("Qwen2")),
            ("apr pull other", "model-required", Some(DEFAULT_MODEL)),
            ("apr run-x m.gguf", "trivial", None),
            ("apr run:x m.gguf", "model-required", Some("m.gguf")),
            ("apr run.gguf", "model-required", Some(DEFAULT_MODEL)),
            ("$ apr chat m.gguf", "model-required", Some("m.gguf")),
            ("  \n$ apr tui\napr run", "interactive", None),
            ("apr run\nm.gguf", "model-required", Some(DEFAULT_MODEL)),
            ("apr\u{3000}run\x1fm.gguf", "model-required", Some("m.gguf")),
            ("apr\x1crun", "model-required", Some(DEFAULT_MODEL)),
            ("echo x\napr gpu", "trivial", None),
        ] {
            assert_eq!(
                classify_bash(code),
                (cost, model.map(str::to_owned)),
                "classify({code:?})"
            );
        }
    }

    #[test]
    fn fence_and_cost_line_match_the_regexes() {
        assert_eq!(fence_lang("```bash"), Some("bash"));
        assert_eq!(fence_lang("```rust \t\x1f\u{a0}"), Some("rust"));
        assert_eq!(fence_lang("```bashx"), None);
        assert_eq!(fence_lang(" ```bash"), None);
        assert_eq!(fence_lang("```"), None);
        assert_eq!(fence_lang("```Bash"), None);
        assert_eq!(fence_lang("``` bash"), None);
        for (s, want) in [
            ("<!-- example-cost: trivial -->", true),
            ("<!--example-cost:x-->", true),
            ("<!-- \x1c example-cost: -->", true),
            ("<!--example-cost:-->", false),
            ("<!-- example-cost: a>b -->", false),
            ("<!-- example-cost: x --> y", false),
            ("<!-- cost: x -->", false),
            ("<!- example-cost: x -->", false),
            ("x<!-- example-cost: x -->", false),
            ("<!-- example-cost: x ->", false),
        ] {
            assert_eq!(is_cost_line(s), want, "cost({s:?})");
        }
    }

    #[test]
    fn annotates_inserting_a_blank_before_text() {
        let (t, a, s) =
            annotate_text("# T\nPara\n```bash\napr run m.gguf\n```\n\n```rust\nfn f(){}\n```\n");
        assert_eq!((a, s), (2, 0));
        assert_eq!(
            t,
            "# T\nPara\n\n<!-- example-cost: model-required model: m.gguf -->\n```bash\napr run m.gguf\n```\n\n<!-- example-cost: trivial -->\n```rust\nfn f(){}\n```\n"
        );
    }

    #[test]
    fn is_idempotent_and_looks_back_two_lines() {
        let once = annotate_text("x\n```bash\napr tui\n```\n").0;
        assert_eq!(annotate_text(&once), (once.clone(), 0, 1));
        // A blank between the annotation and the fence still counts; two blanks do not.
        assert_eq!(
            annotate_text("<!-- example-cost: gpu -->\n\n```bash\n```").2,
            1
        );
        let (t, a, _) = annotate_text("<!-- example-cost: gpu -->\n\n\n```bash\n```");
        assert_eq!(
            (t.as_str(), a),
            (
                "<!-- example-cost: gpu -->\n\n\n<!-- example-cost: trivial -->\n```bash\n```",
                1
            )
        );
        // A non-blank, non-annotation line right above stops the look-back.
        assert_eq!(
            annotate_text("<!-- example-cost: gpu -->\ntext\n```rust\n```").1,
            1
        );
        // A fence on the first line: nothing above, no blank inserted.
        assert_eq!(
            annotate_text("```rust\n```").0,
            "<!-- example-cost: trivial -->\n```rust\n```"
        );
    }

    #[test]
    fn a_closing_fence_that_names_a_language_opens_another() {
        let (t, a, s) = annotate_text("```bash\napr gpu\n```rust\n");
        assert_eq!((a, s), (2, 0));
        assert_eq!(
            t,
            "<!-- example-cost: gpu -->\n```bash\napr gpu\n\n<!-- example-cost: trivial -->\n```rust\n"
        );
    }

    #[test]
    fn the_body_ends_at_any_backtick_fence_or_eof() {
        assert_eq!(
            annotate_text("```bash\n```\napr gpu\n").0,
            "<!-- example-cost: trivial -->\n```bash\n```\napr gpu\n"
        );
        assert_eq!(
            annotate_text("```bash\napr code").0,
            "<!-- example-cost: interactive -->\n```bash\napr code"
        );
    }

    #[test]
    fn line_breaks_and_trailing_newlines_follow_splitlines() {
        // A trailing blank line is lost (splitlines + join), as the original loses it.
        assert_eq!(annotate_text("a\n\n").0, "a\n");
        assert_eq!(annotate_text("a\x0cb\u{2028}c").0, "a\nb\nc");
        assert_eq!(annotate_text("a\x0c").0, "a");
        assert_eq!(annotate_text("").0, "");
        assert_eq!(annotate_text("\n").0, "\n");
    }

    #[test]
    fn run_rewrites_every_chapter_in_order() {
        let t = tree(
            "order",
            &[
                ("book/src/cli/b.md", b"```bash\napr run\n```\r\n"),
                ("book/src/cli/a.md", b"no fences\r"),
                (
                    "book/src/cli/A.md",
                    b"<!-- example-cost: x -->\n```rust\n```",
                ),
                ("book/src/cli/c.MD", b"```rust\n"),
                ("book/src/cli/.h.md", b"```rust\n"),
                ("book/src/lib/z.md", b"```rust\n```rust\n"),
                ("book/src/other/o.md", b"```rust\n"),
            ],
        );
        let out = run(t.path()).expect("run");
        assert_eq!(
            out,
            "  book/src/cli/.h.md: +1 annotation(s)\n  book/src/cli/b.md: +1 annotation(s)\n  book/src/lib/z.md: +2 annotation(s)\n\nAnnotated 4 fence(s) across 5 chapter(s); skipped 1 already-annotated fence(s).\n"
        );
        let read = |r: &str| std::fs::read_to_string(t.path().join(r)).expect("read");
        assert_eq!(read("book/src/cli/a.md"), "no fences\n");
        assert_eq!(
            read("book/src/cli/b.md"),
            format!("<!-- example-cost: model-required model: {DEFAULT_MODEL} -->\n```bash\napr run\n```\n")
        );
        assert_eq!(read("book/src/cli/c.MD"), "```rust\n");
        assert_eq!(read("book/src/other/o.md"), "```rust\n");
    }

    #[test]
    fn run_with_no_book_lists_nothing() {
        let t = tree("nobook", &[("book/src/cli", b"a file, not a dir")]);
        assert_eq!(
            run(t.path()).expect("run"),
            "\nAnnotated 0 fence(s) across 0 chapter(s); skipped 0 already-annotated fence(s).\n"
        );
    }

    #[test]
    fn run_stops_at_the_first_bad_chapter_after_printing() {
        let t = tree(
            "stop",
            &[
                ("book/src/cli/a.md", b"```rust\n"),
                ("book/src/cli/b.md", b"bad \xff utf-8\n"),
                ("book/src/cli/c.md", b"```rust\n"),
            ],
        );
        let (printed, reason) = run(t.path()).expect_err("b.md is not UTF-8");
        assert_eq!(printed, "  book/src/cli/a.md: +1 annotation(s)\n");
        assert!(reason.contains("b.md: not UTF-8"), "{reason}");
        let c = std::fs::read_to_string(t.path().join("book/src/cli/c.md")).expect("read");
        assert_eq!(c, "```rust\n", "c.md is not reached");

        let d = tree("dir", &[("book/src/lib/d.md/x", b"")]);
        let (printed, reason) = run(d.path()).expect_err("d.md is a directory");
        assert_eq!(printed, "");
        assert!(reason.contains("d.md"), "{reason}");
    }

    #[cfg(unix)]
    #[test]
    fn run_refuses_a_non_utf8_name_before_touching_anything() {
        use std::os::unix::ffi::OsStrExt;
        let t = tree("badname", &[("book/src/cli/a.md", b"```rust\n")]);
        let bad = std::ffi::OsStr::from_bytes(b"\xff.md");
        std::fs::write(t.path().join("book/src/cli").join(bad), b"").expect("write");
        let (printed, reason) = run(t.path()).expect_err("non-UTF-8 name");
        assert_eq!(printed, "");
        assert!(reason.contains("not UTF-8"), "{reason}");
        let a = std::fs::read_to_string(t.path().join("book/src/cli/a.md")).expect("read");
        assert_eq!(a, "```rust\n");
    }
}
