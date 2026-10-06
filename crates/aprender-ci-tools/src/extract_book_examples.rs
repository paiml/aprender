//! `extract-book-examples [ROOT]`: one JSON line per ```` ```bash ```` / ```` ```rust ```` block
//! in `ROOT/book/src/{cli,lib}/*.md` (was scripts/extract_book_examples.py).
//!
//! A record is `{"path", "line_start", "line_end", "lang", "cost", "code"}`, plus `"model"`
//! when the block's cost line names one, written as `json.dumps` writes it (`, ` and `: `,
//! everything outside `' '..='~'` escaped). The cost is read from the last non-blank line of
//! the two above the opening fence; without a cost line it is `trivial`.

use crate::pystr::{is_py_space, py_splitlines, py_strip};
use std::path::Path;

const VALID_COSTS: [&str; 6] = [
    "trivial",
    "model-required",
    "gpu",
    "destructive",
    "interactive",
    "skip",
];

/// `FENCE_RE`, `^```(bash|rust)\s*$`: the language of an opening fence.
fn fence_lang(line: &str) -> Option<&'static str> {
    let rest = line.strip_prefix("```")?;
    ["bash", "rust"].into_iter().find(|lang| {
        rest.strip_prefix(lang)
            .is_some_and(|r| r.chars().all(is_py_space))
    })
}

/// `CLOSE_RE`, `^```\s*$`.
fn is_close(line: &str) -> bool {
    line.strip_prefix("```")
        .is_some_and(|r| r.chars().all(is_py_space))
}

/// `COST_RE`, `^<!--\s*example-cost:\s*([^>]+?)\s*-->\s*$`, on a stripped line: after
/// `<!--`, whitespace and `example-cost:`, the text before the final `-->` must be non-empty
/// and hold no `>`. Returns that text stripped, which is `group(1).strip()`.
fn cost_payload(cand: &str) -> Option<&str> {
    let between = cand
        .strip_prefix("<!--")?
        .trim_start_matches(is_py_space)
        .strip_prefix("example-cost:")?
        .strip_suffix("-->")?;
    (!between.is_empty() && !between.contains('>')).then(|| py_strip(between))
}

/// `parse_cost_annotation` on a matched payload: the first word, `trivial` unless it is a
/// valid cost, and the word after the first `model:`. An empty payload (a cost line of
/// whitespace only) is the original's `parts[0]` IndexError.
pub fn parse_cost(payload: &str) -> Result<(&'static str, Option<String>), String> {
    let parts: Vec<&str> = payload
        .split(is_py_space)
        .filter(|t| !t.is_empty())
        .collect();
    let Some(first) = parts.first() else {
        return Err("a cost line with no cost class".to_owned());
    };
    let cost = VALID_COSTS
        .into_iter()
        .find(|c| c == first)
        .unwrap_or("trivial");
    let model = parts
        .iter()
        .position(|t| *t == "model:")
        .and_then(|i| parts.get(i + 1))
        .map(|m| (*m).to_owned());
    Ok((cost, model))
}

/// The cost of the fence at `lines[i]`: the last non-blank line of the two above decides.
fn cost_above(lines: &[&str], i: usize) -> Result<(&'static str, Option<String>), String> {
    for back in 1..=2 {
        let Some(j) = i.checked_sub(back) else {
            break;
        };
        let cand = py_strip(lines[j]);
        if cand.is_empty() {
            continue;
        }
        if let Some(payload) = cost_payload(cand) {
            return parse_cost(payload);
        }
        break;
    }
    Ok(("trivial", None))
}

/// `json.dumps` of a string given as code points (a lone surrogate allowed, for a file name
/// decoded with `surrogateescape`).
fn json_str(cps: impl IntoIterator<Item = u32>) -> String {
    let mut out = String::from('"');
    for cp in cps {
        match cp {
            0x22 => out.push_str("\\\""),
            0x5c => out.push_str("\\\\"),
            0x0a => out.push_str("\\n"),
            0x0d => out.push_str("\\r"),
            0x09 => out.push_str("\\t"),
            0x08 => out.push_str("\\b"),
            0x0c => out.push_str("\\f"),
            0x20..=0x7e => out.push(char::from(cp as u8)),
            0x1_0000.. => {
                let v = cp - 0x1_0000;
                let (hi, lo) = (0xd800 + (v >> 10), 0xdc00 + (v & 0x3ff));
                out.push_str(&format!("\\u{hi:04x}\\u{lo:04x}"));
            }
            _ => out.push_str(&format!("\\u{cp:04x}")),
        }
    }
    out.push('"');
    out
}

/// The JSON lines for one chapter's text (already read in text mode); `rel` is its path as
/// code points.
pub fn extract_text(rel: &[u32], text: &str) -> Result<String, String> {
    let lines = py_splitlines(text);
    let path = json_str(rel.iter().copied());
    let mut out = String::new();
    let mut i = 0;
    while i < lines.len() {
        let Some(lang) = fence_lang(lines[i]) else {
            i += 1;
            continue;
        };
        let (cost, model) = cost_above(&lines, i)?;
        let body_len = lines[i + 1..].iter().take_while(|l| !is_close(l)).count();
        let k = i + 1 + body_len;
        if k >= lines.len() {
            // An unclosed fence ends the chapter.
            break;
        }
        let code = lines[i + 1..k].join("\n");
        out.push_str(&format!(
            "{{\"path\": {path}, \"line_start\": {}, \"line_end\": {}, \"lang\": \"{lang}\", \"cost\": \"{cost}\", \"code\": {}",
            i + 1,
            k + 1,
            json_str(code.chars().map(u32::from)),
        ));
        if let Some(m) = model {
            out.push_str(&format!(
                ", \"model\": {}",
                json_str(m.chars().map(u32::from))
            ));
        }
        out.push_str("}\n");
        i = k + 1;
    }
    Ok(out)
}

/// A file name as Python's `os.fsdecode` sees it: UTF-8, each byte of an invalid sequence
/// becoming the lone surrogate U+DC80 + byte (`surrogateescape`).
fn fs_code_points(name: &[u8]) -> Vec<u32> {
    let mut cps = Vec::with_capacity(name.len());
    for chunk in name.utf8_chunks() {
        cps.extend(chunk.valid().chars().map(u32::from));
        cps.extend(chunk.invalid().iter().map(|&b| 0xdc00 + u32::from(b)));
    }
    cps
}

/// `sorted(d.glob("*.md"))` for `ROOT/book/src/<sub>` when it is a directory: every entry
/// whose name ends `.md` (dot-files, directories and dangling links included), as
/// `(path relative to ROOT in code points, file name)`, in code point order. An unlistable
/// directory lists nothing, as pathlib's glob swallows the error.
fn chapters(root: &Path, sub: &str) -> Vec<(Vec<u32>, std::ffi::OsString)> {
    use std::os::unix::ffi::OsStrExt;
    let d = root.join("book").join("src").join(sub);
    if !d.is_dir() {
        return Vec::new();
    }
    let Ok(entries) = std::fs::read_dir(&d) else {
        return Vec::new();
    };
    let mut names: Vec<(Vec<u32>, std::ffi::OsString)> = entries
        .flatten()
        .map(|e| e.file_name())
        .filter(|n| n.as_bytes().ends_with(b".md"))
        .map(|n| {
            let mut rel: Vec<u32> = format!("book/src/{sub}/").chars().map(u32::from).collect();
            rel.extend(fs_code_points(n.as_bytes()));
            (rel, n)
        })
        .collect();
    names.sort();
    names
}

/// The whole run: stdout, or `(already printed, reason)` at the first chapter that cannot be
/// read, is not UTF-8 or holds an empty cost line; its own records are not printed.
pub fn run(root: &Path) -> Result<String, (String, String)> {
    let mut printed = String::new();
    for sub in ["cli", "lib"] {
        let d = root.join("book").join("src").join(sub);
        for (rel, name) in chapters(root, sub) {
            let path = d.join(&name);
            let fail = |r: String| (printed.clone(), format!("{}: {r}", path.display()));
            let bytes = std::fs::read(&path).map_err(|e| fail(e.to_string()))?;
            let text = String::from_utf8(bytes)
                .map_err(|e| fail(format!("not UTF-8: {e}")))?
                .replace("\r\n", "\n")
                .replace('\r', "\n");
            let records = extract_text(&rel, &text).map_err(fail)?;
            printed.push_str(&records);
        }
    }
    Ok(printed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cps(s: &str) -> Vec<u32> {
        s.chars().map(u32::from).collect()
    }

    fn extract(text: &str) -> String {
        extract_text(&cps("book/src/cli/x.md"), text).expect("extracts")
    }

    /// A tree under the system temp dir, removed on drop.
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
        let root =
            std::env::temp_dir().join(format!("ci-tools-extract-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for (rel, bytes) in files {
            let p = root.join(rel);
            std::fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
            std::fs::write(&p, bytes).expect("write");
        }
        Tree(root)
    }

    #[test]
    fn fences_and_cost_lines_match_the_regexes() {
        assert_eq!(fence_lang("```bash"), Some("bash"));
        assert_eq!(fence_lang("```rust \t\u{3000}\x1f"), Some("rust"));
        for no in [
            "```Bash", "````bash", " ```bash", "```bashx", "```", "```sh",
        ] {
            assert_eq!(fence_lang(no), None, "{no:?}");
        }
        assert!(is_close("```"));
        assert!(is_close("``` \u{a0}"));
        assert!(!is_close("```bash"));
        assert!(!is_close(" ```"));
        assert_eq!(cost_payload("<!-- example-cost: gpu -->"), Some("gpu"));
        assert_eq!(cost_payload("<!--example-cost:gpu-->"), Some("gpu"));
        assert_eq!(cost_payload("<!-- example-cost:  -->"), Some(""));
        for no in [
            "<!-- example-cost:-->",
            "<!-- example-cost: a > b -->",
            "<!-- example-cost: a --> b",
            "<!-- example-costs: a -->",
            "<!- example-cost: a -->",
        ] {
            assert_eq!(cost_payload(no), None, "{no:?}");
        }
    }

    #[test]
    fn parses_cost_and_model_like_the_original() {
        assert_eq!(parse_cost("gpu"), Ok(("gpu", None)));
        assert_eq!(
            parse_cost("model-required model: qwen2.5"),
            Ok(("model-required", Some("qwen2.5".to_owned())))
        );
        assert_eq!(
            parse_cost("bogus model: m"),
            Ok(("trivial", Some("m".to_owned())))
        );
        assert_eq!(
            parse_cost("model: m"),
            Ok(("trivial", Some("m".to_owned())))
        );
        assert_eq!(parse_cost("skip model:"), Ok(("skip", None)));
        assert_eq!(
            parse_cost("gpu model: a model: b"),
            Ok(("gpu", Some("a".to_owned())))
        );
        assert_eq!(
            parse_cost("gpu\x1fmodel:\u{3000}m"),
            Ok(("gpu", Some("m".to_owned())))
        );
        assert!(parse_cost("").is_err());
    }

    #[test]
    fn writes_records_as_json_dumps_does() {
        let got = extract(
            "x\n<!-- example-cost: model-required model: q -->\n\n```bash\napr run \"q\"\n\t\\\x7f é 😀\n```\n",
        );
        assert_eq!(
            got,
            "{\"path\": \"book/src/cli/x.md\", \"line_start\": 4, \"line_end\": 7, \"lang\": \"bash\", \"cost\": \"model-required\", \"code\": \"apr run \\\"q\\\"\\n\\t\\\\\\u007f \\u00e9 \\ud83d\\ude00\", \"model\": \"q\"}\n"
        );
        assert_eq!(
            json_str([0x0d, 0x08, 0x0c, 0x01, 0x1f, 0xdcff, 0x7e, 0x20]),
            "\"\\r\\b\\f\\u0001\\u001f\\udcff~ \""
        );
    }

    #[test]
    fn looks_back_two_lines_to_the_last_non_blank() {
        let rec = |t: &str| extract(t);
        assert!(rec("<!-- example-cost: gpu -->\n```rust\n```\n").contains("\"cost\": \"gpu\""));
        assert!(rec("<!-- example-cost: gpu -->\n \t\n```rust\n```\n").contains("\"gpu\""));
        assert!(!rec("<!-- example-cost: gpu -->\n\n\n```rust\n```\n").contains("\"gpu\""));
        assert!(!rec("<!-- example-cost: gpu -->\ntext\n```rust\n```\n").contains("\"gpu\""));
        assert!(rec("  <!-- example-cost: gpu -->  \n```rust\n```\n").contains("\"gpu\""));
        assert!(rec("```rust\n```\n").contains("\"trivial\""));
        let err = extract_text(&cps("p"), "<!-- example-cost:   -->\n```bash\n```\n");
        assert!(err.is_err());
        // An empty cost line no fence looks back to is never parsed.
        assert_eq!(
            extract("<!-- example-cost:   -->\nx\ny\n```bash\n```\n")
                .lines()
                .count(),
            1
        );
    }

    #[test]
    fn bodies_end_at_a_bare_fence_and_an_unclosed_one_ends_the_chapter() {
        let got = extract("```bash\na\n```rust\nb\n```\n```rust\n\n```\n```bash\nnever\n");
        assert_eq!(got.lines().count(), 2);
        assert!(got.contains("\"line_start\": 1, \"line_end\": 5, \"lang\": \"bash\""));
        assert!(got.contains("\"code\": \"a\\n```rust\\nb\""));
        assert!(got.contains("\"line_start\": 6, \"line_end\": 8, \"lang\": \"rust\", \"cost\": \"trivial\", \"code\": \"\""));
        assert!(!got.contains("never"));
        assert_eq!(extract("```bash\n```"), extract("```bash\n```\n"));
        assert!(extract("```bash\u{2028}x\u{85}```\n").contains("\"code\": \"x\""));
    }

    #[test]
    fn names_decode_with_surrogateescape_in_code_point_order() {
        assert_eq!(
            fs_code_points(b"a\xffb\xe2\x82"),
            vec![0x61, 0xdcff, 0x62, 0xdce2, 0xdc82]
        );
        assert_eq!(fs_code_points("é".as_bytes()), vec![0xe9]);
        // U+DCFF sorts before U+E000, though its source byte 0xff sorts after 0xee.
        let mut v = [
            fs_code_points("\u{e000}.md".as_bytes()),
            fs_code_points(b"\xff.md"),
        ];
        v.sort();
        assert_eq!(v[0][0], 0xdcff);
    }

    #[test]
    fn run_lists_cli_then_lib_in_order() {
        let t = tree(
            "order",
            &[
                ("book/src/lib/a.md", b"```rust\n```\n"),
                ("book/src/cli/b.md", b"```bash\nb\n```\r\n"),
                ("book/src/cli/.md", b"```bash\ndot\n```\n"),
                ("book/src/cli/c.MD", b"```bash\nupper\n```\n"),
                ("book/src/cli/d.txt", b"```bash\ntxt\n```\n"),
            ],
        );
        let got = run(t.path()).expect("runs");
        let paths: Vec<&str> = got
            .lines()
            .map(|l| l.split('"').nth(3).expect("path"))
            .collect();
        assert_eq!(
            paths,
            ["book/src/cli/.md", "book/src/cli/b.md", "book/src/lib/a.md"]
        );
    }

    #[test]
    fn run_with_no_book_prints_nothing() {
        let t = tree("none", &[("book/src/cli", b"a file, not a directory")]);
        assert_eq!(run(t.path()), Ok(String::new()));
    }

    #[test]
    fn run_stops_at_the_first_bad_chapter_after_printing() {
        let t = tree(
            "stop",
            &[
                ("book/src/cli/a.md", b"```rust\n```\n"),
                ("book/src/cli/b.md", b"```rust\n```\nbad \xff\n"),
                ("book/src/cli/c.md", b"```rust\n```\n"),
            ],
        );
        let (printed, reason) = run(t.path()).expect_err("b.md is not UTF-8");
        assert_eq!(printed.lines().count(), 1);
        assert!(printed.contains("cli/a.md"));
        assert!(reason.contains("b.md: not UTF-8"), "{reason}");

        let e = tree(
            "empty-cost",
            &[(
                "book/src/lib/e.md",
                b"```rust\n```\n<!-- example-cost: -->\n```rust\n```\n",
            )],
        );
        let (printed, reason) = run(e.path()).expect_err("an empty cost line");
        assert_eq!(printed, "");
        assert!(
            reason.contains("e.md: a cost line with no cost class"),
            "{reason}"
        );

        let d = tree("dir", &[("book/src/lib/d.md/x", b"")]);
        let (printed, reason) = run(d.path()).expect_err("d.md is a directory");
        assert_eq!(printed, "");
        assert!(reason.contains("d.md"), "{reason}");
    }

    #[test]
    fn run_escapes_a_non_utf8_name() {
        use std::os::unix::ffi::OsStrExt;
        let t = tree("nonutf8", &[]);
        let d = t.path().join("book/src/cli");
        std::fs::create_dir_all(&d).expect("mkdir");
        std::fs::write(
            d.join(std::ffi::OsStr::from_bytes(b"\xff.md")),
            "```rust\n```\n",
        )
        .expect("write");
        std::fs::write(d.join("\u{e000}.md"), "```rust\n```\n").expect("write");
        let got = run(t.path()).expect("runs");
        let paths: Vec<&str> = got
            .lines()
            .map(|l| l.split('"').nth(3).expect("path"))
            .collect();
        assert_eq!(
            paths,
            ["book/src/cli/\\udcff.md", "book/src/cli/\\ue000.md"]
        );
    }
}
