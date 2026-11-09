//! KTEST-02 §0.3: the oracle must not be able to reach an optimized backend, so no source file of
//! `kernel_oracle` may name a path outside that module or splice in source from elsewhere.
//!
//! The files are found at test time (every `.rs` under `src/kernel_oracle/`, at any depth) and
//! read whole, so a new file, a nested module or a test module is scanned like the rest. This scan
//! lives outside the directory it scans: its planted escapes would otherwise be escapes of the
//! oracle itself.

use std::path::{Path, PathBuf};

/// Text that splices source or macros in from outside the scanned files, matched on the whole file
/// with all whitespace (line breaks included) removed: `include!`-family macros, `#[path = …]`
/// (also raw `r"…"`/`r#"…"#`, and inside `cfg_attr`), `#[macro_use]` and `extern crate`.
const SPLICES: [&str; 8] = [
    "include!",
    "include_str!",
    "include_bytes!",
    "path=\"",
    "path=r\"",
    "path=r#",
    "macro_use",
    "externcrate",
];

/// The oracle's own modules, which its files may name by a bare path.
const LOCAL_MODULES: [&str; 4] = ["error_model", "margin", "oracle", "tests"];

/// Primitive types, whose associated items (`f64::from`, `u32::MAX`) are not crate paths.
const PRIMITIVES: [&str; 15] = [
    "f32", "f64", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128",
    "isize", "bool",
];

/// `line` up to its `//` comment; a `//` inside a string literal is not a comment.
fn code_of(line: &str) -> &str {
    let mut in_str = false;
    let mut prev = ' ';
    for (i, c) in line.char_indices() {
        match c {
            '"' if prev != '\\' => in_str = !in_str,
            '/' if prev == '/' && !in_str => return &line[..i - 1],
            _ => {},
        }
        prev = c;
    }
    line
}

/// Every line of `src` (comments stripped) that could reach code outside the oracle: a splice
/// (see [`SPLICES`]), a `crate::` path not under `crate::kernel_oracle::`, a `super::super`, a
/// leading `::`, or a path rooted at any crate other than `std`/`core`/`alloc`. Paths are read
/// across whitespace and line breaks, so `crate\n    ::gguf` is one path.
fn escapes(src: &str) -> Vec<String> {
    let lines: Vec<&str> = src.lines().map(code_of).collect();
    let code = lines.join("\n");
    let line_at = |i: usize| code[..i].matches('\n').count();
    let mut hits = vec![false; lines.len()];
    for line in splice_lines(&code) {
        hits[line] = true;
    }
    let mut pos = 0;
    while let Some(off) = code[pos..].find("::") {
        let i = pos + off;
        let head = code[..i].trim_end();
        let root_len = head
            .chars()
            .rev()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .map(char::len_utf8)
            .sum::<usize>();
        let root = &head[head.len() - root_len..];
        let before = head[..head.len() - root_len].trim_end();
        let after = code[i + 2..].trim_start();
        let bad = match root {
            "crate" => !after.starts_with("kernel_oracle::"),
            "super" => after.starts_with("super"),
            // A leading `::` names an extern crate; after `>` it is a turbofish or a qualified path.
            "" => !head.ends_with('>'),
            // Tool attributes (`#[allow(clippy::…)]`) name no crate.
            "std" | "core" | "alloc" | "self" | "Self" | "clippy" | "rustfmt" => false,
            r => r.starts_with(|c: char| c.is_ascii_lowercase()) && is_crate_root(r, before),
        };
        if bad {
            // The line that names the root, which for `crate\n::x` is not the line of the `::`.
            hits[line_at(head.len() - root_len)] = true;
        }
        pos = i + 2;
    }
    lines
        .iter()
        .zip(hits)
        .filter(|(_, hit)| *hit)
        .map(|(line, _)| line.trim().to_string())
        .collect()
}

/// The line each splice starts on, found in `code` with every whitespace character removed, so a
/// splice broken across lines (`include\n!(…)`, `extern\ncrate x;`) is still one splice.
fn splice_lines(code: &str) -> Vec<usize> {
    let mut squeezed = String::new();
    let mut line_of = Vec::new();
    let mut line = 0;
    for c in code.chars() {
        if c == '\n' {
            line += 1;
        }
        if !c.is_whitespace() {
            squeezed.push(c);
            line_of.extend(std::iter::repeat_n(line, c.len_utf8()));
        }
    }
    let mut found: Vec<usize> = SPLICES
        .iter()
        .flat_map(|s| squeezed.match_indices(s).map(|(i, _)| line_of[i]))
        .collect();
    found.sort_unstable();
    found.dedup();
    found
}

/// A lowercase path segment is a crate root iff nothing path-like precedes it.
fn is_crate_root(root: &str, before: &str) -> bool {
    !(before.ends_with("::")
        || before.ends_with('.')
        || LOCAL_MODULES.contains(&root)
        || PRIMITIVES.contains(&root))
}

/// Every `.rs` file under `dir`, at any depth, sorted.
fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut dirs = vec![dir.to_path_buf()];
    while let Some(d) = dirs.pop() {
        let entries = std::fs::read_dir(&d).unwrap_or_else(|e| panic!("{}: {e}", d.display()));
        for entry in entries {
            let path = entry.expect("directory entry").path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().is_some_and(|x| x == "rs") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

/// Every escape in every `.rs` file under `dir`, as `file: line`.
fn escapes_under(dir: &Path) -> Vec<String> {
    let mut found = Vec::new();
    for file in rust_files(dir) {
        let src =
            std::fs::read_to_string(&file).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
        let name = file
            .strip_prefix(dir)
            .unwrap_or(&file)
            .display()
            .to_string();
        found.extend(
            escapes(&src)
                .into_iter()
                .map(|line| format!("{name}: {line}")),
        );
    }
    found
}

fn oracle_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/kernel_oracle")
}

#[test]
fn oracle_reaches_nothing_outside_itself() {
    let files = rust_files(&oracle_dir());
    let names: Vec<String> = files
        .iter()
        .filter_map(|f| f.file_name().map(|n| n.to_string_lossy().into_owned()))
        .collect();
    for known in ["mod.rs", "error_model.rs", "margin.rs", "oracle.rs"] {
        assert!(
            names.iter().any(|n| n == known),
            "the scan must find {known}: {names:?}"
        );
    }
    let found = escapes_under(&oracle_dir());
    assert!(
        found.is_empty(),
        "KTEST-02 §0.3: the oracle must stay std-only, it reaches out: {found:?}"
    );
}

/// Anti-vacuity: each planted escape is seen, and the oracle's own idioms are not.
#[test]
fn escape_scan_sees_a_planted_path() {
    for planted in [
        "use crate::quantize::fused_q4k_parallel_matvec;",
        "let y = crate::gguf::GGUFConfig::default();",
        "use super::super::kernel_registry;",
        "use trueno::Vector;",
        "let v = realizar::quantize::dequantize(x);",
        "use ::trueno::Vector;",
        "let y = crate :: gguf::GGUFConfig::default();",
        "let u = \"http://x\"; let v = trueno::Vector::new();",
        "extern crate trueno;",
        "include!(\"../quantize/fused.rs\");",
        "const K: &str = include_str!(\"../gguf/mod.rs\");",
        "const B: &[u8] = include_bytes!(\"../gguf/table.bin\");",
        "#[path = \"../quantize/mod.rs\"]",
        "#[cfg_attr(test, path = \"../gguf/mod.rs\")]",
        "#[macro_use]",
        "trueno::dispatch!(x);",
        "#[path = r\"../quantize/mod.rs\"]",
        "#[path = r#\"../quantize/mod.rs\"#]",
    ] {
        assert_eq!(escapes(planted), vec![planted.to_string()], "{planted}");
    }
    // Split across lines: the escape is reported on the line it starts on.
    for (split, first) in [
        (
            "use crate\n    ::quantize::fused_q4k_parallel_matvec;",
            "use crate",
        ),
        ("include\n!(\"../quantize/fused.rs\");", "include"),
        (
            "const K: &str = include_str\n    !(\"../gguf/mod.rs\");",
            "const K: &str = include_str",
        ),
        ("#[path\n= \"../quantize/mod.rs\"]", "#[path"),
        ("#[path =\n    r\"../quantize/mod.rs\"]", "#[path ="),
        ("extern\ncrate trueno;", "extern"),
        ("#[macro\n_use]", "#[macro"),
    ] {
        assert_eq!(escapes(split), vec![first.to_string()], "{split}");
    }
    // Multibyte text before a splice: the offset is a byte offset, so the line map counts bytes.
    let one = "é\ninclude!(\"x\");";
    assert_eq!(escapes(one), vec!["include!(\"x\");".to_string()], "{one}");
    let many = format!("{}\ninclude!(\nx);\nlet y = 1;", "é".repeat(50));
    assert_eq!(escapes(&many), vec!["include!(".to_string()], "{many}");
    for clean in [
        "use super::error_model::{ErrorModel, Refusal};",
        "use crate::kernel_oracle::oracle;",
        "use super::*;",
        "let n = std::f64::consts::PI + f64::EPSILON;",
        "let v: Vec<f64> = x.iter().map(|v| f64::from(*v)).collect();",
        "let v = Vec::<f64>::with_capacity(n);",
        "let e = ErrorModel::Dot { k, acc: Dtype::F32, out: Dtype::F32, ftz: false };",
        "let r = oracle::dot(&a, &b); // crate::gguf in a comment",
        "#[allow(clippy::cast_precision_loss)]",
        "impl std::fmt::Display for Refusal {",
        "fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {",
        "impl std::error::Error for Refusal {}",
    ] {
        assert!(escapes(clean).is_empty(), "{clean}: {:?}", escapes(clean));
    }
}

/// Anti-vacuity for the file walk: a file the scan was never told about, one directory down, is
/// found and its escape reported.
#[test]
fn escape_scan_finds_a_new_file() {
    let dir = std::env::temp_dir().join(format!("kernel-oracle-scan-{}", std::process::id()));
    let nested = dir.join("nested");
    std::fs::create_dir_all(&nested).expect("create planted dir");
    std::fs::write(dir.join("mod.rs"), "pub mod nested;\n").expect("write planted mod.rs");
    std::fs::write(nested.join("fast.rs"), "use trueno::Vector;\n").expect("write planted file");
    let found = escapes_under(&dir);
    std::fs::remove_dir_all(&dir).expect("remove planted dir");
    let want = format!(
        "{}: use trueno::Vector;",
        Path::new("nested").join("fast.rs").display()
    );
    assert_eq!(found, vec![want]);
}
