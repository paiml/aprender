//! `privscan PATH...`: flag lines that would leak private detail into a public file. One
//! tracked tool in place of the untracked `privscan.sh` copies kept per worktree (#4678).
//!
//! Patterns 1-7 are that script's rules, verbatim and in its order: ASCII, case-insensitive
//! (`LC_ALL=C grep -a -i -E`), unbounded unless the rule spells `\b`. Patterns 8-10 are what
//! the older `privscan.py` checked and the script did not: credential tokens (case-sensitive,
//! as there, so prose cannot match them), MAC addresses, and an e-mail address under any
//! top-level domain. `privscan.py`'s allow-list of service addresses is not taken: it removes
//! hits, and the script's rules are the spec.
//!
//! Output, per file, then per pattern, then per line: `FILE:LINE:pattern-N`, never the text,
//! then `privscan: N hit(s)`. A line counts once per pattern it matches. A directory is walked
//! in name order (a symlinked directory is not entered). Exit 0 no hit, 1 any hit, 2 a path
//! that cannot be read or no path at all; the script let both pass with 0 hits.
//!
//! The literals below are split with `concat!` so this file neither matches itself nor
//! spells out the words it hunts for; `own_source_is_clean` holds that.

use regex::bytes::{Regex, RegexBuilder};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// `(rule, case-insensitive)`, numbered from 1 in output.
const RULES: [(&str, bool); 10] = [
    (
        concat!("ti+e", "r:|reve+nue pa", "th|flo", "w[-_]manifest"),
        true,
    ),
    (
        concat!(
            "lam",
            "bda|gx",
            "10|int",
            "el|fw",
            "16|mb",
            "p|mac-ser",
            "ver"
        ),
        true,
    ),
    (
        concat!("no", "ah|mad", "rid|@[a-z0-9.-]+\\.(com|org|net)"),
        true,
    ),
    (r"\bD(9|10|11|12)\b", true),
    (
        concat!(
            "red_a",
            "ge|nightly_gr",
            "eens|release-fac",
            "tory|cop-in",
            "box|cop-st",
            "ate|/ho",
            "me/|/m",
            "nt/"
        ),
        true,
    ),
    (r"\b[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b", true),
    (
        concat!(
            "hand",
            "off:|mem",
            "ory:|in",
            "fra@|\\bM-[A-Z]|\\bR-C[0-9]|\\bF-[A-Z]"
        ),
        true,
    ),
    (
        concat!(
            "gh",
            "[pousr]_[A-Za-z0-9]{20}|git",
            "hub_pat_[A-Za-z0-9_]{20}|BEG",
            "IN [A-Z ]*KEY|AK",
            "IA[0-9A-Z]{12}|xo",
            "x[bpas]-[A-Za-z0-9-]{10}|\\bs",
            "k-[A-Za-z0-9]{20}|Bear",
            "er [A-Za-z0-9._-]{20}"
        ),
        false,
    ),
    (r"\b([0-9a-f]{2}[:-]){5}[0-9a-f]{2}\b", true),
    (
        r"[a-z0-9_.+-]+@[a-z0-9_-]+(\.[a-z0-9_-]+)*\.[a-z]{2,}\b",
        true,
    ),
];

fn rules() -> &'static [Regex] {
    static R: OnceLock<Vec<Regex>> = OnceLock::new();
    R.get_or_init(|| {
        RULES
            .iter()
            .map(|(p, ci)| {
                RegexBuilder::new(p)
                    .case_insensitive(*ci)
                    .unicode(false)
                    .build()
                    .expect("privscan rules are fixed and compile")
            })
            .collect()
    })
}

/// The `FILE:LINE:pattern-N` rows for one file's bytes, in the script's order.
pub fn scan(name: &str, data: &[u8]) -> Vec<String> {
    let mut rows = Vec::new();
    for (i, rx) in rules().iter().enumerate() {
        for (n, line) in data.split(|b| *b == b'\n').enumerate() {
            if rx.is_match(line) {
                rows.push(format!("{name}:{}:pattern-{}", n + 1, i + 1));
            }
        }
    }
    rows
}

fn files_under(path: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let meta = std::fs::symlink_metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if meta.is_dir() {
        let mut kids = std::fs::read_dir(path)
            .and_then(|d| {
                d.map(|e| e.map(|e| e.path()))
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(|e| format!("{}: {e}", path.display()))?;
        kids.sort();
        for k in kids {
            let is_link_dir = std::fs::symlink_metadata(&k)
                .is_ok_and(|m| m.file_type().is_symlink())
                && k.is_dir();
            if !is_link_dir {
                files_under(&k, out)?;
            }
        }
    } else {
        out.push(path.to_path_buf());
    }
    Ok(())
}

/// The report and the hit count, or the reason a path could not be scanned.
pub fn run(paths: &[PathBuf]) -> Result<(String, usize), String> {
    let mut files = Vec::new();
    for p in paths {
        files_under(p, &mut files)?;
    }
    let mut out = String::new();
    let mut hits = 0;
    for f in &files {
        let data = std::fs::read(f).map_err(|e| format!("{}: {e}", f.display()))?;
        for row in scan(&f.display().to_string(), &data) {
            out.push_str(&row);
            out.push('\n');
            hits += 1;
        }
    }
    out.push_str(&format!("privscan: {hits} hit(s)\n"));
    Ok((out, hits))
}

#[cfg(test)]
mod tests {
    use super::{run, scan, RULES};

    fn pats(line: &str) -> Vec<usize> {
        scan("f", line.as_bytes())
            .iter()
            .map(|r| {
                r.rsplit("pattern-")
                    .next()
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Must-hit rows: every alternative of every rule, and the case each rule is matched in.
    #[test]
    fn must_hit() {
        let cases: &[(&str, &str, &[usize])] = &[
            ("1 tier key", concat!("ti", "er: 2"), &[1]),
            (
                "1 tier key, doubled i, any case",
                concat!("TII", "ER: x"),
                &[1],
            ),
            (
                "1 the money-route phrase",
                concat!("the reven", "ue path"),
                &[1],
            ),
            (
                "1 manifest, underscore",
                concat!("flo", "w_manifest.yaml"),
                &[1],
            ),
            (
                "2 host in a word (unbounded)",
                concat!("int", "elligence"),
                &[2],
            ),
            ("2 host, upper case", concat!("GX", "10"), &[2]),
            (
                "2 each host",
                concat!("la", "mbda fw", "16 m", "bp mac-se", "rver"),
                &[2],
            ),
            ("3 a name", concat!("by No", "ah"), &[3]),
            ("3 a city", concat!("Ma", "drid time"), &[3]),
            (
                "3 e-mail .com hits 3 and 10",
                concat!("a@", "b.com"),
                &[3, 10],
            ),
            ("4 a rule tag", concat!("see D", "11 here"), &[4]),
            ("5 a home path", concat!("/ho", "me/x"), &[5]),
            ("5 a mount path", concat!("/m", "nt/raid"), &[5]),
            ("5 an inbox", concat!("cop-in", "box"), &[5]),
            ("5 a state file", concat!("cop-st", "ate.md"), &[5]),
            ("5 a factory", concat!("release-fac", "tory"), &[5]),
            (
                "5 dashboard keys",
                concat!("red_a", "ge nightly_gr", "eens"),
                &[5],
            ),
            (
                "5 a home dir is unbounded: relative hits too",
                concat!("src/ho", "me/mod.rs"),
                &[5],
            ),
            ("6 an ipv4", concat!("at 10.0.", "0.1 now"), &[6]),
            (
                "6 a quad inside five octets",
                concat!("1.2.3.", "4.5"),
                &[6],
            ),
            ("7 a doc ref", concat!("hand", "off: x"), &[7]),
            ("7 a note ref", concat!("mem", "ory: y"), &[7]),
            ("7 a mailbox", concat!("in", "fra@"), &[7]),
            ("7 rule ids", concat!("M", "-X R", "-C1 F", "-Y"), &[7]),
            (
                "8 a classic token",
                concat!("gh", "p_", "aaaaaaaaaaaaaaaaaaaa"),
                &[8],
            ),
            (
                "8 a fine-grained token",
                concat!("git", "hub_pat_", "a_a_a_a_a_a_a_a_a_a_"),
                &[8],
            ),
            (
                "8 a key block",
                concat!("-----BEG", "IN RSA PRIVATE KEY"),
                &[8],
            ),
            ("8 a cloud key", concat!("AK", "IA", "ABCDEFGHIJKL"), &[8]),
            ("8 a chat token", concat!("xo", "xb-", "0123456789"), &[8]),
            (
                "8 an api key",
                concat!(" s", "k-", "aaaaaaaaaaaaaaaaaaaa"),
                &[8],
            ),
            (
                "8 a bearer",
                concat!("Bear", "er ", "aaaaaaaaaaaaaaaaaaaa"),
                &[8],
            ),
            ("9 a mac", concat!("00:1A:2b", ":3c:4D:5e"), &[9]),
            ("9 a mac, dashes", concat!("00-1a-2b", "-3c-4d-5e"), &[9]),
            (
                "10 e-mail, other TLD",
                concat!("x.y+z@", "ex-ample.io"),
                &[10],
            ),
        ];
        for (name, line, want) in cases {
            assert_eq!(pats(line), *want, "{name}: {line}");
        }
    }

    /// Must-not-hit rows: near misses the rules' boundaries are there to keep out.
    #[test]
    fn must_not_hit() {
        let cases: &[(&str, &str)] = &[
            ("1 tear: is not a tier key", "tear: x"),
            ("4 D13 and AD9 are not rule tags", "D13 AD9 D9x"),
            ("5 a home with no slash after", "/homework"),
            ("6 a version", "v0.70.2"),
            ("6 three octets are not four", "1.2.3"),
            ("7 a word-inner M-", "PMAT-4678 llvm-cov if-else"),
            ("8 prose with begin..key", "we begin with the key"),
            ("8 a short token", concat!("gh", "p_", "short")),
            (
                "8 a lower-case bearer",
                concat!("bear", "er ", "aaaaaaaaaaaaaaaaaaaa"),
            ),
            ("9 a sha", "deadbeefcafe0123"),
            ("10 a scoped name", "pkg@0.70.0"),
            ("plain code", "fn main() { let x = 1; }"),
        ];
        for (name, line) in cases {
            assert_eq!(pats(line), Vec::<usize>::new(), "{name}: {line}");
        }
    }

    #[test]
    fn rows_name_file_line_and_pattern_never_the_text() {
        let data = concat!("ok\n", "ip 10.0.", "0.1\n", "a@", "b.com\n");
        assert_eq!(
            scan("p.md", data.as_bytes()),
            ["p.md:3:pattern-3", "p.md:2:pattern-6", "p.md:3:pattern-10"]
        );
        assert_eq!(
            scan("p", concat!("no newline at end 10.0.", "0.1").as_bytes()),
            ["p:1:pattern-6"]
        );
        assert_eq!(
            scan("b", &[b"\x00\xff10.0." as &[u8], b"0.1\x00"].concat()),
            ["b:1:pattern-6"]
        );
        assert!(scan("e", b"").is_empty());
    }

    #[test]
    fn run_walks_dirs_and_refuses_what_it_cannot_read() {
        let d = std::env::temp_dir().join(format!("privscan-t-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("sub")).expect("mkdir");
        std::fs::write(d.join("b.txt"), "clean\n").expect("write");
        std::fs::write(d.join("sub/a.txt"), concat!("10.0.", "0.1\n")).expect("write");
        let (out, hits) = run(std::slice::from_ref(&d)).expect("scan");
        assert_eq!(hits, 1);
        assert_eq!(
            out,
            format!(
                "{}:1:pattern-6\nprivscan: 1 hit(s)\n",
                d.join("sub/a.txt").display()
            )
        );
        let (out, hits) = run(&[d.join("b.txt")]).expect("scan");
        assert_eq!((out.as_str(), hits), ("privscan: 0 hit(s)\n", 0));
        assert!(
            run(&[d.join("missing")]).is_err(),
            "a missing path is not a clean pass"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn own_source_is_clean() {
        assert!(scan("privscan.rs", include_bytes!("privscan.rs")).is_empty());
        assert_eq!(RULES.len(), 10);
    }
}
