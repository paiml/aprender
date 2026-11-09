//! Port of `scripts/lib/git_patch_id.py`: a byte-exact `git patch-id` for runners whose git
//! predates `--verbatim` (2.40).
//!
//! It mirrors the .py, which mirrors `get_one_patchid()` / `flush_one_hunk()` of git v2.53
//! `builtin/patch-id.c` and `diff.c`, line for line. Input is raw bytes split the way
//! `bytes.splitlines(keepends=True)` splits them (`\n`, `\r\n`, lone `\r`). Each patch prints
//! `<patch-id> <commit-oid>`; the oid is the 40-hex line that ended the previous patch, or 40
//! zeros. `scripts/tests/ci_tools_git_patch_id_parity_test.sh` checks it against the .py and,
//! where the git on PATH has the mode, against the native command.

use sha1::{Digest, Sha1};
use std::fmt::Write as _;

/// The oid printed before any `commit <oid>` / `From <oid>` line has been seen.
const ZERO: &str = "0000000000000000000000000000000000000000";
/// C `isspace()` in the "C" locale: what the non-verbatim modes drop from every line.
const SPACE: &[u8] = b" \t\n\x0b\x0c\r";

/// The three modes of `git patch-id`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Stable,
    Verbatim,
    Unstable,
}

impl Mode {
    /// The .py's argv rule: no argument is `--unstable`; anything else unknown is refused.
    #[must_use]
    pub fn parse(arg: Option<&str>) -> Option<Self> {
        match arg.unwrap_or("--unstable") {
            "--stable" => Some(Self::Stable),
            "--verbatim" => Some(Self::Verbatim),
            "--unstable" => Some(Self::Unstable),
            _ => None,
        }
    }
}

/// `bytes.splitlines(keepends=True)`.
fn split_lines(b: &[u8]) -> Vec<&[u8]> {
    let mut lines = Vec::new();
    let (mut start, mut i) = (0, 0);
    while i < b.len() {
        let end = match b[i] {
            b'\n' => i + 1,
            b'\r' if b.get(i + 1) == Some(&b'\n') => i + 2,
            b'\r' => i + 1,
            _ => {
                i += 1;
                continue;
            }
        };
        lines.push(&b[start..end]);
        start = end;
        i = end;
    }
    if start < b.len() {
        lines.push(&b[start..]);
    }
    lines
}

/// Python's `b[start:end]` for non-negative bounds: clamped, empty when inverted.
fn slice(b: &[u8], start: usize, end: usize) -> &[u8] {
    let end = end.min(b.len());
    if start >= end {
        &[]
    } else {
        &b[start..end]
    }
}

/// Python's `b.find(needle, from)`.
fn find(b: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    b.get(from..)?
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| p + from)
}

/// `re.compile(rb"[0-9]*").match(line, at).group(0)`.
fn digits(line: &[u8], at: usize) -> &[u8] {
    let rest = line.get(at..).unwrap_or_default();
    let n = rest.iter().take_while(|c| c.is_ascii_digit()).count();
    &rest[..n]
}

/// `int(d or b"0")`. Saturates where Python's int would grow; no diff has that many lines.
fn int(d: &[u8]) -> i64 {
    d.iter().fold(0_i64, |n, c| {
        n.saturating_mul(10).saturating_add(i64::from(c - b'0'))
    })
}

/// `diff.c flush_one_hunk`: add the hunk's sha1 into `result` bytewise with carry, and
/// start a fresh hunk.
fn flush_one_hunk(result: &mut [u8; 20], ctx: &mut Sha1) {
    let digest = std::mem::take(ctx).finalize();
    let mut carry = 0_u32;
    for (r, d) in result.iter_mut().zip(digest.iter()) {
        carry += u32::from(*r) + u32::from(*d);
        *r = carry.to_le_bytes()[0];
        carry >>= 8;
    }
}

/// `(before, after)` of an `@@ -a,b +c,d @@` header, or `None` when it does not parse.
fn scan_hunk_header(line: &[u8]) -> Option<(i64, i64)> {
    let mut q = 4;
    let mut n = digits(line, q).len();
    let before = if line.get(q + n) == Some(&b',') {
        q += n + 1;
        let d = digits(line, q);
        n = d.len();
        int(d)
    } else {
        1
    };
    if n == 0 || line.get(q + n) != Some(&b' ') || line.get(q + n + 1) != Some(&b'+') {
        return None;
    }
    let mut r = q + n + 2;
    n = digits(line, r).len();
    let after = if line.get(r + n) == Some(&b',') {
        r += n + 1;
        let d = digits(line, r);
        n = d.len();
        int(d)
    } else {
        1
    };
    (n != 0).then_some((before, after))
}

/// `index <pre>..<post>[ mode]` -> `(pre_oid, post_oid)`, or `None` when there is no `..`.
fn parse_index_line(line: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    let oid1_end = find(line, b"..", 0)?;
    // Without a mode the post oid runs to the line's last byte, which is dropped as the
    // newline (the .py's `len(line) - 1`); `..` makes the line at least 2 bytes long.
    let oid2_end = find(line, b" ", oid1_end).unwrap_or(line.len() - 1);
    let pre = slice(line, 6, oid1_end);
    let post = slice(line, oid1_end + 2, oid2_end);
    Some((slice(pre, 0, 40).to_vec(), slice(post, 0, 40).to_vec()))
}

fn is_binary_marker(line: &[u8]) -> bool {
    line.starts_with(b"GIT binary patch") || line.starts_with(b"Binary files")
}

/// The text after a `commit ` / `From ` prefix, else the line itself.
fn oid_candidate(line: &[u8]) -> &[u8] {
    line.strip_prefix(b"commit ")
        .or_else(|| line.strip_prefix(b"From "))
        .unwrap_or(line)
}

/// `HEX40.match(p)`: the candidate opens with 40 hex digits; they are the next oid.
fn leading_oid(p: &[u8]) -> Option<String> {
    let head = p.get(..40)?;
    head.iter()
        .all(u8::is_ascii_hexdigit)
        .then(|| String::from_utf8_lossy(head).to_ascii_lowercase())
}

/// What a scanner step tells the line loop to do next.
enum Act {
    Continue,
    Break,
    Fall,
}

/// The scanner state `get_one_patchid` threads through one patch.
struct State {
    before: i64,
    after: i64,
    diff_is_binary: bool,
    pre_oid: Vec<u8>,
    post_oid: Vec<u8>,
    ctx: Sha1,
    result: [u8; 20],
    stable: bool,
    verbatim: bool,
}

impl State {
    fn new(mode: Mode) -> Self {
        Self {
            before: -1,
            after: -1,
            diff_is_binary: false,
            pre_oid: Vec::new(),
            post_oid: Vec::new(),
            ctx: Sha1::new(),
            result: [0; 20],
            stable: mode != Mode::Unstable,
            verbatim: mode == Mode::Verbatim,
        }
    }

    /// A line seen before a hunk header count is known.
    fn header_line(&mut self, line: &[u8]) -> Act {
        if is_binary_marker(line) {
            self.diff_is_binary = true;
            self.before = 0;
            self.ctx.update(&self.pre_oid);
            self.ctx.update(&self.post_oid);
            if self.stable {
                flush_one_hunk(&mut self.result, &mut self.ctx);
            }
            return Act::Continue;
        }
        if line.starts_with(b"index ") {
            if let Some((pre, post)) = parse_index_line(line) {
                self.pre_oid = pre;
                self.post_oid = post;
            }
            return Act::Continue;
        }
        if line.starts_with(b"--- ") {
            self.before = 1;
            self.after = 1;
            return Act::Fall;
        }
        if line.first().is_some_and(u8::is_ascii_alphabetic) {
            Act::Fall
        } else {
            Act::Break
        }
    }

    /// A line seen when the previous hunk is fully counted.
    fn between_hunks(&mut self, line: &[u8]) -> Act {
        if line.starts_with(b"@@ -") {
            if let Some((before, after)) = scan_hunk_header(line) {
                self.before = before;
                self.after = after;
            }
            return Act::Continue;
        }
        if !line.starts_with(b"diff ") {
            return Act::Break;
        }
        if self.stable {
            flush_one_hunk(&mut self.result, &mut self.ctx);
        }
        self.before = -1;
        self.after = -1;
        Act::Fall
    }

    /// Count a hunk line and hash it; returns the bytes it adds to the patch length.
    fn hash_line(&mut self, line: &[u8]) -> usize {
        match line.first() {
            Some(b'-') => self.before -= 1,
            Some(b' ') => {
                self.before -= 1;
                self.after -= 1;
            }
            Some(b'+') => self.after -= 1,
            _ => {}
        }
        if self.verbatim {
            self.ctx.update(line);
            line.len()
        } else {
            let data: Vec<u8> = line
                .iter()
                .copied()
                .filter(|b| !SPACE.contains(b))
                .collect();
            self.ctx.update(&data);
            data.len()
        }
    }

    /// The steps between the oid check and the hash, in the .py's order.
    fn route(&mut self, line: &[u8]) -> Act {
        if self.before == -1 {
            match self.header_line(line) {
                Act::Fall => {}
                act => return act,
            }
        }
        if self.diff_is_binary {
            if line.starts_with(b"diff ") {
                self.diff_is_binary = false;
                self.before = -1;
            }
            return Act::Continue;
        }
        if self.before == 0 && self.after == 0 {
            return self.between_hunks(line);
        }
        Act::Fall
    }

    /// One line of the .py's loop body up to the hash; `started` is `patchlen != 0`.
    fn step(&mut self, line: &[u8], started: bool) -> Step {
        if line.starts_with(b"\\ ") && line.len() > 12 {
            if self.verbatim {
                self.ctx.update(line);
            }
            return Step::Skip;
        }
        if let Some(oid) = leading_oid(oid_candidate(line)) {
            return Step::End(Some(oid));
        }
        if !started && !line.starts_with(b"diff ") {
            return Step::Skip;
        }
        match self.route(line) {
            Act::Continue => Step::Skip,
            Act::Break => Step::End(None),
            Act::Fall => Step::Hash,
        }
    }
}

/// What one line does to the patch: nothing, end it (with the next patch's oid, if the
/// line named one), or add to its hash.
enum Step {
    Skip,
    End(Option<String>),
    Hash,
}

/// One patch: `(patch length, patch-id, oid that ended it, next line)`.
fn get_one_patchid(
    lines: &[&[u8]],
    mut pos: usize,
    mode: Mode,
) -> (usize, [u8; 20], Option<String>, usize) {
    let mut patchlen = 0;
    let mut st = State::new(mode);
    let mut next_oid = None;
    while let Some(&line) = lines.get(pos) {
        pos += 1;
        match st.step(line, patchlen != 0) {
            Step::Skip => {}
            Step::End(oid) => {
                next_oid = oid;
                break;
            }
            Step::Hash => patchlen += st.hash_line(line),
        }
    }
    flush_one_hunk(&mut st.result, &mut st.ctx);
    (patchlen, st.result, next_oid, pos)
}

/// Every `<patch-id> <oid>` line `git_patch_id.py <mode>` prints for `input`.
#[must_use]
pub fn run(input: &[u8], mode: Mode) -> String {
    let lines = split_lines(input);
    let mut out = String::new();
    let mut pos = 0;
    let mut oid = ZERO.to_owned();
    loop {
        let (patchlen, result, next, p) = get_one_patchid(&lines, pos, mode);
        pos = p;
        if patchlen > 0 {
            for b in result {
                let _ = write!(out, "{b:02x}");
            }
            let _ = writeln!(out, " {oid}");
        }
        oid = next.unwrap_or_else(|| ZERO.to_owned());
        if pos >= lines.len() {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{run, scan_hunk_header, split_lines, Mode};

    /// The whitespace fixture of `scripts/lib/pr_review_patch_id.sh --self-test`: a binary
    /// hunk, CRLF, trailing blanks, a `\ No newline` line and a new file.
    const WS_DIFF: &[u8] = b"diff --git a/b.bin b/b.bin\n\
index 88768efdf77ec78c9a995f94881793be6a41752b..3e3315e1b02129d197721a8a0b56dd88862f454d 100644\n\
Binary files a/b.bin and b/b.bin differ\n\
diff --git a/f b/f\n\
index 7c766e90d5b572097b55657d7df3099e2bc9e2e4..2495bbe3521f6350b9012f2bd26ab83ca88c5c3a 100644\n\
--- a/f\n\
+++ b/f\n\
@@ -1,3 +1,3 @@\n\
-a \n\
-\tb\r\n\
-c\n\
\\ No newline at end of file\n\
+a  \n\
+\tb\n\
+c\n\
diff --git a/g b/g\n\
new file mode 100644\n\
index 0000000000000000000000000000000000000000..3e757656cf36eca53338e520d134963a44f793f8\n\
--- /dev/null\n\
+++ b/g\n\
@@ -0,0 +1 @@\n\
+new\n";

    const Z: &str = "0000000000000000000000000000000000000000";

    /// The golden ids that self-test pins (native git 2.53.0).
    #[test]
    fn golden_ids_of_the_self_test_fixture() {
        for (mode, want) in [
            (Mode::Verbatim, "4866d77a6281c72194256e3826f262bcf982de28"),
            (Mode::Stable, "1379117dba9a33896e4076e98048a064488ae8db"),
            (Mode::Unstable, "92a2dd2834633681d83627e08e08cd2b6b4c6ab4"),
        ] {
            assert_eq!(run(WS_DIFF, mode), format!("{want} {Z}\n"), "{mode:?}");
        }
    }

    #[test]
    fn modes_parse_like_the_py_argv() {
        assert_eq!(Mode::parse(None), Some(Mode::Unstable));
        assert_eq!(Mode::parse(Some("--stable")), Some(Mode::Stable));
        assert_eq!(Mode::parse(Some("--verbatim")), Some(Mode::Verbatim));
        assert_eq!(Mode::parse(Some("--unstable")), Some(Mode::Unstable));
        for bad in ["", "stable", "--Stable", "-s", "--verbatim "] {
            assert_eq!(Mode::parse(Some(bad)), None, "{bad:?}");
        }
    }

    #[test]
    fn lines_split_like_bytes_splitlines() {
        let cases: &[(&[u8], &[&[u8]])] = &[
            (b"", &[]),
            (b"a", &[b"a"]),
            (b"a\nb", &[b"a\n", b"b"]),
            (b"a\r\nb\rc\n", &[b"a\r\n", b"b\r", b"c\n"]),
            (b"\r\r\n\n", &[b"\r", b"\r\n", b"\n"]),
            // Only \n and \r end a bytes line; \v, \f and \x1c stay inside it.
            (b"a\x0bb\x0cc\x1cd\n", &[b"a\x0bb\x0cc\x1cd\n"]),
        ];
        for (input, want) in cases {
            assert_eq!(split_lines(input), *want, "{input:?}");
        }
    }

    #[test]
    fn hunk_headers_parse_like_the_py() {
        let cases: &[(&[u8], Option<(i64, i64)>)] = &[
            (b"@@ -1,3 +1,3 @@\n", Some((3, 3))),
            (b"@@ -0,0 +1 @@\n", Some((0, 1))),
            (b"@@ -5 +7,2 @@\n", Some((1, 2))),
            (b"@@ -1, +1 @@\n", None),
            (b"@@ -1,3 -1,3 @@\n", None),
            (b"@@ -x +1 @@\n", None),
            (b"@@ -1 +\n", None),
            (b"@@ -", None),
        ];
        for (line, want) in cases {
            assert_eq!(scan_hunk_header(line), *want, "{line:?}");
        }
    }

    /// A `commit <oid>` line ends one patch and names the next; uppercase hex is lowered.
    #[test]
    fn commit_lines_carry_the_oid_to_the_next_patch() {
        let one = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n";
        let oid = "ABCDEF0123456789ABCDEF0123456789ABCDEF01";
        let input = format!("{one}commit {oid}\n{one}");
        let out = run(input.as_bytes(), Mode::Stable);
        let rows: Vec<&str> = out.lines().collect();
        assert_eq!(rows.len(), 2, "{out}");
        assert!(rows[0].ends_with(Z), "{out}");
        assert!(rows[1].ends_with(&oid.to_ascii_lowercase()), "{out}");
        assert_eq!(rows[0][..40], rows[1][..40], "same patch, same id");
    }

    #[test]
    fn no_diff_prints_nothing() {
        for input in [
            &b""[..],
            b"\n",
            b"not a diff\n",
            b"commit 0123456789012345678901234567890123456789\n",
        ] {
            assert_eq!(run(input, Mode::Unstable), "", "{input:?}");
        }
    }
}
