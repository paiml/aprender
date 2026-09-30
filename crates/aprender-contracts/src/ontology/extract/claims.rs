//! ONT-001 §5 ONT-4c (v4.14 claim-fence clause) — the Markdown machinery `extract:readme` and
//! `extract:llm-context` share: frontmatter, headings, inline code spans, fenced code blocks found by CommonMark,
//! which of them are CLAIMS, and the set of commands a merge-path workflow actually runs.
//!
//! **A claim** is a fence whose first info-string token, lower-cased with a leading `{` and `.` stripped, is
//! `bash`, `sh` or `shell` — so ```` ```bash ````, ```` ~~~ sh ````, ```` ```{.Bash} ```` are claims and
//! ```` ```text ````, ```` ```console ```` are not. Fences are found at any nesting depth: leading whitespace
//! and blockquote `>` markers are stripped before the fence test.
//!
//! **Normalisation**, applied to claim bodies and workflow `run:` values alike (a `run:` value is already
//! YAML-decoded by the parser): `\` continuations joined; a shell comment (`#` at the start or after whitespace,
//! outside quotes) dropped; ends trimmed; whitespace collapsed outside quotes; empty lines dropped. One
//! normalised line is one command.
//!
//! **The CI set** is every normalised `run:` line of a step, under `.github/workflows/*.{yml,yaml}`, of a
//! workflow triggered by `push`, `pull_request`, `pull_request_target` or `merge_group`, in a step without a
//! literal `if: false`. A claim that is not in the set is unresolved — the extractor names it.

use std::collections::BTreeSet;
use std::path::Path;

/// The triggers that put a workflow on the merge path.
pub const MERGE_PATH_EVENTS: [&str; 4] =
    ["push", "pull_request", "pull_request_target", "merge_group"];

/// The info-string tokens that make a fence a claim.
pub const CLAIM_LANGS: [&str; 3] = ["bash", "sh", "shell"];

/// One fenced code block: its info string, the 1-based line of its opening fence, and its body lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fence {
    pub info: String,
    pub line: usize,
    pub body: Vec<String>,
}

impl Fence {
    /// Is this fence a claim (its first info token is a shell language)?
    #[must_use]
    pub fn is_claim(&self) -> bool {
        is_claim_info(&self.info)
    }
}

/// The claim rule on an info string alone.
#[must_use]
pub fn is_claim_info(info: &str) -> bool {
    let first = info.split_whitespace().next().unwrap_or("");
    let tok = first.strip_prefix('{').unwrap_or(first);
    let tok = tok.strip_prefix('.').unwrap_or(tok);
    let tok = tok.trim_end_matches('}').to_ascii_lowercase();
    CLAIM_LANGS.contains(&tok.as_str())
}

/// A line with its container prefix (indentation and `>` blockquote markers, any depth) removed.
fn strip_container(line: &str) -> &str {
    let mut s = line;
    loop {
        let t = s.trim_start();
        match t.strip_prefix('>') {
            Some(rest) => s = rest,
            None => return t,
        }
    }
}

/// `Some((char, len, info))` when `s` opens a fence: ≥3 backticks or tildes; a backtick fence's info string
/// may not contain a backtick (CommonMark §4.5).
fn fence_open(s: &str) -> Option<(char, usize, String)> {
    let c = s.chars().next()?;
    if c != '`' && c != '~' {
        return None;
    }
    let n = s.chars().take_while(|x| *x == c).count();
    if n < 3 {
        return None;
    }
    let info = s[n..].trim().to_string();
    if c == '`' && info.contains('`') {
        return None;
    }
    Some((c, n, info))
}

/// Does `s` close a fence opened with `n` × `c`?
fn fence_close(s: &str, c: char, n: usize) -> bool {
    let m = s.chars().take_while(|x| *x == c).count();
    m >= n && s[m * c.len_utf8()..].trim().is_empty()
}

/// Every fenced code block in `text`, in order. An unclosed fence runs to the end of the document.
#[must_use]
pub fn fences(text: &str) -> Vec<Fence> {
    let mut out = Vec::new();
    let mut open: Option<(char, usize, Fence)> = None;
    for (i, raw) in text.lines().enumerate() {
        let s = strip_container(raw);
        if let Some((c, n, mut f)) = open.take() {
            if fence_close(s, c, n) {
                out.push(f);
            } else {
                f.body.push(s.to_string());
                open = Some((c, n, f));
            }
            continue;
        }
        if let Some((c, n, info)) = fence_open(s) {
            open = Some((
                c,
                n,
                Fence {
                    info,
                    line: i + 1,
                    body: Vec::new(),
                },
            ));
        }
    }
    if let Some((_, _, f)) = open {
        out.push(f);
    }
    out
}

/// The lines of `text` that are OUTSIDE every fence (and outside the frontmatter), with their 1-based numbers.
#[must_use]
pub fn prose_lines(text: &str) -> Vec<(usize, String)> {
    let skip = frontmatter_span(text).map_or(0, |(_, end)| end);
    let mut out = Vec::new();
    let mut open: Option<(char, usize)> = None;
    for (i, raw) in text.lines().enumerate() {
        if i < skip {
            continue;
        }
        let s = strip_container(raw);
        if let Some((c, n)) = open {
            if fence_close(s, c, n) {
                open = None;
            }
            continue;
        }
        if let Some((c, n, _)) = fence_open(s) {
            open = Some((c, n));
            continue;
        }
        out.push((i + 1, raw.to_string()));
    }
    out
}

/// ATX headings outside fences: `(level, text)`, the text trimmed and any closing `#` run removed.
#[must_use]
pub fn headings(text: &str) -> Vec<(usize, String)> {
    prose_lines(text)
        .into_iter()
        .filter_map(|(_, l)| {
            let t = l.trim_start();
            let level = t.chars().take_while(|c| *c == '#').count();
            if !(1..=6).contains(&level) {
                return None;
            }
            let rest = &t[level..];
            if !rest.is_empty() && !rest.starts_with(' ') && !rest.starts_with('\t') {
                return None;
            }
            let body = rest.trim();
            // a closing `#` run counts only when a space precedes it (or it is the whole content)
            let stripped = body.trim_end_matches('#');
            let body = if stripped.is_empty() {
                ""
            } else if stripped.ends_with([' ', '\t']) {
                stripped.trim_end()
            } else {
                body
            };
            Some((level, body.to_string()))
        })
        .collect()
}

/// The length of the backtick run starting at `b[i]`.
fn tick_run(b: &[u8], i: usize) -> usize {
    b[i..].iter().take_while(|x| **x == b'`').count()
}

/// The index of the backtick run of exactly `n` that closes a span opened before `start`, if any.
fn closing_run(b: &[u8], start: usize, n: usize) -> Option<usize> {
    let mut j = start;
    while j < b.len() {
        if b[j] != b'`' {
            j += 1;
            continue;
        }
        let m = tick_run(b, j);
        if m == n {
            return Some(j);
        }
        j += m;
    }
    None
}

/// The code spans of one prose line, appended to `out`.
fn line_spans(line: &str, out: &mut Vec<String>) {
    let b = line.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'`' {
            i += 1;
            continue;
        }
        let n = tick_run(b, i);
        let start = i + n;
        match closing_run(b, start, n) {
            Some(end) => {
                out.push(line[start..end].trim().to_string());
                i = end + n;
            }
            None => i = start,
        }
    }
}

/// Inline code spans outside fences (single-backtick-run delimited, CommonMark's matching-length rule).
#[must_use]
pub fn code_spans(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (_, line) in prose_lines(text) {
        line_spans(&line, &mut out);
    }
    out
}

/// The frontmatter's line span `(first body line index, index after the closing `---`)`, when `text` opens
/// with a `---` line and a later `---` line closes it.
fn frontmatter_span(text: &str) -> Option<(usize, usize)> {
    let mut lines = text.lines();
    if lines.next()?.trim_end() != "---" {
        return None;
    }
    lines
        .position(|l| l.trim_end() == "---")
        .map(|p| (1, p + 2))
}

/// The frontmatter as a YAML mapping. `None` when there is none; `Err` when it is there and is not a mapping.
pub fn frontmatter(text: &str) -> Result<Option<serde_yaml::Mapping>, String> {
    let Some((start, end)) = frontmatter_span(text) else {
        return Ok(None);
    };
    let body: Vec<&str> = text.lines().skip(start).take(end - 1 - start).collect();
    let v: serde_yaml::Value =
        serde_yaml::from_str(&body.join("\n")).map_err(|e| format!("frontmatter: {e}"))?;
    match v {
        serde_yaml::Value::Mapping(m) => Ok(Some(m)),
        serde_yaml::Value::Null => Ok(Some(serde_yaml::Mapping::new())),
        _ => Err("frontmatter is not a mapping".into()),
    }
}

/// `snake_case` / `kebab-case` → `camelCase`, the predicate spelling of a frontmatter key.
#[must_use]
pub fn camel(key: &str) -> String {
    let mut out = String::new();
    let mut up = false;
    for c in key.chars() {
        if c == '_' || c == '-' {
            up = true;
        } else if up {
            out.extend(c.to_uppercase());
            up = false;
        } else {
            out.push(c);
        }
    }
    out
}

/// Normalise a command block into its commands (see the module doc).
#[must_use]
pub fn normalise(block: &str) -> Vec<String> {
    let mut joined: Vec<String> = Vec::new();
    let mut acc = String::new();
    for line in block.lines() {
        let l = line.trim_end();
        if let Some(head) = l.strip_suffix('\\') {
            acc.push_str(head);
            acc.push(' ');
            continue;
        }
        acc.push_str(l);
        joined.push(std::mem::take(&mut acc));
    }
    if !acc.is_empty() {
        joined.push(acc);
    }
    joined
        .iter()
        .map(|l| collapse(&drop_comment(l)))
        .filter(|l| !l.is_empty())
        .collect()
}

/// The line with a shell comment removed: a `#` at the start or after whitespace, outside quotes.
fn drop_comment(line: &str) -> String {
    let mut out = String::new();
    let (mut sq, mut dq, mut esc) = (false, false, false);
    let mut prev_ws = true;
    for c in line.chars() {
        if esc {
            esc = false;
            out.push(c);
            prev_ws = false;
            continue;
        }
        match c {
            '\\' if !sq => esc = true,
            '\'' if !dq => sq = !sq,
            '"' if !sq => dq = !dq,
            '#' if !sq && !dq && prev_ws => break,
            _ => {}
        }
        prev_ws = c.is_whitespace();
        out.push(c);
    }
    out
}

/// Shell quoting state: inside single quotes, double quotes, or just after a backslash.
#[derive(Default)]
struct Quotes {
    sq: bool,
    dq: bool,
    esc: bool,
}

impl Quotes {
    fn outside(&self) -> bool {
        !self.sq && !self.dq && !self.esc
    }

    fn step(&mut self, c: char) {
        if self.esc {
            self.esc = false;
            return;
        }
        match c {
            '\\' if !self.sq => self.esc = true,
            '\'' if !self.dq => self.sq = !self.sq,
            '"' if !self.sq => self.dq = !self.dq,
            _ => {}
        }
    }
}

/// Trim, and collapse runs of whitespace outside quotes to one space.
fn collapse(line: &str) -> String {
    let mut out = String::new();
    let mut q = Quotes::default();
    let mut pending_ws = false;
    for c in line.trim().chars() {
        if q.outside() && c.is_whitespace() {
            pending_ws = true;
            continue;
        }
        if pending_ws {
            out.push(' ');
            pending_ws = false;
        }
        q.step(c);
        out.push(c);
    }
    out
}

/// Every normalised command in the claim fences of `text`, in document order (duplicates kept).
#[must_use]
pub fn claim_commands(text: &str) -> Vec<String> {
    fences(text)
        .iter()
        .filter(|f| f.is_claim())
        .flat_map(|f| normalise(&f.body.join("\n")))
        .collect()
}

/// Is a workflow's `on:` on the merge path? `on` may be a string, a list, or a map keyed by event.
fn on_merge_path(doc: &serde_yaml::Value) -> bool {
    let on = doc
        .get("on")
        .or_else(|| doc.get(serde_yaml::Value::Bool(true)));
    let hit = |s: &str| MERGE_PATH_EVENTS.contains(&s);
    match on {
        Some(serde_yaml::Value::String(s)) => hit(s),
        Some(serde_yaml::Value::Sequence(seq)) => seq.iter().filter_map(|v| v.as_str()).any(hit),
        Some(serde_yaml::Value::Mapping(m)) => m.keys().filter_map(|k| k.as_str()).any(hit),
        _ => false,
    }
}

/// A step's literal `if: false` (the boolean, or the string `false`).
fn disabled(step: &serde_yaml::Value) -> bool {
    match step.get("if") {
        Some(serde_yaml::Value::Bool(false)) => true,
        Some(serde_yaml::Value::String(s)) => s.trim() == "false",
        _ => false,
    }
}

/// The raw `run:` bodies of the enabled steps of the jobs `keep` accepts, by job name.
fn job_runs(doc: &serde_yaml::Value, keep: impl Fn(&str) -> bool) -> Vec<&str> {
    let Some(jobs) = doc.get("jobs").and_then(serde_yaml::Value::as_mapping) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (name, job) in jobs {
        if !name.as_str().is_some_and(&keep) {
            continue;
        }
        let Some(steps) = job.get("steps").and_then(serde_yaml::Value::as_sequence) else {
            continue;
        };
        for step in steps {
            if disabled(step) {
                continue;
            }
            if let Some(run) = step.get("run").and_then(serde_yaml::Value::as_str) {
                out.push(run);
            }
        }
    }
    out
}

/// The normalised `run:` lines of one workflow document (empty unless it is on the merge path).
#[must_use]
pub fn workflow_run_lines(doc: &serde_yaml::Value) -> BTreeSet<String> {
    if !on_merge_path(doc) {
        return BTreeSet::new();
    }
    job_runs(doc, |_| true)
        .into_iter()
        .flat_map(normalise)
        .collect()
}

/// #4433: the five fat CI jobs run the job bodies of `ci/sections.yml` through
/// `scripts/ci/fat_driver.py run --sections '<spec>'`. The spec parts a merge-path workflow names, in order.
#[must_use]
pub fn section_specs(doc: &serde_yaml::Value) -> Vec<String> {
    if !on_merge_path(doc) {
        return Vec::new();
    }
    let mut out = Vec::new();
    for run in job_runs(doc, |_| true) {
        if !run.contains("fat_driver.py") {
            continue;
        }
        let mut words = run.split_whitespace();
        while let Some(w) = words.next() {
            if w == "--sections" {
                if let Some(spec) = words.next() {
                    let spec = spec.trim_matches(|c| c == '\'' || c == '"');
                    out.extend(
                        spec.split(',')
                            .map(str::trim)
                            .filter(|p| !p.is_empty())
                            .map(String::from),
                    );
                }
            }
        }
    }
    out
}

/// fat_driver's `fnmatch` over a section name, for the `*` and `?` its specs use. A matrix pin
/// (`determinism[X64]`) selects its job (`determinism`), whose steps are the pin's.
fn section_selected(name: &str, part: &str) -> bool {
    fn glob(n: &[u8], p: &[u8]) -> bool {
        match (p.first(), n.first()) {
            (None, None) => true,
            (Some(b'*'), _) => glob(n, &p[1..]) || (!n.is_empty() && glob(&n[1..], p)),
            (Some(b'?'), Some(_)) => glob(&n[1..], &p[1..]),
            (Some(a), Some(b)) if a == b => glob(&n[1..], &p[1..]),
            _ => false,
        }
    }
    let part = part.split('[').next().unwrap_or(part);
    glob(name.as_bytes(), part.as_bytes())
}

/// The normalised `run:` lines of the `ci/sections.yml` jobs that `specs` select. A section no merge-path fat
/// job names contributes nothing: a claim it "runs" is not run.
///
/// #4415 moved guard-cargo's guard steps into a manifest job, `guard-cargo-steps`, that GitHub never runs:
/// the selected `guard-cargo` runs them all through its one `bash scripts/ci_guards.sh guard-cargo` step. So a
/// `<X>-steps` job counts as selected when `X` is selected and one of `X`'s steps runs `ci_guards.sh X`.
#[must_use]
pub fn section_run_lines(sections: &serde_yaml::Value, specs: &[String]) -> BTreeSet<String> {
    let selected = |name: &str| specs.iter().any(|p| section_selected(name, p));
    let runs_manifest = |base: &str| {
        let call = format!("scripts/ci_guards.sh {base}");
        job_runs(sections, |n| n == base).iter().any(|r| {
            r.split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .contains(&call)
        })
    };
    job_runs(sections, |name| {
        selected(name)
            || name
                .strip_suffix("-steps")
                .is_some_and(|base| selected(base) && runs_manifest(base))
    })
    .into_iter()
    .flat_map(normalise)
    .collect()
}

/// The CI set: every merge-path `run:` line under `<root>/.github/workflows/`, plus the `run:` lines of the
/// `<root>/ci/sections.yml` jobs a merge-path workflow runs through `fat_driver.py --sections` (#4433). Unparseable
/// files contribute nothing (a workflow GitHub cannot parse runs nothing either).
#[must_use]
pub fn ci_run_lines(root: &Path) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut specs = Vec::new();
    let Ok(rd) = std::fs::read_dir(root.join(".github/workflows")) else {
        return out;
    };
    let mut files: Vec<_> = rd
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("yml" | "yaml")))
        .collect();
    files.sort();
    for f in files {
        let Ok(text) = std::fs::read_to_string(&f) else {
            continue;
        };
        if let Ok(doc) = serde_yaml::from_str::<serde_yaml::Value>(&text) {
            out.extend(workflow_run_lines(&doc));
            specs.extend(section_specs(&doc));
        }
    }
    if let Some(sections) = std::fs::read_to_string(root.join("ci/sections.yml"))
        .ok()
        .and_then(|t| serde_yaml::from_str::<serde_yaml::Value>(&t).ok())
    {
        out.extend(section_run_lines(&sections, &specs));
    }
    out
}

/// What the claim check found in one document: the commands that resolved, and the ones that did not.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClaimCheck {
    pub resolved: BTreeSet<String>,
    pub unresolved: Vec<String>,
}

/// Check every claim in `text` against `ci`.
#[must_use]
pub fn check_claims(text: &str, ci: &BTreeSet<String>) -> ClaimCheck {
    let mut c = ClaimCheck::default();
    for cmd in claim_commands(text) {
        if ci.contains(&cmd) {
            c.resolved.insert(cmd);
        } else if !c.unresolved.contains(&cmd) {
            c.unresolved.push(cmd);
        }
    }
    c
}

/// What one document extractor (`readme`, `llm-context`, `csv`) did: files read, the claim commands that
/// resolved (the MEASURED set F-33 compares), and the files or values it refused.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DocStats {
    pub files_read: usize,
    pub verified_commands: BTreeSet<String>,
    pub errors: Vec<super::gguf::ExtractError>,
}

impl DocStats {
    /// Record a refusal of `file`.
    pub fn refuse(&mut self, file: &str, what: impl Into<String>) {
        self.errors.push(super::gguf::ExtractError {
            file: file.to_string(),
            what: what.into(),
        });
    }
}

/// `(stem, entity.ref, raw YAML)` for every contract whose `entity.type` is `ty` and whose `entity.ref` names a
/// file.
#[must_use]
pub fn entity_docs(contract_dir: &Path, ty: &str) -> Vec<(String, String, serde_yaml::Value)> {
    use super::pv_contract::scalar;
    super::pv_contract::documents(contract_dir)
        .into_iter()
        .filter_map(|(stem, _rel, doc)| {
            let entity = doc.get("entity");
            if scalar(entity.and_then(|e| e.get("type"))).as_deref() != Some(ty) {
                return None;
            }
            let r = scalar(entity.and_then(|e| e.get("ref")))?;
            Some((stem, r, doc))
        })
        .collect()
}

/// A frontmatter value as the scalars it holds: a scalar is one, a sequence is each scalar element.
#[must_use]
pub fn scalars(v: &serde_yaml::Value) -> Vec<String> {
    use super::pv_contract::scalar;
    match v {
        serde_yaml::Value::Sequence(seq) => seq.iter().filter_map(|x| scalar(Some(x))).collect(),
        other => scalar(Some(other)).into_iter().collect(),
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_claim_is_bash_sh_or_shell_by_its_first_token_in_any_case_and_brace_form() {
        for info in [
            "bash",
            "sh",
            "shell",
            "BASH",
            "Shell title=x",
            "{.bash}",
            "{bash}",
            ".sh",
        ] {
            assert!(is_claim_info(info), "{info}");
        }
        for info in ["", "text", "console", "rust", "bashx", "zsh", "yaml bash"] {
            assert!(!is_claim_info(info), "{info}");
        }
    }

    #[test]
    fn fences_are_found_with_backticks_or_tildes_at_any_nesting_depth() {
        let md = "a\n```bash\nmake x\n```\n> ~~~sh\n> cargo y\n> ~~~\n- item\n    ````shell\n    ls\n    ````\n";
        let f = fences(md);
        assert_eq!(f.len(), 3);
        assert_eq!(f[0].body, vec!["make x"]);
        assert_eq!(f[1].body, vec!["cargo y"]);
        assert_eq!(f[2].body, vec!["ls"]);
        assert!(f.iter().all(Fence::is_claim));
    }

    /// #4588: the backtick-in-info rule is for BACKTICK fences only (CommonMark 4.5); a tilde fence may carry one.
    #[test]
    fn only_a_backtick_fence_refuses_a_backtick_in_its_info() {
        assert_eq!(fence_open("```a`b"), None);
        assert_eq!(fence_open("```a b"), Some(('`', 3, "a b".to_string())));
        assert_eq!(fence_open("~~~ a`b"), Some(('~', 3, "a`b".to_string())));
        assert!(fences("```a`b\nx\n").is_empty());
    }

    /// #4588: fence and prose line numbers are 1-based, even on the first line.
    #[test]
    fn fence_and_prose_line_numbers_are_one_based() {
        let f = fences("```bash\nmake\n```\n");
        assert_eq!(f[0].line, 1);
        let f = fences("a\n\n~~~\nx\n~~~\n");
        assert_eq!(f[0].line, 3);
        let p = prose_lines("one\n```\nx\n```\nfive\n");
        assert_eq!(p, vec![(1, "one".to_string()), (5, "five".to_string())]);
    }

    #[test]
    fn a_shorter_or_different_run_does_not_close_a_fence() {
        let md = "````bash\n```\n~~~\nmake z\n````\n";
        let f = fences(md);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].body, vec!["```", "~~~", "make z"]);
    }

    #[test]
    fn normalisation_joins_continuations_drops_comments_and_collapses_outside_quotes() {
        assert_eq!(
            normalise(
                "cargo  test \\\n   -p x   # why\n\n# only a comment\necho 'a  #b'  \"c  d\""
            ),
            vec!["cargo test -p x", "echo 'a  #b' \"c  d\""]
        );
        assert_eq!(normalise("echo a#b"), vec!["echo a#b"]);
    }

    fn wf(yaml: &str) -> BTreeSet<String> {
        workflow_run_lines(&serde_yaml::from_str(yaml).unwrap())
    }

    #[test]
    fn only_merge_path_workflows_and_enabled_steps_count() {
        let on_push = "on: push\njobs:\n  a:\n    steps:\n      - run: make lint\n      - if: false\n        run: make off\n      - if: 'false'\n        run: make off2\n      - if: github.event_name == 'push'\n        run: make cond\n";
        assert_eq!(
            wf(on_push),
            BTreeSet::from(["make lint".to_string(), "make cond".to_string()])
        );
        assert!(wf(
            "on: [workflow_dispatch, schedule]\njobs:\n  a:\n    steps:\n      - run: make x\n"
        )
        .is_empty());
        assert_eq!(wf("on:\n  merge_group:\njobs:\n  a:\n    steps:\n      - run: |\n          make a\n          make b \\\n            --c\n").len(), 2);
        assert_eq!(
            wf("on: [pull_request_target]\njobs:\n  a:\n    steps:\n      - run: x\n").len(),
            1
        );
    }

    #[test]
    fn a_steps_manifest_counts_only_through_its_selected_guard_call() {
        // #4415: guard-cargo-steps is `if: false`; guard-cargo runs it via ci_guards.sh.
        let sections: serde_yaml::Value = serde_yaml::from_str(
            "jobs:\n  guard-cargo:\n    steps:\n      - run: bash scripts/ci_guards.sh guard-cargo\n  guard-cargo-steps:\n    if: false\n    steps:\n      - run: make m\n  lone:\n    steps:\n      - run: make lone\n  lone-steps:\n    steps:\n      - run: make orphan\n",
        )
        .unwrap();
        let specs = vec!["guard-cargo".to_string(), "lone".to_string()];
        let got = section_run_lines(&sections, &specs);
        assert!(got.contains("make m"), "{got:?}");
        assert!(got.contains("make lone"));
        assert!(
            !got.contains("make orphan"),
            "a -steps job with no ci_guards.sh call is not run"
        );
        assert!(!section_run_lines(&sections, &["lone".to_string()]).contains("make m"));
    }

    #[test]
    fn a_section_counts_only_when_a_merge_path_fat_job_names_it() {
        let ci: serde_yaml::Value = serde_yaml::from_str(
            "on: [pull_request]\njobs:\n  x86:\n    steps:\n      - run: >-\n          python3 scripts/ci/fat_driver.py run\n          --sections 'sov.*,guard-*,determinism[X64]'\n          --results r.json\n",
        )
        .unwrap();
        let nightly: serde_yaml::Value = serde_yaml::from_str(
            "on: [schedule]\njobs:\n  n:\n    steps:\n      - run: python3 scripts/ci/fat_driver.py run --sections 'nightly-only'\n",
        )
        .unwrap();
        let specs = section_specs(&ci);
        assert_eq!(specs, vec!["sov.*", "guard-*", "determinism[X64]"]);
        assert!(section_specs(&nightly).is_empty());
        let sections: serde_yaml::Value = serde_yaml::from_str(
            "jobs:\n  guard-tree:\n    steps:\n      - run: make readme\n      - if: false\n        run: make off\n  determinism:\n    steps:\n      - run: make det\n  nightly-only:\n    steps:\n      - run: make nightly\n",
        )
        .unwrap();
        assert_eq!(
            section_run_lines(&sections, &specs),
            BTreeSet::from(["make readme".to_string(), "make det".to_string()])
        );
        assert!(section_run_lines(&sections, &[]).is_empty());
    }

    #[test]
    fn claims_resolve_against_the_ci_set_and_unresolved_are_named_once() {
        let ci = BTreeSet::from(["make lint".to_string()]);
        let md = "```bash\nmake lint\nmake bogus\n```\n```text\nmake other\n```\n```sh\nmake bogus\n```\n";
        let c = check_claims(md, &ci);
        assert_eq!(c.resolved, BTreeSet::from(["make lint".to_string()]));
        assert_eq!(c.unresolved, vec!["make bogus".to_string()]);
    }

    #[test]
    fn frontmatter_headings_and_spans_skip_fences() {
        let md = "---\nkind: library\nschema_version: \"1.0\"\n---\n# Title\n```bash\n# not a heading\n`not/a/span`\n```\n## Two ##\nsee `crates/x/` and ``a`b``\n";
        let fm = frontmatter(md).unwrap().unwrap();
        assert_eq!(fm.get("kind").and_then(|v| v.as_str()), Some("library"));
        assert_eq!(
            headings(md),
            vec![(1, "Title".to_string()), (2, "Two".to_string())]
        );
        assert_eq!(
            code_spans(md),
            vec!["crates/x/".to_string(), "a`b".to_string()]
        );
        assert!(frontmatter("# no\n").unwrap().is_none());
    }

    #[test]
    fn camel_case_of_a_frontmatter_key() {
        assert_eq!(camel("schema_version"), "schemaVersion");
        assert_eq!(camel("contract-count"), "contractCount");
        assert_eq!(camel("kind"), "kind");
    }

    #[test]
    fn a_section_glob_star_needs_the_tail_and_question_mark_takes_any_one_byte() {
        // `*` may absorb bytes, never skip the rest of the pattern.
        assert!(section_selected("determinism", "det*"));
        assert!(section_selected("determinism", "*ism"));
        assert!(!section_selected("determinism", "*x"));
        assert!(!section_selected("", "*x"));
        // `?` is exactly one byte of any value, not a literal '?'.
        assert!(section_selected("ab", "a?"));
        assert!(section_selected("gpu-touched", "gpu?touched"));
        assert!(!section_selected("a", "a?"));
        // a matrix pin selects its job
        assert!(section_selected("determinism", "determinism[X64]"));
    }
}
