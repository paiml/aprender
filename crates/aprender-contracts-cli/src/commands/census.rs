//! `pv census` — ONT-001 v4.3 ONT-1: ONE cardinality over the contract corpus,
//! reported `by_kind`, `by_entity_type` and `by_anchoring`.
//!
//! WHY THIS WALKS WITH `pv lint`, AND WHY IT READS RAW YAML TOO.
//!
//! ONT-001 §1 quote-freezes three different corpus sizes — 1818, 1460, 1331 —
//! because three readers each had their own idea of what a contract file is.
//! This command therefore collects with `provable_contracts::lint`'s walker, the
//! one `pv lint` and `pv validate` already share: same directory exclusions, same
//! `is_contract_yaml` rule. A census that counted its own universe would just be
//! a fourth number.
//!
//! The ANCHORING breakdown cannot come from the typed `Contract`, though.
//! `Contract` is not `#[serde(deny_unknown_fields)]`, so serde silently DROPS an
//! `entity:` block it does not model (measured 2026-09-14: `pv validate` prints
//! "Contract is valid" for a contract whose `entity:` it ignored). A census built
//! only on the struct would report `unanchored` for every contract FOREVER,
//! including after the corpus was anchored, and ONT-001 §11.2's ratchet reading it
//! would record 0 while the work was done. So each file is parsed twice: through
//! the shared parser, for the file rule and `by_kind`, and as raw text, for the
//! anchor the struct cannot see.
//!
//! R-2 applies to the instrument itself:
//!
//! - nothing to read → `ZeroContracts`, exit 2, `decline: 0 contracts under <path>`;
//! - a file that will not parse → `ParseErrors`, exit 1,
//!   `reject: N parse error(s) under <path>`.
//!
//! A census of 4 over 3 readable contracts is the shape R-2 exists to refuse: the
//! count would include a file nobody read.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

use provable_contracts::lint::collect_yaml_files;
use provable_contracts::schema::parse_contract;
/// The declaration's shape, defined ONCE — in the schema module, beside the
/// `pv validate` rules that check it (PMAT-1098). It used to be declared here,
/// which made `pv census` and `pv validate` two readers each holding half of
/// what the file is; `EXT-CORPORA-009` now validates THROUGH this struct, so a
/// declaration pv calls valid is one this command can count, by construction.
pub use provable_contracts::schema::{parse_external_corpora_str, ExternalCorpus};

use crate::contract_walk::{ParseErrors, ZeroContracts};

/// Schema id of `contracts/census.json`.
pub const SCHEMA: &str = "ont.paiml.dev/census/v1alpha1";

/// Runs the timing baseline is defined over (ONT-001 §5 ONT-1). The values stay
/// `null` until PVL EV-9's `provable-ladder` job measures them on the CI host
/// class; R-12 is unarmed until then. A number measured HERE would be a number
/// from the wrong host, and would also make two runs of this command disagree —
/// which the ONT-1 probe (tracked census == fresh census) would catch as drift.
pub const TIMING_RUNS: usize = 5;

/// Contracts the census does not walk, kept for the count only.
const QUARANTINE_DIR: &str = "quarantine";

/// How a contract names the thing it is a contract for (ONT-001 §0.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anchoring {
    /// No `entity:` block. Not a defect: R-5 makes `entity:` optional by design.
    Unanchored,
    /// `entity: { type }` — a shape over every entity of that type.
    Class,
    /// `entity: { type, ref }` — one named thing.
    Instance,
}

/// `by_anchoring` — the three anchoring levels of §0.0, always all three keys.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AnchoringCounts {
    pub unanchored: usize,
    pub class: usize,
    pub instance: usize,
}

/// `timing` — the baseline's definition, not a measurement taken here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Timing {
    pub census_cpu_ms_p50: Option<u64>,
    pub lint_cpu_ms_p50: Option<u64>,
    pub host_class: Option<String>,
    pub n_runs: usize,
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            census_cpu_ms_p50: None,
            lint_cpu_ms_p50: None,
            host_class: None,
            n_runs: TIMING_RUNS,
        }
    }
}

/// The counts ONT-1 owes, in the order `contracts/census.json` carries them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Census {
    pub schema: String,
    /// ALWAYS null, and that is the decision, not an omission: the ONT-1 probe
    /// requires the tracked census to equal a fresh one byte for byte, and a
    /// commit sha is unknowable for the commit that will contain it — PVL-001
    /// recorded the same defect for `discharge.json` ("bytes change per run →
    /// verdict lapse every run"). `id_set_sha256` is the content address, and it
    /// is checkable without a commit. Operator ruling, 2026-09-16.
    pub git_sha: Option<String>,
    pub n_files: usize,
    pub n_parsed: usize,
    /// Always 0 in a TRACKED census, by construction: a file that will not parse
    /// makes the whole run a reject (exit 1), so no census.json can carry one.
    /// The field exists because the probe asserts the identity below, which is
    /// what makes "nothing was silently skipped" checkable.
    pub n_parse_errors: usize,
    pub parse_errors: Vec<String>,
    pub quarantined_n: usize,
    pub by_kind: BTreeMap<String, usize>,
    pub by_entity_type: BTreeMap<String, usize>,
    pub by_anchoring: AnchoringCounts,
    /// sha256 over the sorted, unique contract ids (file stems), newline
    /// separated. Two corpora with the same ids hash the same; adding, removing
    /// or renaming one changes it.
    pub id_set_sha256: String,
    pub declared_external: Vec<ExternalCorpus>,
    pub timing: Timing,
}

/// Classify ONE raw YAML document. Pure, so the case table can pin every row
/// without touching a filesystem.
#[must_use]
pub fn classify(yaml: &str) -> (Anchoring, Option<String>) {
    let Some(block) = entity_block(yaml) else {
        return (Anchoring::Unanchored, None);
    };
    let ty = scalar_field(&block, "type");
    let has_ref = scalar_field(&block, "ref").is_some();
    match (ty, has_ref) {
        (Some(t), true) => (Anchoring::Instance, Some(t)),
        (Some(t), false) => (Anchoring::Class, Some(t)),
        // `entity:` present but naming no type is not an anchor. Reporting it as
        // one would let a malformed block inflate the ratchet.
        (None, _) => (Anchoring::Unanchored, None),
    }
}

/// The text of the top-level `entity:` block, inline or nested, or None.
fn entity_block(yaml: &str) -> Option<String> {
    let mut lines = yaml.lines();
    while let Some(line) = lines.next() {
        let Some(rest) = line.strip_prefix("entity:") else {
            continue;
        };
        let rest = rest.trim();
        if rest.is_empty() {
            // nested: `entity:` alone, the block is the indented lines below it
            return Some(indented_block(&mut lines));
        }
        // inline: `entity: { type: code, ref: src/x.rs }`
        return Some(rest.to_string());
    }
    None
}

/// The run of more-indented lines that follow a bare `key:`, flattened.
fn indented_block<'a>(lines: &mut impl Iterator<Item = &'a str>) -> String {
    let mut block = String::new();
    for next in lines {
        if next.trim().is_empty() {
            continue;
        }
        if !next.starts_with([' ', '\t']) {
            break;
        }
        block.push_str(next.trim());
        block.push('\n');
    }
    block
}

/// `type: code` / `{ type: code }` -> "code".
fn scalar_field(block: &str, key: &str) -> Option<String> {
    let needle = format!("{key}:");
    let idx = block.find(&needle)?;
    let after = &block[idx + needle.len()..];
    let val: String = after
        .trim_start()
        .chars()
        .take_while(|c| !matches!(c, ',' | '}' | '\n' | '#'))
        .collect();
    let val = val.trim().trim_matches(['"', '\'']).to_string();
    if val.is_empty() {
        None
    } else {
        Some(val)
    }
}

fn stem_of(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// sha256 over the sorted, unique ids, newline separated.
fn id_set_sha256(ids: &BTreeSet<String>) -> String {
    let mut hasher = Sha256::new();
    for id in ids {
        hasher.update(id.as_bytes());
        hasher.update(b"\n");
    }
    hasher
        .finalize()
        .iter()
        .fold(String::with_capacity(64), |mut s, b| {
            use std::fmt::Write;
            let _ = write!(s, "{b:02x}");
            s
        })
}

/// `contracts/external-corpora.yaml` beside the corpus, or an empty declaration.
/// A malformed file is an ERROR, never an empty list: "no external corpora" and
/// "the declaration could not be read" are different facts.
fn declared_external(root: &Path) -> Result<Vec<ExternalCorpus>, Box<dyn std::error::Error>> {
    let path = root.join("external-corpora.yaml");
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(&path)?;
    let parsed = parse_external_corpora_str(&text)
        .map_err(|e| format!("{} does not parse: {e}", path.display()))?;
    let mut corpora = parsed.corpora;
    corpora.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(corpora)
}

/// Walk `dir` with `pv lint`'s file rule and count. Returns the census, or the
/// refusal the corpus earned (see the module docs).
pub fn census_of(dir: &Path) -> Result<Census, Box<dyn std::error::Error>> {
    let mut all = Vec::new();
    if dir.is_dir() {
        collect_yaml_files(dir, &mut all);
    }
    let mut files = all;
    if files.is_empty() {
        return Err(ZeroContracts {
            path: dir.to_path_buf(),
            filter: None,
        }
        .into());
    }
    files.sort();
    let mut census = empty_census(files.len(), quarantined_n(dir));
    let mut errors = Vec::new();
    let mut ids = BTreeSet::new();
    for path in &files {
        tally(path, &mut census, &mut ids, &mut errors);
    }
    if !errors.is_empty() {
        return Err(ParseErrors {
            path: dir.to_path_buf(),
            files: files.len(),
            errors,
        }
        .into());
    }
    census.id_set_sha256 = id_set_sha256(&ids);
    census.declared_external = declared_external(dir)?;
    Ok(census)
}

/// Contracts held OUT of the corpus, counted but never parsed. The shared walker
/// skips `quarantine/`, so this is its own scan: a number the census reports is a
/// number the census measured.
fn quarantined_n(root: &Path) -> usize {
    let mut out = Vec::new();
    let dir = root.join(QUARANTINE_DIR);
    if dir.is_dir() {
        collect_quarantined(&dir, &mut out);
    }
    out.len()
}

fn collect_quarantined(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_quarantined(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
            out.push(path);
        }
    }
}

fn empty_census(n_files: usize, quarantined_n: usize) -> Census {
    Census {
        schema: SCHEMA.to_string(),
        git_sha: None,
        n_files,
        n_parsed: 0,
        n_parse_errors: 0,
        parse_errors: Vec::new(),
        quarantined_n,
        by_kind: BTreeMap::new(),
        by_entity_type: BTreeMap::new(),
        by_anchoring: AnchoringCounts::default(),
        id_set_sha256: String::new(),
        declared_external: Vec::new(),
        timing: Timing::default(),
    }
}

/// One file: the shared parser decides whether it counts, the raw text decides
/// what it is anchored to.
fn tally(
    path: &Path,
    census: &mut Census,
    ids: &mut BTreeSet<String>,
    errors: &mut Vec<(PathBuf, String)>,
) {
    let contract = match parse_contract(path) {
        Ok(c) => c,
        Err(e) => {
            errors.push((path.to_path_buf(), e.to_string()));
            census.n_parse_errors += 1;
            census.parse_errors.push(path.display().to_string());
            return;
        }
    };
    census.n_parsed += 1;
    ids.insert(stem_of(path));
    *census
        .by_kind
        .entry(contract.kind().to_string())
        .or_insert(0) += 1;
    let Ok(text) = std::fs::read_to_string(path) else {
        census.by_anchoring.unanchored += 1;
        return;
    };
    let (anchoring, ty) = classify(&text);
    match anchoring {
        Anchoring::Unanchored => census.by_anchoring.unanchored += 1,
        Anchoring::Class => census.by_anchoring.class += 1,
        Anchoring::Instance => census.by_anchoring.instance += 1,
    }
    if let Some(t) = ty {
        *census.by_entity_type.entry(t).or_insert(0) += 1;
    }
}

/// The tracked `contracts/census.json` bytes: deterministic, one trailing
/// newline. Two runs over one tree are byte-identical — the ONT-1 probe compares
/// the tracked file with a fresh run.
pub fn render_json(census: &Census) -> Result<String, Box<dyn std::error::Error>> {
    let mut json = serde_json::to_string_pretty(census)?;
    json.push('\n');
    Ok(json)
}

fn render_table(census: &Census) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(out, "== pv census (ONT-001 ONT-1) ==");
    let _ = writeln!(
        out,
        "contracts: {} parsed ({} file(s), {} parse error(s), {} quarantined)",
        census.n_parsed, census.n_files, census.n_parse_errors, census.quarantined_n
    );
    let _ = writeln!(out, "id_set_sha256: {}", census.id_set_sha256);
    let _ = writeln!(out, "\nby_anchoring");
    let _ = writeln!(
        out,
        "  unanchored {:>6}   (no entity: — a law, pattern or policy; optional by R-5)",
        census.by_anchoring.unanchored
    );
    let _ = writeln!(
        out,
        "  class      {:>6}   (entity: {{type}} — a shape over every entity of that type)",
        census.by_anchoring.class
    );
    let _ = writeln!(
        out,
        "  instance   {:>6}   (entity: {{type, ref}} — one named thing)",
        census.by_anchoring.instance
    );
    let _ = writeln!(out, "\nby_entity_type");
    if census.by_entity_type.is_empty() {
        let _ = writeln!(
            out,
            "  (none — no contract in this corpus names an entity type)"
        );
    }
    for (k, v) in &census.by_entity_type {
        let _ = writeln!(out, "  {k:<28} {v:>6}");
    }
    let _ = writeln!(out, "\nby_kind");
    for (k, v) in &census.by_kind {
        let _ = writeln!(out, "  {k:<28} {v:>6}");
    }
    for ext in &census.declared_external {
        let _ = writeln!(
            out,
            "\ndeclared external: {} — {} file(s), NOT in the count above ({})",
            ext.name,
            ext.n_files,
            ext.mark.as_deref().unwrap_or("[U]")
        );
    }
    out
}

/// `pv census <dir> [--format json]`.
pub fn run(contract_dir: &Path, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    let census = census_of(contract_dir)?;
    if json {
        print!("{}", render_json(&census)?);
    } else {
        print!("{}", render_table(&census));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract_walk::{exit_code_for, verdict_for, ZERO_CONTRACTS_EXIT};

    fn write_valid(dir: &Path, name: &str) {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("fixture dir is creatable");
        }
        std::fs::write(
            &path,
            "metadata:\n  version: 1.0.0\n  description: ONT-1 fixture\n",
        )
        .expect("fixture contract is writable");
    }

    fn corpus(names: &[&str]) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().expect("temp dir is creatable");
        for name in names {
            write_valid(tmp.path(), name);
        }
        tmp
    }

    // ---- ONT-001 §5 ONT-1: the two refusals -------------------------------------

    /// R-2 on the instrument: nothing measured is a DECLINE, exit 2, and the line
    /// says so in PVL-001 §0's vocabulary. `error:` at exit 1 claims a measurement.
    #[test]
    fn an_empty_corpus_declines_at_exit_2() {
        let tmp = tempfile::tempdir().expect("temp dir is creatable");
        let err = census_of(tmp.path()).expect_err("an empty corpus is refused");
        assert_eq!(
            exit_code_for(err.as_ref()),
            ZERO_CONTRACTS_EXIT,
            "an empty corpus must decline (exit 2), not fail: {err}"
        );
        assert_eq!(verdict_for(err.as_ref()), "decline");
        assert_eq!(
            err.to_string(),
            format!("0 contracts under {}", tmp.path().display())
        );
    }

    /// A file the census cannot parse is a REJECT, never a counted contract: the
    /// alternative is a cardinality that includes what was never read.
    #[test]
    fn a_parse_error_rejects_at_exit_1_and_is_never_counted() {
        let tmp = corpus(&["a.yaml", "b.yaml", "c.yaml"]);
        std::fs::write(tmp.path().join("garbage.yaml"), "{{{ not yaml at all: [\n")
            .expect("fixture file is writable");
        let err = census_of(tmp.path())
            .expect_err("a corpus with an unparseable file is rejected, never censused");
        assert_eq!(
            exit_code_for(err.as_ref()),
            1,
            "a parse error rejects: {err}"
        );
        assert_eq!(verdict_for(err.as_ref()), "reject");
        assert_eq!(
            err.to_string().lines().next().unwrap_or_default(),
            format!("1 parse error under {}", tmp.path().display())
        );
    }

    // ---- the schema contracts/census.json carries --------------------------------

    #[test]
    fn n_files_equals_n_parsed_plus_n_parse_errors() {
        let tmp = corpus(&["a.yaml", "b.yaml", "nested/c.yaml"]);
        let c = census_of(tmp.path()).expect("a valid corpus censuses");
        assert_eq!(c.n_files, 3);
        assert_eq!(c.n_parsed + c.n_parse_errors, c.n_files);
        assert_eq!(c.n_parse_errors, 0);
        assert!(c.parse_errors.is_empty());
    }

    /// The walker's own rule, not a second one: `binding.yaml` is not a contract.
    #[test]
    fn the_shared_walker_decides_what_a_contract_file_is() {
        let tmp = corpus(&["a.yaml", "binding.yaml", "kaizen/k.yaml"]);
        let c = census_of(tmp.path()).expect("a valid corpus censuses");
        assert_eq!(
            c.n_files, 1,
            "binding.yaml and kaizen/ are excluded by provable_contracts::lint's walker"
        );
    }

    #[test]
    fn git_sha_is_null_and_the_schema_is_named() {
        let tmp = corpus(&["a.yaml"]);
        let c = census_of(tmp.path()).expect("a valid corpus censuses");
        assert_eq!(
            c.git_sha, None,
            "a commit sha is unknowable for its own commit"
        );
        assert_eq!(c.schema, SCHEMA);
    }

    #[test]
    fn the_timing_baseline_declares_five_runs_and_measures_nothing_here() {
        let tmp = corpus(&["a.yaml"]);
        let c = census_of(tmp.path()).expect("a valid corpus censuses");
        assert_eq!(c.timing.n_runs, 5);
        assert_eq!(c.timing.census_cpu_ms_p50, None);
        assert_eq!(c.timing.lint_cpu_ms_p50, None);
        assert_eq!(c.timing.host_class, None);
    }

    #[test]
    fn id_set_sha256_is_64_hex_stable_and_moves_with_the_id_set() {
        let tmp = corpus(&["a.yaml", "b.yaml"]);
        let first = census_of(tmp.path()).expect("censuses");
        let again = census_of(tmp.path()).expect("censuses");
        assert_eq!(first.id_set_sha256, again.id_set_sha256);
        assert_eq!(first.id_set_sha256.len(), 64);
        assert!(first.id_set_sha256.chars().all(|c| c.is_ascii_hexdigit()));
        write_valid(tmp.path(), "c.yaml");
        let after = census_of(tmp.path()).expect("censuses");
        assert_ne!(first.id_set_sha256, after.id_set_sha256);
    }

    #[test]
    fn by_kind_uses_the_contract_kind_vocabulary() {
        let tmp = corpus(&["a.yaml"]);
        std::fs::write(
            tmp.path().join("p.yaml"),
            "metadata:\n  version: 1.0.0\n  kind: pattern\n  description: fixture\n",
        )
        .expect("fixture is writable");
        let c = census_of(tmp.path()).expect("censuses");
        assert_eq!(c.by_kind.get("pattern"), Some(&1));
        assert_eq!(c.by_kind.get("kernel"), Some(&1), "kind defaults to kernel");
    }

    #[test]
    fn quarantined_contracts_are_counted_and_not_censused() {
        let tmp = corpus(&["a.yaml", "quarantine/broken.yaml"]);
        let c = census_of(tmp.path()).expect("censuses");
        assert_eq!(c.n_files, 1);
        assert_eq!(c.quarantined_n, 1);
    }

    #[test]
    fn declared_external_is_read_from_the_declaration_and_empty_without_one() {
        let tmp = corpus(&["a.yaml"]);
        assert!(census_of(tmp.path())
            .expect("censuses")
            .declared_external
            .is_empty());
        std::fs::write(
            tmp.path().join("external-corpora.yaml"),
            "schema: ont.paiml.dev/external-corpora/v1alpha1\ncorpora:\n  - name: archived\n    n_files: 397\n    counted_by: gh api ...\n",
        )
        .expect("declaration is writable");
        let c = census_of(tmp.path()).expect("censuses");
        assert_eq!(c.declared_external.len(), 1);
        assert_eq!(c.declared_external[0].n_files, 397);
        assert_eq!(
            c.n_files, 1,
            "a declared external corpus is never added to the cardinality"
        );
    }

    /// The probe compares the tracked census with a fresh one, byte for byte.
    #[test]
    fn render_json_is_deterministic_and_ends_in_one_newline() {
        let tmp = corpus(&["a.yaml", "b.yaml"]);
        let first = render_json(&census_of(tmp.path()).expect("censuses")).expect("renders");
        let again = render_json(&census_of(tmp.path()).expect("censuses")).expect("renders");
        assert_eq!(first, again);
        assert!(first.ends_with("}\n"));
        let parsed: serde_json::Value = serde_json::from_str(&first).expect("valid JSON");
        for key in [
            "schema",
            "git_sha",
            "n_files",
            "n_parsed",
            "n_parse_errors",
            "parse_errors",
            "quarantined_n",
            "by_kind",
            "by_entity_type",
            "by_anchoring",
            "id_set_sha256",
            "declared_external",
            "timing",
        ] {
            assert!(parsed.get(key).is_some(), "census.json must carry {key}");
        }
        assert!(parsed["git_sha"].is_null());
    }

    // ---- anchoring, unchanged from the classifier's own case table ---------------

    #[test]
    fn absent_entity_is_unanchored() {
        assert_eq!(classify("id: X\nkind: Kernel\n").0, Anchoring::Unanchored);
    }

    #[test]
    fn inline_type_only_is_class_level() {
        let (a, t) = classify("entity: { type: readme }\n");
        assert_eq!(a, Anchoring::Class);
        assert_eq!(t.as_deref(), Some("readme"));
    }

    #[test]
    fn inline_type_and_ref_is_instance_level() {
        let (a, t) = classify("entity: { type: readme, ref: README.md }\n");
        assert_eq!(a, Anchoring::Instance);
        assert_eq!(t.as_deref(), Some("readme"));
    }

    #[test]
    fn nested_block_is_read_too() {
        let (a, t) = classify("id: X\nentity:\n  type: gguf\n  ref: m.gguf\nshape: {}\n");
        assert_eq!(a, Anchoring::Instance);
        assert_eq!(t.as_deref(), Some("gguf"));
    }

    #[test]
    fn nested_type_only_is_class_level() {
        let (a, t) = classify("entity:\n  type: csv\nshape: {}\n");
        assert_eq!(a, Anchoring::Class);
        assert_eq!(t.as_deref(), Some("csv"));
    }

    /// An `entity:` block naming no type is NOT an anchor. Counting it would let
    /// a malformed block inflate the §11.2 ratchet.
    #[test]
    fn entity_without_a_type_is_not_an_anchor() {
        assert_eq!(
            classify("entity: { ref: README.md }\n").0,
            Anchoring::Unanchored
        );
    }

    /// `entity:` must be top-level. An indented one belongs to another key.
    #[test]
    fn indented_entity_is_not_the_top_level_block() {
        assert_eq!(
            classify("metadata:\n  entity: { type: code }\n").0,
            Anchoring::Unanchored
        );
    }

    #[test]
    fn quoted_values_are_unquoted() {
        let (_, t) = classify("entity: { type: \"apr-model\", ref: 'm.apr' }\n");
        assert_eq!(t.as_deref(), Some("apr-model"));
    }

    #[test]
    fn a_trailing_comment_is_not_part_of_the_type() {
        let (_, t) = classify("entity:\n  type: sqlite  # the db\n");
        assert_eq!(t.as_deref(), Some("sqlite"));
    }

    /// R-2 on the instrument: an unreadable corpus declines, never censuses zero.
    #[test]
    fn a_missing_directory_is_a_decline_not_a_zero_census() {
        let err = census_of(Path::new("/nonexistent/contracts")).expect_err("refused");
        assert_eq!(exit_code_for(err.as_ref()), ZERO_CONTRACTS_EXIT);
    }
}
