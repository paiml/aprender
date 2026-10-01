//! ONT-4g (aprender#4476) — `extract:binary`: every cargo bin target of the workspace becomes an `ont:Binary`.
//!
//! The extractor never runs a binary (R-15, §3.7). It reads the tracked snapshot [`SNAPSHOT`] that
//! `scripts/dogfood_surfaces.sh --snapshot` writes: one JSON line per bin target, `schema` = [`SCHEMA`], carrying the
//! package, target, name, source path, the commit it was built from, its `--version` string and the sha256 of that
//! string, the binary and `--help` sha256, and the sorted `commands` / `routes` / `tools` it exposes. A line with any
//! other `schema`, a missing key, or a duplicate `(package, target)` is refused by name (PV-ONT-012).
//!
//! Key: `binary/<package>/<target>` — never the bin name. `apr` is declared by both `apr-cli` and the root facade,
//! and keyed by name the pair would collapse into one node (FALSIFY-BIN-002). Each node is typed `ont:Binary`,
//! `prov:Entity`, and `bin:target/<package>/<target>` — the class a per-binary contract's shape targets.
//!
//! Derived facts, each a triple the shared shapes in `contracts/binary-surface-v1.yaml` hold at `maxCount 0`:
//! - `bin:versionLacksSha` (G0.1): the version string does not contain the first 9 hex of `bin:gitSha`.
//! - `bin:namePairMismatch` (G0.5): on BOTH nodes of a pair that share `bin:name` and differ in help or version sha;
//!   the object is the other node.
//! - `bin:helpFailed` (G1.2): the snapshot says `--help` failed, or recorded the sha256 of empty output.
//! - `bin:unledgered{Command,Route,Tool}` and `bin:ledgerOrphan{,Route}`: the join with [`LEDGER`], resolved by
//!   `bin:name` (the ledger's `binary` column cannot say which `apr` target a row means). A command row is
//!   `"<name> <path> …"`, its path cut at the first token starting with `-`, `<`, `[` or `(`; a bare `"<name>"` or
//!   `"<name> --flag"` row is the binary itself, never an orphan. A route row is `"VERB /path …"`; a tool row is
//!   `"mcp:<tool> …"`. Rows of any other form are counted in `bin:auditRow` and joined to nothing.
//!
//! Census (the extractor-missing check): at a `[workspace]` root the member manifests are read for their bin targets
//! (`[[bin]]` tables plus cargo's `src/main.rs` / `src/bin/*` auto-discovery), and every census target without a
//! snapshot line — or snapshot line naming no census target — is an extractor error. So a dropped line is RED at
//! extract time (FALSIFY-BIN-001), not only in a probe that compares counts. At a `[workspace]` root, a missing or
//! empty snapshot is an error too (vacuity); a fixture corpus with no workspace manifest is not measured.
//!
//! No blank nodes; byte-ordered (R-15).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::code::{manifest_names, workspace_membership};
use super::gguf::ExtractError;
use crate::ontology::rdf::{iri_path, ont, Graph, Term, PROV_ENTITY, RDF_TYPE};

/// The tracked snapshot, relative to the repo root.
pub const SNAPSHOT: &str = "evidence/binary/snapshot.jsonl";
/// The only snapshot schema this build reads.
pub const SCHEMA: &str = "binary-snapshot/v1";
/// The surface-audit ledger, relative to the repo root.
pub const LEDGER: &str = "docs/audits/surface_audit.csv";
/// The ledger header this reader understands; any other header is refused.
pub const LEDGER_HEADER: &str =
    "binary,feature,quality_1_10,verified_hardware,top_competitor,in_dogfood_skill,\
cluster_id,cluster_label,evidence_path,confidence";
/// sha256 of the empty string: a `--help` that printed nothing.
pub const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

const HTTP_VERBS: &[&str] = &["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"];

/// A `bin:*` vocabulary term.
#[must_use]
pub fn bin(name: &str) -> String {
    ont(&format!("bin/{name}"))
}

/// The node of one bin target.
#[must_use]
pub fn node(package: &str, target: &str) -> String {
    iri_path("binary", &[package, target])
}

/// One snapshot line.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Target {
    pub package: String,
    pub target: String,
    pub name: String,
    pub src_path: String,
    pub git_sha: String,
    pub version: String,
    pub version_sha256: String,
    pub binary_sha256: String,
    pub help_sha256: String,
    pub help_failed: bool,
    pub commands: BTreeSet<String>,
    pub routes: BTreeSet<String>,
    pub tools: BTreeSet<String>,
}

impl Target {
    /// G0.1: the version string names the commit (its first 9 hex).
    #[must_use]
    pub fn version_names_sha(&self) -> bool {
        self.git_sha
            .get(..9)
            .is_some_and(|sha9| self.version.contains(sha9))
    }
}

/// What the ledger says about one bin name.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LedgerEntry {
    /// Every row's `feature`, as written.
    pub rows: BTreeSet<String>,
    pub commands: BTreeSet<String>,
    pub routes: BTreeSet<String>,
    pub tools: BTreeSet<String>,
}

/// Counts reported beside the graph.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BinaryStats {
    /// The repo root carries a `[workspace]` manifest — the only case in which the census and vacuity are checked.
    pub at_workspace_root: bool,
    /// Snapshot lines read into nodes.
    pub targets: usize,
    /// Distinct `bin:name`s among them.
    pub names: usize,
    /// Bin targets the member manifests declare (`None` off a workspace root).
    pub census: Option<usize>,
    /// Ledger rows read.
    pub ledger_rows: usize,
    pub version_lacks_sha: usize,
    /// Nodes carrying `bin:namePairMismatch`.
    pub name_pair_mismatch: usize,
    pub help_failed: usize,
    pub unledgered: usize,
    pub orphans: usize,
    /// Nodes with no ledger row at all.
    pub no_audit_row: usize,
    pub errors: Vec<ExtractError>,
}

fn refuse(errors: &mut Vec<ExtractError>, file: &str, what: String) {
    errors.push(ExtractError {
        file: file.to_string(),
        what,
    });
}

fn str_key(v: &serde_json::Value, k: &str) -> Result<String, String> {
    v.get(k)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("key `{k}` is missing or not a string"))
}

fn set_key(v: &serde_json::Value, k: &str) -> Result<BTreeSet<String>, String> {
    let arr = v
        .get(k)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            format!("key `{k}` is missing or not an array (an empty surface is `[]`)")
        })?;
    arr.iter()
        .map(|x| {
            x.as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("`{k}` holds a non-string"))
        })
        .collect()
}

fn hex_of(s: &str, len: usize) -> bool {
    s.len() == len
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// One snapshot line, or why it is refused.
fn parse_line(line: &str) -> Result<Target, String> {
    let v: serde_json::Value = serde_json::from_str(line).map_err(|e| format!("not JSON: {e}"))?;
    let schema = str_key(&v, "schema")?;
    if schema != SCHEMA {
        return Err(format!("schema `{schema}` is not `{SCHEMA}`"));
    }
    let t = Target {
        package: str_key(&v, "package")?,
        target: str_key(&v, "target")?,
        name: str_key(&v, "name")?,
        src_path: str_key(&v, "src_path")?,
        git_sha: str_key(&v, "git_sha")?,
        version: str_key(&v, "version")?,
        version_sha256: str_key(&v, "version_sha256")?,
        binary_sha256: str_key(&v, "binary_sha256")?,
        help_sha256: str_key(&v, "help_sha256")?,
        help_failed: v
            .get("help_failed")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        commands: set_key(&v, "commands")?,
        routes: set_key(&v, "routes")?,
        tools: set_key(&v, "tools")?,
    };
    if t.package.is_empty() || t.target.is_empty() || t.name.is_empty() {
        return Err("package, target and name must be non-empty".into());
    }
    if !hex_of(&t.git_sha, 40) {
        return Err(format!("git_sha `{}` is not 40 lowercase hex", t.git_sha));
    }
    Ok(t)
}

/// The snapshot's targets, byte-ordered by `(package, target)`, plus the lines refused.
#[must_use]
pub fn parse_snapshot(text: &str) -> (Vec<Target>, Vec<ExtractError>) {
    let mut errors = Vec::new();
    let mut by_key: BTreeMap<(String, String), Target> = BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let file = format!("{SNAPSHOT}:{}", i + 1);
        match parse_line(line) {
            Ok(t) => match by_key.entry((t.package.clone(), t.target.clone())) {
                std::collections::btree_map::Entry::Occupied(e) => refuse(
                    &mut errors,
                    &file,
                    format!("duplicate bin target {}/{}", e.key().0, e.key().1),
                ),
                std::collections::btree_map::Entry::Vacant(e) => {
                    e.insert(t);
                }
            },
            Err(what) => refuse(&mut errors, &file, what),
        }
    }
    (by_key.into_values().collect(), errors)
}

/// RFC 4180 records: quoted fields may hold commas, doubled quotes and newlines.
fn csv_records(text: &str) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    let mut rec = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match (quoted, c) {
            (true, '"') if chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            (true, '"') => quoted = false,
            (true, c) => field.push(c),
            (false, '"') => quoted = true,
            (false, ',') => rec.push(std::mem::take(&mut field)),
            (false, '\r') => {}
            (false, '\n') => {
                rec.push(std::mem::take(&mut field));
                out.push(std::mem::take(&mut rec));
            }
            (false, c) => field.push(c),
        }
    }
    if !field.is_empty() || !rec.is_empty() {
        rec.push(field);
        out.push(rec);
    }
    out
}

/// The command path a `"<name> <path> …"` feature names: the tokens after the name, cut at the first token that
/// starts with `-`, `<`, `[` or `(`. `Some("")` is the binary itself; `None` is not a command row of `name`.
#[must_use]
pub fn command_path(name: &str, feature: &str) -> Option<String> {
    let mut tokens = feature.split_whitespace();
    if tokens.next() != Some(name) {
        return None;
    }
    let path: Vec<&str> = tokens
        .take_while(|t| !t.starts_with(['-', '<', '[', '(']))
        .collect();
    Some(path.join(" "))
}

/// `"VERB /path"` for a route feature, or `None`.
#[must_use]
pub fn route_key(feature: &str) -> Option<String> {
    let mut tokens = feature.split_whitespace();
    let verb = tokens.next()?;
    let path = tokens.next()?;
    (HTTP_VERBS.contains(&verb) && path.starts_with('/')).then(|| format!("{verb} {path}"))
}

/// The tool a `"mcp:<tool> …"` feature names, or `None`.
#[must_use]
pub fn tool_key(feature: &str) -> Option<String> {
    let first = feature.split_whitespace().next()?;
    first
        .strip_prefix("mcp:")
        .filter(|t| !t.is_empty())
        .map(str::to_string)
}

/// The ledger by bin name, or why it is refused.
pub fn parse_ledger(text: &str) -> Result<BTreeMap<String, LedgerEntry>, String> {
    let mut records = csv_records(text).into_iter();
    let header = records.next().ok_or("the ledger is empty")?.join(",");
    if header != LEDGER_HEADER {
        return Err(format!("header `{header}` is not the surface-audit header"));
    }
    let mut out: BTreeMap<String, LedgerEntry> = BTreeMap::new();
    for rec in records {
        let (Some(binary), Some(feature)) = (rec.first(), rec.get(1)) else {
            continue;
        };
        if binary.is_empty() {
            continue;
        }
        let e = out.entry(binary.clone()).or_default();
        e.rows.insert(feature.clone());
        if let Some(r) = route_key(feature) {
            e.routes.insert(r);
        } else if let Some(t) = tool_key(feature) {
            e.tools.insert(t);
        } else if let Some(p) = command_path(binary, feature).filter(|p| !p.is_empty()) {
            e.commands.insert(p);
        }
    }
    Ok(out)
}

/// `(package, target)` of every bin target one manifest declares: its `[[bin]]` tables, plus cargo's
/// auto-discovered `src/main.rs` (named for the package) and `src/bin/<x>.rs` / `src/bin/<x>/main.rs` unless
/// `autobins = false` or a `[[bin]]` already uses that file. A `[[bin]]` of the same name replaces the discovered
/// one (cargo's rule), so names are a set.
#[must_use]
pub fn bins_of(dir: &Path, manifest: &str) -> Vec<(String, String)> {
    let Some(package) = manifest_names(manifest).package else {
        return Vec::new();
    };
    let (mut names, paths, autobins) = declared_bins(manifest);
    if autobins {
        // cargo skips a discovered file a declared [[bin]] already uses
        if dir.join("src/main.rs").is_file() && !paths.contains("src/main.rs") {
            names.insert(package.clone());
        }
        if let Ok(rd) = std::fs::read_dir(dir.join("src/bin")) {
            names.extend(
                rd.flatten()
                    .filter_map(|e| discovered_bin(&e.path(), &paths)),
            );
        }
    }
    names.into_iter().map(|n| (package.clone(), n)).collect()
}

/// The `[[bin]]` names and paths one manifest declares, and its `autobins` setting.
fn declared_bins(manifest: &str) -> (BTreeSet<String>, BTreeSet<String>, bool) {
    let mut names = BTreeSet::new();
    let mut paths = BTreeSet::new();
    let mut autobins = true;
    let mut section = String::new();
    for line in manifest.lines() {
        let t = line.trim();
        if t.starts_with('[') && t.ends_with(']') {
            section = t.trim_matches(['[', ']']).trim().to_string();
            continue;
        }
        let Some((k, v)) = t.split_once('=') else {
            continue;
        };
        let v = v.trim().trim_matches('"');
        match (section.as_str(), k.trim()) {
            ("bin", "name") => {
                names.insert(v.to_string());
            }
            ("bin", "path") => {
                paths.insert(v.trim_start_matches("./").to_string());
            }
            ("package", "autobins") => autobins = v != "false",
            _ => {}
        }
    }
    (names, paths, autobins)
}

/// The bin cargo auto-discovers at `src/bin/<x>.rs` or `src/bin/<x>/main.rs`, unless a `[[bin]]` uses that file.
fn discovered_bin(p: &Path, declared_paths: &BTreeSet<String>) -> Option<String> {
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
    let (file, is_target) = if p.is_dir() {
        (
            format!("src/bin/{stem}/main.rs"),
            p.join("main.rs").is_file(),
        )
    } else {
        (
            format!("src/bin/{stem}.rs"),
            p.extension().is_some_and(|e| e == "rs"),
        )
    };
    (is_target && !stem.is_empty() && !declared_paths.contains(&file)).then(|| stem.to_string())
}

/// The bin targets of every member of the workspace under `root`, or `None` when `root` is not a workspace root.
#[must_use]
pub fn census(root: &Path) -> Option<BTreeSet<(String, String)>> {
    let membership = std::fs::read_to_string(root.join("Cargo.toml"))
        .ok()
        .and_then(|t| workspace_membership(&t))?;
    let mut out = BTreeSet::new();
    for manifest in super::example::manifests(root) {
        let dir = manifest.parent().unwrap_or(root);
        if !membership.admits(root, dir) {
            continue;
        }
        if let Ok(text) = std::fs::read_to_string(&manifest) {
            out.extend(bins_of(dir, &text));
        }
    }
    Some(out)
}

/// One target into `g`, joined to its ledger entry and to the targets sharing its name.
pub fn emit(g: &mut Graph, t: &Target, ledger: Option<&LedgerEntry>, pair_mismatch: &[&Target]) {
    let n = node(&t.package, &t.target);
    g.insert(n.clone(), RDF_TYPE, Term::iri(ont("Binary")));
    g.insert(n.clone(), RDF_TYPE, Term::iri(PROV_ENTITY));
    g.insert(
        n.clone(),
        RDF_TYPE,
        Term::iri(bin(&format!("target/{}/{}", t.package, t.target))),
    );
    for (p, v) in [
        ("package", &t.package),
        ("target", &t.target),
        ("name", &t.name),
        ("srcPath", &t.src_path),
        ("gitSha", &t.git_sha),
        ("versionString", &t.version),
        ("versionSha256", &t.version_sha256),
        ("sha256", &t.binary_sha256),
        ("helpSha256", &t.help_sha256),
    ] {
        g.insert(n.clone(), bin(p), Term::string(v.as_str()));
    }
    for c in &t.commands {
        g.insert(n.clone(), bin("command"), Term::string(c.as_str()));
    }
    for r in &t.routes {
        g.insert(n.clone(), bin("route"), Term::string(r.as_str()));
    }
    for x in &t.tools {
        g.insert(n.clone(), bin("mcpTool"), Term::string(x.as_str()));
    }
    if !t.version_names_sha() {
        let sha9 = t.git_sha.get(..9).unwrap_or(&t.git_sha);
        g.insert(n.clone(), bin("versionLacksSha"), Term::string(sha9));
    }
    if t.help_failed || t.help_sha256 == EMPTY_SHA256 {
        g.insert(n.clone(), bin("helpFailed"), Term::boolean(true));
    }
    for other in pair_mismatch {
        g.insert(
            n.clone(),
            bin("namePairMismatch"),
            Term::iri(node(&other.package, &other.target)),
        );
    }
    let empty = LedgerEntry::default();
    let l = ledger.unwrap_or(&empty);
    for row in &l.rows {
        g.insert(n.clone(), bin("auditRow"), Term::string(row.as_str()));
    }
    for (pred, snap, led) in [
        ("unledgeredCommand", &t.commands, &l.commands),
        ("unledgeredRoute", &t.routes, &l.routes),
        ("unledgeredTool", &t.tools, &l.tools),
        ("ledgerOrphan", &l.commands, &t.commands),
        ("ledgerOrphanRoute", &l.routes, &t.routes),
    ] {
        for x in snap.difference(led) {
            g.insert(n.clone(), bin(pred), Term::string(x.as_str()));
        }
    }
}

/// For each target, the targets sharing its name that differ in help or version sha (G0.5).
#[must_use]
pub fn pair_mismatches(targets: &[Target]) -> Vec<Vec<&Target>> {
    targets
        .iter()
        .map(|t| {
            targets
                .iter()
                .filter(|o| {
                    o.name == t.name
                        && !(o.package == t.package && o.target == t.target)
                        && (o.help_sha256 != t.help_sha256 || o.version_sha256 != t.version_sha256)
                })
                .collect()
        })
        .collect()
}

/// Every target into `g`, and the counts.
pub fn emit_all(
    g: &mut Graph,
    targets: &[Target],
    ledger: &BTreeMap<String, LedgerEntry>,
    stats: &mut BinaryStats,
) {
    let pairs = pair_mismatches(targets);
    for (t, mismatch) in targets.iter().zip(&pairs) {
        let l = ledger.get(&t.name);
        emit(g, t, l, mismatch);
        stats.version_lacks_sha += usize::from(!t.version_names_sha());
        stats.name_pair_mismatch += usize::from(!mismatch.is_empty());
        stats.help_failed += usize::from(t.help_failed || t.help_sha256 == EMPTY_SHA256);
        stats.no_audit_row += usize::from(l.is_none_or(|l| l.rows.is_empty()));
        let empty = LedgerEntry::default();
        let l = l.unwrap_or(&empty);
        stats.unledgered += t.commands.difference(&l.commands).count()
            + t.routes.difference(&l.routes).count()
            + t.tools.difference(&l.tools).count();
        stats.orphans +=
            l.commands.difference(&t.commands).count() + l.routes.difference(&t.routes).count();
    }
    stats.targets = targets.len();
    stats.names = targets
        .iter()
        .map(|t| t.name.as_str())
        .collect::<BTreeSet<_>>()
        .len();
}

/// The snapshot and ledger under `root`, into `g`.
pub fn extract_at(root: &Path, g: &mut Graph) -> BinaryStats {
    let census = census(root);
    let mut stats = BinaryStats {
        at_workspace_root: census.is_some(),
        census: census.as_ref().map(BTreeSet::len),
        ..BinaryStats::default()
    };
    let snapshot = std::fs::read_to_string(root.join(SNAPSHOT));
    let (targets, mut errors) = match &snapshot {
        Ok(text) => parse_snapshot(text),
        Err(_) => (Vec::new(), Vec::new()),
    };
    stats.errors.append(&mut errors);
    let ledger = match std::fs::read_to_string(root.join(LEDGER)) {
        Ok(text) => parse_ledger(&text).unwrap_or_else(|what| {
            refuse(&mut stats.errors, LEDGER, what);
            BTreeMap::new()
        }),
        Err(_) => BTreeMap::new(),
    };
    stats.ledger_rows = ledger.values().map(|e| e.rows.len()).sum();
    emit_all(g, &targets, &ledger, &mut stats);
    if let Some(census) = &census {
        // a workspace whose manifests declare no bin target has nothing to snapshot: census and snapshot agree
        // on zero, which is a measurement. The vacuous case is a census that names targets and a snapshot
        // that reads none.
        if targets.is_empty() && !census.is_empty() {
            refuse(
                &mut stats.errors,
                SNAPSHOT,
                format!(
                    "a [workspace] root with {} bin target(s) read zero snapshot lines ({}) — the extractor \
                     measured nothing",
                    census.len(),
                    if snapshot.is_ok() { "empty" } else { "absent" }
                ),
            );
        } else {
            let read: BTreeSet<(String, String)> = targets
                .iter()
                .map(|t| (t.package.clone(), t.target.clone()))
                .collect();
            for (p, t) in census.difference(&read) {
                refuse(
                    &mut stats.errors,
                    SNAPSHOT,
                    format!(
                        "bin target {p}/{t} is declared by its manifest and has no snapshot line"
                    ),
                );
            }
            for (p, t) in read.difference(census) {
                refuse(
                    &mut stats.errors,
                    SNAPSHOT,
                    format!("snapshot line {p}/{t} names no bin target of a workspace member"),
                );
            }
        }
    }
    stats
}

/// The bin targets of the workspace under `contract_dir`'s parent, into `g`.
pub fn extract(contract_dir: &Path, g: &mut Graph) -> BinaryStats {
    extract_at(&super::repo_root(contract_dir), g)
}

/// The planted apr pair [`positive_control`] runs on: two packages declaring `apr`, whose help differs and one of
/// whose versions lacks the commit.
#[must_use]
pub fn control_sample() -> Vec<Target> {
    let sha = "0123456789abcdef0123456789abcdef01234567";
    let base = Target {
        name: "apr".into(),
        git_sha: sha.into(),
        version: "apr 0.70.0 (012345678)".into(),
        help_sha256: "a".repeat(64),
        version_sha256: "b".repeat(64),
        commands: ["run".to_string()].into_iter().collect(),
        ..Target::default()
    };
    let facade = Target {
        package: "aprender".into(),
        target: "apr".into(),
        ..base.clone()
    };
    let cli = Target {
        package: "apr-cli".into(),
        target: "apr".into(),
        version: "apr 0.70.0".into(),
        help_sha256: "c".repeat(64),
        ..base
    };
    vec![cli, facade]
}

/// A planted apr pair fires, this run: exactly two `Binary` nodes; one whose help differs puts
/// `bin:namePairMismatch` on both; a version without the commit puts `bin:versionLacksSha` on its node; and the
/// one ledger row for a command the snapshot lacks is the only orphan. It takes the pair ([`control_sample`]) so a
/// test can hand it a sample that must NOT fire — over the planted pair alone it is always true (#4587).
#[must_use]
pub fn positive_control(targets: &[Target]) -> bool {
    let ledger = parse_ledger(&format!(
        "{LEDGER_HEADER}\napr,apr run,,,,,,,,\napr,apr gone,,,,,,,,\n"
    ))
    .unwrap_or_default();
    let mut g = Graph::default();
    let mut stats = BinaryStats::default();
    emit_all(&mut g, targets, &ledger, &mut stats);
    let a = node("apr-cli", "apr");
    let b = node("aprender", "apr");
    g.instances_of(&ont("Binary")).len() == 2
        && !g.objects(&a, &bin("namePairMismatch")).is_empty()
        && !g.objects(&b, &bin("namePairMismatch")).is_empty()
        && !g.objects(&a, &bin("versionLacksSha")).is_empty()
        && g.objects(&b, &bin("versionLacksSha")).is_empty()
        && g.objects(&b, &bin("ledgerOrphan")).len() == 1
}

#[cfg(test)]
#[path = "binary_tests.rs"]
mod tests;
