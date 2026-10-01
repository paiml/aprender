//! ONT-001 §5 ONT-4c, B.5 — `extract:llm-context`: an LLM context file (`CLAUDE.md`) becomes an
//! `llm:LlmContext` node.
//!
//! - frontmatter `tools: [...]` → one `llm:declaredTool` each (B.5 grades them `in:` Σ's tool names); any other
//!   frontmatter key → `llm:<camelCase>`, which a `closed: true` shape rejects by name;
//! - every `## ` heading → `llm:section` (informational), and — v4.16 D5b — a heading listed under a role in Σ
//!   `llm_context_role_synonyms` ALSO becomes that role's own predicate (`Purpose` → `llm:purposeSection`, …), so
//!   each role is a plain `minCount: 1` on its own path; a heading no role lists contributes to no role;
//! - an inline code span outside fences that is a RELATIVE path (`dir/`, `dir/file.rs`, `.github/workflows/`: no
//!   leading `/` or `~`, no glob, no `..`, no `:`) → `llm:referencedPath` when it exists under the repo root, and
//!   refused as `llm:referencedPath unresolved: <p>` when it does not (`resolves: path`);
//! - every CLAIM-fence command a merge-path workflow runs → `llm:command`; one none runs is refused,
//!   `llm:command unresolved: <cmd>` (`resolves: ci-step`, [`super::claims`]);
//! - every prose line that opens with `Never:` (optionally bulleted, optionally bold) → `llm:neverRule`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::OnceLock;

use regex::Regex;

use crate::ontology::rdf::{iri, Graph, Term, ONT_BASE, RDF_TYPE};

use super::claims::{self, DocStats};

/// The `llm:` vocabulary: `https://ont.paiml.dev/v1alpha1/llm/<name>`.
#[must_use]
pub fn llm(name: &str) -> String {
    format!("{ONT_BASE}llm/{name}")
}

/// Σ `llm_context_role_synonyms`: role → the `## ` headings that satisfy it.
pub type Synonyms = BTreeMap<String, Vec<String>>;

/// Read Σ `llm_context_role_synonyms` from `<contract_dir>/ontology.yaml` (empty when absent).
#[must_use]
pub fn synonyms(contract_dir: &Path) -> Synonyms {
    std::fs::read_to_string(contract_dir.join("ontology.yaml"))
        .ok()
        .and_then(|t| serde_yaml::from_str::<serde_yaml::Value>(&t).ok())
        .and_then(|d| {
            d.get("llm_context_role_synonyms")
                .and_then(|m| serde_yaml::from_value::<Synonyms>(m.clone()).ok())
        })
        .unwrap_or_default()
}

/// `Purpose` → `purposeSection`.
#[must_use]
pub fn role_predicate(role: &str) -> String {
    let mut c = role.chars();
    let head: String = c
        .next()
        .map(|x| x.to_lowercase().collect())
        .unwrap_or_default();
    format!("{head}{}Section", c.as_str())
}

fn path_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^(\.?[A-Za-z0-9_][A-Za-z0-9_.-]*/)+[A-Za-z0-9_.-]*$").expect("static regex")
    })
}

fn never_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)^\s*(?:[-*]\s+)?(?:\*\*)?never(?:\*\*)?:").expect("static regex")
    })
}

/// Is this code span a relative path reference (see the module doc)?
#[must_use]
pub fn is_path_ref(span: &str) -> bool {
    path_re().is_match(span) && !span.split('/').any(|seg| seg == "..")
}

/// What an LLM context file resolves against.
pub struct Ctx<'a> {
    /// Does a repo-relative path exist? (A closure so the positive control needs no filesystem.)
    pub exists: &'a dyn Fn(&str) -> bool,
    pub ci: &'a BTreeSet<String>,
    pub synonyms: &'a Synonyms,
}

/// Frontmatter keys as `llm:` literals (`tools` → `llm:declaredTool`).
fn emit_frontmatter(g: &mut Graph, s: &str, rel: &str, text: &str, stats: &mut DocStats) {
    let fm = match claims::frontmatter(text) {
        Ok(Some(fm)) => fm,
        Ok(None) => return,
        Err(e) => return stats.refuse(rel, e),
    };
    for (k, v) in &fm {
        let Some(key) = k.as_str() else { continue };
        let pred = if key == "tools" {
            llm("declaredTool")
        } else {
            llm(&claims::camel(key))
        };
        for x in claims::scalars(v) {
            g.insert(s.to_string(), pred.clone(), Term::string(x));
        }
    }
}

/// Every level-2 heading as `llm:section`, plus the role predicate of each Σ synonym it matches.
fn emit_sections(g: &mut Graph, s: &str, text: &str, synonyms: &Synonyms) {
    for (level, h) in claims::headings(text) {
        if level != 2 {
            continue;
        }
        g.insert(s.to_string(), llm("section"), Term::string(&h));
        for (role, heads) in synonyms {
            if heads.contains(&h) {
                g.insert(s.to_string(), llm(&role_predicate(role)), Term::string(&h));
            }
        }
    }
}

/// Every distinct path-shaped code span: `llm:referencedPath` when it exists, refused by name when not.
fn emit_paths(g: &mut Graph, s: &str, rel: &str, text: &str, ctx: &Ctx<'_>, stats: &mut DocStats) {
    let mut seen = BTreeSet::new();
    for span in claims::code_spans(text) {
        if !is_path_ref(&span) || !seen.insert(span.clone()) {
            continue;
        }
        if (ctx.exists)(&span) {
            g.insert(s.to_string(), llm("referencedPath"), Term::string(&span));
        } else {
            stats.refuse(rel, format!("llm:referencedPath unresolved: {span}"));
        }
    }
}

/// Every prose `NEVER:` line as `llm:neverRule`.
fn emit_never(g: &mut Graph, s: &str, text: &str) {
    for (_, line) in claims::prose_lines(text) {
        if never_re().is_match(&line) {
            g.insert(s.to_string(), llm("neverRule"), Term::string(line.trim()));
        }
    }
}

/// Claim-fence commands: `llm:command` when a merge-path CI step runs it, refused by name when not.
fn emit_commands(
    g: &mut Graph,
    s: &str,
    rel: &str,
    text: &str,
    ci: &BTreeSet<String>,
    stats: &mut DocStats,
) {
    let check = claims::check_claims(text, ci);
    for cmd in &check.resolved {
        g.insert(s.to_string(), llm("command"), Term::string(cmd));
    }
    for cmd in check.unresolved {
        stats.refuse(rel, format!("llm:command unresolved: {cmd}"));
    }
    stats.verified_commands = check.resolved;
}

/// One context file into `g`.
pub fn emit(g: &mut Graph, stem: &str, rel: &str, text: &str, ctx: &Ctx<'_>) -> DocStats {
    let mut stats = DocStats::default();
    let s = iri("llm", stem);
    g.insert(s.clone(), RDF_TYPE, Term::iri(llm("LlmContext")));
    emit_frontmatter(g, &s, rel, text, &mut stats);
    emit_sections(g, &s, text, ctx.synonyms);
    emit_paths(g, &s, rel, text, ctx, &mut stats);
    emit_never(g, &s, text);
    emit_commands(g, &s, rel, text, ctx.ci, &mut stats);
    stats
}

/// Every `entity: {type: llm-context}` contract under `contract_dir`.
pub fn extract(contract_dir: &Path, g: &mut Graph, ci: &BTreeSet<String>) -> DocStats {
    let root = super::repo_root(contract_dir);
    let syn = synonyms(contract_dir);
    let exists = |p: &str| root.join(p).exists();
    let ctx = Ctx {
        exists: &exists,
        ci,
        synonyms: &syn,
    };
    let mut stats = DocStats::default();
    for (stem, r, _doc) in claims::entity_docs(contract_dir, "llm-context") {
        match std::fs::read_to_string(root.join(&r)) {
            Ok(text) => {
                let one = emit(g, &stem, &r, &text, &ctx);
                stats.files_read += 1;
                stats.verified_commands.extend(one.verified_commands);
                stats.errors.extend(one.errors);
            }
            Err(e) => stats.refuse(&r, format!("unreadable: {e}")),
        }
    }
    stats
}

/// The positive control (R-3): a context file naming a path that does not exist is refused by name, and the
/// same file naming one that does is not — every run, in memory, against an in-memory tree.
#[must_use]
pub fn positive_control() -> bool {
    let exists = |p: &str| p == "src/";
    let ci = BTreeSet::new();
    let syn = Synonyms::new();
    let ctx = Ctx {
        exists: &exists,
        ci: &ci,
        synonyms: &syn,
    };
    let bad = "## Key Files\nsee `src/__pc_llm_context__/`\n";
    let good = "## Key Files\nsee `src/`\n";
    let refused = emit(&mut Graph::new(), "__pc__", "CLAUDE.md", bad, &ctx)
        .errors
        .iter()
        .any(|e| e.what == "llm:referencedPath unresolved: src/__pc_llm_context__/");
    let clean = emit(&mut Graph::new(), "__pc__", "CLAUDE.md", good, &ctx)
        .errors
        .is_empty();
    refused && clean
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(g: &Graph, p: &str) -> Vec<String> {
        g.objects(&iri("llm", "c"), &llm(p))
            .iter()
            .filter_map(|t| t.as_literal().map(|l| l.0.to_string()))
            .collect()
    }

    #[test]
    fn roles_tools_paths_commands_and_never_rules_become_the_node() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let exists = |p: &str| root.join(p).exists();
        let ci = BTreeSet::from(["cargo test".to_string()]);
        let syn = Synonyms::from([
            ("Purpose".to_string(), vec!["Project Overview".to_string()]),
            ("Layout".to_string(), vec!["Key Files".to_string()]),
        ]);
        let md = "---\ntools: [pv, cargo]\n---\n# Top\n## Project Overview\n## Testing\n## Key Files\n`src/` and `/abs/x` and `*.rs` and `a:b/c` and `../up/`\n- **Never:** push to main.\n```bash\ncargo test\n```\n";
        let mut g = Graph::new();
        let st = emit(
            &mut g,
            "c",
            "CLAUDE.md",
            md,
            &Ctx {
                exists: &exists,
                ci: &ci,
                synonyms: &syn,
            },
        );
        assert!(st.errors.is_empty(), "{:?}", st.errors);
        assert_eq!(values(&g, "declaredTool").len(), 2);
        assert_eq!(values(&g, "section").len(), 3);
        assert_eq!(values(&g, "purposeSection"), vec!["Project Overview"]);
        assert_eq!(values(&g, "layoutSection"), vec!["Key Files"]);
        assert!(values(&g, "rulesSection").is_empty());
        assert_eq!(values(&g, "referencedPath"), vec!["src/"]);
        assert_eq!(values(&g, "neverRule").len(), 1);
        assert_eq!(values(&g, "command"), vec!["cargo test"]);
    }

    #[test]
    fn a_missing_path_and_an_unrun_command_are_refused_by_name() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let exists = |p: &str| root.join(p).exists();
        let ci = BTreeSet::new();
        let syn = Synonyms::new();
        let md = "`no/such/path.rs`\n```shell\nmake ghost\n```\n";
        let st = emit(
            &mut Graph::new(),
            "c",
            "CLAUDE.md",
            md,
            &Ctx {
                exists: &exists,
                ci: &ci,
                synonyms: &syn,
            },
        );
        let what: Vec<&str> = st.errors.iter().map(|e| e.what.as_str()).collect();
        assert!(
            what.contains(&"llm:referencedPath unresolved: no/such/path.rs"),
            "{what:?}"
        );
        assert!(
            what.contains(&"llm:command unresolved: make ghost"),
            "{what:?}"
        );
    }

    #[test]
    fn path_refs_are_relative_paths_only() {
        for p in ["src/", "src/lib.rs", ".github/workflows/", "a/b/c.yaml"] {
            assert!(is_path_ref(p), "{p}");
        }
        for p in [
            "/abs/x", "~/x/", "*.rs", "src/*.rs", "../x/", "a:b/c", "lib.rs", "a b/c", "x/…",
        ] {
            assert!(!is_path_ref(p), "{p}");
        }
    }

    #[test]
    fn a_role_is_its_own_predicate() {
        assert_eq!(role_predicate("Purpose"), "purposeSection");
        assert_eq!(role_predicate("Commands"), "commandsSection");
    }

    #[test]
    fn the_positive_control_fires() {
        assert!(positive_control());
    }
}
