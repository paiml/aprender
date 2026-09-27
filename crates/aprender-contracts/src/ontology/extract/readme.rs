//! ONT-001 §5 ONT-4c, B.4 — `extract:readme`: a `README.md` becomes a `readme:Readme` node.
//!
//! The frontmatter (emitted by `readme_gen` / `scripts/readme_sync.sh`, never hand-written) maps key by key:
//! `schema_version` → `readme:schemaVersion`, `kind` → `readme:kind`, each of `entrypoints` → `readme:entrypoint`
//! (`resolves: path` — it must exist under the repo root), `contract_count` → `readme:contractCount`
//! (`resolves: census` — it must equal `<contract_dir>/census.json` `n_files`); any other key becomes
//! `readme:<camelCase>`, which a `closed: true` shape rejects by name. Every heading outside a fence is a
//! `readme:section`. Every CLAIM-fence command ([`super::claims`]) that a merge-path workflow runs is a
//! `readme:verifiedCommand` (`resolves: ci-step`); one that none runs is refused, named:
//! `readme:verifiedCommand unresolved: <cmd>`.

use std::collections::BTreeSet;
use std::path::Path;

use crate::ontology::rdf::{iri, Graph, Term, ONT_BASE, RDF_TYPE};

use super::claims::{self, DocStats};

/// The `readme:` vocabulary: `https://ont.paiml.dev/v1alpha1/readme/<name>`.
#[must_use]
pub fn readme(name: &str) -> String {
    format!("{ONT_BASE}readme/{name}")
}

/// What a README's claims and frontmatter resolve against.
pub struct Ctx<'a> {
    pub root: &'a Path,
    pub ci: &'a BTreeSet<String>,
    /// `census.json` `n_files`, when the census is readable.
    pub census_n: Option<u64>,
}

/// `<contract_dir>/census.json` `n_files`.
#[must_use]
pub fn census_n(contract_dir: &Path) -> Option<u64> {
    let text = std::fs::read_to_string(contract_dir.join("census.json")).ok()?;
    serde_json::from_str::<serde_json::Value>(&text)
        .ok()?
        .get("n_files")?
        .as_u64()
}

/// `entrypoints`: each path as `readme:entrypoint` when it exists, refused by name when not.
fn emit_entrypoints(
    g: &mut Graph,
    s: &str,
    rel: &str,
    v: &serde_yaml::Value,
    ctx: &Ctx<'_>,
    stats: &mut DocStats,
) {
    for p in claims::scalars(v) {
        if ctx.root.join(&p).exists() {
            g.insert(s.to_string(), readme("entrypoint"), Term::string(&p));
        } else {
            stats.refuse(rel, format!("readme:entrypoint unresolved: {p}"));
        }
    }
}

/// `contract_count`: `readme:contractCount`, refused unless it equals `census.json` `n_files`.
fn emit_contract_count(
    g: &mut Graph,
    s: &str,
    rel: &str,
    v: &serde_yaml::Value,
    ctx: &Ctx<'_>,
    stats: &mut DocStats,
) {
    let Some(n) = v.as_u64() else {
        return stats.refuse(rel, "readme:contractCount is not a non-negative integer");
    };
    g.insert(s.to_string(), readme("contractCount"), Term::integer(n));
    match ctx.census_n {
        Some(c) if c == n => {}
        Some(c) => stats.refuse(
            rel,
            format!("readme:contractCount unresolved: {n} != census.json n_files {c}"),
        ),
        None => stats.refuse(
            rel,
            format!("readme:contractCount unresolved: {n}, census.json unreadable"),
        ),
    }
}

fn emit_frontmatter(
    g: &mut Graph,
    s: &str,
    rel: &str,
    fm: &serde_yaml::Mapping,
    ctx: &Ctx<'_>,
    stats: &mut DocStats,
) {
    for (k, v) in fm {
        let Some(key) = k.as_str() else { continue };
        match key {
            "entrypoints" => emit_entrypoints(g, s, rel, v, ctx, stats),
            "contract_count" => emit_contract_count(g, s, rel, v, ctx, stats),
            other => {
                let pred = readme(&claims::camel(other));
                for x in claims::scalars(v) {
                    g.insert(s.to_string(), pred.clone(), Term::string(x));
                }
            }
        }
    }
}

/// One README into `g`. Returns what it resolved and refused.
pub fn emit(g: &mut Graph, stem: &str, rel: &str, text: &str, ctx: &Ctx<'_>) -> DocStats {
    let mut stats = DocStats::default();
    let s = iri("readme", stem);
    g.insert(s.clone(), RDF_TYPE, Term::iri(readme("Readme")));
    match claims::frontmatter(text) {
        Ok(Some(fm)) => emit_frontmatter(g, &s, rel, &fm, ctx, &mut stats),
        Ok(None) => {}
        Err(e) => stats.refuse(rel, e),
    }
    for (_, h) in claims::headings(text) {
        g.insert(s.clone(), readme("section"), Term::string(h));
    }
    let check = claims::check_claims(text, ctx.ci);
    for cmd in &check.resolved {
        g.insert(s.clone(), readme("verifiedCommand"), Term::string(cmd));
    }
    for cmd in check.unresolved {
        stats.refuse(rel, format!("readme:verifiedCommand unresolved: {cmd}"));
    }
    stats.verified_commands = check.resolved;
    stats
}

/// Every `entity: {type: readme}` contract under `contract_dir`.
pub fn extract(contract_dir: &Path, g: &mut Graph, ci: &BTreeSet<String>) -> DocStats {
    let root = super::repo_root(contract_dir);
    let ctx = Ctx {
        root: &root,
        ci,
        census_n: census_n(contract_dir),
    };
    let mut stats = DocStats::default();
    for (stem, r, _doc) in claims::entity_docs(contract_dir, "readme") {
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

/// The positive control (R-3): a claim fence whose command no merge-path workflow runs is refused by name,
/// while the same document with that fence relabelled `text` is not — every run, in memory.
#[must_use]
pub fn positive_control() -> bool {
    let ci = BTreeSet::from(["make lint".to_string()]);
    let ctx = Ctx {
        root: Path::new("."),
        ci: &ci,
        census_n: None,
    };
    let bad =
        "---\nschema_version: \"1.0\"\n---\n# R\n```{.Bash}\nmake lint\nmake __pc_readme__\n```\n";
    let good = bad.replacen("```{.Bash}", "```text", 1);
    let mut g = Graph::new();
    let refused = emit(&mut g, "__pc__", "README.md", bad, &ctx)
        .errors
        .iter()
        .any(|e| e.what == "readme:verifiedCommand unresolved: make __pc_readme__");
    let clean = emit(&mut Graph::new(), "__pc__", "README.md", &good, &ctx)
        .errors
        .is_empty();
    refused && clean
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx<'a>(root: &'a Path, ci: &'a BTreeSet<String>, n: Option<u64>) -> Ctx<'a> {
        Ctx {
            root,
            ci,
            census_n: n,
        }
    }

    fn values(g: &Graph, s: &str, p: &str) -> Vec<String> {
        g.objects(s, &readme(p))
            .iter()
            .filter_map(|t| t.as_literal().map(|l| l.0.to_string()))
            .collect()
    }

    #[test]
    fn frontmatter_headings_and_resolved_claims_become_the_node() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let ci = BTreeSet::from(["cargo test".to_string()]);
        let md = "---\nschema_version: \"1.0\"\nkind: library\nentrypoints: [src]\ncontract_count: 7\n---\n# A\n## B\n### C\n```bash\ncargo   test   # all\n```\n";
        let mut g = Graph::new();
        let st = emit(&mut g, "r", "README.md", md, &ctx(root, &ci, Some(7)));
        assert!(st.errors.is_empty(), "{:?}", st.errors);
        let s = iri("readme", "r");
        assert_eq!(values(&g, &s, "schemaVersion"), vec!["1.0"]);
        assert_eq!(values(&g, &s, "kind"), vec!["library"]);
        assert_eq!(values(&g, &s, "entrypoint"), vec!["src"]);
        assert_eq!(values(&g, &s, "contractCount"), vec!["7"]);
        assert_eq!(values(&g, &s, "section").len(), 3);
        assert_eq!(values(&g, &s, "verifiedCommand"), vec!["cargo test"]);
        assert_eq!(
            st.verified_commands,
            BTreeSet::from(["cargo test".to_string()])
        );
    }

    #[test]
    fn an_unrun_claim_a_missing_entrypoint_and_a_stale_count_are_each_refused_by_name() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let ci = BTreeSet::new();
        let md = "---\nentrypoints: [no/such/dir]\ncontract_count: 3\n---\n```sh\nmake nope\n```\n";
        let st = emit(
            &mut Graph::new(),
            "r",
            "README.md",
            md,
            &ctx(root, &ci, Some(4)),
        );
        let what: Vec<&str> = st.errors.iter().map(|e| e.what.as_str()).collect();
        assert!(
            what.contains(&"readme:verifiedCommand unresolved: make nope"),
            "{what:?}"
        );
        assert!(
            what.contains(&"readme:entrypoint unresolved: no/such/dir"),
            "{what:?}"
        );
        assert!(
            what.iter()
                .any(|w| w.contains("3 != census.json n_files 4")),
            "{what:?}"
        );
    }

    #[test]
    fn an_undeclared_frontmatter_key_becomes_a_camel_case_predicate_a_closed_shape_can_name() {
        let ci = BTreeSet::new();
        let md = "---\nschema_version: \"1.0\"\nextra_key: x\n---\n";
        let mut g = Graph::new();
        emit(
            &mut g,
            "r",
            "README.md",
            md,
            &ctx(Path::new("."), &ci, None),
        );
        assert_eq!(values(&g, &iri("readme", "r"), "extraKey"), vec!["x"]);
    }

    #[test]
    fn the_positive_control_fires() {
        assert!(positive_control());
    }
}
