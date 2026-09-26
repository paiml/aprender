//! #3560 R3 — `extract:recipe`: apr-cookbook's `cookbook-recipe/v1` recipes as nodes, derived by pv itself.
//!
//! A contract says `entity: {type: recipe, ref: recipes/apr}` and carries the same `vocabulary` a `json` contract
//! does. Every `*.yaml` below `ref` (recursively, in path-component order) is one recipe, and it becomes one node
//! typed `root_class` (and `prov:Entity`) through the SAME [`super::node`] a `json` document goes through.
//!
//! **Why an extractor and not a JSONL file.** Until R3 the cookbook shaped a DERIVED file,
//! `evidence/recipes/recipes.jsonl`, written by a script outside pv: SHACL (pv's subset) cannot see a list
//! position (`argv[0]`) or join two fields (every `{model:slot}` names a pinned model), so the script computed those
//! facts as scalars and the shape required them. The shape judged the script's copy, and a `--check` step had to
//! prove the copy was current. Here the derivation runs inside the one walk the shapes gate and `pv extract`
//! share (R-18), so the shape judges the recipes themselves and there is no copy to go stale.
//!
//! **The row is the script's row, key for key** (apr-cookbook `scripts/recipes_to_jsonl.py`, #440):
//! `id`, `schema`, `verb` (= `argv[1]`), `recipe_sha256` (of the file's bytes), `argv0_is_apr`,
//! `placeholders_resolve` (the `{model:<slot>}` args name exactly the declared slots), `model_sha256` /
//! `model_ref` (one per slot), `expect_exit`, `has_output_check`, `min_apr`, `host`, `backend`, and
//! `receipt_current_pass` (for EVERY declared host, `receipts/<CURRENT>/<host>/<id>.json` says PASS, names the
//! CURRENT binary's sha, and carries this recipe's sha256). An absent value emits no triple, so `minCount` catches
//! it. The contract's closed shape is unchanged by the move; only its `entity:` block is.
//!
//! **Fail closed.** No recipe under `ref`, or a recipe that is not a YAML mapping, is [`ExtractError::Recipe`] —
//! the declaration's fault, exit 3 — never a smaller corpus graded green.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::{node, scalar, vocabulary, ExtractError, Vocabulary};
use crate::ontology::rdf::Graph;

/// The receipts directory, relative to the repo root, when the contract's `entity` names none.
pub const DEFAULT_RECEIPTS: &str = "receipts";

/// What the walk read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecipeStats {
    /// Recipes extracted, over every `recipe` contract.
    pub recipes: usize,
}

/// Is this contract one this extractor reads? `entity.type == recipe`.
#[must_use]
pub fn applies(doc: &serde_yaml::Value) -> bool {
    scalar(doc.get("entity").and_then(|e| e.get("type"))).as_deref() == Some("recipe")
}

/// Reads every recipe under the contract's `entity.ref` (relative to `root`) and adds one node per recipe to `g`.
/// `Ok(n)` is the number of recipes; `Err` is the declaration's fault.
pub fn extract_into(
    g: &mut Graph,
    stem: &str,
    doc: &serde_yaml::Value,
    root: &Path,
) -> Result<usize, ExtractError> {
    let entity = doc.get("entity");
    let rel =
        scalar(entity.and_then(|e| e.get("ref"))).ok_or_else(|| ExtractError::RefUnreadable {
            contract: stem.to_string(),
            path: String::new(),
            why: "entity.ref is absent".into(),
        })?;
    let receipts = scalar(entity.and_then(|e| e.get("receipts")))
        .unwrap_or_else(|| DEFAULT_RECEIPTS.to_string());
    let vocab = vocabulary(stem, doc)?;
    let dir = root.join(&rel);
    let mut files = Vec::new();
    walk(&dir, &mut files).map_err(|e| ExtractError::RefUnreadable {
        contract: stem.to_string(),
        path: rel.clone(),
        why: e.to_string(),
    })?;
    // Path-component order, as the cookbook's derivation sorted: `a/x` before `a-b/x`.
    files.sort_by_cached_key(|p| {
        p.components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
    });
    let mut recipes = Vec::with_capacity(files.len());
    for f in &files {
        let raw = std::fs::read(f).map_err(|e| recipe_error(stem, root, f, &e.to_string()))?;
        recipes.push((f.clone(), raw));
    }
    extract_recipes(
        g,
        stem,
        &rel,
        &recipes,
        &Receipts::at(&root.join(receipts)),
        &vocab,
    )
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            walk(&path, out)?;
        } else if path.extension().is_some_and(|x| x == "yaml") {
            out.push(path);
        }
    }
    Ok(())
}

fn recipe_error(stem: &str, root: &Path, file: &Path, why: &str) -> ExtractError {
    ExtractError::Recipe {
        contract: stem.to_string(),
        path: file
            .strip_prefix(root)
            .unwrap_or(file)
            .display()
            .to_string(),
        why: why.to_string(),
    }
}

/// Everything after the reads: the recipes' bytes as nodes. Shared by [`extract_into`] and [`positive_control`], so
/// the control exercises the code the gate runs. Staged, so a refused recipe adds nothing of the others.
fn extract_recipes(
    g: &mut Graph,
    stem: &str,
    rel: &str,
    recipes: &[(PathBuf, Vec<u8>)],
    receipts: &Receipts,
    vocab: &Vocabulary,
) -> Result<usize, ExtractError> {
    if recipes.is_empty() {
        return Err(ExtractError::Recipe {
            contract: stem.to_string(),
            path: rel.to_string(),
            why: "no *.yaml recipe below it — an empty corpus would let the shapes gate decline quietly".into(),
        });
    }
    let mut staged = Graph::new();
    for (i, (path, raw)) in recipes.iter().enumerate() {
        let value = row(raw, receipts).map_err(|why| ExtractError::Recipe {
            contract: stem.to_string(),
            path: path.display().to_string(),
            why,
        })?;
        node(
            &mut staged,
            stem,
            &format!("{stem}.{}", i + 1),
            &vocab.root_class,
            true,
            &value,
            vocab,
        )?;
    }
    g.extend(&staged);
    Ok(recipes.len())
}

/// The receipts a recipe's `receipt_current_pass` is judged against: `<dir>/CURRENT` names the release directory
/// (`<version>-<sha>`), and `<dir>/<CURRENT>/<host>/<id>.json` is one recipe's run on one host.
pub(crate) struct Receipts {
    dir: PathBuf,
    current: Option<String>,
}

impl Receipts {
    pub(crate) fn at(dir: &Path) -> Self {
        let current = std::fs::read_to_string(dir.join("CURRENT"))
            .ok()
            .map(|s| s.trim().to_string());
        Self {
            dir: dir.to_path_buf(),
            current,
        }
    }

    /// For EVERY host: the receipt says PASS, names the CURRENT binary's sha, and carries `recipe_sha`. No
    /// CURRENT, no sha in it, or no host at all is `false` — an unjudged recipe is not a passed one.
    fn current_pass(&self, id: &str, hosts: &[String], recipe_sha: &str) -> bool {
        let Some(cur) = self.current.as_deref() else {
            return false;
        };
        let Some((_, sha)) = cur.split_once('-') else {
            return false;
        };
        if sha.is_empty() || hosts.is_empty() {
            return false;
        }
        hosts.iter().all(|h| {
            let p = self.dir.join(cur).join(h).join(format!("{id}.json"));
            let Some(r) = std::fs::read_to_string(p)
                .ok()
                .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            else {
                return false;
            };
            r.get("verdict").and_then(|v| v.as_str()) == Some("PASS")
                && r.get("apr_version")
                    .and_then(|v| v.as_str())
                    .is_some_and(|v| v.contains(sha))
                && r.get("recipe_sha256").and_then(|v| v.as_str()) == Some(recipe_sha)
        })
    }
}

/// One recipe's bytes as its row. `Err` names why the file is not a recipe at all.
fn row(raw: &[u8], receipts: &Receipts) -> Result<serde_json::Value, String> {
    use serde_json::{json, Map, Value};
    let yaml: serde_yaml::Value =
        serde_yaml::from_slice(raw).map_err(|e| format!("not YAML: {e}"))?;
    let r = serde_json::to_value(&yaml).map_err(|e| format!("not a JSON-shaped document: {e}"))?;
    let Value::Object(r) = r else {
        return Err("the root is not a mapping".into());
    };
    let get = |k: &str| r.get(k).cloned().unwrap_or(Value::Null);
    let empty = Map::new();
    let argv = r
        .get("argv")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let models = r.get("models").and_then(Value::as_object).unwrap_or(&empty);
    let expect = r.get("expect").and_then(Value::as_object).unwrap_or(&empty);
    let used: std::collections::BTreeSet<&str> = argv
        .iter()
        .filter_map(Value::as_str)
        .filter_map(model_slot)
        .collect();
    let declared: std::collections::BTreeSet<&str> = models.keys().map(String::as_str).collect();
    let per_model = |k: &str| -> Value {
        if models.is_empty() {
            return Value::Null;
        }
        Value::Array(
            models
                .values()
                .map(|m| m.get(k).cloned().unwrap_or(Value::Null))
                .collect(),
        )
    };
    // `hosts: []` is absent, as the derivation had it (`or None`), so `minCount` catches it.
    let non_empty = |k: &str| match get(k) {
        Value::Array(a) if a.is_empty() => Value::Null,
        v => v,
    };
    let hosts: Vec<String> = r
        .get("hosts")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|h| h.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let host_count = r.get("hosts").and_then(Value::as_array).map_or(0, Vec::len);
    let sha = hex(&Sha256::digest(raw));
    let id = r.get("id").and_then(Value::as_str);
    // A non-string host names no receipt path, so it cannot have passed.
    let receipt_pass =
        id.is_some_and(|id| hosts.len() == host_count && receipts.current_pass(id, &hosts, &sha));
    let output_check = ["stdout_contains", "stdout_json", "stdout_json_is_object"]
        .iter()
        .any(|k| expect.contains_key(*k));
    let row = json!({
        "id": get("id"),
        "schema": get("schema"),
        "verb": argv.get(1).cloned().unwrap_or(Value::Null),
        "recipe_sha256": sha,
        "argv0_is_apr": argv.first().and_then(Value::as_str) == Some("apr"),
        "placeholders_resolve": used == declared,
        "model_sha256": per_model("sha256"),
        "model_ref": per_model("ref"),
        "expect_exit": expect.get("exit").cloned().unwrap_or(Value::Null),
        "has_output_check": output_check,
        "min_apr": get("min_apr"),
        "host": non_empty("hosts"),
        "backend": non_empty("backends"),
        "receipt_current_pass": receipt_pass,
    });
    Ok(row)
}

/// `{model:<slot>}` → `slot`, for a whole argv element only; anything else is not a placeholder.
fn model_slot(arg: &str) -> Option<&str> {
    let slot = arg.strip_prefix("{model:")?.strip_suffix('}')?;
    (!slot.is_empty()
        && slot
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'))
    .then_some(slot)
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

/// The positive control (R-3), in memory every gate run: a recipe whose argv starts `apr` and whose one
/// placeholder names its one model must extract with `argv0_is_apr` and `placeholders_resolve` true; the planted
/// copy whose placeholder names an undeclared slot must extract with `placeholders_resolve` FALSE — the join the
/// JSONL could only carry because a script computed it, now computed here. And an empty corpus is refused.
#[must_use]
pub fn positive_control() -> bool {
    use crate::ontology::rdf::{iri, Term};
    use crate::ontology::shapes::expand;
    let vocab = Vocabulary {
        prefix: "pc".into(),
        root_class: "pc:Recipe".into(),
        nested: Vec::new(),
    };
    let none = Receipts {
        dir: PathBuf::new(),
        current: None,
    };
    let recipe = |slot: &str| {
        format!("id: r\nargv: [apr, run, \"{{model:{slot}}}\"]\nmodels: {{main: {{sha256: x}}}}\n")
            .into_bytes()
    };
    let resolves = |raw: Vec<u8>| -> Option<bool> {
        let mut g = Graph::new();
        extract_recipes(
            &mut g,
            "__pc_recipe__",
            "pc",
            &[(PathBuf::from("r.yaml"), raw)],
            &none,
            &vocab,
        )
        .ok()?;
        let s = iri("pc", "__pc_recipe__.1");
        let flag = |k: &str| {
            g.objects(&s, &expand(&format!("pc:{k}")))
                .first()
                .map(|t| **t == Term::boolean(true))
        };
        let typed = g.instances_of(&expand("pc:Recipe")).contains(&s.as_str());
        (flag("argv0_is_apr")? && typed).then_some(flag("placeholders_resolve")?)
    };
    let empty_refused = matches!(
        extract_recipes(&mut Graph::new(), "__pc_recipe__", "pc", &[], &none, &vocab),
        Err(ExtractError::Recipe { .. })
    );
    resolves(recipe("main")) == Some(true)
        && resolves(recipe("draft")) == Some(false)
        && empty_refused
}

#[cfg(test)]
#[path = "recipe_tests.rs"]
mod tests;
