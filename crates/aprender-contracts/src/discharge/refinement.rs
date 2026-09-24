//! ONT-001 ONT-3b (#4073) — refinement: a Lean module earns L4 only when `formalization.yaml` names the Rust item it
//! models (`model_of`), that item resolves in the workspace (ONT-3a's `syn` walk), the module's theorems are
//! discharged (PVL-001 EV-8b), and the relation is `extraction` or `simulation`. `test_witnessed` earns L3. A
//! statement with no model earns no L4: that closes PVL F5, "nothing checks the Rust implements what Lean proves".
//!
//! ```yaml
//! models:
//!   - module: ProvableContracts/Theorems/Softmax/Kernel.lean   # relative to the lean dir, as the summary lists it
//!     model_of: trueno::softmax::softmax_row                   # a Rust path the workspace resolves
//!     relation: { kind: extraction, evidence: "how the Lean model was obtained from the Rust" }
//! unrefined_baseline: 42   # theorem-bearing modules with no L4 model; shrink-only
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// How the Lean model relates to the Rust item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// The Lean definition was extracted from the Rust (e.g. Aeneas/Charon).
    Extraction,
    /// A simulation proof relates the Lean model to the Rust semantics.
    Simulation,
    /// Only tests witness the correspondence.
    TestWitnessed,
}

impl Kind {
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "extraction" => Some(Self::Extraction),
            "simulation" => Some(Self::Simulation),
            "test_witnessed" => Some(Self::TestWitnessed),
            _ => None,
        }
    }

    /// The highest verification level this relation can earn (spec Q1 default).
    #[must_use]
    pub fn level(self) -> u8 {
        match self {
            Self::Extraction | Self::Simulation => 4,
            Self::TestWitnessed => 3,
        }
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Extraction => "extraction",
            Self::Simulation => "simulation",
            Self::TestWitnessed => "test_witnessed",
        }
    }
}

/// One `models:` entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Model {
    pub module: String,
    pub model_of: String,
    pub kind: Kind,
    pub evidence: String,
}

/// `formalization.yaml`'s refinement record: the well-formed models, what was malformed, and the baseline.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Record {
    pub models: Vec<Model>,
    pub malformed: Vec<String>,
    pub unrefined_baseline: Option<usize>,
}

/// Read `<lean_dir>/formalization.yaml`'s `models` and `unrefined_baseline`. `Err` only when the file is unreadable.
pub fn load(lean_dir: &Path) -> Result<Record, String> {
    let p = lean_dir.join("formalization.yaml");
    let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    let doc: serde_yaml::Value =
        serde_yaml::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok(parse(&doc))
}

fn parse(doc: &serde_yaml::Value) -> Record {
    let mut rec = Record {
        unrefined_baseline: doc
            .get("unrefined_baseline")
            .and_then(serde_yaml::Value::as_u64)
            .and_then(|n| usize::try_from(n).ok()),
        ..Record::default()
    };
    let entries = doc
        .get("models")
        .and_then(serde_yaml::Value::as_sequence)
        .cloned()
        .unwrap_or_default();
    let mut seen = BTreeSet::new();
    for (i, e) in entries.iter().enumerate() {
        let s = |k: &str| e.get(k).and_then(serde_yaml::Value::as_str).map(str::trim);
        let rel = |k: &str| {
            e.get("relation")
                .and_then(|r| r.get(k))
                .and_then(serde_yaml::Value::as_str)
                .map(str::trim)
        };
        let (Some(module), Some(model_of)) = (s("module"), s("model_of")) else {
            rec.malformed
                .push(format!("models[{i}] needs `module` and `model_of`"));
            continue;
        };
        let Some(kind) = rel("kind").and_then(Kind::parse) else {
            rec.malformed.push(format!(
                "models[{i}] ({module}): relation.kind {:?} is not extraction, simulation or test_witnessed",
                rel("kind")
            ));
            continue;
        };
        let evidence = rel("evidence").unwrap_or_default();
        if module.is_empty() || !model_of.contains("::") || evidence.is_empty() {
            rec.malformed.push(format!(
                "models[{i}] ({module}): needs a module, a `crate::path::item` model_of and relation.evidence"
            ));
            continue;
        }
        if !seen.insert(module.to_string()) {
            rec.malformed
                .push(format!("models[{i}]: {module} is modelled twice"));
            continue;
        }
        rec.models.push(Model {
            module: module.into(),
            model_of: model_of.into(),
            kind,
            evidence: evidence.into(),
        });
    }
    rec
}

/// One model, judged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Judged {
    pub model: Model,
    /// `Err(why)`: `model_of` names no workspace item (a ghost formalization).
    pub resolved: Result<(), String>,
    /// The module is one the discharge summary lists.
    pub in_tree: bool,
}

impl Judged {
    /// The level this model grants its module: none unless it resolves and names a module in the tree.
    #[must_use]
    pub fn level(&self) -> Option<u8> {
        (self.resolved.is_ok() && self.in_tree).then(|| self.model.kind.level())
    }
}

/// Judge every model. `resolve` answers for a Rust path (`crate::module::item`); `tree_modules` are the summary's paths.
pub fn judge(
    models: &[Model],
    tree_modules: &BTreeSet<String>,
    mut resolve: impl FnMut(&str) -> Result<(), String>,
) -> Vec<Judged> {
    models
        .iter()
        .map(|m| Judged {
            resolved: resolve(&m.model_of),
            in_tree: tree_modules.contains(&m.module),
            model: m.clone(),
        })
        .collect()
}

/// The modules whose model earns L4.
#[must_use]
pub fn l4_modules(judged: &[Judged]) -> BTreeSet<String> {
    judged
        .iter()
        .filter(|j| j.level() == Some(4))
        .map(|j| j.model.module.clone())
        .collect()
}

/// The theorem-bearing modules (`module -> theorems`) that no L4 model covers.
#[must_use]
pub fn unrefined<'a>(
    theorem_modules: &'a BTreeMap<String, Vec<String>>,
    l4: &BTreeSet<String>,
) -> Vec<&'a str> {
    theorem_modules
        .iter()
        .filter(|(m, ts)| !ts.is_empty() && !l4.contains(*m))
        .map(|(m, _)| m.as_str())
        .collect()
}

/// A resolver over the workspace at `root` (ONT-3a's walk), as the closure [`judge`] takes.
pub fn workspace_resolver(root: &Path) -> impl FnMut(&str) -> Result<(), String> {
    let ws = crate::ontology::extract::code::Workspace::scan(root);
    let mut cache: BTreeMap<String, Result<(), String>> = BTreeMap::new();
    move |path: &str| {
        cache
            .entry(path.to_string())
            .or_insert_with(|| {
                let Some((module, item)) = path.rsplit_once("::") else {
                    return Err(format!("`{path}` is not a crate::path::item"));
                };
                let mut r = crate::ontology::extract::code::Resolver::new(&ws);
                r.resolve(module, item).map(|_| ()).map_err(|u| u.reason)
            })
            .clone()
    }
}

#[cfg(test)]
#[path = "refinement_tests.rs"]
mod tests;
