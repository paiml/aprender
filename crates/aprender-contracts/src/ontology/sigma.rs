//! ONT-001 §4.1 — Σ, the ontology's own declaration (ONT-2b, PMAT-3471).
//!
//! Σ lives in `contracts/ontology.yaml` and declares what the corpus is allowed to say: the `concepts`, the `roles`
//! (with `domain`/`range`), the `symbols` a `formal:` expression may use, the `worlds`, the `agents`, the
//! `entity_types` and the `extractors` that read them, and the keys the ontology deliberately cannot express
//! (`not_expressible`).
//!
//! **Four ways Σ itself is malformed, all exit 3** (`error:`, never `reject:` — the corpus is not at fault):
//!
//! 1. an `entity_types` entry naming no extractor, or naming one `extractors[]` does not declare;
//! 2. a Σ key no declared reader claims — an anchor nothing reads is decoration, the defect ONT-1 refused;
//! 3. a `not_expressible` entry without a `reader`;
//! 4. an `extractors[]` entry without a `reader`.
//!
//! A `reader` is a DECLARED NAME carried beside `implemented: true|false`, not a path that must resolve today
//! (plan v2 ruling 3): §4.1 gives `entity_types[{name, extractor, implemented}]` and ONT-4c marks four of them
//! `implemented: true` later, so a Σ that could only name built extractors could not be written at all in this row.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::Deserialize;

/// Σ as `contracts/ontology.yaml` carries it.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Sigma {
    pub schema: String,
    #[serde(default)]
    pub concepts: BTreeMap<String, Concept>,
    #[serde(default)]
    pub roles: BTreeMap<String, Role>,
    #[serde(default)]
    pub symbols: Vec<Symbol>,
    #[serde(default)]
    pub worlds: BTreeMap<String, World>,
    #[serde(default)]
    pub agents: Vec<String>,
    #[serde(default)]
    pub entity_types: Vec<EntityTypeDecl>,
    #[serde(default)]
    pub extractors: Vec<ExtractorDecl>,
    #[serde(default)]
    pub not_expressible: Vec<NotExpressible>,
    /// Which reader claims each Σ key. The anti-decoration rule: a key nobody reads is refused (exit 3).
    #[serde(default)]
    pub readers: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Concept {
    pub doc: String,
}

/// A binary relation the corpus may use in `relations:`. `domain`/`range` name concepts.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Role {
    pub domain: String,
    pub range: String,
    #[serde(default)]
    pub symmetric: bool,
    #[serde(default)]
    pub acyclic: bool,
    #[serde(default)]
    pub doc: String,
}

/// One token a `formal:` expression may carry.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    #[serde(default)]
    pub doc: String,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    Operator,
    Function,
    Predicate,
    Constant,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct World {
    pub doc: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EntityTypeDecl {
    pub name: String,
    /// The extractor that reads this entity type. Must name an `extractors[]` entry.
    #[serde(default)]
    pub extractor: String,
    pub implemented: bool,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExtractorDecl {
    pub name: String,
    /// A declared name, with `implemented` saying whether it exists yet (plan v2 ruling 3).
    #[serde(default)]
    pub reader: String,
    pub implemented: bool,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NotExpressible {
    pub key: String,
    #[serde(default)]
    pub reader: String,
}

/// Σ is malformed. Every variant is exit 3 (`error:`): the corpus is not at fault, the declaration is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SigmaError {
    /// `contracts/ontology.yaml` did not parse.
    Parse(String),
    /// An `entity_types` entry names no extractor, or names one `extractors[]` does not declare.
    EntityTypeWithoutExtractor {
        entity_type: String,
        extractor: String,
    },
    /// A Σ key that no `readers` entry claims.
    KeyWithoutReader { key: String },
    /// A `not_expressible` entry without a `reader`.
    NotExpressibleWithoutReader { key: String },
    /// An `extractors[]` entry without a `reader`.
    ExtractorWithoutReader { extractor: String },
}

impl fmt::Display for SigmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "ontology.yaml: {e}"),
            Self::EntityTypeWithoutExtractor {
                entity_type,
                extractor,
            } => {
                if extractor.is_empty() {
                    write!(f, "entity_type `{entity_type}` names no extractor")
                } else {
                    write!(
                        f,
                        "entity_type `{entity_type}` names extractor `{extractor}`, which extractors[] does not declare"
                    )
                }
            }
            Self::KeyWithoutReader { key } => {
                write!(
                    f,
                    "Σ key `{key}` has no reader — a key nothing reads is decoration"
                )
            }
            Self::NotExpressibleWithoutReader { key } => {
                write!(f, "not_expressible `{key}` has no reader")
            }
            Self::ExtractorWithoutReader { extractor } => {
                write!(f, "extractor `{extractor}` has no reader")
            }
        }
    }
}

impl std::error::Error for SigmaError {}

/// The Σ keys that must be claimed by a reader when they are present and non-empty.
pub const READABLE_KEYS: [&str; 8] = [
    "concepts",
    "roles",
    "symbols",
    "worlds",
    "agents",
    "entity_types",
    "extractors",
    "not_expressible",
];

impl Sigma {
    /// Parse Σ. Does NOT check integrity — [`Sigma::check_integrity`] does that, so a caller can report a parse
    /// failure and a malformed declaration differently.
    pub fn from_yaml(yaml: &str) -> Result<Self, SigmaError> {
        serde_yaml::from_str(yaml).map_err(|e| SigmaError::Parse(e.to_string()))
    }

    /// The four malformed-Σ classes of §5 ONT-2b, in the order the row lists them.
    pub fn check_integrity(&self) -> Result<(), SigmaError> {
        // RED: every integrity test must fail here.
        Ok(())
    }

    /// Which Σ keys are present and non-empty — the set `readers` must cover.
    #[must_use]
    pub fn populated_keys(&self) -> BTreeSet<&'static str> {
        let mut keys = BTreeSet::new();
        if !self.concepts.is_empty() {
            keys.insert("concepts");
        }
        if !self.roles.is_empty() {
            keys.insert("roles");
        }
        if !self.symbols.is_empty() {
            keys.insert("symbols");
        }
        if !self.worlds.is_empty() {
            keys.insert("worlds");
        }
        if !self.agents.is_empty() {
            keys.insert("agents");
        }
        if !self.entity_types.is_empty() {
            keys.insert("entity_types");
        }
        if !self.extractors.is_empty() {
            keys.insert("extractors");
        }
        if !self.not_expressible.is_empty() {
            keys.insert("not_expressible");
        }
        keys
    }

    #[must_use]
    pub fn declares_role(&self, _name: &str) -> bool {
        true // RED
    }

    #[must_use]
    pub fn declares_entity_type(&self, _name: &str) -> bool {
        true // RED
    }

    #[must_use]
    pub fn declares_symbol(&self, _token: &str) -> bool {
        true // RED
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Σ small enough to read, with every key populated and claimed.
    fn good() -> &'static str {
        r#"
schema: ont-sigma-v1
concepts:
  Contract: {doc: "a pv contract file"}
  Code: {doc: "a Rust item"}
roles:
  binds: {domain: Contract, range: Code, acyclic: false, doc: "a contract binds a symbol"}
symbols:
  - {name: "∀", kind: operator}
  - {name: isFinite, kind: predicate}
worlds:
  default: {doc: "the tree as committed"}
agents: [pv]
entity_types:
  - {name: pv-contract, extractor: pv_contract, implemented: false}
extractors:
  - {name: pv_contract, reader: ontology/extract/pv_contract.rs, implemented: false}
not_expressible:
  - {key: taste, reader: ontology/sigma.rs}
readers:
  concepts: ontology/sigma.rs
  roles: lint/sigma_gate.rs
  symbols: lint/sigma_symbols.rs
  worlds: ontology/sigma.rs
  agents: ontology/sigma.rs
  entity_types: lint/sigma_gate.rs
  extractors: ontology/sigma.rs
  not_expressible: ontology/sigma.rs
"#
    }

    #[test]
    fn a_well_formed_sigma_parses_and_passes_integrity() {
        let s = Sigma::from_yaml(good()).expect("Σ parses");
        assert_eq!(s.schema, "ont-sigma-v1");
        assert_eq!(s.roles["binds"].domain, "Contract");
        assert!(s.check_integrity().is_ok(), "{:?}", s.check_integrity());
    }

    #[test]
    fn an_entity_type_naming_no_extractor_is_malformed() {
        let y = good().replace(
            "  - {name: pv-contract, extractor: pv_contract, implemented: false}",
            "  - {name: pv-contract, extractor: \"\", implemented: false}",
        );
        let s = Sigma::from_yaml(&y).expect("parses");
        assert_eq!(
            s.check_integrity(),
            Err(SigmaError::EntityTypeWithoutExtractor {
                entity_type: "pv-contract".into(),
                extractor: String::new()
            })
        );
    }

    #[test]
    fn an_entity_type_naming_an_undeclared_extractor_is_malformed() {
        let y = good().replace("extractor: pv_contract,", "extractor: ghost,");
        let s = Sigma::from_yaml(&y).expect("parses");
        assert_eq!(
            s.check_integrity(),
            Err(SigmaError::EntityTypeWithoutExtractor {
                entity_type: "pv-contract".into(),
                extractor: "ghost".into()
            })
        );
    }

    #[test]
    fn a_sigma_key_with_no_reader_is_malformed() {
        let y = good().replace("  symbols: lint/sigma_symbols.rs\n", "");
        let s = Sigma::from_yaml(&y).expect("parses");
        assert_eq!(
            s.check_integrity(),
            Err(SigmaError::KeyWithoutReader {
                key: "symbols".into()
            })
        );
    }

    #[test]
    fn an_extractor_without_a_reader_is_malformed() {
        let y = good().replace(
            "  - {name: pv_contract, reader: ontology/extract/pv_contract.rs, implemented: false}",
            "  - {name: pv_contract, reader: \"\", implemented: false}",
        );
        let s = Sigma::from_yaml(&y).expect("parses");
        assert_eq!(
            s.check_integrity(),
            Err(SigmaError::ExtractorWithoutReader {
                extractor: "pv_contract".into()
            })
        );
    }

    #[test]
    fn a_not_expressible_without_a_reader_is_malformed() {
        let y = good().replace(
            "  - {key: taste, reader: ontology/sigma.rs}",
            "  - {key: taste, reader: \"\"}",
        );
        let s = Sigma::from_yaml(&y).expect("parses");
        assert_eq!(
            s.check_integrity(),
            Err(SigmaError::NotExpressibleWithoutReader {
                key: "taste".into()
            })
        );
    }

    #[test]
    fn an_empty_key_needs_no_reader() {
        // `populated_keys` is what `readers` must cover: a key Σ does not use is not decoration.
        let y = good()
            .replace("agents: [pv]", "agents: []")
            .replace("  agents: ontology/sigma.rs\n", "");
        let s = Sigma::from_yaml(&y).expect("parses");
        assert!(!s.populated_keys().contains("agents"));
        assert!(s.check_integrity().is_ok(), "{:?}", s.check_integrity());
    }

    #[test]
    fn lookups_answer_only_for_what_sigma_declares() {
        let s = Sigma::from_yaml(good()).expect("parses");
        assert!(s.declares_role("binds"));
        assert!(!s.declares_role("refines"), "undeclared role");
        assert!(s.declares_entity_type("pv-contract"));
        assert!(!s.declares_entity_type("csv"), "undeclared entity type");
        assert!(s.declares_symbol("∀"));
        assert!(s.declares_symbol("isFinite"));
        assert!(!s.declares_symbol("⊗"), "undeclared symbol");
    }

    #[test]
    fn an_unknown_key_is_refused_by_the_parser() {
        let y = format!("{}\nsurprise: 1\n", good());
        assert!(matches!(Sigma::from_yaml(&y), Err(SigmaError::Parse(_))));
    }

    #[test]
    fn the_repo_sigma_parses_and_is_well_formed() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/ontology.yaml");
        let text = std::fs::read_to_string(&path).expect("contracts/ontology.yaml is readable");
        let s = Sigma::from_yaml(&text).expect("the repo's Σ parses");
        assert!(s.check_integrity().is_ok(), "{:?}", s.check_integrity());
        assert!(!s.entity_types.is_empty(), "Σ declares entity types");
        assert!(!s.extractors.is_empty(), "Σ declares extractors");
        assert!(
            s.symbols.len() >= 20,
            "Σ declares the corpus's operator vocabulary, got {}",
            s.symbols.len()
        );
    }
}
