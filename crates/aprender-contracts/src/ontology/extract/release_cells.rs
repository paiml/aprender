//! aprender#3745 S2 (#3777) — the release cells, DERIVED from the release candidate's surface. Nothing here names
//! a verb or a flag: `VERBS` is gone, and every cell is a row of a deterministic covering array computed from
//! `apr surface --json` and the hosts' measured inventory.
//!
//! Cop rulings (#3745 issuecomment-5765981405 thread):
//! - **D1**: per (model command, model, host), ONE strength-2 covering array over the command's mode args (its own
//!   and the propagated globals) ∪ {input shape} ∪ — when the command GENERATES — {thinking, context rung}. Every
//!   pair of levels, every (thinking, rung) pair included, appears in some cell.
//! - **D2**: "generates" is the surface's typed `generates` field, never inferred from a name.
//! - A command with no model arg owes one PROBE cell per host: it runs, or refuses by name.
//! - **D3 / S2.5**: per (command, representative model per architecture, host), one BASE cell (every mode arg
//!   unset) and one EFFECT cell per (mode arg, set level), identical to the base but for that one arg — so a flag
//!   that changes nothing is visible as a flag, not lost in a pairwise row.
//!
//! A cell's id is readable and stable: `<command>/<host>/<model file|->/<think-x|->/<rung|->/<shape|->/<args>`,
//! `<args>` being the SET mode args in factor order (`--chat+--format=json`, or `defaults`). The producer reads the
//! derived list (`pv extract --release-* --surface F --cells-out C`) and keys each receipt row by `cell_id`.

use std::collections::BTreeSet;

use crate::ontology::extract::cli_surface::{Arg, Command, Surface, UNSET};
use crate::ontology::extract::covering::{self, Pair};

/// What kind of obligation a cell is.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CellKind {
    /// A row of the D1 covering array.
    Matrix,
    /// A command with no model arg: runs, or refuses by name.
    Probe,
    /// S2.5: the one-factor-at-a-time base (every mode arg unset).
    Base,
    /// S2.5: the base with exactly one mode arg set; `base` is the base cell's id.
    Effect {
        arg: String,
        level: String,
        base: String,
    },
    /// S2.5, a8's sampling controls: one of `t0`, `topk1`, `seed-a`, `seed-a-again`, `seed-b` for a generating
    /// command's typed `sampling` args (the knob values are the controls' own constants, not apr names).
    Sampling { control: String },
}

/// One derived cell — what the producer runs and what the shape grades.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct CellSpec {
    pub id: String,
    pub host: String,
    pub command: String,
    pub generates: bool,
    pub model_sha256: Option<String>,
    pub model_file: Option<String>,
    /// The SET mode args, `(spelling, level)`, in factor order; an unset arg is absent.
    pub args: Vec<(String, String)>,
    pub shape: Option<String>,
    pub thinking: Option<String>,
    pub rung: Option<String>,
    /// The rung's target size for THIS model (a `declared` rung is the model's own length).
    pub rung_tokens: Option<u64>,
    #[serde(flatten)]
    pub kind: CellKind,
}

/// One model a host must prove, as the derivation needs it.
#[derive(Debug, Clone)]
pub struct ModelRef<'a> {
    pub sha: &'a str,
    pub file: &'a str,
    pub arch: Option<&'a str>,
    /// The thinking modes the model owes (#3723 derivation).
    pub modes: Vec<&'static str>,
    /// The rungs it owes, with each rung's target size for this model.
    pub rungs: Vec<(&'a str, Option<u64>)>,
}

/// One required host and its universe.
#[derive(Debug, Clone)]
pub struct HostModels<'a> {
    pub host: &'a str,
    pub models: Vec<ModelRef<'a>>,
}

/// One factor of a command's covering array.
struct Factor {
    /// `Arg(spelling)` for a mode arg; the input shape, thinking and rung are their own kinds.
    kind: FactorKind,
    levels: Vec<String>,
}

#[derive(Clone, PartialEq, Eq)]
enum FactorKind {
    Arg(String),
    Shape,
    Thinking,
    Rung,
}

/// Derive every cell of the release. Deterministic (R-15): commands in surface order, hosts and models in the
/// order given, rows in covering order.
#[must_use]
pub fn derive(surface: &Surface, hosts: &[HostModels<'_>]) -> Vec<CellSpec> {
    let mut out = Vec::new();
    for c in surface.commands.iter().filter(|c| c.leaf) {
        for h in hosts {
            if c.takes_model() {
                for m in &h.models {
                    out.extend(matrix(surface, c, h.host, m));
                }
                out.extend(effects(surface, c, h));
                out.extend(sampling(c, h));
            } else {
                out.push(probe(c, h.host));
            }
        }
    }
    out
}

fn mode_factors<'a>(surface: &'a Surface, c: &'a Command) -> Vec<(&'a Arg, Factor)> {
    let mut seen = BTreeSet::new();
    c.mode_args(&surface.global_args)
        .filter(|a| seen.insert(a.spelling()))
        .map(|a| {
            (
                a,
                Factor {
                    kind: FactorKind::Arg(a.spelling()),
                    levels: a.levels(),
                },
            )
        })
        .collect()
}

/// The forbidden pairs: two args the surface declares `conflicts_with`, both SET (level ≠ 0). Level 0 (`unset`)
/// never conflicts, which is what lets the covering array always complete a row.
fn forbidden(args: &[&Arg], factors: &[Factor]) -> BTreeSet<Pair> {
    let mut out = BTreeSet::new();
    for (i, a) in args.iter().enumerate() {
        for (j, b) in args.iter().enumerate().skip(i + 1) {
            let clash = a.conflicts_with.contains(&b.id) || b.conflicts_with.contains(&a.id);
            if !clash {
                continue;
            }
            for la in 1..factors[i].levels.len() {
                for lb in 1..factors[j].levels.len() {
                    out.insert((i, la, j, lb));
                }
            }
        }
    }
    out
}

fn matrix(surface: &Surface, c: &Command, host: &str, m: &ModelRef<'_>) -> Vec<CellSpec> {
    let modes = mode_factors(surface, c);
    let arg_refs: Vec<&Arg> = modes.iter().map(|(a, _)| *a).collect();
    let mut factors: Vec<Factor> = modes.into_iter().map(|(_, f)| f).collect();
    let fb = forbidden(&arg_refs, &factors);
    let shapes: Vec<String> = c.input_shapes().iter().map(|a| a.spelling()).collect();
    if !shapes.is_empty() {
        factors.push(Factor {
            kind: FactorKind::Shape,
            levels: shapes,
        });
    }
    let generates = c.generates == Some(true);
    if generates {
        factors.push(Factor {
            kind: FactorKind::Thinking,
            levels: m.modes.iter().map(|t| format!("think-{t}")).collect(),
        });
        // a model that owes no rung (its length is under the smallest) still owes its cells, without a rung
        // factor: a factor with zero levels would make the covering array EMPTY, and the model would vanish
        if !m.rungs.is_empty() {
            factors.push(Factor {
                kind: FactorKind::Rung,
                levels: m.rungs.iter().map(|(r, _)| (*r).to_string()).collect(),
            });
        }
    }
    let levels: Vec<usize> = factors.iter().map(|f| f.levels.len()).collect();
    covering::pairwise(&levels, &fb)
        .into_iter()
        .map(|row| cell_from_row(c, host, m, &factors, &row, CellKind::Matrix))
        .collect()
}

fn cell_from_row(
    c: &Command,
    host: &str,
    m: &ModelRef<'_>,
    factors: &[Factor],
    row: &[usize],
    kind: CellKind,
) -> CellSpec {
    let mut spec = CellSpec {
        id: String::new(),
        host: host.to_string(),
        command: c.key.clone(),
        generates: c.generates == Some(true),
        model_sha256: Some(m.sha.to_string()),
        model_file: Some(m.file.to_string()),
        args: Vec::new(),
        shape: None,
        thinking: None,
        rung: None,
        rung_tokens: None,
        kind,
    };
    for (f, &l) in factors.iter().zip(row) {
        let level = f.levels[l].clone();
        match &f.kind {
            FactorKind::Arg(sp) if level != UNSET => spec.args.push((sp.clone(), level)),
            FactorKind::Arg(_) => {}
            FactorKind::Shape => spec.shape = Some(level),
            FactorKind::Thinking => {
                spec.thinking = Some(level.trim_start_matches("think-").to_string())
            }
            FactorKind::Rung => {
                spec.rung_tokens = m
                    .rungs
                    .iter()
                    .find(|(r, _)| *r == level)
                    .and_then(|(_, t)| *t);
                spec.rung = Some(level);
            }
        }
    }
    spec.id = cell_id(&spec);
    spec
}

/// `<command>/<host>/<model file|->/<think-x|->/<rung|->/<shape|->/<args>`.
#[must_use]
pub fn cell_id(s: &CellSpec) -> String {
    let args = if s.args.is_empty() {
        "defaults".to_string()
    } else {
        s.args
            .iter()
            .map(|(a, l)| {
                if l == "true" {
                    a.clone()
                } else {
                    format!("{a}={l}")
                }
            })
            .collect::<Vec<_>>()
            .join("+")
    };
    let prefix = match &s.kind {
        CellKind::Base => "base:",
        CellKind::Effect { .. } => "effect:",
        CellKind::Sampling { .. } => "sampling:",
        CellKind::Matrix | CellKind::Probe => "",
    };
    [
        format!("{prefix}{}", s.command),
        s.host.clone(),
        s.model_file.clone().unwrap_or_else(|| "-".into()),
        s.thinking
            .as_ref()
            .map_or_else(|| "-".into(), |t| format!("think-{t}")),
        s.rung.clone().unwrap_or_else(|| "-".into()),
        s.shape.clone().unwrap_or_else(|| "-".into()),
        args,
    ]
    .join("/")
}

fn probe(c: &Command, host: &str) -> CellSpec {
    let mut spec = CellSpec {
        id: String::new(),
        host: host.to_string(),
        command: c.key.clone(),
        generates: false,
        model_sha256: None,
        model_file: None,
        args: Vec::new(),
        shape: None,
        thinking: None,
        rung: None,
        rung_tokens: None,
        kind: CellKind::Probe,
    };
    spec.id = cell_id(&spec);
    spec
}

/// S2.5: per (command, representative model per architecture, host) — the representative is the first model of
/// each architecture in the host's (sorted) universe — one BASE cell and one EFFECT cell per set level of each
/// mode arg. The base uses the first input shape, `off` when owed (else the first mode) and the smallest rung.
fn effects(surface: &Surface, c: &Command, h: &HostModels<'_>) -> Vec<CellSpec> {
    let mut archs = BTreeSet::new();
    let reps = h
        .models
        .iter()
        .filter(|m| archs.insert(m.arch.unwrap_or("-")));
    let modes = mode_factors(surface, c);
    let mut out = Vec::new();
    for m in reps {
        let mut base = cell_from_row(c, h.host, m, &[], &[], CellKind::Base);
        base.shape = c.input_shapes().first().map(|a| a.spelling());
        if c.generates == Some(true) {
            let off = m
                .modes
                .iter()
                .find(|t| **t == "off")
                .or_else(|| m.modes.first());
            base.thinking = off.map(|t| (*t).to_string());
            if let Some((r, t)) = m.rungs.first() {
                base.rung = Some((*r).to_string());
                base.rung_tokens = *t;
            }
        }
        base.id = cell_id(&base);
        for (_, f) in &modes {
            let FactorKind::Arg(sp) = &f.kind else {
                continue;
            };
            for level in f.levels.iter().filter(|l| l.as_str() != UNSET) {
                let mut e = base.clone();
                e.args = vec![(sp.clone(), level.clone())];
                e.kind = CellKind::Effect {
                    arg: sp.clone(),
                    level: level.clone(),
                    base: base.id.clone(),
                };
                e.id = cell_id(&e);
                out.push(e);
            }
        }
        out.push(base);
    }
    out
}

/// a8's controls need these knobs; each control is the base cell with only these sampling args set.
/// (knob kind, value) per control; a control whose kind the command lacks is not derived (and is named).
pub const SAMPLING_CONTROLS: [(&str, &[(&str, &str)]); 5] = [
    ("t0", &[("temperature", "0")]),
    ("topk1", &[("top_k", "1")]),
    ("seed-a", &[("temperature", "0.8"), ("seed", "1")]),
    ("seed-a-again", &[("temperature", "0.8"), ("seed", "1")]),
    ("seed-b", &[("temperature", "0.8"), ("seed", "2")]),
];

/// The sampling kinds a8's four controls need together.
pub const SAMPLING_KINDS_NEEDED: [&str; 3] = ["temperature", "top_k", "seed"];

/// S2.5 sampling controls: per GENERATING command with typed sampling args, per representative model per
/// architecture, per host — every control whose knobs the command has, on the effect base.
fn sampling(c: &Command, h: &HostModels<'_>) -> Vec<CellSpec> {
    if c.generates != Some(true) {
        return Vec::new();
    }
    let knob = |kind: &str| {
        c.args
            .iter()
            .find(|a| a.role == "sampling" && a.sampling_kind.as_deref() == Some(kind))
            .map(Arg::spelling)
    };
    if !c.args.iter().any(|a| a.role == "sampling") {
        return Vec::new();
    }
    let mut archs = BTreeSet::new();
    let reps = h
        .models
        .iter()
        .filter(|m| archs.insert(m.arch.unwrap_or("-")));
    let mut out = Vec::new();
    for m in reps {
        for (control, knobs) in SAMPLING_CONTROLS {
            let set: Option<Vec<(String, String)>> = knobs
                .iter()
                .map(|(k, v)| knob(k).map(|sp| (sp, (*v).to_string())))
                .collect();
            let Some(args) = set else { continue };
            let mut cell = cell_from_row(
                c,
                h.host,
                m,
                &[],
                &[],
                CellKind::Sampling {
                    control: control.to_string(),
                },
            );
            cell.shape = c.input_shapes().first().map(|a| a.spelling());
            let off = m
                .modes
                .iter()
                .find(|t| **t == "off")
                .or_else(|| m.modes.first());
            cell.thinking = off.map(|t| (*t).to_string());
            if let Some((r, t)) = m.rungs.first() {
                cell.rung = Some((*r).to_string());
                cell.rung_tokens = *t;
            }
            cell.args = args;
            cell.id = format!("{}#{control}", cell_id(&cell));
            out.push(cell);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::extract::cli_surface;

    const SURFACE: &str = r#"{"schema":"apr-cli-surface/v1","binary":{"version":"0.0.0","git_sha":"t"},
"global_args":[{"id":"json","long":"json","value_type":"flag","values":["true","false"],"role":"mode"}],
"commands":[
 {"path":["gen"],"key":"gen","leaf":true,"generates":true,"args":[
   {"id":"model","positional":true,"required":true,"value_type":"path","role":"model"},
   {"id":"prompt","long":"prompt","value_type":"text","role":"prompt"},
   {"id":"input","short":"i","value_type":"path","role":"input-file"},
   {"id":"chat","long":"chat","value_type":"flag","values":["true","false"],"role":"mode","conflicts_with":["raw"]},
   {"id":"raw","long":"raw","value_type":"flag","values":["true","false"],"role":"mode","conflicts_with":["chat"]},
   {"id":"format","long":"format","value_type":"enum","values":["text","json"],"role":"mode"}]},
 {"path":["look"],"key":"look","leaf":true,"generates":false,"args":[
   {"id":"file","positional":true,"required":true,"value_type":"path","role":"model"}]},
 {"path":["group"],"key":"group","leaf":false,"args":[]},
 {"path":["list"],"key":"list","leaf":true,"args":[]}]}"#;

    fn hosts() -> Vec<HostModels<'static>> {
        let m = |sha, file, arch| ModelRef {
            sha,
            file,
            arch: Some(arch),
            modes: vec!["on", "off"],
            rungs: vec![("4k", Some(4096)), ("declared", Some(10_000))],
        };
        vec![HostModels {
            host: "h1",
            models: vec![m("aa", "a.gguf", "qwen35"), m("bb", "b.gguf", "qwen35")],
        }]
    }

    #[test]
    fn the_matrix_covers_every_pair_including_thinking_by_rung_and_never_a_conflict() {
        let s = cli_surface::parse("t.json", SURFACE).expect("parses");
        let cells = derive(&s, &hosts());
        let gen_a: Vec<&CellSpec> = cells
            .iter()
            .filter(|c| {
                c.command == "gen"
                    && c.kind == CellKind::Matrix
                    && c.model_file.as_deref() == Some("a.gguf")
            })
            .collect();
        for t in ["on", "off"] {
            for r in ["4k", "declared"] {
                assert!(
                    gen_a
                        .iter()
                        .any(|c| c.thinking.as_deref() == Some(t) && c.rung.as_deref() == Some(r)),
                    "(think-{t}, {r}) is covered"
                );
            }
        }
        for sh in ["--prompt", "-i"] {
            assert!(gen_a.iter().any(|c| c.shape.as_deref() == Some(sh)), "{sh}");
            assert!(
                gen_a
                    .iter()
                    .any(|c| c.shape.as_deref() == Some(sh)
                        && c.args.iter().any(|(a, _)| a == "--chat")),
                "--chat with {sh}: the #3743 pair (run --prompt --chat) is a cell"
            );
        }
        assert!(
            !gen_a
                .iter()
                .any(|c| c.args.iter().any(|(a, _)| a == "--chat")
                    && c.args.iter().any(|(a, _)| a == "--raw")),
            "conflicts_with pairs never share a cell"
        );
        let decl = gen_a
            .iter()
            .find(|c| c.rung.as_deref() == Some("declared"))
            .expect("declared");
        assert_eq!(decl.rung_tokens, Some(10_000), "the model's own length");
    }

    #[test]
    fn a_non_generating_model_command_owes_no_thinking_or_rung_and_a_modelless_one_owes_a_probe() {
        let s = cli_surface::parse("t.json", SURFACE).expect("parses");
        let cells = derive(&s, &hosts());
        let look: Vec<&CellSpec> = cells
            .iter()
            .filter(|c| c.command == "look" && c.kind == CellKind::Matrix)
            .collect();
        assert!(!look.is_empty());
        assert!(look
            .iter()
            .all(|c| c.thinking.is_none() && c.rung.is_none()));
        let list: Vec<&CellSpec> = cells.iter().filter(|c| c.command == "list").collect();
        assert_eq!(list.len(), 1, "one probe per host");
        assert_eq!(list[0].kind, CellKind::Probe);
        assert_eq!(list[0].id, "list/h1/-/-/-/-/defaults");
        assert!(
            cells.iter().all(|c| c.command != "group"),
            "a non-leaf owes nothing itself"
        );
    }

    #[test]
    fn effect_cells_flip_exactly_one_arg_against_one_base_per_architecture() {
        let s = cli_surface::parse("t.json", SURFACE).expect("parses");
        let cells = derive(&s, &hosts());
        let bases: Vec<&CellSpec> = cells
            .iter()
            .filter(|c| c.command == "gen" && c.kind == CellKind::Base)
            .collect();
        assert_eq!(
            bases.len(),
            1,
            "one qwen35 representative (a.gguf), not one per model"
        );
        assert!(bases[0].args.is_empty());
        let effects: Vec<&CellSpec> = cells
            .iter()
            .filter(|c| c.command == "gen" && matches!(c.kind, CellKind::Effect { .. }))
            .collect();
        // --chat, --raw, --format=text, --format=json, --json
        assert_eq!(effects.len(), 5);
        assert!(effects.iter().all(|e| e.args.len() == 1));
        assert_eq!(derive(&s, &hosts()), cells, "deterministic (R-15)");
        let mut tiny = hosts();
        tiny[0].models[1].rungs.clear();
        let tiny_cells = derive(&s, &tiny);
        assert!(
            tiny_cells
                .iter()
                .any(|c| c.model_file.as_deref() == Some("b.gguf") && c.command == "gen"),
            "a model owing no rung keeps its cells (a zero-level factor must not empty the array)"
        );
        let ids: BTreeSet<&str> = cells.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids.len(), cells.len(), "every id names one cell");
    }
}
