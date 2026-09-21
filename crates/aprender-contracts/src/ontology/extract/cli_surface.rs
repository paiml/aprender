//! aprender#3745 S2 (#3777) — `extract:cli-surface`: the release candidate's OWN interface becomes the graph the
//! release cells are derived from.
//!
//! Operator, 2026-09-21: *"the way to prevent testing leakage is tests are dervived from surface and SHACL
//! enforced...WE ARE NOT ENFORCING our interface AT ALL..it is hand coded not derived"*. So nothing here names a
//! verb or a flag: the input is `apr surface --json` (schema `apr-cli-surface/v1`, #3749), emitted by the binary
//! under test and passed to pv as a file, and every node is read from it.
//!
//! | node | IRI | from |
//! |---|---|---|
//! | `cli:Command` (+ `cli:ModelCommand` when an arg has role `model`) | `cli-command/<path…>` | `commands[]` |
//! | `cli:Arg` | `cli-arg/<path…>/<id>` | `commands[].args[]`, and `global_args[]` under `cli-arg/-/<id>` |
//! | `cli:ModeValue` | `cli-mode/<path…>/<id>/<level>` | every level of a `mode` arg (level `unset` = the default) |
//! | `cli:InputShape` | `cli-shape/<path…>/<id>` | every `prompt` / `input-file` arg |
//!
//! Additive fields are allowed within v1 (#3745 amendment 2), so a key this reader does not know is ignored; an
//! unknown `schema` or an unknown `role` is refused BY NAME — a role nobody can interpret is not read as `other`.

use std::collections::BTreeMap;
use std::path::Path;

use crate::ontology::rdf::{iri_path, Graph, Term, ONT_BASE, RDF_TYPE};

pub const SCHEMA: &str = "apr-cli-surface/v1";
/// Every schema this reader accepts: v1.1 adds `commands[].generates` (#3745 amendment 3, additive).
pub const SCHEMAS: [&str; 2] = [SCHEMA, "apr-cli-surface/v1.1"];
/// Every role v1 may carry (#3745 note + amendment 1).
pub const ROLES: [&str; 8] = [
    "model",
    "prompt",
    "input-file",
    "backend",
    "sampling",
    "mode",
    "other",
    "unknown",
];
/// The level every mode arg has first: not passed, i.e. its default. It never conflicts with anything.
pub const UNSET: &str = "unset";

/// The vocabulary root: `https://ont.paiml.dev/v1alpha1/cli/<name>` (`cli:` in a shape).
#[must_use]
pub fn cli(name: &str) -> String {
    format!("{ONT_BASE}cli/{name}")
}

/// One argument as the surface declares it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arg {
    pub id: String,
    pub long: Option<String>,
    pub short: Option<String>,
    pub positional: bool,
    pub required: bool,
    pub value_type: String,
    pub values: Vec<String>,
    pub role: String,
    pub marker: Option<String>,
    pub hidden: bool,
    pub conflicts_with: Vec<String>,
    /// Role `sampling` (#3745, fc 17dfb1291): which sampling knob this arg is — read from the typed group the arg
    /// joined, never its name. `None` on every other arg.
    pub sampling_kind: Option<String>,
}

impl Arg {
    /// How the arg is spelled on a command line: `--long`, else `-s`, else `<id>` for a positional.
    #[must_use]
    pub fn spelling(&self) -> String {
        match (&self.long, &self.short, self.positional) {
            (Some(l), _, _) => format!("--{l}"),
            (None, Some(s), _) => format!("-{s}"),
            _ => format!("<{}>", self.id),
        }
    }

    /// A mode arg's levels, `unset` first: a flag is `unset | true`; a finite set is `unset` (when not required)
    /// plus each value. Any other role has no levels.
    #[must_use]
    pub fn levels(&self) -> Vec<String> {
        if self.role != "mode" {
            return Vec::new();
        }
        let mut out = Vec::new();
        if !self.required {
            out.push(UNSET.to_string());
        }
        if self.value_type == "flag" || self.value_type == "count" {
            out.push("true".to_string());
        } else {
            out.extend(self.values.iter().cloned());
        }
        out
    }
}

/// One subcommand path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Command {
    pub key: String,
    pub path: Vec<String>,
    pub leaf: bool,
    pub hidden: bool,
    pub foreign: bool,
    /// D2 (#3745): does the command GENERATE text — read from a typed marker in S1, never inferred. `None` when
    /// the surface predates the field.
    pub generates: Option<bool>,
    pub args: Vec<Arg>,
}

impl Command {
    #[must_use]
    pub fn takes_model(&self) -> bool {
        self.args.iter().any(|a| a.role == "model")
    }

    /// The input shapes: every prompt / input-file arg, by spelling.
    #[must_use]
    pub fn input_shapes(&self) -> Vec<&Arg> {
        self.args
            .iter()
            .filter(|a| a.role == "prompt" || a.role == "input-file")
            .collect()
    }

    /// The mode args (the command's own, then the globals the surface propagates to every path).
    pub fn mode_args<'a>(&'a self, globals: &'a [Arg]) -> impl Iterator<Item = &'a Arg> {
        self.args
            .iter()
            .chain(globals.iter())
            .filter(|a| a.role == "mode" && !a.levels().is_empty())
    }
}

/// The surface as read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Surface {
    pub file: String,
    pub version: String,
    pub git_sha: String,
    pub global_args: Vec<Arg>,
    pub commands: Vec<Command>,
}

/// A surface this reader refuses, by name. The DECLARATION's fault (exit 3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurfaceError {
    pub file: String,
    pub what: String,
}

impl std::fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.file, self.what)
    }
}

impl std::error::Error for SurfaceError {}

pub fn read(path: &Path) -> Result<Surface, SurfaceError> {
    let file = path.display().to_string();
    let text = std::fs::read_to_string(path).map_err(|e| SurfaceError {
        file: file.clone(),
        what: format!("unreadable: {e}"),
    })?;
    parse(&file, &text)
}

fn str_of(v: &serde_json::Value, key: &str) -> Option<String> {
    v.get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

fn bool_of(v: &serde_json::Value, key: &str) -> bool {
    v.get(key)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn strings(v: &serde_json::Value, key: &str) -> Vec<String> {
    v.get(key)
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_string)
        .collect()
}

/// Parse one surface document. An unknown `schema` or `role` is an error naming it.
pub fn parse(file: &str, text: &str) -> Result<Surface, SurfaceError> {
    let err = |what: String| SurfaceError {
        file: file.to_string(),
        what,
    };
    let v: serde_json::Value =
        serde_json::from_str(text).map_err(|e| err(format!("not JSON: {e}")))?;
    let schema = v
        .get("schema")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if !SCHEMAS.contains(&schema) {
        return Err(err(format!(
            "schema {schema:?} is not {} — refused by name",
            SCHEMAS.join(" or ")
        )));
    }
    let args_of = |node: &serde_json::Value, key: &str, owner: &str| {
        node.get(key)
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .map(|a| parse_arg(a).map_err(|w| err(format!("{owner}: {w}"))))
            .collect::<Result<Vec<Arg>, SurfaceError>>()
    };
    let global_args = args_of(&v, "global_args", "global_args")?;
    let mut commands = Vec::new();
    for c in v
        .get("commands")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
    {
        let key = str_of(c, "key").unwrap_or_default();
        commands.push(Command {
            path: strings(c, "path"),
            leaf: bool_of(c, "leaf"),
            hidden: bool_of(c, "hidden"),
            foreign: bool_of(c, "foreign"),
            generates: c.get("generates").and_then(serde_json::Value::as_bool),
            args: args_of(c, "args", &key)?,
            key,
        });
    }
    let binary = v.get("binary").cloned().unwrap_or_default();
    Ok(Surface {
        file: file.to_string(),
        version: str_of(&binary, "version").unwrap_or_default(),
        git_sha: str_of(&binary, "git_sha").unwrap_or_default(),
        global_args,
        commands,
    })
}

fn parse_arg(a: &serde_json::Value) -> Result<Arg, String> {
    let id = str_of(a, "id").unwrap_or_default();
    let role = str_of(a, "role").unwrap_or_default();
    if !ROLES.contains(&role.as_str()) {
        return Err(format!(
            "arg {id:?} has role {role:?}, which apr-cli-surface/v1 does not define — refused by name"
        ));
    }
    Ok(Arg {
        long: str_of(a, "long"),
        short: str_of(a, "short"),
        positional: bool_of(a, "positional"),
        required: bool_of(a, "required"),
        value_type: str_of(a, "value_type").unwrap_or_default(),
        values: strings(a, "values"),
        role,
        marker: str_of(a, "marker"),
        hidden: bool_of(a, "hidden"),
        conflicts_with: strings(a, "conflicts_with"),
        sampling_kind: str_of(a, "sampling_kind"),
        id,
    })
}

/// What the extraction counted — the receipt's derived-surface numbers and the two shrink-only ratchets.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct SurfaceStats {
    pub commands: usize,
    pub leaves: usize,
    pub model_commands: usize,
    pub generating_commands: usize,
    /// Did every model command's surface say whether it generates (D2)? `false` is a violation at the release.
    pub generates_declared: bool,
    pub by_role: BTreeMap<String, usize>,
    /// Shrink-only ratchet (#3745, cop): args no role type claims yet. Baseline 259, all in foreign subtrees.
    pub unknown_args: usize,
    pub unknown_outside_foreign: usize,
    /// Shrink-only ratchet: model / input-file / prompt POSITIONALS whose stdin form v1 cannot declare.
    pub stdin_undeclared: usize,
}

fn path_of(c: &Command) -> Vec<&str> {
    c.path.iter().map(String::as_str).collect()
}

/// Write the surface's nodes into `g`.
pub fn emit(g: &mut Graph, s: &Surface) -> SurfaceStats {
    let mut st = SurfaceStats {
        generates_declared: true,
        ..SurfaceStats::default()
    };
    for a in &s.global_args {
        emit_arg(g, &["-"], a);
    }
    for c in &s.commands {
        emit_command(g, c, &mut st);
    }
    st
}

fn emit_command(g: &mut Graph, c: &Command, st: &mut SurfaceStats) {
    st.commands += 1;
    let path = path_of(c);
    let n = iri_path("cli-command", &path);
    g.insert(n.clone(), RDF_TYPE, Term::iri(cli("Command")));
    g.insert(n.clone(), cli("key"), Term::string(&c.key));
    g.insert(n.clone(), cli("leaf"), Term::boolean(c.leaf));
    g.insert(n.clone(), cli("hidden"), Term::boolean(c.hidden));
    g.insert(n.clone(), cli("foreign"), Term::boolean(c.foreign));
    if let Some(gen) = c.generates {
        g.insert(n.clone(), cli("generates"), Term::boolean(gen));
    }
    if c.leaf {
        st.leaves += 1;
    }
    if c.takes_model() {
        st.model_commands += 1;
        g.insert(n.clone(), RDF_TYPE, Term::iri(cli("ModelCommand")));
        match c.generates {
            Some(true) => st.generating_commands += 1,
            Some(false) => {}
            None => st.generates_declared = false,
        }
    }
    for a in &c.args {
        *st.by_role.entry(a.role.clone()).or_default() += 1;
        if a.role == "unknown" {
            st.unknown_args += 1;
            if !c.foreign {
                st.unknown_outside_foreign += 1;
            }
        }
        let stdin_capable = matches!(a.role.as_str(), "model" | "input-file" | "prompt");
        if a.positional && stdin_capable {
            st.stdin_undeclared += 1;
        }
        let an = emit_arg(g, &path, a);
        g.insert(n.clone(), cli("arg"), Term::iri(an));
    }
}

fn emit_arg(g: &mut Graph, path: &[&str], a: &Arg) -> String {
    let mut segs: Vec<&str> = path.to_vec();
    segs.push(&a.id);
    let n = iri_path("cli-arg", &segs);
    g.insert(n.clone(), RDF_TYPE, Term::iri(cli("Arg")));
    g.insert(n.clone(), cli("id"), Term::string(&a.id));
    g.insert(n.clone(), cli("spelling"), Term::string(a.spelling()));
    g.insert(n.clone(), cli("role"), Term::string(&a.role));
    g.insert(n.clone(), cli("valueType"), Term::string(&a.value_type));
    g.insert(n.clone(), cli("positional"), Term::boolean(a.positional));
    g.insert(n.clone(), cli("required"), Term::boolean(a.required));
    if let Some(m) = &a.marker {
        g.insert(n.clone(), cli("marker"), Term::string(m));
    }
    for c in &a.conflicts_with {
        g.insert(n.clone(), cli("conflictsWith"), Term::string(c));
    }
    for level in a.levels() {
        let mut s2 = segs.clone();
        s2.push(&level);
        let m = iri_path("cli-mode", &s2);
        g.insert(m.clone(), RDF_TYPE, Term::iri(cli("ModeValue")));
        g.insert(m.clone(), cli("level"), Term::string(&level));
        g.insert(n.clone(), cli("modeValue"), Term::iri(m));
    }
    if a.role == "prompt" || a.role == "input-file" {
        let s = iri_path("cli-shape", &segs);
        g.insert(s.clone(), RDF_TYPE, Term::iri(cli("InputShape")));
        g.insert(s.clone(), cli("spelling"), Term::string(a.spelling()));
        g.insert(n.clone(), cli("inputShape"), Term::iri(s));
    }
    n
}

/// The positive control (R-3, the #3706 Σ-tie), drawn every gate run: a planted two-command surface whose `run`
/// declares a model positional must yield `cli:ModelCommand` for `run` with a role-`model` arg; the same surface
/// with that arg's role changed to `other` must NOT. An extractor that stopped reading `role` — or read it from
/// the arg's NAME — would type both the same, and this goes `not-fired`.
#[must_use]
pub fn positive_control() -> bool {
    let typed = |role: &str| {
        let text = PC_SURFACE.replace("__ROLE__", role);
        let Ok(s) = parse("__pc_surface__.json", &text) else {
            return None;
        };
        let mut g = Graph::new();
        let st = emit(&mut g, &s);
        let run = iri_path("cli-command", &["run"]);
        let is_model = g
            .objects(&run, RDF_TYPE)
            .iter()
            .any(|t| t.as_iri() == Some(cli("ModelCommand").as_str()));
        Some((is_model, st.model_commands))
    };
    matches!(
        (typed("model"), typed("other")),
        (Some((true, 1)), Some((false, 0)))
    )
}

const PC_SURFACE: &str = r#"{"schema":"apr-cli-surface/v1","binary":{"name":"apr","version":"0.0.0-pc","git_sha":"pc"},
"global_args":[{"id":"json","long":"json","value_type":"flag","values":["true","false"],"role":"mode"}],
"commands":[
 {"path":["run"],"key":"run","leaf":true,"hidden":false,"foreign":false,"generates":true,"args":[
   {"id":"model","positional":true,"required":true,"value_type":"path","role":"__ROLE__","marker":"ModelRef"},
   {"id":"prompt","long":"prompt","value_type":"text","role":"prompt","marker":"PromptText"},
   {"id":"chat","long":"chat","value_type":"flag","values":["true","false"],"role":"mode"}]},
 {"path":["list"],"key":"list","leaf":true,"hidden":false,"foreign":false,"args":[]}]}"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_control_fires_and_a_foreign_schema_or_role_is_refused_by_name() {
        assert!(positive_control());
        let e = parse("x.json", r#"{"schema":"apr-cli-surface/v9"}"#).expect_err("refused");
        assert!(e.what.contains("v9") && e.what.contains(SCHEMA), "{e}");
        let bad = PC_SURFACE.replace("__ROLE__", "modelish");
        let e = parse("x.json", &bad).expect_err("unknown role refused");
        assert!(e.what.contains("modelish"), "{e}");
    }

    #[test]
    fn levels_start_at_unset_and_the_ratchets_count_what_v1_cannot_say() {
        let s = parse("pc.json", &PC_SURFACE.replace("__ROLE__", "model")).expect("parses");
        let run = &s.commands[0];
        let modes: Vec<(String, Vec<String>)> = run
            .mode_args(&s.global_args)
            .map(|a| (a.spelling(), a.levels()))
            .collect();
        assert_eq!(
            modes,
            vec![
                ("--chat".to_string(), vec!["unset".into(), "true".into()]),
                ("--json".to_string(), vec!["unset".into(), "true".into()]),
            ],
            "the command's own mode args, then the propagated globals"
        );
        let mut g = Graph::new();
        let st = emit(&mut g, &s);
        assert_eq!(st.model_commands, 1);
        assert_eq!(st.generating_commands, 1);
        assert!(st.generates_declared);
        assert_eq!(st.stdin_undeclared, 1, "the model positional");
        let no_gen = PC_SURFACE
            .replace("__ROLE__", "model")
            .replace(r#""generates":true,"#, "");
        let st2 = emit(
            &mut Graph::new(),
            &parse("pc.json", &no_gen).expect("parses"),
        );
        assert!(
            !st2.generates_declared,
            "a model command that does not say is named"
        );
    }
}
