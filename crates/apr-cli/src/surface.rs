//! `apr surface`: the interface the binary actually has, emitted by the binary
//! (#3745 S1, PMAT-3749).
//!
//! Operator, 2026-09-21, verbatim: "the way to prevent testing leakage is tests
//! are dervived from surface and SHACL enforced...WE ARE NOT ENFORCING our
//! interface AT ALL..it is hand coded not derived".
//!
//! Every release cell used to come from a list someone typed: the ladder's
//! rungs, `{run, chat, serve, code}`, `release_evidence.rs`'s `VERBS`, the
//! hand-maintained `apr-cli-commands-v1.yaml`. A verb, flag or input shape
//! nobody typed was invisible to the gate, and 0.69.0 leaked three that way
//! (`serve` on Qwen3.5 #3571, `code` #3719, `run --prompt P --chat` #3743).
//!
//! This module walks `<Cli as CommandFactory>::command()` and returns every
//! subcommand path and every argument, with the argument's ROLE read from how it
//! is BUILT:
//!
//! * `model` / `prompt` / `input-file` / `other`: the value parser's output type
//!   is one of the role types in `batuta_common::cli_roles` (compared by
//!   `TypeId`, never by name);
//! * `backend`: the argument belongs to the group the shared [`BackendArg`]
//!   flatten creates (`<BackendArg as clap::Args>::group_id()`);
//! * `mode`: structural, meaning a flag action or a finite value set;
//! * `unknown`: a free-form value (raw `PathBuf`, or raw `String` without a
//!   finite value set) that no role type declares. [`unknown_outside_foreign`]
//!   is the marker guard: it must be empty.
//!
//! The JSON (`apr-cli-surface/v1`, agreed on #3745) is produced from the binary
//! under test at gate time and is never committed. [`emit`] is the library entry
//! point that in-process gates call; `apr surface --json` is a thin wrapper
//! over it, and a test pins the two equal.

use std::any::TypeId;
use std::path::PathBuf;

use batuta_common::cli_roles::{self, Carrier, Role};
use clap::{ArgAction, CommandFactory};
use serde::Serialize;

use crate::{BackendArg, Cli};

/// The schema string consumers refuse to read if it is not one they know.
///
/// v1.1 (cop ruling on #3745, after S2 found `serve run`'s prompts arrive over
/// HTTP) adds `commands[].generates`.
pub const SCHEMA: &str = "apr-cli-surface/v1.1";

/// Stack for the recursive walk. Building and walking the full `apr` clap tree
/// overflows a default test-thread stack (see the note at
/// `commands/tokenize.rs` on FALSIFY-APR-TOK-PAR-004), so [`emit`] always runs
/// it on a thread of its own.
const WALK_STACK: usize = 64 * 1024 * 1024;

/// The whole surface: one entry per subcommand path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Surface {
    /// Always [`SCHEMA`].
    pub schema: &'static str,
    /// The binary this surface was read from.
    pub binary: BinaryId,
    /// Every role an argument can carry, in the order the schema lists them.
    pub roles: Vec<&'static str>,
    /// Arguments the root declares with `global = true`; clap propagates each
    /// of them to every command path, and they are not repeated per command.
    pub global_args: Vec<ArgEntry>,
    /// Every subcommand path, nested and group nodes included, sorted by path.
    pub commands: Vec<CommandEntry>,
}

/// Name, version and source revision of the binary under test.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BinaryId {
    pub name: String,
    pub version: String,
    pub git_sha: String,
}

/// One subcommand path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CommandEntry {
    /// The subcommand names from the root, e.g. `["serve", "run"]`.
    pub path: Vec<String>,
    /// `path` joined with single spaces: the key #3739's CRUX mapping uses.
    pub key: String,
    /// Other names clap accepts for this subcommand (visible and hidden).
    pub aliases: Vec<String>,
    /// Absent from `--help`.
    pub hidden: bool,
    /// Declared by another crate's `Subcommand` type rather than by apr-cli.
    pub foreign: bool,
    /// True iff the command has no subcommands.
    pub leaf: bool,
    /// The command generates text from prompts (v1.1): it has a `PromptText`
    /// argument, or it carries the `ServesGeneration` marker because its
    /// prompts arrive another way (`serve run` over HTTP, `chat` on stdin,
    /// `mcp` as JSON-RPC). Read from how the command is built, never its name.
    pub generates: bool,
    /// Names of the direct subcommands, in declaration order.
    pub subcommands: Vec<String>,
    /// Arguments declared on this command (not the propagated globals).
    pub args: Vec<ArgEntry>,
}

/// One argument.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ArgEntry {
    pub id: String,
    /// Long name, without dashes.
    pub long: Option<String>,
    pub short: Option<char>,
    /// Every other long name clap accepts (visible and hidden).
    pub long_aliases: Vec<String>,
    /// Every other short name clap accepts (visible and hidden).
    pub short_aliases: Vec<char>,
    pub positional: bool,
    /// 1-based position of a positional argument.
    pub index: Option<usize>,
    pub required: bool,
    pub takes_value: bool,
    pub num_args: NumArgs,
    /// `flag`, `count`, `enum`, `path`, `text`, `int`, `float` or `other`.
    pub value_type: &'static str,
    /// The finite set of accepted values (hidden values excluded), else empty.
    pub values: Vec<String>,
    pub default: Vec<String>,
    /// One of [`Surface::roles`].
    pub role: &'static str,
    /// The role type that built this argument (`ModelPath`, `BackendArg`, …).
    pub marker: Option<&'static str>,
    /// For role `sampling` (v1.1): `seed`, `temperature`, `top_k`, `top_p`,
    /// `min_p`, `repeat_penalty` or `repeat_last_n`, read from the sampling
    /// group the argument joined, never from its name. `null` otherwise.
    pub sampling_kind: Option<&'static str>,
    pub hidden: bool,
    /// Ids of the arguments this one conflicts with.
    pub conflicts_with: Vec<String>,
}

/// How many values one occurrence takes. `max: None` is unbounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct NumArgs {
    pub min: usize,
    pub max: Option<usize>,
}

impl Surface {
    /// The versioned JSON document `apr surface --json` prints.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("surface is plain data and always serializes")
    }

    /// The entry for a subcommand path, e.g. `["serve", "run"]`.
    #[must_use]
    pub fn command(&self, path: &[&str]) -> Option<&CommandEntry> {
        self.commands
            .iter()
            .find(|c| c.path.iter().map(String::as_str).eq(path.iter().copied()))
    }
}

impl CommandEntry {
    /// The argument with this id.
    #[must_use]
    pub fn arg(&self, id: &str) -> Option<&ArgEntry> {
        self.args.iter().find(|a| a.id == id)
    }
}

/// The surface of the `apr` binary this library is linked into.
///
/// Runs the walk on its own thread with an explicit stack (see [`WALK_STACK`]),
/// so it is safe to call from a test thread or from `main`.
#[must_use]
pub fn emit() -> Surface {
    std::thread::Builder::new()
        .name("apr-surface".to_string())
        .stack_size(WALK_STACK)
        .spawn(|| emit_from(&Cli::command()))
        .expect("spawn the surface walk thread")
        .join()
        .expect("the surface walk panicked")
}

/// The surface of an arbitrary clap tree rooted at `root`.
///
/// Membership (which commands and arguments exist) is read from `root` as the
/// code declared it. Finalised attributes (`num_args`, positional indices,
/// conflicts) are read from a BUILT clone. Building also adds clap's generated
/// `help` subcommand and the `--help`/`--version` flags; those are clap's
/// surface, not apr's, so they come from nowhere in the unbuilt tree and are
/// not emitted.
///
/// Call this on a thread with a large stack when `root` is the full `apr` tree.
#[must_use]
pub fn emit_from(root: &clap::Command) -> Surface {
    let mut built = root.clone();
    built.build();
    let foreign = foreign_mounts(root).mounts;

    let global_args = root
        .get_arguments()
        .filter(|a| a.is_global_set())
        .map(|a| arg_entry(a, &built, root))
        .collect();

    let mut commands = Vec::new();
    for sub in root.get_subcommands() {
        let built_sub = child(&built, sub.get_name());
        walk(sub, built_sub, &mut vec![], &foreign, false, &mut commands);
    }
    commands.sort_by(|a, b| a.path.cmp(&b.path));

    Surface {
        schema: SCHEMA,
        binary: BinaryId {
            name: root.get_name().to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            git_sha: env!("APR_GIT_SHA").to_string(),
        },
        roles: ROLES.iter().map(|r| r.as_str()).collect(),
        global_args,
        commands,
    }
}

/// Every role, in schema order.
const ROLES: [Role; 8] = [
    Role::Model,
    Role::Prompt,
    Role::InputFile,
    Role::Backend,
    Role::Sampling,
    Role::Mode,
    Role::Other,
    Role::Unknown,
];

fn child<'a>(built: &'a clap::Command, name: &str) -> &'a clap::Command {
    built
        .get_subcommands()
        .find(|s| s.get_name() == name)
        .expect("a built tree has every subcommand its unbuilt source declares")
}

fn walk(
    cmd: &clap::Command,
    built: &clap::Command,
    parent: &mut Vec<String>,
    foreign: &[Vec<String>],
    inherited_foreign: bool,
    out: &mut Vec<CommandEntry>,
) {
    let mut path = parent.clone();
    path.push(cmd.get_name().to_string());
    let is_foreign = inherited_foreign || foreign.iter().any(|m| *m == path);

    let subcommands: Vec<String> = cmd
        .get_subcommands()
        .map(|s| s.get_name().to_string())
        .collect();
    let args: Vec<ArgEntry> = cmd
        .get_arguments()
        .filter(|a| !a.is_global_set())
        .map(|a| arg_entry(a, built, cmd))
        .collect();
    out.push(CommandEntry {
        key: path.join(" "),
        path: path.clone(),
        aliases: cmd.get_all_aliases().map(str::to_string).collect(),
        hidden: cmd.is_hide_set(),
        foreign: is_foreign,
        leaf: subcommands.is_empty(),
        generates: generates(cmd, &args),
        subcommands,
        args,
    });

    for sub in cmd.get_subcommands() {
        walk(
            sub,
            child(built, sub.get_name()),
            &mut path,
            foreign,
            is_foreign,
            out,
        );
    }
}

fn arg_entry(arg: &clap::Arg, built_cmd: &clap::Command, cmd: &clap::Command) -> ArgEntry {
    let built_arg = built_cmd
        .get_arguments()
        .find(|b| b.get_id() == arg.get_id())
        .expect("a built command keeps every argument its source declares");
    let class = classify(arg, cmd, built_cmd);
    let takes_value = built_arg.get_action().takes_values();
    let num_args = built_arg
        .get_num_args()
        .map(|r| NumArgs {
            min: r.min_values(),
            max: (r.max_values() != usize::MAX).then_some(r.max_values()),
        })
        .unwrap_or(if takes_value {
            NumArgs {
                min: 1,
                max: Some(1),
            }
        } else {
            NumArgs {
                min: 0,
                max: Some(0),
            }
        });
    ArgEntry {
        id: arg.get_id().to_string(),
        long: arg.get_long().map(str::to_string),
        short: arg.get_short(),
        long_aliases: arg
            .get_all_aliases()
            .unwrap_or_default()
            .into_iter()
            .map(str::to_string)
            .collect(),
        short_aliases: arg.get_all_short_aliases().unwrap_or_default(),
        positional: arg.is_positional(),
        index: built_arg.get_index(),
        required: arg.is_required_set(),
        takes_value,
        num_args,
        value_type: class.value_type,
        values: finite_values(arg),
        default: arg
            .get_default_values()
            .iter()
            .map(|v| v.to_string_lossy().into_owned())
            .collect(),
        role: class.role.as_str(),
        marker: class.marker,
        sampling_kind: cli_roles::SamplingArg::kind_of(built_cmd, built_arg)
            .map(cli_roles::SamplingKind::as_str),
        hidden: arg.is_hide_set(),
        conflicts_with: built_cmd
            .get_arg_conflicts_with(built_arg)
            .into_iter()
            .map(|c| c.get_id().to_string())
            .collect(),
    }
}

/// True iff the command generates from prompts: a `PromptText` argument, or the
/// command-level `ServesGeneration` marker. Both are read from construction.
fn generates(cmd: &clap::Command, args: &[ArgEntry]) -> bool {
    cli_roles::ServesGeneration::is_on(cmd) || args.iter().any(|a| a.role == Role::Prompt.as_str())
}

/// An argument's role, the type that declared it, and the shape of its value.
struct Class {
    role: Role,
    marker: Option<&'static str>,
    value_type: &'static str,
}

/// Decide an argument's role from how it is built. Nothing here reads a name.
fn classify(arg: &clap::Arg, cmd: &clap::Command, built: &clap::Command) -> Class {
    let ty = arg.get_value_parser().type_id();
    let finite = !finite_values(arg).is_empty();
    let value_type = match arg.get_action() {
        ArgAction::SetTrue | ArgAction::SetFalse => "flag",
        ArgAction::Count => "count",
        _ if finite => "enum",
        _ => scalar_type(arg.get_value_parser()),
    };

    if in_backend_group(arg, cmd) {
        return Class {
            role: Role::Backend,
            marker: Some("BackendArg"),
            value_type,
        };
    }
    // v1.1: membership in the SamplingArg group, which is visible only on the
    // BUILT command (clap merges `#[arg(group = …)]` there).
    if cli_roles::SamplingArg::kind_of(built, arg).is_some() {
        return Class {
            role: Role::Sampling,
            marker: Some("SamplingArg"),
            value_type,
        };
    }
    if let Some(m) = cli_roles::MARKERS.iter().find(|m| ty == m.type_id()) {
        let value_type = match m.carrier {
            Carrier::Path => "path",
            Carrier::Text => "text",
        };
        return Class {
            role: m.role,
            marker: Some(m.name),
            value_type,
        };
    }
    let role = match value_type {
        "flag" | "count" | "enum" => Role::Mode,
        "path" | "text" => Role::Unknown,
        _ => Role::Other,
    };
    Class {
        role,
        marker: None,
        value_type,
    }
}

/// `path`/`text` for the two raw free-form types, `int`/`float` for numbers,
/// `other` for anything else a value parser can yield.
fn scalar_type(parser: &clap::builder::ValueParser) -> &'static str {
    let ty = parser.type_id();
    let is = |t: TypeId| ty == t;
    if is(TypeId::of::<PathBuf>()) {
        "path"
    } else if is(TypeId::of::<String>()) {
        "text"
    } else if [
        TypeId::of::<u8>(),
        TypeId::of::<u16>(),
        TypeId::of::<u32>(),
        TypeId::of::<u64>(),
        TypeId::of::<u128>(),
        TypeId::of::<usize>(),
        TypeId::of::<i8>(),
        TypeId::of::<i16>(),
        TypeId::of::<i32>(),
        TypeId::of::<i64>(),
        TypeId::of::<i128>(),
        TypeId::of::<isize>(),
    ]
    .into_iter()
    .any(is)
    {
        "int"
    } else if is(TypeId::of::<f32>()) || is(TypeId::of::<f64>()) {
        "float"
    } else {
        "other"
    }
}

/// Values the argument's parser declares as its whole domain. clap's bool
/// parser lists `true`/`false`, which is a finite domain and so a mode.
fn finite_values(arg: &clap::Arg) -> Vec<String> {
    arg.get_possible_values()
        .iter()
        .filter(|v| !v.is_hide_set())
        .map(|v| v.get_name().to_string())
        .collect()
}

/// True iff `arg` is declared through the shared [`BackendArg`] flatten. The
/// group id comes from the type, through `clap::Args`, never from a literal.
fn in_backend_group(arg: &clap::Arg, cmd: &clap::Command) -> bool {
    let Some(group) = <BackendArg as clap::Args>::group_id() else {
        return false;
    };
    cmd.get_groups()
        .any(|g| g.get_id() == &group && g.get_args().any(|a| a == arg.get_id()))
}

/// The subtrees of `root` that another crate's `Subcommand` type declares.
///
/// apr embeds these CLIs whole (`apr pv …` is `aprender-contracts-cli`'s own
/// enum). Their free-form arguments cannot all carry apr's role types yet, so
/// they are emitted as `foreign: true` and a remaining untyped argument there
/// is role `unknown`, which S2 charges a runs-or-refuses cell and ratchets
/// shrink-only (cop ruling on #3745).
///
/// A mount is found by TYPE: each listed type renders its own subcommand set,
/// and a subtree of `root` whose children are exactly that set (order ignored)
/// is its mount.
/// The list fails closed both ways. A foreign CLI missing from it has its
/// untyped arguments counted as apr-cli's, which the marker guard refuses. A
/// listed type that is mounted zero times or more than once is reported in
/// [`ForeignScan::problems`], which the marker guard requires to be empty for
/// the real `apr` tree, so a stale entry cannot sit in the list unnoticed.
fn foreign_mounts(root: &clap::Command) -> ForeignScan {
    let probes: [(&str, clap::Command); 6] = [
        (
            "aprender_contracts_cli::cli::Commands",
            probe::<aprender_contracts_cli::cli::Commands>(),
        ),
        (
            "alimentar::cli::Commands",
            probe::<alimentar::cli::Commands>(),
        ),
        ("simular::cli::Commands", probe::<simular::cli::Commands>()),
        (
            "aprender_rag_cli::Commands",
            probe::<aprender_rag_cli::Commands>(),
        ),
        (
            "aprender_zram_cli::Commands",
            probe::<aprender_zram_cli::Commands>(),
        ),
        ("cgp::cli::Commands", probe::<cgp::cli::Commands>()),
    ];
    let mut scan = ForeignScan::default();
    for (ty, p) in &probes {
        // A set, not a sequence: declaration order is not identity, and clap's
        // own `mut_subcommand` moves the child it mutates to the end.
        let mut want: Vec<&str> = p.get_subcommands().map(clap::Command::get_name).collect();
        want.sort_unstable();
        let mut hits = Vec::new();
        find_mounts(root, &mut Vec::new(), &want, &mut hits);
        if hits.len() != 1 {
            scan.problems.push(format!(
                "foreign CLI `{ty}` must be mounted exactly once in the tree, found {} mount(s): {hits:?}",
                hits.len()
            ));
        }
        scan.mounts.extend(hits);
    }
    scan
}

/// Where the foreign CLIs are mounted, and what is wrong with the list.
#[derive(Debug, Default)]
pub(crate) struct ForeignScan {
    /// Path of each mount point.
    pub(crate) mounts: Vec<Vec<String>>,
    /// A listed type mounted zero times, or more than once.
    pub(crate) problems: Vec<String>,
}

/// The foreign-list problems for the real `apr` tree. The marker guard requires
/// this to be empty.
#[must_use]
pub fn foreign_list_problems() -> Vec<String> {
    std::thread::Builder::new()
        .stack_size(WALK_STACK)
        .spawn(|| foreign_mounts(&Cli::command()).problems)
        .expect("spawn")
        .join()
        .expect("join")
}

fn probe<T: clap::Subcommand>() -> clap::Command {
    T::augment_subcommands(clap::Command::new("probe"))
}

fn find_mounts(
    cmd: &clap::Command,
    path: &mut Vec<String>,
    want: &[&str],
    hits: &mut Vec<Vec<String>>,
) {
    for sub in cmd.get_subcommands() {
        path.push(sub.get_name().to_string());
        let mut names: Vec<&str> = sub.get_subcommands().map(clap::Command::get_name).collect();
        names.sort_unstable();
        if !names.is_empty() && names == want {
            hits.push(path.clone());
        }
        find_mounts(sub, path, want, hits);
        path.pop();
    }
}

/// Every argument outside a foreign subtree whose role is `unknown`, as
/// `"<key> <arg id>"`. This is the marker guard's reading, and it must be empty:
/// a free-form argument that no role type declares is how a model or prompt
/// argument hides from the cells derived from this surface.
#[must_use]
pub fn unknown_outside_foreign(surface: &Surface) -> Vec<String> {
    let mut out: Vec<String> = surface
        .global_args
        .iter()
        .filter(|a| a.role == Role::Unknown.as_str())
        .map(|a| format!("<global> {}", a.id))
        .collect();
    for c in surface.commands.iter().filter(|c| !c.foreign) {
        for a in c.args.iter().filter(|a| a.role == Role::Unknown.as_str()) {
            out.push(format!("{} {}", c.key, a.id));
        }
    }
    out
}

/// `apr surface`: print [`emit`]'s JSON and nothing else on stdout.
///
/// Written with `std::io::Write`, not the crate's shadowed `println!`, so
/// `--quiet` cannot suppress the only thing the verb exists to produce.
pub(crate) fn run() -> Result<(), crate::CliError> {
    write_surface(&mut std::io::stdout().lock())
        .map_err(|e| crate::CliError::ValidationFailed(format!("apr surface: writing stdout: {e}")))
}

/// The bytes `apr surface` writes: [`emit`]'s JSON and one newline.
fn write_surface(out: &mut impl std::io::Write) -> std::io::Result<()> {
    out.write_all(emit().to_json().as_bytes())?;
    out.write_all(b"\n")?;
    out.flush()
}

#[cfg(test)]
#[path = "surface_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "surface_guard_tests.rs"]
mod guard;
