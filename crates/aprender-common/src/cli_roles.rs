//! Typed CLI argument roles (#3745 S1, PMAT-3749).
//!
//! A release cell can only be derived for an argument whose ROLE the gate can
//! see. Before these types, the role of an `apr` argument was knowable only
//! from its name: `apr run SOURCE`, `--prompt` and `-i FILE` were a `String`, a
//! `String` and a `PathBuf`, the same types as `--name`, `--tag` and
//! `--output`. Every release list was therefore typed by hand, and a verb, flag
//! or input shape missing from that list was invisible to the gate. That is how
//! `apr run --prompt P --chat` shipped double-templated (#3743) while
//! `-i file --chat` was covered.
//!
//! Each type here is a transparent newtype over `PathBuf` or `String` whose clap
//! value parser is the SAME parser clap uses for the raw type
//! (`PathBufValueParser` / `StringValueParser`), mapped into the newtype. So a
//! migrated argument accepts exactly the values it accepted before. The only
//! thing that changes is the parser's output TYPE, and that is what `apr surface`
//! reads, through `ValueParser::type_id()`, to decide the role. The role is
//! declared by how the argument is BUILT; its name never enters into it.
//!
//! The types live here rather than in `apr-cli` because a marker compared by
//! `TypeId` needs ONE defining crate, and some of `apr`'s surface is declared by
//! crates that do not depend on `apr-cli` (`apr pv …` from
//! `aprender-contracts-cli`, `apr rag …` from `aprender-rag-cli`).

use std::any::TypeId;
use std::ffi::OsStr;
use std::fmt;
use std::ops::Deref;
use std::path::{Path, PathBuf};

use clap::builder::{MapValueParser, PathBufValueParser, StringValueParser, TypedValueParser};

/// The role an argument plays in a release cell (#3745).
///
/// `Mode` and `Backend` are not listed in [`MARKERS`]. Mode is structural (a
/// flag action, or a finite value set), and backend is membership in the shared
/// `BackendArg` group; `apr surface` decides both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Role {
    /// A model: a local file, or a reference `apr` resolves to one.
    Model,
    /// Text the model is prompted with.
    Prompt,
    /// A file the command consumes as input data (audio, text, a dataset).
    InputFile,
    /// The compute backend override.
    Backend,
    /// A flag, or a value drawn from a finite set.
    Mode,
    /// Declared, and none of the above: outputs, directories, configs, names.
    Other,
    /// A free-form value no role type declares. Only a foreign subtree may
    /// carry one; the marker guard is RED on it anywhere else.
    Unknown,
}

impl Role {
    /// The spelling used in the `apr-cli-surface` JSON.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Prompt => "prompt",
            Self::InputFile => "input-file",
            Self::Backend => "backend",
            Self::Mode => "mode",
            Self::Other => "other",
            Self::Unknown => "unknown",
        }
    }
}

/// What a marker's underlying value is: a filesystem path, or free text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Carrier {
    /// Backed by `PathBuf`.
    Path,
    /// Backed by `String`.
    Text,
}

/// One role type: the `TypeId` its value parser yields, and what it declares.
#[derive(Debug, Clone, Copy)]
pub struct Marker {
    /// The type's name as written in code, e.g. `ModelPath`.
    pub name: &'static str,
    /// The role this type declares.
    pub role: Role,
    /// Path- or text-backed.
    pub carrier: Carrier,
    type_id: fn() -> TypeId,
}

impl Marker {
    /// The `TypeId` of the marker type, which is the `TypeId` its clap value
    /// parser reports.
    #[must_use]
    pub fn type_id(&self) -> TypeId {
        (self.type_id)()
    }
}

macro_rules! path_role {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(PathBuf);

        // `Debug` is the inner value's, so `{:?}` of a migrated field prints
        // `"m.gguf"` exactly as the raw `PathBuf` did, never `ModelPath("m.gguf")`.
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Debug::fmt(&self.0, f)
            }
        }

        impl $name {
            /// Wrap a path.
            #[must_use]
            pub fn new(path: impl Into<PathBuf>) -> Self {
                Self(path.into())
            }
            /// The path, borrowed.
            #[must_use]
            pub fn as_path(&self) -> &Path {
                &self.0
            }
            /// The path, owned.
            #[must_use]
            pub fn into_path_buf(self) -> PathBuf {
                self.0
            }
        }

        // `Path`, not `PathBuf`: the same target `PathBuf` derefs to, so
        // `Option<ModelPath>::as_deref()` is `Option<&Path>` exactly as
        // `Option<PathBuf>::as_deref()` was, and a migrated field reads the same.
        impl Deref for $name {
            type Target = Path;
            fn deref(&self) -> &Path {
                &self.0
            }
        }
        impl AsRef<Path> for $name {
            fn as_ref(&self) -> &Path {
                &self.0
            }
        }
        impl AsRef<OsStr> for $name {
            fn as_ref(&self) -> &OsStr {
                self.0.as_os_str()
            }
        }
        impl From<PathBuf> for $name {
            fn from(p: PathBuf) -> Self {
                Self(p)
            }
        }
        impl From<&Path> for $name {
            fn from(p: &Path) -> Self {
                Self(p.to_path_buf())
            }
        }
        impl From<&str> for $name {
            fn from(p: &str) -> Self {
                Self(PathBuf::from(p))
            }
        }
        impl From<String> for $name {
            fn from(p: String) -> Self {
                Self(PathBuf::from(p))
            }
        }
        impl From<$name> for PathBuf {
            fn from(p: $name) -> PathBuf {
                p.0
            }
        }
        impl PartialEq<PathBuf> for $name {
            fn eq(&self, other: &PathBuf) -> bool {
                &self.0 == other
            }
        }
        impl PartialEq<Path> for $name {
            fn eq(&self, other: &Path) -> bool {
                self.0 == other
            }
        }
        impl clap::builder::ValueParserFactory for $name {
            type Parser = MapValueParser<PathBufValueParser, fn(PathBuf) -> $name>;
            fn value_parser() -> Self::Parser {
                PathBufValueParser::new().map($name as fn(PathBuf) -> $name)
            }
        }
    };
}

macro_rules! text_role {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(String);

        // `Debug` is the inner value's: `{:?}` prints `"text"`, as `String` did.
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Debug::fmt(&self.0, f)
            }
        }

        impl $name {
            /// Wrap a string.
            #[must_use]
            pub fn new(s: impl Into<String>) -> Self {
                Self(s.into())
            }
            /// The text, borrowed.
            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }
            /// The text as a `&String`, for callees that take one.
            #[must_use]
            pub fn as_string(&self) -> &String {
                &self.0
            }
            /// The text, owned.
            #[must_use]
            pub fn into_string(self) -> String {
                self.0
            }
        }

        // `str`, not `String`: the same target `String` derefs to, so
        // `Option<PromptText>::as_deref()` is `Option<&str>` exactly as
        // `Option<String>::as_deref()` was.
        impl Deref for $name {
            type Target = str;
            fn deref(&self) -> &str {
                &self.0
            }
        }
        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }
        impl AsRef<Path> for $name {
            fn as_ref(&self) -> &Path {
                Path::new(&self.0)
            }
        }
        impl AsRef<OsStr> for $name {
            fn as_ref(&self) -> &OsStr {
                OsStr::new(&self.0)
            }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
        impl From<String> for $name {
            fn from(s: String) -> Self {
                Self(s)
            }
        }
        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(s.to_string())
            }
        }
        impl From<$name> for String {
            fn from(s: $name) -> String {
                s.0
            }
        }
        impl PartialEq<str> for $name {
            fn eq(&self, other: &str) -> bool {
                self.0 == other
            }
        }
        impl PartialEq<&str> for $name {
            fn eq(&self, other: &&str) -> bool {
                self.0 == *other
            }
        }
        impl PartialEq<String> for $name {
            fn eq(&self, other: &String) -> bool {
                &self.0 == other
            }
        }
        impl clap::builder::ValueParserFactory for $name {
            type Parser = MapValueParser<StringValueParser, fn(String) -> $name>;
            fn value_parser() -> Self::Parser {
                StringValueParser::new().map($name as fn(String) -> $name)
            }
        }
    };
}

path_role!(
    /// A local model file (`.apr`, `.gguf`, `.safetensors`, …).
    ModelPath
);
path_role!(
    /// A file the command reads as input data, not as a model.
    InputFile
);
path_role!(
    /// A file the command writes.
    OutputPath
);
path_role!(
    /// A directory: a workspace, a cache, a corpus root, an output tree.
    DirPath
);
path_role!(
    /// A configuration, manifest, contract, schema or receipt the command
    /// reads to decide what to do, as distinct from the data it operates on.
    ConfigPath
);
text_role!(
    /// A model reference: a local path, `hf://org/repo`, a URL, or a cache
    /// name, which `apr` resolves to a model file.
    ModelRef
);
text_role!(
    /// Text the model is prompted with.
    PromptText
);
text_role!(
    /// Any other free text: a name, id, tag, URL, pattern or list.
    FreeText
);

/// Command-level marker: this command GENERATES text from prompts that do not
/// arrive through a `PromptText` argument (#3745 S1, schema v1.1).
///
/// `apr serve run` takes its prompts over HTTP, `apr chat` reads them from
/// stdin, and `apr mcp` receives them as JSON-RPC tool calls. None of them has a
/// prompt ARGUMENT, so a rule that reads "generates" off the prompt role alone
/// would derive no prompt-shape cells for the very verb whose 0.69.0 leak
/// (#3571) started this. The marker is attached to the subcommand itself, with
/// no argument and no field:
///
/// ```ignore
/// #[command(group(batuta_common::cli_roles::ServesGeneration::group()))]
/// Run { /* … */ },
/// ```
///
/// `apr surface` reports `generates: true` for a command that has a
/// `PromptText` argument OR carries this group, and it matches the group by
/// [`ServesGeneration::ID`], never by the command's name.
#[derive(Debug, Clone, Copy)]
pub struct ServesGeneration;

impl ServesGeneration {
    /// The group id the marker is recognised by.
    pub const ID: &'static str = "batuta_common::cli_roles::ServesGeneration";

    /// The argument-less group that marks a command as a generator. It names
    /// no argument, so it changes neither parsing nor `--help`.
    #[must_use]
    pub fn group() -> clap::ArgGroup {
        clap::ArgGroup::new(Self::ID).multiple(true)
    }

    /// True iff `cmd` carries the marker.
    #[must_use]
    pub fn is_on(cmd: &clap::Command) -> bool {
        cmd.get_groups().any(|g| g.get_id() == Self::ID)
    }
}

/// Owned `String`s from a slice of text role values, for callees that take
/// `&[String]`.
#[must_use]
pub fn strings<T: AsRef<str>>(values: &[T]) -> Vec<String> {
    values.iter().map(|v| v.as_ref().to_string()).collect()
}

/// Owned `PathBuf`s from a slice of path role values, for callees that take
/// `&[PathBuf]`.
#[must_use]
pub fn path_bufs<T: AsRef<Path>>(values: &[T]) -> Vec<PathBuf> {
    values.iter().map(|v| v.as_ref().to_path_buf()).collect()
}

/// Every role type, in one table. `apr surface` reads roles through it, and the
/// marker guard refuses a free-form argument whose parser yields none of these.
pub const MARKERS: &[Marker] = &[
    Marker {
        name: "ModelPath",
        role: Role::Model,
        carrier: Carrier::Path,
        type_id: TypeId::of::<ModelPath>,
    },
    Marker {
        name: "ModelRef",
        role: Role::Model,
        carrier: Carrier::Text,
        type_id: TypeId::of::<ModelRef>,
    },
    Marker {
        name: "PromptText",
        role: Role::Prompt,
        carrier: Carrier::Text,
        type_id: TypeId::of::<PromptText>,
    },
    Marker {
        name: "InputFile",
        role: Role::InputFile,
        carrier: Carrier::Path,
        type_id: TypeId::of::<InputFile>,
    },
    Marker {
        name: "OutputPath",
        role: Role::Other,
        carrier: Carrier::Path,
        type_id: TypeId::of::<OutputPath>,
    },
    Marker {
        name: "DirPath",
        role: Role::Other,
        carrier: Carrier::Path,
        type_id: TypeId::of::<DirPath>,
    },
    Marker {
        name: "ConfigPath",
        role: Role::Other,
        carrier: Carrier::Path,
        type_id: TypeId::of::<ConfigPath>,
    },
    Marker {
        name: "FreeText",
        role: Role::Other,
        carrier: Carrier::Text,
        type_id: TypeId::of::<FreeText>,
    },
];

/// The marker whose type a value parser yields, if any.
#[must_use]
pub fn marker_for(type_id: TypeId) -> Option<&'static Marker> {
    MARKERS.iter().find(|m| m.type_id() == type_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{Arg, Command};

    fn parser_type<T: clap::builder::ValueParserFactory>() -> TypeId
    where
        T::Parser: TypedValueParser + Send + Sync + 'static,
        <T::Parser as TypedValueParser>::Value: Send + Sync + Clone + 'static,
    {
        let vp: clap::builder::ValueParser = T::value_parser().into();
        let arg = Arg::new("x").value_parser(vp);
        // `AnyValueId` is clap's; it compares equal to a std `TypeId`.
        let id = arg.get_value_parser().type_id();
        MARKERS
            .iter()
            .map(Marker::type_id)
            .chain([TypeId::of::<PathBuf>(), TypeId::of::<String>()])
            .find(|t| id == *t)
            .expect("value parser yields a known type")
    }

    #[test]
    fn every_marker_parser_reports_its_own_type_id() {
        assert_eq!(parser_type::<ModelPath>(), TypeId::of::<ModelPath>());
        assert_eq!(parser_type::<ModelRef>(), TypeId::of::<ModelRef>());
        assert_eq!(parser_type::<PromptText>(), TypeId::of::<PromptText>());
        assert_eq!(parser_type::<InputFile>(), TypeId::of::<InputFile>());
        assert_eq!(parser_type::<OutputPath>(), TypeId::of::<OutputPath>());
        assert_eq!(parser_type::<DirPath>(), TypeId::of::<DirPath>());
        assert_eq!(parser_type::<ConfigPath>(), TypeId::of::<ConfigPath>());
        assert_eq!(parser_type::<FreeText>(), TypeId::of::<FreeText>());
    }

    #[test]
    fn marker_table_is_one_to_one_with_types() {
        let mut ids: Vec<TypeId> = MARKERS.iter().map(Marker::type_id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), MARKERS.len(), "two markers share a TypeId");
        assert!(
            marker_for(TypeId::of::<PathBuf>()).is_none(),
            "raw PathBuf must not be a marker"
        );
        assert!(
            marker_for(TypeId::of::<String>()).is_none(),
            "raw String must not be a marker"
        );
        assert_eq!(
            marker_for(TypeId::of::<ModelPath>()).map(|m| m.role),
            Some(Role::Model)
        );
        assert_eq!(
            marker_for(TypeId::of::<PromptText>()).map(|m| m.role),
            Some(Role::Prompt)
        );
    }

    /// A migrated argument must accept exactly what the raw type accepted: the
    /// marker parser is the raw parser, mapped. An empty path is refused by
    /// both, a normal value is accepted by both.
    #[test]
    fn marker_parsers_accept_what_the_raw_parsers_accept() {
        let raw = Command::new("t").arg(Arg::new("p").value_parser(clap::value_parser!(PathBuf)));
        let typed =
            Command::new("t").arg(Arg::new("p").value_parser(clap::value_parser!(ModelPath)));
        for argv in [vec!["t", "m.gguf"], vec!["t", ""], vec!["t", "-"]] {
            let a = raw.clone().try_get_matches_from(&argv).is_ok();
            let b = typed.clone().try_get_matches_from(&argv).is_ok();
            assert_eq!(a, b, "PathBuf vs ModelPath disagree on {argv:?}");
        }
        let m = typed.try_get_matches_from(["t", "m.gguf"]).expect("parses");
        assert_eq!(
            m.get_one::<ModelPath>("p").map(ModelPath::as_path),
            Some(Path::new("m.gguf"))
        );

        let raw = Command::new("t").arg(Arg::new("s").value_parser(clap::value_parser!(String)));
        let typed =
            Command::new("t").arg(Arg::new("s").value_parser(clap::value_parser!(PromptText)));
        for argv in [vec!["t", "What is 2+2?"], vec!["t", ""]] {
            let a = raw.clone().try_get_matches_from(&argv).is_ok();
            let b = typed.clone().try_get_matches_from(&argv).is_ok();
            assert_eq!(a, b, "String vs PromptText disagree on {argv:?}");
        }
    }

    /// The generation marker names no argument, so it changes neither what
    /// parses nor what `--help` prints, and it is found by id alone.
    #[test]
    fn serves_generation_marks_a_command_without_changing_it() {
        let plain = Command::new("t").arg(Arg::new("port").long("port"));
        let marked = plain.clone().group(ServesGeneration::group());
        assert!(!ServesGeneration::is_on(&plain));
        assert!(ServesGeneration::is_on(&marked));
        for argv in [vec!["t"], vec!["t", "--port", "8080"], vec!["t", "--nope"]] {
            let a = plain.clone().try_get_matches_from(&argv).is_ok();
            let b = marked.clone().try_get_matches_from(&argv).is_ok();
            assert_eq!(a, b, "the marker changed parsing of {argv:?}");
        }
        let help = |c: &Command| c.clone().render_long_help().to_string();
        assert_eq!(help(&plain), help(&marked), "the marker changed --help");
        // A look-alike group spelled with the type's short name is not the marker.
        let lookalike = plain.group(clap::ArgGroup::new("ServesGeneration").multiple(true));
        assert!(!ServesGeneration::is_on(&lookalike));
    }

    /// A migrated field must print the same under `{:?}` as the raw type did:
    /// error messages and tests format arguments that way.
    #[test]
    fn debug_output_is_the_raw_types() {
        assert_eq!(
            format!("{:?}", ModelPath::from("m.gguf")),
            format!("{:?}", PathBuf::from("m.gguf"))
        );
        assert_eq!(
            format!("{:?}", Some(OutputPath::from("o"))),
            format!("{:?}", Some(PathBuf::from("o")))
        );
        assert_eq!(
            format!("{:?}", PromptText::from("hi")),
            format!("{:?}", String::from("hi"))
        );
        assert_eq!(
            format!("{:?}", vec![FreeText::from("a")]),
            format!("{:?}", vec![String::from("a")])
        );
    }

    #[test]
    fn derive_picks_the_marker_parser_for_a_marker_field() {
        #[derive(clap::Parser)]
        struct T {
            model: ModelPath,
            #[arg(long)]
            prompt: Option<PromptText>,
            #[arg(long)]
            inputs: Vec<InputFile>,
        }
        use clap::{CommandFactory, Parser};
        let cmd = T::command();
        let id = |name: &str| {
            cmd.get_arguments()
                .find(|a| a.get_id() == name)
                .map(|a| a.get_value_parser().type_id())
                .expect("arg exists")
        };
        assert!(id("model") == TypeId::of::<ModelPath>());
        assert!(id("prompt") == TypeId::of::<PromptText>());
        assert!(id("inputs") == TypeId::of::<InputFile>());
        let t = T::try_parse_from(["t", "a.apr", "--prompt", "hi", "--inputs", "x.wav"])
            .expect("parses");
        assert_eq!(t.model.as_path(), Path::new("a.apr"));
        assert_eq!(t.prompt.as_deref(), Some("hi"));
        assert_eq!(t.inputs.len(), 1);
    }
}
