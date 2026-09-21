//! #3745 S1.3: the marker guard.
//!
//! Issue #3745 mutant 3, verbatim: "a command that loads a model with a plain
//! `PathBuf` arg instead of `ModelArg` → the marker guard is RED."
//!
//! The guard does not try to detect model loading; apr-cli has no loader funnel
//! to detect it at (about 13 loader APIs across realizar and aprender-core), and
//! a list of loaders would rot like the verb lists #3745 removes. Instead it
//! makes an undeclared role inexpressible: outside a foreign subtree, every
//! free-form argument (raw `PathBuf`, or raw `String` without a finite value
//! set) is RED until it is built through a role type. A model argument can
//! therefore only be written down as a model argument, or as a typed lie that
//! is visible in the diff.

use super::*;
use clap::Arg;

fn big_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(WALK_STACK)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("join")
}

const REMEDY: &str = "declare each through a role type from batuta_common::cli_roles \
     (ModelPath/ModelRef for a model, PromptText for a prompt, InputFile for input data, \
     OutputPath/DirPath/ConfigPath/FreeText otherwise) — a raw PathBuf/String hides the \
     argument's role from the cells derived from `apr surface` (#3745)";

/// The guard itself, on the tree this binary ships.
#[test]
fn no_free_form_arg_is_undeclared_outside_foreign_subtrees() {
    let unknown = big_stack(|| unknown_outside_foreign(&emit()));
    assert!(
        unknown.is_empty(),
        "{} free-form argument(s) with no declared role:\n  {}\n{REMEDY}",
        unknown.len(),
        unknown.join("\n  ")
    );
}

/// The foreign-CLI list fails closed: every listed type is mounted exactly once.
#[test]
fn the_foreign_list_matches_the_tree() {
    let problems = foreign_list_problems();
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// The verbs whose leaks #3745 names, and the two foreign model arguments the
/// cop ruled into S1, carry the roles their cells are derived from.
#[test]
fn the_leak_verbs_declare_model_prompt_and_input_roles() {
    let s = big_stack(emit);
    let role = |path: &[&str], id: &str| {
        s.command(path)
            .and_then(|c| c.arg(id))
            .map(|a| a.role)
            .unwrap_or_else(|| panic!("{path:?} {id} is on the surface"))
    };
    // #3743: `run --prompt P --chat` vs `-i file --chat`.
    assert_eq!(role(&["run"], "source"), "model");
    assert_eq!(role(&["run"], "prompt"), "prompt");
    assert_eq!(role(&["run"], "positional_prompt"), "prompt");
    assert_eq!(role(&["run"], "input"), "input-file");
    assert_eq!(role(&["chat"], "file"), "model");
    // #3571: `apr serve` on Qwen3.5.
    assert_eq!(role(&["serve", "run"], "file"), "model");
    // #3719: `apr code`'s backend is `apr serve`.
    assert_eq!(role(&["code"], "model"), "model");
    assert_eq!(role(&["code"], "prompt"), "prompt");
    // Cop ruling on #3745: foreign, but typed in S1 itself.
    assert_eq!(role(&["pv", "verify-structure"], "model"), "model");
    assert_eq!(role(&["rag", "transcribe"], "model"), "model");
}

// ── mutants, planted in the real tree on every run ──────────────────────────

fn unknown_after(
    mutate: impl FnOnce(clap::Command) -> clap::Command + Send + 'static,
) -> Vec<String> {
    big_stack(move || unknown_outside_foreign(&emit_from(&mutate(Cli::command()))))
}

/// Mutant 3a: `inspect`'s model argument rebuilt as a plain `PathBuf`.
#[test]
fn mutant_an_existing_model_arg_reverted_to_plain_pathbuf_is_red() {
    let unknown = unknown_after(|root| {
        root.mut_subcommand("inspect", |c| {
            c.mut_arg("file", |a| a.value_parser(clap::value_parser!(PathBuf)))
        })
    });
    assert_eq!(
        unknown,
        ["inspect file"],
        "the guard must name exactly the reverted argument"
    );
}

/// Mutant 3b: a new command that takes its model through a plain `PathBuf`.
#[test]
fn mutant_a_new_command_with_a_plain_pathbuf_model_is_red() {
    let unknown = unknown_after(|root| {
        root.subcommand(
            clap::Command::new("zz-new-verb").arg(
                Arg::new("model")
                    .long("model")
                    .value_parser(clap::value_parser!(PathBuf)),
            ),
        )
    });
    assert_eq!(unknown, ["zz-new-verb model"]);
}

/// Mutant 3c: a prompt taken as a plain `String` (the #3743 class).
#[test]
fn mutant_a_prompt_as_plain_string_is_red() {
    let unknown = unknown_after(|root| {
        root.mut_subcommand("run", |c| {
            c.mut_arg("prompt", |a| a.value_parser(clap::value_parser!(String)))
        })
    });
    assert_eq!(unknown, ["run prompt"]);
}

/// A raw argument in a nested apr-cli subcommand is caught as well.
#[test]
fn mutant_a_nested_plain_pathbuf_is_red() {
    let unknown = unknown_after(|root| {
        root.mut_subcommand("serve", |serve| {
            serve.mut_subcommand("run", |c| {
                c.mut_arg("file", |a| a.value_parser(clap::value_parser!(PathBuf)))
            })
        })
    });
    assert_eq!(unknown, ["serve run file"]);
}

/// A raw global argument is caught: globals reach every command path.
#[test]
fn mutant_a_plain_global_is_red() {
    let unknown = unknown_after(|root| {
        root.arg(
            Arg::new("zz-config")
                .long("zz-config")
                .global(true)
                .value_parser(clap::value_parser!(PathBuf)),
        )
    });
    assert_eq!(unknown, ["<global> zz-config"]);
}

/// Inside a foreign subtree a raw argument is emitted as `unknown` (S2 owes
/// its verb a runs-or-refuses cell and ratchets the count), and it is not the
/// marker guard's to refuse. (Reverting THIS argument is still caught, by
/// `the_leak_verbs_declare_model_prompt_and_input_roles`.)
#[test]
fn a_foreign_raw_arg_is_unknown_not_red() {
    let s = big_stack(|| {
        emit_from(&Cli::command().mut_subcommand("pv", |pv| {
            pv.mut_subcommand("verify-structure", |c| {
                c.mut_arg("model", |a| a.value_parser(clap::value_parser!(PathBuf)))
            })
        }))
    });
    let arg = s
        .command(&["pv", "verify-structure"])
        .and_then(|c| c.arg("model"))
        .expect("pv verify-structure --model");
    assert_eq!(arg.role, "unknown");
    assert!(
        unknown_outside_foreign(&s).is_empty(),
        "foreign subtrees are not the marker guard's"
    );
}

/// v1.1: the generators S2 derives prompt-shape cells for. `serve run`, `chat`
/// and `mcp` take their prompts over HTTP, stdin and JSON-RPC, so only the
/// `ServesGeneration` marker makes them generators; `run` and `code` are by
/// their `PromptText` args. Model tools that never generate are not.
#[test]
fn the_generators_are_declared() {
    let s = big_stack(emit);
    let generates = |path: &[&str]| {
        s.command(path)
            .map(|c| c.generates)
            .unwrap_or_else(|| panic!("{path:?} is on the surface"))
    };
    for path in [
        &["serve", "run"][..],
        &["chat"],
        &["mcp"],
        &["run"],
        &["code"],
    ] {
        assert!(generates(path), "{path:?} must be a generator");
    }
    for path in [
        &["inspect"][..],
        &["tensors"],
        &["serve", "plan"],
        &["surface"],
        // Encoders and scorers take EncodeText, not PromptText (aprender-97):
        // thinking × context-rung cells mean nothing for them.
        &["embed"],
        &["rerank"],
        &["eval"],
    ] {
        assert!(!generates(path), "{path:?} does not generate");
    }
}

/// Mutant: `serve run` with its ServesGeneration marker dropped. Its real
/// argument set, rebuilt without the marker, reads `generates: false`. So the
/// marker, not any argument, is what makes it a generator (#3571's verb).
#[test]
fn mutant_serve_run_without_its_marker_does_not_generate() {
    let (real, stripped) = big_stack(|| {
        let root = Cli::command();
        let run = root
            .find_subcommand("serve")
            .and_then(|s| s.find_subcommand("run"))
            .expect("serve run")
            .clone();
        let mut bare = clap::Command::new("run");
        for a in run.get_arguments() {
            bare = bare.arg(a.clone());
        }
        let real = emit_from(&clap::Command::new("apr").subcommand(run));
        let stripped = emit_from(&clap::Command::new("apr").subcommand(bare));
        (real, stripped)
    });
    assert!(real.command(&["run"]).expect("run").generates);
    assert!(
        !stripped.command(&["run"]).expect("run").generates,
        "without the marker serve run must read generates=false — the marker is load-bearing"
    );
}
