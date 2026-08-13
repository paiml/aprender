// FALSIFY-CLI-HELP-TRUTH-001 — every `apr …` invocation quoted in help text
// must be runnable by the binary that prints it.
//
// Dogfooding 0.63.0 found 8 of the 16 `*-lint` commands documenting a
// producer command the shipped binary does not have: `apr attn-viz`,
// `apr dataset audio-inspect --format json`, `apr trace --check-finite`,
// `apr finetune --parallel ddp`, `apr debug embed-viz`,
// `apr explain --format jsonl`, `apr profile --gpu-memory-trace`,
// `apr kernel parity`. Every one exits 2 — `unrecognized subcommand` or
// `unexpected argument` — so the documented end-to-end workflow
// (`apr X … > body.json && apr X-lint --file body.json`) is unreachable for
// a `cargo install aprender` user. `apr attn-viz` even got clap's
// "did you mean 'attn-viz-lint'" pointing back at the consumer (issue
// #2377 finding 3).
//
// Prose that merely *mentions* a command is not the target — the guard only
// reads backtick-quoted invocations, which is how this codebase writes a
// command a user is meant to type.

/// A backtick-quoted `apr …` invocation found in some help string.
#[derive(Debug, Clone)]
struct Quoted {
    /// Where it was found, e.g. `attn-viz-lint --attn-file`.
    site: String,
    /// The full quoted text, e.g. `apr trace --check-finite`.
    text: String,
}

/// Pull every `` `apr …` `` span out of a help string.
fn quoted_apr_invocations(site: &str, help: &str) -> Vec<Quoted> {
    let mut found = Vec::new();
    let mut rest = help;
    while let Some(open) = rest.find('`') {
        let after = &rest[open + 1..];
        let Some(close) = after.find('`') else { break };
        let span = &after[..close];
        if span == "apr" || span.starts_with("apr ") {
            found.push(Quoted {
                site: site.to_string(),
                text: span.to_string(),
            });
        }
        rest = &after[close + 1..];
    }
    found
}

/// Resolve a quoted invocation against the real clap tree.
///
/// Returns `Err(reason)` naming the first token the parser would reject.
fn resolve(root: &clap::Command, quoted: &str) -> Result<(), String> {
    let mut cmd = root;
    let mut tokens = quoted.split_whitespace();
    let _apr = tokens.next(); // "apr"
    let mut path = String::from("apr");
    let mut positionals_used = 0usize;

    // Set when the previous token was a long flag that takes a value, so the
    // next bare word is that value and not a subcommand or positional.
    let mut awaiting_value = false;

    for tok in tokens {
        if let Some(flag) = tok.strip_prefix("--") {
            awaiting_value = false;
            let name = flag.split('=').next().unwrap_or(flag);
            if name.is_empty() {
                continue; // bare `--`
            }
            let matches_long = |a: &clap::Arg| {
                a.get_long() == Some(name)
                    || a.get_all_aliases()
                        .is_some_and(|al| al.iter().any(|x| *x == name))
            };
            // clap propagates `global = true` args from the root to every
            // subcommand, so `--json` is legal on any of them.
            let Some(arg) = cmd
                .get_arguments()
                .find(|a| matches_long(a))
                .or_else(|| root.get_arguments().find(|a| a.is_global_set() && matches_long(a)))
            else {
                return Err(format!("`{path}` has no flag `--{name}`"));
            };
            awaiting_value = !flag.contains('=')
                && arg
                    .get_num_args()
                    .is_none_or(|r| r.takes_values())
                && arg.get_action().takes_values();
            continue;
        }
        if let Some(short) = tok.strip_prefix('-') {
            // Short flags are not name-checked (help text uses them rarely),
            // but a short flag that takes a value consumes the next word.
            let c = short.chars().next();
            awaiting_value = c.is_some_and(|c| {
                cmd.get_arguments()
                    .chain(root.get_arguments().filter(|a| a.is_global_set()))
                    .any(|a| a.get_short() == Some(c) && a.get_action().takes_values())
            }) && short.len() == 1;
            continue;
        }
        if tok.starts_with('<') || tok.starts_with('"') {
            awaiting_value = false;
            continue; // placeholders are not checked
        }
        if awaiting_value {
            awaiting_value = false;
            continue; // this bare word is the previous flag's value
        }
        // A bare word is a subcommand while the command still has subcommands
        // and has not started consuming positionals; otherwise it is a
        // positional VALUE — and there are only so many of those.
        let sub = if positionals_used == 0 {
            cmd.get_subcommands()
                .find(|s| s.get_name() == tok || s.get_all_aliases().any(|a| a == tok))
        } else {
            None
        };
        if let Some(sub) = sub {
            cmd = sub;
            path = format!("{path} {tok}");
            continue;
        }
        if positionals_used < positional_capacity(cmd) {
            positionals_used += 1;
            continue;
        }
        if cmd.get_subcommands().next().is_some() {
            return Err(format!("`{path}` has no subcommand `{tok}`"));
        }
        return Err(format!(
            "`{path}` takes {} positional(s); `{tok}` is one too many",
            positional_capacity(cmd)
        ));
    }
    Ok(())
}

/// How many bare words this command can swallow as positional values.
/// A variadic positional (`num_args` with no upper bound) means unlimited.
fn positional_capacity(cmd: &clap::Command) -> usize {
    let mut n = 0usize;
    for p in cmd.get_positionals() {
        match p.get_num_args() {
            Some(r) if r.max_values() > 1 => return usize::MAX,
            _ => n += 1,
        }
    }
    n
}

/// Collect every quoted `apr …` invocation reachable from the CLI tree.
fn collect(cmd: &clap::Command, prefix: &str, out: &mut Vec<Quoted>) {
    let site = if prefix.is_empty() {
        cmd.get_name().to_string()
    } else {
        format!("{prefix} {}", cmd.get_name())
    };
    for help in [cmd.get_about(), cmd.get_long_about()].into_iter().flatten() {
        out.extend(quoted_apr_invocations(&site, &help.to_string()));
    }
    for arg in cmd.get_arguments() {
        let arg_site = format!("{site} --{}", arg.get_long().unwrap_or(arg.get_id().as_str()));
        for help in [arg.get_help(), arg.get_long_help()].into_iter().flatten() {
            out.extend(quoted_apr_invocations(&arg_site, &help.to_string()));
        }
    }
    for sub in cmd.get_subcommands() {
        collect(sub, &site, out);
    }
}

/// The 16 CRUX `*-lint` consumers are the surface the audit measured: each
/// documents the producer whose output it reads.
#[test]
fn every_apr_command_quoted_in_lint_help_exists() {
    let violations = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            use clap::CommandFactory;
            let root = Cli::command();
            let mut quoted = Vec::new();
            collect(&root, "", &mut quoted);
            quoted
                .into_iter()
                .filter(|q| q.site.contains("-lint"))
                .filter_map(|q| {
                    resolve(&root, &q.text)
                        .err()
                        .map(|why| format!("{}: `{}` — {why}", q.site, q.text))
                })
                .collect::<Vec<_>>()
        })
        .expect("spawn")
        .join()
        .expect("join");

    assert!(
        violations.is_empty(),
        "help text names {} apr invocation(s) the binary cannot run:\n  {}",
        violations.len(),
        violations.join("\n  ")
    );
}

/// The guard itself must be able to fail — a check that cannot go red is
/// theater. These are the exact strings 0.63.0 shipped.
#[test]
fn resolver_rejects_the_invocations_dogfooding_found() {
    let cases = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            use clap::CommandFactory;
            let root = Cli::command();
            // aprender#2377 finding 3 IMPLEMENTED three of the eight producers,
            // so `apr dataset audio-inspect …`, `apr kernel parity …` and
            // `apr debug embed-viz …` moved to the accepts-list below. What is
            // left here is still missing, and `apr debug model.gguf embed-viz`
            // stays: `embed-viz` is a subcommand of `debug`, not a word that
            // may follow the model path.
            [
                "apr attn-viz model.gguf",
                "apr trace model.gguf --check-finite",
                "apr finetune --parallel ddp",
                "apr debug model.gguf embed-viz",
                "apr profile model.gguf --gpu-memory-trace",
                "apr quantize model.apr --imatrix calib.jsonl",
            ]
            .into_iter()
            .map(|t| (t, resolve(&root, t)))
            .collect::<Vec<_>>()
        })
        .expect("spawn")
        .join()
        .expect("join");

    for (text, result) in cases {
        assert!(
            result.is_err(),
            "resolver accepted `{text}`, which the shipped parser rejects with exit 2"
        );
    }
}

/// Control: invocations that DO work must resolve, or the guard would just be
/// deleting help text.
#[test]
fn resolver_accepts_real_invocations() {
    let cases = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            use clap::CommandFactory;
            let root = Cli::command();
            [
                "apr inspect model.apr",
                "apr inspect model.apr --json",
                "apr serve run model.gguf",
                "apr validate model.apr --quality",
                "apr imatrix-lint --observation-file obs.json",
                "apr imatrix-lint --observation-file obs.json --json",
                "apr quantize model.gguf --scheme q4k -o out.apr",
                "apr export model.apr --format gguf",
                "apr rm model",
                // aprender#2377 finding 3: the three producers this batch added.
                "apr dataset audio-inspect clip.wav --format json",
                "apr dataset audio-inspect clip.wav --format json -o audio.json",
                "apr kernel parity --impl tiled --ref naive --json",
                "apr kernel parity --impl flash2 --ref naive --head-dim 96 --json",
                "apr debug embed-viz --model model.apr --seed 42 -o emb.csv",
                "apr debug model.apr --hex",
            ]
            .into_iter()
            .map(|t| (t, resolve(&root, t)))
            .collect::<Vec<_>>()
        })
        .expect("spawn")
        .join()
        .expect("join");

    for (text, result) in cases {
        assert!(result.is_ok(), "resolver rejected a real command: {text} — {result:?}");
    }
}
