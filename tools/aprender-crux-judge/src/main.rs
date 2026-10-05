//! `aprender-crux-judge collect ...`: the Rust port of the CRUX inference
//! judge (`crux_inference_judge.py`). Same flags, same receipt, same table,
//! same exit codes: 0 PASS, 1 RED, 2 DECLINE. README.md lists where it
//! differs.
//!
//! `aprender-crux-judge sandbox` (port only, not in the usage line) probes the
//! code_tests sandbox and prints the interpreter version a receipt records:
//! rc 0 when the sandbox works, rc 2 `sandbox_unavailable: <why>` when not.
//!
//! The retired Python CLIs live on as subcommands, with their own exit codes:
//! `eval` / `lint` (`crux_oracles.py`, see `oracles::cli`; `extract` is new) and `certify` /
//! `check` (`crux_prompt_certify.py`, see `certify::cli`).

mod certify;
mod collect;
mod judge;
mod oracles;
mod pyerr;
mod pyio;
mod pyjson;
mod pyre;
mod pyval;
mod sandbox;
mod serve_routes;
mod unicode_tables;

use std::process::ExitCode;

const PROG: &str = "aprender-crux-judge";

const USAGE: &str =
    "usage: aprender-crux-judge collect [-h] --manifest MANIFEST --prompts PROMPTS \
                     --meta META --out-json OUT_JSON --out-md OUT_MD \
                     [--certification CERTIFICATION]";

/// The `collect` flags; the last is the only optional one.
const FLAGS: [&str; 6] = [
    "--manifest",
    "--prompts",
    "--meta",
    "--out-json",
    "--out-md",
    "--certification",
];

enum Parsed {
    Run(collect::Args),
    Exit(u8),
}

/// argparse's usage error: the usage line and the message, exit code 2.
fn usage_error(msg: &str) -> u8 {
    eprintln!("{USAGE}\n{PROG}: error: {msg}");
    2
}

/// argparse's long-option lookup: an exact name, else a unique prefix.
fn resolve(opt: &str) -> Result<Option<usize>, String> {
    if opt == "--help" {
        return Ok(None);
    }
    if let Some(i) = FLAGS.iter().position(|f| *f == opt) {
        return Ok(Some(i));
    }
    let mut hits: Vec<&str> = FLAGS
        .iter()
        .copied()
        .filter(|f| f.starts_with(opt))
        .collect();
    if "--help".starts_with(opt) {
        hits.push("--help");
    }
    match hits.as_slice() {
        [] => Err(format!("unrecognized arguments: {opt}")),
        ["--help"] => Ok(None),
        [one] => Ok(FLAGS.iter().position(|f| f == one)),
        many => Err(format!(
            "ambiguous option: {opt} could match {}",
            many.join(", ")
        )),
    }
}

fn help() -> u8 {
    println!("{USAGE}");
    0
}

/// The subcommand: `None` when it is `collect`, else the exit code.
fn check_cmd(cmd: Option<&str>) -> Option<u8> {
    match cmd {
        None => Some(usage_error("the following arguments are required: cmd")),
        Some("-h" | "--help") => Some(help()),
        Some("collect") => None,
        Some(other) => Some(usage_error(&format!(
            "argument cmd: invalid choice: '{other}' (choose from 'collect')"
        ))),
    }
}

/// One option argument: its flag index and any `=value`, or the exit code.
fn option_index(arg: &str) -> Result<(usize, Option<String>), u8> {
    if arg == "-h" {
        return Err(help());
    }
    if !arg.starts_with("--") || arg.len() <= 2 {
        return Err(usage_error(&format!("unrecognized arguments: {arg}")));
    }
    let (opt, inline) = match arg.split_once('=') {
        Some((o, v)) => (o, Some(v.to_string())),
        None => (arg, None),
    };
    match resolve(opt) {
        Ok(Some(idx)) => Ok((idx, inline)),
        Ok(None) => Err(help()),
        Err(msg) => Err(usage_error(&msg)),
    }
}

/// An option's value: the inline `=value`, else the next argument unless it
/// looks like an option.
fn option_value(
    argv: &[String],
    i: &mut usize,
    inline: Option<String>,
    idx: usize,
) -> Result<String, u8> {
    if let Some(v) = inline {
        return Ok(v);
    }
    match argv.get(*i) {
        Some(v) if !(v.starts_with('-') && v.len() > 1) => {
            *i += 1;
            Ok(v.clone())
        }
        _ => Err(usage_error(&format!(
            "argument {}: expected one argument",
            FLAGS[idx]
        ))),
    }
}

fn parse_flags(argv: &[String]) -> Result<[Option<String>; 6], u8> {
    let mut vals: [Option<String>; 6] = Default::default();
    let mut i = 1;
    while i < argv.len() {
        let (idx, inline) = option_index(&argv[i])?;
        i += 1;
        vals[idx] = Some(option_value(argv, &mut i, inline, idx)?);
    }
    Ok(vals)
}

fn parse(argv: &[String]) -> Parsed {
    if let Some(rc) = check_cmd(argv.first().map(String::as_str)) {
        return Parsed::Exit(rc);
    }
    let vals = match parse_flags(argv) {
        Ok(vals) => vals,
        Err(rc) => return Parsed::Exit(rc),
    };
    let missing: Vec<&str> = FLAGS[..5]
        .iter()
        .zip(&vals)
        .filter(|(_, v)| v.is_none())
        .map(|(f, _)| *f)
        .collect();
    if !missing.is_empty() {
        return Parsed::Exit(usage_error(&format!(
            "the following arguments are required: {}",
            missing.join(", ")
        )));
    }
    let [manifest, prompts, meta, out_json, out_md, certification] = vals;
    Parsed::Run(collect::Args {
        manifest: manifest.unwrap_or_default(),
        prompts: prompts.unwrap_or_default(),
        meta: meta.unwrap_or_default(),
        out_json: out_json.unwrap_or_default(),
        out_md: out_md.unwrap_or_default(),
        certification,
    })
}

/// The `sandbox` report line and its exit code.
fn sandbox_report(probed: Result<&sandbox::Sandbox, &str>) -> (String, u8) {
    match probed {
        Ok(sb) => (
            format!(
                "sandbox: {}; prlimit cpu/as/fsize, unshare -rn, python3 -I, env PATH=/usr/bin:/bin",
                sb.version
            ),
            0,
        ),
        Err(why) => (format!("sandbox_unavailable: {why}"), 2),
    }
}

fn main() -> ExitCode {
    let mut argv = Vec::new();
    for a in std::env::args_os().skip(1) {
        match a.into_string() {
            Ok(s) => argv.push(s),
            Err(a) => {
                return ExitCode::from(usage_error(&format!(
                    "argument is not UTF-8: {}",
                    a.to_string_lossy()
                )));
            }
        }
    }
    match argv.first().map(String::as_str) {
        Some("eval" | "extract" | "lint") => return ExitCode::from(oracles::cli(&argv)),
        Some("certify" | "check") => return ExitCode::from(certify::cli(&argv)),
        _ => {}
    }
    if argv.first().map(String::as_str) == Some("sandbox") {
        if let Some(extra) = argv.get(1) {
            return ExitCode::from(usage_error(&format!("unrecognized arguments: {extra}")));
        }
        let (line, rc) = sandbox_report(sandbox::sandbox());
        println!("{line}");
        return ExitCode::from(rc);
    }
    let args = match parse(&argv) {
        Parsed::Run(args) => args,
        Parsed::Exit(rc) => return ExitCode::from(rc),
    };
    match collect::collect(&args) {
        Ok(rc) => ExitCode::from(u8::try_from(rc).unwrap_or(1)),
        Err(e) if e.declines() => {
            eprintln!("decline: {}: {e}", e.kind);
            ExitCode::from(2)
        }
        Err(e) => {
            eprintln!("crash: {}: {e}", e.kind);
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn flags_resolve_like_argparse() {
        assert_eq!(resolve("--meta"), Ok(Some(2)));
        assert_eq!(resolve("--cert"), Ok(Some(5)));
        assert_eq!(resolve("--he"), Ok(None));
        assert!(resolve("--out")
            .unwrap_err()
            .contains("--out-json, --out-md"));
        assert!(resolve("--m").unwrap_err().starts_with("ambiguous"));
        assert!(resolve("--nope").unwrap_err().starts_with("unrecognized"));
    }

    #[test]
    fn parse_collect() {
        let Parsed::Run(a) = parse(&argv(
            "collect --manifest m --prompts=p --meta x --out-json j --out-md d --cert c",
        )) else {
            panic!("should parse");
        };
        assert_eq!(
            (a.manifest, a.prompts, a.meta, a.out_json, a.out_md),
            ("m".into(), "p".into(), "x".into(), "j".into(), "d".into())
        );
        assert_eq!(a.certification.as_deref(), Some("c"));
        assert!(matches!(
            parse(&argv("collect --manifest m")),
            Parsed::Exit(2)
        ));
        assert!(matches!(parse(&argv("")), Parsed::Exit(2)));
        assert!(matches!(parse(&argv("judge")), Parsed::Exit(2)));
        assert!(matches!(parse(&argv("collect -h")), Parsed::Exit(0)));
    }

    #[test]
    fn sandbox_report_lines() {
        let (line, rc) = sandbox_report(Ok(sandbox::tests::host_sandbox()));
        assert_eq!(rc, 0);
        assert!(line.starts_with("sandbox: Python 3."), "{line}");
        assert_eq!(
            sandbox_report(Err("no python3 on PATH")),
            ("sandbox_unavailable: no python3 on PATH".to_string(), 2)
        );
    }
}
