//! ttop surface gate (#4511 item 4): the code and `contracts/aprender-viz-ttop-surface-v1.yaml`
//! declare the same flags, key bindings and panels, and every flag has a ledger row.
//!
//! Every check is two-directional. A flag, key or panel the code gained without a
//! declaration is RED; one the contract declares but the code dropped is RED too.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Command;

use aprender_viz_ttop::runtime::parse_panel_type;
use presentar_terminal::ptop::PanelType;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(repo_root().join(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"))
}

/// The whitespace-separated tokens of `equations.<name>.formula` in the contract.
fn contract_tokens(name: &str) -> BTreeSet<String> {
    let yaml = read("contracts/aprender-viz-ttop-surface-v1.yaml");
    let header = format!("  {name}:");
    let mut lines = yaml.lines().skip_while(|l| l.trim_end() != header).skip(1);
    let formula = lines
        .find_map(|l| l.trim().strip_prefix("formula: "))
        .unwrap_or_else(|| panic!("contract has no equations.{name}.formula"));
    formula
        .trim_matches('"')
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

fn diff(label: &str, code: &BTreeSet<String>, contract: &BTreeSet<String>) -> Vec<String> {
    let mut out = Vec::new();
    for t in code.difference(contract) {
        out.push(format!(
            "{label}: `{t}` is in the code but not declared in the contract"
        ));
    }
    for t in contract.difference(code) {
        out.push(format!(
            "{label}: `{t}` is declared in the contract but the code has none"
        ));
    }
    out
}

/// Long flags in `--help` text: every `--name` token that opens an option line.
fn help_flags(help: &str) -> BTreeSet<String> {
    help.lines()
        .filter_map(|l| {
            let t = l.trim_start();
            // `-r, --refresh`: drop the short form. Only the `-x, ` prefix, never the
            // first ", " in the line: a description with a comma in it would eat the flag.
            let t = match t.as_bytes() {
                [b'-', c, b',', b' ', ..] if *c != b'-' => &t[4..],
                _ => t,
            };
            t.strip_prefix("--").map(|r| {
                let name: String = r
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '-')
                    .collect();
                format!("--{name}")
            })
        })
        .collect()
}

/// Key tokens a handler's match arms name, spelled as the contract spells them.
fn handler_keys(body: &str) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    for line in body.lines() {
        let ctrl = line.contains("KeyModifiers::CONTROL") && !line.contains("!modifiers");
        let arm = line.split("=>").next().unwrap_or("");
        if !line.contains("=>") {
            continue;
        }
        let mut rest = arm;
        while let Some(i) = rest.find("KeyCode::") {
            rest = &rest[i + "KeyCode::".len()..];
            let name: String = rest
                .chars()
                .take_while(char::is_ascii_alphanumeric)
                .collect();
            match name.as_str() {
                "Char" => {
                    let inner = &rest[5..rest.find(')').expect("Char( closes")];
                    if !inner.contains('\'') {
                        keys.insert("Char(any)".to_string());
                    }
                    for part in inner.split('|') {
                        let c = part.trim().trim_matches('\'');
                        if c.is_empty() || !part.contains('\'') {
                            continue;
                        }
                        let tok = if c == " " {
                            "Space".to_string()
                        } else {
                            c.to_string()
                        };
                        keys.insert(if ctrl { format!("C-{tok}") } else { tok });
                    }
                }
                "F" => {
                    let n: String = rest[2..].chars().take_while(char::is_ascii_digit).collect();
                    keys.insert(format!("F{n}"));
                }
                _ => {
                    keys.insert(name);
                }
            }
        }
    }
    keys
}

/// The body of `fn <name>` in app.rs: from its signature to the next `fn ` at the same indent.
fn fn_body<'a>(src: &'a str, name: &str) -> &'a str {
    let sig = format!("fn {name}(");
    let start = src
        .find(&sig)
        .unwrap_or_else(|| panic!("app.rs has no {sig}"));
    let after = &src[start + sig.len()..];
    let end = after
        .find("\n    fn ")
        .or_else(|| after.find("\n    pub fn "))
        .unwrap_or(after.len());
    &after[..end]
}

#[test]
fn flags_match_help() {
    let out = Command::new(env!("CARGO_BIN_EXE_aprender-viz-ttop"))
        .arg("--help")
        .output()
        .expect("run ttop --help");
    assert!(out.status.success(), "ttop --help exited {:?}", out.status);
    let code = help_flags(&String::from_utf8_lossy(&out.stdout));
    assert!(
        code.len() >= 10,
        "parsed only {} flags from --help: {code:?}",
        code.len()
    );
    let errs = diff("flag", &code, &contract_tokens("cli_flags"));
    assert!(errs.is_empty(), "{}", errs.join("\n"));
}

#[test]
fn every_flag_has_a_ledger_row() {
    let csv = read("docs/audits/surface_audit.csv");
    let rows: Vec<&str> = csv
        .lines()
        .filter(|l| l.starts_with("aprender-viz-ttop,"))
        .map(|l| l.split(',').nth(1).unwrap_or(""))
        .collect();
    let missing: Vec<String> = contract_tokens("cli_flags")
        .into_iter()
        .filter(|f| {
            !rows
                .iter()
                .any(|feat| feat.split_whitespace().any(|w| w == f))
        })
        .collect();
    assert!(
        missing.is_empty(),
        "no aprender-viz-ttop row in surface_audit.csv for: {missing:?}"
    );
}

#[test]
fn keys_match_handlers() {
    let src = read("crates/aprender-present-terminal/src/ptop/app.rs");
    let mut errs = Vec::new();
    for (mode, handler) in [
        ("normal", "handle_normal_mode_key"),
        ("help", "handle_help_mode_key"),
        ("signal_confirmation", "handle_signal_confirmation_key"),
        ("exploded", "handle_exploded_mode_key"),
        ("filter_input", "handle_filter_input_key"),
    ] {
        let code = handler_keys(fn_body(&src, handler));
        assert!(
            !code.is_empty(),
            "parsed no keys from {handler}: the parser is blind, not the handler empty"
        );
        errs.extend(diff(
            &format!("keys_{mode}"),
            &code,
            &contract_tokens(&format!("keys_{mode}")),
        ));
    }
    assert!(errs.is_empty(), "{}", errs.join("\n"));
}

#[test]
fn panels_match_panel_type() {
    let declared = contract_tokens("panels");
    let mut errs = Vec::new();
    for p in PanelType::all() {
        if !declared
            .iter()
            .any(|n| parse_panel_type(n).ok() == Some(*p))
        {
            errs.push(format!(
                "PanelType::{p:?} has no declared panel name that --explode parses to it"
            ));
        }
    }
    for n in &declared {
        match parse_panel_type(n) {
            Ok(p) if PanelType::all().contains(&p) => {}
            other => errs.push(format!(
                "declared panel `{n}` does not parse to a PanelType::all() entry: {other:?}"
            )),
        }
    }
    assert_eq!(
        declared.len(),
        PanelType::all().len(),
        "one declared name per panel"
    );
    assert!(errs.is_empty(), "{}", errs.join("\n"));
}

/// The parsers above are the gate; prove they see what they claim to (case table).
#[test]
fn parser_case_table() {
    let keys = handler_keys(
        "match code {\n KeyCode::Char('q') | KeyCode::Esc => return true,\n KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => x,\n KeyCode::Tab if !modifiers.contains(KeyModifiers::SHIFT) => y,\n KeyCode::Char('/' | 'f') => z,\n KeyCode::Enter | KeyCode::Char(' ') => w,\n KeyCode::F(1) => v,\n KeyCode::Char(c) => self.filter.push(c),\n _ => {}\n}",
    );
    let want: BTreeSet<String> = [
        "q",
        "Esc",
        "C-c",
        "Tab",
        "/",
        "f",
        "Enter",
        "Space",
        "F1",
        "Char(any)",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(keys, want);
    let flags = help_flags("Usage: x\n\nOptions:\n  -r, --refresh <MS>  Refresh (a, b)\n      --no-color  x\n      --explode <PANEL>  Explode (cpu, memory, disk)\n  -h, --help  Print help\n  -V, --version  v\n");
    let want: BTreeSet<String> = [
        "--refresh",
        "--no-color",
        "--explode",
        "--help",
        "--version",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(flags, want);
}
