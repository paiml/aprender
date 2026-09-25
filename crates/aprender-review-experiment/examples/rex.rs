//! `cargo run -p aprender-review-experiment --example rex -- <command>`
//!
//! An example, not a `[[bin]]`: the workspace binary register only shrinks
//! (monorepo_invariants::test_no_unauthorized_binaries).
//!
//! Commands:
//! - `prereg`        print the prereg lock the tree implies
//! - `prereg-check`  exit 1 unless the committed lock matches the tree

use aprender_review_experiment::prereg;
use std::process::ExitCode;

fn main() -> ExitCode {
    let cmd = std::env::args().nth(1).unwrap_or_default();
    let Some(c) = prereg::Components::in_tree() else {
        eprintln!("rex: spec lacks a §2..§6 span");
        return ExitCode::from(2);
    };
    match cmd.as_str() {
        "prereg" => {
            print!("{}", c.render_lock());
            ExitCode::SUCCESS
        }
        "prereg-check" => {
            let bad = prereg::verify(prereg::LOCK, &c);
            if bad.is_empty() {
                println!("rex-prereg-v1 OK prereg_sha={}", c.prereg_sha());
                ExitCode::SUCCESS
            } else {
                for b in &bad {
                    eprintln!("rex-prereg-v1 DRIFT {b}");
                }
                ExitCode::from(1)
            }
        }
        _ => {
            eprintln!("usage: rex <prereg|prereg-check>");
            ExitCode::from(2)
        }
    }
}
