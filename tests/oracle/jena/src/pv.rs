//! pv's side of R-INPUT and R-RETURN (S3): `pv ontology read X` dumps X as N-Triples, or refuses it naming a line.

use std::path::Path;
use std::process::Command;

use crate::nt::{self, Triple};

/// What `pv ontology read` did with one file.
pub enum Read {
    /// Exit 0: the dump as pv wrote it, and the harness's parse of it.
    Dump(String, Vec<Triple>),
    /// Exit 1: pv refused the file. The line it names, and its first stderr line.
    Refused(Option<usize>, String),
    /// pv ran and broke the command's promise: another exit code, a signal, or a dump that is not N-Triples.
    Broken(String),
}

/// Run `pv ontology read <file> --base <base>`. `Err` only when pv cannot be started, so nothing is known.
pub fn read(pv: &Path, file: &Path, base: &str) -> Result<Read, String> {
    let o = Command::new(pv)
        .args(["ontology", "read"])
        .arg(file)
        .args(["--base", base])
        .output()
        .map_err(|e| format!("{}: {e}", pv.display()))?;
    let stderr = String::from_utf8_lossy(&o.stderr);
    let first = stderr.lines().next().unwrap_or("").to_string();
    Ok(match o.status.code() {
        Some(0) => match String::from_utf8(o.stdout) {
            Ok(text) => match nt::parse(&text) {
                Ok(t) => Read::Dump(text, t),
                Err(e) => Read::Broken(format!("pv's dump is not N-Triples: {e}")),
            },
            Err(e) => Read::Broken(format!("pv's dump is not UTF-8: {e}")),
        },
        Some(1) => Read::Refused(error_line(&first), first),
        c => Read::Broken(format!("pv ontology read exited {c:?}: {first}")),
    })
}

/// The first run of digits after `marker`.
fn number_after(text: &str, marker: &str) -> Option<usize> {
    let rest = &text[text.find(marker)? + marker.len()..];
    rest.split(|c: char| !c.is_ascii_digit())
        .next()?
        .parse()
        .ok()
}

/// The line pv names: `error: <file>: line L, col C: …`.
pub fn error_line(message: &str) -> Option<usize> {
    number_after(message, ": line ")
}

/// The line riot names: `… [line: L, col: C ] …`.
pub fn riot_line(err: &str) -> Option<usize> {
    number_after(err, "[line: ")
}

/// How many distinct blank nodes `t` holds.
pub fn blank_nodes(t: &[Triple]) -> usize {
    t.iter()
        .flat_map(|t| [&t[0], &t[2]])
        .filter(|x| x.is_blank())
        .collect::<std::collections::BTreeSet<_>>()
        .len()
}
