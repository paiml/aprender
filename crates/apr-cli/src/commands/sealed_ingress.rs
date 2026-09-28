//! train-ingress-sealed-refusal-v1, the part that can be met today: an
//! rc-bound training run must name a non-empty sealed manifest (FALSIFY-TIS-005).
//!
//! Matching samples against the manifest (TIS-001/002, exact and near-duplicate
//! with the trace-dedup-v1 normaliser) is NOT implemented yet. A run that loads a
//! manifest therefore says so on stderr and makes no sealed-hits claim: the gate
//! only guarantees that "0 sealed hits" can never be reported against an empty
//! or absent set.

use crate::error::CliError;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

/// A loaded sealed manifest. `matching` stays false until TIS-001/002 land, so a
/// receipt writer cannot record `sealed_hits = 0` from this value alone.
#[derive(Debug)]
pub(crate) struct SealedGate {
    pub(crate) manifest_sha256: String,
    pub(crate) items: usize,
    pub(crate) matching: bool,
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Parse a sealed manifest: an optional title line, then one item per line,
/// `<id> <sha256> [hunk_sha256,…]` (the review-corpus-v1 test-manifest format).
/// Blank and `#` lines are skipped. Any other line is an error, never ignored.
pub(crate) fn parse_manifest(text: &str) -> Result<usize, String> {
    let mut items = 0usize;
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields.as_slice() {
            [_, sha, ..] if is_sha256(sha) => items += 1,
            // A title line such as "review-corpus-v1 test-manifest", first line only.
            _ if n == 0 && items == 0 => {}
            _ => {
                return Err(format!(
                    "line {}: expected `<id> <sha256> [hunks]`: {line}",
                    n + 1
                ))
            }
        }
    }
    Ok(items)
}

/// FALSIFY-TIS-005. Runs before any model or data I/O of `verb`.
pub(crate) fn gate(
    verb: &str,
    rc_bound: bool,
    manifest: Option<&Path>,
) -> Result<Option<SealedGate>, CliError> {
    let refuse = |why: String| {
        CliError::ValidationFailed(format!(
            "apr {verb}: {why} (train-ingress-sealed-refusal-v1 TIS-005)"
        ))
    };
    let Some(path) = manifest else {
        return if rc_bound {
            Err(refuse(
                "--rc-bound needs --sealed-manifest <PATH>; an rc run without a sealed set \
                 cannot claim 0 sealed items in its training data"
                    .to_string(),
            ))
        } else {
            Ok(None)
        };
    };
    let bytes =
        fs::read(path).map_err(|e| refuse(format!("cannot read {}: {e}", path.display())))?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|e| refuse(format!("{} is not UTF-8: {e}", path.display())))?;
    let items = parse_manifest(text).map_err(|e| refuse(format!("{}: {e}", path.display())))?;
    if items == 0 {
        return Err(refuse(format!(
            "sealed manifest {} has 0 items; an empty set makes every sealed-hits claim vacuous",
            path.display()
        )));
    }
    let manifest_sha256 = format!("{:x}", Sha256::digest(&bytes));
    eprintln!(
        "[TIS-005] sealed manifest {} sha256 {manifest_sha256} ({items} items). \
         Sample matching (TIS-001/002) is not implemented: this run makes NO sealed-hits claim.",
        path.display()
    );
    Ok(Some(SealedGate {
        manifest_sha256,
        items,
        matching: false,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str = "dbf2e76f07bdf31bde4a9c92e2d8698c46b80224ad101876525711f71afca173";
    const SHA_B: &str = "52b05d908f3e82ae8c8a0a2fd25a91b4a1a8fbc035892ca41a35b779ec7c5296";

    fn write(dir: &tempfile::TempDir, name: &str, body: &str) -> std::path::PathBuf {
        let p = dir.path().join(name);
        fs::write(&p, body).expect("write manifest");
        p
    }

    #[test]
    fn falsify_tis_005_rc_bound_without_manifest_is_refused() {
        let err = gate("finetune", true, None).expect_err("rc run with no manifest");
        let msg = err.to_string();
        assert!(
            msg.contains("--sealed-manifest") && msg.contains("TIS-005"),
            "{msg}"
        );
    }

    #[test]
    fn falsify_tis_005_empty_manifest_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        for (name, body) in [
            ("empty.txt", ""),
            ("title-only.txt", "review-corpus-v1 test-manifest\n"),
            ("comments.txt", "# nothing sealed\n\n"),
        ] {
            let p = write(&dir, name, body);
            // Refused whether or not the run is rc-bound: an empty set is never a claim.
            for rc in [true, false] {
                let err = gate("distill", rc, Some(&p)).expect_err(name).to_string();
                assert!(err.contains("0 items"), "{name}: {err}");
            }
        }
    }

    #[test]
    fn falsify_tis_005_unreadable_manifest_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("absent.txt");
        assert!(gate("pretrain", true, Some(&missing)).is_err());
        let garbled = write(
            &dir,
            "garbled.txt",
            &format!("G-1 {SHA_A}\nnot a manifest line\n"),
        );
        let err = gate("pretrain", true, Some(&garbled))
            .expect_err("garbled")
            .to_string();
        assert!(err.contains("line 2"), "{err}");
    }

    #[test]
    fn manifest_with_items_passes_and_claims_no_matching() {
        let dir = tempfile::tempdir().expect("tempdir");
        let body = format!(
            "review-corpus-v1 test-manifest\nG-pr1 {SHA_A} {SHA_B},{SHA_A}\nG-pr2 {SHA_B}\n"
        );
        let p = write(&dir, "sealed.txt", &body);
        let g = gate("finetune", true, Some(&p))
            .expect("non-empty manifest passes")
            .expect("gate value");
        assert_eq!(g.items, 2);
        assert_eq!(
            g.manifest_sha256,
            format!("{:x}", Sha256::digest(body.as_bytes()))
        );
        assert!(!g.matching, "no sample matching exists yet (TIS-001/002)");
    }

    #[test]
    fn non_rc_run_without_manifest_is_unchanged() {
        assert!(gate("finetune", false, None).expect("non-rc run").is_none());
    }
}
