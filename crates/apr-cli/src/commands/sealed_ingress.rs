//! train-ingress-sealed-refusal-v1: no sealed eval item reaches a training run.
//!
//! - TIS-005: an rc-bound run must name a non-empty sealed manifest.
//! - TIS-001: a finetune or distill sample byte-equal to a sealed item, or to one
//!   of its hunks, is refused before step 0, naming the sealed id. The manifest
//!   (review-corpus-v1 test-manifest) carries sha256 digests only, so this is
//!   exact matching: each data line, and each JSON string in it, is hashed.
//! - TIS-003: `apr pretrain` reads u32 token shards, which cannot be matched
//!   against a sha-only manifest; an rc-bound pretrain is refused by name rather
//!   than run unchecked.
//! - TIS-002 (near-duplicates via the trace-dedup-v1 normaliser) is NOT
//!   implemented: the normaliser exists nowhere yet. A perturbed sealed item
//!   still passes, and the run says so.

use crate::error::CliError;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

/// A loaded sealed manifest. `matching` is true only for verbs whose ingress
/// is checked (finetune, distill); a receipt may record `sealed_hits = 0` only
/// from a gate with `matching` set.
#[derive(Debug, Clone)]
pub(crate) struct SealedGate {
    pub(crate) manifest_sha256: String,
    pub(crate) items: usize,
    pub(crate) matching: bool,
    /// sha256 (item or hunk) -> sealed id.
    digests: HashMap<String, String>,
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// One manifest line's item digest, then each of its hunk digests, all
/// lowercase. `None` if any hunk is not a sha256.
fn line_digests(sha: &str, hunks: &[&str]) -> Option<Vec<String>> {
    let mut shas = vec![sha.to_ascii_lowercase()];
    for hunk in hunks
        .iter()
        .flat_map(|h| h.split(','))
        .filter(|h| !h.is_empty())
    {
        if !is_sha256(hunk) {
            return None;
        }
        shas.push(hunk.to_ascii_lowercase());
    }
    Some(shas)
}

/// Parse a review-corpus-v1 test manifest: an optional title line, then
/// `<id> <sha256> [hunk-sha256,...]` per item. Returns each id with its item
/// and hunk digests (lowercase).
pub(crate) fn parse_manifest(text: &str) -> Result<Vec<(String, Vec<String>)>, String> {
    let mut items = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let bad = || format!("line {}: expected `<id> <sha256> [hunks]`: {line}", n + 1);
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields.as_slice() {
            [id, sha, rest @ ..] if is_sha256(sha) => {
                let shas = line_digests(sha, rest).ok_or_else(bad)?;
                items.push(((*id).to_string(), shas));
            }
            _ if n == 0 && items.is_empty() => {}
            _ => return Err(bad()),
        }
    }
    Ok(items)
}

fn refusal(verb: &str, rule: &str, why: String) -> CliError {
    CliError::ValidationFailed(format!(
        "apr {verb}: {why} (train-ingress-sealed-refusal-v1 {rule})"
    ))
}

/// Load the sealed manifest for `verb`. Refuses an rc-bound run with no
/// manifest (TIS-005), any empty or malformed manifest (TIS-005), and an
/// rc-bound run of a verb whose ingress cannot be checked (TIS-003).
pub(crate) fn gate(
    verb: &str,
    rc_bound: bool,
    manifest: Option<&Path>,
) -> Result<Option<SealedGate>, CliError> {
    let refuse = |why: String| refusal(verb, "TIS-005", why);
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
    let entries = parse_manifest(text).map_err(|e| refuse(format!("{}: {e}", path.display())))?;
    if entries.is_empty() {
        return Err(refuse(format!(
            "sealed manifest {} has 0 items; an empty set makes every sealed-hits claim vacuous",
            path.display()
        )));
    }
    let matching = verb != "pretrain";
    if !matching && rc_bound {
        return Err(refusal(
            verb,
            "TIS-003",
            "pretrain reads u32 token shards, which cannot be checked against a sha-only \
             sealed manifest; an rc-bound pretrain would train on unchecked data"
                .to_string(),
        ));
    }
    let manifest_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let items = entries.len();
    let digests = entries
        .into_iter()
        .flat_map(|(id, shas)| shas.into_iter().map(move |s| (s, id.clone())))
        .collect();
    if matching {
        eprintln!(
            "[TIS-001] sealed manifest {} sha256 {manifest_sha256} ({items} items): exact \
             matching on; near-duplicate matching (TIS-002) is NOT implemented.",
            path.display()
        );
    } else {
        eprintln!(
            "[TIS-003] sealed manifest {} sha256 {manifest_sha256} ({items} items): {verb} \
             ingress is not checked; this run makes NO sealed-hits claim.",
            path.display()
        );
    }
    Ok(Some(SealedGate {
        manifest_sha256,
        items,
        matching,
        digests,
    }))
}

/// Gate `verb` (TIS-005/003), make the gate active for loaders deep in the
/// verb, and check the data files named on the command line (TIS-001).
pub(crate) fn enforce(
    verb: &str,
    rc_bound: bool,
    manifest: Option<&Path>,
    data: &[Option<&Path>],
) -> Result<(), CliError> {
    let g = gate(verb, rc_bound, manifest)?;
    activate(verb, g.as_ref());
    match &g {
        Some(g) => check_files(verb, g, &data.iter().flatten().copied().collect::<Vec<_>>()),
        None => Ok(()),
    }
}

/// The gate of the running verb, for loaders that open data files deep in a
/// verb (distill reads its prompts path from a config file).
static ACTIVE: Mutex<Option<(String, SealedGate)>> = Mutex::new(None);

/// Serialises tests that set [`ACTIVE`].
#[cfg(test)]
pub(crate) static TEST_ACTIVE_LOCK: Mutex<()> = Mutex::new(());

/// Make `gate` the one that [`check_ingress`] applies for the rest of the run.
pub(crate) fn activate(verb: &str, gate: Option<&SealedGate>) {
    if let Ok(mut active) = ACTIVE.lock() {
        *active = gate.map(|g| (verb.to_string(), g.clone()));
    }
}

/// Refuse `path` if any of its samples is a sealed item (TIS-001). A no-op
/// when no manifest was given.
pub(crate) fn check_ingress(path: &Path) -> Result<(), CliError> {
    let active = ACTIVE.lock().ok().and_then(|a| a.clone());
    match active {
        Some((verb, gate)) => check_files(&verb, &gate, &[path]),
        None => Ok(()),
    }
}

/// Refuse any sample in `paths` that is a sealed item or hunk (TIS-001). A
/// directory cannot be checked line by line, so it is refused.
pub(crate) fn check_files(verb: &str, gate: &SealedGate, paths: &[&Path]) -> Result<(), CliError> {
    if !gate.matching {
        return Ok(());
    }
    let mut hits = Vec::new();
    let mut rows = 0usize;
    for path in paths {
        if path.is_dir() {
            return Err(refusal(
                verb,
                "TIS-003",
                format!(
                    "{} is a directory; sealed matching reads data files line by line, so \
                     name the data file",
                    path.display()
                ),
            ));
        }
        let bytes = fs::read(path).map_err(|e| {
            refusal(
                verb,
                "TIS-001",
                format!("cannot read {}: {e}", path.display()),
            )
        })?;
        let text = String::from_utf8_lossy(&bytes);
        for (n, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            rows += 1;
            if let Some(id) = sealed_hit(line, &gate.digests) {
                hits.push(format!("{} row {} = {id}", path.display(), n + 1));
            }
        }
    }
    if !hits.is_empty() {
        let shown = hits.iter().take(10).cloned().collect::<Vec<_>>().join("; ");
        return Err(refusal(
            verb,
            "TIS-001",
            format!(
                "{} sample(s) are sealed eval items: {shown}. Refusing before step 0",
                hits.len()
            ),
        ));
    }
    eprintln!("[TIS-001] {rows} samples checked, 0 exact sealed hits");
    Ok(())
}

/// The sealed id a data line carries, if any: the whole line, or any JSON
/// string in it (with or without one trailing newline), hashed.
fn sealed_hit(line: &str, digests: &HashMap<String, String>) -> Option<String> {
    let hit = |s: &str| {
        let bare = s.strip_suffix('\n').unwrap_or(s);
        [bare.to_string(), format!("{bare}\n")]
            .iter()
            .find_map(|c| digests.get(&format!("{:x}", Sha256::digest(c.as_bytes()))))
            .cloned()
    };
    if let Some(id) = hit(line.trim()) {
        return Some(id);
    }
    let json: serde_json::Value = serde_json::from_str(line).ok()?;
    let mut stack = vec![&json];
    while let Some(v) = stack.pop() {
        match v {
            serde_json::Value::String(s) => {
                if let Some(id) = hit(s) {
                    return Some(id);
                }
            }
            serde_json::Value::Array(a) => stack.extend(a),
            serde_json::Value::Object(o) => stack.extend(o.values()),
            _ => {}
        }
    }
    None
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
        // A well-formed line whose hunk list holds a digest that is not a
        // sha256 is refused as well, not loaded without that hunk.
        for hunks in [
            format!("{SHA_B},not-a-sha256"),
            format!("{SHA_B} {}", "g".repeat(64)),
        ] {
            let p = write(
                &dir,
                "bad-hunk.txt",
                &format!("G-1 {SHA_A}\nG-2 {SHA_A} {hunks}\n"),
            );
            let err = gate("finetune", true, Some(&p))
                .expect_err(&hunks)
                .to_string();
            assert!(err.contains("line 2"), "{hunks}: {err}");
        }
    }

    #[test]
    fn manifest_with_items_passes_and_turns_matching_on() {
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
        assert!(g.matching, "finetune ingress is checked (TIS-001)");
    }

    #[test]
    fn non_rc_run_without_manifest_is_unchanged() {
        assert!(gate("finetune", false, None).expect("non-rc run").is_none());
    }

    fn sha(s: &str) -> String {
        format!("{:x}", Sha256::digest(s.as_bytes()))
    }

    const SEALED_2: &str = "def is_prime(n): return n > 1 and all(n % d for d in range(2, n))";
    const HUNK_3: &str = "-    return a\n+    return b";

    /// Three sealed items; item 3 is sealed by a hunk digest as well.
    fn sealed_manifest(dir: &tempfile::TempDir) -> std::path::PathBuf {
        let body = format!(
            "review-corpus-v1 test-manifest\nS-1 {}\nS-2 {}\nS-3 {} {}\n",
            sha("sealed one"),
            sha(SEALED_2),
            sha("sealed three"),
            sha(HUNK_3)
        );
        write(dir, "manifest.txt", &body)
    }

    fn corpus(insert_at: Option<(usize, String)>) -> String {
        let mut rows: Vec<String> = (0..20)
            .map(|i| serde_json::json!({"instruction": format!("task {i}"), "response": format!("answer {i}")}).to_string())
            .collect();
        if let Some((at, row)) = insert_at {
            rows.insert(at, row);
        }
        rows.join("\n") + "\n"
    }

    /// FALSIFY-TIS-001: a finetune corpus carrying sealed item S-2 at row 11
    /// is refused before step 0, naming the id and the row.
    #[test]
    fn falsify_tis_001_sealed_item_in_corpus_is_refused() {
        let _l = TEST_ACTIVE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().expect("tempdir");
        let m = sealed_manifest(&dir);
        let row = serde_json::json!({"instruction": "review", "response": SEALED_2}).to_string();
        let data = write(&dir, "train.jsonl", &corpus(Some((10, row))));
        let err = enforce("finetune", true, Some(&m), &[Some(&data)])
            .expect_err("sealed item must be refused")
            .to_string();
        activate("finetune", None);
        assert!(
            err.contains("S-2") && err.contains("row 11") && err.contains("TIS-001"),
            "{err}"
        );

        // A hunk digest seals too, and a raw (non-JSON) line is matched whole.
        let hunk = serde_json::json!({"text": format!("{HUNK_3}\n")}).to_string();
        let data = write(
            &dir,
            "hunk.jsonl",
            &format!("{}\n{hunk}\nsealed one\n", corpus(None)),
        );
        let err = enforce("distill", false, Some(&m), &[None, Some(&data)])
            .expect_err("hunk must be refused")
            .to_string();
        activate("distill", None);
        assert!(
            err.contains("S-3") && err.contains("S-1") && err.contains("2 sample"),
            "{err}"
        );
    }

    /// FALSIFY-TIS-004: the same corpus without the sealed row passes.
    #[test]
    fn falsify_tis_004_clean_corpus_passes() {
        let _l = TEST_ACTIVE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().expect("tempdir");
        let m = sealed_manifest(&dir);
        let data = write(&dir, "train.jsonl", &corpus(None));
        let r = enforce("finetune", true, Some(&m), &[Some(&data)]);
        activate("finetune", None);
        r.expect("clean corpus passes");
    }

    /// FALSIFY-TIS-003: loaders deep in a verb are checked, pretrain is
    /// refused by name when rc-bound, and a directory is not waved through.
    #[test]
    fn falsify_tis_003_every_ingress_is_covered() {
        let _l = TEST_ACTIVE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().expect("tempdir");
        let m = sealed_manifest(&dir);
        enforce("distill", false, Some(&m), &[]).expect("no data on the command line");
        let prompts = write(
            &dir,
            "prompts.jsonl",
            &format!("{}\n", serde_json::json!({"prompt": SEALED_2})),
        );
        let err = check_ingress(&prompts);
        let dir_err = enforce("distill", false, Some(&m), &[Some(dir.path())]);
        activate("distill", None);
        assert!(err
            .expect_err("config-named prompts are checked")
            .to_string()
            .contains("S-2"));
        let dir_err = dir_err.expect_err("directory").to_string();
        assert!(
            dir_err.contains("is a directory") && dir_err.contains("TIS-003"),
            "{dir_err}"
        );
        assert!(check_ingress(&prompts).is_ok(), "no active gate, no check");

        let err = gate("pretrain", true, Some(&m))
            .expect_err("rc pretrain")
            .to_string();
        assert!(err.contains("TIS-003") && err.contains("u32"), "{err}");
        let g = gate("pretrain", false, Some(&m))
            .expect("non-rc pretrain runs")
            .expect("gate");
        assert!(!g.matching, "pretrain makes no sealed-hits claim");
    }
}
