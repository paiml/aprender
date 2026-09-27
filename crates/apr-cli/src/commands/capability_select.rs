//! `apr capability --select <consumer>` — which Qwen3.5 size to run, and the receipts (#3558).
//!
//! THE ANSWER IS NOT COMPUTED HERE. It is embedded from
//! `crates/apr-cli/contracts/qwen35-size-selection.json`, the packaged mirror of the
//! newest `evidence/dogfood/qwen35-selection/<ver>.json`, which
//! `scripts/qwen35_size_select.py` derives from committed ladder receipts and
//! re-derives byte-identical (`--check`). A second selection rule in Rust would be a
//! second answer to the same question, and the two would drift.
//!
//! The test below asserts the mirror equals the newest derived table, so a table
//! re-derived for a new release without refreshing the mirror is RED, not stale.

use crate::error::{CliError, Result};

const SELECTION: &str = include_str!("../../contracts/qwen35-size-selection.json");

fn table() -> Result<serde_json::Value> {
    serde_json::from_str(SELECTION).map_err(|e| {
        CliError::ValidationFailed(format!(
            "the embedded Qwen3.5 selection table did not parse: {e} \
             (crates/apr-cli/contracts/qwen35-size-selection.json)"
        ))
    })
}

fn keys(v: Option<&serde_json::Value>) -> Vec<String> {
    v.and_then(|x| x.as_object())
        .map(|o| o.keys().cloned().collect())
        .unwrap_or_default()
}

/// The answer for one (host, consumer), or why there is none.
fn lookup<'a>(
    t: &'a serde_json::Value,
    consumer: &str,
    host: &str,
) -> Result<(&'a serde_json::Value, &'a serde_json::Value)> {
    let hosts = t.get("hosts");
    let h = hosts.and_then(|x| x.get(host)).ok_or_else(|| {
        CliError::InvalidInput(format!(
            "no selection for host `{host}`; the table covers: {}",
            keys(hosts).join(", ")
        ))
    })?;
    let consumers = h.get("consumers");
    let c = consumers.and_then(|x| x.get(consumer)).ok_or_else(|| {
        CliError::InvalidInput(format!(
            "no selection for consumer `{consumer}`; the table covers: {}",
            keys(consumers).join(", ")
        ))
    })?;
    Ok((h, c))
}

fn s<'a>(v: &'a serde_json::Value, path: &[&str]) -> &'a str {
    path.iter()
        .try_fold(v, |acc, k| acc.get(k))
        .and_then(|x| x.as_str())
        .unwrap_or("")
}

/// Print the selected size for `consumer` on `host`. Nothing admissible is an error
/// exit: a consumer scripting this must not read an empty answer as "use anything".
pub fn run(consumer: &str, host: &str, json: bool) -> Result<()> {
    let t = table()?;
    let (h, c) = lookup(&t, consumer, host)?;
    let selected = c.get("selected").and_then(|x| x.as_str());
    let cand = c
        .get("candidates")
        .and_then(|x| x.as_array())
        .and_then(|a| {
            a.iter()
                .find(|r| r.get("rung").and_then(|x| x.as_str()) == selected)
        });

    if json {
        let out = serde_json::json!({
            "source": "evidence/dogfood/qwen35-selection",
            "ladder_version": t.get("ladder_version"),
            "host": host,
            "consumer": consumer,
            "max_prompt_tokens": c.get("max_prompt_tokens"),
            "selected": selected,
            "admissible": c.get("admissible"),
            "capability_receipt": h.get("receipt"),
            "candidate": cand,
            "context_basis_note": t.get("context_basis_note"),
        });
        let text = serde_json::to_string_pretty(&out)
            .map_err(|e| CliError::ValidationFailed(format!("selection did not serialize: {e}")))?;
        println!("{text}");
    } else {
        println!(
            "{host}/{consumer} (budget {} tokens, ladder {}): {}",
            c.get("max_prompt_tokens")
                .unwrap_or(&serde_json::Value::Null),
            s(&t, &["ladder_version"]),
            selected.unwrap_or("NOTHING ADMISSIBLE")
        );
        if let Some(r) = cand {
            println!(
                "  capability: {} ({})",
                s(h, &["receipt"]),
                s(r, &["capability", "why"])
            );
            println!(
                "  context:    {} ({} tokens, {})",
                s(r, &["context", "receipt"]),
                r.pointer("/context/prompt_tokens")
                    .unwrap_or(&serde_json::Value::Null),
                s(r, &["context", "basis"])
            );
        }
        for r in c
            .get("candidates")
            .and_then(|x| x.as_array())
            .into_iter()
            .flatten()
        {
            if r.get("admissible").and_then(|x| x.as_bool()) == Some(true) {
                continue;
            }
            let why = if r.pointer("/capability/ok").and_then(|x| x.as_bool()) == Some(true) {
                s(r, &["context", "why"])
            } else {
                s(r, &["capability", "why"])
            };
            println!("  not admissible: {} — {why}", s(r, &["rung"]));
        }
        println!("  note: {}", s(&t, &["context_basis_note"]));
    }
    if selected.is_none() {
        return Err(CliError::ValidationFailed(format!(
            "no Qwen3.5 size is admissible for {consumer} on {host}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn the_embedded_table_parses_and_answers_every_host_and_consumer() {
        let t = table().expect("embedded selection table parses");
        let hosts = t["hosts"].as_object().expect("hosts object");
        assert!(!hosts.is_empty(), "selection table has no hosts");
        for (host, h) in hosts {
            for consumer in h["consumers"].as_object().expect("consumers").keys() {
                lookup(&t, consumer, host).expect("every listed pair resolves");
            }
        }
    }

    #[test]
    fn an_unknown_consumer_or_host_is_refused_not_defaulted() {
        let t = table().expect("parses");
        assert!(lookup(&t, "no-such-consumer", "lambda").is_err());
        assert!(lookup(&t, "arbiter", "no-such-host").is_err());
    }

    /// The mirror must equal the NEWEST derived table. Refreshing the table for a new
    /// release without refreshing the mirror would ship last release's answer.
    #[test]
    fn the_mirror_is_the_newest_derived_table() {
        let dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../evidence/dogfood/qwen35-selection");
        let mut vs: Vec<(Vec<u64>, std::path::PathBuf)> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter_map(|p| {
                let stem = p.file_stem()?.to_str()?.to_string();
                let v: Option<Vec<u64>> = stem.split('.').map(|x| x.parse().ok()).collect();
                Some((v?, p))
            })
            .collect();
        vs.sort();
        let (_, newest) = vs.last().expect("at least one derived selection table");
        let want = std::fs::read_to_string(newest).expect("read newest table");
        assert!(
            want == SELECTION,
            "crates/apr-cli/contracts/qwen35-size-selection.json is not {} — copy it",
            newest.display()
        );
    }
}
