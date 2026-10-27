//! README.md rendering half of the `pv` README update (#6394, epic #2870 PVL-001), part 1 of 2.
//!
//! The generated parts of README.md are the `CONTRACT_COUNT` inline blocks (the walked count of contract
//! files, the one definition `pv census` and `scripts/readme_sync.sh` share) and the frontmatter
//! `contract_count:` line (the tracked `census.json` `n_files`, which lags the walk by design). This is the
//! Rust port of `scripts/readme_sync.sh`'s rewrite; the script stays until the fold.
//!
//! This part is the pure render and the contract walk, with their tests. The SHACL validation, the
//! check/write driver and the `pv-readme` binary land in the stacked PR (#6432).

use std::path::Path;

const START: &str = "<!-- CONTRACT_COUNT_START -->";
const END: &str = "<!-- CONTRACT_COUNT_END -->";

/// A reason the run did not pass, with the exit code it maps to.
#[derive(Debug, PartialEq, Eq)]
pub struct Refusal {
    pub code: u8,
    pub what: String,
}

fn refuse<T>(code: u8, what: impl Into<String>) -> Result<T, Refusal> {
    Err(Refusal {
        code,
        what: what.into(),
    })
}

/// How many `CONTRACT_COUNT` blocks `text` carries.
#[must_use]
pub fn block_count(text: &str) -> usize {
    text.matches(START).count()
}

/// The README with every `CONTRACT_COUNT` block body set to `walked` and every whole-line `contract_count: N`
/// set to `census`. Nothing else is touched. A block body is the text up to the next `<`, so a hand-edited
/// body is replaced just as a stale number is (the same rule as `readme_sync.sh`).
#[must_use]
pub fn render(text: &str, walked: u64, census: u64) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find(START) {
        let after = &rest[i + START.len()..];
        let body = after.find('<').unwrap_or(after.len());
        if after[body..].starts_with(END) {
            out.push_str(&rest[..i]);
            out.push_str(START);
            out.push_str(&walked.to_string());
            rest = &after[body..];
        } else {
            // an opening marker with no closing marker after its body: leave it, keep scanning past it
            out.push_str(&rest[..i + START.len()]);
            rest = after;
        }
    }
    out.push_str(rest);
    let lines: Vec<String> = out
        .split('\n')
        .map(|l| {
            let numeric = l
                .strip_prefix("contract_count: ")
                .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()));
            if numeric {
                format!("contract_count: {census}")
            } else {
                l.to_string()
            }
        })
        .collect();
    lines.join("\n")
}

/// `contract_files_only` of `readme_sync.sh`: a `*.yaml` under `contracts/` that is not in a retired directory,
/// not one of the four named non-contracts, and not a dotfile.
fn is_contract_file(rel: &[String]) -> bool {
    let Some(name) = rel.last() else { return false };
    if !name.ends_with(".yaml") || name.starts_with('.') {
        return false;
    }
    if matches!(
        name.as_str(),
        "binding.yaml" | "binding.yml" | "external-corpora.yaml" | "ontology.yaml"
    ) {
        return false;
    }
    !rel[..rel.len() - 1].iter().any(|d| {
        matches!(
            d.as_str(),
            "legacy" | "pipelines" | "publish-manifests" | "quarantine"
        )
    })
}

fn walk(dir: &Path, rel: &mut Vec<String>, n: &mut u64) -> std::io::Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.collect::<Result<_, _>>()?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for e in entries {
        let ft = e.file_type()?;
        rel.push(e.file_name().to_string_lossy().into_owned());
        if ft.is_dir() {
            walk(&e.path(), rel, n)?;
        } else if ft.is_file() && is_contract_file(rel) {
            *n += 1;
        }
        rel.pop();
    }
    Ok(())
}

/// The walked contract count: the same universe `scripts/readme_sync.sh` measures.
pub fn walked_count(contract_dir: &Path) -> Result<u64, Refusal> {
    let mut n = 0;
    walk(contract_dir, &mut Vec::new(), &mut n)
        .or_else(|e| refuse(2, format!("cannot walk {}: {e}", contract_dir.display())))?;
    if n == 0 {
        return refuse(
            2,
            format!(
                "{} holds 0 contract files: a broken measurement, not a README to render",
                contract_dir.display()
            ),
        );
    }
    Ok(n)
}

#[cfg(test)]
#[path = "readme_cmd_tests.rs"]
mod tests;
