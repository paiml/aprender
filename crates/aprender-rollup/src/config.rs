//! Per-repo rollup settings (PMAT-5737 G1). One table per repo; the default is aprender's.

use serde::Deserialize;

/// Settings the pure core reads instead of constants, so one batcher serves many repos.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// `owner/name`, passed as `--repo` to every `gh` call; empty means the current checkout.
    pub repo: String,
    /// Branch the rollup is cut from.
    pub base: String,
    /// Largest PR (additions + deletions) that may join.
    pub max_lines: u32,
    /// Most constituents in one rollup.
    pub max_entries: usize,
    /// Label a PR owner sets to opt in.
    pub label: String,
    /// Paths a rollup never carries: an entry ending in `/` is a prefix, any other is exact.
    pub forbidden: Vec<String>,
    /// The repo's own local gate, run on the built rollup.
    pub local_gate: String,
}

impl Default for Config {
    /// aprender: b = 3 until p_red is measured; no CI, contract or lockfile changes.
    fn default() -> Self {
        Self {
            repo: String::new(),
            base: "main".to_string(),
            max_lines: 200,
            max_entries: 3,
            label: "rollup-ok".to_string(),
            forbidden: [".github/workflows/", "contracts/", "Cargo.lock"]
                .map(String::from)
                .to_vec(),
            local_gate: "make gate".to_string(),
        }
    }
}

impl Config {
    /// True when `path` is one the rollup never carries.
    #[must_use]
    pub fn is_forbidden(&self, path: &str) -> bool {
        self.forbidden.iter().any(|f| {
            if f.ends_with('/') {
                path.starts_with(f.as_str())
            } else {
                path == f
            }
        })
    }

    /// `gh` arguments that pin the repo, empty for the current checkout.
    #[must_use]
    pub fn repo_args(&self) -> Vec<String> {
        if self.repo.is_empty() {
            Vec::new()
        } else {
            vec!["--repo".to_string(), self.repo.clone()]
        }
    }

    /// The table for `name` out of a TOML file of `[repo.<name>]` tables; keys left out keep
    /// the defaults, unknown keys are an error, an absent table is an error.
    ///
    /// # Errors
    /// Malformed TOML, an unknown key, or no `[repo.<name>]` table.
    pub fn from_toml(text: &str, name: &str) -> Result<Self, String> {
        let mut doc: toml::Table = text.parse().map_err(|e| format!("config: {e}"))?;
        let table = doc
            .remove("repo")
            .and_then(|r| match r {
                toml::Value::Table(mut t) => t.remove(name),
                _ => None,
            })
            .ok_or_else(|| format!("config: no [repo.{name}] table"))?;
        table
            .try_into()
            .map_err(|e| format!("config [repo.{name}]: {e}"))
    }
}

#[cfg(test)]
mod tests;
