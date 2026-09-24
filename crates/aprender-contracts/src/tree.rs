//! Workspace files read by tests at RUN time: skip by name out of tree, FAIL in tree (#4175).
//!
//! Many tests read files that live outside their crate (`contracts/…`, a sibling crate's
//! fixtures). A published `.crate` compiles those tests but carries no workspace around it, so a
//! bare `expect` PANICS there (#4129, #4149). The rule is two-sided:
//!
//! * out of tree (an unpacked `.crate`): the test prints `SKIP <test>: out of tree …` at column 0
//!   on stderr and returns;
//! * in tree (the aprender checkout): a missing or unreadable file FAILS the test. Deleting
//!   `contracts/` from a checkout must turn these tests red, never make them skip.
//!
//! So "in tree" is decided by the WORKSPACE, never by the file being looked for (a
//! `contracts/.is_dir()` test skips exactly when it must fail). The manifest two levels up
//! declares `[workspace]` AND its `crates/<this dir name>` is this crate's own directory. The
//! second half matters: the packaged-tarball gate unpacks every crate into `<ws>/pkgs/<name>-<ver>/`
//! under a generated `[workspace]` manifest, and that must read as out of tree.
//!
//! This is the ONE copy. Every crate calls it through the macros, which capture the CALLER's
//! `CARGO_MANIFEST_DIR` (a plain fn here would see aprender-contracts' own directory). A crate
//! that uses it needs `aprender-contracts` (or the `provable-contracts` alias) as a
//! `{ workspace = true }` dependency: a path-only dev-dep is stripped by `cargo package`.

use std::path::{Path, PathBuf};

/// The aprender workspace root when `manifest_dir` is a member crate of it under `crates/`, else
/// `None`.
pub fn workspace_root_of(manifest_dir: &Path) -> Option<PathBuf> {
    let root = manifest_dir.join("../..");
    let text = std::fs::read_to_string(root.join("Cargo.toml")).ok()?;
    let declares = text
        .lines()
        .any(|l| l.split('#').next().unwrap_or("").trim() == "[workspace]");
    if !declares {
        return None;
    }
    let me = manifest_dir.canonicalize().ok()?;
    let listed = root
        .join("crates")
        .join(me.file_name()?)
        .canonicalize()
        .ok()?;
    if listed != me {
        return None;
    }
    root.canonicalize().ok()
}

/// `root/rel` in tree, where it MUST exist (panics, naming the path, when it does not). Out of
/// tree: `None`, after printing which test skipped and why. Call it through
/// [`workspace_path_or_skip!`](crate::workspace_path_or_skip).
pub fn workspace_path_or_skip_at(test: &str, manifest_dir: &Path, rel: &str) -> Option<PathBuf> {
    let Some(root) = workspace_root_of(manifest_dir) else {
        eprintln!(
            "SKIP {test}: out of tree ({} is not a member of the aprender workspace) - {rel} \
             lives in the workspace, which a published crate does not carry",
            manifest_dir.display()
        );
        return None;
    };
    let path = root.join(rel);
    assert!(
        path.exists(),
        "{test}: in tree, {} must exist (only an out-of-tree build may skip)",
        path.display()
    );
    Some(path)
}

/// The contents of `root/rel`, on the same two-sided rule as [`workspace_path_or_skip_at`]. Call
/// it through [`workspace_file_or_skip!`](crate::workspace_file_or_skip).
pub fn workspace_file_or_skip_at(test: &str, manifest_dir: &Path, rel: &str) -> Option<String> {
    let path = workspace_path_or_skip_at(test, manifest_dir, rel)?;
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{test}: in tree, {} must be readable: {e}", path.display()));
    Some(text)
}

/// `workspace_path_or_skip!(test, rel) -> Option<PathBuf>`: the workspace file `rel` in tree (it
/// must exist), `None` plus a named `SKIP` out of tree. See [`crate::tree`].
#[macro_export]
macro_rules! workspace_path_or_skip {
    ($test:expr, $rel:expr) => {
        $crate::tree::workspace_path_or_skip_at(
            $test,
            ::std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
            $rel,
        )
    };
}

/// `workspace_file_or_skip!(test, rel) -> Option<String>`: the contents of the workspace file
/// `rel` in tree (it must exist and be readable), `None` plus a named `SKIP` out of tree. See
/// [`crate::tree`].
#[macro_export]
macro_rules! workspace_file_or_skip {
    ($test:expr, $rel:expr) => {
        $crate::tree::workspace_file_or_skip_at(
            $test,
            ::std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
            $rel,
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// A fake workspace: `<tmp>/Cargo.toml` (with `manifest` as its text, or none), one crate at
    /// `<tmp>/<crate_rel>`, and `contracts/x.yaml` when `with_contracts`.
    fn fixture(
        manifest: Option<&str>,
        crate_rel: &str,
        with_contracts: bool,
    ) -> (tempfile::TempDir, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        if let Some(text) = manifest {
            fs::write(tmp.path().join("Cargo.toml"), text).unwrap();
        }
        let krate = tmp.path().join(crate_rel);
        fs::create_dir_all(&krate).unwrap();
        fs::write(krate.join("Cargo.toml"), "[package]\nname = \"foo\"\n").unwrap();
        if with_contracts {
            fs::create_dir_all(tmp.path().join("contracts")).unwrap();
            fs::write(tmp.path().join("contracts/x.yaml"), "k: v\n").unwrap();
        }
        (tmp, krate)
    }

    const WS: &str = "[workspace]\nmembers = [\"crates/*\"]\n";

    fn outcome(manifest: Option<&str>, crate_rel: &str, with_contracts: bool) -> &'static str {
        let (_tmp, krate) = fixture(manifest, crate_rel, with_contracts);
        match std::panic::catch_unwind(|| {
            workspace_file_or_skip_at("case", &krate, "contracts/x.yaml")
        }) {
            Ok(Some(_)) => "run",
            Ok(None) => "skip",
            Err(_) => "FAIL",
        }
    }

    /// The case table. Each row names the checkout shape and what a test reading
    /// `contracts/x.yaml` must do there.
    #[test]
    fn case_table() {
        let rows: &[(&str, Option<&str>, &str, bool, &str)] = &[
            (
                "in tree, contracts/ present",
                Some(WS),
                "crates/foo",
                true,
                "run",
            ),
            (
                "in tree, contracts/ absent (a broken checkout)",
                Some(WS),
                "crates/foo",
                false,
                "FAIL",
            ),
            (
                "in tree, [workspace] with a trailing comment",
                Some("[workspace] # root\n"),
                "crates/foo",
                false,
                "FAIL",
            ),
            (
                "tarball: no manifest two levels up (the registry)",
                None,
                "src/foo-1.2.3",
                true,
                "skip",
            ),
            (
                "tarball: a parent manifest without [workspace]",
                Some("[package]\nname = \"p\"\n"),
                "crates/foo",
                true,
                "skip",
            ),
            (
                "tarball gate: a generated [workspace], crate under pkgs/",
                Some(WS),
                "pkgs/foo-1.2.3",
                true,
                "skip",
            ),
            (
                "[workspace] only in a comment",
                Some("# [workspace]\n[package]\n"),
                "crates/foo",
                true,
                "skip",
            ),
        ];
        let mut bad = Vec::new();
        for (name, manifest, crate_rel, with_contracts, want) in rows {
            let got = outcome(*manifest, crate_rel, *with_contracts);
            if got != *want {
                bad.push(format!("{name}: want {want}, got {got}"));
            }
        }
        assert!(
            bad.is_empty(),
            "case table rows failed:\n{}",
            bad.join("\n")
        );
    }

    /// The macro captures THIS crate's manifest dir. In the checkout that is in tree and this
    /// file exists; in the published tarball it is out of tree and skips by name.
    #[test]
    fn the_macro_reads_the_callers_manifest_dir() {
        if let Some(text) = crate::workspace_file_or_skip!(
            "the_macro_reads_the_callers_manifest_dir",
            "crates/aprender-contracts/Cargo.toml"
        ) {
            assert!(text.contains("name = \"aprender-contracts\""));
        }
    }
}
