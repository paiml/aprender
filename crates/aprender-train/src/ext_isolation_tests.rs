//! FALSIFY-EXT-001 (EXT-001 row EXT-02, #4384): under nextest, no test can reach the real HOME.
//!
//! entrenar's global store is `dirs::home_dir()/.entrenar/experiments.db` (`PretrainTracker::open`),
//! and pacha and apr-cli resolve theirs the same way. `.config/nextest.toml` runs
//! `scripts/nextest/isolate_home.sh` as a setup script for both profiles. That script points HOME at
//! a fresh temp dir and drops a marker there naming the real home. This test fails if the script did
//! not run, if a profile stops binding it, or if the dir it made is the real home or sits inside it.

use std::path::{Path, PathBuf};

fn env_path(var: &str) -> PathBuf {
    std::env::var_os(var)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("FALSIFY-EXT-001: {var} is unset under nextest"))
}

#[test]
fn falsify_ext_001_nextest_runs_every_test_with_an_isolated_home() {
    if std::env::var_os("NEXTEST_RUN_ID").is_none() {
        // The isolation is a nextest profile setting; plain `cargo test` has no setup scripts.
        eprintln!("FALSIFY-EXT-001: not under nextest, nothing to check");
        return;
    }
    let home = env_path("HOME");
    let marker = home.join(".ext-001-isolated");
    let real = std::fs::read_to_string(&marker).unwrap_or_else(|e| {
        panic!(
            "FALSIFY-EXT-001: HOME={} has no {}: the nextest setup script \
             [scripts.setup.ext-isolate-home] did not run for this profile ({e})",
            home.display(),
            marker.display()
        )
    });
    let real = PathBuf::from(real.trim());
    assert!(
        !real.as_os_str().is_empty() && !home.starts_with(&real),
        "FALSIFY-EXT-001: the isolated HOME {} is the real home {} or inside it",
        home.display(),
        real.display()
    );
    assert_eq!(env_path("EXT001_REAL_HOME"), real, "FALSIFY-EXT-001: marker and env disagree");

    // The resolver the writers call, not only the variable.
    assert_eq!(
        dirs::home_dir().as_deref(),
        Some(home.as_path()),
        "FALSIFY-EXT-001: dirs::home_dir() (entrenar's global store root) escapes the isolated HOME"
    );
    for (var, sub) in [("ENTRENAR_HOME", ".entrenar"), ("APR_HOME", ".apr")] {
        assert_eq!(env_path(var), home.join(sub), "FALSIFY-EXT-001: {var}");
    }
    let under_home = |p: Option<PathBuf>, what: &str| {
        let p = p.unwrap_or_else(|| panic!("FALSIFY-EXT-001: {what} is None"));
        assert!(p.starts_with(&home), "FALSIFY-EXT-001: {what}={} escapes HOME", p.display());
    };
    under_home(dirs::config_dir(), "dirs::config_dir()");
    under_home(dirs::cache_dir(), "dirs::cache_dir()");
    under_home(dirs::data_dir(), "dirs::data_dir()");
    under_home(dirs::state_dir(), "dirs::state_dir()");

    // The toolchain stays reachable for tests that shell out to cargo.
    for var in ["CARGO_HOME", "RUSTUP_HOME"] {
        let p = env_path(var);
        assert!(!p.starts_with(&home), "FALSIFY-EXT-001: {var} was moved into the throwaway HOME");
        assert!(
            Path::new(&p).is_absolute(),
            "FALSIFY-EXT-001: {var}={} is not absolute",
            p.display()
        );
    }
}
