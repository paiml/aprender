//! FALSIFY-EXT-001 (EXT-001 row EXT-02, #4384): under nextest, no test can reach the real HOME.
//!
//! entrenar's global store is `dirs::home_dir()/.entrenar/experiments.db` (`PretrainTracker::open`),
//! and pacha and apr-cli resolve theirs the same way. `.config/nextest.toml` runs
//! `scripts/nextest/isolate_home.sh` as a setup script for every profile. That script points HOME at
//! a fresh temp dir and drops a marker there naming the real home. The first test fails if the script did
//! not run, if a profile stops binding it, or if the dir it made is the real home or sits inside it.
//! It checks the live env, so it needs nextest. The second checks the binding and the script directly,
//! so plain `cargo test` (what the contract gate runs) exercises the property too.

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

/// Parses `KEY=VALUE` lines the way nextest reads `$NEXTEST_ENV`.
fn read_env_file(path: &Path) -> std::collections::HashMap<String, String> {
    std::fs::read_to_string(path)
        .expect("FALSIFY-EXT-001: read NEXTEST_ENV")
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// The same property without nextest, so a plain `cargo test` cannot pass it vacuously: the config
/// binds the setup script to every test, and the script, run against a stand-in real home, exports
/// an isolated HOME and keeps the toolchain on the real one.
#[test]
fn falsify_ext_001_setup_script_is_bound_to_every_test_and_isolates_home() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cfg_path = root.join(".config/nextest.toml");
    let cfg: toml::Value = std::fs::read_to_string(&cfg_path)
        .expect("FALSIFY-EXT-001: read .config/nextest.toml")
        .parse()
        .expect("FALSIFY-EXT-001: parse .config/nextest.toml");

    let experimental = cfg.get("experimental").and_then(toml::Value::as_array);
    assert!(
        experimental.is_some_and(|a| a.iter().any(|v| v.as_str() == Some("setup-scripts"))),
        "FALSIFY-EXT-001: nextest.toml does not enable experimental setup-scripts"
    );
    let command = cfg
        .get("scripts")
        .and_then(|s| s.get("setup"))
        .and_then(|s| s.get("ext-isolate-home"))
        .and_then(|s| s.get("command"))
        .and_then(toml::Value::as_str)
        .expect("FALSIFY-EXT-001: [scripts.setup.ext-isolate-home] command is missing");
    let names_script = |v: &toml::Value| match v {
        toml::Value::String(s) => s == "ext-isolate-home",
        toml::Value::Array(a) => a.iter().any(|x| x.as_str() == Some("ext-isolate-home")),
        _ => false,
    };
    let bound = cfg
        .get("profile")
        .and_then(|p| p.get("default"))
        .and_then(|d| d.get("scripts"))
        .and_then(toml::Value::as_array)
        .is_some_and(|rules| {
            rules.iter().any(|r| {
                r.get("filter").and_then(toml::Value::as_str) == Some("all()")
                    && r.get("setup").is_some_and(names_script)
            })
        });
    assert!(
        bound,
        "FALSIFY-EXT-001: [[profile.default.scripts]] does not bind ext-isolate-home to all() \
         (profile.ci inherits default's scripts, so default is the one binding)"
    );

    let script = command.split_whitespace().last().expect("FALSIFY-EXT-001: empty command");
    let real = tempfile::tempdir().expect("stand-in real home");
    let tmp = tempfile::tempdir().expect("TMPDIR for the script");
    let env_file = tmp.path().join("nextest-env");
    std::fs::write(&env_file, "").expect("create NEXTEST_ENV");
    let status = std::process::Command::new("bash")
        .arg(root.join(script))
        .env("NEXTEST_ENV", &env_file)
        .env("HOME", real.path())
        .env("TMPDIR", tmp.path())
        .env_remove("CARGO_HOME")
        .env_remove("RUSTUP_HOME")
        .status()
        .expect("FALSIFY-EXT-001: run the setup script");
    assert!(status.success(), "FALSIFY-EXT-001: {script} exited {status}");

    let env = read_env_file(&env_file);
    let get = |k: &str| {
        PathBuf::from(
            env.get(k).unwrap_or_else(|| panic!("FALSIFY-EXT-001: {script} does not export {k}")),
        )
    };
    let home = get("HOME");
    assert!(
        !home.starts_with(real.path()) && home.is_dir(),
        "FALSIFY-EXT-001: exported HOME {} is the real home or missing",
        home.display()
    );
    let marker = std::fs::read_to_string(home.join(".ext-001-isolated"))
        .expect("FALSIFY-EXT-001: the setup script wrote no .ext-001-isolated marker");
    assert_eq!(
        Path::new(marker.trim()),
        real.path(),
        "FALSIFY-EXT-001: marker names the wrong home"
    );
    assert_eq!(get("EXT001_REAL_HOME"), real.path());
    for (k, sub) in [
        ("ENTRENAR_HOME", ".entrenar"),
        ("APR_HOME", ".apr"),
        ("XDG_CONFIG_HOME", ".config"),
        ("XDG_CACHE_HOME", ".cache"),
        ("XDG_DATA_HOME", ".local/share"),
        ("XDG_STATE_HOME", ".local/state"),
    ] {
        assert_eq!(get(k), home.join(sub), "FALSIFY-EXT-001: {k}");
    }
    assert_eq!(get("CARGO_HOME"), real.path().join(".cargo"), "FALSIFY-EXT-001: CARGO_HOME moved");
    assert_eq!(
        get("RUSTUP_HOME"),
        real.path().join(".rustup"),
        "FALSIFY-EXT-001: RUSTUP_HOME moved"
    );
}
