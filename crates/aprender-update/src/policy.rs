//! When the check may run, and whether this host may install.
//!
//! Pure functions over an environment lookup, so every rule is a table row.

/// Why the background check does not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Skip {
    /// `SOVEREIGN_DISABLE_UPDATE_CHECK` is set to a non-empty value other than `0`.
    Disabled,
    /// `CI` is set.
    Ci,
    /// stderr is not a terminal, so nobody would read the notice.
    NotTty,
    /// `--json` or `--quiet`/`-q` on the command line: output is for a machine.
    MachineOutput,
}

pub const DISABLE_ENV: &str = "SOVEREIGN_DISABLE_UPDATE_CHECK";
/// Overrides where the fleet arbiter's manifest lives.
pub const ARBITER_ENV: &str = "SOVEREIGN_ARBITER_MANIFEST";
/// The arbiter manifest, relative to `$HOME` (infra-8d, EPIC #4232). forjar
/// owns it; every writer runs as the user, so it is not under `/etc`.
pub const ARBITER_MANIFEST: &str = ".config/sovereign/arbiter-manifest.json";
/// Fleet hosts: report-only whether or not a manifest exists yet (infra-8d:
/// "no manifest means report-only, never auto-update"). Matched
/// case-insensitively as a substring of the hostname.
pub const FLEET_HOSTS: [&str; 3] = ["lambda-vector", "gx10", "yoga"];

fn set(env: &dyn Fn(&str) -> Option<String>, k: &str) -> bool {
    env(k).is_some_and(|v| !v.is_empty() && v != "0")
}

/// `Ok(())` when the check may run.
pub fn check_allowed(
    env: &dyn Fn(&str) -> Option<String>,
    stderr_is_tty: bool,
    args: &[String],
) -> Result<(), Skip> {
    if set(env, DISABLE_ENV) {
        return Err(Skip::Disabled);
    }
    if env("CI").is_some() {
        return Err(Skip::Ci);
    }
    if args
        .iter()
        .any(|a| a == "--json" || a == "--quiet" || a == "-q" || a.starts_with("--format=json"))
    {
        return Err(Skip::MachineOutput);
    }
    if !stderr_is_tty {
        return Err(Skip::NotTty);
    }
    Ok(())
}

/// Who owns installs on this host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Installs {
    /// The user: `update` may swap the binary.
    User,
    /// The fleet arbiter: report only, never self-install. Carries why.
    Arbiter(String),
}

/// Where the arbiter manifest is: `$SOVEREIGN_ARBITER_MANIFEST` if set and
/// non-empty, else `$HOME/.config/sovereign/arbiter-manifest.json`.
#[must_use]
pub fn arbiter_manifest(env: &dyn Fn(&str) -> Option<String>) -> Option<std::path::PathBuf> {
    env(ARBITER_ENV)
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
        .or_else(|| {
            env("HOME")
                .filter(|h| !h.is_empty())
                .map(|h| std::path::Path::new(&h).join(ARBITER_MANIFEST))
        })
}

/// Decide who owns installs. An existing arbiter manifest (the file IS the
/// marker) or a fleet hostname makes it the arbiter's.
#[must_use]
pub fn installs(hostname: &str, manifest: Option<&std::path::Path>) -> Installs {
    if let Some(m) = manifest {
        return Installs::Arbiter(format!("arbiter manifest {}", m.display()));
    }
    if hostname.trim().is_empty() {
        // Fail closed: an undetectable name could be a fleet host.
        return Installs::Arbiter("hostname unknown".into());
    }
    let h = hostname.to_ascii_lowercase();
    if let Some(f) = FLEET_HOSTS.iter().find(|f| h.contains(*f)) {
        return Installs::Arbiter(format!("fleet host {hostname} ({f})"));
    }
    Installs::User
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_of(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |k| {
            pairs
                .iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| (*v).to_string())
        }
    }
    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn check_allowed_case_table() {
        type Row = (
            &'static str,
            &'static [(&'static str, &'static str)],
            bool,
            &'static [&'static str],
            Result<(), Skip>,
        );
        let rows: [Row; 10] = [
            ("interactive, no env", &[], true, &["run"], Ok(())),
            (
                "disabled by env",
                &[(DISABLE_ENV, "1")],
                true,
                &[],
                Err(Skip::Disabled),
            ),
            (
                "disable=0 means enabled",
                &[(DISABLE_ENV, "0")],
                true,
                &[],
                Ok(()),
            ),
            (
                "disable empty means enabled",
                &[(DISABLE_ENV, "")],
                true,
                &[],
                Ok(()),
            ),
            (
                "CI set, even empty",
                &[("CI", "")],
                true,
                &[],
                Err(Skip::Ci),
            ),
            ("stderr not a tty", &[], false, &[], Err(Skip::NotTty)),
            (
                "--json",
                &[],
                true,
                &["list", "--json"],
                Err(Skip::MachineOutput),
            ),
            ("--quiet", &[], true, &["--quiet"], Err(Skip::MachineOutput)),
            ("-q", &[], true, &["-q"], Err(Skip::MachineOutput)),
            (
                "--format=json",
                &[],
                true,
                &["--format=json"],
                Err(Skip::MachineOutput),
            ),
        ];
        let mut bad = Vec::new();
        for (name, env, tty, a, want) in rows {
            let got = check_allowed(&env_of(env), tty, &args(a));
            if got != want {
                bad.push(format!("{name}: got {got:?}, want {want:?}"));
            }
        }
        assert!(
            bad.is_empty(),
            "check_allowed rows failed:\n{}",
            bad.join("\n")
        );
    }

    #[test]
    fn installs_case_table() {
        use std::path::{Path, PathBuf};
        assert_eq!(installs("laptop", None), Installs::User);
        for h in ["", "  "] {
            assert!(
                matches!(installs(h, None), Installs::Arbiter(_)),
                "unknown hostname {h:?} is report-only"
            );
        }
        assert!(matches!(
            installs("laptop", Some(Path::new("/h/m.json"))),
            Installs::Arbiter(_)
        ));
        for h in ["noah-Lambda-Vector", "gx10-5e1a", "YOGA"] {
            assert!(
                matches!(installs(h, None), Installs::Arbiter(_)),
                "{h} is a fleet host even with no manifest"
            );
        }
        assert_eq!(
            arbiter_manifest(&env_of(&[(ARBITER_ENV, "/m.json"), ("HOME", "/h")])),
            Some(PathBuf::from("/m.json")),
            "the env override wins"
        );
        assert_eq!(
            arbiter_manifest(&env_of(&[(ARBITER_ENV, ""), ("HOME", "/h")])),
            Some(PathBuf::from("/h/.config/sovereign/arbiter-manifest.json")),
            "an empty override falls back to HOME"
        );
        assert_eq!(arbiter_manifest(&env_of(&[])), None);
    }
}
