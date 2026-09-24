//! Claude-Code-style update check for sovereign binaries (EPIC #4232, #4242).
//!
//! * [`startup`] — call first thing in `main`. Prints at most one stderr line
//!   from the cache, and refreshes a stale cache (24 h) on a detached thread.
//!   It never blocks, never fails the command, and is silent on every error.
//!   It is skipped by `SOVEREIGN_DISABLE_UPDATE_CHECK=1`, `CI`, a non-TTY
//!   stderr, or `--json`/`--quiet`.
//! * [`update_command`] — the body of `<bin> update [--check]`. It installs the
//!   NEWER of the latest release and the green nightly (#4189 manifest), by
//!   main ancestry, and never downgrades. It verifies the sha256 (fail
//!   closed), smoke-tests `--version`, and renames the new binary over the
//!   running one, keeping `<exe>.prev`. Fleet hosts report and do not install.
//!
//! ```no_run
//! const PRODUCT: sovereign_update::Product = sovereign_update::Product {
//!     bin: "pv",
//!     repo: "paiml/aprender",
//!     version: env!("CARGO_PKG_VERSION"),
//!     build_sha: None,
//!     release_asset: Some("{bin}-{tag}-{target}.tar.gz"),
//!     nightly: true,
//! };
//! sovereign_update::startup(&PRODUCT, &std::env::args().collect::<Vec<_>>());
//! ```

pub mod cache;
pub mod decide;
pub mod fetch;
pub mod install;
pub mod policy;

use decide::{Candidate, Current, Decision};
use fetch::Net;
use policy::Installs;
use std::io::IsTerminal;
use std::path::Path;
use std::time::Duration;

/// What a binary says about itself. All fields are compile-time constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Product {
    /// Executable name, e.g. `apr`.
    pub bin: &'static str,
    /// `owner/name` on GitHub.
    pub repo: &'static str,
    /// `env!("CARGO_PKG_VERSION")`.
    pub version: &'static str,
    /// Build commit, when the binary embeds one (#4219).
    pub build_sha: Option<&'static str>,
    /// Release asset name template over `{bin}` `{tag}` `{version}` `{target}`;
    /// `None` when releases carry no asset for this bin.
    pub release_asset: Option<&'static str>,
    /// Whether the repo's `nightly` release carries an #4189 manifest.
    pub nightly: bool,
}

/// Per request, for the background refresh child.
const TIMEOUT: Duration = Duration::from_secs(10);
/// The refresh child exits after this, whatever the network is doing.
const REFRESH_DEADLINE: Duration = Duration::from_secs(60);
/// Hidden `update` argument the startup check runs its refresh child with.
const REFRESH_ARG: &str = "--refresh-cache";
const UPDATE_TIMEOUT: Duration = Duration::from_secs(120);

/// The target triple this binary was built for, as release assets name it.
#[must_use]
pub fn host_target() -> String {
    let os = match std::env::consts::OS {
        "linux" if cfg!(target_env = "musl") => "unknown-linux-musl",
        "linux" => "unknown-linux-gnu",
        "macos" => "apple-darwin",
        "windows" => "pc-windows-msvc",
        other => other,
    };
    format!("{}-{os}", std::env::consts::ARCH)
}

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok()
}

/// This host's name: procfs, `/etc/hostname`, `$HOSTNAME`, then `uname -n`
/// (macOS has neither file). Empty if all fail, which [`policy::installs`]
/// treats as report-only.
fn hostname() -> String {
    let file = |p: &str| std::fs::read_to_string(p).ok();
    let uname = || {
        std::process::Command::new("uname")
            .arg("-n")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
    };
    file("/proc/sys/kernel/hostname")
        .or_else(|| file("/etc/hostname"))
        .or_else(|| env("HOSTNAME"))
        .or_else(uname)
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn this_host_installs() -> Installs {
    let manifest = policy::arbiter_manifest(&env);
    policy::installs(&hostname(), manifest.as_deref().filter(|m| m.exists()))
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn user_agent(p: &Product) -> String {
    format!("{}/{} sovereign-update", p.bin, p.version)
}

/// Fresh check against the network: the release, the nightly, and ancestry.
pub fn check(net: &dyn Net, p: &Product, target: &str) -> Result<Decision, String> {
    let release = fetch::latest_release(net, p, target)?;
    let nightly = fetch::nightly(net, p, target)?;
    let cur = Current {
        version: p.version.to_string(),
        build_sha: p.build_sha.map(str::to_string),
    };
    let anc = |a: &str, b: &str| fetch::ancestry(net, p.repo, a, b);
    Ok(decide::decide(
        &cur,
        release.as_ref(),
        nightly.as_ref(),
        &anc,
    ))
}

/// Run a check and record it. A failure keeps the previous `available`, so a
/// flaky network neither hides a known update nor retries more than daily.
pub fn refresh(net: &dyn Net, p: &Product, target: &str, path: &Path, now: u64) {
    let old = cache::load(path);
    let (available, error) = match check(net, p, target) {
        Ok(Decision::Available(c)) => (Some(c), None),
        Ok(Decision::UpToDate) => (None, None),
        Err(e) => (old.and_then(|c| c.available), Some(e)),
    };
    let c = cache::Cache {
        schema: cache::SCHEMA.into(),
        checked_at: now,
        current_version: p.version.into(),
        current_sha: p.build_sha.map(Into::into),
        available,
        error,
    };
    let _ = cache::store(path, &c);
}

/// Call at the top of `main`. Never blocks, never fails, never errors aloud.
pub fn startup(p: &Product, args: &[String]) {
    // `update` is unix-only (install.rs); never announce what it cannot do.
    if !cfg!(unix) {
        return;
    }
    if policy::check_allowed(&env, std::io::stderr().is_terminal(), args).is_err() {
        return;
    }
    let Some(path) = cache::path(&env, p.bin) else {
        return;
    };
    if let Some(line) = startup_with(p, args, &path, now(), &this_host_installs(), &spawn_refresh) {
        eprintln!("{line}");
    }
}

/// The startup check after the skip rules. It takes no [`Net`]: it cannot
/// block on the network by construction. It reads the cache, claims the day
/// and hands the refresh to `spawn`. Returns the notice to print, if any.
pub fn startup_with(
    p: &Product,
    args: &[String],
    path: &Path,
    now: u64,
    installs: &Installs,
    spawn: &dyn Fn(),
) -> Option<String> {
    if args.get(1).is_some_and(|a| a == "update") {
        return None;
    }
    let c = cache::load(path);
    let line = c.as_ref().and_then(|c| cache::notice(p, c, installs));
    // Claim the day before spawning, so concurrent runs start one refresh,
    // not one each. A refresh that dies leaves the claim: next try in 24 h.
    if cache::needs_refresh(p, c.as_ref(), now)
        && cache::store(path, &cache::claim(p, c.as_ref(), now)).is_ok()
    {
        spawn();
    }
    line
}

/// Run `<exe> update --refresh-cache` as a detached child. A thread would die
/// with a short command (`pv --version` exits long before GitHub answers); a
/// child outlives it. Its stdio is null, so it can never print into the user's
/// terminal, and a waiter thread reaps it if this process lives longer.
fn spawn_refresh() {
    use std::process::{Command, Stdio};
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let mut cmd = Command::new(exe);
    cmd.args(["update", REFRESH_ARG])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
    if let Ok(mut child) = cmd.spawn() {
        std::thread::spawn(move || child.wait());
    }
}

fn describe(c: &Candidate) -> String {
    let src = match c.source {
        decide::Source::Release => format!("release {}", c.git_ref),
        decide::Source::Nightly => {
            format!("nightly {}", c.git_ref.chars().take(12).collect::<String>())
        }
    };
    format!("{} ({src})", c.version)
}

/// Body of `<bin> update [--check]`, with every effect injected. Returns the
/// exit code and the lines to print (stdout).
pub fn update_with(
    net: &dyn Net,
    p: &Product,
    target: &str,
    installs: &Installs,
    exe: &Path,
    check_only: bool,
) -> (i32, Vec<String>) {
    let c = match check(net, p, target) {
        Ok(Decision::UpToDate) => {
            return (0, vec![format!("{} {} is up to date", p.bin, p.version)])
        }
        Ok(Decision::Available(c)) => c,
        Err(e) => return (1, vec![format!("{}: update check failed: {e}", p.bin)]),
    };
    let mut out = vec![format!(
        "{} {} -> {} available",
        p.bin,
        p.version,
        describe(&c)
    )];
    if check_only {
        return (0, out);
    }
    if let Installs::Arbiter(why) = installs {
        out.push(format!(
            "not installing: {why}; the fleet arbiter manages installs on this host"
        ));
        return (0, out);
    }
    match install::install(net, p.bin, &c, exe) {
        Ok(prev) => {
            out.push(format!(
                "updated {} -> {} (previous kept at {})",
                exe.display(),
                c.version,
                prev.display()
            ));
            (0, out)
        }
        Err(e) => {
            out.push(format!("{}: update refused: {e}", p.bin));
            (1, out)
        }
    }
}

/// `<bin> update [--check]` against the real network and the running exe.
#[must_use]
pub fn update_command(p: &Product, check_only: bool) -> i32 {
    if !cfg!(unix) {
        eprintln!(
            "{}: self-update is unix-only; download from https://github.com/{}/releases",
            p.bin, p.repo
        );
        return 1;
    }
    // Say what it is waiting on: DNS is not bounded by the request timeout.
    eprintln!("{}: checking https://github.com/{} ...", p.bin, p.repo);
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("{}: cannot locate the running executable: {e}", p.bin);
            return 1;
        }
    };
    let net = fetch::Http::new(UPDATE_TIMEOUT, &user_agent(p));
    let (code, lines) = update_with(
        &net,
        p,
        &host_target(),
        &this_host_installs(),
        &exe,
        check_only,
    );
    for l in lines {
        if code == 0 {
            println!("{l}");
        } else {
            eprintln!("{l}");
        }
    }
    // A successful check also refreshes the cache, so the startup notice agrees.
    if let Some(path) = cache::path(&env, p.bin) {
        refresh(&net, p, &host_target(), &path, now());
    }
    code
}

/// What `<bin> update <args>` asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateArgs {
    Install,
    Check,
    Help,
    /// The startup check's detached child: refresh the cache, print nothing.
    RefreshCache,
    Bad,
}

/// Parse the arguments after `update`.
#[must_use]
pub fn parse_update_args(args: &[String]) -> UpdateArgs {
    match args {
        [] => UpdateArgs::Install,
        [a] if a == "--check" => UpdateArgs::Check,
        [a] if a == "--help" || a == "-h" => UpdateArgs::Help,
        [a] if a == REFRESH_ARG => UpdateArgs::RefreshCache,
        _ => UpdateArgs::Bad,
    }
}

/// Entry for a binary that dispatches `update` before its own CLI parser:
/// `if args.get(1) == Some("update") { exit(update_main(&P, &args[2..])) }`.
/// Intercepting in the binary's own `main` means the executable replaced is
/// always the one named `p.bin`, never a host binary that embeds its library.
#[must_use]
pub fn update_main(p: &Product, args: &[String]) -> i32 {
    let usage = format!(
        "usage: {0} update [--check]\n  Install the newer of the latest release and the green nightly (sha256-verified).\n  --check  report only\n  Disable the startup check with {1}=1.",
        p.bin,
        policy::DISABLE_ENV
    );
    match parse_update_args(args) {
        UpdateArgs::Install => update_command(p, false),
        UpdateArgs::Check => update_command(p, true),
        UpdateArgs::Help => {
            println!("{usage}");
            0
        }
        UpdateArgs::RefreshCache => {
            // ureq's timeout does not bound DNS resolution; a hung resolver
            // must not leave one stuck child per day. Hard deadline.
            std::thread::spawn(|| {
                std::thread::sleep(REFRESH_DEADLINE);
                std::process::exit(0);
            });
            if let Some(path) = cache::path(&env, p.bin) {
                let net = fetch::Http::new(TIMEOUT, &user_agent(p));
                refresh(&net, p, &host_target(), &path, now());
            }
            0
        }
        UpdateArgs::Bad => {
            eprintln!("{usage}");
            2
        }
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to Result::unwrap
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct Fake(HashMap<String, Vec<u8>>, bool);
    impl Net for Fake {
        fn get(&self, url: &str) -> Result<Option<Vec<u8>>, String> {
            if self.1 {
                return Err("offline".into());
            }
            Ok(self.0.get(url).cloned())
        }
    }

    const P: Product = Product {
        bin: "pv",
        repo: "o/r",
        version: "0.69.0",
        build_sha: None,
        release_asset: Some("{bin}-{tag}-{target}.tar.gz"),
        nightly: true,
    };
    const T: &str = "x86_64-unknown-linux-gnu";
    const LATEST: &str = "https://api.github.com/repos/o/r/releases/latest";

    fn with_release(tag: &str) -> HashMap<String, Vec<u8>> {
        let n = format!("pv-{tag}-{T}.tar.gz");
        let body = serde_json::json!({"tag_name": tag, "assets": [{"name": n, "browser_download_url": format!("https://dl/{n}")}]});
        HashMap::from([(LATEST.to_string(), serde_json::to_vec(&body).expect("json"))])
    }

    #[test]
    fn update_rows() {
        let exe = Path::new("/nonexistent/pv");
        let (c, o) = update_with(
            &Fake(with_release("v0.69.0"), false),
            &P,
            T,
            &Installs::User,
            exe,
            false,
        );
        assert_eq!((c, o[0].as_str()), (0, "pv 0.69.0 is up to date"));
        let (c, o) = update_with(
            &Fake(with_release("v0.69.1"), false),
            &P,
            T,
            &Installs::User,
            exe,
            true,
        );
        assert_eq!(
            (c, o.as_slice()),
            (
                0,
                ["pv 0.69.0 -> 0.69.1 (release v0.69.1) available".to_string()].as_slice()
            )
        );
        let (c, o) = update_with(
            &Fake(with_release("v0.69.1"), false),
            &P,
            T,
            &Installs::Arbiter("fleet host yoga".into()),
            exe,
            false,
        );
        assert_eq!(c, 0, "fleet host reports, does not install");
        assert!(o[1].starts_with("not installing: fleet host yoga"), "{o:?}");
        let (c, o) = update_with(
            &Fake(with_release("v0.69.1"), false),
            &P,
            T,
            &Installs::User,
            exe,
            false,
        );
        assert_eq!(c, 1, "no published digest -> refused: {o:?}");
        assert!(o[1].contains("no published digest"), "{o:?}");
        let (c, _) = update_with(
            &Fake(HashMap::new(), true),
            &P,
            T,
            &Installs::User,
            exe,
            true,
        );
        assert_eq!(
            c, 1,
            "an explicit check that cannot reach the network says so"
        );
        let (c, o) = update_with(
            &Fake(HashMap::new(), false),
            &P,
            T,
            &Installs::User,
            exe,
            true,
        );
        assert_eq!(
            (c, o[0].as_str()),
            (0, "pv 0.69.0 is up to date"),
            "nothing published (404s)"
        );
    }

    #[test]
    fn refresh_keeps_known_update_when_offline() {
        let d = tempfile::tempdir().expect("tempdir");
        let path = d.path().join("pv/update-check.json");
        refresh(&Fake(with_release("v0.69.1"), false), &P, T, &path, 100);
        let c = cache::load(&path).expect("stored");
        assert_eq!(
            c.available.as_ref().map(|a| a.version.as_str()),
            Some("0.69.1")
        );
        refresh(&Fake(HashMap::new(), true), &P, T, &path, 200);
        let c2 = cache::load(&path).expect("stored");
        assert_eq!(
            (c2.checked_at, c2.available, c2.error.is_some()),
            (200, c.available, true)
        );
    }

    #[test]
    fn startup_spawns_once_a_day_and_never_refreshes_inline() {
        let d = tempfile::tempdir().expect("tempdir");
        let path = d.path().join("pv/update-check.json");
        let spawned = std::cell::Cell::new(0);
        let spawn = || spawned.set(spawned.get() + 1);
        let run = |args: &[&str], now| {
            let a: Vec<String> = args.iter().map(|s| (*s).to_string()).collect();
            startup_with(&P, &a, &path, now, &Installs::User, &spawn)
        };
        assert_eq!(
            run(&["pv", "validate"], 1000),
            None,
            "cold cache: no notice yet"
        );
        assert_eq!(
            spawned.get(),
            1,
            "cold cache: the refresh is spawned, not run"
        );
        let claim = cache::load(&path).expect("the day is claimed before spawning");
        assert_eq!((claim.checked_at, claim.available), (1000, None));
        assert_eq!(run(&["pv", "validate"], 2000), None);
        assert_eq!(spawned.get(), 1, "within 24 h: no second refresh");
        refresh(&Fake(with_release("v0.69.1"), false), &P, T, &path, 3000);
        let n = run(&["pv"], 4000).expect("the cached update is announced");
        assert!(n.starts_with("pv 0.69.0 -> 0.69.1 available"), "{n}");
        assert_eq!(run(&["pv", "update"], 1 + 3000 + cache::TTL_SECS), None);
        assert_eq!(spawned.get(), 1, "`update` itself never spawns a refresh");
        assert!(run(&["pv"], 3000 + cache::TTL_SECS).is_some());
        assert_eq!(spawned.get(), 2, "after 24 h: one more refresh");
    }

    #[test]
    fn update_args_rows() {
        let a =
            |v: &[&str]| parse_update_args(&v.iter().map(|s| (*s).to_string()).collect::<Vec<_>>());
        assert_eq!(a(&[]), UpdateArgs::Install);
        assert_eq!(a(&["--check"]), UpdateArgs::Check);
        assert_eq!(a(&["-h"]), UpdateArgs::Help);
        assert_eq!(a(&["--refresh-cache"]), UpdateArgs::RefreshCache);
        assert_eq!(a(&["--force"]), UpdateArgs::Bad);
        assert_eq!(a(&["--check", "x"]), UpdateArgs::Bad);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn host_target_is_a_triple() {
        let t = host_target();
        assert_eq!(t.split('-').count(), 4, "{t}");
    }
}
