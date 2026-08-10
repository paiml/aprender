//! Offline mode detection and enforcement for `apr --offline` (CRUX-A-20).
//!
//! Contract: `contracts/crux-A-20-v1.yaml`.
//!
//! Two layers live here:
//!
//! 1. [`is_offline`] — a pure classifier over the parsed `--offline` flag and
//!    an environment snapshot. No I/O, deterministic, unit-testable.
//! 2. The **enforcement point** — [`latch`], [`scope`], [`network_forbidden`]
//!    and [`guard`]. `--offline` used to be a per-command `bool` that each
//!    command had to remember to forward to each download helper; three of
//!    them (`pull`, `chat`, `showcase`) forgot, so the flag was inert and
//!    `apr pull --offline hf://...` happily downloaded. A per-process latch
//!    removes the forwarding step entirely: the flag is recorded once in
//!    `execute_command`, and every outbound-network helper asks [`guard`].
//!    Forgetting to plumb a parameter can no longer disarm the control.

/// Environment variables that trigger offline mode. APR-native and
/// HF-compatible. Both MUST be observationally equivalent to the
/// `--offline` CLI flag per FALSIFY-CRUX-A-20-004.
pub const OFFLINE_ENV_VARS: &[&str] = &["APR_OFFLINE", "HF_HUB_OFFLINE"];

/// Return true iff the raw env value is a truthy offline signal.
/// HF's own convention (`huggingface_hub.constants.HF_HUB_OFFLINE`) treats
/// "1", "true", "TRUE", "yes" as true and "0", "false", "" as false.
fn env_is_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Resolve offline mode from the CLI flag + environment snapshot.
///
/// Precedence: `cli_flag=true` OR any offline env var truthy → offline.
/// If none are set, or all env vars are falsy and the flag is false,
/// offline is false (default online behavior).
///
/// The environment snapshot is passed in explicitly (rather than read
/// from `std::env`) so callers can test this function deterministically
/// without mutating process-global state.
pub fn is_offline<'a, I>(cli_flag: bool, env: I) -> bool
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    if cli_flag {
        return true;
    }
    for (k, v) in env {
        if OFFLINE_ENV_VARS.contains(&k) && env_is_truthy(v) {
            return true;
        }
    }
    false
}

/// Read the two offline-relevant env vars out of the real process
/// environment. Thin wrapper so callers don't sprinkle `std::env::var`
/// across the codebase.
pub fn read_offline_env() -> Vec<(String, String)> {
    OFFLINE_ENV_VARS
        .iter()
        .filter_map(|k| std::env::var(k).ok().map(|v| ((*k).to_string(), v)))
        .collect()
}

/// Process-wide latch, set once from the parsed CLI in `execute_command`.
/// Monotonic: it can only ever be turned ON, so no later code path can
/// silently re-enable the network for a run the user asked to be offline.
static PROCESS_OFFLINE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

thread_local! {
    /// Thread-scoped override used by [`scope`] so a command that receives
    /// `offline` as a parameter (e.g. `pull::run`) feeds the very same
    /// enforcement point without mutating global state — which also keeps
    /// parallel unit tests isolated, since each test runs on its own thread.
    static THREAD_OFFLINE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Record the process-wide offline decision. Only ever sets the latch.
pub fn latch(cli_flag: bool) {
    if cli_flag {
        PROCESS_OFFLINE.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

/// RAII guard returned by [`scope`]; restores the previous thread value.
pub struct OfflineScope(bool);

impl Drop for OfflineScope {
    fn drop(&mut self) {
        THREAD_OFFLINE.with(|c| c.set(self.0));
    }
}

/// Force offline for the current thread until the returned guard drops.
/// `scope(false)` is a no-op that still restores on drop.
#[must_use]
pub fn scope(on: bool) -> OfflineScope {
    let previous = THREAD_OFFLINE.with(|c| {
        let previous = c.get();
        c.set(previous || on);
        previous
    });
    OfflineScope(previous)
}

/// True iff outbound network I/O is forbidden for this call.
///
/// Reads the process latch, the thread scope, and the environment — so a
/// command that never received the `--offline` parameter still cannot make
/// a request. This is the property that makes the control structural.
pub fn network_forbidden() -> bool {
    if PROCESS_OFFLINE.load(std::sync::atomic::Ordering::SeqCst)
        || THREAD_OFFLINE.with(std::cell::Cell::get)
    {
        return true;
    }
    let env = read_offline_env();
    is_offline(false, env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
}

/// The single enforcement point. Every helper that is about to perform
/// outbound network I/O calls this first. `activity` completes the sentence
/// "Cannot {activity} in --offline mode".
pub fn guard(activity: &str) -> crate::error::Result<()> {
    if network_forbidden() {
        return Err(crate::error::CliError::NetworkError(format!(
            "Cannot {activity} in --offline mode. \
             Network access is disabled by --offline (or APR_OFFLINE=1 / \
             HF_HUB_OFFLINE=1). Use an already-cached model or a local path."
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_alone_triggers_offline() {
        assert!(is_offline(true, std::iter::empty::<(&str, &str)>()));
    }

    #[test]
    fn no_flag_no_env_is_online() {
        assert!(!is_offline(false, std::iter::empty::<(&str, &str)>()));
    }

    #[test]
    fn apr_offline_one_triggers_offline() {
        assert!(is_offline(false, [("APR_OFFLINE", "1")]));
    }

    #[test]
    fn hf_hub_offline_one_triggers_offline() {
        assert!(is_offline(false, [("HF_HUB_OFFLINE", "1")]));
    }

    #[test]
    fn apr_offline_zero_is_online() {
        assert!(!is_offline(false, [("APR_OFFLINE", "0")]));
    }

    #[test]
    fn apr_offline_empty_is_online() {
        assert!(!is_offline(false, [("APR_OFFLINE", "")]));
    }

    #[test]
    fn truthy_variants_all_work() {
        for v in ["1", "true", "TRUE", "yes", "on", "  1  "] {
            assert!(
                is_offline(false, [("APR_OFFLINE", v)]),
                "APR_OFFLINE={v:?} must be truthy",
            );
        }
    }

    #[test]
    fn falsy_variants_all_fail() {
        for v in ["0", "false", "no", "off", "", "random-string"] {
            assert!(
                !is_offline(false, [("APR_OFFLINE", v)]),
                "APR_OFFLINE={v:?} must be falsy",
            );
        }
    }

    #[test]
    fn unrelated_env_var_ignored() {
        assert!(!is_offline(false, [("SOME_OTHER_VAR", "1")]));
    }

    #[test]
    fn flag_overrides_falsy_env() {
        assert!(is_offline(true, [("APR_OFFLINE", "0")]));
    }

    #[test]
    fn is_deterministic() {
        let a = is_offline(false, [("HF_HUB_OFFLINE", "1")]);
        let b = is_offline(false, [("HF_HUB_OFFLINE", "1")]);
        assert_eq!(a, b);
    }

    // ---------------------------------------------------------------
    // CRUX-A-20 enforcement. These assert the REFUSAL, not the shape:
    // each one drives a real command entry point and requires it to come
    // back with an error naming --offline mode. With the enforcement
    // removed, each command instead reaches the network and returns some
    // HTTP/DNS error (or, for `pull`, succeeds and writes to the cache),
    // and every assertion below fails.
    // ---------------------------------------------------------------

    /// The scope must be an override, not a toggle: it can turn offline ON
    /// and must restore the previous value when it drops.
    #[test]
    fn scope_forces_offline_then_restores() {
        let baseline = network_forbidden();
        {
            let _s = scope(true);
            assert!(network_forbidden(), "scope(true) must forbid the network");
            assert!(guard("fetch x").is_err(), "guard must refuse inside scope");
        }
        assert_eq!(
            network_forbidden(),
            baseline,
            "scope must restore the previous decision on drop"
        );
    }

    /// `scope(false)` must not weaken an enclosing offline scope.
    #[test]
    fn inner_online_scope_cannot_re_enable_network() {
        let _outer = scope(true);
        let _inner = scope(false);
        assert!(
            network_forbidden(),
            "an inner scope(false) must NOT re-enable the network"
        );
    }

    /// The refusal must be a `NetworkError` naming `--offline mode`, so it
    /// reads like `apr run --offline` and exits with the network exit code
    /// rather than looking like a download failure.
    #[test]
    fn guard_error_names_offline_mode() {
        let _s = scope(true);
        let err = guard("download hf://org/repo").expect_err("must refuse");
        let msg = err.to_string();
        assert!(
            msg.contains("--offline mode"),
            "refusal must name --offline mode, got: {msg}"
        );
        assert!(
            matches!(err, crate::error::CliError::NetworkError(_)),
            "refusal must be a NetworkError"
        );
    }

    /// `apr pull --offline hf://org/repo` downloaded 3.9 MB and exited 0 in
    /// v0.63.0: `pull::run` read its `offline` parameter only inside the
    /// `--dry-run` branch. Drive `pull::run` with `offline = true` and
    /// require a refusal. The repo name is deliberately nonexistent so the
    /// pre-fix path fails on the HTTP status instead of downloading — that
    /// error does not mention `--offline mode`, so this test is RED before
    /// the fix and needs no network to be GREEN after it.
    #[test]
    fn pull_run_refuses_uncached_model_when_offline() {
        let err = crate::commands::pull::run(
            "hf://hf-internal-testing/apr-offline-falsifier-does-not-exist",
            false,
            false,
            None,
            true,
        )
        .expect_err("pull --offline must refuse an uncached model");
        let msg = err.to_string();
        assert!(
            msg.contains("--offline mode"),
            "pull --offline must refuse before any network I/O, got: {msg}"
        );
    }

    /// `apr pull dataset <repo> --offline` reached HuggingFace and returned
    /// `HTTP 404 Not Found` in v0.63.0 — proof that the listing request went
    /// out. The dataset lister goes through the same `hf_get` choke point.
    #[test]
    fn pull_dataset_refuses_listing_when_offline() {
        let _s = scope(true);
        let err = crate::commands::pull::run_dataset(
            "hf-internal-testing/apr-offline-falsifier-does-not-exist",
            &[],
            None,
            Some(std::path::Path::new("/nonexistent/apr-offline-falsifier")),
            false,
        )
        .expect_err("pull dataset --offline must refuse");
        let msg = err.to_string();
        assert!(
            msg.contains("--offline mode"),
            "dataset listing must be refused offline, got: {msg}"
        );
    }

    /// `apr chat` passed a hardcoded `offline = false` into `resolve_model`
    /// (chat.rs:126), so `apr --offline chat hf://org/repo` downloaded the
    /// model. `resolve_model` now consults the enforcement point itself, so
    /// the refusal no longer depends on any caller remembering to forward
    /// the flag — which is what this asserts: `offline` is passed as
    /// `false`, exactly as chat used to, and the model must STILL be refused.
    #[test]
    fn resolve_model_refuses_uncached_model_when_caller_forgot_the_flag() {
        let _s = scope(true);
        let source = crate::commands::run::ModelSource::parse(
            "hf://hf-internal-testing/apr-offline-falsifier-does-not-exist",
        )
        .expect("parse hf uri");
        let err = crate::commands::run::resolve_model(&source, false, false)
            .expect_err("uncached model must be refused while offline");
        let msg = err.to_string();
        assert!(
            msg.contains("OFFLINE MODE"),
            "resolve_model must refuse regardless of the caller's flag, got: {msg}"
        );
    }

    /// The refusal must not claim something it cannot know.
    ///
    /// Enforcing `--offline` on the Hub API costs `apr run --offline
    /// hf://org/repo` the ability to learn WHICH file a bare repo means — in
    /// 0.63.0 that lookup went out over the network under `--offline`, and
    /// the answer is what made the pacha cache probe possible. Offline,
    /// `file` is therefore `None` and the cache — keyed on the full
    /// `hf://org/repo/<file>` — cannot be probed, so "not cached" would be
    /// an unsupported claim about a file that may well be sitting in it.
    ///
    /// Same repo, two URIs, two different truths: the bare form must say it
    /// cannot resolve, and only the named-file form may say "not cached".
    #[test]
    fn bare_repo_refusal_does_not_claim_the_model_is_uncached() {
        let _s = scope(true);

        let bare = crate::commands::run::ModelSource::parse(
            "hf://hf-internal-testing/apr-offline-falsifier-does-not-exist",
        )
        .expect("parse bare hf uri");
        let bare_msg = crate::commands::run::resolve_model(&bare, false, true)
            .expect_err("bare repo must be refused while offline")
            .to_string();
        assert!(
            !bare_msg.contains("not cached"),
            "a bare repo cannot be known to be uncached offline, got: {bare_msg}"
        );
        assert!(
            bare_msg.contains("cannot resolve"),
            "the bare-repo refusal must say resolution is what failed, got: {bare_msg}"
        );

        let named = crate::commands::run::ModelSource::parse(
            "hf://hf-internal-testing/apr-offline-falsifier-does-not-exist/model.safetensors",
        )
        .expect("parse named hf uri");
        let named_msg = crate::commands::run::resolve_model(&named, false, true)
            .expect_err("named file must be refused while offline")
            .to_string();
        assert!(
            named_msg.contains("not cached"),
            "a named file IS probed in the cache, so this one may say so, got: {named_msg}"
        );
    }

    /// The whole CLI path, parsed exactly as a user types it.
    ///
    /// `apr pull dataset R --offline` reached the Hub and returned
    /// `HTTP 404 Not Found` in v0.63.0 — proof an outbound request went out
    /// under a flag that forbids them. It is the branch furthest from the
    /// flag: the dataset puller never enters `pull::run`, so no amount of
    /// parameter-threading inside `pull` would have covered it.
    ///
    /// `--offline` is declared twice — once as a clap global on `Cli` and
    /// once on the `Pull` variant — and clap populates BOTH, which this
    /// asserts so a future refactor that drops either one is caught here
    /// rather than by a silent download.
    ///
    /// The repo does not exist, so without enforcement this reaches the Hub
    /// and comes back with an HTTP status error — which does not name
    /// `--offline mode`, making this RED before the fix and needing no
    /// network to be GREEN after it.
    #[test]
    fn pull_dataset_subcommand_flag_is_enforced_end_to_end() {
        // The dispatcher matches over the whole `Commands` enum in one frame,
        // which does not fit a 2 MiB test thread. `main` gets 8 MiB, so give
        // this the same room rather than testing a smaller stack than
        // production uses.
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(pull_dataset_offline_body)
            .expect("spawn")
            .join()
            .expect("dataset offline falsifier panicked");
    }

    fn pull_dataset_offline_body() {
        use clap::Parser;

        let cli = crate::Cli::parse_from([
            "apr",
            "pull",
            "dataset",
            "hf-internal-testing/apr-offline-falsifier-does-not-exist",
            "--offline",
        ]);

        // Both carriers of the flag must be populated — the enforcement reads
        // the global (via the latch in `execute_command`) and the variant (via
        // the scope armed in `dispatch_model_commands`), so losing either one
        // would silently narrow the control.
        assert!(cli.offline, "the clap global --offline must be set");
        assert!(
            matches!(*cli.command, crate::Commands::Pull { offline: true, .. }),
            "the flag must also land on the Pull variant"
        );

        // Enter through the real dispatcher rather than `execute_command`:
        // the latch is a process-wide static and this test shares its process
        // with ~6 600 others, several of which do reach the network.
        // `dispatch_model_commands` is the frame that arms the scope for all
        // three `pull` branches, so it is the narrowest entry that still
        // proves the flag travels from `clap` to the refusal. The
        // `execute_command` half — the latch itself — is covered by
        // `latch_survives_into_a_command_that_never_receives_the_flag`.
        let err = crate::dispatch_model_commands(&cli)
            .expect("pull must be handled by the model dispatcher")
            .expect_err("apr pull dataset --offline must refuse");
        let msg = err.to_string();
        assert!(
            msg.contains("--offline mode"),
            "apr pull dataset --offline must refuse before any network I/O, got: {msg}"
        );
    }

    /// `apr --offline chat hf://org/repo` downloaded 478 440 bytes into an
    /// empty cache in v0.63.0: `chat` has no `--offline` of its own, so the
    /// only carrier is the clap global, and `chat.rs` passed a hardcoded
    /// `false` into `resolve_model`. This is the half of the plumbing the
    /// dispatcher-level test above cannot see — that `execute_command`
    /// latches `cli.offline` for a command that never receives it as a
    /// parameter.
    ///
    /// The latch is a process-wide static and cannot be un-set, so setting it
    /// in-process would force every other test in this binary offline (five
    /// `pull` companion tests do real HTTP and go red). It therefore runs in
    /// a child process: this same test binary, re-invoked with a filter and
    /// `APR_OFFLINE_LATCH_CHILD=1`, which takes the branch below.
    ///
    /// The repo does not exist, so without the fix the child reaches the Hub
    /// and prints a download/HTTP failure that says nothing about offline
    /// mode — RED before the fix, and GREEN after it without a network.
    #[test]
    fn latch_survives_into_a_command_that_never_receives_the_flag() {
        const CHILD_ENV: &str = "APR_OFFLINE_LATCH_CHILD";
        const TEST_PATH: &str =
            "commands::offline::tests::latch_survives_into_a_command_that_never_receives_the_flag";

        if std::env::var(CHILD_ENV).is_ok() {
            child_runs_offline_chat();
            return;
        }

        let exe = std::env::current_exe().expect("current test binary");
        let out = std::process::Command::new(exe)
            .args(["--exact", TEST_PATH, "--nocapture", "--test-threads=1"])
            .env(CHILD_ENV, "1")
            .output()
            .expect("re-run this test binary as a child");
        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            combined.contains("OFFLINE-REFUSAL:"),
            "child did not reach the refusal at all; output was:\n{combined}"
        );
        assert!(
            combined.to_ascii_lowercase().contains("offline mode"),
            "apr --offline chat must be refused without any download, got:\n{combined}"
        );
    }

    fn child_runs_offline_chat() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                use clap::Parser;

                let cli = crate::Cli::parse_from([
                    "apr",
                    "--offline",
                    "chat",
                    "hf://hf-internal-testing/apr-offline-falsifier-does-not-exist",
                ]);
                assert!(cli.offline, "the clap global --offline must be set");

                // `dispatch_run.rs` is include!()d into the crate root.
                let err = crate::execute_command(&cli)
                    .expect_err("apr --offline chat must refuse an uncached model");
                println!("OFFLINE-REFUSAL: {err}");
            })
            .expect("spawn")
            .join()
            .expect("offline chat falsifier panicked");
    }
}
