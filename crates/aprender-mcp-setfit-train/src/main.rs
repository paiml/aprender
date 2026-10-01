//! Local runner for the thin SetFit training MCP server — stdio transport.
//!
//! Stdio is what MCP clients (Claude Desktop, Claude Code, Cursor) spawn
//! directly, and what the E2E test drives. Everything human-readable goes to
//! stderr: stdout belongs to the protocol.
//!
//! ```bash
//! . scripts/apr_bin.sh || exit 1   # never a bare `apr`, never a hardcoded path
//! aprender-mcp-setfit-train \
//!   --apr-bin "$APR" \
//!   --data data/tweet-eval-stance \
//!   --selection data/tweet-eval-stance/selection-manifest.json \
//!   --model-dir ~/.cache/aprender/minilm-l6-v2-1110a243 \
//!   --output-dir /tmp/setfit-train-out
//! ```
//!
//! Every flag also reads an `APRENDER_SETFIT_TRAIN_*` env var, so a transport
//! wrapper (the Lambda phase) can configure the same server without argv.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use aprender_mcp_setfit_train::{
    build_server, AprenderTaskStore, CancelSink, Dispatcher, InMemoryTaskBackend, LocalDispatcher,
    RunningJobs, TrainerPaths, ENV_APR_BIN, ENV_DATA, ENV_MODEL_DIR, ENV_OUTPUT_DIR, ENV_SELECTION,
    SERVER_NAME,
};

/// Which transport to serve.
///
/// HTTP is the DEFAULT because a remote MCP server is the deployment this org
/// encourages; stdio is the local-development opt-in. That is also the shape
/// the in-house `approval-mcp` server uses ("HTTP-first", its D-02/D-03).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Transport {
    /// Streamable HTTP on `addr` — remote clients, many sessions.
    Http(std::net::SocketAddr),
    /// Stdio — what a local client (Claude Desktop, Claude Code, Cursor)
    /// spawns directly.
    Stdio,
}

/// What argv asked for: a configured server, or just its own documentation.
#[derive(Debug)]
enum Parsed {
    Serve {
        paths: TrainerPaths,
        transport: Transport,
    },
    /// `--help`/`-h`/`--version`/`-V`: print and exit 0.
    Message(String),
}

const USAGE: &str = "\
aprender-mcp-setfit-train — thin single-algorithm MCP training server (SetFit)

USAGE:
    aprender-mcp-setfit-train --apr-bin <PATH> --data <DIR> --selection <FILE> \\
                              --model-dir <DIR> --output-dir <DIR> [--addr <ADDR> | --stdio]

TRANSPORT (streamable HTTP by default — this is a remote server):
    --addr <ADDR>    bind address for streamable HTTP (default 127.0.0.1:8080;
                     use 0.0.0.0:<port> to accept remote clients, and :0 to let
                     the OS pick, which is printed on startup)
    --stdio          serve stdio instead, for a local client that spawns this
                     binary directly

Every flag also reads an environment variable; the flag wins:
    --apr-bin     APRENDER_SETFIT_TRAIN_APR_BIN     a pinned apr built with --features setfit
    --data        APRENDER_SETFIT_TRAIN_DATA        attested benchmark dir (apr data ...)
    --selection   APRENDER_SETFIT_TRAIN_SELECTION   selection-manifest.json (apr data select)
    --model-dir   APRENDER_SETFIT_TRAIN_MODEL_DIR   pinned all-MiniLM-L6-v2 checkout
    --output-dir  APRENDER_SETFIT_TRAIN_OUTPUT_DIR  where configs and artifacts are written

    -h, --help       print this and exit
    -V, --version    print the version and exit
";

/// Flag beats env var; absent in both is an error naming the pair.
fn resolve(flag: &str, env: &str, value: Option<PathBuf>) -> Result<PathBuf, String> {
    value
        .or_else(|| std::env::var_os(env).map(PathBuf::from))
        .ok_or_else(|| format!("missing {flag} (or {env})"))
}

/// Parse argv into the five operator-owned paths.
///
/// Each flag names its destination in the `match`, so the compiler checks the
/// association. An earlier revision kept the settings in an array and unpacked
/// them positionally — with five same-typed fields, reordering the array
/// silently routed one flag's value into another's slot and still compiled.
///
/// The `match` has the SAME hazard (all five slots are `Option<PathBuf>`, so
/// swapping two arms still compiles), which is why this takes an iterator
/// rather than the concrete `std::env::ArgsOs`: a synthetic argv cannot be
/// built from `ArgsOs`, so the guard could not be tested at all. See
/// `each_flag_lands_in_its_own_field`.
///
/// `OsString`, not `String`: `std::env::args()` PANICS on any non-UTF-8
/// argument, and every value here is a filesystem path, which on Unix need not
/// be UTF-8. That panic aborted with a backtrace before a single one of main's
/// deliberate `ExitCode::from(2)` messages could run — and the env-var half of
/// `resolve` already used `var_os`, so the two halves disagreed. Only the FLAG
/// needs to be text; the value is handed to `PathBuf` as an `OsString`.
fn parse_paths(mut args: impl Iterator<Item = std::ffi::OsString>) -> Result<Parsed, String> {
    let (mut apr_bin, mut data, mut selection, mut model_dir, mut output_dir) =
        (None, None, None, None, None);
    let mut stdio = false;
    let mut addr: Option<std::net::SocketAddr> = None;
    let _argv0 = args.next();
    while let Some(arg) = args.next() {
        let flag = arg.to_string_lossy().into_owned();
        let slot = match flag.as_str() {
            "--apr-bin" => &mut apr_bin,
            "--data" => &mut data,
            "--selection" => &mut selection,
            "--model-dir" => &mut model_dir,
            "--output-dir" => &mut output_dir,
            // A binary that documents five flags and five env vars must be
            // able to say so. These used to fall into `other` and exit 2 with
            // "unknown argument --help".
            "--stdio" => {
                stdio = true;
                continue;
            }
            "--addr" => {
                let value = args
                    .next()
                    .ok_or_else(|| String::from("--addr requires a bind address"))?;
                let text = value.to_string_lossy().into_owned();
                addr = Some(text.parse().map_err(|e| {
                    format!("--addr {text} is not a bind address (host:port): {e}")
                })?);
                continue;
            }
            "--help" | "-h" => return Ok(Parsed::Message(USAGE.to_string())),
            "--version" | "-V" => {
                return Ok(Parsed::Message(format!(
                    "{SERVER_NAME} {}",
                    env!("CARGO_PKG_VERSION")
                )))
            }
            other => {
                return Err(format!(
                    "unknown argument {other}; expected \
                     --apr-bin/--data/--selection/--model-dir/--output-dir (--help for usage)"
                ))
            }
        };
        let value = args
            .next()
            .ok_or_else(|| format!("{flag} requires a path"))?;
        *slot = Some(PathBuf::from(value));
    }
    if stdio && addr.is_some() {
        return Err(String::from(
            "--stdio and --addr are mutually exclusive: pick one transport",
        ));
    }
    let transport = if stdio {
        Transport::Stdio
    } else {
        // The env var exists so a container can set the bind address without
        // argv, exactly like the five paths.
        let from_env = std::env::var("APRENDER_SETFIT_TRAIN_ADDR")
            .ok()
            .map(|text| {
                text.parse::<std::net::SocketAddr>().map_err(|e| {
                    format!("APRENDER_SETFIT_TRAIN_ADDR {text} is not a bind address: {e}")
                })
            })
            .transpose()?;
        Transport::Http(
            addr.or(from_env)
                .unwrap_or_else(|| std::net::SocketAddr::from(([127, 0, 0, 1], 8080))),
        )
    };
    Ok(Parsed::Serve {
        paths: TrainerPaths {
            apr_bin: resolve("--apr-bin", ENV_APR_BIN, apr_bin)?,
            data: resolve("--data", ENV_DATA, data)?,
            selection: resolve("--selection", ENV_SELECTION, selection)?,
            model_dir: resolve("--model-dir", ENV_MODEL_DIR, model_dir)?,
            output_dir: resolve("--output-dir", ENV_OUTPUT_DIR, output_dir)?,
        },
        transport,
    })
}

/// A current-thread runtime: this server supervises one child process and
/// frames stdio. Every task it runs is async I/O — a timer-free waiter, two
/// pipe drains inside `wait_with_output`, and the protocol loop — so a worker
/// thread per core would only add scheduler contention against a trainer that
/// is already saturating the CPU at ~4 GB RSS.
#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let (paths, transport) = match parse_paths(std::env::args_os()) {
        Ok(Parsed::Serve { paths, transport }) => (paths, transport),
        Ok(Parsed::Message(text)) => {
            println!("{text}");
            return ExitCode::SUCCESS;
        }
        Err(message) => {
            eprintln!("error: {message}");
            return ExitCode::from(2);
        }
    };
    if let Err(message) = paths.validate() {
        eprintln!("error: {message}");
        return ExitCode::from(2);
    }

    let paths = Arc::new(paths);
    // Single-flight admission AND the cancel relay. The store owns the task
    // RECORD; this owns the running child; `CancelSink` is the seam. Cancelling
    // a record without stopping the work would leave a CPU-saturating trainer
    // alive and every later submit refused.
    let running = Arc::new(RunningJobs::new());
    // `InMemoryTaskBackend` is the right backend for a long-lived stdio
    // process. The serverless deployment swaps in a DynamoDB backend behind the
    // same seam — the store, the tools and this wiring do not change.
    let store = Arc::new(
        AprenderTaskStore::new(Arc::new(InMemoryTaskBackend::new()))
            .with_cancel_sink(Arc::clone(&running) as Arc<dyn CancelSink>),
    );

    eprintln!(
        "{SERVER_NAME}: apr={} data={} out={}",
        paths.apr_bin.display(),
        paths.data.display(),
        paths.output_dir.display(),
    );

    // This runner trains IN THIS PROCESS: it is a long-lived server, so the
    // local dispatcher is the right one. The serverless request Lambda builds
    // the same server around a Step Functions dispatcher and needs none of the
    // paths above.
    let dispatcher: Arc<dyn Dispatcher> =
        Arc::new(LocalDispatcher::new(paths, running, Arc::clone(&store)));

    let server = match build_server(store, dispatcher, SERVER_NAME, env!("CARGO_PKG_VERSION")) {
        Ok(server) => server,
        Err(error) => {
            eprintln!("error: server construction refused: {error}");
            return ExitCode::FAILURE;
        }
    };

    match transport {
        Transport::Http(addr) => {
            // The bound address is PRINTED, not assumed: `--addr :0` is the
            // only way an E2E can take a free port, and it needs to be told
            // which one it got. stderr, because stdout is the stdio protocol's
            // and a server that serves both must not differ in that habit.
            match aprender_mcp_setfit_train::serve_http(server, addr).await {
                Ok((bound, handle)) => {
                    eprintln!("{SERVER_NAME}: streamable HTTP listening on http://{bound}");
                    if let Err(error) = handle.await {
                        eprintln!("error: HTTP server terminated: {error}");
                        return ExitCode::FAILURE;
                    }
                }
                Err(error) => {
                    eprintln!("error: cannot serve HTTP on {addr}: {error}");
                    return ExitCode::FAILURE;
                }
            }
        }
        Transport::Stdio => {
            eprintln!("{SERVER_NAME}: serving `train`/`train_status` on stdio");
            if let Err(error) = server.run_stdio().await {
                eprintln!("error: stdio server terminated: {error}");
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::{parse_paths, Parsed, Transport};
    use std::ffi::OsString;

    fn argv(args: &[&str]) -> Vec<OsString> {
        std::iter::once("aprender-mcp-setfit-train")
            .chain(args.iter().copied())
            .map(OsString::from)
            .collect()
    }

    /// The guard the `match` exists for: five same-typed slots, so a swapped
    /// arm compiles and only shows up as `validate()` blaming the wrong flag.
    /// Every value here is distinct, so any mis-routing fails an assertion.
    #[test]
    fn each_flag_lands_in_its_own_field() {
        let parsed = parse_paths(
            argv(&[
                "--apr-bin",
                "/bin/apr",
                "--data",
                "/d/data",
                "--selection",
                "/s/selection.json",
                "--model-dir",
                "/m/model",
                "--output-dir",
                "/o/out",
            ])
            .into_iter(),
        )
        .expect("a complete argv parses");
        let Parsed::Serve { paths, transport } = parsed else {
            panic!("a complete argv is not a --help request")
        };
        assert_eq!(
            transport,
            Transport::Http(std::net::SocketAddr::from(([127, 0, 0, 1], 8080))),
            "HTTP is the default transport — this is a remote server"
        );
        assert_eq!(paths.apr_bin, std::path::Path::new("/bin/apr"));
        assert_eq!(paths.data, std::path::Path::new("/d/data"));
        assert_eq!(paths.selection, std::path::Path::new("/s/selection.json"));
        assert_eq!(paths.model_dir, std::path::Path::new("/m/model"));
        assert_eq!(paths.output_dir, std::path::Path::new("/o/out"));
    }

    /// The org default. A regression here silently turns a remote server into
    /// a local-only one, which no other assertion would notice.
    #[test]
    fn http_is_the_default_and_stdio_is_opt_in() {
        let base = [
            "--apr-bin",
            "/bin/apr",
            "--data",
            "/d",
            "--selection",
            "/s",
            "--model-dir",
            "/m",
            "--output-dir",
            "/o",
        ];
        let parsed = parse_paths(argv(&base).into_iter()).expect("parses");
        let Parsed::Serve { transport, .. } = parsed else {
            panic!("not a --help request")
        };
        assert_eq!(
            transport,
            Transport::Http(std::net::SocketAddr::from(([127, 0, 0, 1], 8080)))
        );

        let mut with_stdio = base.to_vec();
        with_stdio.push("--stdio");
        let Parsed::Serve { transport, .. } =
            parse_paths(argv(&with_stdio).into_iter()).expect("parses")
        else {
            panic!("not a --help request")
        };
        assert_eq!(transport, Transport::Stdio);
    }

    #[test]
    fn an_explicit_addr_is_honoured_and_conflicts_are_refused() {
        let mut args = vec![
            "--apr-bin",
            "/bin/apr",
            "--data",
            "/d",
            "--selection",
            "/s",
            "--model-dir",
            "/m",
            "--output-dir",
            "/o",
            "--addr",
            "0.0.0.0:9000",
        ];
        let Parsed::Serve { transport, .. } = parse_paths(argv(&args).into_iter()).expect("parses")
        else {
            panic!("not a --help request")
        };
        assert_eq!(
            transport,
            Transport::Http(std::net::SocketAddr::from(([0, 0, 0, 0], 9000)))
        );

        args.push("--stdio");
        let err = parse_paths(argv(&args).into_iter())
            .expect_err("two transports is a configuration error, not a silent precedence rule");
        assert!(err.contains("mutually exclusive"), "{err}");

        let bad = vec![
            "--apr-bin",
            "/bin/apr",
            "--data",
            "/d",
            "--selection",
            "/s",
            "--model-dir",
            "/m",
            "--output-dir",
            "/o",
            "--addr",
            "not-an-address",
        ];
        let err = parse_paths(argv(&bad).into_iter()).expect_err("a bad addr must be refused");
        assert!(err.contains("--addr"), "{err}");
    }

    #[test]
    fn help_and_version_are_answered_not_refused() {
        for flag in ["--help", "-h", "--version", "-V"] {
            let parsed = parse_paths(argv(&[flag]).into_iter())
                .unwrap_or_else(|e| panic!("{flag} must be answered, got: {e}"));
            assert!(
                matches!(parsed, Parsed::Message(_)),
                "{flag} must print, not configure a server"
            );
        }
    }

    #[test]
    fn a_missing_value_names_the_flag_and_a_missing_flag_names_its_env_var() {
        let err = parse_paths(argv(&["--data"]).into_iter()).expect_err("--data has no value");
        assert!(err.contains("--data"), "{err}");
        let err = parse_paths(argv(&["--apr-bin", "/bin/apr"]).into_iter())
            .expect_err("the other four are unset");
        assert!(
            err.contains("--data") && err.contains("APRENDER_SETFIT_TRAIN_DATA"),
            "a refusal must name BOTH doors: {err}"
        );
    }

    #[test]
    fn a_non_utf8_path_is_carried_through_rather_than_panicking() {
        // `std::env::args()` panicked here; `OsString` does not. Only the FLAG
        // is required to be text.
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let weird = OsString::from_vec(b"/models/caf\xe9-minilm".to_vec());
            let mut args = argv(&[
                "--apr-bin",
                "/bin/apr",
                "--data",
                "/d",
                "--selection",
                "/s",
                "--output-dir",
                "/o",
                "--model-dir",
            ]);
            args.push(weird.clone());
            let parsed = parse_paths(args.into_iter()).expect("non-UTF-8 paths parse");
            let Parsed::Serve { paths, .. } = parsed else {
                panic!("not a --help request")
            };
            assert_eq!(paths.model_dir.as_os_str(), weird);
        }
    }
}
