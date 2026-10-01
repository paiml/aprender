//! Local runner for the thin forecast MCP server.
//!
//! Default transport is stdio — what MCP clients (Claude Desktop, Claude Code, Cursor)
//! spawn directly. `--http [PORT] [--pool K]` serves the same one tool over
//! streamable-HTTP plus the same-origin demo page, behind K independent MCP routers
//! (D-12: pmcp 2.19.3 holds one `Arc<Mutex<Server>>` across the whole tool future, so a
//! single router serialises concurrent fits); `--bench [sizes...]` prints a fit/predict
//! table on synthetic series. Everything human-readable goes to stderr: stdout belongs to the protocol. The
//! `--bench` table is a report on a non-protocol run, so it prints to stdout by design.
//!
//! ```bash
//! aprender-mcp-forecast                 # stdio
//! aprender-mcp-forecast --http 8765               # demo page + /mcp, 8 routers
//! aprender-mcp-forecast --http 8765 --pool 16     # 16 concurrent fits in flight
//! aprender-mcp-forecast --http 8765 --pool 1      # one router (the pre-pool behaviour)
//! aprender-mcp-forecast --bench 1000              # timing table
//! ```

use std::process::ExitCode;
use std::time::Instant;

use aprender_forecast::prophet::{days_from_civil, format_ymd, Rng};
use aprender_forecast::ForecastArgs;

/// Server name advertised in `initialize`.
const SERVER_NAME: &str = "aprender-forecast";
/// Default `--http` port.
const DEFAULT_PORT: u16 = 8765;
/// Default router-pool size. Re-exported from the library, where it is ASSERTED equal to
/// `constants.pool_default` by `types::tests::cost_bounds_match_contract` — this crate has
/// no contract reader, so a literal here could only ever CLAIM to mirror the YAML.
/// Size it to the blocking-thread budget: K is the number of fits that can be in flight at
/// once, and each one is seconds of CPU.
use aprender_forecast::types::DEFAULT_POOL;
/// Ceiling on `--pool`. Each router is a whole `pmcp::Server` plus a spawned outbound-drain
/// task, and the useful range is the blocking-thread budget (tokio's default is 512), so
/// anything past this is a resource mistake rather than a configuration.
const MAX_POOL: usize = 256;

/// A synthetic daily series with drifting slope, yearly + weekly terms and noise.
fn synth(n: usize, seed: u64) -> (Vec<String>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let start = days_from_civil(2010, 1, 1);
    let mut slope = 0.002;
    let mut ds = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut level = 10.0;
    for i in 0..n {
        let day = start + i as i64;
        if i % 400 == 0 && i > 0 {
            slope += (rng.uniform() - 0.5) * 0.004;
        }
        level += slope;
        let f = day as f64;
        let v = level
            + 0.8 * (2.0 * std::f64::consts::PI * f / 365.25).sin()
            + 0.3 * (2.0 * std::f64::consts::PI * f / 7.0).cos()
            + 0.2 * rng.normal();
        ds.push(format_ymd(day));
        y.push(v);
    }
    (ds, y)
}

/// The `--bench` report. Not a protocol path: these `println!`s are the report itself.
fn bench(sizes: &[usize]) {
    println!("| points | model | fit s | predict s | total s | L-BFGS rounds/iters/evals |");
    println!("|---|---|---|---|---|---|");
    for &n in sizes {
        let (ds, y) = synth(n, 3);
        let args = ForecastArgs {
            ds,
            y,
            horizon: 365,
            freq: None,
            model: Some("prophet".into()),
            growth: None,
            cap: None,
            seasonality_mode: None,
            interval_width: None,
            holidays: None,
            n_lags: None,
            seed: None,
            regressors: None,
        };
        let t0 = Instant::now();
        match aprender_forecast::forecast(&args) {
            Ok(r) => {
                let total = t0.elapsed().as_secs_f64();
                let lb = &r.diagnostics["lbfgs"];
                println!(
                    "| {n} | prophet | {:.3} | {:.3} | {total:.3} | {}/{}/{} |",
                    r.fit_seconds,
                    r.predict_seconds,
                    lb["rounds"],
                    lb["iterations"],
                    lb["evaluations"]
                );
            }
            Err(e) => eprintln!("error: {n} points refused: {e}"),
        }
    }
}

fn build() -> Option<pmcp::Server> {
    match aprender_mcp_forecast::build_server(SERVER_NAME, env!("CARGO_PKG_VERSION")) {
        Ok(server) => Some(server),
        Err(error) => {
            eprintln!("error: server construction refused: {error}");
            None
        }
    }
}

/// Parse the tail of `--http [PORT] [--pool K]` in either order.
///
/// Returns `None` for a usage error (an unparseable port, a `--pool` with no value, a
/// `--pool` value that is not a number, or a `--pool` above [`MAX_POOL`]) so `main` can
/// exit 2 rather than silently defaulting — the same refuse-never-default rule the tool
/// boundary itself follows.
fn parse_http_args(rest: &[String]) -> Option<(u16, usize)> {
    let mut port = DEFAULT_PORT;
    let mut pool = DEFAULT_POOL;
    let mut i = 0;
    while i < rest.len() {
        if rest[i] == "--pool" {
            pool = rest.get(i + 1)?.parse().ok()?;
            // `pooled_app` does `Vec::with_capacity(pool)` and builds one `Server` plus one
            // tokio drain task per router, so an unbounded K is a crash (`--pool
            // 18446744073709551615` panics with "capacity overflow") or a task storm before
            // the listener is ever bound. A usage error is the documented outcome.
            if pool > MAX_POOL {
                return None;
            }
            i += 2;
        } else {
            port = rest[i].parse().ok()?;
            i += 1;
        }
    }
    Some((port, pool))
}

async fn serve_http(port: u16, pool: usize) -> ExitCode {
    let app = match aprender_mcp_forecast::pooled_app(pool, SERVER_NAME, env!("CARGO_PKG_VERSION"))
    {
        Ok(app) => app,
        Err(error) => {
            eprintln!("error: server construction refused: {error}");
            return ExitCode::FAILURE;
        }
    };
    let listener = match tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("error: bind {port}: {error}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "{SERVER_NAME}: demo page http://127.0.0.1:{port}/ — MCP streamable-http at \
         http://127.0.0.1:{port}/mcp — router pool {pool} ({})",
        if pool <= 1 {
            "single router: concurrent fits serialise behind pmcp's server mutex"
        } else {
            "K concurrent fits in flight"
        }
    );
    if let Err(error) = axum::serve(listener, app).await {
        eprintln!("error: {error}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

async fn serve_stdio() -> ExitCode {
    let Some(server) = build() else {
        return ExitCode::FAILURE;
    };
    eprintln!(
        "{SERVER_NAME}: serving `{}` on stdio",
        aprender_mcp_forecast::TOOL_NAME
    );
    if let Err(error) = server.run_stdio().await {
        eprintln!("error: stdio server terminated: {error}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--bench") => {
            let sizes: Vec<usize> = args[1..].iter().filter_map(|a| a.parse().ok()).collect();
            bench(if sizes.is_empty() {
                &[1_000, 3_000, 10_000, 20_000]
            } else {
                &sizes
            });
            ExitCode::SUCCESS
        }
        Some("--http") => {
            let Some((port, pool)) = parse_http_args(&args[1..]) else {
                eprintln!(
                    "error: usage: aprender-mcp-forecast --http [PORT] [--pool K] \
                     (PORT is a number, K is a number; K <= 1 means one router)"
                );
                return ExitCode::from(2);
            };
            serve_http(port, pool).await
        }
        Some(other) if other.starts_with('-') => {
            eprintln!(
                "error: unknown argument {other}; usage: aprender-mcp-forecast \
                 [--http PORT [--pool K]] [--bench SIZES...]"
            );
            ExitCode::from(2)
        }
        _ => serve_stdio().await,
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_http_args, DEFAULT_POOL, DEFAULT_PORT};

    fn argv(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn http_args_default_to_the_contract_pool_size() {
        assert_eq!(
            parse_http_args(&argv(&[])),
            Some((DEFAULT_PORT, DEFAULT_POOL))
        );
        assert_eq!(
            parse_http_args(&argv(&["9000"])),
            Some((9000, DEFAULT_POOL))
        );
    }

    #[test]
    fn http_args_accept_pool_in_either_order() {
        assert_eq!(
            parse_http_args(&argv(&["9000", "--pool", "16"])),
            Some((9000, 16))
        );
        assert_eq!(
            parse_http_args(&argv(&["--pool", "16", "9000"])),
            Some((9000, 16))
        );
        assert_eq!(
            parse_http_args(&argv(&["--pool", "1"])),
            Some((DEFAULT_PORT, 1))
        );
    }

    #[test]
    fn a_usage_error_refuses_rather_than_defaulting() {
        // Refuse, never default (D-11) — the same rule the tool boundary follows.
        assert_eq!(
            parse_http_args(&argv(&["--pool"])),
            None,
            "--pool with no K"
        );
        assert_eq!(parse_http_args(&argv(&["--pool", "many"])), None);
        assert_eq!(parse_http_args(&argv(&["not-a-port"])), None);
        assert_eq!(
            parse_http_args(&argv(&["70000"])),
            None,
            "port must fit u16"
        );
    }
}
