//! Local runner for the thin Chronos MCP server.
//!
//! Default transport is stdio — what MCP clients (Claude Desktop, Claude Code, Cursor)
//! spawn directly. `--http [PORT]` serves the same one tool over streamable-HTTP plus the
//! same-origin demo page. `--coldstart [N]` spawns THIS binary as a stdio server N times
//! and times exec -> initialize -> first forecast (the Lambda cold-start number, minus the
//! container). `--bench [DIRS...]` prints the kernel-variant and latency tables.
//!
//! Everything human-readable goes to stderr: stdout belongs to the protocol. The `--bench`
//! and `--coldstart` tables are reports on non-protocol runs, so they print to stdout by
//! design.
//!
//! ```bash
//! aprender-mcp-chronos                       # stdio
//! aprender-mcp-chronos --http 8766           # demo page + /mcp
//! aprender-mcp-chronos --coldstart 5         # cold-start table
//! aprender-mcp-chronos --bench               # kernel/latency tables
//! ```

// The report tables and the coldstart harness are the deliberate exceptions: they are the
// output, not a protocol path, and the harness asserts on a spawned child.
#![allow(clippy::disallowed_methods)]

use std::io::{BufRead, Write};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use aprender_forecast::chronos::{load_csv, load_model_from_dir};

/// Server name advertised in `initialize`.
const SERVER_NAME: &str = "aprender-chronos";
/// Default `--http` port (8765 belongs to `aprender-mcp-forecast`).
const DEFAULT_PORT: u16 = 8766;
/// The sample series both reports run on, embedded so no relative path is assumed.
const PEYTON_CSV: &str = include_str!("../fixtures/peyton_manning.csv");

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    v[v.len() / 2]
}

/// Variants of the same model (loop vs GEMM routing), latency by context and horizon, and
/// the max |Δ| between variants on the direct 64-step block.
///
/// The spike's rayon-parallel GEMM row is deliberately absent: D-14 fixed the production
/// routing at `fast + attn_gemm + dot8` single rows, single-threaded, and a benchmark row
/// for a path this server never takes is a claim nothing else in the crate stands behind.
fn bench(model_dirs: &[String]) {
    let (_, y) = load_csv(PEYTON_CSV).expect("peyton");
    let full: Vec<f32> = y.iter().map(|v| v.map_or(f32::NAN, |x| x as f32)).collect();
    for dir in model_dirs {
        let t0 = Instant::now();
        let mut m = match load_model_from_dir(std::path::Path::new(dir)) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("error: {dir}: {e}");
                continue;
            }
        };
        println!(
            "\n### {} — {} params, weights {}, load {:.0} ms (parse + f32 decode + transposes)\n",
            m.name,
            m.n_params,
            m.dtype,
            t0.elapsed().as_secs_f64() * 1e3
        );
        println!("| variant | forward ms (2048 ctx) | max abs Δ vs all-GEMM (q, 64 steps) |");
        println!("|---|---|---|");
        let ctx = &full[full.len() - 2048..];
        let reference = {
            m.bolt.fast = true;
            m.bolt.attn_gemm = true;
            m.bolt.row1_gemv = false;
            m.bolt.forward(ctx)
        };
        for (name, fast, attn, row1) in [
            ("plain loops (spike 005 default)", false, false, false),
            (
                "GEMM projections + FF + embedding, single rows via trueno gemv",
                true,
                false,
                true,
            ),
            ("+ attention scores/context via GEMM", true, true, true),
            (
                "+ single-row projections as contiguous dot8 instead of gemv (D-14 production)",
                true,
                true,
                false,
            ),
        ] {
            m.bolt.fast = fast;
            m.bolt.attn_gemm = attn;
            m.bolt.row1_gemv = row1;
            let out = m.bolt.forward(ctx);
            let d = out
                .iter()
                .zip(&reference)
                .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
                .fold(0.0f32, f32::max);
            let ms = median(
                (0..7)
                    .map(|_| {
                        let t = Instant::now();
                        let _ = m.bolt.forward(ctx);
                        t.elapsed().as_secs_f64() * 1e3
                    })
                    .collect(),
            );
            println!("| {name} | {ms:.1} | {d:.1e} |");
        }
        m.bolt.fast = true;
        m.bolt.attn_gemm = true;
        m.bolt.row1_gemv = false;
        let (_, stages) = m.bolt.forward_stages(ctx);
        println!(
            "\nStage breakdown (2048 ctx, all-GEMM, dot8 single rows): {}",
            stages
                .iter()
                .map(|(n, ms)| format!("{n} {ms:.1} ms"))
                .collect::<Vec<_>>()
                .join(" · ")
        );
        println!("\n| context | horizon | forwards | predict ms (median of 5) |");
        println!("|---|---|---|---|");
        for &ctx_len in &[100usize, 512, 2048] {
            for &h in &[12usize, 64, 365] {
                let c = &full[full.len() - ctx_len..];
                let mut forwards = 0;
                let ms = median(
                    (0..5)
                        .map(|_| {
                            let t = Instant::now();
                            let (_, f) = m.bolt.predict(c, h);
                            forwards = f;
                            t.elapsed().as_secs_f64() * 1e3
                        })
                        .collect(),
                );
                println!("| {ctx_len} | {h} | {forwards} | {ms:.1} |");
            }
        }
    }
}

/// Spawn this binary as a stdio MCP server and time initialize -> first forecast: cold
/// start as a Lambda would see it, minus the container (process exec + weight decode +
/// first call).
fn coldstart(runs: usize) -> ExitCode {
    let exe = std::env::current_exe().expect("exe");
    let (ds, y) = load_csv(PEYTON_CSV).expect("peyton");
    let call = serde_json::json!({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "forecast", "arguments": {"ds": ds, "y": y, "horizon": 64}}
    })
    .to_string();
    println!("| run | exec → initialize response ms | → forecast response ms |");
    println!("|---|---|---|");
    let mut times = Vec::with_capacity(runs);
    for run in 0..runs {
        let t0 = Instant::now();
        let mut child = std::process::Command::new(&exe)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn");
        let mut stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let mut lines = std::io::BufReader::new(stdout).lines();
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"2025-06-18","capabilities":{{}},"clientInfo":{{"name":"coldstart","version":"0"}}}}}}"#
        )
        .expect("write");
        let Some(Ok(init)) = lines.next() else {
            eprintln!("error: the spawned server produced no initialize reply");
            let _ = child.kill();
            let _ = child.wait();
            return ExitCode::FAILURE;
        };
        let t_init = t0.elapsed().as_secs_f64() * 1e3;
        assert!(
            init.contains("serverInfo"),
            "unexpected initialize reply: {init}"
        );
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","method":"notifications/initialized"}}"#
        )
        .expect("write");
        writeln!(stdin, "{call}").expect("write");
        let Some(Ok(resp)) = lines.next() else {
            eprintln!("error: the spawned server produced no forecast reply");
            let _ = child.kill();
            let _ = child.wait();
            return ExitCode::FAILURE;
        };
        let t_call = t0.elapsed().as_secs_f64() * 1e3;
        let v: serde_json::Value = serde_json::from_str(&resp).expect("json");
        assert!(v.get("error").is_none(), "forecast error: {resp}");
        drop(stdin);
        let _ = child.kill();
        let _ = child.wait();
        println!("| {} | {t_init:.0} | {t_call:.0} |", run + 1);
        times.push((t_init, t_call));
    }
    if !times.is_empty() {
        println!(
            "\nmedian: initialize {:.0} ms, forecast {:.0} ms (n = {runs}, debug build unless \
             --release)",
            median(times.iter().map(|t| t.0).collect()),
            median(times.iter().map(|t| t.1).collect())
        );
    }
    ExitCode::SUCCESS
}

/// Load the model once and announce it on stderr — the ONE banner (T-06-05: model name,
/// params, dtype, source and load time; nothing about the host or the caller).
fn load_and_announce() -> Option<Arc<aprender_forecast::chronos::Model>> {
    match aprender_mcp_chronos::resolve_model() {
        Ok(model) => {
            eprintln!(
                "{SERVER_NAME}: {} ({} params, {}, {}) loaded in {:.0} ms",
                model.name,
                model.n_params,
                model.dtype,
                model.source,
                model.load_seconds * 1e3
            );
            Some(Arc::new(model))
        }
        Err(error) => {
            eprintln!("error: {error}");
            None
        }
    }
}

async fn serve_http(port: u16) -> ExitCode {
    let Some(model) = load_and_announce() else {
        return ExitCode::FAILURE;
    };
    let server =
        match aprender_mcp_chronos::build_server(model, SERVER_NAME, env!("CARGO_PKG_VERSION")) {
            Ok(server) => server,
            Err(error) => {
                eprintln!("error: server construction refused: {error}");
                return ExitCode::FAILURE;
            }
        };
    let app = aprender_mcp_chronos::http_app(server);
    let listener = match tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("error: bind {port}: {error}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "{SERVER_NAME}: demo page http://127.0.0.1:{port}/ — MCP streamable-http at \
         http://127.0.0.1:{port}/mcp"
    );
    if let Err(error) = axum::serve(listener, app).await {
        eprintln!("error: {error}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

async fn serve_stdio() -> ExitCode {
    let Some(model) = load_and_announce() else {
        return ExitCode::FAILURE;
    };
    let server =
        match aprender_mcp_chronos::build_server(model, SERVER_NAME, env!("CARGO_PKG_VERSION")) {
            Ok(server) => server,
            Err(error) => {
                eprintln!("error: server construction refused: {error}");
                return ExitCode::FAILURE;
            }
        };
    eprintln!(
        "{SERVER_NAME}: serving `{}` on stdio",
        aprender_mcp_chronos::TOOL_NAME
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
            let dirs: Vec<String> = if args.len() > 1 {
                args[1..].to_vec()
            } else {
                vec![
                    "models/chronos-bolt-tiny/f32".into(),
                    "models/chronos-bolt-tiny/f16".into(),
                ]
            };
            bench(&dirs);
            ExitCode::SUCCESS
        }
        Some("--coldstart") => {
            let Some(runs) = args.get(1).map_or(Some(3), |s| s.parse().ok()) else {
                eprintln!("error: usage: aprender-mcp-chronos --coldstart [N] (N is a number)");
                return ExitCode::from(2);
            };
            coldstart(runs)
        }
        Some("--http") => {
            let Some(port) = args.get(1).map_or(Some(DEFAULT_PORT), |s| s.parse().ok()) else {
                eprintln!("error: usage: aprender-mcp-chronos --http [PORT] (PORT is a number)");
                return ExitCode::from(2);
            };
            serve_http(port).await
        }
        Some(other) if other.starts_with('-') => {
            eprintln!(
                "error: unknown argument {other}; usage: aprender-mcp-chronos [--http PORT] \
                 [--coldstart N] [--bench DIRS...]"
            );
            ExitCode::from(2)
        }
        _ => serve_stdio().await,
    }
}
