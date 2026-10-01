//! chronos-mcp — stdio MCP server (default), `--http [port]` for the browser demo, `--bench`,
//! `--coldstart` (spawns itself over stdio and times the first responses).
#![allow(clippy::disallowed_methods)]
use std::io::{BufRead, Write};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

const SERVER_NAME: &str = "aprender-chronos";

fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).expect("finite")); v[v.len() / 2] }

/// Variants of the same model (loops vs GEMM paths), latency by context and horizon, and the
/// max |Δ| between variants on the direct 64-step block.
fn bench(model_dirs: &[String]) {
    let (_, y) = chronos_mcp::load_csv("fixtures/peyton_manning.csv").expect("peyton");
    let full: Vec<f32> = y.iter().map(|v| v.map_or(f32::NAN, |x| x as f32)).collect();
    for dir in model_dirs {
        let t0 = Instant::now();
        let mut m = chronos_mcp::load_model_from_dir(std::path::Path::new(dir)).expect("model");
        println!("\n### {} — {} params, weights {}, load {:.0} ms (parse + f32 decode + transposes)\n", m.name, m.n_params, m.dtype, t0.elapsed().as_secs_f64() * 1e3);
        println!("| variant | forward ms (2048 ctx) | max abs Δ vs all-GEMM (q, 64 steps) |\n|---|---|---|");
        let ctx = &full[full.len() - 2048..];
        let reference = { m.bolt.fast = true; m.bolt.attn_gemm = true; m.bolt.forward(ctx) };
        for (name, fast, attn, row1, par) in [("plain loops (spike 005 default)", false, false, false, false), ("GEMM projections + FF + embedding, single rows via trueno gemv", true, false, true, false), ("+ attention scores/context via GEMM", true, true, true, false), ("+ single-row projections as contiguous dot8 instead of gemv", true, true, false, false), ("+ rayon-parallel `blis::gemm` (crate feature `parallel`, 10 cores)", true, true, false, true)] {
            m.bolt.fast = fast; m.bolt.attn_gemm = attn; m.bolt.row1_gemv = row1; chronos_mcp::bolt::PARALLEL_GEMM.store(par, std::sync::atomic::Ordering::Relaxed);
            let out = m.bolt.forward(ctx);
            let d = out.iter().zip(&reference).flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs())).fold(0.0f32, f32::max);
            let ms = median((0..7).map(|_| { let t = Instant::now(); let _ = m.bolt.forward(ctx); t.elapsed().as_secs_f64() * 1e3 }).collect());
            println!("| {name} | {ms:.1} | {d:.1e} |");
        }
        m.bolt.fast = true; m.bolt.attn_gemm = true; m.bolt.row1_gemv = false; chronos_mcp::bolt::PARALLEL_GEMM.store(false, std::sync::atomic::Ordering::Relaxed);
        let (_, stages) = m.bolt.forward_stages(ctx);
        let (_, stages) = { let _ = stages; m.bolt.forward_stages(ctx) };
        println!("\nStage breakdown (2048 ctx, all-GEMM, dot8 single rows): {}", stages.iter().map(|(n, ms)| format!("{n} {ms:.1} ms")).collect::<Vec<_>>().join(" · "));
        println!("\n| context | horizon | forwards | predict ms (median of 5) |\n|---|---|---|---|");
        for &ctx_len in &[100usize, 512, 2048] {
            for &h in &[12usize, 64, 365] {
                let c = &full[full.len() - ctx_len..];
                let mut forwards = 0;
                let ms = median((0..5).map(|_| { let t = Instant::now(); let (_, f) = m.bolt.predict(c, h); forwards = f; t.elapsed().as_secs_f64() * 1e3 }).collect());
                println!("| {ctx_len} | {h} | {forwards} | {ms:.1} |");
            }
        }
    }
}

/// Spawn this binary as a stdio MCP server and time initialize → first forecast (cold start as a
/// Lambda would see it, minus the container: process exec + embedded-weight decode + first call).
fn coldstart(runs: usize) {
    let exe = std::env::current_exe().expect("exe");
    let (ds, y) = chronos_mcp::load_csv("fixtures/peyton_manning.csv").expect("peyton");
    let call = serde_json::json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "forecast", "arguments": {"ds": ds, "y": y, "horizon": 64}}}).to_string();
    println!("| run | exec → initialize response ms | → forecast response ms |\n|---|---|---|");
    for run in 0..runs {
        let t0 = Instant::now();
        let mut child = std::process::Command::new(&exe).stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().expect("spawn");
        let mut stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let mut lines = std::io::BufReader::new(stdout).lines();
        writeln!(stdin, r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"2025-06-18","capabilities":{{}},"clientInfo":{{"name":"coldstart","version":"0"}}}}}}"#).expect("write");
        let init = lines.next().expect("init line").expect("init io");
        let t_init = t0.elapsed().as_secs_f64() * 1e3;
        assert!(init.contains("serverInfo"), "unexpected initialize reply: {init}");
        writeln!(stdin, r#"{{"jsonrpc":"2.0","method":"notifications/initialized"}}"#).expect("write");
        writeln!(stdin, "{call}").expect("write");
        let resp = lines.next().expect("call line").expect("call io");
        let t_call = t0.elapsed().as_secs_f64() * 1e3;
        let v: serde_json::Value = serde_json::from_str(&resp).expect("json");
        assert!(v.get("error").is_none(), "forecast error: {resp}");
        drop(stdin);
        let _ = child.kill(); let _ = child.wait();
        println!("| {} | {t_init:.0} | {t_call:.0} |", run + 1);
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--bench") => { let dirs: Vec<String> = if args.len() > 1 { args[1..].to_vec() } else { vec!["models/tiny".into(), "models/tiny-f16".into(), "models/small-f16".into()] }; bench(&dirs); ExitCode::SUCCESS }
        Some("--coldstart") => { coldstart(args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3)); ExitCode::SUCCESS }
        Some("--http") => {
            let port: u16 = args.get(1).and_then(|p| p.parse().ok()).unwrap_or(8766);
            let model = match chronos_mcp::resolve_model() { Ok(m) => Arc::new(m), Err(e) => { eprintln!("error: {e}"); return ExitCode::FAILURE; } };
            eprintln!("{SERVER_NAME}: {} ({} params, {}, {}) loaded in {:.0} ms", model.name, model.n_params, model.dtype, model.source, model.load_seconds * 1e3);
            let server = match chronos_mcp::build_server(model, SERVER_NAME, env!("CARGO_PKG_VERSION")) { Ok(s) => s, Err(e) => { eprintln!("error: {e}"); return ExitCode::FAILURE; } };
            let app = chronos_mcp::http_app(server);
            let listener = match tokio::net::TcpListener::bind(("127.0.0.1", port)).await { Ok(l) => l, Err(e) => { eprintln!("error: bind {port}: {e}"); return ExitCode::FAILURE; } };
            eprintln!("{SERVER_NAME}: demo page http://127.0.0.1:{port}/  — MCP streamable-http at http://127.0.0.1:{port}/mcp");
            if let Err(e) = axum::serve(listener, app).await { eprintln!("error: {e}"); return ExitCode::FAILURE; }
            ExitCode::SUCCESS
        }
        _ => {
            let t0 = Instant::now();
            let model = match chronos_mcp::resolve_model() { Ok(m) => Arc::new(m), Err(e) => { eprintln!("error: {e}"); return ExitCode::FAILURE; } };
            eprintln!("{SERVER_NAME}: {} ({} params, {}, {}) loaded in {:.0} ms; serving `{}` on stdio", model.name, model.n_params, model.dtype, model.source, t0.elapsed().as_secs_f64() * 1e3, chronos_mcp::TOOL_NAME);
            let server = match chronos_mcp::build_server(model, SERVER_NAME, env!("CARGO_PKG_VERSION")) { Ok(s) => s, Err(e) => { eprintln!("error: {e}"); return ExitCode::FAILURE; } };
            if let Err(e) = server.run_stdio().await { eprintln!("error: {e}"); return ExitCode::FAILURE; }
            ExitCode::SUCCESS
        }
    }
}
