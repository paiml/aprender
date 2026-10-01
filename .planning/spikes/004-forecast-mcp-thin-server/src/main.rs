//! forecast-mcp — stdio MCP server (default), `--http [port]` for the browser demo, `--bench`.
#![allow(clippy::disallowed_methods)]
use std::process::ExitCode;
use std::time::Instant;

const SERVER_NAME: &str = "aprender-forecast";

fn synth(n: usize, seed: u64) -> (Vec<String>, Vec<f64>) {
    let mut rng = forecast_mcp::prophet::Rng::new(seed);
    let start = forecast_mcp::prophet::days_from_civil(2010, 1, 1);
    let mut slope = 0.002;
    let (mut ds, mut y) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let mut level = 10.0;
    for i in 0..n {
        let day = start + i as i64;
        if i % 400 == 0 && i > 0 { slope += (rng.uniform() - 0.5) * 0.004; }
        level += slope;
        let f = day as f64;
        let v = level + 0.8 * (2.0 * std::f64::consts::PI * f / 365.25).sin() + 0.3 * (2.0 * std::f64::consts::PI * f / 7.0).cos() + 0.2 * rng.normal();
        ds.push(forecast_mcp::prophet::format_ymd(day)); y.push(v);
    }
    (ds, y)
}

fn bench(sizes: &[usize]) {
    println!("| points | model | fit s | predict s | total s | L-BFGS rounds/iters/evals |\n|---|---|---|---|---|---|");
    for &n in sizes {
        let (ds, y) = synth(n, 3);
        for (model, n_lags) in [("prophet", 0usize), ("neuralprophet", 0), ("neuralprophet", 30)] {
            if model == "neuralprophet" && n > 10_000 { continue; }
            let args = forecast_mcp::ForecastArgs { ds: ds.clone(), y: y.clone(), horizon: 365, freq: None, model: Some(model.into()), growth: None, cap: None, seasonality_mode: None, interval_width: None, holidays: None, n_lags: Some(n_lags), seed: None };
            let t0 = Instant::now();
            let r = forecast_mcp::forecast(&args).expect("forecast");
            let total = t0.elapsed().as_secs_f64();
            let lb = r.diagnostics.get("lbfgs").map(|v| format!("{}/{}/{}", v["rounds"], v["iterations"], v["evaluations"])).unwrap_or_else(|| format!("epochs {} steps {}", r.diagnostics["epochs"], r.diagnostics["steps"]));
            println!("| {n} | {model}{} | {:.3} | {:.3} | {total:.3} | {lb} |", if n_lags > 0 { format!(" n_lags={n_lags}") } else { String::new() }, r.fit_seconds, r.predict_seconds);
        }
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--bench") => { let sizes: Vec<usize> = args[1..].iter().filter_map(|a| a.parse().ok()).collect(); bench(if sizes.is_empty() { &[1_000, 3_000, 10_000, 20_000] } else { &sizes }); ExitCode::SUCCESS }
        Some("--http") => {
            let port: u16 = args.get(1).and_then(|p| p.parse().ok()).unwrap_or(8765);
            let server = match forecast_mcp::build_server(SERVER_NAME, env!("CARGO_PKG_VERSION")) { Ok(s) => s, Err(e) => { eprintln!("error: {e}"); return ExitCode::FAILURE; } };
            let app = forecast_mcp::http_app(server);
            let listener = match tokio::net::TcpListener::bind(("127.0.0.1", port)).await { Ok(l) => l, Err(e) => { eprintln!("error: bind {port}: {e}"); return ExitCode::FAILURE; } };
            eprintln!("{SERVER_NAME}: demo page http://127.0.0.1:{port}/  — MCP streamable-http at http://127.0.0.1:{port}/mcp");
            if let Err(e) = axum::serve(listener, app).await { eprintln!("error: {e}"); return ExitCode::FAILURE; }
            ExitCode::SUCCESS
        }
        _ => {
            let server = match forecast_mcp::build_server(SERVER_NAME, env!("CARGO_PKG_VERSION")) { Ok(s) => s, Err(e) => { eprintln!("error: {e}"); return ExitCode::FAILURE; } };
            eprintln!("{SERVER_NAME}: serving `{}` on stdio", forecast_mcp::TOOL_NAME);
            if let Err(e) = server.run_stdio().await { eprintln!("error: {e}"); return ExitCode::FAILURE; }
            ExitCode::SUCCESS
        }
    }
}
