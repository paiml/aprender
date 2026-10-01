//! Spike 010: the spike-004 Prophet/NeuralProphet server under concurrent requests. Eight fits at
//! once over live streamable-HTTP must each equal their sequential result bit for bit (the autograd
//! tape is thread-local and `spawn_blocking` threads are reused), repeatedly, and the wall time must
//! show the fits actually ran in parallel.
//! Run from the spike directory: CARGO_TARGET_DIR=../../../target cargo run --release
#![allow(clippy::disallowed_methods)]
use std::time::Instant;

struct Client { http: reqwest::Client, url: String }
impl Client {
    async fn call(&self, id: u64, method: &str, params: serde_json::Value) -> serde_json::Value {
        let body = serde_json::json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let r = self.http.post(&self.url).header("content-type", "application/json").header("accept", "application/json, text/event-stream").body(body.to_string()).send().await.expect("send");
        let text = r.text().await.expect("text");
        let payload = text.lines().find_map(|l| l.strip_prefix("data: ")).unwrap_or(&text);
        serde_json::from_str(payload).unwrap_or_else(|e| panic!("{method}: non-JSON: {e}\n{text}"))
    }
}
fn tool_output(v: &serde_json::Value) -> serde_json::Value {
    if let Some(s) = v["result"].get("structuredContent") { return s.clone(); }
    let text = v["result"]["content"][0]["text"].as_str().unwrap_or_else(|| panic!("no text content: {v}"));
    serde_json::from_str(text).expect("tool JSON")
}
fn csv(path: &str) -> (Vec<String>, Vec<f64>) {
    let s = std::fs::read_to_string(path).expect("csv");
    let mut rows: Vec<(String, f64)> = s.lines().skip(1).filter_map(|l| { let mut it = l.split(','); let d = it.next()?.trim_matches('"').get(..10)?.to_string(); let v = it.next()?.trim_matches('"').parse().ok()?; Some((d, v)) }).collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0)); rows.dedup_by(|l, e| { if l.0 == e.0 { *e = l.clone(); true } else { false } });
    rows.into_iter().unzip()
}
/// Deterministic synthetic daily series (trend + yearly + weekly + noise), different per seed.
fn synth(n: usize, seed: u64) -> (Vec<String>, Vec<f64>) {
    let mut rng = forecast_mcp::prophet::Rng::new(seed);
    let start = forecast_mcp::prophet::days_from_civil(2012, 1, 1);
    let (mut ds, mut y) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let mut level = 10.0 + seed as f64;
    for i in 0..n {
        let day = start + i as i64; level += 0.002; let f = day as f64;
        y.push(level + 0.8 * (2.0 * std::f64::consts::PI * f / 365.25).sin() + 0.3 * (2.0 * std::f64::consts::PI * f / 7.0).cos() + 0.2 * rng.normal());
        ds.push(forecast_mcp::prophet::format_ymd(day));
    }
    (ds, y)
}
/// The fields whose bits must not depend on what else the server was doing.
fn signature(out: &serde_json::Value) -> String {
    serde_json::json!({"ds": out["ds"], "yhat": out["yhat"], "lo": out["yhat_lower"], "hi": out["yhat_upper"], "trend": out["trend"], "components": out["components"]}).to_string()
}
fn max_abs_diff(a: &serde_json::Value, b: &serde_json::Value) -> f64 {
    let f = |v: &serde_json::Value| v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect::<Vec<_>>();
    f(&a["yhat"]).iter().zip(f(&b["yhat"])).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// `pool = 1`: the spike-004 app as shipped (one `Arc<Mutex<Server>>` behind pmcp's router).
/// `pool > 1`: K independent pmcp routers, each over its own `Server`, with a round-robin front
/// handler — K tool calls can then be in flight at once.
fn app(pool: usize) -> axum::Router {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::ServiceExt;
    if pool <= 1 { return forecast_mcp::http_app(forecast_mcp::build_server("aprender-forecast-concurrency", "0.0.0").expect("server")); }
    let routers: std::sync::Arc<Vec<axum::Router>> = std::sync::Arc::new((0..pool).map(|_| forecast_mcp::http_app(forecast_mcp::build_server("aprender-forecast-concurrency", "0.0.0").expect("server"))).collect());
    let next = std::sync::Arc::new(AtomicUsize::new(0));
    axum::Router::new().fallback(move |req: axum::extract::Request| {
        let routers = routers.clone(); let next = next.clone();
        async move {
            let i = next.fetch_add(1, Ordering::Relaxed) % routers.len();
            routers[i].clone().oneshot(req).await.unwrap_or_else(|e| match e {})
        }
    })
}

async fn run(pool: usize) {
    let app = app(pool);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move { axum::serve(listener, app).await.expect("serve"); });
    let c = std::sync::Arc::new(Client { http: reqwest::Client::new(), url: format!("http://{addr}/mcp") });
    let init = c.call(1, "initialize", serde_json::json!({"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "spike-010", "version": "0"}})).await;
    assert!(init.get("error").is_none(), "{init}");

    let (pds, py) = csv("fixtures/peyton_manning.csv");
    let (ads, ay) = csv("fixtures/air_passengers.csv");
    let (s1d, s1y) = synth(3000, 1);
    let (s2d, s2y) = synth(3000, 2);
    // eight distinct requests: four series × {prophet, neuralprophet n_lags=30}
    let mut reqs: Vec<(String, serde_json::Value)> = Vec::new();
    for (name, ds, y, freq) in [("peyton", &pds, &py, "D"), ("air", &ads, &ay, "MS"), ("synth1", &s1d, &s1y, "D"), ("synth2", &s2d, &s2y, "D")] {
        reqs.push((format!("{name}/prophet"), serde_json::json!({"ds": ds, "y": y, "horizon": 30, "freq": freq, "model": "prophet"})));
        if freq == "D" { reqs.push((format!("{name}/neuralprophet-lags30"), serde_json::json!({"ds": ds, "y": y, "horizon": 30, "model": "neuralprophet", "n_lags": 30}))); }
        else { reqs.push((format!("{name}/prophet-multiplicative"), serde_json::json!({"ds": ds, "y": y, "horizon": 12, "freq": freq, "seasonality_mode": "multiplicative"}))); }
    }
    println!("\n# Configuration: pool = {pool} ({})\n", if pool <= 1 { "spike-004 as shipped: one `Arc<Mutex<Server>>` behind pmcp's router" } else { "K independent pmcp routers, round-robin front handler" });
    println!("Server in-process, streamable-HTTP, {} requests: {}\n", reqs.len(), reqs.iter().map(|r| r.0.as_str()).collect::<Vec<_>>().join(", "));

    // 1. sequential baseline
    let t0 = Instant::now();
    let mut base: Vec<(String, f64)> = Vec::new();
    for (i, (name, args)) in reqs.iter().enumerate() {
        let t = Instant::now();
        let r = c.call(100 + i as u64, "tools/call", serde_json::json!({"name": "forecast", "arguments": args})).await;
        assert!(r.get("error").is_none() && r["result"]["isError"] != true, "{name}: {r}");
        let out = tool_output(&r);
        base.push((signature(&out), t.elapsed().as_secs_f64()));
        let _ = name;
    }
    let seq_total = t0.elapsed().as_secs_f64();
    println!("## 1. Sequential baseline\n\n| request | seconds |\n|---|---|");
    for ((name, _), (_, s)) in reqs.iter().zip(&base) { println!("| {name} | {s:.2} |"); }
    println!("| **total** | **{seq_total:.2}** |");

    // 2. concurrent rounds: all requests at once, three times
    println!("\n## 2. Concurrent rounds (all {} at once)\n\n| round | wall s | speed-up vs sequential | responses identical to baseline | max |Δ yhat| |\n|---|---|---|---|---|", reqs.len());
    let mut all_identical = true;
    for round in 0..3 {
        let t0 = Instant::now();
        let handles: Vec<_> = reqs.iter().enumerate().map(|(i, (_, args))| { let c = c.clone(); let args = args.clone(); tokio::spawn(async move { let r = c.call(1000 + (round * 100 + i) as u64, "tools/call", serde_json::json!({"name": "forecast", "arguments": args})).await; tool_output(&r) }) }).collect();
        let mut outs = Vec::new();
        for h in handles { outs.push(h.await.expect("join")); }
        let wall = t0.elapsed().as_secs_f64();
        let identical = outs.iter().zip(&base).filter(|(o, (sig, _))| signature(o) == *sig).count();
        let worst = outs.iter().zip(&base).map(|(o, (sig, _))| { let b: serde_json::Value = serde_json::from_str(sig).unwrap(); max_abs_diff(o, &b) }).fold(0.0, f64::max);
        if identical != reqs.len() { all_identical = false; }
        println!("| {} | {wall:.2} | {:.1}× | {identical}/{} | {worst:.1e} |", round + 1, seq_total / wall, reqs.len());
    }

    // 3. stress: 16 simultaneous NeuralProphet fits with lags (the tape-heaviest path), two series alternating
    let stress: Vec<serde_json::Value> = (0..16).map(|i| { let (ds, y) = if i % 2 == 0 { (&s1d, &s1y) } else { (&s2d, &s2y) }; serde_json::json!({"ds": ds, "y": y, "horizon": 30, "model": "neuralprophet", "n_lags": 30}) }).collect();
    let expect: Vec<String> = { let mut v = Vec::new(); for i in 0..2 { let r = c.call(5000 + i, "tools/call", serde_json::json!({"name": "forecast", "arguments": stress[i as usize]})).await; v.push(signature(&tool_output(&r))); } v };
    let t0 = Instant::now();
    let handles: Vec<_> = stress.iter().enumerate().map(|(i, args)| { let c = c.clone(); let args = args.clone(); tokio::spawn(async move { let r = c.call(6000 + i as u64, "tools/call", serde_json::json!({"name": "forecast", "arguments": args})).await; tool_output(&r) }) }).collect();
    let mut ok = 0; let mut errors = 0;
    for (i, h) in handles.into_iter().enumerate() { let o = h.await.expect("join"); if o.get("yhat").is_none() { errors += 1; } else if signature(&o) == expect[i % 2] { ok += 1; } }
    let wall = t0.elapsed().as_secs_f64();
    println!("\n## 3. Stress: 16 simultaneous NeuralProphet n_lags=30 fits\n\n{ok}/16 identical to their sequential result, {errors} errors, wall {wall:.2} s (one such fit alone: {:.2} s)\n", base[1].1);
    println!("Correctness: {}", if all_identical && ok == 16 && errors == 0 { "every concurrent response equals its sequential result; the thread-local tape and per-request seeds hold under load" } else { "MISMATCH — see tables" });
}

#[tokio::main]
async fn main() {
    println!("# Spike 010 — spike-004 server under concurrent requests ({} cores)", std::thread::available_parallelism().map_or(0, |n| n.get()));
    let pools: Vec<usize> = { let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect(); if a.is_empty() { vec![1, 8] } else { a } };
    for pool in pools { run(pool).await; }
}
