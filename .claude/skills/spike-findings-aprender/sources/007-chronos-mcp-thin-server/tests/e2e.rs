//! The thin Chronos server over live streamable-HTTP MCP, in-process, with the Python oracle from
//! spike 005 as the parity bar and the f16 weights measured against the f32 ones.
#![allow(clippy::disallowed_methods)]
use std::sync::Arc;
use std::time::Instant;

struct Client { http: reqwest::Client, url: String, next: u64 }
impl Client {
    async fn call(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        self.next += 1;
        let body = serde_json::json!({"jsonrpc": "2.0", "id": self.next, "method": method, "params": params});
        let r = self.http.post(&self.url).header("content-type", "application/json").header("accept", "application/json, text/event-stream").body(body.to_string()).send().await.expect("send");
        let status = r.status();
        let text = r.text().await.expect("text");
        assert!(status.is_success(), "{method}: HTTP {status}\n{text}");
        let payload = text.lines().find_map(|l| l.strip_prefix("data: ")).unwrap_or(&text);
        serde_json::from_str(payload).unwrap_or_else(|e| panic!("{method}: non-JSON: {e}\n{text}"))
    }
}

fn tool_output(v: &serde_json::Value) -> serde_json::Value {
    if let Some(s) = v["result"].get("structuredContent") { return s.clone(); }
    let text = v["result"]["content"][0]["text"].as_str().expect("text content");
    serde_json::from_str(text).expect("tool JSON")
}

fn f64s(v: &serde_json::Value) -> Vec<f64> { v.as_array().expect("array").iter().map(|x| x.as_f64().expect("f64")).collect() }

async fn serve(model_dir: &str) -> (Client, String) {
    let model = Arc::new(chronos_mcp::load_model_from_dir(std::path::Path::new(model_dir)).expect("model"));
    let server = chronos_mcp::build_server(model, "aprender-chronos-test", "0.0.0").expect("server");
    let app = chronos_mcp::http_app(server);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move { axum::serve(listener, app).await.expect("serve"); });
    (Client { http: reqwest::Client::new(), url: format!("http://{addr}/mcp"), next: 0 }, format!("http://{addr}"))
}

#[tokio::test]
async fn forecast_tool_over_streamable_http() {
    let (mut c, base) = serve("models/tiny").await;
    let page = reqwest::get(format!("{base}/")).await.expect("page").text().await.expect("html");
    assert!(page.contains("tools/call") && page.contains("forecast"), "demo page must be an MCP client");

    let init = c.call("initialize", serde_json::json!({"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "e2e", "version": "0"}})).await;
    assert!(init.get("error").is_none(), "initialize: {init}");
    let tools = c.call("tools/list", serde_json::json!({})).await;
    let names: Vec<&str> = tools["result"]["tools"].as_array().expect("tools").iter().map(|t| t["name"].as_str().expect("name")).collect();
    assert_eq!(names, vec!["forecast"], "exactly one tool");
    let schema = &tools["result"]["tools"][0]["inputSchema"];
    assert!(schema["properties"].get("allow_long_horizon").is_some() && schema["required"].as_array().expect("req").iter().any(|v| v == "ds"), "schema: {schema}");

    // Peyton, 64 ahead — parity with the Python oracle through the whole server
    let (ds, y) = chronos_mcp::load_csv("fixtures/peyton_manning.csv").expect("csv");
    let oracle: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("fixtures/peyton_tiny_oracle.json").expect("oracle")).expect("json");
    let t0 = Instant::now();
    let r = c.call("tools/call", serde_json::json!({"name": "forecast", "arguments": {"ds": ds, "y": y, "horizon": 64}})).await;
    let rt = t0.elapsed().as_secs_f64();
    assert!(r.get("error").is_none() && r["result"]["isError"] != true, "forecast failed: {r}");
    let out = tool_output(&r);
    assert_eq!(out["ds"].as_array().expect("ds").len(), 64);
    assert_eq!(out["ds"][0], "2016-01-21");
    assert_eq!(out["context_used"], 2048);
    assert!(out.get("warning").is_none(), "no warning at the native horizon: {out}");
    let mut worst = 0.0f64;
    for (qi, lvl) in ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"].iter().enumerate() {
        let ours = f64s(&out["quantiles"][*lvl]);
        let theirs = f64s(&oracle["quantiles_64"][qi]);
        for (a, b) in ours.iter().zip(&theirs) { worst = worst.max((a - b).abs()); }
    }
    println!("PARITY tiny f32, 64 steps: max |Δ| vs Python oracle = {worst:.2e}; round trip {rt:.3}s; predict {:.4}s; {}", out["predict_seconds"], out["diagnostics"]);
    assert!(worst < 5e-5, "quantiles must match the spike-005 oracle (f32 rounding): {worst}");
    assert!(f64s(&out["yhat"]) == f64s(&out["quantiles"]["0.5"]) && f64s(&out["yhat_lower"]) == f64s(&out["quantiles"]["0.1"]), "yhat is the median, band is q10/q90");
    assert!(rt < 0.5, "round trip {rt}s (bar: 0.1 s in release; debug test builds are slower)");

    // horizon beyond 64 without the flag → refused with the reason; with it → rollout, warning, oracle parity
    let r = c.call("tools/call", serde_json::json!({"name": "forecast", "arguments": {"ds": ds, "y": y, "horizon": 365}})).await;
    assert!(r.to_string().contains("allow_long_horizon"), "long horizon must be refused by default: {r}");
    let r = c.call("tools/call", serde_json::json!({"name": "forecast", "arguments": {"ds": ds, "y": y, "horizon": 365, "allow_long_horizon": true}})).await;
    let out = tool_output(&r);
    assert_eq!(out["ds"].as_array().expect("ds").len(), 365);
    assert!(out["warning"].as_str().expect("warning").contains("rollout"), "{out}");
    assert_eq!(out["diagnostics"]["forwards"], 46);
    let mut worst = 0.0f64;
    for (qi, lvl) in ["0.1", "0.5", "0.9"].iter().zip([0usize, 4, 8]) {
        for (a, b) in f64s(&out["quantiles"][*qi]).iter().zip(f64s(&oracle["quantiles_365"][lvl])) { worst = worst.max((a - b).abs()); }
    }
    println!("PARITY tiny f32, 365 steps (46 forwards): max |Δ| = {worst:.2e}; predict {:.3}s", out["predict_seconds"]);
    assert!(worst < 1e-4, "rollout parity: {worst}");

    // missing values: null in y is accepted and reported
    let mut y_gap = y.clone(); for i in (100..2900).step_by(37) { y_gap[i] = None; }
    let r = c.call("tools/call", serde_json::json!({"name": "forecast", "arguments": {"ds": ds, "y": y_gap, "horizon": 12}})).await;
    let out = tool_output(&r);
    assert_eq!(out["diagnostics"]["missing_values"], 76, "{}", out["diagnostics"]);
    assert!(f64s(&out["yhat"]).iter().all(|v| v.is_finite()));

    // refusals: unknown field, too few points, bad date, horizon 0, all-null y
    let r = c.call("tools/call", serde_json::json!({"name": "forecast", "arguments": {"ds": ds, "y": y, "horizon": 10, "model": "prophet"}})).await;
    assert!(r.get("error").is_some() || r["result"]["isError"] == true, "unknown field must be refused: {r}");
    let r = c.call("tools/call", serde_json::json!({"name": "forecast", "arguments": {"ds": ds[..3], "y": y[..3], "horizon": 10}})).await;
    assert!(r.to_string().contains("at least"), "{r}");
    let mut bad = ds.clone(); bad[3] = "2008-02-30".into();
    let r = c.call("tools/call", serde_json::json!({"name": "forecast", "arguments": {"ds": bad, "y": y, "horizon": 10}})).await;
    assert!(r.to_string().contains("calendar"), "{r}");
    let r = c.call("tools/call", serde_json::json!({"name": "forecast", "arguments": {"ds": ds, "y": y, "horizon": 0}})).await;
    assert!(r.to_string().contains("at least 1"), "{r}");
    let nulls: Vec<Option<f64>> = vec![None; ds.len()];
    let r = c.call("tools/call", serde_json::json!({"name": "forecast", "arguments": {"ds": ds, "y": nulls, "horizon": 10}})).await;
    assert!(r.to_string().contains("non-null"), "{r}");

    // monthly series, MS future dates
    let (ads, ay) = chronos_mcp::load_csv("fixtures/air_passengers.csv").expect("csv");
    let r = c.call("tools/call", serde_json::json!({"name": "forecast", "arguments": {"ds": ads, "y": ay, "horizon": 24, "freq": "MS"}})).await;
    let out = tool_output(&r);
    assert_eq!(out["ds"][0], "1961-01-01"); assert_eq!(out["ds"][23], "1962-12-01");
    assert_eq!(out["context_used"], 144);
    let (lo, hi) = (f64s(&out["yhat_lower"]), f64s(&out["yhat_upper"]));
    assert!(lo.iter().zip(&hi).all(|(l, h)| l <= h));

    // short series (below one patch)
    let r = c.call("tools/call", serde_json::json!({"name": "forecast", "arguments": {"ds": ds[..7], "y": y[..7], "horizon": 5}})).await;
    let out = tool_output(&r);
    assert_eq!(out["ds"].as_array().expect("ds").len(), 5);
}

/// The f16 weights halve the embedded size; this measures what they cost in output.
#[tokio::test]
async fn f16_weights_vs_f32() {
    let (ds, y) = chronos_mcp::load_csv("fixtures/peyton_manning.csv").expect("csv");
    let args = chronos_mcp::ForecastArgs { ds, y, horizon: 64, freq: None, allow_long_horizon: false };
    let f32m = chronos_mcp::load_model_from_dir(std::path::Path::new("models/tiny")).expect("f32");
    let f16m = chronos_mcp::load_model_from_dir(std::path::Path::new("models/tiny-f16")).expect("f16");
    assert_eq!(f16m.dtype, "F16"); assert_eq!(f32m.dtype, "F32");
    let a = chronos_mcp::forecast(&f32m, &args).expect("f32 forecast");
    let b = chronos_mcp::forecast(&f16m, &args).expect("f16 forecast");
    let scale = 0.8718; // series std (oracle loc/scale)
    let mut worst = 0.0f64;
    for (lvl, va) in &a.quantiles { let vb = &b.quantiles[lvl]; for (x, y) in va.as_array().unwrap().iter().zip(vb.as_array().unwrap()) { worst = worst.max((x.as_f64().unwrap() - y.as_f64().unwrap()).abs()); } }
    println!("F16 vs F32 (tiny, Peyton, 64 steps): max |Δ| = {worst:.2e} = {:.3}% of the series std; load f32 {:.0} ms, f16 {:.0} ms", worst / scale * 100.0, f32m.load_seconds * 1e3, f16m.load_seconds * 1e3);
    assert!(worst / scale < 0.02, "f16 weights moved a quantile by more than 2 % of the series std: {worst}");
}
