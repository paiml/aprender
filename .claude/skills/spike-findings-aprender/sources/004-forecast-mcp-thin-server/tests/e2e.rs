//! The thin forecast server over live streamable-HTTP MCP, in-process.
#![allow(clippy::disallowed_methods)]
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

fn csv(path: &str) -> (Vec<String>, Vec<f64>) {
    let s = std::fs::read_to_string(path).expect("csv");
    let mut ds = Vec::new(); let mut y = Vec::new();
    for line in s.lines().skip(1) { let mut it = line.split(','); ds.push(it.next().unwrap().trim_matches('"').to_string()); y.push(it.next().unwrap().trim_matches('"').parse().unwrap()); }
    (ds, y)
}

#[tokio::test]
async fn forecast_tool_over_streamable_http() {
    let server = forecast_mcp::build_server("aprender-forecast-test", "0.0.0").expect("server");
    let app = forecast_mcp::http_app(server);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move { axum::serve(listener, app).await.expect("serve"); });
    let mut c = Client { http: reqwest::Client::new(), url: format!("http://{addr}/mcp"), next: 0 };

    // page + samples are same-origin
    let page = reqwest::get(format!("http://{addr}/")).await.expect("page").text().await.expect("html");
    assert!(page.contains("tools/call") && page.contains("forecast"), "demo page must be an MCP client");

    let init = c.call("initialize", serde_json::json!({"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "e2e", "version": "0"}})).await;
    assert!(init.get("error").is_none(), "initialize: {init}");
    let tools = c.call("tools/list", serde_json::json!({})).await;
    let names: Vec<&str> = tools["result"]["tools"].as_array().expect("tools").iter().map(|t| t["name"].as_str().expect("name")).collect();
    assert_eq!(names, vec!["forecast"], "exactly one tool");
    let schema = &tools["result"]["tools"][0]["inputSchema"];
    assert!(schema["properties"].get("horizon").is_some() && schema["required"].as_array().expect("req").iter().any(|v| v == "ds"), "schema advertises ds/y/horizon: {schema}");

    // Peyton, 365 days ahead
    let (ds, y) = csv("fixtures/peyton_manning.csv");
    let t0 = Instant::now();
    let r = c.call("tools/call", serde_json::json!({"name": "forecast", "arguments": {"ds": ds, "y": y, "horizon": 365}})).await;
    let rt = t0.elapsed().as_secs_f64();
    assert!(r.get("error").is_none() && r["result"]["isError"] != true, "forecast failed: {r}");
    let out = tool_output(&r);
    assert_eq!(out["ds"].as_array().expect("ds").len(), 365);
    assert_eq!(out["ds"][0], "2016-01-21");
    let (yl, yh, yu) = (out["yhat_lower"].as_array().unwrap(), out["yhat"].as_array().unwrap(), out["yhat_upper"].as_array().unwrap());
    assert!(yl.iter().zip(yh).zip(yu).all(|((l, h), u)| l.as_f64() <= h.as_f64() && h.as_f64() <= u.as_f64()), "band must bracket yhat");
    assert!(out["components"].get("yearly").is_some() && out["components"].get("weekly").is_some());
    println!("PEYTON 2905 pts → 365 ahead: fit {:.3}s predict {:.3}s round-trip {rt:.3}s; diagnostics {}", out["fit_seconds"].as_f64().unwrap(), out["predict_seconds"].as_f64().unwrap(), out["diagnostics"]);
    assert!(rt < 10.0, "round trip {rt}s");

    // unknown field → refused, not ignored
    let r = c.call("tools/call", serde_json::json!({"name": "forecast", "arguments": {"ds": ds, "y": y, "horizon": 10, "temperature": 0.7}})).await;
    assert!(r.get("error").is_some() || r["result"]["isError"] == true, "unknown field must be refused: {r}");
    // too few points
    let r = c.call("tools/call", serde_json::json!({"name": "forecast", "arguments": {"ds": ds[..5], "y": y[..5], "horizon": 10}})).await;
    let msg = r.to_string();
    assert!(msg.contains("at least"), "too-few-points must be a validation error: {r}");
    // bad date
    let mut bad = ds.clone(); bad[3] = "2008-02-30".into();
    let r = c.call("tools/call", serde_json::json!({"name": "forecast", "arguments": {"ds": bad, "y": y, "horizon": 10}})).await;
    assert!(r.to_string().contains("calendar"), "impossible date must be refused: {r}");

    // monthly, multiplicative, MS future
    let (ads, ay) = csv("fixtures/air_passengers.csv");
    let r = c.call("tools/call", serde_json::json!({"name": "forecast", "arguments": {"ds": ads, "y": ay, "horizon": 24, "freq": "MS", "seasonality_mode": "multiplicative"}})).await;
    let out = tool_output(&r);
    assert_eq!(out["ds"][0], "1961-01-01"); assert_eq!(out["ds"][23], "1962-12-01");
    assert!(out["components"].get("multiplicative_terms").is_some());

    // logistic with holidays
    let r = c.call("tools/call", serde_json::json!({"name": "forecast", "arguments": {"ds": ds, "y": y, "horizon": 30, "growth": "logistic", "cap": 13.0, "holidays": [{"name": "superbowl", "dates": ["2010-02-07", "2014-02-02", "2016-02-07"], "upper_window": 1}]}})).await;
    let out = tool_output(&r);
    assert!(out["components"].get("superbowl").is_some() && out["components"].get("holidays").is_some(), "{out}");

    // neuralprophet, with and without lags
    for n_lags in [0, 30] {
        let t0 = Instant::now();
        let r = c.call("tools/call", serde_json::json!({"name": "forecast", "arguments": {"ds": ds, "y": y, "horizon": 60, "model": "neuralprophet", "n_lags": n_lags}})).await;
        let out = tool_output(&r);
        assert_eq!(out["ds"].as_array().expect("ds").len(), 60, "{r}");
        println!("NEURALPROPHET n_lags={n_lags}: fit {:.3}s round-trip {:.3}s; {}", out["fit_seconds"].as_f64().unwrap(), t0.elapsed().as_secs_f64(), out["diagnostics"]);
    }
}
