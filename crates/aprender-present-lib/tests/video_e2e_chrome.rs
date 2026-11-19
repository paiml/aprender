//! Real-Chrome probar run for the Video widget (aprender#7261): play and pause a Video in a
//! grid cell and in a modal, reading `paused` and `currentTime` back from the page.
//!
//! `#[ignore]`d for plain `cargo test`: it needs the page built with
//! `cargo build --example video_e2e --target wasm32-unknown-unknown` and `wasm-bindgen --target web
//! --out-dir $VIDEO_E2E_PKG <that .wasm>`; then `-- --ignored`. `CHROME` may name the browser.
#![cfg(not(target_arch = "wasm32"))]

use jugar_probar::{ChromiumDriver, DriverConfig, ProbarDriver};
use serde_json::Value;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn serve(dirs: Vec<PathBuf>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    std::thread::spawn(move || {
        for mut s in listener.incoming().flatten() {
            let mut buf = [0u8; 2048];
            let n = s.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]).into_owned();
            let name = req
                .split_whitespace()
                .nth(1)
                .unwrap_or("/")
                .trim_matches('/');
            let name = if name.is_empty() { "index.html" } else { name };
            let body = dirs.iter().find_map(|d| {
                std::fs::read(d.join(name))
                    .ok()
                    .filter(|_| !name.contains(".."))
            });
            let ctype = match name.rsplit('.').next() {
                Some("html") => "text/html",
                Some("js") => "text/javascript",
                _ => "application/octet-stream",
            };
            let (status, body) = body.map_or(("404 Not Found", vec![]), |b| ("200 OK", b));
            let head = format!("HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
            let _ = s
                .write_all(head.as_bytes())
                .and_then(|()| s.write_all(&body));
        }
    });
    port
}

async fn js(d: &ChromiumDriver, script: &str) -> Value {
    d.execute_js(script)
        .await
        .unwrap_or_else(|e| panic!("js {script}: {e}"))
}

/// Re-evaluate `expr` until it is true; panic with it after ten seconds.
async fn poll(d: &ChromiumDriver, expr: &str) {
    let end = Instant::now() + Duration::from_secs(10);
    while js(d, expr).await != Value::Bool(true) {
        assert!(Instant::now() < end, "never true: {expr}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test]
#[ignore = "needs the built page; see the file header"]
async fn video_plays_and_pauses_in_grid_cell_and_modal() {
    let pkg = PathBuf::from(std::env::var("VIDEO_E2E_PKG").expect("VIDEO_E2E_PKG"));
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/video_e2e");
    let port = serve(vec![pkg, here]);
    let mut cfg = DriverConfig::new().viewport(800, 600);
    cfg.executable_path = std::env::var("CHROME").ok();
    let mut d = ChromiumDriver::launch(cfg).await.expect("launch Chrome");
    d.navigate(&format!("http://127.0.0.1:{port}/"))
        .await
        .expect("navigate");
    poll(&d, "window.ready === true").await;

    // Two elements side by side in the grid's row, one for the modal below it (index.html).
    poll(&d, "layoutOk()").await;

    for id in ["grid-0", "grid-1", "modal"] {
        let st = format!("JSON.parse(e2e.state('{id}'))");
        poll(&d, &format!("{st}.paused === true")).await;
        js(&d, &format!("e2e.toggle('{id}')")).await;
        poll(&d, &format!("{st}.paused === false && {st}.time > 0.3")).await;
        js(&d, &format!("e2e.toggle('{id}')")).await;
        poll(&d, &format!("{st}.paused === true")).await;
        js(&d, &format!("window.t0 = {st}.time")).await;
        tokio::time::sleep(Duration::from_millis(400)).await;
        poll(&d, &format!("{st}.time === window.t0")).await; // frozen while paused
    }
}
