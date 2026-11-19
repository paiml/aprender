//! Real-Chrome probar run for the Video widget (aprender#7261): play and pause a Video in a
//! grid cell and in a modal, reading `paused` and `currentTime` back from the page.
//!
//! Needs a built page, so it is `#[ignore]`d for plain `cargo test`. Run:
//!   cargo build --example video_e2e --target wasm32-unknown-unknown
//!   wasm-bindgen --target web --out-dir "$VIDEO_E2E_PKG" target/wasm32-unknown-unknown/debug/examples/video_e2e.wasm
//!   VIDEO_E2E_PKG=... cargo test -p aprender-present-lib --test video_e2e_chrome -- --ignored
//! `CHROME` may name the browser binary.
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
            let path = req.split_whitespace().nth(1).unwrap_or("/");
            let name = path.trim_start_matches('/').split('?').next().unwrap_or("");
            let name = if name.is_empty() { "index.html" } else { name };
            let body = (!name.contains(".."))
                .then(|| dirs.iter().find_map(|d| std::fs::read(d.join(name)).ok()))
                .flatten();
            let ctype = match name.rsplit('.').next() {
                Some("html") => "text/html",
                Some("js") => "text/javascript",
                Some("wasm") => "application/wasm",
                Some("webm") => "video/webm",
                _ => "application/octet-stream",
            };
            let (status, body) = body.map_or(("404 Not Found", Vec::new()), |b| ("200 OK", b));
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

async fn state(d: &ChromiumDriver, id: &str) -> Value {
    let raw = js(d, &format!("e2e_state('{id}')")).await;
    serde_json::from_str(raw.as_str().expect("state string")).expect("state json")
}

async fn until(d: &ChromiumDriver, id: &str, want_paused: bool) -> Value {
    let end = Instant::now() + Duration::from_secs(10);
    loop {
        let s = state(d, id).await;
        if s["paused"] == want_paused && (want_paused || s["time"].as_f64() > Some(0.3)) {
            return s;
        }
        assert!(
            Instant::now() < end,
            "{id}: never reached paused={want_paused}, last {s}"
        );
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
    let end = Instant::now() + Duration::from_secs(20);
    while js(&d, "window.ready === true").await != Value::Bool(true) {
        assert!(Instant::now() < end, "page never became ready");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Three elements (newest first in the DOM): two side by side in the grid's row, one for the modal.
    let rects = js(&d, "JSON.stringify([...document.querySelectorAll('video')].map(v => { const r = v.getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; }))").await;
    let mut rects: Vec<[f64; 4]> =
        serde_json::from_str(rects.as_str().expect("rects")).expect("rects json");
    assert_eq!(rects.len(), 3, "{rects:?}");
    rects.sort_by(|a, b| (a[1], a[0]).partial_cmp(&(b[1], b[0])).expect("finite"));
    let [cell0, cell1, modal] = [rects[0], rects[1], rects[2]];
    assert!(
        cell0[1] == cell1[1] && cell1[0] >= cell0[0] + cell0[2],
        "grid cells: {rects:?}"
    );
    assert!(
        modal[1] >= cell0[1] + cell0[3] && modal[2] > 0.0,
        "modal below the grid row: {rects:?}"
    );

    for id in ["grid-0", "grid-1", "modal"] {
        assert_eq!(state(&d, id).await["paused"], true, "{id} starts paused");
        js(&d, &format!("e2e_toggle('{id}')")).await;
        let playing = until(&d, id, false).await;
        assert!(
            playing["time"].as_f64() > Some(0.3),
            "{id} clock runs: {playing}"
        );
        js(&d, &format!("e2e_toggle('{id}')")).await;
        let paused = until(&d, id, true).await;
        tokio::time::sleep(Duration::from_millis(400)).await;
        let later = state(&d, id).await;
        assert_eq!(
            later["time"], paused["time"],
            "{id} clock frozen while paused"
        );
    }

    // Closing the modal removes its element; reopening brings back one, paused.
    js(&d, "e2e_modal(false)").await;
    assert_eq!(js(&d, "document.querySelectorAll('video').length").await, 2);
    assert_eq!(js(&d, "e2e_state('modal')").await, "null");
    js(&d, "e2e_modal(true)").await;
    assert_eq!(js(&d, "document.querySelectorAll('video').length").await, 3);
}
