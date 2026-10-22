//! #5718: `DevServer::run`, driven over real TCP on a free port.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use probador::dev_server::{DevServer, DevServerConfig, HotReloadMessage};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn free_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    l.local_addr().expect("addr").port()
}

/// Start `run` in the background and wait for its ServerReady broadcast.
async fn start(
    dir: &std::path::Path,
    cors: bool,
    coop: bool,
) -> (u16, tokio::task::JoinHandle<()>) {
    let port = free_port();
    let config = DevServerConfig::builder()
        .directory(dir)
        .port(port)
        .cors(cors)
        .cross_origin_isolated(coop)
        .build();
    let server = DevServer::new(config);
    let mut rx = server.reload_sender().subscribe();
    let handle = tokio::spawn(async move {
        let _ = server.run().await;
    });
    let msg = tokio::time::timeout(Duration::from_secs(10), rx.recv())
        .await
        .expect("ServerReady within 10s")
        .expect("channel open");
    assert!(matches!(msg, HotReloadMessage::ServerReady), "{msg:?}");
    (port, handle)
}

/// One GET with `Connection: close`; returns the raw response, lower-cased headers preserved as sent.
async fn get(port: u16, path: &str, extra: &str) -> String {
    let mut last = None;
    for _ in 0..50 {
        match tokio::net::TcpStream::connect(("127.0.0.1", port)).await {
            Ok(mut s) => {
                let req =
                    format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n{extra}\r\n");
                s.write_all(req.as_bytes()).await.expect("write");
                let mut buf = Vec::new();
                s.read_to_end(&mut buf).await.expect("read");
                return String::from_utf8_lossy(&buf).into_owned();
            }
            Err(e) => {
                last = Some(e);
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
    panic!("could not connect: {last:?}");
}

fn site() -> tempfile::TempDir {
    let d = tempfile::tempdir().expect("tempdir");
    std::fs::write(d.path().join("index.html"), "<h1>root-5718</h1>").expect("index");
    std::fs::write(d.path().join("app.wasm"), b"\0asm").expect("wasm");
    std::fs::create_dir(d.path().join("sub")).expect("sub");
    std::fs::write(d.path().join("sub/index.html"), "sub-index-5718").expect("sub index");
    std::fs::create_dir(d.path().join("bare")).expect("bare");
    d
}

#[tokio::test]
async fn run_serves_index_static_wasm_and_404s() {
    let d = site();
    let (port, h) = start(d.path(), false, false).await;

    let root = get(port, "/", "").await;
    assert!(root.starts_with("HTTP/1.1 200"), "{root}");
    assert!(root.contains("<h1>root-5718</h1>"), "{root}");

    let wasm = get(port, "/app.wasm", "").await;
    assert!(wasm.starts_with("HTTP/1.1 200"), "{wasm}");
    assert!(
        wasm.to_lowercase()
            .contains("content-type: application/wasm"),
        "{wasm}"
    );
    assert!(
        wasm.to_lowercase().contains("cache-control: no-cache"),
        "{wasm}"
    );

    let sub = get(port, "/sub/", "").await;
    assert!(sub.contains("sub-index-5718"), "{sub}");
    let sub_no_slash = get(port, "/sub", "").await;
    assert!(sub_no_slash.contains("sub-index-5718"), "{sub_no_slash}");

    let bare = get(port, "/bare", "").await;
    assert!(bare.starts_with("HTTP/1.1 404"), "{bare}");
    assert!(bare.contains("No index.html found in directory"), "{bare}");

    let missing = get(port, "/nope.js", "").await;
    assert!(missing.starts_with("HTTP/1.1 404"), "{missing}");
    assert!(missing.contains("File not found"), "{missing}");
    h.abort();
}

#[tokio::test]
async fn run_adds_cors_and_isolation_headers_only_when_enabled() {
    let d = site();
    let (port, h) = start(d.path(), true, true).await;
    let on = get(port, "/", "Origin: http://example.test\r\n")
        .await
        .to_lowercase();
    assert!(on.contains("access-control-allow-origin: *"), "{on}");
    assert!(
        on.contains("cross-origin-opener-policy: same-origin"),
        "{on}"
    );
    assert!(
        on.contains("cross-origin-embedder-policy: require-corp"),
        "{on}"
    );
    h.abort();

    let (port, h) = start(d.path(), false, false).await;
    let off = get(port, "/", "Origin: http://example.test\r\n")
        .await
        .to_lowercase();
    assert!(!off.contains("access-control-allow-origin"), "{off}");
    assert!(!off.contains("cross-origin-opener-policy"), "{off}");
    assert!(!off.contains("cross-origin-embedder-policy"), "{off}");
    h.abort();
}

#[tokio::test]
async fn run_gzips_when_the_client_accepts_it() {
    let d = site();
    std::fs::write(d.path().join("big.js"), "x".repeat(8192)).expect("big");
    let (port, h) = start(d.path(), false, false).await;
    let r = get(port, "/big.js", "Accept-Encoding: gzip\r\n")
        .await
        .to_lowercase();
    assert!(r.contains("content-encoding: gzip"), "{r}");
    h.abort();
}

#[tokio::test]
async fn run_returns_the_bind_error_when_the_port_is_taken() {
    let held = std::net::TcpListener::bind("0.0.0.0:0").expect("hold a port");
    let port = held.local_addr().expect("addr").port();
    let config = DevServerConfig::builder().port(port).build();
    let err = DevServer::new(config)
        .run()
        .await
        .expect_err("a second bind must fail");
    assert_eq!(err.kind(), std::io::ErrorKind::AddrInUse, "{err}");
}
