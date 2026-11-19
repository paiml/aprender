//! probar end-to-end run of `aprender-present bundle` (#7264): build the bundle, serve it,
//! load it in real headless Chrome over CDP, click the counter button, read `data-count` back
//! (0 -> 1 -> 2). Needs `APR_PROBAR_CHROMIUM`, `wasm-pack` and the wasm32 target, so it is
//! skipped unless set; `APR_PROBAR_REQUIRE=1` turns a missing prerequisite into a failure.

use probar::{Browser, BrowserConfig};
use std::path::{Path, PathBuf};
use std::process::Command;

const CONTROLS: &str = "button, input, select, a[href], [data-control]";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root")
}

fn content_type(p: &Path) -> &'static str {
    match p.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html",
        Some("js") => "text/javascript",
        Some("wasm") => "application/wasm",
        Some("json") => "application/json",
        Some("css") => "text/css",
        _ => "application/octet-stream",
    }
}

/// Serve `dir` on an ephemeral 127.0.0.1 port; returns the port.
fn serve(dir: PathBuf) -> u16 {
    let server = tiny_http::Server::http("127.0.0.1:0").expect("bind");
    let port = server.server_addr().to_ip().expect("ip").port();
    std::thread::spawn(move || {
        for req in server.incoming_requests() {
            let rel = req
                .url()
                .split('?')
                .next()
                .unwrap_or("/")
                .trim_start_matches('/')
                .to_string();
            let rel = if rel.is_empty() {
                "index.html".to_string()
            } else {
                rel
            };
            let path = dir.join(&rel);
            if !rel.contains("..") && path.is_file() {
                let ct = tiny_http::Header::from_bytes("Content-Type", content_type(&path))
                    .expect("header");
                let mut body = std::fs::read(&path).expect("read served file");
                if rel == "index.html" {
                    let html = String::from_utf8(body).expect("utf8 page");
                    let trap = format!("<head><script>{TRAP}</script>");
                    body = html.replacen("<head>", &trap, 1).into_bytes();
                }
                let r = tiny_http::Response::from_data(body);
                let _ = req.respond(r.with_header(ct));
            } else {
                let _ = req
                    .respond(tiny_http::Response::from_string("not found").with_status_code(404));
            }
        }
    });
    port
}

const PAINTED: &str = "(() => { const c = document.querySelector('canvas'); \
     if (!c) return false; \
     const d = c.getContext('2d').getImageData(0,0,c.width,c.height).data; \
     for (let i = 3; i < d.length; i += 4) { if (d[i] !== 0) return true; } return false; })()";

/// Runs before any page script: record every error and unhandled rejection.
const TRAP: &str = "window.__errs = []; \
    window.addEventListener('error', e => window.__errs.push(String(e.message))); \
    window.addEventListener('unhandledrejection', e => window.__errs.push(String(e.reason)));";

/// Click a grid over the canvas (mousedown + mouseup) until `data-count` changes; remember
/// the point that did it in `window.__plus`.
const FIND_PLUS: &str = "(() => { const c = document.querySelector('canvas'); \
    const r = c.getBoundingClientRect(); const start = c.getAttribute('data-count'); \
    for (let y = 2; y < c.height; y += 6) { for (let x = 2; x < c.width; x += 6) { \
      for (const t of ['mousedown', 'mouseup']) { c.dispatchEvent(new MouseEvent(t, \
        {bubbles: true, clientX: r.left + x, clientY: r.top + y, button: 0})); } \
      if (c.getAttribute('data-count') !== start) { window.__plus = [x, y]; return true; } } } \
    return false; })()";

/// Click the remembered point once more.
const CLICK_LAST: &str = "(() => { const c = document.querySelector('canvas'); \
    const r = c.getBoundingClientRect(); const [x, y] = window.__plus; \
    for (const t of ['mousedown', 'mouseup']) { c.dispatchEvent(new MouseEvent(t, \
      {bubbles: true, clientX: r.left + x, clientY: r.top + y, button: 0})); } return true; })()";

async fn eval<T: serde::de::DeserializeOwned>(page: &probar::Page, js: &str) -> T {
    page.evaluate(js)
        .await
        .unwrap_or_else(|e| panic!("evaluate {js:?}: {e}"))
        .into_value()
        .unwrap_or_else(|e| panic!("decode {js:?}: {e}"))
}

#[test]
fn bundle_page_every_control_driven_in_real_chrome() {
    let Ok(chrome) = std::env::var("APR_PROBAR_CHROMIUM") else {
        assert!(
            std::env::var("APR_PROBAR_REQUIRE").is_err(),
            "APR_PROBAR_CHROMIUM unset"
        );
        eprintln!("SKIP: APR_PROBAR_CHROMIUM not set");
        return;
    };
    let root = repo_root();
    let out = root.join(format!("target/bundle-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);

    let st = Command::new(env!("CARGO_BIN_EXE_aprender-present"))
        .args(["bundle", "--no-optimize", "--output"])
        .arg(&out)
        .current_dir(&root)
        .status()
        .expect("run bundle");
    assert!(st.success(), "aprender-present bundle failed");
    assert!(out.join("pkg/presentar_bg.wasm").is_file(), "wasm missing");
    assert!(out.join("pkg/presentar.js").is_file(), "js glue missing");

    assert!(
        out.join("index.html").is_file(),
        "bundle wrote no index.html"
    );
    assert!(out.join("pkg/boot.js").is_file(), "bundle wrote no loader");
    let port = serve(out.clone());

    let rt = tokio::runtime::Runtime::new().expect("rt");
    rt.block_on(async {
        let cfg = BrowserConfig {
            headless: false,
            sandbox: false,
            ..BrowserConfig::default()
        }
        .with_chromium_path(chrome);
        let browser = Browser::launch(cfg).await.expect("launch chrome");
        let mut page = browser.new_page().await.expect("new page");
        let url = format!("http://127.0.0.1:{port}/index.html");
        page.goto(&url).await.expect("goto");

        let mut painted = false;
        for _ in 0..150 {
            if let Ok(r) = page.evaluate(PAINTED).await {
                painted = r.into_value::<bool>().unwrap_or(false);
            }
            if painted {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
        assert!(painted, "canvas is blank: the wasm app never rendered");

        let n: u64 = eval(&page, &format!("document.querySelectorAll('{CONTROLS}').length")).await;
        for i in 0..n {
            let step = format!(
                "(() => {{ const el = document.querySelectorAll('{CONTROLS}')[{i}]; \
                 if (el.tagName === 'A') {{ el.addEventListener('click', e => e.preventDefault(), {{once: true}}); }} \
                 el.click(); return true; }})()"
            );
            let _: bool = eval(&page, &step).await;
        }

        let read_count = "document.querySelector('canvas').getAttribute('data-count')";
        let before: Option<String> = eval(&page, read_count).await;
        assert_eq!(before.as_deref(), Some("0"), "counter did not start at 0");
        let found: bool = eval(&page, FIND_PLUS).await;
        assert!(found, "no click on the canvas moved data-count: the + button does nothing");
        let after_first: Option<String> = eval(&page, read_count).await;
        assert_eq!(after_first.as_deref(), Some("1"), "first click must make it 1");
        let again: bool = eval(&page, CLICK_LAST).await;
        assert!(again);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let after_second: Option<String> = eval(&page, read_count).await;
        assert_eq!(after_second.as_deref(), Some("2"), "second click must make it 2");

        let errs: Vec<String> = eval(&page, "window.__errs").await;
        assert!(errs.is_empty(), "page errors since load: {errs:?}");
        let still: bool = eval(&page, PAINTED).await;
        assert!(still, "canvas went blank after driving the controls");
        eprintln!("bundle e2e: painted=true controls_driven={n} count=0->1->2 errors=0");

        let _ = browser.close().await;
    });
    let _ = std::fs::remove_dir_all(&out);
}
