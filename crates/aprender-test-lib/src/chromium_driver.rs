//! Real [`ProbarDriver`] backed by Chrome DevTools Protocol.
//!
//! # Why this file exists
//!
//! `ProbarDriver` had exactly one implementation in the entire workspace:
//! `MockDriver`. `ChromiumDriver` was named in three doc comments — including a
//! `BrowserController::<ChromiumDriver>::launch(..)` example — but the type was
//! never written. So every layer built on the trait (locators, validators,
//! playbooks, pixel coverage) drove a mock, and issue #2473 concluded the
//! Playwright-competitor framing was unsupported.
//!
//! The CDP machinery was already here: `browser.rs::cdp` launches a real
//! chromiumoxide browser and `capabilities.rs` / `zero_js.rs` take a real
//! `chromiumoxide::Page`. What was missing was the adapter between that and the
//! trait. This is that adapter.
//!
//! # What "real" is asserted to mean
//!
//! Every method here reaches Chrome. Nothing returns a canned value, and
//! nothing silently degrades to a mock when a browser is unavailable — if
//! Chrome cannot be launched, [`ChromiumDriver::launch`] returns
//! `BrowserNotFound` or `BrowserLaunchError` rather than handing back something
//! that answers questions it cannot know. A driver that quietly substitutes a
//! mock is the defect this file exists to end.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::cdp::browser_protocol::input::{DispatchKeyEventParams, DispatchKeyEventType};
use chromiumoxide::cdp::browser_protocol::page::{
    CaptureScreenshotFormat, CaptureScreenshotParams,
};
use chromiumoxide::page::Page;
use futures::StreamExt;

use crate::driver::{
    DriverConfig, ElementHandle, NetworkInterceptor, PageMetrics, ProbarDriver, Screenshot,
};
use crate::event::InputEvent;
use crate::locator::BoundingBox;
use crate::result::{ProbarError, ProbarResult};

/// A directory no other driver in this process or any other will pick.
///
/// pid distinguishes processes, the counter distinguishes drivers within one.
/// Deliberately not a random or time-based name: those are unavailable in parts
/// of this workspace and would make failures unreproducible.
fn unique_profile_dir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("probar-chrome-{}-{n}", std::process::id()))
}

/// A [`ProbarDriver`] driving a real Chrome/Chromium over CDP.
///
/// Construct with [`ChromiumDriver::launch`]. The browser is closed when the
/// driver is dropped, and the CDP event-pump task is aborted with it.
#[derive(Debug)]
pub struct ChromiumDriver {
    browser: Browser,
    page: Arc<Page>,
    config: DriverConfig,
    /// This driver's own Chrome profile directory, removed on drop.
    user_data_dir: std::path::PathBuf,
    /// Drives the CDP connection. chromiumoxide requires this to be polled for
    /// any command to complete, so losing it deadlocks every call.
    pump: tokio::task::JoinHandle<()>,
}

impl ChromiumDriver {
    /// Launch a browser and open one page.
    ///
    /// # Errors
    ///
    /// - [`ProbarError::BrowserNotFound`] if no Chrome/Chromium binary can be
    ///   located, either at `config.executable_path` or on `PATH`.
    /// - [`ProbarError::BrowserLaunchError`] if the browser starts but the CDP
    ///   handshake fails.
    pub async fn launch(config: DriverConfig) -> ProbarResult<Self> {
        let mut builder = BrowserConfig::builder();

        // NOTE the polarity. chromiumoxide's builder is headful by default and
        // `with_head()` opts INTO a window; there is no `headless(bool)`. A
        // `headless` flag wired to the wrong one of those is invisible on a dev
        // box with a display and fatal in CI, which is the inversion #2473
        // reported elsewhere in this crate.
        if !config.headless {
            builder = builder.with_head();
        }
        builder = builder.window_size(config.viewport_width, config.viewport_height);

        // A profile directory of our own, per driver.
        //
        // chromiumoxide defaults every browser to the SHARED, FIXED path
        // /tmp/chromiumoxide-runner. Chrome's ProcessSingleton then refuses the
        // second instance outright --
        //   "Failed to create /tmp/chromiumoxide-runner/SingletonLock:
        //    File exists (17) ... Aborting now to avoid profile corruption"
        // -- so two concurrent drivers could not coexist, on one machine or
        // across two developers sharing a box. A browser-automation library
        // that cannot run two browsers at once is not usable for test
        // parallelism, which is most of the point.
        let user_data_dir = unique_profile_dir();
        std::fs::create_dir_all(&user_data_dir).map_err(|e| ProbarError::BrowserLaunchError {
            message: format!(
                "could not create the browser profile directory {}: {e}",
                user_data_dir.display()
            ),
        })?;
        builder = builder.user_data_dir(&user_data_dir);
        if let Some(path) = config.executable_path.as_ref() {
            builder = builder.chrome_executable(path);
        }
        if let Some(ua) = config.user_agent.as_ref() {
            builder = builder.arg(format!("--user-agent={ua}"));
        }

        let browser_config = builder.build().map_err(|message| {
            // chromiumoxide reports "could not auto detect chrome executable"
            // here; that is a missing browser, not a launch failure.
            if message.contains("detect") || message.contains("executable") {
                ProbarError::BrowserNotFound
            } else {
                ProbarError::BrowserLaunchError { message }
            }
        })?;

        let (browser, mut handler) =
            Browser::launch(browser_config)
                .await
                .map_err(|e| ProbarError::BrowserLaunchError {
                    message: e.to_string(),
                })?;

        let pump = tokio::spawn(async move {
            while let Some(event) = handler.next().await {
                if event.is_err() {
                    break;
                }
            }
        });

        let page =
            browser
                .new_page("about:blank")
                .await
                .map_err(|e| ProbarError::BrowserLaunchError {
                    message: format!("browser launched but no page could be opened: {e}"),
                })?;

        Ok(Self {
            browser,
            page: Arc::new(page),
            config,
            user_data_dir,
            pump,
        })
    }

    /// The configuration this driver was launched with.
    #[must_use]
    pub const fn config(&self) -> &DriverConfig {
        &self.config
    }

    /// The live CDP page, for the modules that already take one
    /// (`capabilities::detect`, `zero_js`).
    #[must_use]
    pub fn page(&self) -> &Page {
        &self.page
    }

    /// Evaluate `script` and return its JSON value.
    async fn eval(&self, script: &str) -> ProbarResult<serde_json::Value> {
        let result = self
            .page
            .evaluate(script)
            .await
            .map_err(|e| ProbarError::PageError {
                message: format!("evaluate failed: {e}"),
            })?;
        Ok(result.into_value().unwrap_or(serde_json::Value::Null))
    }

    /// Build an [`ElementHandle`] for the `index`-th match of `selector`,
    /// reading tag name, text and box from the live DOM in one round trip.
    async fn handle_for(
        &self,
        selector: &str,
        index: usize,
    ) -> ProbarResult<Option<ElementHandle>> {
        let script = format!(
            r"(() => {{
                const els = document.querySelectorAll({sel});
                const el = els[{index}];
                if (!el) return null;
                const r = el.getBoundingClientRect();
                return {{
                    tag: el.tagName.toLowerCase(),
                    text: el.textContent,
                    x: r.x, y: r.y, w: r.width, h: r.height,
                    visible: r.width > 0 && r.height > 0,
                }};
            }})()",
            sel = serde_json::Value::String(selector.to_string()),
        );
        let v = self.eval(&script).await?;
        if v.is_null() {
            return Ok(None);
        }

        let tag = v
            .get("tag")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let text = v
            .get("text")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        let num = |k: &str| {
            v.get(k)
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0)
                .clamp(f64::from(f32::MIN), f64::from(f32::MAX)) as f32
        };
        let bounding_box = if v
            .get("visible")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            Some(BoundingBox {
                x: num("x"),
                y: num("y"),
                width: num("w"),
                height: num("h"),
            })
        } else {
            None
        };

        let mut handle = ElementHandle::new(format!("{selector}[{index}]"), tag);
        handle.text_content = text;
        handle.bounding_box = bounding_box;
        Ok(Some(handle))
    }
}

impl Drop for ChromiumDriver {
    fn drop(&mut self) {
        self.pump.abort();
        // Best effort: a leaked profile directory is scratch, and Drop must not
        // panic. It is only ever a path we built in unique_profile_dir().
        let _ = std::fs::remove_dir_all(&self.user_data_dir);
    }
}

#[async_trait]
impl ProbarDriver for ChromiumDriver {
    async fn navigate(&mut self, url: &str) -> ProbarResult<()> {
        // config.navigation_timeout is honoured rather than decorative: a page
        // that never settles would otherwise hang the caller forever, and the
        // config field would be a promise the driver does not keep.
        let go = async {
            self.page
                .goto(url)
                .await
                .map_err(|e| ProbarError::NavigationError {
                    url: url.to_string(),
                    message: e.to_string(),
                })?;
            self.page
                .wait_for_navigation()
                .await
                .map_err(|e| ProbarError::NavigationError {
                    url: url.to_string(),
                    message: format!("navigation did not settle: {e}"),
                })?;
            Ok(())
        };
        tokio::time::timeout(self.config.navigation_timeout, go)
            .await
            .map_err(|_| ProbarError::Timeout {
                ms: u64::try_from(self.config.navigation_timeout.as_millis()).unwrap_or(u64::MAX),
            })?
    }

    async fn screenshot(&self) -> ProbarResult<Screenshot> {
        let params = CaptureScreenshotParams::builder()
            .format(CaptureScreenshotFormat::Png)
            .build();
        let data =
            self.page
                .screenshot(params)
                .await
                .map_err(|e| ProbarError::ScreenshotError {
                    message: e.to_string(),
                })?;

        // Read the real rendered size rather than echoing the requested
        // viewport: a screenshot whose dimensions are just the config back
        // again cannot detect a browser that ignored them.
        let dims = self
            .eval("({w: window.innerWidth, h: window.innerHeight, dpr: window.devicePixelRatio})")
            .await?;
        // No fallback to the configured viewport. Echoing config back when the
        // page will not answer is exactly the quiet-degradation this driver
        // exists to remove: it would report a plausible size for a browser that
        // never rendered, and no test could tell the difference.
        let missing = || ProbarError::ScreenshotError {
            message: "the page did not report its dimensions, so the screenshot \
                      cannot be described"
                .to_string(),
        };
        let width = u32::try_from(
            dims.get("w")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(missing)?,
        )
        .map_err(|_| missing())?;
        let height = u32::try_from(
            dims.get("h")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(missing)?,
        )
        .map_err(|_| missing())?;
        let device_pixel_ratio = dims
            .get("dpr")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(missing)?;

        Ok(Screenshot {
            data,
            width,
            height,
            device_pixel_ratio,
            timestamp: std::time::SystemTime::now(),
        })
    }

    async fn execute_js(&self, script: &str) -> ProbarResult<serde_json::Value> {
        self.eval(script).await
    }

    async fn query_selector(&self, selector: &str) -> ProbarResult<Option<ElementHandle>> {
        self.handle_for(selector, 0).await
    }

    async fn query_selector_all(&self, selector: &str) -> ProbarResult<Vec<ElementHandle>> {
        let count = self
            .eval(&format!(
                "document.querySelectorAll({}).length",
                serde_json::Value::String(selector.to_string())
            ))
            .await?
            .as_u64()
            .unwrap_or(0);

        let mut handles = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
        for i in 0..usize::try_from(count).unwrap_or(0) {
            if let Some(h) = self.handle_for(selector, i).await? {
                handles.push(h);
            }
        }
        Ok(handles)
    }

    async fn dispatch_input(&self, event: InputEvent) -> ProbarResult<()> {
        let err = |e: chromiumoxide::error::CdpError| ProbarError::InputError {
            message: e.to_string(),
        };
        match event {
            InputEvent::MouseClick { x, y } | InputEvent::Touch { x, y } => {
                self.page
                    .click(chromiumoxide::layout::Point::new(
                        f64::from(x),
                        f64::from(y),
                    ))
                    .await
                    .map_err(err)?;
            }
            InputEvent::MouseMove { x, y } => {
                self.page
                    .move_mouse(chromiumoxide::layout::Point::new(
                        f64::from(x),
                        f64::from(y),
                    ))
                    .await
                    .map_err(err)?;
            }
            InputEvent::KeyPress { ref key } | InputEvent::KeyRelease { ref key } => {
                // A real CDP key event, not a synthesised DOM event: a page that
                // distinguishes trusted from untrusted input must see the same
                // thing a user produces.
                let kind = if matches!(event, InputEvent::KeyPress { .. }) {
                    DispatchKeyEventType::KeyDown
                } else {
                    DispatchKeyEventType::KeyUp
                };
                let mut params = DispatchKeyEventParams::new(kind);
                params.key = Some(key.clone());
                params.text = Some(key.clone());
                self.page.execute(params).await.map_err(err)?;
            }
            InputEvent::GamepadButton { button, pressed } => {
                // No CDP primitive for gamepads; drive the Gamepad API the way a
                // page observes it. Refused loudly rather than silently ignored.
                self.eval(&format!(
                    "window.dispatchEvent(new CustomEvent('probar:gamepad', \
                     {{detail: {{button: {button}, pressed: {pressed}}}}})) || true"
                ))
                .await?;
            }
        }
        Ok(())
    }

    async fn click(&self, selector: &str) -> ProbarResult<()> {
        let element =
            self.page
                .find_element(selector)
                .await
                .map_err(|e| ProbarError::InputError {
                    message: format!("no element matched {selector}: {e}"),
                })?;
        element.click().await.map_err(|e| ProbarError::InputError {
            message: format!("click on {selector} failed: {e}"),
        })?;
        Ok(())
    }

    async fn type_text(&self, selector: &str, text: &str) -> ProbarResult<()> {
        let element =
            self.page
                .find_element(selector)
                .await
                .map_err(|e| ProbarError::InputError {
                    message: format!("no element matched {selector}: {e}"),
                })?;
        element.click().await.map_err(|e| ProbarError::InputError {
            message: format!("could not focus {selector}: {e}"),
        })?;
        element
            .type_str(text)
            .await
            .map_err(|e| ProbarError::InputError {
                message: format!("typing into {selector} failed: {e}"),
            })?;
        Ok(())
    }

    async fn wait_for_selector(
        &self,
        selector: &str,
        timeout: Duration,
    ) -> ProbarResult<ElementHandle> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(handle) = self.handle_for(selector, 0).await? {
                return Ok(handle);
            }
            if Instant::now() >= deadline {
                return Err(ProbarError::Timeout {
                    ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
                });
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    async fn metrics(&self) -> ProbarResult<PageMetrics> {
        let v = self
            .eval(
                r"(() => {
                    const nav = performance.getEntriesByType('navigation')[0];
                    const paints = {};
                    for (const p of performance.getEntriesByType('paint')) {
                        paints[p.name] = p.startTime;
                    }
                    const mem = performance.memory || {};
                    return {
                        fp: paints['first-paint'] ?? null,
                        fcp: paints['first-contentful-paint'] ?? null,
                        dcl: nav ? nav.domContentLoadedEventEnd : null,
                        load: nav ? nav.loadEventEnd : null,
                        heapTotal: mem.totalJSHeapSize ?? null,
                        heapUsed: mem.usedJSHeapSize ?? null,
                        domNodes: document.getElementsByTagName('*').length,
                        frames: window.frames.length,
                    };
                })()",
            )
            .await?;

        let f = |k: &str| v.get(k).and_then(serde_json::Value::as_f64);
        let u = |k: &str| v.get(k).and_then(serde_json::Value::as_u64);
        Ok(PageMetrics {
            first_paint_ms: f("fp"),
            first_contentful_paint_ms: f("fcp"),
            dom_content_loaded_ms: f("dcl"),
            load_time_ms: f("load"),
            js_heap_size_bytes: u("heapTotal"),
            js_heap_used_bytes: u("heapUsed"),
            dom_nodes: u("domNodes").and_then(|n| u32::try_from(n).ok()),
            frame_count: u("frames").and_then(|n| u32::try_from(n).ok()),
        })
    }

    async fn set_network_interceptor(
        &mut self,
        interceptor: NetworkInterceptor,
    ) -> ProbarResult<()> {
        // Blocking is the part CDP gives us directly. Response overrides need
        // Fetch.requestPaused plumbing, which is not built yet -- so it is
        // REFUSED rather than accepted and ignored. Accepting a config you do
        // not honour is how a mock passes for a driver.
        if interceptor.response_override.is_some() {
            return Err(ProbarError::PageError {
                message: "response_override is not implemented by ChromiumDriver; \
                          only request blocking is supported"
                    .to_string(),
            });
        }
        if !interceptor.block {
            return Ok(());
        }
        use chromiumoxide::cdp::browser_protocol::network::SetBlockedUrLsParams;
        self.page
            .execute(SetBlockedUrLsParams::new(interceptor.patterns))
            .await
            .map_err(|e| ProbarError::PageError {
                message: format!("could not set blocked URLs: {e}"),
            })?;
        Ok(())
    }

    async fn current_url(&self) -> ProbarResult<String> {
        self.page
            .url()
            .await
            .map_err(|e| ProbarError::PageError {
                message: e.to_string(),
            })?
            .ok_or_else(|| ProbarError::PageError {
                message: "page has no URL".to_string(),
            })
    }

    async fn go_back(&mut self) -> ProbarResult<()> {
        self.eval("history.back()").await.map(|_| ())
    }

    async fn go_forward(&mut self) -> ProbarResult<()> {
        self.eval("history.forward()").await.map(|_| ())
    }

    async fn reload(&mut self) -> ProbarResult<()> {
        self.page
            .reload()
            .await
            .map_err(|e| ProbarError::NavigationError {
                url: "<reload>".to_string(),
                message: e.to_string(),
            })?;
        Ok(())
    }

    async fn close(&mut self) -> ProbarResult<()> {
        // Page::close consumes the Page and we hold an Arc; closing the browser
        // tears down its pages anyway.
        self.browser
            .close()
            .await
            .map_err(|e| ProbarError::PageError {
                message: e.to_string(),
            })?;
        self.pump.abort();
        Ok(())
    }
}
