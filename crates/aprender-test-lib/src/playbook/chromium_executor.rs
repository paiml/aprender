//! Real [`ActionExecutor`] backed by [`ChromiumDriver`].
//!
//! # Why this file exists
//!
//! `ActionExecutor` is the trait playbooks run through, and it had **no
//! production implementation at all** — the only two `impl ActionExecutor` in
//! the workspace were `MockExecutor`, both inside `#[cfg(test)]` modules
//! (`executor.rs:485`, `runner.rs:457`). So a playbook could be authored,
//! parsed, validated and "run", and nothing ever reached a browser.
//!
//! This is the same shape as the `ProbarDriver`/`MockDriver` gap in #2473, one
//! layer up: an interface with a mock behind it. `ChromiumDriver` closed that
//! one; this closes the layer above it, so `click`, `navigate` and `wait` in a
//! playbook drive Chrome.
//!
//! # Sync over async
//!
//! `ActionExecutor` is synchronous and `ChromiumDriver` is not, so this bridges
//! with a runtime it owns. Calling it from inside an async context would panic
//! inside tokio, so [`ChromiumExecutor::launch`] **refuses** with a clear error
//! instead — see [`ExecutorError::ScriptError`]. A panic buried in a playbook
//! step is a bad way to learn about a threading rule.

use std::time::{Duration, Instant};

use tokio::runtime::Runtime;

use super::executor::{ActionExecutor, ExecutorError};
use super::schema::WaitCondition;
use crate::chromium_driver::ChromiumDriver;
use crate::driver::{DriverConfig, ProbarDriver};

/// Executes playbook actions against a real browser.
#[derive(Debug)]
pub struct ChromiumExecutor {
    driver: ChromiumDriver,
    runtime: Runtime,
    /// Where `screenshot` writes. Each shot is a real PNG from the compositor.
    screenshot_dir: std::path::PathBuf,
}

impl ChromiumExecutor {
    /// Launch a browser to execute playbook actions against.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutorError::ScriptError`] if called from inside an async
    /// context, if no runtime can be built, or if the browser will not launch.
    pub fn launch(
        config: DriverConfig,
        screenshot_dir: impl Into<std::path::PathBuf>,
    ) -> Result<Self, ExecutorError> {
        if tokio::runtime::Handle::try_current().is_ok() {
            return Err(ExecutorError::ScriptError {
                message: "ChromiumExecutor::launch was called from inside an async \
                          context; it owns a runtime and would deadlock. Construct it \
                          on a blocking thread (tokio::task::spawn_blocking) instead."
                    .to_string(),
            });
        }

        let runtime = Runtime::new().map_err(|e| ExecutorError::ScriptError {
            message: format!("could not build a tokio runtime for the browser: {e}"),
        })?;
        let driver = runtime
            .block_on(ChromiumDriver::launch(config))
            .map_err(|e| ExecutorError::ScriptError {
                message: format!("could not launch a browser: {e}"),
            })?;

        Ok(Self {
            driver,
            runtime,
            screenshot_dir: screenshot_dir.into(),
        })
    }

    /// The underlying driver, for callers that need the full async surface.
    #[must_use]
    pub const fn driver(&self) -> &ChromiumDriver {
        &self.driver
    }

    /// How long a wait condition may take before it is a [`ExecutorError::Timeout`].
    ///
    /// Taken from `DriverConfig::element_timeout` rather than hardcoded: the
    /// field existed and was ignored, which is a promise the executor did not
    /// keep -- and it made the timeout path of the test suite take 30s.
    fn wait_timeout(&self) -> Duration {
        self.driver.config().element_timeout
    }

    /// Read a single JSON value out of the page.
    fn eval_json(&self, script: &str) -> Result<serde_json::Value, ExecutorError> {
        self.runtime
            .block_on(self.driver.execute_js(script))
            .map_err(|e| ExecutorError::ScriptError {
                message: e.to_string(),
            })
    }

    /// A JS string literal for `s`, so a selector containing quotes cannot
    /// break out of the expression it is pasted into.
    fn js_str(s: &str) -> String {
        serde_json::Value::String(s.to_string()).to_string()
    }
}

impl ActionExecutor for ChromiumExecutor {
    fn click(&mut self, selector: &str) -> Result<(), ExecutorError> {
        self.runtime
            .block_on(self.driver.click(selector))
            .map_err(|_| ExecutorError::ElementNotFound {
                selector: selector.to_string(),
            })
    }

    fn type_text(&mut self, selector: &str, text: &str) -> Result<(), ExecutorError> {
        self.runtime
            .block_on(self.driver.type_text(selector, text))
            .map_err(|_| ExecutorError::ElementNotFound {
                selector: selector.to_string(),
            })
    }

    fn wait(&mut self, condition: &WaitCondition) -> Result<(), ExecutorError> {
        match condition {
            WaitCondition::Visible { selector } => self
                .runtime
                .block_on(self.driver.wait_for_selector(selector, self.wait_timeout()))
                .map(|_| ())
                .map_err(|_| ExecutorError::Timeout),

            WaitCondition::Hidden { selector } => {
                // Polled rather than assumed: "hidden" is the absence of a laid
                // out box, which only the renderer can report.
                let deadline = Instant::now() + self.wait_timeout();
                let probe = format!(
                    "(() => {{ const e = document.querySelector({});
                       if (!e) return true;
                       const r = e.getBoundingClientRect();
                       return r.width === 0 || r.height === 0
                              || getComputedStyle(e).visibility === 'hidden'; }})()",
                    Self::js_str(selector)
                );
                loop {
                    if self.eval_json(&probe)?.as_bool().unwrap_or(false) {
                        return Ok(());
                    }
                    if Instant::now() >= deadline {
                        return Err(ExecutorError::Timeout);
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
            }

            WaitCondition::Duration { ms } => {
                std::thread::sleep(Duration::from_millis(*ms));
                Ok(())
            }

            WaitCondition::NetworkIdle => {
                // Approximated by "no resource finished loading recently", read
                // from the page's own Resource Timing. Documented as an
                // approximation rather than presented as CDP-accurate.
                let deadline = Instant::now() + self.wait_timeout();
                loop {
                    let quiet = self
                        .eval_json(
                            "(() => { const es = performance.getEntriesByType('resource');
                               if (es.length === 0) return true;
                               const last = es[es.length - 1];
                               return performance.now() - (last.responseEnd || 0) > 500; })()",
                        )?
                        .as_bool()
                        .unwrap_or(false);
                    if quiet {
                        return Ok(());
                    }
                    if Instant::now() >= deadline {
                        return Err(ExecutorError::Timeout);
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            }

            WaitCondition::Condition { expression } => {
                let deadline = Instant::now() + self.wait_timeout();
                loop {
                    if self.eval_json(expression)?.as_bool().unwrap_or(false) {
                        return Ok(());
                    }
                    if Instant::now() >= deadline {
                        return Err(ExecutorError::Timeout);
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        }
    }

    fn navigate(&mut self, url: &str) -> Result<(), ExecutorError> {
        self.runtime
            .block_on(self.driver.navigate(url))
            .map_err(|_| ExecutorError::NavigationFailed {
                url: url.to_string(),
            })
    }

    fn execute_script(&mut self, code: &str) -> Result<String, ExecutorError> {
        let v = self.eval_json(code)?;
        // A JSON string is returned bare; anything else keeps its JSON form, so
        // a caller can tell 42 from "42".
        Ok(match v {
            serde_json::Value::String(s) => s,
            other => other.to_string(),
        })
    }

    fn screenshot(&mut self, name: &str) -> Result<(), ExecutorError> {
        let shot = self
            .runtime
            .block_on(self.driver.screenshot())
            .map_err(|e| ExecutorError::ScriptError {
                message: format!("screenshot failed: {e}"),
            })?;
        std::fs::create_dir_all(&self.screenshot_dir).map_err(|e| ExecutorError::ScriptError {
            message: format!("could not create {}: {e}", self.screenshot_dir.display()),
        })?;
        let path = self.screenshot_dir.join(format!("{name}.png"));
        std::fs::write(&path, &shot.data).map_err(|e| ExecutorError::ScriptError {
            message: format!("could not write {}: {e}", path.display()),
        })
    }

    fn element_exists(&self, selector: &str) -> Result<bool, ExecutorError> {
        Ok(self
            .eval_json(&format!(
                "document.querySelector({}) !== null",
                Self::js_str(selector)
            ))?
            .as_bool()
            .unwrap_or(false))
    }

    fn get_text(&self, selector: &str) -> Result<String, ExecutorError> {
        let v = self.eval_json(&format!(
            "(() => {{ const e = document.querySelector({}); return e ? e.textContent : null; }})()",
            Self::js_str(selector)
        ))?;
        // A missing element is ElementNotFound, not an empty string: an empty
        // string is a real answer for an element that exists and is blank.
        v.as_str()
            .map(str::to_string)
            .ok_or_else(|| ExecutorError::ElementNotFound {
                selector: selector.to_string(),
            })
    }

    fn get_attribute(&self, selector: &str, attribute: &str) -> Result<String, ExecutorError> {
        let v = self.eval_json(&format!(
            "(() => {{ const e = document.querySelector({}); \
               return e ? e.getAttribute({}) : null; }})()",
            Self::js_str(selector),
            Self::js_str(attribute)
        ))?;
        v.as_str()
            .map(str::to_string)
            .ok_or_else(|| ExecutorError::ElementNotFound {
                selector: selector.to_string(),
            })
    }

    fn get_url(&self) -> Result<String, ExecutorError> {
        self.runtime
            .block_on(self.driver.current_url())
            .map_err(|e| ExecutorError::ScriptError {
                message: e.to_string(),
            })
    }

    fn evaluate(&self, expression: &str) -> Result<bool, ExecutorError> {
        // Truthiness is decided by the page, not by us guessing at JSON shapes:
        // `!!(expr)` is what the author of the playbook expression means.
        Ok(self
            .eval_json(&format!("!!({expression})"))?
            .as_bool()
            .unwrap_or(false))
    }
}
