//! WASM Dashboard Bindings (ENT-004)
//!
//! Provides browser-compatible dashboard implementation using IndexedDB
//! for storage and wasm_bindgen for JavaScript interop.
//!
//! # Features
//!
//! - `IndexedDbStorage`: Persistent storage in browser IndexedDB
//! - `WasmRun`: WASM-compatible run wrapper
//! - Callback-based metric subscriptions
//!
//! # Usage
//!
//! ```javascript
//! import { WasmRun } from 'entrenar';
//!
//! const run = await WasmRun.new('experiment-1');
//! run.log_metric('loss', 0.5);
//! run.log_metric('loss', 0.4);
//!
//! const metrics = run.get_metrics_json();
//! console.log(JSON.parse(metrics));
//!
//! run.subscribe_metrics((key, value) => {
//!     console.log(`${key}: ${value}`);
//! });
//! ```

mod run;
mod storage;

#[cfg(test)]
mod tests;

pub use run::WasmRun;
pub use storage::IndexedDbStorage;
