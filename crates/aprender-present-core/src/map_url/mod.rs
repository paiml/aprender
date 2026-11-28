//! MAP-6 slice 1: map URL hash codec and history policy.
//!
//! Pure logic, no browser. Contract: `contracts/aprender-viz-map-url-state-v1.yaml`
//! (MAP-URL-INV-001, MAP-URL-EQ-002, MAP-URL-INV-003, MAP-URL-INV-004).

mod codec;
mod history;
mod state;

#[cfg(test)]
mod falsify_tests;
#[cfg(test)]
mod generators;

pub use codec::{decode, encode};
pub use history::{FakeHistory, HistoryPolicy, HistorySink};
pub use state::{Camera, MapUrlError, MapUrlState, EXTENSION_KEYS, FIXED_KEYS};
