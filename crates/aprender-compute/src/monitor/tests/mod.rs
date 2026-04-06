//! Tests for GPU monitoring (EXTREME TDD - Tests First!)

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod backend;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod config_error;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod device_info;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod integration;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod metrics;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod monitor_mock;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod vendor;
