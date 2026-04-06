//! Device Emulation and Environment Mocking (Features 15-16)
//!
//! Emulate mobile devices, screen sizes, geolocation, and other environment settings.
//!
//! ## EXTREME TDD: Tests written FIRST per spec
//!
//! ## Toyota Way Application:
//! - **Poka-Yoke**: Pre-defined device profiles prevent configuration errors
//! - **Genchi Genbutsu**: Accurate real-device specifications from actual devices

mod audio;
mod device;
mod geolocation;

pub use audio::{AudioEmulator, AudioEmulatorConfig, AudioEmulatorError, AudioSource};
pub use device::{DeviceDescriptor, DeviceEmulator, TouchMode, Viewport};
pub use geolocation::{GeolocationMock, GeolocationPosition};
