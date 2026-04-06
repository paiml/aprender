//! Device Emulation (Feature 15)
//!
//! Emulate mobile devices, screen sizes, and device capabilities.
//!
//! ## EXTREME TDD: Tests written FIRST per spec

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Viewport dimensions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Viewport {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
}

impl Viewport {
    /// Create a new viewport
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Create a landscape version of this viewport
    #[must_use]
    pub const fn landscape(self) -> Self {
        if self.width > self.height {
            self
        } else {
            Self {
                width: self.height,
                height: self.width,
            }
        }
    }

    /// Create a portrait version of this viewport
    #[must_use]
    pub const fn portrait(self) -> Self {
        if self.height > self.width {
            self
        } else {
            Self {
                width: self.height,
                height: self.width,
            }
        }
    }

    /// Check if viewport is in landscape orientation
    #[must_use]
    pub const fn is_landscape(&self) -> bool {
        self.width > self.height
    }

    /// Check if viewport is in portrait orientation
    #[must_use]
    pub const fn is_portrait(&self) -> bool {
        self.height > self.width
    }
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
        }
    }
}

/// Touch mode configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TouchMode {
    /// No touch support
    #[default]
    None,
    /// Single touch
    Single,
    /// Multi-touch
    Multi,
}

impl TouchMode {
    /// Check if touch is enabled
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Device descriptor with all emulation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceDescriptor {
    /// Device name (e.g., "iPhone 14 Pro")
    pub name: String,
    /// Viewport dimensions
    pub viewport: Viewport,
    /// User agent string
    pub user_agent: String,
    /// Device pixel ratio (e.g., 2.0 for Retina, 3.0 for iPhone)
    pub device_scale_factor: f64,
    /// Whether the device is mobile
    pub is_mobile: bool,
    /// Touch support
    pub touch: TouchMode,
    /// Whether device supports hover
    pub has_hover: bool,
}

impl DeviceDescriptor {
    /// Create a new device descriptor
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            viewport: Viewport::default(),
            user_agent: String::new(),
            device_scale_factor: 1.0,
            is_mobile: false,
            touch: TouchMode::None,
            has_hover: true,
        }
    }

    /// Set viewport
    #[must_use]
    pub const fn with_viewport(mut self, viewport: Viewport) -> Self {
        self.viewport = viewport;
        self
    }

    /// Set viewport dimensions
    #[must_use]
    pub const fn with_viewport_size(mut self, width: u32, height: u32) -> Self {
        self.viewport = Viewport::new(width, height);
        self
    }

    /// Set user agent
    #[must_use]
    pub fn with_user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = ua.into();
        self
    }

    /// Set device scale factor
    #[must_use]
    pub const fn with_device_scale_factor(mut self, factor: f64) -> Self {
        self.device_scale_factor = factor;
        self
    }

    /// Set mobile mode
    #[must_use]
    pub const fn with_mobile(mut self, is_mobile: bool) -> Self {
        self.is_mobile = is_mobile;
        self
    }

    /// Set touch mode
    #[must_use]
    pub const fn with_touch(mut self, touch: TouchMode) -> Self {
        self.touch = touch;
        self
    }

    /// Set hover support
    #[must_use]
    pub const fn with_hover(mut self, has_hover: bool) -> Self {
        self.has_hover = has_hover;
        self
    }
}

/// Device emulator with preset device profiles
#[derive(Debug, Default)]
pub struct DeviceEmulator {
    presets: HashMap<String, DeviceDescriptor>,
}

impl DeviceEmulator {
    /// Create a new device emulator with built-in presets
    #[must_use]
    pub fn new() -> Self {
        let mut emulator = Self {
            presets: HashMap::new(),
        };

        // Add built-in device presets
        emulator.register_preset(Self::iphone_14());
        emulator.register_preset(Self::iphone_14_pro());
        emulator.register_preset(Self::iphone_14_pro_max());
        emulator.register_preset(Self::ipad_pro());
        emulator.register_preset(Self::ipad_mini());
        emulator.register_preset(Self::pixel_7());
        emulator.register_preset(Self::pixel_7_pro());
        emulator.register_preset(Self::samsung_galaxy_s23());
        emulator.register_preset(Self::desktop_1080p());
        emulator.register_preset(Self::desktop_1440p());
        emulator.register_preset(Self::desktop_4k());

        emulator
    }

    /// Register a custom device preset
    pub fn register_preset(&mut self, device: DeviceDescriptor) {
        let _ = self.presets.insert(device.name.clone(), device);
    }

    /// Get a device preset by name
    #[must_use]
    pub fn get_preset(&self, name: &str) -> Option<&DeviceDescriptor> {
        self.presets.get(name)
    }

    /// Get all preset names
    #[must_use]
    pub fn preset_names(&self) -> Vec<&str> {
        self.presets.keys().map(String::as_str).collect()
    }

    /// Create a custom device
    #[must_use]
    pub fn custom(viewport: Viewport, user_agent: &str) -> DeviceDescriptor {
        DeviceDescriptor::new("Custom")
            .with_viewport(viewport)
            .with_user_agent(user_agent)
    }

    // ========================================================================
    // iPhone Presets
    // ========================================================================

    /// iPhone 14 device preset
    #[must_use]
    pub fn iphone_14() -> DeviceDescriptor {
        DeviceDescriptor::new("iPhone 14")
            .with_viewport_size(390, 844)
            .with_user_agent(
                "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) \
                AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            )
            .with_device_scale_factor(3.0)
            .with_mobile(true)
            .with_touch(TouchMode::Multi)
            .with_hover(false)
    }

    /// iPhone 14 Pro device preset
    #[must_use]
    pub fn iphone_14_pro() -> DeviceDescriptor {
        DeviceDescriptor::new("iPhone 14 Pro")
            .with_viewport_size(393, 852)
            .with_user_agent(
                "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) \
                AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            )
            .with_device_scale_factor(3.0)
            .with_mobile(true)
            .with_touch(TouchMode::Multi)
            .with_hover(false)
    }

    /// iPhone 14 Pro Max device preset
    #[must_use]
    pub fn iphone_14_pro_max() -> DeviceDescriptor {
        DeviceDescriptor::new("iPhone 14 Pro Max")
            .with_viewport_size(430, 932)
            .with_user_agent(
                "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) \
                AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            )
            .with_device_scale_factor(3.0)
            .with_mobile(true)
            .with_touch(TouchMode::Multi)
            .with_hover(false)
    }

    // ========================================================================
    // iPad Presets
    // ========================================================================

    /// iPad Pro 12.9" device preset
    #[must_use]
    pub fn ipad_pro() -> DeviceDescriptor {
        DeviceDescriptor::new("iPad Pro")
            .with_viewport_size(1024, 1366)
            .with_user_agent(
                "Mozilla/5.0 (iPad; CPU OS 16_0 like Mac OS X) \
                AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            )
            .with_device_scale_factor(2.0)
            .with_mobile(true)
            .with_touch(TouchMode::Multi)
            .with_hover(false)
    }

    /// iPad Mini device preset
    #[must_use]
    pub fn ipad_mini() -> DeviceDescriptor {
        DeviceDescriptor::new("iPad Mini")
            .with_viewport_size(768, 1024)
            .with_user_agent(
                "Mozilla/5.0 (iPad; CPU OS 16_0 like Mac OS X) \
                AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            )
            .with_device_scale_factor(2.0)
            .with_mobile(true)
            .with_touch(TouchMode::Multi)
            .with_hover(false)
    }

    // ========================================================================
    // Android Presets
    // ========================================================================

    /// Google Pixel 7 device preset
    #[must_use]
    pub fn pixel_7() -> DeviceDescriptor {
        DeviceDescriptor::new("Pixel 7")
            .with_viewport_size(412, 915)
            .with_user_agent(
                "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 \
                (KHTML, like Gecko) Chrome/116.0.0.0 Mobile Safari/537.36",
            )
            .with_device_scale_factor(2.625)
            .with_mobile(true)
            .with_touch(TouchMode::Multi)
            .with_hover(false)
    }

    /// Google Pixel 7 Pro device preset
    #[must_use]
    pub fn pixel_7_pro() -> DeviceDescriptor {
        DeviceDescriptor::new("Pixel 7 Pro")
            .with_viewport_size(412, 892)
            .with_user_agent(
                "Mozilla/5.0 (Linux; Android 13; Pixel 7 Pro) AppleWebKit/537.36 \
                (KHTML, like Gecko) Chrome/116.0.0.0 Mobile Safari/537.36",
            )
            .with_device_scale_factor(3.5)
            .with_mobile(true)
            .with_touch(TouchMode::Multi)
            .with_hover(false)
    }

    /// Samsung Galaxy S23 device preset
    #[must_use]
    pub fn samsung_galaxy_s23() -> DeviceDescriptor {
        DeviceDescriptor::new("Samsung Galaxy S23")
            .with_viewport_size(360, 780)
            .with_user_agent(
                "Mozilla/5.0 (Linux; Android 13; SM-S911B) AppleWebKit/537.36 \
                (KHTML, like Gecko) Chrome/116.0.0.0 Mobile Safari/537.36",
            )
            .with_device_scale_factor(3.0)
            .with_mobile(true)
            .with_touch(TouchMode::Multi)
            .with_hover(false)
    }

    // ========================================================================
    // Desktop Presets
    // ========================================================================

    /// 1080p Desktop preset
    #[must_use]
    pub fn desktop_1080p() -> DeviceDescriptor {
        DeviceDescriptor::new("Desktop 1080p")
            .with_viewport_size(1920, 1080)
            .with_user_agent(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
            )
            .with_device_scale_factor(1.0)
            .with_mobile(false)
            .with_touch(TouchMode::None)
            .with_hover(true)
    }

    /// 1440p Desktop preset
    #[must_use]
    pub fn desktop_1440p() -> DeviceDescriptor {
        DeviceDescriptor::new("Desktop 1440p")
            .with_viewport_size(2560, 1440)
            .with_user_agent(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
            )
            .with_device_scale_factor(1.0)
            .with_mobile(false)
            .with_touch(TouchMode::None)
            .with_hover(true)
    }

    /// 4K Desktop preset
    #[must_use]
    pub fn desktop_4k() -> DeviceDescriptor {
        DeviceDescriptor::new("Desktop 4K")
            .with_viewport_size(3840, 2160)
            .with_user_agent(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
            )
            .with_device_scale_factor(1.5)
            .with_mobile(false)
            .with_touch(TouchMode::None)
            .with_hover(true)
    }
}

// ============================================================================
// EXTREME TDD: Tests written FIRST per spec
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    mod viewport_tests {
        use super::*;

        #[test]
        fn test_new() {
            let viewport = Viewport::new(800, 600);
            assert_eq!(viewport.width, 800);
            assert_eq!(viewport.height, 600);
        }

        #[test]
        fn test_default() {
            let viewport = Viewport::default();
            assert_eq!(viewport.width, 1920);
            assert_eq!(viewport.height, 1080);
        }

        #[test]
        fn test_landscape() {
            let portrait = Viewport::new(600, 800);
            let landscape = portrait.landscape();
            assert_eq!(landscape.width, 800);
            assert_eq!(landscape.height, 600);

            // Already landscape stays landscape
            let already = Viewport::new(800, 600);
            let still = already.landscape();
            assert_eq!(still.width, 800);
            assert_eq!(still.height, 600);
        }

        #[test]
        fn test_portrait() {
            let landscape = Viewport::new(800, 600);
            let portrait = landscape.portrait();
            assert_eq!(portrait.width, 600);
            assert_eq!(portrait.height, 800);

            // Already portrait stays portrait
            let already = Viewport::new(600, 800);
            let still = already.portrait();
            assert_eq!(still.width, 600);
            assert_eq!(still.height, 800);
        }

        #[test]
        fn test_is_landscape() {
            assert!(Viewport::new(800, 600).is_landscape());
            assert!(!Viewport::new(600, 800).is_landscape());
        }

        #[test]
        fn test_is_portrait() {
            assert!(Viewport::new(600, 800).is_portrait());
            assert!(!Viewport::new(800, 600).is_portrait());
        }
    }

    mod touch_mode_tests {
        use super::*;

        #[test]
        fn test_default() {
            let touch = TouchMode::default();
            assert_eq!(touch, TouchMode::None);
        }

        #[test]
        fn test_is_enabled() {
            assert!(!TouchMode::None.is_enabled());
            assert!(TouchMode::Single.is_enabled());
            assert!(TouchMode::Multi.is_enabled());
        }
    }

    mod device_descriptor_tests {
        use super::*;

        #[test]
        fn test_new() {
            let device = DeviceDescriptor::new("Test Device");
            assert_eq!(device.name, "Test Device");
            assert!(!device.is_mobile);
            assert!(device.user_agent.is_empty());
        }

        #[test]
        fn test_builder_pattern() {
            let device = DeviceDescriptor::new("Custom")
                .with_viewport_size(390, 844)
                .with_user_agent("Mozilla/5.0")
                .with_device_scale_factor(3.0)
                .with_mobile(true)
                .with_touch(TouchMode::Multi)
                .with_hover(false);

            assert_eq!(device.viewport.width, 390);
            assert_eq!(device.viewport.height, 844);
            assert_eq!(device.user_agent, "Mozilla/5.0");
            assert!((device.device_scale_factor - 3.0).abs() < f64::EPSILON);
            assert!(device.is_mobile);
            assert_eq!(device.touch, TouchMode::Multi);
            assert!(!device.has_hover);
        }

        #[test]
        fn test_with_viewport() {
            let viewport = Viewport::new(400, 800);
            let device = DeviceDescriptor::new("Test").with_viewport(viewport);
            assert_eq!(device.viewport, viewport);
        }
    }

    mod device_emulator_tests {
        use super::*;

        #[test]
        fn test_new_has_presets() {
            let emulator = DeviceEmulator::new();
            assert!(!emulator.preset_names().is_empty());
        }

        #[test]
        fn test_get_preset() {
            let emulator = DeviceEmulator::new();
            let iphone = emulator.get_preset("iPhone 14");
            assert!(iphone.is_some());
            let device = iphone.unwrap();
            assert!(device.is_mobile);
            assert_eq!(device.touch, TouchMode::Multi);
        }

        #[test]
        fn test_get_nonexistent_preset() {
            let emulator = DeviceEmulator::new();
            assert!(emulator.get_preset("NonExistent").is_none());
        }

        #[test]
        fn test_register_custom_preset() {
            let mut emulator = DeviceEmulator::new();
            let custom = DeviceDescriptor::new("My Device").with_viewport_size(500, 900);

            emulator.register_preset(custom);

            let device = emulator.get_preset("My Device").unwrap();
            assert_eq!(device.viewport.width, 500);
        }

        #[test]
        fn test_custom_device() {
            let viewport = Viewport::new(500, 800);
            let device = DeviceEmulator::custom(viewport, "Custom UA");
            assert_eq!(device.name, "Custom");
            assert_eq!(device.viewport.width, 500);
            assert_eq!(device.user_agent, "Custom UA");
        }
    }

    mod preset_tests {
        use super::*;

        #[test]
        fn test_iphone_14() {
            let device = DeviceEmulator::iphone_14();
            assert_eq!(device.name, "iPhone 14");
            assert_eq!(device.viewport.width, 390);
            assert_eq!(device.viewport.height, 844);
            assert!((device.device_scale_factor - 3.0).abs() < f64::EPSILON);
            assert!(device.is_mobile);
            assert!(!device.user_agent.is_empty());
        }

        #[test]
        fn test_iphone_14_pro() {
            let device = DeviceEmulator::iphone_14_pro();
            assert_eq!(device.name, "iPhone 14 Pro");
            assert_eq!(device.viewport.width, 393);
            assert!(device.is_mobile);
        }

        #[test]
        fn test_iphone_14_pro_max() {
            let device = DeviceEmulator::iphone_14_pro_max();
            assert_eq!(device.name, "iPhone 14 Pro Max");
            assert_eq!(device.viewport.width, 430);
            assert!(device.is_mobile);
        }

        #[test]
        fn test_ipad_pro() {
            let device = DeviceEmulator::ipad_pro();
            assert_eq!(device.name, "iPad Pro");
            assert_eq!(device.viewport.width, 1024);
            assert!(device.is_mobile);
        }

        #[test]
        fn test_ipad_mini() {
            let device = DeviceEmulator::ipad_mini();
            assert_eq!(device.name, "iPad Mini");
            assert!(device.is_mobile);
        }

        #[test]
        fn test_pixel_7() {
            let device = DeviceEmulator::pixel_7();
            assert_eq!(device.name, "Pixel 7");
            assert_eq!(device.viewport.width, 412);
            assert!(device.is_mobile);
            assert!(device.user_agent.contains("Android"));
        }

        #[test]
        fn test_pixel_7_pro() {
            let device = DeviceEmulator::pixel_7_pro();
            assert_eq!(device.name, "Pixel 7 Pro");
            assert!(device.is_mobile);
        }

        #[test]
        fn test_samsung_galaxy_s23() {
            let device = DeviceEmulator::samsung_galaxy_s23();
            assert_eq!(device.name, "Samsung Galaxy S23");
            assert!(device.is_mobile);
            assert!(device.user_agent.contains("Android"));
        }

        #[test]
        fn test_desktop_1080p() {
            let device = DeviceEmulator::desktop_1080p();
            assert_eq!(device.name, "Desktop 1080p");
            assert_eq!(device.viewport.width, 1920);
            assert_eq!(device.viewport.height, 1080);
            assert!(!device.is_mobile);
            assert!(device.has_hover);
            assert_eq!(device.touch, TouchMode::None);
        }

        #[test]
        fn test_desktop_1440p() {
            let device = DeviceEmulator::desktop_1440p();
            assert_eq!(device.viewport.width, 2560);
            assert!(!device.is_mobile);
        }

        #[test]
        fn test_desktop_4k() {
            let device = DeviceEmulator::desktop_4k();
            assert_eq!(device.viewport.width, 3840);
            assert!(!device.is_mobile);
        }
    }

    mod preset_names {
        use super::*;

        #[test]
        fn test_all_presets_available() {
            let emulator = DeviceEmulator::new();
            let names = emulator.preset_names();

            assert!(names.contains(&"iPhone 14"));
            assert!(names.contains(&"iPhone 14 Pro"));
            assert!(names.contains(&"iPhone 14 Pro Max"));
            assert!(names.contains(&"iPad Pro"));
            assert!(names.contains(&"iPad Mini"));
            assert!(names.contains(&"Pixel 7"));
            assert!(names.contains(&"Pixel 7 Pro"));
            assert!(names.contains(&"Samsung Galaxy S23"));
            assert!(names.contains(&"Desktop 1080p"));
            assert!(names.contains(&"Desktop 1440p"));
            assert!(names.contains(&"Desktop 4K"));
        }
    }

    // =========================================================================
    // H₀ EXTREME TDD: Device Emulation Tests (G.3 P1)
    // =========================================================================

    mod h0_viewport_tests {
        use super::*;

        #[test]
        fn h0_device_01_viewport_new() {
            let viewport = Viewport::new(1024, 768);
            assert_eq!(viewport.width, 1024);
            assert_eq!(viewport.height, 768);
        }

        #[test]
        fn h0_device_02_viewport_default() {
            let viewport = Viewport::default();
            assert_eq!(viewport.width, 1920);
            assert_eq!(viewport.height, 1080);
        }

        #[test]
        fn h0_device_03_viewport_landscape_from_portrait() {
            let portrait = Viewport::new(600, 800);
            let landscape = portrait.landscape();
            assert_eq!(landscape.width, 800);
            assert_eq!(landscape.height, 600);
        }

        #[test]
        fn h0_device_04_viewport_landscape_stays_landscape() {
            let landscape = Viewport::new(800, 600);
            let result = landscape.landscape();
            assert_eq!(result.width, 800);
        }

        #[test]
        fn h0_device_05_viewport_portrait_from_landscape() {
            let landscape = Viewport::new(800, 600);
            let portrait = landscape.portrait();
            assert_eq!(portrait.width, 600);
            assert_eq!(portrait.height, 800);
        }

        #[test]
        fn h0_device_06_viewport_portrait_stays_portrait() {
            let portrait = Viewport::new(600, 800);
            let result = portrait.portrait();
            assert_eq!(result.height, 800);
        }

        #[test]
        fn h0_device_07_viewport_is_landscape() {
            assert!(Viewport::new(1920, 1080).is_landscape());
            assert!(!Viewport::new(1080, 1920).is_landscape());
        }

        #[test]
        fn h0_device_08_viewport_is_portrait() {
            assert!(Viewport::new(1080, 1920).is_portrait());
            assert!(!Viewport::new(1920, 1080).is_portrait());
        }

        #[test]
        fn h0_device_09_viewport_square() {
            let square = Viewport::new(800, 800);
            assert!(!square.is_landscape());
            assert!(!square.is_portrait());
        }

        #[test]
        fn h0_device_10_viewport_equality() {
            let v1 = Viewport::new(800, 600);
            let v2 = Viewport::new(800, 600);
            assert_eq!(v1, v2);
        }
    }

    mod h0_touch_mode_tests {
        use super::*;

        #[test]
        fn h0_device_11_touch_mode_default() {
            let touch = TouchMode::default();
            assert_eq!(touch, TouchMode::None);
        }

        #[test]
        fn h0_device_12_touch_mode_none_not_enabled() {
            assert!(!TouchMode::None.is_enabled());
        }

        #[test]
        fn h0_device_13_touch_mode_single_enabled() {
            assert!(TouchMode::Single.is_enabled());
        }

        #[test]
        fn h0_device_14_touch_mode_multi_enabled() {
            assert!(TouchMode::Multi.is_enabled());
        }

        #[test]
        fn h0_device_15_touch_mode_equality() {
            assert_eq!(TouchMode::Single, TouchMode::Single);
            assert_ne!(TouchMode::Single, TouchMode::Multi);
        }
    }

    mod h0_device_descriptor_tests {
        use super::*;

        #[test]
        fn h0_device_16_descriptor_new() {
            let device = DeviceDescriptor::new("Test");
            assert_eq!(device.name, "Test");
        }

        #[test]
        fn h0_device_17_descriptor_default_viewport() {
            let device = DeviceDescriptor::new("Test");
            assert_eq!(device.viewport.width, 1920);
        }

        #[test]
        fn h0_device_18_descriptor_default_not_mobile() {
            let device = DeviceDescriptor::new("Test");
            assert!(!device.is_mobile);
        }

        #[test]
        fn h0_device_19_descriptor_default_has_hover() {
            let device = DeviceDescriptor::new("Test");
            assert!(device.has_hover);
        }

        #[test]
        fn h0_device_20_descriptor_with_viewport() {
            let viewport = Viewport::new(400, 800);
            let device = DeviceDescriptor::new("Test").with_viewport(viewport);
            assert_eq!(device.viewport.width, 400);
        }

        #[test]
        fn h0_device_21_descriptor_with_viewport_size() {
            let device = DeviceDescriptor::new("Test").with_viewport_size(390, 844);
            assert_eq!(device.viewport.width, 390);
            assert_eq!(device.viewport.height, 844);
        }

        #[test]
        fn h0_device_22_descriptor_with_user_agent() {
            let device = DeviceDescriptor::new("Test").with_user_agent("Mozilla/5.0");
            assert_eq!(device.user_agent, "Mozilla/5.0");
        }

        #[test]
        fn h0_device_23_descriptor_with_scale_factor() {
            let device = DeviceDescriptor::new("Test").with_device_scale_factor(3.0);
            assert!((device.device_scale_factor - 3.0).abs() < f64::EPSILON);
        }

        #[test]
        fn h0_device_24_descriptor_with_mobile() {
            let device = DeviceDescriptor::new("Test").with_mobile(true);
            assert!(device.is_mobile);
        }

        #[test]
        fn h0_device_25_descriptor_with_touch() {
            let device = DeviceDescriptor::new("Test").with_touch(TouchMode::Multi);
            assert_eq!(device.touch, TouchMode::Multi);
        }

        #[test]
        fn h0_device_26_descriptor_with_hover() {
            let device = DeviceDescriptor::new("Test").with_hover(false);
            assert!(!device.has_hover);
        }

        #[test]
        fn h0_device_27_descriptor_builder_chain() {
            let device = DeviceDescriptor::new("iPhone")
                .with_viewport_size(390, 844)
                .with_mobile(true)
                .with_touch(TouchMode::Multi)
                .with_hover(false);
            assert!(device.is_mobile);
            assert!(!device.has_hover);
        }
    }

    mod h0_emulator_tests {
        use super::*;

        #[test]
        fn h0_device_28_emulator_new_has_presets() {
            let emulator = DeviceEmulator::new();
            assert!(!emulator.preset_names().is_empty());
        }

        #[test]
        fn h0_device_29_emulator_get_preset_exists() {
            let emulator = DeviceEmulator::new();
            assert!(emulator.get_preset("iPhone 14").is_some());
        }

        #[test]
        fn h0_device_30_emulator_get_preset_not_exists() {
            let emulator = DeviceEmulator::new();
            assert!(emulator.get_preset("Unknown Device").is_none());
        }

        #[test]
        fn h0_device_31_emulator_register_custom() {
            let mut emulator = DeviceEmulator::new();
            let custom = DeviceDescriptor::new("Custom Device");
            emulator.register_preset(custom);
            assert!(emulator.get_preset("Custom Device").is_some());
        }

        #[test]
        fn h0_device_32_emulator_custom_device() {
            let viewport = Viewport::new(500, 900);
            let device = DeviceEmulator::custom(viewport, "Custom UA");
            assert_eq!(device.name, "Custom");
            assert_eq!(device.user_agent, "Custom UA");
        }

        #[test]
        fn h0_device_33_emulator_preset_count() {
            let emulator = DeviceEmulator::new();
            assert!(emulator.preset_names().len() >= 11);
        }
    }

    mod h0_preset_iphone_tests {
        use super::*;

        #[test]
        fn h0_device_34_iphone_14_viewport() {
            let device = DeviceEmulator::iphone_14();
            assert_eq!(device.viewport.width, 390);
            assert_eq!(device.viewport.height, 844);
        }

        #[test]
        fn h0_device_35_iphone_14_is_mobile() {
            let device = DeviceEmulator::iphone_14();
            assert!(device.is_mobile);
        }

        #[test]
        fn h0_device_36_iphone_14_touch_multi() {
            let device = DeviceEmulator::iphone_14();
            assert_eq!(device.touch, TouchMode::Multi);
        }

        #[test]
        fn h0_device_37_iphone_14_no_hover() {
            let device = DeviceEmulator::iphone_14();
            assert!(!device.has_hover);
        }

        #[test]
        fn h0_device_38_iphone_14_pro_viewport() {
            let device = DeviceEmulator::iphone_14_pro();
            assert_eq!(device.viewport.width, 393);
        }

        #[test]
        fn h0_device_39_iphone_14_pro_max_viewport() {
            let device = DeviceEmulator::iphone_14_pro_max();
            assert_eq!(device.viewport.width, 430);
        }
    }

    mod h0_preset_android_tests {
        use super::*;

        #[test]
        fn h0_device_40_pixel_7_viewport() {
            let device = DeviceEmulator::pixel_7();
            assert_eq!(device.viewport.width, 412);
        }

        #[test]
        fn h0_device_41_pixel_7_user_agent() {
            let device = DeviceEmulator::pixel_7();
            assert!(device.user_agent.contains("Android"));
        }

        #[test]
        fn h0_device_42_pixel_7_pro_scale_factor() {
            let device = DeviceEmulator::pixel_7_pro();
            assert!((device.device_scale_factor - 3.5).abs() < 0.01);
        }

        #[test]
        fn h0_device_43_samsung_galaxy_viewport() {
            let device = DeviceEmulator::samsung_galaxy_s23();
            assert_eq!(device.viewport.width, 360);
        }

        #[test]
        fn h0_device_44_samsung_galaxy_mobile() {
            let device = DeviceEmulator::samsung_galaxy_s23();
            assert!(device.is_mobile);
        }
    }

    mod h0_preset_tablet_tests {
        use super::*;

        #[test]
        fn h0_device_45_ipad_pro_viewport() {
            let device = DeviceEmulator::ipad_pro();
            assert_eq!(device.viewport.width, 1024);
            assert_eq!(device.viewport.height, 1366);
        }

        #[test]
        fn h0_device_46_ipad_mini_viewport() {
            let device = DeviceEmulator::ipad_mini();
            assert_eq!(device.viewport.width, 768);
        }

        #[test]
        fn h0_device_47_ipad_pro_scale_factor() {
            let device = DeviceEmulator::ipad_pro();
            assert!((device.device_scale_factor - 2.0).abs() < 0.01);
        }
    }

    mod h0_preset_desktop_tests {
        use super::*;

        #[test]
        fn h0_device_48_desktop_1080p() {
            let device = DeviceEmulator::desktop_1080p();
            assert_eq!(device.viewport.width, 1920);
            assert_eq!(device.viewport.height, 1080);
            assert!(!device.is_mobile);
        }

        #[test]
        fn h0_device_49_desktop_1440p() {
            let device = DeviceEmulator::desktop_1440p();
            assert_eq!(device.viewport.width, 2560);
            assert_eq!(device.viewport.height, 1440);
        }

        #[test]
        fn h0_device_50_desktop_4k() {
            let device = DeviceEmulator::desktop_4k();
            assert_eq!(device.viewport.width, 3840);
            assert_eq!(device.viewport.height, 2160);
            assert!(device.has_hover);
        }
    }
}
