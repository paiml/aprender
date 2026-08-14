# Device Emulation

> **Toyota Way**: Poka-Yoke (Mistake-Proofing) - Test on real device profiles

Emulate mobile and desktop devices for responsive testing with type-safe viewport and device configuration.

## Running the Example

```bash
cargo run --example locator_demo
```

## Quick Start

```rust
use probar::emulation::{DeviceDescriptor, TouchMode, Viewport};

// Create a custom device
let iphone = DeviceDescriptor::new("iPhone 14 Pro")
    .with_viewport_size(393, 852)
    .with_device_scale_factor(3.0)
    .with_mobile(true)
    .with_touch(TouchMode::Multi)
    .with_user_agent("Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)");

// Use preset devices
let iphone_preset = DeviceDescriptor::iphone_14_pro();
let pixel_preset = DeviceDescriptor::pixel_7();
let ipad_preset = DeviceDescriptor::ipad_pro_12_9();
```

## Viewport Management

```rust
use probar::emulation::Viewport;

// Create viewports
let desktop = Viewport::new(1920, 1080);
let tablet = Viewport::new(768, 1024);
let mobile = Viewport::new(375, 812);

// Orientation helpers
let landscape = tablet.landscape();  // 1024x768
let portrait = tablet.portrait();    // 768x1024

// Check orientation
assert!(desktop.is_landscape());
assert!(mobile.is_portrait());
```

## Touch Mode Configuration

```rust
use probar::emulation::TouchMode;

// Touch modes available
let no_touch = TouchMode::None;      // Desktop without touch
let single = TouchMode::Single;       // Basic touch (e.g., older tablets)
let multi = TouchMode::Multi;         // Multi-touch (modern phones/tablets)

// Check if touch is enabled
assert!(!no_touch.is_enabled());
assert!(multi.is_enabled());
```

## Device Presets

Probar includes accurate presets for popular devices:

| Device | Viewport | Scale | Mobile | Touch |
|--------|----------|-------|--------|-------|
| iPhone 14 Pro | 393×852 | 3.0 | Yes | Multi |
| iPhone 14 Pro Max | 430×932 | 3.0 | Yes | Multi |
| Pixel 7 | 412×915 | 2.625 | Yes | Multi |
| iPad Pro 12.9" | 1024×1366 | 2.0 | Yes | Multi |
| Samsung Galaxy S23 | 360×780 | 3.0 | Yes | Multi |
| MacBook Pro 16" | 1728×1117 | 2.0 | No | None |

```rust
use probar::emulation::DeviceDescriptor;

// Mobile devices
let iphone = DeviceDescriptor::iphone_14_pro();
let pixel = DeviceDescriptor::pixel_7();
let galaxy = DeviceDescriptor::galaxy_s23();

// Tablets
let ipad = DeviceDescriptor::ipad_pro_12_9();

// Desktop
let macbook = DeviceDescriptor::macbook_pro_16();
```

## Custom Device Configuration

```rust
use probar::emulation::{DeviceDescriptor, TouchMode, Viewport};

// Full custom configuration
let gaming_device = DeviceDescriptor::new("Steam Deck")
    .with_viewport(Viewport::new(1280, 800))
    .with_device_scale_factor(1.0)
    .with_mobile(false)  // Not a phone
    .with_touch(TouchMode::Single)
    .with_hover(true)    // Has cursor
    .with_user_agent("Mozilla/5.0 (X11; Linux x86_64; Steam Deck)");

// Access device properties
println!("Device: {}", gaming_device.name);
println!("Viewport: {}x{}",
    gaming_device.viewport.width,
    gaming_device.viewport.height);
println!("Is mobile: {}", gaming_device.is_mobile);
println!("Touch: {:?}", gaming_device.touch);
```

## Device Emulator Usage

```rust
use probar::emulation::{DeviceEmulator, DeviceDescriptor};

// Create emulator
let mut emulator = DeviceEmulator::new();

// Register devices
emulator.register("iphone", DeviceDescriptor::iphone_14_pro());
emulator.register("pixel", DeviceDescriptor::pixel_7());
emulator.register("desktop", DeviceDescriptor::macbook_pro_16());

// Get device by name
if let Some(device) = emulator.get("iphone") {
    println!("Testing on: {}", device.name);
}

// List all registered devices
for name in emulator.device_names() {
    println!("- {}", name);
}
```

## Testing Responsive Layouts

```rust
use probar::emulation::{DeviceDescriptor, Viewport};

// Test breakpoints
let breakpoints = [
    ("mobile", Viewport::new(320, 568)),
    ("tablet", Viewport::new(768, 1024)),
    ("desktop", Viewport::new(1440, 900)),
    ("wide", Viewport::new(1920, 1080)),
];

for (name, viewport) in breakpoints {
    let device = DeviceDescriptor::new(name)
        .with_viewport(viewport);

    // Run tests at this viewport size
    println!("Testing at {} ({}x{})",
        name, viewport.width, viewport.height);
}
```

## Best Practices

1. **Use Presets**: Start with device presets for accurate real-world testing
2. **Test Orientations**: Use `.landscape()` and `.portrait()` helpers
3. **Consider Touch**: Ensure touch-specific interactions work correctly
4. **Test Scale Factors**: High-DPI displays may reveal rendering issues
5. **Mobile User Agents**: Some features depend on UA string detection
