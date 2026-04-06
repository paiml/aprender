# Geolocation Mocking

> **Toyota Way**: Poka-Yoke (Mistake-Proofing) - Deterministic location testing

Mock GPS coordinates and location data for testing location-based features with type-safe coordinate validation.

## Running the Example

```bash
cargo run --example locator_demo
```

## Quick Start

```rust
use probar::emulation::GeolocationPosition;

// Create a custom position
let position = GeolocationPosition::new(
    37.7749,   // latitude
    -122.4194, // longitude
    10.0       // accuracy in meters
);

// Use preset locations
let nyc = GeolocationPosition::new_york();
let tokyo = GeolocationPosition::tokyo();
let london = GeolocationPosition::london();
```

## Geographic Position

```rust
use probar::emulation::GeolocationPosition;

// Basic position with coordinates and accuracy
let basic = GeolocationPosition::new(40.758896, -73.985130, 10.0);

// Position with full data
let detailed = GeolocationPosition::new(37.820587, -122.478264, 5.0)
    .with_altitude(67.0, 3.0)     // altitude: 67m, accuracy: 3m
    .with_heading(45.0)           // heading: 45 degrees (NE)
    .with_speed(1.5);             // speed: 1.5 m/s (walking)

// Access position data
println!("Latitude: {}", detailed.latitude);
println!("Longitude: {}", detailed.longitude);
println!("Accuracy: {}m", detailed.accuracy);
println!("Altitude: {:?}m", detailed.altitude);
println!("Heading: {:?}°", detailed.heading);
println!("Speed: {:?} m/s", detailed.speed);
```

## Preset Locations

Probar includes accurate coordinates for major world cities:

| City | Landmark | Coordinates |
|------|----------|-------------|
| New York | Times Square | 40.7589°N, 73.9851°W |
| Tokyo | Shibuya Crossing | 35.6595°N, 139.7005°E |
| London | Trafalgar Square | 51.5080°N, 0.1281°W |
| Paris | Eiffel Tower | 48.8584°N, 2.2945°E |
| Sydney | Opera House | 33.8568°S, 151.2153°E |
| San Francisco | Golden Gate Bridge | 37.8206°N, 122.4783°W |
| Berlin | Brandenburg Gate | 52.5163°N, 13.3777°E |
| Singapore | Marina Bay Sands | 1.2834°N, 103.8604°E |
| Dubai | Burj Khalifa | 25.1972°N, 55.2744°E |
| São Paulo | Paulista Avenue | 23.5632°S, 46.6543°W |

```rust
use probar::emulation::GeolocationPosition;

// Major city presets
let new_york = GeolocationPosition::new_york();
let tokyo = GeolocationPosition::tokyo();
let london = GeolocationPosition::london();
let paris = GeolocationPosition::paris();
let sydney = GeolocationPosition::sydney();
let san_francisco = GeolocationPosition::san_francisco();
let berlin = GeolocationPosition::berlin();
let singapore = GeolocationPosition::singapore();
let dubai = GeolocationPosition::dubai();
let sao_paulo = GeolocationPosition::sao_paulo();
```

## Geolocation Mock System

```rust
use probar::emulation::{GeolocationMock, GeolocationPosition};

// Create mock geolocation system
let mut mock = GeolocationMock::new();

// Set initial position
mock.set_position(GeolocationPosition::tokyo());

// Get current position
let current = mock.current_position();
println!("Current: {:.4}°N, {:.4}°E",
    current.latitude, current.longitude);

// Simulate position error
mock.set_error("Position unavailable");
assert!(mock.current_error().is_some());

// Clear error
mock.clear_error();
assert!(mock.current_error().is_none());
```

## Movement Simulation

```rust
use probar::emulation::{GeolocationMock, GeolocationPosition};

let mut mock = GeolocationMock::new();

// Define a route (e.g., walking through a city)
let route = [
    GeolocationPosition::new(40.758896, -73.985130, 10.0), // Times Square
    GeolocationPosition::new(40.762093, -73.979112, 10.0), // 5th Ave
    GeolocationPosition::new(40.764912, -73.973017, 10.0), // Central Park
];

// Add waypoints
for position in &route {
    mock.add_waypoint(position.clone());
}

// Simulate movement along route
while mock.has_waypoints() {
    mock.advance_to_next_waypoint();
    let pos = mock.current_position();
    println!("Now at: {:.4}°N, {:.4}°W", pos.latitude, pos.longitude);
}
```

## Testing Location-Based Features

```rust
use probar::emulation::{GeolocationMock, GeolocationPosition};

fn test_location_based_content() {
    let mut geo = GeolocationMock::new();

    // Test US content
    geo.set_position(GeolocationPosition::new_york());
    // assert!(app.shows_us_content());

    // Test EU content
    geo.set_position(GeolocationPosition::berlin());
    // assert!(app.shows_eu_content());

    // Test Asia content
    geo.set_position(GeolocationPosition::tokyo());
    // assert!(app.shows_asia_content());
}

fn test_geofencing() {
    let mut geo = GeolocationMock::new();

    // Inside geofence
    geo.set_position(GeolocationPosition::new(
        37.7749, -122.4194, 10.0  // SF downtown
    ));
    // assert!(app.is_in_service_area());

    // Outside geofence
    geo.set_position(GeolocationPosition::new(
        40.7128, -74.0060, 10.0  // NYC
    ));
    // assert!(!app.is_in_service_area());
}
```

## Coordinate Validation

Probar's type system ensures coordinates are always valid:

```rust
use probar::emulation::GeolocationPosition;

// Valid coordinates work
let valid = GeolocationPosition::new(45.0, 90.0, 10.0);

// Invalid latitude (must be -90 to 90) - panics in debug
// let invalid = GeolocationPosition::new(91.0, 0.0, 10.0);

// Invalid longitude (must be -180 to 180) - panics in debug
// let invalid = GeolocationPosition::new(0.0, 181.0, 10.0);

// Invalid accuracy (must be non-negative) - panics in debug
// let invalid = GeolocationPosition::new(0.0, 0.0, -1.0);
```

## Best Practices

1. **Use Presets**: Start with city presets for realistic testing
2. **Test Edge Cases**: Test equator (0,0), poles, and date line
3. **Accuracy Matters**: Different accuracy values affect UX decisions
4. **Simulate Errors**: Test "permission denied" and "position unavailable"
5. **Movement Testing**: Use waypoints to test location tracking features
