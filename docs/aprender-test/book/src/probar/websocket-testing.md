# WebSocket Testing

> **Toyota Way**: Genchi Genbutsu (Go and See) - Monitor real-time connections

Monitor and test WebSocket connections with message capture, mocking, and state tracking.

## Running the Example

```bash
cargo run --example websocket_monitor
```

## Quick Start

```rust
use probar::websocket::{WebSocketMonitor, WebSocketMessage};

// Create a WebSocket monitor
let monitor = WebSocketMonitor::new();

// Monitor messages
monitor.on_message(|msg| {
    println!("Message: {} - {:?}", msg.direction, msg.data);
});

// Start monitoring
monitor.start("ws://localhost:8080/game")?;
```

## WebSocket Monitor

```rust
use probar::websocket::{WebSocketMonitor, WebSocketMonitorBuilder};

// Build a monitor with options
let monitor = WebSocketMonitorBuilder::new()
    .capture_binary(true)
    .capture_text(true)
    .max_messages(1000)
    .on_open(|| println!("Connected"))
    .on_close(|| println!("Disconnected"))
    .on_error(|e| eprintln!("Error: {}", e))
    .build();

// Get captured messages
let messages = monitor.messages();
println!("Captured {} messages", messages.len());
```

## WebSocket Messages

```rust
use probar::websocket::{WebSocketMessage, MessageDirection, MessageType};

// Message structure
let message = WebSocketMessage {
    direction: MessageDirection::Incoming,
    message_type: MessageType::Text,
    data: r#"{"action": "move", "x": 100, "y": 200}"#.to_string(),
    timestamp_ms: 1234567890,
};

// Check direction
match message.direction {
    MessageDirection::Incoming => println!("Server → Client"),
    MessageDirection::Outgoing => println!("Client → Server"),
}

// Check type
match message.message_type {
    MessageType::Text => println!("Text message: {}", message.data),
    MessageType::Binary => println!("Binary message ({} bytes)", message.data.len()),
}
```

## Message Direction

```rust
use probar::websocket::MessageDirection;

// Message directions
let directions = [
    MessageDirection::Incoming,  // Server to client
    MessageDirection::Outgoing,  // Client to server
];

// Filter by direction
fn filter_incoming(messages: &[probar::websocket::WebSocketMessage]) -> Vec<&probar::websocket::WebSocketMessage> {
    messages.iter()
        .filter(|m| m.direction == MessageDirection::Incoming)
        .collect()
}
```

## WebSocket State

```rust
use probar::websocket::WebSocketState;

// Connection states
let states = [
    WebSocketState::Connecting,   // Connection in progress
    WebSocketState::Connected,    // Connected and ready
    WebSocketState::Closing,      // Close in progress
    WebSocketState::Closed,       // Connection closed
];

// Monitor state changes
fn describe_state(state: WebSocketState) {
    match state {
        WebSocketState::Connecting => println!("Connecting..."),
        WebSocketState::Connected => println!("Ready to send/receive"),
        WebSocketState::Closing => println!("Closing connection"),
        WebSocketState::Closed => println!("Connection closed"),
    }
}
```

## WebSocket Mocking

```rust
use probar::websocket::{WebSocketMock, MockWebSocketResponse};

// Create a mock WebSocket server
let mock = WebSocketMock::new()
    .on_connect(|| {
        MockWebSocketResponse::send(r#"{"type": "welcome"}"#)
    })
    .on_message("ping", || {
        MockWebSocketResponse::send(r#"{"type": "pong"}"#)
    })
    .on_message_pattern(r"move:(\d+),(\d+)", |captures| {
        let x = captures.get(1).map(|m| m.as_str()).unwrap_or("0");
        let y = captures.get(2).map(|m| m.as_str()).unwrap_or("0");
        MockWebSocketResponse::send(format!(r#"{{"type": "moved", "x": {}, "y": {}}}"#, x, y))
    });

// Use in tests
// let response = mock.handle_message("ping");
// assert_eq!(response.data, r#"{"type": "pong"}"#);
```

## WebSocket Connection

```rust
use probar::websocket::WebSocketConnection;

// Track connection details
let connection = WebSocketConnection {
    url: "ws://localhost:8080/game".to_string(),
    protocol: Some("game-protocol-v1".to_string()),
    state: probar::websocket::WebSocketState::Connected,
    messages_sent: 42,
    messages_received: 38,
    bytes_sent: 2048,
    bytes_received: 1536,
};

println!("URL: {}", connection.url);
println!("Protocol: {:?}", connection.protocol);
println!("Messages: {} sent, {} received",
    connection.messages_sent, connection.messages_received);
```

## Testing Game Protocol

```rust
use probar::websocket::{WebSocketMonitor, MessageDirection};

fn test_game_protocol() {
    let monitor = WebSocketMonitor::new();

    // Connect to game server
    // monitor.start("ws://localhost:8080/game")?;

    // Send player action
    // monitor.send(r#"{"action": "join", "player": "test"}"#)?;

    // Wait for response
    // let response = monitor.wait_for_message(|msg| {
    //     msg.direction == MessageDirection::Incoming
    //         && msg.data.contains("joined")
    // })?;

    // Verify protocol
    // assert!(response.data.contains(r#""status": "ok""#));
}
```

## Message Assertions

```rust
use probar::websocket::{WebSocketMonitor, WebSocketMessage};

fn assert_message_received(monitor: &WebSocketMonitor, expected_type: &str) {
    let messages = monitor.messages();

    let found = messages.iter().any(|msg| {
        msg.data.contains(&format!(r#""type": "{}""#, expected_type))
    });

    assert!(found, "Expected message type '{}' not found", expected_type);
}

fn assert_message_count(monitor: &WebSocketMonitor, expected: usize) {
    let actual = monitor.messages().len();
    assert_eq!(actual, expected,
        "Expected {} messages, got {}", expected, actual);
}
```

## Binary Messages

```rust
use probar::websocket::{WebSocketMessage, MessageType};

// Handle binary messages (e.g., game state updates)
fn handle_binary(message: &WebSocketMessage) {
    if message.message_type == MessageType::Binary {
        // Binary data is base64 encoded
        // let bytes = base64::decode(&message.data)?;
        // Parse game state from bytes
    }
}

// Send binary data
fn send_binary(monitor: &probar::websocket::WebSocketMonitor, data: &[u8]) {
    let encoded = base64::encode(data);
    // monitor.send_binary(encoded)?;
}
```

## Connection Lifecycle

```rust
use probar::websocket::WebSocketMonitor;

fn test_connection_lifecycle() {
    let monitor = WebSocketMonitor::new();

    // Test connection
    // assert!(monitor.connect("ws://localhost:8080").is_ok());
    // assert!(monitor.is_connected());

    // Test messaging
    // monitor.send("hello")?;
    // let response = monitor.wait_for_message()?;

    // Test disconnection
    // monitor.close()?;
    // assert!(!monitor.is_connected());

    // Verify clean shutdown
    // assert!(monitor.close_code() == Some(1000));  // Normal closure
}
```

## Multiplayer Game Testing

```rust
use probar::websocket::WebSocketMonitor;

fn test_multiplayer_sync() {
    let player1 = WebSocketMonitor::new();
    let player2 = WebSocketMonitor::new();

    // Both players connect
    // player1.connect("ws://server/game/room1")?;
    // player2.connect("ws://server/game/room1")?;

    // Player 1 moves
    // player1.send(r#"{"action": "move", "x": 100}"#)?;

    // Player 2 should receive update
    // let update = player2.wait_for_message(|m| m.data.contains("player_moved"))?;
    // assert!(update.data.contains(r#""x": 100"#));
}
```

## Best Practices

1. **Message Validation**: Verify message format before processing
2. **Connection Handling**: Handle reconnection and errors gracefully
3. **Binary vs Text**: Choose appropriate message types for data
4. **Protocol Testing**: Test both client-to-server and server-to-client flows
5. **State Transitions**: Verify connection state changes
6. **Cleanup**: Always close connections in test teardown
