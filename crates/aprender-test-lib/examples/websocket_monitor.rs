//! Example: WebSocket Monitoring (Feature 8)
//!
//! Demonstrates: Monitoring and mocking WebSocket connections
//!
//! Run with: `cargo run --example websocket_monitor`
//!
//! Toyota Way: Mieruka (Visibility) - Clear view of all WebSocket traffic

use jugar_probar::prelude::*;

fn main() -> ProbarResult<()> {
    println!("=== WebSocket Monitoring Example ===\n");

    // 1. Create WebSocket monitor
    println!("1. Creating WebSocket monitor...");
    let mut monitor = WebSocketMonitor::new();
    monitor.start();

    println!("   Monitor created and started");

    // 2. WebSocket connection states
    println!("\n2. WebSocket connection states...");
    let states = [
        WebSocketState::Connecting,
        WebSocketState::Open,
        WebSocketState::Closing,
        WebSocketState::Closed,
    ];

    for state in &states {
        println!("   {:?}", state);
    }

    // 3. Create WebSocket connection
    println!("\n3. Creating WebSocket connection...");
    let conn_id = monitor.connect("wss://example.com/ws");

    println!("   Connection ID: {}", conn_id);

    // 4. Message types
    println!("\n4. WebSocket message types...");
    let text_msg = WebSocketMessage::text("Hello, WebSocket!", MessageDirection::Sent, 100);
    let binary_msg =
        WebSocketMessage::binary(vec![0x01, 0x02, 0x03, 0x04], MessageDirection::Sent, 200);
    let ping_msg = WebSocketMessage::ping(300);
    let pong_msg = WebSocketMessage::pong(400);

    println!(
        "   Text: {:?}, {} chars",
        text_msg.message_type,
        text_msg.data.len()
    );
    println!(
        "   Binary: {:?}, {} bytes",
        binary_msg.message_type,
        binary_msg.data.len()
    );
    println!("   Ping: {:?}", ping_msg.message_type);
    println!("   Pong: {:?}", pong_msg.message_type);

    // 5. Message directions
    println!("\n5. Message directions...");
    let directions = [MessageDirection::Sent, MessageDirection::Received];

    for direction in &directions {
        println!("   {:?}", direction);
    }

    // 6. Send and receive messages
    println!("\n6. Sending and receiving messages...");
    monitor.send(&conn_id, "Hello from client!");
    monitor.receive(&conn_id, "Hello from server!");

    println!("   Sent: Hello from client!");
    println!("   Received: Hello from server!");

    // 7. Create mock WebSocket response
    println!("\n7. Creating mock WebSocket responses...");
    let _mock_response = MockWebSocketResponse::new().with_text("Mock response data");

    println!("   Mock response created");

    // 8. WebSocket mock for pattern matching
    println!("\n8. Creating WebSocket mock...");
    let mock = WebSocketMock::new("wss://api.example.com/*")
        .on_open(MockWebSocketResponse::new().with_text("Welcome!"))
        .on_message("ping", MockWebSocketResponse::new().with_text("pong"));

    println!("   Mock pattern: {}", mock.url_pattern);

    monitor.mock(mock);

    // 9. Message type enum
    println!("\n9. Message type variants...");
    let types = [
        MessageType::Text,
        MessageType::Binary,
        MessageType::Ping,
        MessageType::Pong,
        MessageType::Close,
    ];

    for msg_type in &types {
        println!("   {:?}", msg_type);
    }

    // 10. View all captured messages
    println!("\n10. Viewing captured messages...");
    let messages = monitor.all_messages();
    println!("   Total messages captured: {}", messages.len());

    for (i, msg) in messages.iter().enumerate() {
        let direction = match msg.direction {
            MessageDirection::Sent => "->",
            MessageDirection::Received => "<-",
        };
        match &msg.message_type {
            MessageType::Text => {
                println!("   {} Message {}: Text \"{}\"", direction, i + 1, &msg.data);
            }
            _ => println!("   {} Message {}: {:?}", direction, i + 1, msg.message_type),
        }
    }

    // 11. Cleanup
    println!("\n11. Cleanup...");
    monitor.disconnect(&conn_id, 1000, "Normal closure");
    monitor.stop();

    println!("   Connection closed, monitor stopped");

    println!("\n✅ WebSocket monitoring example completed!");
    Ok(())
}
