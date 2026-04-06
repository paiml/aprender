//! WebSocket handler for real-time event streaming.
//!
//! GET /api/v1/ws — upgrade to WebSocket, receive JSON events.

use axum::{
    extract::{
        ws::{Message, WebSocket},
        State, WebSocketUpgrade,
    },
    response::Response,
};

use super::state::BancoState;

/// GET /api/v1/ws — WebSocket upgrade for real-time events.
pub async fn ws_handler(ws: WebSocketUpgrade, State(state): State<BancoState>) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Handle a connected WebSocket — forward events from the bus.
async fn handle_socket(mut socket: WebSocket, state: BancoState) {
    let mut rx = state.events.subscribe();

    // Send a welcome message
    let welcome = serde_json::json!({
        "type": "connected",
        "data": {
            "endpoints": 66,
            "model_loaded": state.model.is_loaded(),
        }
    });
    if socket.send(Message::Text(welcome.to_string())).await.is_err() {
        return;
    }

    // Forward events until the client disconnects
    loop {
        tokio::select! {
            // Event from bus → send to client
            event = rx.recv() => {
                match event {
                    Ok(json) => {
                        if socket.send(Message::Text(json)).await.is_err() {
                            break; // Client disconnected
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        // Client too slow — notify and continue
                        let lag_msg = serde_json::json!({
                            "type": "system_event",
                            "data": {"message": format!("Missed {n} events (slow consumer)")}
                        });
                        let _ = socket.send(Message::Text(lag_msg.to_string())).await;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
            // Client message (ping/pong handled by axum, we ignore text)
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Ping(data))) => {
                        let _ = socket.send(Message::Pong(data)).await;
                    }
                    _ => {} // Ignore other messages
                }
            }
        }
    }
}
