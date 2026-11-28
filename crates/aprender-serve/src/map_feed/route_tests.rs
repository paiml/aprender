//! Integration test: a real WebSocket client against the route on 127.0.0.1:0.
//!
//! Each test: client connects, applies frames, is dropped; the server applies more batches;
//! the client reconnects with its last seq and must end with the server's column hash.
//! Replay path: the ring still covers the gap. Snapshot path: the ring overflowed.

use std::time::Duration;

use futures::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

use crate::map_feed::codec::{decode_server_frame, encode_hello, ServerFrame};
use crate::map_feed::column_hash;
use crate::map_feed::route::{router, FeedHub};

/// Headless client mirror of one column, driven only by decoded frames.
struct Mirror {
    seq: u64,
    values: Vec<f32>,
}

impl Mirror {
    fn new(cells: usize) -> Self {
        Self {
            seq: 0,
            values: vec![0.0; cells],
        }
    }

    fn apply(&mut self, frame: ServerFrame) -> Result<(), String> {
        match frame {
            ServerFrame::Batch(batch) => {
                if batch.seq != self.seq + 1 {
                    return Err(format!(
                        "gap: mirror at {} got batch {}",
                        self.seq, batch.seq
                    ));
                }
                for (cell, value) in batch.changes {
                    let slot = usize::try_from(cell)
                        .ok()
                        .and_then(|i| self.values.get_mut(i))
                        .ok_or_else(|| format!("cell {cell} out of range"))?;
                    *slot = value;
                }
                self.seq = batch.seq;
                Ok(())
            },
            ServerFrame::Snapshot { seq, hash, values } => {
                if column_hash(&values) != hash {
                    return Err("snapshot hash does not match its values".to_owned());
                }
                self.seq = seq;
                self.values = values;
                Ok(())
            },
        }
    }

    fn hash(&self) -> u64 {
        column_hash(&self.values)
    }
}

type Socket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Apply `n` deterministic batches to column "col", continuing from the hub's current seq.
fn apply_n(hub: &FeedHub, n: u32) {
    for _ in 0..n {
        let seq = hub.with_feed(|f| f.seq("col").expect("seq"));
        let cell = u32::try_from(seq % 4).expect("cell fits u32");
        hub.apply("col", &[(cell, seq as f32 * 0.5 - 3.0)])
            .expect("apply");
    }
}

async fn spawn_server(hub: FeedHub) -> std::net::SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind 127.0.0.1:0");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(listener, router(hub)).await.expect("serve");
    });
    addr
}

async fn connect(addr: std::net::SocketAddr, column: &str, last_seen: u64) -> Socket {
    let url = format!("ws://{addr}/map/feed/{column}");
    let (mut ws, _response) = tokio_tungstenite::connect_async(url)
        .await
        .expect("websocket connect");
    ws.send(Message::Binary(encode_hello(last_seen)))
        .await
        .expect("send hello");
    ws
}

/// Read exactly `n` binary frames, failing rather than hanging if the server stalls.
async fn read_frames(ws: &mut Socket, n: usize) -> Vec<ServerFrame> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let next = tokio::time::timeout(Duration::from_secs(10), ws.next())
            .await
            .expect("server sent a frame in time");
        match next {
            Some(Ok(Message::Binary(bytes))) => {
                out.push(decode_server_frame(&bytes).expect("server frame decodes"));
            },
            other => panic!("expected a binary frame, got {other:?}"),
        }
    }
    out
}

fn server_state(hub: &FeedHub) -> (u64, u64) {
    hub.with_feed(|f| {
        (
            f.seq("col").expect("seq"),
            column_hash(f.values("col").expect("values")),
        )
    })
}

#[tokio::test]
async fn map_feed_route_replay_path_reconnects_to_server_hash() {
    let hub = FeedHub::new(8, 16);
    hub.add_column("col", 4).expect("add column");
    apply_n(&hub, 3);
    let addr = spawn_server(hub.clone()).await;

    let mut mirror = Mirror::new(4);
    let mut ws = connect(addr, "col", 0).await;
    for frame in read_frames(&mut ws, 3).await {
        mirror.apply(frame).expect("first connection applies");
    }
    assert_eq!(mirror.seq, 3);
    drop(ws);

    apply_n(&hub, 4); // seq 4..=7 while the client is away; the ring still holds them
    let mut ws = connect(addr, "col", mirror.seq).await;
    let frames = read_frames(&mut ws, 4).await;
    assert!(
        frames.iter().all(|f| matches!(f, ServerFrame::Batch(_))),
        "replay path must send batches, not a snapshot"
    );
    for frame in frames {
        mirror.apply(frame).expect("replay applies in order");
    }

    // Live forwarding after the resume.
    apply_n(&hub, 1);
    for frame in read_frames(&mut ws, 1).await {
        mirror.apply(frame).expect("live batch applies");
    }

    let (server_seq, server_hash) = server_state(&hub);
    assert_eq!(mirror.seq, server_seq);
    assert_eq!(mirror.hash(), server_hash);
}

#[tokio::test]
async fn map_feed_route_snapshot_path_reconnects_to_server_hash() {
    let hub = FeedHub::new(2, 16); // ring of 2: a 10-batch gap overflows it
    hub.add_column("col", 4).expect("add column");
    apply_n(&hub, 2);
    let addr = spawn_server(hub.clone()).await;

    let mut mirror = Mirror::new(4);
    let mut ws = connect(addr, "col", 0).await;
    for frame in read_frames(&mut ws, 2).await {
        mirror.apply(frame).expect("first connection applies");
    }
    assert_eq!(mirror.seq, 2);
    drop(ws);

    apply_n(&hub, 10); // seq 3..=12; only 11 and 12 survive in the ring
    let mut ws = connect(addr, "col", mirror.seq).await;
    let frames = read_frames(&mut ws, 1).await;
    assert!(
        matches!(frames[0], ServerFrame::Snapshot { seq: 12, .. }),
        "overflowed ring must answer with a snapshot at seq 12, got {:?}",
        frames[0]
    );
    for frame in frames {
        mirror.apply(frame).expect("snapshot applies");
    }

    apply_n(&hub, 1);
    for frame in read_frames(&mut ws, 1).await {
        mirror
            .apply(frame)
            .expect("live batch after snapshot applies");
    }

    let (server_seq, server_hash) = server_state(&hub);
    assert_eq!(mirror.seq, server_seq);
    assert_eq!(mirror.hash(), server_hash);
}
