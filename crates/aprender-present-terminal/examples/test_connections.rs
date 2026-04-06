//! Test connections analyzer
use presentar_terminal::ptop::analyzers::{Analyzer, ConnectionsAnalyzer, TcpState};

fn main() {
    println!("Testing ConnectionsAnalyzer...");

    let mut analyzer = ConnectionsAnalyzer::new();

    // Collect data
    if let Err(e) = analyzer.collect() {
        println!("Error collecting: {:?}", e);
        return;
    }

    let data = analyzer.data();
    println!("Total connections: {}", data.connections.len());

    let established = data
        .connections
        .iter()
        .filter(|c| c.state == TcpState::Established)
        .count();
    let listen = data
        .connections
        .iter()
        .filter(|c| c.state == TcpState::Listen)
        .count();

    println!("Established: {}", established);
    println!("Listen: {}", listen);

    println!("\nFirst 10 connections:");
    for (i, conn) in data.connections.iter().take(10).enumerate() {
        println!(
            "{}: {} {} -> {} ({})",
            i,
            conn.state.short(),
            conn.local_display(),
            conn.remote_display(),
            conn.process_display()
        );
    }
}
