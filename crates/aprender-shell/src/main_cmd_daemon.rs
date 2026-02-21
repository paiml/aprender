
/// Check daemon status
fn cmd_daemon_status(socket_path: &std::path::Path) {
    #[cfg(unix)]
    {
        use std::io::{BufRead, BufReader, Write};
        use std::os::unix::net::UnixStream;

        if !socket_path.exists() {
            println!("❌ Daemon not running (socket not found)");
            std::process::exit(1);
        }

        let mut stream = match UnixStream::connect(socket_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("❌ Could not connect to daemon: {e}");
                eprintln!("   Socket exists but daemon may have crashed");
                std::process::exit(1);
            }
        };

        // Send STATS command
        writeln!(stream, "STATS").ok();
        stream.flush().ok();

        println!("✅ Daemon is running");
        println!("   Socket: {}", socket_path.display());

        // Read PID if available
        let pid_path = socket_path.with_extension("pid");
        if let Ok(pid) = std::fs::read_to_string(&pid_path) {
            println!("   PID:    {}", pid.trim());
        }

        // Read stats
        let reader = BufReader::new(&stream);
        println!("\n📊 Statistics:");
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            if line.is_empty() {
                break;
            }
            println!("   {line}");
        }
    }

    #[cfg(not(unix))]
    {
        let _ = socket_path;
        eprintln!("❌ Daemon mode is only supported on Unix systems");
        std::process::exit(1);
    }
}
