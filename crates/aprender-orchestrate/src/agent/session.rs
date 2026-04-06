//! Session persistence for `apr code`.
//!
//! Serializes conversation history to `~/.apr/sessions/{id}/messages.jsonl`
//! for crash recovery and `/resume`. Each message is one JSON line.
//!
//! See: apr-code.md §6

use std::fs;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::driver::Message;

/// Session manifest stored alongside messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionManifest {
    /// Unique session ID.
    pub id: String,
    /// Agent name (e.g., "apr-code").
    pub agent: String,
    /// Working directory at session start.
    pub cwd: String,
    /// Timestamp of session creation (ISO 8601).
    pub created: String,
    /// Number of turns completed.
    pub turns: u32,
}

/// Persistent session storage backed by JSONL files.
pub struct SessionStore {
    /// Root directory for this session.
    pub dir: PathBuf,
    /// Session manifest.
    pub manifest: SessionManifest,
}

impl SessionStore {
    /// Create a new session in `~/.apr/sessions/{id}/`.
    pub fn create(agent_name: &str) -> anyhow::Result<Self> {
        let id = generate_session_id();
        let sessions_dir = sessions_root()?;
        let dir = sessions_dir.join(&id);
        fs::create_dir_all(&dir)?;

        let cwd =
            std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_else(|_| ".".into());

        let manifest = SessionManifest {
            id,
            agent: agent_name.to_string(),
            cwd,
            created: chrono_now(),
            turns: 0,
        };

        // Write manifest
        let manifest_path = dir.join("manifest.json");
        let json = serde_json::to_string_pretty(&manifest)?;
        fs::write(&manifest_path, json)?;

        Ok(Self { dir, manifest })
    }

    /// Resume an existing session by ID.
    pub fn resume(session_id: &str) -> anyhow::Result<Self> {
        let dir = sessions_root()?.join(session_id);
        if !dir.is_dir() {
            anyhow::bail!("session not found: {session_id}");
        }

        let manifest_path = dir.join("manifest.json");
        let json = fs::read_to_string(&manifest_path)?;
        let manifest: SessionManifest = serde_json::from_str(&json)?;

        Ok(Self { dir, manifest })
    }

    /// Find the most recent session for the current working directory.
    ///
    /// Only returns sessions modified within `max_age` (default 24h).
    /// PMAT-165: age filter prevents offering stale sessions.
    pub fn find_recent_for_cwd() -> Option<SessionManifest> {
        Self::find_recent_for_cwd_within(std::time::Duration::from_secs(24 * 3600))
    }

    /// Find the most recent session for cwd within the given age limit.
    pub fn find_recent_for_cwd_within(max_age: std::time::Duration) -> Option<SessionManifest> {
        let sessions_dir = sessions_root().ok()?;
        if !sessions_dir.is_dir() {
            return None;
        }

        let cwd = std::env::current_dir().ok()?.display().to_string();
        let now = std::time::SystemTime::now();

        let mut best: Option<(SessionManifest, std::time::SystemTime)> = None;
        for entry in fs::read_dir(&sessions_dir).ok()?.flatten() {
            let manifest_path = entry.path().join("manifest.json");
            if !manifest_path.is_file() {
                continue;
            }
            if let Ok(json) = fs::read_to_string(&manifest_path) {
                if let Ok(m) = serde_json::from_str::<SessionManifest>(&json) {
                    if m.cwd == cwd && m.turns > 0 {
                        let mtime = entry.metadata().ok()?.modified().ok()?;
                        // PMAT-165: skip sessions older than max_age
                        if now.duration_since(mtime).unwrap_or(max_age) >= max_age {
                            continue;
                        }
                        if best.as_ref().is_none_or(|(_, t)| mtime > *t) {
                            best = Some((m, mtime));
                        }
                    }
                }
            }
        }

        best.map(|(m, _)| m)
    }

    /// Session ID.
    pub fn id(&self) -> &str {
        &self.manifest.id
    }

    /// Append a message to the JSONL log.
    pub fn append_message(&self, msg: &Message) -> anyhow::Result<()> {
        let path = self.dir.join("messages.jsonl");
        let mut file = fs::OpenOptions::new().create(true).append(true).open(&path)?;
        let json = serde_json::to_string(msg)?;
        writeln!(file, "{json}")?;
        Ok(())
    }

    /// Append multiple messages (e.g., after a turn completes).
    pub fn append_messages(&self, msgs: &[Message]) -> anyhow::Result<()> {
        let path = self.dir.join("messages.jsonl");
        let mut file = fs::OpenOptions::new().create(true).append(true).open(&path)?;
        for msg in msgs {
            let json = serde_json::to_string(msg)?;
            writeln!(file, "{json}")?;
        }
        Ok(())
    }

    /// Load all messages from the JSONL log.
    pub fn load_messages(&self) -> anyhow::Result<Vec<Message>> {
        let path = self.dir.join("messages.jsonl");
        if !path.is_file() {
            return Ok(Vec::new());
        }

        let file = fs::File::open(&path)?;
        let reader = std::io::BufReader::new(file);
        let mut messages = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let msg: Message = serde_json::from_str(&line)?;
            messages.push(msg);
        }

        Ok(messages)
    }

    /// Update the turn count in the manifest.
    pub fn record_turn(&mut self) -> anyhow::Result<()> {
        self.manifest.turns += 1;
        let manifest_path = self.dir.join("manifest.json");
        let json = serde_json::to_string_pretty(&self.manifest)?;
        fs::write(&manifest_path, json)?;
        Ok(())
    }
}

/// Root directory for all sessions.
fn sessions_root() -> anyhow::Result<PathBuf> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("cannot determine home directory"))?;
    Ok(home.join(".apr").join("sessions"))
}

/// Generate a short session ID from timestamp + nanos for uniqueness.
fn generate_session_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let ts = dur.as_secs();
    let nanos = dur.subsec_nanos();
    format!("{ts:x}-{nanos:08x}")
}

/// PMAT-165/174: Interactive auto-resume prompt for recent sessions.
///
/// Checks if stdin is a TTY (skips when piped). Finds sessions <24h old
/// for cwd. Shows [Y/n] prompt with age display.
pub fn offer_auto_resume() -> Option<String> {
    if !std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        return None;
    }
    let manifest = SessionStore::find_recent_for_cwd()?;
    let age = manifest
        .created
        .parse::<chrono::DateTime<chrono::Utc>>()
        .ok()
        .map(|created| {
            let elapsed = chrono::Utc::now().signed_duration_since(created);
            if elapsed.num_hours() > 0 {
                format!("{}h ago", elapsed.num_hours())
            } else {
                format!("{}m ago", elapsed.num_minutes().max(1))
            }
        })
        .unwrap_or_else(|| "recently".to_string());
    eprintln!("  Found previous session ({age}, {} turns)", manifest.turns);
    eprint!("  Resume? [Y/n] ");
    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_err() {
        return None;
    }
    let input = input.trim().to_lowercase();
    if input.is_empty() || input == "y" || input == "yes" {
        Some(manifest.id)
    } else {
        None
    }
}

/// Current UTC time as ISO 8601 string (no chrono dependency).
fn chrono_now() -> String {
    // Simple UTC timestamp without external dependency
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    // Approximate: days since epoch, then h/m/s
    let days = secs / 86400;
    let rem = secs % 86400;
    let h = rem / 3600;
    let m = (rem % 3600) / 60;
    let s = rem % 60;
    // Year calculation (good enough for display)
    let years = 1970 + days / 365;
    let day_of_year = days % 365;
    let month = day_of_year / 30 + 1;
    let day = day_of_year % 30 + 1;
    format!("{years:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a session store in a temp dir (isolated from ~/.apr/sessions/).
    fn create_test_store() -> SessionStore {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let id = generate_session_id();
        let tmp_path = tmp.path().to_path_buf();
        // Prevent cleanup so test can use the dir
        std::mem::forget(tmp);
        let dir = tmp_path.join(&id);
        fs::create_dir_all(&dir).expect("mkdir");

        let manifest = SessionManifest {
            id,
            agent: "test-agent".into(),
            cwd: ".".into(),
            created: chrono_now(),
            turns: 0,
        };
        let json = serde_json::to_string_pretty(&manifest).expect("json");
        fs::write(dir.join("manifest.json"), json).expect("write");
        SessionStore { dir, manifest }
    }

    #[test]
    fn test_session_create_and_persist() {
        let store = create_test_store();
        assert!(!store.id().is_empty());
        assert!(store.dir.is_dir());

        store.append_message(&Message::User("hello".into())).expect("append");
        store.append_message(&Message::Assistant("hi".into())).expect("append");

        let msgs = store.load_messages().expect("load");
        assert_eq!(msgs.len(), 2);
        assert!(matches!(&msgs[0], Message::User(s) if s == "hello"));
        assert!(matches!(&msgs[1], Message::Assistant(s) if s == "hi"));

        let _ = fs::remove_dir_all(&store.dir);
    }

    #[test]
    fn test_session_resume_by_path() {
        let store = create_test_store();
        store.append_message(&Message::User("test".into())).expect("append");

        // Resume by reading from same dir
        let manifest_json = fs::read_to_string(store.dir.join("manifest.json")).expect("read");
        let manifest: SessionManifest = serde_json::from_str(&manifest_json).expect("parse");
        let resumed = SessionStore { dir: store.dir.clone(), manifest };
        let msgs = resumed.load_messages().expect("load");
        assert_eq!(msgs.len(), 1);

        let _ = fs::remove_dir_all(&store.dir);
    }

    #[test]
    fn test_session_resume_nonexistent() {
        let result = SessionStore::resume("nonexistent-id-12345");
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_session_id_unique() {
        let id1 = generate_session_id();
        // Small sleep to ensure different nanos
        std::thread::sleep(std::time::Duration::from_millis(1));
        let id2 = generate_session_id();
        assert_ne!(id1, id2, "IDs should be unique");
        assert!(id1.contains('-'));
    }

    #[test]
    fn test_append_and_load_empty() {
        let store = create_test_store();
        let msgs = store.load_messages().expect("load");
        assert!(msgs.is_empty());
        let _ = fs::remove_dir_all(&store.dir);
    }

    #[test]
    fn test_record_turn() {
        let mut store = create_test_store();
        assert_eq!(store.manifest.turns, 0);
        store.record_turn().expect("record");
        assert_eq!(store.manifest.turns, 1);

        // Reload manifest from disk
        let json = fs::read_to_string(store.dir.join("manifest.json")).expect("read");
        let reloaded: SessionManifest = serde_json::from_str(&json).expect("parse");
        assert_eq!(reloaded.turns, 1);

        let _ = fs::remove_dir_all(&store.dir);
    }

    // PMAT-165: age filter tests
    #[test]
    fn test_find_recent_for_cwd_within_zero_age() {
        // With zero max_age, nothing should match
        let result = SessionStore::find_recent_for_cwd_within(std::time::Duration::ZERO);
        assert!(result.is_none(), "zero age should return nothing");
    }

    #[test]
    fn test_find_recent_for_cwd_delegates_to_within() {
        // find_recent_for_cwd uses 24h, should not panic
        let _ = SessionStore::find_recent_for_cwd();
    }

    // ═══ apr-chat-session-v1 contract (PMAT-189) ═══

    #[test]
    fn falsify_session_001_jsonl_roundtrip() {
        let store = create_test_store();
        let original = vec![
            Message::User("question".into()),
            Message::Assistant("answer".into()),
            Message::User("follow-up".into()),
            Message::Assistant("response".into()),
        ];
        store.append_messages(&original).expect("append");
        let loaded = store.load_messages().expect("load");
        assert_eq!(loaded.len(), original.len(), "FALSIFY-SESSION-001: message count preserved");
        for (i, (orig, load)) in original.iter().zip(loaded.iter()).enumerate() {
            assert_eq!(
                format!("{orig:?}"),
                format!("{load:?}"),
                "FALSIFY-SESSION-001: message {i} roundtrip mismatch"
            );
        }
        let _ = fs::remove_dir_all(&store.dir);
    }

    #[test]
    fn falsify_session_002_resume_preserves_messages() {
        let store = create_test_store();
        store.append_message(&Message::User("turn1".into())).expect("append");
        store.append_message(&Message::Assistant("reply1".into())).expect("append");
        store.append_message(&Message::User("turn2".into())).expect("append");

        // Simulate resume: re-read manifest + messages
        let manifest_json = fs::read_to_string(store.dir.join("manifest.json")).expect("read");
        let manifest: SessionManifest = serde_json::from_str(&manifest_json).expect("parse");
        let resumed = SessionStore { dir: store.dir.clone(), manifest };
        let msgs = resumed.load_messages().expect("load");
        assert_eq!(msgs.len(), 3, "FALSIFY-SESSION-002: all messages survive resume");
        assert!(matches!(&msgs[2], Message::User(s) if s == "turn2"));
        let _ = fs::remove_dir_all(&store.dir);
    }

    #[test]
    fn falsify_session_003_manifest_serde_roundtrip() {
        let manifest = SessionManifest {
            id: "test-123".into(),
            agent: "apr-code".into(),
            cwd: "/home/user/project".into(),
            created: "2026-04-04T12:00:00Z".into(),
            turns: 5,
        };
        let json = serde_json::to_string(&manifest).expect("serialize");
        let loaded: SessionManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(loaded.id, manifest.id, "FALSIFY-SESSION-003: id preserved");
        assert_eq!(loaded.turns, manifest.turns, "FALSIFY-SESSION-003: turns preserved");
        assert_eq!(loaded.cwd, manifest.cwd, "FALSIFY-SESSION-003: cwd preserved");
    }

    #[test]
    fn falsify_session_004_age_filter_24h() {
        // FALSIFY-SESSION-004: Sessions older than max_age must NOT be returned.
        // Zero duration = nothing qualifies, so find_recent should return None
        // regardless of how many sessions exist.
        let result = SessionStore::find_recent_for_cwd_within(std::time::Duration::ZERO);
        assert!(
            result.is_none(),
            "FALSIFY-SESSION-004: zero max_age must return None (no session is 0s old)"
        );

        // Very large duration = if any session exists, it qualifies
        // (can't assert Some because the dir may be empty in CI)
        let _ = SessionStore::find_recent_for_cwd_within(std::time::Duration::from_secs(
            365 * 24 * 3600,
        ));
    }

    #[test]
    fn falsify_session_005_unicode_roundtrip() {
        // FALSIFY-SESSION-005: Messages with unicode, newlines, special chars
        // must survive JSONL roundtrip without corruption.
        let store = create_test_store();
        let special_messages = vec![
            Message::User("Hello \u{1F600} emoji".into()),
            Message::Assistant("Line1\nLine2\nLine3".into()),
            Message::User("Tabs\there\tand\tthere".into()),
            Message::Assistant("Quotes: \"double\" and 'single'".into()),
            Message::User("\u{00e9}\u{00e8}\u{00ea} accented".into()),
        ];
        store.append_messages(&special_messages).expect("append unicode");
        let loaded = store.load_messages().expect("load unicode");
        assert_eq!(
            loaded.len(),
            special_messages.len(),
            "FALSIFY-SESSION-005: all unicode messages preserved"
        );
        for (i, (orig, load)) in special_messages.iter().zip(loaded.iter()).enumerate() {
            assert_eq!(
                format!("{orig:?}"),
                format!("{load:?}"),
                "FALSIFY-SESSION-005: unicode message {i} corrupted"
            );
        }
        let _ = fs::remove_dir_all(&store.dir);
    }

    #[test]
    fn falsify_session_006_append_only_monotonic() {
        // FALSIFY-SESSION-006: Multiple appends grow the log monotonically.
        let store = create_test_store();
        store.append_message(&Message::User("first".into())).expect("1");
        let after_one = store.load_messages().expect("load1");
        assert_eq!(after_one.len(), 1);

        store.append_message(&Message::Assistant("second".into())).expect("2");
        let after_two = store.load_messages().expect("load2");
        assert_eq!(after_two.len(), 2);

        store.append_message(&Message::User("third".into())).expect("3");
        let after_three = store.load_messages().expect("load3");
        assert_eq!(after_three.len(), 3);

        // Verify monotonicity: earlier messages unchanged
        assert_eq!(
            format!("{:?}", after_two[0]),
            format!("{:?}", after_three[0]),
            "FALSIFY-SESSION-006: earlier messages must not change"
        );
        assert_eq!(
            format!("{:?}", after_two[1]),
            format!("{:?}", after_three[1]),
            "FALSIFY-SESSION-006: earlier messages must not change"
        );
        let _ = fs::remove_dir_all(&store.dir);
    }
}
