//! Enhanced Deterministic Replay System (Feature 23 - EDD Compliance)
//!
//! Provides advanced deterministic replay capabilities for WASM games.
//! Supports recording, replaying, and verifying game sessions with
//! frame-accurate determinism.
//!
//! ## EXTREME TDD: Tests written FIRST per spec
//!
//! ## Toyota Way Application
//!
//! - **Poka-Yoke**: Type-safe replay files with version checking
//! - **Muda**: Efficient binary/YAML serialization
//! - **Genchi Genbutsu**: Frame-by-frame state verification
//! - **Jidoka**: Fail-fast on determinism violations

use crate::event::InputEvent;
use crate::result::{ProbarError, ProbarResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Version of the replay format
pub const REPLAY_FORMAT_VERSION: u32 = 1;

/// Replay file header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayHeader {
    /// Format version
    pub version: u32,
    /// Name of the game/application
    pub game_name: String,
    /// Game version
    pub game_version: String,
    /// Replay creation timestamp (Unix epoch)
    pub created_at: u64,
    /// Initial random seed
    pub seed: u64,
    /// Total number of frames
    pub total_frames: u64,
    /// Target FPS
    pub fps: u32,
    /// Checksum of replay data
    pub checksum: String,
}

impl ReplayHeader {
    /// Create a new replay header
    #[must_use]
    pub fn new(game_name: &str, game_version: &str, seed: u64) -> Self {
        Self {
            version: REPLAY_FORMAT_VERSION,
            game_name: game_name.to_string(),
            game_version: game_version.to_string(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            seed,
            total_frames: 0,
            fps: 60,
            checksum: String::new(),
        }
    }

    /// Set the FPS
    #[must_use]
    pub const fn with_fps(mut self, fps: u32) -> Self {
        self.fps = fps;
        self
    }
}

/// A single input event with frame timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimedInput {
    /// Frame number when input occurred
    pub frame: u64,
    /// The input event
    pub event: InputEvent,
}

impl TimedInput {
    /// Create a new timed input
    #[must_use]
    pub const fn new(frame: u64, event: InputEvent) -> Self {
        Self { frame, event }
    }
}

/// State checkpoint for verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateCheckpoint {
    /// Frame number
    pub frame: u64,
    /// Hash of the game state at this frame
    pub state_hash: String,
    /// Optional state data (for debugging)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_data: Option<HashMap<String, serde_json::Value>>,
}

impl StateCheckpoint {
    /// Create a new checkpoint with hash
    #[must_use]
    pub fn new(frame: u64, state_hash: &str) -> Self {
        Self {
            frame,
            state_hash: state_hash.to_string(),
            state_data: None,
        }
    }

    /// Create a checkpoint with state data
    #[must_use]
    pub fn with_data(
        frame: u64,
        state_hash: &str,
        data: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            frame,
            state_hash: state_hash.to_string(),
            state_data: Some(data),
        }
    }
}

/// A complete replay recording
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Replay {
    /// Replay header
    pub header: ReplayHeader,
    /// All timed inputs
    pub inputs: Vec<TimedInput>,
    /// State checkpoints (for verification)
    pub checkpoints: Vec<StateCheckpoint>,
    /// Metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl Replay {
    /// Create a new replay
    #[must_use]
    pub fn new(header: ReplayHeader) -> Self {
        Self {
            header,
            inputs: Vec::new(),
            checkpoints: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add an input event
    pub fn add_input(&mut self, frame: u64, event: InputEvent) {
        self.inputs.push(TimedInput::new(frame, event));
        self.header.total_frames = self.header.total_frames.max(frame + 1);
    }

    /// Add a state checkpoint
    pub fn add_checkpoint(&mut self, checkpoint: StateCheckpoint) {
        self.header.total_frames = self.header.total_frames.max(checkpoint.frame + 1);
        self.checkpoints.push(checkpoint);
    }

    /// Add metadata
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Get inputs for a specific frame
    #[must_use]
    pub fn inputs_at_frame(&self, frame: u64) -> Vec<&InputEvent> {
        self.inputs
            .iter()
            .filter(|i| i.frame == frame)
            .map(|i| &i.event)
            .collect()
    }

    /// Get checkpoint at or before a frame
    #[must_use]
    pub fn checkpoint_at_or_before(&self, frame: u64) -> Option<&StateCheckpoint> {
        self.checkpoints
            .iter()
            .filter(|c| c.frame <= frame)
            .max_by_key(|c| c.frame)
    }

    /// Compute checksum of replay data
    #[must_use]
    pub fn compute_checksum(&self) -> String {
        let mut hasher = Sha256::new();

        // Hash header fields
        hasher.update(self.header.seed.to_le_bytes());
        hasher.update(self.header.total_frames.to_le_bytes());
        hasher.update(self.header.fps.to_le_bytes());

        // Hash inputs
        for input in &self.inputs {
            hasher.update(input.frame.to_le_bytes());
            hasher.update(format!("{:?}", input.event).as_bytes());
        }

        // Hash checkpoints
        for checkpoint in &self.checkpoints {
            hasher.update(checkpoint.frame.to_le_bytes());
            hasher.update(checkpoint.state_hash.as_bytes());
        }

        let result = hasher.finalize();
        format!("{result:x}")
    }

    /// Finalize the replay (compute checksum)
    pub fn finalize(&mut self) {
        self.header.checksum = self.compute_checksum();
    }

    /// Verify replay checksum
    #[must_use]
    pub fn verify_checksum(&self) -> bool {
        // Create a copy without checksum to compute
        let computed = self.compute_checksum();
        computed == self.header.checksum
    }

    /// Save replay to YAML file
    pub fn save_yaml(&self, path: &Path) -> ProbarResult<()> {
        let yaml = serde_yaml_ng::to_string(self).map_err(|e| {
            ProbarError::SnapshotSerializationError {
                message: format!("Failed to serialize replay: {e}"),
            }
        })?;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(path, yaml)?;
        Ok(())
    }

    /// Load replay from YAML file
    pub fn load_yaml(path: &Path) -> ProbarResult<Self> {
        let yaml = fs::read_to_string(path)?;
        let replay: Replay = serde_yaml_ng::from_str(&yaml).map_err(|e| {
            ProbarError::SnapshotSerializationError {
                message: format!("Failed to deserialize replay: {e}"),
            }
        })?;
        Ok(replay)
    }

    /// Save replay to JSON file
    pub fn save_json(&self, path: &Path) -> ProbarResult<()> {
        let json = serde_json::to_string_pretty(self)?;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(path, json)?;
        Ok(())
    }

    /// Load replay from JSON file
    pub fn load_json(path: &Path) -> ProbarResult<Self> {
        let json = fs::read_to_string(path)?;
        let replay: Replay = serde_json::from_str(&json)?;
        Ok(replay)
    }
}

/// Replay recorder for capturing gameplay
#[derive(Debug)]
pub struct ReplayRecorder {
    /// The replay being recorded
    replay: Replay,
    /// Current frame
    current_frame: u64,
    /// Checkpoint interval (frames between checkpoints)
    checkpoint_interval: u64,
    /// Whether recording is active
    recording: bool,
}

impl ReplayRecorder {
    /// Create a new replay recorder
    #[must_use]
    pub fn new(game_name: &str, game_version: &str, seed: u64) -> Self {
        let header = ReplayHeader::new(game_name, game_version, seed);
        Self {
            replay: Replay::new(header),
            current_frame: 0,
            checkpoint_interval: 60, // Default: checkpoint every second at 60fps
            recording: true,
        }
    }

    /// Set the checkpoint interval
    #[must_use]
    pub const fn with_checkpoint_interval(mut self, interval: u64) -> Self {
        self.checkpoint_interval = interval;
        self
    }

    /// Set FPS
    #[must_use]
    pub fn with_fps(mut self, fps: u32) -> Self {
        self.replay.header = self.replay.header.with_fps(fps);
        self
    }

    /// Record an input event
    pub fn record_input(&mut self, event: InputEvent) {
        if self.recording {
            self.replay.add_input(self.current_frame, event);
        }
    }

    /// Record multiple input events
    pub fn record_inputs(&mut self, events: &[InputEvent]) {
        for event in events {
            self.record_input(event.clone());
        }
    }

    /// Advance to next frame and optionally record checkpoint
    pub fn next_frame(&mut self, state_hash: Option<&str>) {
        self.current_frame += 1;

        // Record checkpoint if at interval
        if let Some(hash) = state_hash {
            if self.current_frame % self.checkpoint_interval == 0 {
                self.replay
                    .add_checkpoint(StateCheckpoint::new(self.current_frame, hash));
            }
        }
    }

    /// Record a checkpoint at current frame
    pub fn checkpoint(&mut self, state_hash: &str) {
        self.replay
            .add_checkpoint(StateCheckpoint::new(self.current_frame, state_hash));
    }

    /// Record a checkpoint with state data
    pub fn checkpoint_with_data(
        &mut self,
        state_hash: &str,
        data: HashMap<String, serde_json::Value>,
    ) {
        self.replay.add_checkpoint(StateCheckpoint::with_data(
            self.current_frame,
            state_hash,
            data,
        ));
    }

    /// Stop recording
    pub fn stop(&mut self) {
        self.recording = false;
    }

    /// Finalize and get the replay
    #[must_use]
    pub fn finalize(mut self) -> Replay {
        self.replay.finalize();
        self.replay
    }

    /// Get current frame
    #[must_use]
    pub const fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Check if recording
    #[must_use]
    pub const fn is_recording(&self) -> bool {
        self.recording
    }
}

/// Replay playback controller
#[derive(Debug)]
pub struct ReplayPlayer {
    /// The replay being played
    replay: Replay,
    /// Current frame
    current_frame: u64,
    /// Playback speed (1.0 = normal)
    speed: f64,
    /// Whether playback is active
    playing: bool,
    /// Index into inputs array for efficiency
    input_index: usize,
}

impl ReplayPlayer {
    /// Create a new replay player
    #[must_use]
    pub fn new(replay: Replay) -> Self {
        Self {
            replay,
            current_frame: 0,
            speed: 1.0,
            playing: true,
            input_index: 0,
        }
    }

    /// Set playback speed
    #[must_use]
    pub fn with_speed(mut self, speed: f64) -> Self {
        self.speed = speed;
        self
    }

    /// Get inputs for current frame and advance
    #[must_use]
    pub fn get_frame_inputs(&mut self) -> Vec<InputEvent> {
        if !self.playing {
            return Vec::new();
        }

        let mut inputs = Vec::new();

        // Collect all inputs for current frame
        while self.input_index < self.replay.inputs.len() {
            let timed = &self.replay.inputs[self.input_index];
            if timed.frame == self.current_frame {
                inputs.push(timed.event.clone());
                self.input_index += 1;
            } else if timed.frame > self.current_frame {
                break;
            } else {
                self.input_index += 1;
            }
        }

        self.current_frame += 1;

        // Check if replay is done
        if self.current_frame >= self.replay.header.total_frames {
            self.playing = false;
        }

        inputs
    }

    /// Get expected state hash for current frame (if checkpoint exists)
    #[must_use]
    pub fn expected_checkpoint(&self) -> Option<&StateCheckpoint> {
        self.replay
            .checkpoints
            .iter()
            .find(|c| c.frame == self.current_frame - 1)
    }

    /// Verify state against checkpoint
    pub fn verify_state(&self, state_hash: &str) -> ProbarResult<()> {
        if let Some(checkpoint) = self.expected_checkpoint() {
            if checkpoint.state_hash != state_hash {
                return Err(ProbarError::AssertionFailed {
                    message: format!(
                        "State divergence at frame {}: expected hash '{}', got '{}'",
                        checkpoint.frame, checkpoint.state_hash, state_hash
                    ),
                });
            }
        }
        Ok(())
    }

    /// Get current frame
    #[must_use]
    pub const fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Check if playback is active
    #[must_use]
    pub const fn is_playing(&self) -> bool {
        self.playing
    }

    /// Get total frames in replay
    #[must_use]
    pub const fn total_frames(&self) -> u64 {
        self.replay.header.total_frames
    }

    /// Get progress (0.0 to 1.0)
    #[must_use]
    pub fn progress(&self) -> f64 {
        if self.replay.header.total_frames == 0 {
            return 1.0;
        }
        self.current_frame as f64 / self.replay.header.total_frames as f64
    }

    /// Seek to a specific frame
    pub fn seek(&mut self, frame: u64) {
        self.current_frame = frame.min(self.replay.header.total_frames);
        self.playing = self.current_frame < self.replay.header.total_frames;

        // Reset input index and search for correct position
        self.input_index = 0;
        while self.input_index < self.replay.inputs.len()
            && self.replay.inputs[self.input_index].frame < self.current_frame
        {
            self.input_index += 1;
        }
    }

    /// Pause playback
    pub fn pause(&mut self) {
        self.playing = false;
    }

    /// Resume playback
    pub fn resume(&mut self) {
        if self.current_frame < self.replay.header.total_frames {
            self.playing = true;
        }
    }

    /// Get the underlying replay
    #[must_use]
    pub fn replay(&self) -> &Replay {
        &self.replay
    }
}

/// Result of replay verification
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether verification passed
    pub passed: bool,
    /// Number of frames verified
    pub frames_verified: u64,
    /// Number of checkpoints verified
    pub checkpoints_verified: usize,
    /// Frame where divergence occurred (if any)
    pub divergence_frame: Option<u64>,
    /// Divergence details
    pub divergence_details: Option<String>,
}

impl VerificationResult {
    /// Create a successful verification result
    #[must_use]
    pub const fn success(frames_verified: u64, checkpoints_verified: usize) -> Self {
        Self {
            passed: true,
            frames_verified,
            checkpoints_verified,
            divergence_frame: None,
            divergence_details: None,
        }
    }

    /// Create a failed verification result
    #[must_use]
    pub fn failure(frame: u64, details: &str) -> Self {
        Self {
            passed: false,
            frames_verified: frame,
            checkpoints_verified: 0,
            divergence_frame: Some(frame),
            divergence_details: Some(details.to_string()),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    mod replay_header_tests {
        use super::*;

        #[test]
        fn test_new() {
            let header = ReplayHeader::new("test_game", "1.0.0", 42);
            assert_eq!(header.game_name, "test_game");
            assert_eq!(header.game_version, "1.0.0");
            assert_eq!(header.seed, 42);
            assert_eq!(header.version, REPLAY_FORMAT_VERSION);
        }

        #[test]
        fn test_with_fps() {
            let header = ReplayHeader::new("game", "1.0", 0).with_fps(30);
            assert_eq!(header.fps, 30);
        }
    }

    mod timed_input_tests {
        use super::*;

        #[test]
        fn test_new() {
            let event = InputEvent::key_press("Space");
            let timed = TimedInput::new(100, event);
            assert_eq!(timed.frame, 100);
        }
    }

    mod state_checkpoint_tests {
        use super::*;

        #[test]
        fn test_new() {
            let cp = StateCheckpoint::new(50, "abc123");
            assert_eq!(cp.frame, 50);
            assert_eq!(cp.state_hash, "abc123");
            assert!(cp.state_data.is_none());
        }

        #[test]
        fn test_with_data() {
            let mut data = HashMap::new();
            data.insert("score".to_string(), serde_json::json!(100));
            let cp = StateCheckpoint::with_data(50, "abc123", data);
            assert!(cp.state_data.is_some());
        }
    }

    mod replay_tests {
        use super::*;

        #[test]
        fn test_new() {
            let header = ReplayHeader::new("game", "1.0", 42);
            let replay = Replay::new(header);
            assert!(replay.inputs.is_empty());
            assert!(replay.checkpoints.is_empty());
        }

        #[test]
        fn test_add_input() {
            let header = ReplayHeader::new("game", "1.0", 42);
            let mut replay = Replay::new(header);

            replay.add_input(0, InputEvent::key_press("A"));
            replay.add_input(10, InputEvent::key_press("B"));

            assert_eq!(replay.inputs.len(), 2);
            assert_eq!(replay.header.total_frames, 11);
        }

        #[test]
        fn test_inputs_at_frame() {
            let header = ReplayHeader::new("game", "1.0", 42);
            let mut replay = Replay::new(header);

            replay.add_input(5, InputEvent::key_press("A"));
            replay.add_input(5, InputEvent::key_press("B"));
            replay.add_input(10, InputEvent::key_press("C"));

            let inputs = replay.inputs_at_frame(5);
            assert_eq!(inputs.len(), 2);
        }

        #[test]
        fn test_add_checkpoint() {
            let header = ReplayHeader::new("game", "1.0", 42);
            let mut replay = Replay::new(header);

            replay.add_checkpoint(StateCheckpoint::new(60, "hash1"));
            replay.add_checkpoint(StateCheckpoint::new(120, "hash2"));

            assert_eq!(replay.checkpoints.len(), 2);
        }

        #[test]
        fn test_checkpoint_at_or_before() {
            let header = ReplayHeader::new("game", "1.0", 42);
            let mut replay = Replay::new(header);

            replay.add_checkpoint(StateCheckpoint::new(60, "hash1"));
            replay.add_checkpoint(StateCheckpoint::new(120, "hash2"));

            let cp = replay.checkpoint_at_or_before(100);
            assert!(cp.is_some());
            assert_eq!(cp.unwrap().frame, 60);

            let cp = replay.checkpoint_at_or_before(120);
            assert!(cp.is_some());
            assert_eq!(cp.unwrap().frame, 120);

            let cp = replay.checkpoint_at_or_before(50);
            assert!(cp.is_none());
        }

        #[test]
        fn test_metadata() {
            let header = ReplayHeader::new("game", "1.0", 42);
            let mut replay = Replay::new(header);

            replay.set_metadata("player", "Alice");
            replay.set_metadata("difficulty", "hard");

            assert_eq!(replay.metadata.get("player"), Some(&"Alice".to_string()));
        }

        #[test]
        fn test_compute_checksum() {
            let header = ReplayHeader::new("game", "1.0", 42);
            let mut replay1 = Replay::new(header.clone());
            let mut replay2 = Replay::new(header);

            replay1.add_input(0, InputEvent::key_press("A"));
            replay2.add_input(0, InputEvent::key_press("A"));

            assert_eq!(replay1.compute_checksum(), replay2.compute_checksum());

            replay2.add_input(1, InputEvent::key_press("B"));
            assert_ne!(replay1.compute_checksum(), replay2.compute_checksum());
        }

        #[test]
        fn test_finalize_and_verify() {
            let header = ReplayHeader::new("game", "1.0", 42);
            let mut replay = Replay::new(header);
            replay.add_input(0, InputEvent::key_press("A"));
            replay.finalize();

            assert!(!replay.header.checksum.is_empty());
            assert!(replay.verify_checksum());
        }
    }

    mod replay_recorder_tests {
        use super::*;

        #[test]
        fn test_new() {
            let recorder = ReplayRecorder::new("game", "1.0", 42);
            assert_eq!(recorder.current_frame(), 0);
            assert!(recorder.is_recording());
        }

        #[test]
        fn test_record_input() {
            let mut recorder = ReplayRecorder::new("game", "1.0", 42);
            recorder.record_input(InputEvent::key_press("A"));
            recorder.next_frame(None);
            recorder.record_input(InputEvent::key_press("B"));

            let replay = recorder.finalize();
            assert_eq!(replay.inputs.len(), 2);
        }

        #[test]
        fn test_checkpoint() {
            let mut recorder = ReplayRecorder::new("game", "1.0", 42).with_checkpoint_interval(10);

            for i in 0..25 {
                recorder.next_frame(Some(&format!("hash{}", i)));
            }

            let replay = recorder.finalize();
            // Checkpoints at frames 10, 20
            assert_eq!(replay.checkpoints.len(), 2);
        }

        #[test]
        fn test_stop_recording() {
            let mut recorder = ReplayRecorder::new("game", "1.0", 42);
            recorder.record_input(InputEvent::key_press("A"));
            recorder.stop();
            recorder.record_input(InputEvent::key_press("B"));

            let replay = recorder.finalize();
            assert_eq!(replay.inputs.len(), 1); // Only A recorded before stop
        }
    }

    mod replay_player_tests {
        use super::*;

        fn create_test_replay() -> Replay {
            let header = ReplayHeader::new("game", "1.0", 42);
            let mut replay = Replay::new(header);

            replay.add_input(0, InputEvent::key_press("A"));
            replay.add_input(5, InputEvent::key_press("B"));
            replay.add_input(5, InputEvent::key_press("C"));
            replay.add_input(10, InputEvent::key_press("D"));
            replay.add_checkpoint(StateCheckpoint::new(5, "hash5"));
            replay.header.total_frames = 15;
            replay
        }

        #[test]
        fn test_new() {
            let replay = create_test_replay();
            let player = ReplayPlayer::new(replay);

            assert_eq!(player.current_frame(), 0);
            assert!(player.is_playing());
        }

        #[test]
        fn test_get_frame_inputs() {
            let replay = create_test_replay();
            let mut player = ReplayPlayer::new(replay);

            // Frame 0: has input A
            let inputs = player.get_frame_inputs();
            assert_eq!(inputs.len(), 1);
            assert_eq!(player.current_frame(), 1);

            // Frames 1-4: no inputs
            for _ in 1..5 {
                let inputs = player.get_frame_inputs();
                assert!(inputs.is_empty());
            }

            // Frame 5: has inputs B and C
            let inputs = player.get_frame_inputs();
            assert_eq!(inputs.len(), 2);
        }

        #[test]
        fn test_progress() {
            let replay = create_test_replay();
            let mut player = ReplayPlayer::new(replay);

            assert!((player.progress() - 0.0).abs() < f64::EPSILON);

            for _ in 0..7 {
                let _ = player.get_frame_inputs();
            }

            // 7/15 ≈ 0.467
            assert!((player.progress() - 7.0 / 15.0).abs() < 0.01);
        }

        #[test]
        fn test_seek() {
            let replay = create_test_replay();
            let mut player = ReplayPlayer::new(replay);

            player.seek(10);
            assert_eq!(player.current_frame(), 10);

            // Should still get input at frame 10
            let inputs = player.get_frame_inputs();
            assert_eq!(inputs.len(), 1);
        }

        #[test]
        fn test_pause_resume() {
            let replay = create_test_replay();
            let mut player = ReplayPlayer::new(replay);

            player.pause();
            assert!(!player.is_playing());

            let inputs = player.get_frame_inputs();
            assert!(inputs.is_empty());
            assert_eq!(player.current_frame(), 0); // Didn't advance

            player.resume();
            assert!(player.is_playing());
        }

        #[test]
        fn test_playback_completion() {
            let replay = create_test_replay();
            let mut player = ReplayPlayer::new(replay);

            // Play through entire replay
            while player.is_playing() {
                let _ = player.get_frame_inputs();
            }

            assert_eq!(player.current_frame(), 15);
            assert!(!player.is_playing());
        }

        #[test]
        fn test_verify_state_pass() {
            let replay = create_test_replay();
            let mut player = ReplayPlayer::new(replay);

            // Advance to frame 5 (checkpoint is at frame 5)
            for _ in 0..6 {
                let _ = player.get_frame_inputs();
            }

            // Should verify successfully
            assert!(player.verify_state("hash5").is_ok());
        }

        #[test]
        fn test_verify_state_fail() {
            let replay = create_test_replay();
            let mut player = ReplayPlayer::new(replay);

            // Advance to frame 5
            for _ in 0..6 {
                let _ = player.get_frame_inputs();
            }

            // Should fail with wrong hash
            assert!(player.verify_state("wrong_hash").is_err());
        }
    }

    mod file_io_tests {
        use super::*;
        use tempfile::TempDir;

        #[test]
        fn test_save_and_load_yaml() {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("replay.yaml");

            let header = ReplayHeader::new("game", "1.0", 42);
            let mut replay = Replay::new(header);
            replay.add_input(0, InputEvent::key_press("A"));
            replay.finalize();

            replay.save_yaml(&path).unwrap();
            assert!(path.exists());

            let loaded = Replay::load_yaml(&path).unwrap();
            assert_eq!(loaded.header.seed, 42);
            assert_eq!(loaded.inputs.len(), 1);
            assert!(loaded.verify_checksum());
        }

        #[test]
        fn test_save_and_load_json() {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("replay.json");

            let header = ReplayHeader::new("game", "1.0", 42);
            let mut replay = Replay::new(header);
            replay.add_input(0, InputEvent::key_press("A"));
            replay.finalize();

            replay.save_json(&path).unwrap();
            assert!(path.exists());

            let loaded = Replay::load_json(&path).unwrap();
            assert_eq!(loaded.header.seed, 42);
            assert!(loaded.verify_checksum());
        }
    }

    mod verification_result_tests {
        use super::*;

        #[test]
        fn test_success() {
            let result = VerificationResult::success(100, 5);
            assert!(result.passed);
            assert_eq!(result.frames_verified, 100);
            assert_eq!(result.checkpoints_verified, 5);
            assert!(result.divergence_frame.is_none());
            assert!(result.divergence_details.is_none());
        }

        #[test]
        fn test_failure() {
            let result = VerificationResult::failure(50, "State mismatch");
            assert!(!result.passed);
            assert_eq!(result.frames_verified, 50);
            assert_eq!(result.divergence_frame, Some(50));
            assert!(result
                .divergence_details
                .as_ref()
                .unwrap()
                .contains("State mismatch"));
        }
    }

    mod additional_replay_tests {
        use super::*;

        #[test]
        fn test_replay_header_default_fps() {
            let header = ReplayHeader::new("game", "1.0", 0);
            assert_eq!(header.fps, 60);
        }

        #[test]
        fn test_replay_header_created_at() {
            let header = ReplayHeader::new("game", "1.0", 0);
            assert!(header.created_at > 0);
        }

        #[test]
        fn test_replay_header_checksum_empty() {
            let header = ReplayHeader::new("game", "1.0", 0);
            assert!(header.checksum.is_empty());
        }

        #[test]
        fn test_timed_input_event_types() {
            let events = vec![
                InputEvent::key_press("A"),
                InputEvent::key_release("B"),
                InputEvent::mouse_click(100.0, 200.0),
            ];

            for (i, event) in events.into_iter().enumerate() {
                let timed = TimedInput::new(i as u64, event);
                assert_eq!(timed.frame, i as u64);
            }
        }

        #[test]
        fn test_replay_inputs_at_frame_none() {
            let header = ReplayHeader::new("game", "1.0", 0);
            let replay = Replay::new(header);
            let inputs = replay.inputs_at_frame(0);
            assert!(inputs.is_empty());
        }

        #[test]
        fn test_replay_checkpoint_none() {
            let header = ReplayHeader::new("game", "1.0", 0);
            let replay = Replay::new(header);
            assert!(replay.checkpoint_at_or_before(0).is_none());
        }

        #[test]
        fn test_replay_verify_checksum_mismatch() {
            let header = ReplayHeader::new("game", "1.0", 42);
            let mut replay = Replay::new(header);
            replay.add_input(0, InputEvent::key_press("A"));
            replay.finalize();

            // Tamper with checksum
            replay.header.checksum = "invalid".to_string();
            assert!(!replay.verify_checksum());
        }
    }

    mod additional_recorder_tests {
        use super::*;

        #[test]
        fn test_recorder_record_inputs() {
            let mut recorder = ReplayRecorder::new("game", "1.0", 42);
            let events = vec![
                InputEvent::key_press("A"),
                InputEvent::key_press("B"),
                InputEvent::key_press("C"),
            ];
            recorder.record_inputs(&events);

            let replay = recorder.finalize();
            assert_eq!(replay.inputs.len(), 3);
        }

        #[test]
        fn test_recorder_checkpoint() {
            let mut recorder = ReplayRecorder::new("game", "1.0", 42);
            recorder.checkpoint("hash_at_0");
            recorder.next_frame(None);
            recorder.checkpoint("hash_at_1");

            let replay = recorder.finalize();
            assert_eq!(replay.checkpoints.len(), 2);
        }

        #[test]
        fn test_recorder_checkpoint_with_data() {
            let mut recorder = ReplayRecorder::new("game", "1.0", 42);
            let mut data = HashMap::new();
            data.insert("score".to_string(), serde_json::json!(100));
            data.insert("level".to_string(), serde_json::json!(5));
            recorder.checkpoint_with_data("hash", data);

            let replay = recorder.finalize();
            assert_eq!(replay.checkpoints.len(), 1);
            assert!(replay.checkpoints[0].state_data.is_some());
        }

        #[test]
        fn test_recorder_with_fps() {
            let recorder = ReplayRecorder::new("game", "1.0", 42).with_fps(30);
            let replay = recorder.finalize();
            assert_eq!(replay.header.fps, 30);
        }

        #[test]
        fn test_recorder_not_recording() {
            let mut recorder = ReplayRecorder::new("game", "1.0", 42);
            recorder.stop();
            assert!(!recorder.is_recording());

            // Should not record after stop
            recorder.record_input(InputEvent::key_press("A"));
            let replay = recorder.finalize();
            assert!(replay.inputs.is_empty());
        }
    }

    mod additional_player_tests {
        use super::*;

        fn create_simple_replay(total_frames: u64) -> Replay {
            let header = ReplayHeader::new("game", "1.0", 42);
            let mut replay = Replay::new(header);
            replay.header.total_frames = total_frames;
            replay
        }

        #[test]
        fn test_player_with_speed() {
            let replay = create_simple_replay(100);
            let player = ReplayPlayer::new(replay).with_speed(2.0);
            assert!((player.speed - 2.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_player_total_frames() {
            let replay = create_simple_replay(100);
            let player = ReplayPlayer::new(replay);
            assert_eq!(player.total_frames(), 100);
        }

        #[test]
        fn test_player_progress_zero_frames() {
            let replay = create_simple_replay(0);
            let player = ReplayPlayer::new(replay);
            assert!((player.progress() - 1.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_player_seek_beyond_total() {
            let replay = create_simple_replay(50);
            let mut player = ReplayPlayer::new(replay);
            player.seek(100);
            assert_eq!(player.current_frame(), 50);
            assert!(!player.is_playing());
        }

        #[test]
        fn test_player_resume_at_end() {
            let replay = create_simple_replay(10);
            let mut player = ReplayPlayer::new(replay);

            // Play to end
            while player.is_playing() {
                let _ = player.get_frame_inputs();
            }

            player.resume();
            // Should still not be playing since we're at the end
            assert!(!player.is_playing());
        }

        #[test]
        fn test_player_replay_accessor() {
            let replay = create_simple_replay(50);
            let player = ReplayPlayer::new(replay);
            let accessed_replay = player.replay();
            assert_eq!(accessed_replay.header.total_frames, 50);
        }

        #[test]
        fn test_player_expected_checkpoint_none() {
            let replay = create_simple_replay(50);
            let player = ReplayPlayer::new(replay);
            // At frame 0, previous frame would be -1, no checkpoint
            assert!(player.expected_checkpoint().is_none());
        }

        #[test]
        fn test_player_verify_state_no_checkpoint() {
            let replay = create_simple_replay(50);
            let player = ReplayPlayer::new(replay);
            // Should be Ok since there's no checkpoint to verify against
            assert!(player.verify_state("any_hash").is_ok());
        }

        #[test]
        fn test_player_inputs_with_gaps() {
            let header = ReplayHeader::new("game", "1.0", 42);
            let mut replay = Replay::new(header);

            // Inputs at frames 0, 10, 20
            replay.add_input(0, InputEvent::key_press("A"));
            replay.add_input(10, InputEvent::key_press("B"));
            replay.add_input(20, InputEvent::key_press("C"));
            replay.header.total_frames = 25;

            let mut player = ReplayPlayer::new(replay);

            // Frame 0: has A
            let inputs = player.get_frame_inputs();
            assert_eq!(inputs.len(), 1);

            // Frames 1-9: no inputs
            for _ in 1..10 {
                let inputs = player.get_frame_inputs();
                assert!(inputs.is_empty());
            }

            // Frame 10: has B
            let inputs = player.get_frame_inputs();
            assert_eq!(inputs.len(), 1);
        }

        #[test]
        fn test_player_seek_updates_input_index() {
            let header = ReplayHeader::new("game", "1.0", 42);
            let mut replay = Replay::new(header);

            replay.add_input(0, InputEvent::key_press("A"));
            replay.add_input(5, InputEvent::key_press("B"));
            replay.add_input(10, InputEvent::key_press("C"));
            replay.header.total_frames = 15;

            let mut player = ReplayPlayer::new(replay);

            // Seek to frame 7 (between B and C)
            player.seek(7);

            // Advance, should get no input
            let inputs = player.get_frame_inputs();
            assert!(inputs.is_empty());

            // Continue to frame 10, should get C
            let _ = player.get_frame_inputs(); // frame 8
            let _ = player.get_frame_inputs(); // frame 9
            let inputs = player.get_frame_inputs(); // frame 10
            assert_eq!(inputs.len(), 1);
        }
    }

    mod additional_file_io_tests {
        use super::*;
        use tempfile::TempDir;

        #[test]
        fn test_save_yaml_creates_parent_dirs() {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir
                .path()
                .join("subdir")
                .join("deep")
                .join("replay.yaml");

            let header = ReplayHeader::new("game", "1.0", 42);
            let replay = Replay::new(header);

            replay.save_yaml(&path).unwrap();
            assert!(path.exists());
        }

        #[test]
        fn test_save_json_creates_parent_dirs() {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir
                .path()
                .join("subdir")
                .join("deep")
                .join("replay.json");

            let header = ReplayHeader::new("game", "1.0", 42);
            let replay = Replay::new(header);

            replay.save_json(&path).unwrap();
            assert!(path.exists());
        }

        #[test]
        fn test_load_yaml_not_found() {
            let result = Replay::load_yaml(Path::new("/nonexistent/replay.yaml"));
            assert!(result.is_err());
        }

        #[test]
        fn test_load_json_not_found() {
            let result = Replay::load_json(Path::new("/nonexistent/replay.json"));
            assert!(result.is_err());
        }

        #[test]
        fn test_load_yaml_invalid() {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("invalid.yaml");
            std::fs::write(&path, "not: valid: yaml: {{{").unwrap();

            let result = Replay::load_yaml(&path);
            assert!(result.is_err());
        }

        #[test]
        fn test_load_json_invalid() {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("invalid.json");
            std::fs::write(&path, "not valid json {{{").unwrap();

            let result = Replay::load_json(&path);
            assert!(result.is_err());
        }

        #[test]
        fn test_replay_with_metadata_yaml() {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("replay.yaml");

            let header = ReplayHeader::new("game", "1.0", 42);
            let mut replay = Replay::new(header);
            replay.set_metadata("player", "Alice");
            replay.set_metadata("score", "1000");
            replay.finalize();

            replay.save_yaml(&path).unwrap();
            let loaded = Replay::load_yaml(&path).unwrap();

            assert_eq!(loaded.metadata.get("player"), Some(&"Alice".to_string()));
            assert_eq!(loaded.metadata.get("score"), Some(&"1000".to_string()));
        }

        #[test]
        fn test_replay_with_checkpoints_json() {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("replay.json");

            let header = ReplayHeader::new("game", "1.0", 42);
            let mut replay = Replay::new(header);

            let mut data = HashMap::new();
            data.insert("level".to_string(), serde_json::json!(3));
            replay.add_checkpoint(StateCheckpoint::with_data(60, "hash1", data));

            replay.finalize();
            replay.save_json(&path).unwrap();

            let loaded = Replay::load_json(&path).unwrap();
            assert_eq!(loaded.checkpoints.len(), 1);
            assert!(loaded.checkpoints[0].state_data.is_some());
        }
    }

    mod additional_edge_case_tests {
        use super::*;

        #[test]
        fn test_replay_format_version() {
            assert_eq!(REPLAY_FORMAT_VERSION, 1);
        }

        #[test]
        fn test_state_checkpoint_without_data() {
            let cp = StateCheckpoint::new(100, "hash");
            assert!(cp.state_data.is_none());
            assert_eq!(cp.frame, 100);
            assert_eq!(cp.state_hash, "hash");
        }

        #[test]
        fn test_replay_checksum_with_checkpoints() {
            let header = ReplayHeader::new("game", "1.0", 42);
            let mut replay = Replay::new(header);

            replay.add_checkpoint(StateCheckpoint::new(10, "hash1"));
            replay.add_checkpoint(StateCheckpoint::new(20, "hash2"));

            let checksum1 = replay.compute_checksum();

            replay.add_checkpoint(StateCheckpoint::new(30, "hash3"));
            let checksum2 = replay.compute_checksum();

            assert_ne!(checksum1, checksum2);
        }

        #[test]
        fn test_replay_total_frames_from_checkpoint() {
            let header = ReplayHeader::new("game", "1.0", 42);
            let mut replay = Replay::new(header);

            assert_eq!(replay.header.total_frames, 0);

            replay.add_checkpoint(StateCheckpoint::new(100, "hash"));
            assert_eq!(replay.header.total_frames, 101);
        }

        #[test]
        fn test_replay_total_frames_from_input() {
            let header = ReplayHeader::new("game", "1.0", 42);
            let mut replay = Replay::new(header);

            replay.add_input(50, InputEvent::key_press("A"));
            assert_eq!(replay.header.total_frames, 51);
        }

        #[test]
        fn test_player_skipped_inputs() {
            let header = ReplayHeader::new("game", "1.0", 42);
            let mut replay = Replay::new(header);

            // Add input at frame 0
            replay.add_input(0, InputEvent::key_press("A"));
            replay.header.total_frames = 5;

            let mut player = ReplayPlayer::new(replay);

            // Seek past frame 0
            player.seek(3);

            // Input at frame 0 should be skipped
            let inputs = player.get_frame_inputs();
            assert!(inputs.is_empty());
        }
    }
}
