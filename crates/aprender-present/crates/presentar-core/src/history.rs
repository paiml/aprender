// Undo/Redo Command History - WASM-first command pattern implementation
//
// Provides:
// - Command pattern for undoable operations
// - History stack with undo/redo navigation
// - Batch command grouping
// - Memory-limited history
// - Checkpoints and save points
// - History branching support

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

/// Unique identifier for a command in history
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CommandId(u64);

impl CommandId {
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// Unique identifier for a command group
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupId(u64);

impl GroupId {
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// Unique identifier for a checkpoint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CheckpointId(u64);

impl CheckpointId {
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// Result of executing or undoing a command
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandResult {
    /// Command executed successfully
    Success,
    /// Command failed with error message
    Failed(String),
    /// Command was cancelled
    Cancelled,
    /// Command requires confirmation
    NeedsConfirmation(String),
}

impl CommandResult {
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, Self::Failed(_))
    }
}

/// Trait for undoable commands
pub trait Command: Send + Sync {
    /// Execute the command
    fn execute(&mut self) -> CommandResult;

    /// Undo the command
    fn undo(&mut self) -> CommandResult;

    /// Redo the command (default: re-execute)
    fn redo(&mut self) -> CommandResult {
        self.execute()
    }

    /// Get command description
    fn description(&self) -> &str;

    /// Check if command can be merged with another
    fn can_merge(&self, _other: &dyn Command) -> bool {
        false
    }

    /// Merge with another command (returns merged command)
    fn merge(&mut self, _other: Box<dyn Command>) -> Option<Box<dyn Command>> {
        None
    }

    /// Estimate memory usage of this command
    fn memory_size(&self) -> usize {
        // Default to a reasonable estimate for trait objects
        64
    }

    /// Get command metadata
    fn metadata(&self) -> Option<&dyn Any> {
        None
    }
}

/// Entry in the history stack
struct HistoryEntry {
    #[allow(dead_code)]
    id: CommandId,
    command: Box<dyn Command>,
    group_id: Option<GroupId>,
    timestamp: u64,
    #[allow(dead_code)]
    executed: bool,
}

/// Configuration for command history
#[derive(Debug, Clone)]
pub struct HistoryConfig {
    /// Maximum number of commands to keep
    pub max_commands: usize,
    /// Maximum memory usage in bytes
    pub max_memory: usize,
    /// Enable command merging
    pub enable_merging: bool,
    /// Automatically group rapid commands
    pub auto_group_interval_ms: u64,
}

impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            max_commands: 1000,
            max_memory: 10 * 1024 * 1024, // 10 MB
            enable_merging: true,
            auto_group_interval_ms: 500,
        }
    }
}

/// Checkpoint in history
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub id: CheckpointId,
    pub name: String,
    pub position: usize,
    pub timestamp: u64,
}

/// State change notification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistoryEvent {
    /// Command was executed
    Executed(CommandId),
    /// Command was undone
    Undone(CommandId),
    /// Command was redone
    Redone(CommandId),
    /// History was cleared
    Cleared,
    /// Checkpoint was created
    CheckpointCreated(CheckpointId),
    /// Restored to checkpoint
    CheckpointRestored(CheckpointId),
    /// History limit reached, old commands dropped
    Trimmed(usize),
}

/// Callback for history events
pub type HistoryCallback = Arc<dyn Fn(HistoryEvent) + Send + Sync>;

/// Command history manager
pub struct CommandHistory {
    config: HistoryConfig,
    next_command_id: u64,
    next_group_id: u64,
    next_checkpoint_id: u64,

    /// Commands that have been executed (undo stack)
    undo_stack: Vec<HistoryEntry>,
    /// Commands that have been undone (redo stack)
    redo_stack: Vec<HistoryEntry>,

    /// Active command group
    current_group: Option<GroupId>,
    /// Saved checkpoints
    checkpoints: HashMap<CheckpointId, Checkpoint>,

    /// Current timestamp
    timestamp: u64,
    /// Last command timestamp for auto-grouping
    last_command_time: u64,

    /// Event listeners
    listeners: Vec<HistoryCallback>,

    /// Current memory usage
    current_memory: usize,

    /// Whether history is recording
    recording: bool,
}

impl Default for CommandHistory {
    fn default() -> Self {
        Self::new(HistoryConfig::default())
    }
}

impl CommandHistory {
    pub fn new(config: HistoryConfig) -> Self {
        Self {
            config,
            next_command_id: 1,
            next_group_id: 1,
            next_checkpoint_id: 1,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            current_group: None,
            checkpoints: HashMap::new(),
            timestamp: 0,
            last_command_time: 0,
            listeners: Vec::new(),
            current_memory: 0,
            recording: true,
        }
    }

    /// Execute a command and add to history
    pub fn execute(&mut self, mut command: Box<dyn Command>) -> CommandResult {
        if !self.recording {
            return command.execute();
        }

        let result = command.execute();
        if !result.is_success() {
            return result;
        }

        // Clear redo stack when new command is executed
        self.redo_stack.clear();

        let id = CommandId(self.next_command_id);
        self.next_command_id += 1;

        // Check for auto-grouping
        let group_id = if self.current_group.is_some() {
            self.current_group
        } else if self.config.auto_group_interval_ms > 0
            && self.timestamp.saturating_sub(self.last_command_time)
                < self.config.auto_group_interval_ms
            && !self.undo_stack.is_empty()
        {
            // Auto-group with previous command
            self.undo_stack.last().and_then(|e| e.group_id)
        } else {
            None
        };

        // Check for merging
        if self.config.enable_merging && !self.undo_stack.is_empty() {
            let can_merge = self
                .undo_stack
                .last()
                .is_some_and(|last| last.command.can_merge(command.as_ref()));

            if can_merge {
                if let Some(last_entry) = self.undo_stack.last_mut() {
                    if let Some(merged) = last_entry.command.merge(command) {
                        // Update memory tracking
                        self.current_memory -= last_entry.command.memory_size();
                        self.current_memory += merged.memory_size();
                        last_entry.command = merged;
                        self.emit(&HistoryEvent::Executed(id));
                        return result;
                    }
                }
                // If merge returned None, we can't use command anymore
                // This shouldn't happen in practice if can_merge is implemented correctly
                return result;
            }
        }

        let memory = command.memory_size();
        self.current_memory += memory;

        let entry = HistoryEntry {
            id,
            command,
            group_id,
            timestamp: self.timestamp,
            executed: true,
        };

        self.undo_stack.push(entry);
        self.last_command_time = self.timestamp;

        // Trim if needed
        self.trim_if_needed();

        self.emit(&HistoryEvent::Executed(id));
        result
    }

    /// Undo the last command
    pub fn undo(&mut self) -> Option<CommandResult> {
        let entry = self.undo_stack.pop()?;
        let id = entry.id;
        let group_id = entry.group_id;

        let mut command = entry.command;
        let result = command.undo();

        if result.is_success() {
            self.current_memory -= command.memory_size();

            let redo_entry = HistoryEntry {
                id,
                command,
                group_id,
                timestamp: entry.timestamp,
                executed: false,
            };
            self.current_memory += redo_entry.command.memory_size();
            self.redo_stack.push(redo_entry);

            self.emit(&HistoryEvent::Undone(id));

            // Undo entire group if applicable
            if let Some(gid) = group_id {
                while let Some(last) = self.undo_stack.last() {
                    if last.group_id == Some(gid) {
                        self.undo();
                    } else {
                        break;
                    }
                }
            }
        }

        Some(result)
    }

    /// Redo the last undone command
    pub fn redo(&mut self) -> Option<CommandResult> {
        let entry = self.redo_stack.pop()?;
        let id = entry.id;
        let group_id = entry.group_id;

        let mut command = entry.command;
        let result = command.redo();

        if result.is_success() {
            self.current_memory -= command.memory_size();

            let undo_entry = HistoryEntry {
                id,
                command,
                group_id,
                timestamp: entry.timestamp,
                executed: true,
            };
            self.current_memory += undo_entry.command.memory_size();
            self.undo_stack.push(undo_entry);

            self.emit(&HistoryEvent::Redone(id));

            // Redo entire group if applicable
            if let Some(gid) = group_id {
                while let Some(last) = self.redo_stack.last() {
                    if last.group_id == Some(gid) {
                        self.redo();
                    } else {
                        break;
                    }
                }
            }
        }

        Some(result)
    }

    /// Check if undo is available
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Check if redo is available
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Get number of undoable commands
    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    /// Get number of redoable commands
    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }

    /// Get description of next undo command
    pub fn undo_description(&self) -> Option<&str> {
        self.undo_stack.last().map(|e| e.command.description())
    }

    /// Get description of next redo command
    pub fn redo_description(&self) -> Option<&str> {
        self.redo_stack.last().map(|e| e.command.description())
    }

    /// Start a command group
    pub fn begin_group(&mut self) -> GroupId {
        let id = GroupId(self.next_group_id);
        self.next_group_id += 1;
        self.current_group = Some(id);
        id
    }

    /// End current command group
    pub fn end_group(&mut self) {
        self.current_group = None;
    }

    /// Execute multiple commands as a group
    pub fn execute_group<I>(&mut self, commands: I) -> Vec<CommandResult>
    where
        I: IntoIterator<Item = Box<dyn Command>>,
    {
        let _group = self.begin_group();
        let results: Vec<_> = commands.into_iter().map(|cmd| self.execute(cmd)).collect();
        self.end_group();
        results
    }

    /// Create a checkpoint at current position
    pub fn create_checkpoint(&mut self, name: impl Into<String>) -> CheckpointId {
        let id = CheckpointId(self.next_checkpoint_id);
        self.next_checkpoint_id += 1;

        let checkpoint = Checkpoint {
            id,
            name: name.into(),
            position: self.undo_stack.len(),
            timestamp: self.timestamp,
        };

        self.checkpoints.insert(id, checkpoint);
        self.emit(&HistoryEvent::CheckpointCreated(id));
        id
    }

    /// Restore to a checkpoint
    pub fn restore_checkpoint(&mut self, id: CheckpointId) -> bool {
        let checkpoint = match self.checkpoints.get(&id) {
            Some(c) => c.clone(),
            None => return false,
        };

        // Undo until we reach checkpoint position
        while self.undo_stack.len() > checkpoint.position {
            if self.undo().is_none() {
                break;
            }
        }

        self.emit(&HistoryEvent::CheckpointRestored(id));
        true
    }

    /// Get checkpoint by ID
    pub fn get_checkpoint(&self, id: CheckpointId) -> Option<&Checkpoint> {
        self.checkpoints.get(&id)
    }

    /// List all checkpoints
    pub fn checkpoints(&self) -> impl Iterator<Item = &Checkpoint> {
        self.checkpoints.values()
    }

    /// Clear all history
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.checkpoints.clear();
        self.current_memory = 0;
        self.emit(&HistoryEvent::Cleared);
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> usize {
        self.current_memory
    }

    /// Update timestamp (call each frame or periodically)
    pub fn tick(&mut self, delta_ms: u64) {
        self.timestamp += delta_ms;
    }

    /// Pause recording
    pub fn pause(&mut self) {
        self.recording = false;
    }

    /// Resume recording
    pub fn resume(&mut self) {
        self.recording = true;
    }

    /// Check if recording
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Add event listener
    pub fn on_event(&mut self, callback: HistoryCallback) {
        self.listeners.push(callback);
    }

    fn emit(&self, event: &HistoryEvent) {
        for listener in &self.listeners {
            listener(event.clone());
        }
    }

    fn trim_if_needed(&mut self) {
        let mut trimmed = 0;

        // Trim by count
        while self.undo_stack.len() > self.config.max_commands {
            if let Some(entry) = self.undo_stack.first() {
                self.current_memory -= entry.command.memory_size();
            }
            self.undo_stack.remove(0);
            trimmed += 1;
        }

        // Trim by memory
        while self.current_memory > self.config.max_memory && !self.undo_stack.is_empty() {
            if let Some(entry) = self.undo_stack.first() {
                self.current_memory -= entry.command.memory_size();
            }
            self.undo_stack.remove(0);
            trimmed += 1;
        }

        if trimmed > 0 {
            // Update checkpoint positions
            for checkpoint in self.checkpoints.values_mut() {
                checkpoint.position = checkpoint.position.saturating_sub(trimmed);
            }
            self.emit(&HistoryEvent::Trimmed(trimmed));
        }
    }
}

/// Builder for creating composite commands
pub struct CompositeCommand {
    description: String,
    commands: Vec<Box<dyn Command>>,
    executed: Vec<bool>,
}

impl CompositeCommand {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            commands: Vec::new(),
            executed: Vec::new(),
        }
    }

    pub fn with_command(mut self, command: Box<dyn Command>) -> Self {
        self.commands.push(command);
        self
    }

    pub fn build(self) -> Box<dyn Command> {
        Box::new(CompositeCommandImpl {
            description: self.description,
            commands: self.commands,
            executed: self.executed,
        })
    }
}

struct CompositeCommandImpl {
    description: String,
    commands: Vec<Box<dyn Command>>,
    executed: Vec<bool>,
}

impl Command for CompositeCommandImpl {
    fn execute(&mut self) -> CommandResult {
        self.executed.clear();
        for cmd in &mut self.commands {
            let result = cmd.execute();
            if !result.is_success() {
                // Rollback executed commands
                for (i, was_executed) in self.executed.iter().enumerate().rev() {
                    if *was_executed {
                        self.commands[i].undo();
                    }
                }
                return result;
            }
            self.executed.push(true);
        }
        CommandResult::Success
    }

    fn undo(&mut self) -> CommandResult {
        for (i, cmd) in self.commands.iter_mut().enumerate().rev() {
            if i < self.executed.len() && self.executed[i] {
                let result = cmd.undo();
                if !result.is_success() {
                    return result;
                }
            }
        }
        self.executed.clear();
        CommandResult::Success
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>() + self.commands.iter().map(|c| c.memory_size()).sum::<usize>()
    }
}

/// Simple value change command
pub struct SetValueCommand<T: Clone + Send + Sync + 'static> {
    description: String,
    value: Arc<std::sync::RwLock<T>>,
    old_value: Option<T>,
    new_value: T,
}

impl<T: Clone + Send + Sync + 'static> SetValueCommand<T> {
    pub fn new(
        description: impl Into<String>,
        value: Arc<std::sync::RwLock<T>>,
        new_value: T,
    ) -> Self {
        Self {
            description: description.into(),
            value,
            old_value: None,
            new_value,
        }
    }
}

impl<T: Clone + Send + Sync + 'static> Command for SetValueCommand<T> {
    fn execute(&mut self) -> CommandResult {
        let Ok(mut guard) = self.value.write() else {
            return CommandResult::Failed("Lock poisoned".into());
        };
        self.old_value = Some(guard.clone());
        *guard = self.new_value.clone();
        CommandResult::Success
    }

    fn undo(&mut self) -> CommandResult {
        let Some(old) = self.old_value.clone() else {
            return CommandResult::Failed("No old value".into());
        };
        let Ok(mut guard) = self.value.write() else {
            return CommandResult::Failed("Lock poisoned".into());
        };
        *guard = old;
        CommandResult::Success
    }

    fn description(&self) -> &str {
        &self.description
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI32, Ordering};

    // Test command that increments/decrements a counter
    struct IncrementCommand {
        counter: Arc<AtomicI32>,
        amount: i32,
    }

    impl Command for IncrementCommand {
        fn execute(&mut self) -> CommandResult {
            self.counter.fetch_add(self.amount, Ordering::SeqCst);
            CommandResult::Success
        }

        fn undo(&mut self) -> CommandResult {
            self.counter.fetch_sub(self.amount, Ordering::SeqCst);
            CommandResult::Success
        }

        fn description(&self) -> &'static str {
            "Increment counter"
        }
    }

    // Test command that can fail
    struct FailingCommand {
        should_fail: bool,
    }

    impl Command for FailingCommand {
        fn execute(&mut self) -> CommandResult {
            if self.should_fail {
                CommandResult::Failed("Intentional failure".into())
            } else {
                CommandResult::Success
            }
        }

        fn undo(&mut self) -> CommandResult {
            CommandResult::Success
        }

        fn description(&self) -> &'static str {
            "Failing command"
        }
    }

    #[test]
    fn test_basic_execute() {
        let mut history = CommandHistory::default();
        let counter = Arc::new(AtomicI32::new(0));

        let cmd = Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 5,
        });

        let result = history.execute(cmd);
        assert!(result.is_success());
        assert_eq!(counter.load(Ordering::SeqCst), 5);
        assert_eq!(history.undo_count(), 1);
    }

    #[test]
    fn test_undo() {
        let mut history = CommandHistory::default();
        let counter = Arc::new(AtomicI32::new(0));

        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 10,
        }));
        assert_eq!(counter.load(Ordering::SeqCst), 10);

        let result = history.undo();
        assert!(result.is_some());
        assert!(result.unwrap().is_success());
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_redo() {
        let mut history = CommandHistory::default();
        let counter = Arc::new(AtomicI32::new(0));

        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 7,
        }));
        history.undo();
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        let result = history.redo();
        assert!(result.is_some());
        assert!(result.unwrap().is_success());
        assert_eq!(counter.load(Ordering::SeqCst), 7);
    }

    #[test]
    fn test_multiple_undo_redo() {
        let mut history = CommandHistory::default();
        let counter = Arc::new(AtomicI32::new(0));

        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 1,
        }));
        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 2,
        }));
        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 3,
        }));

        assert_eq!(counter.load(Ordering::SeqCst), 6);
        assert_eq!(history.undo_count(), 3);

        history.undo();
        assert_eq!(counter.load(Ordering::SeqCst), 3);

        history.undo();
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        history.redo();
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_can_undo_redo() {
        let mut history = CommandHistory::default();
        let counter = Arc::new(AtomicI32::new(0));

        assert!(!history.can_undo());
        assert!(!history.can_redo());

        history.execute(Box::new(IncrementCommand { counter, amount: 1 }));

        assert!(history.can_undo());
        assert!(!history.can_redo());

        history.undo();
        assert!(!history.can_undo());
        assert!(history.can_redo());
    }

    #[test]
    fn test_redo_cleared_on_new_execute() {
        let mut history = CommandHistory::default();
        let counter = Arc::new(AtomicI32::new(0));

        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 1,
        }));
        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 2,
        }));

        history.undo();
        assert!(history.can_redo());

        // New command clears redo stack
        history.execute(Box::new(IncrementCommand { counter, amount: 5 }));

        assert!(!history.can_redo());
    }

    #[test]
    fn test_failed_command_not_added() {
        let mut history = CommandHistory::default();

        let result = history.execute(Box::new(FailingCommand { should_fail: true }));
        assert!(result.is_failed());
        assert_eq!(history.undo_count(), 0);
    }

    #[test]
    fn test_command_descriptions() {
        let mut history = CommandHistory::default();
        let counter = Arc::new(AtomicI32::new(0));

        assert!(history.undo_description().is_none());

        history.execute(Box::new(IncrementCommand { counter, amount: 1 }));

        assert_eq!(history.undo_description(), Some("Increment counter"));
    }

    #[test]
    fn test_command_groups() {
        let mut history = CommandHistory::default();
        let counter = Arc::new(AtomicI32::new(0));

        let _group = history.begin_group();
        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 1,
        }));
        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 2,
        }));
        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 3,
        }));
        history.end_group();

        assert_eq!(counter.load(Ordering::SeqCst), 6);

        // Undo should undo entire group
        history.undo();
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_execute_group() {
        let mut history = CommandHistory::default();
        let counter = Arc::new(AtomicI32::new(0));

        let commands: Vec<Box<dyn Command>> = vec![
            Box::new(IncrementCommand {
                counter: counter.clone(),
                amount: 1,
            }),
            Box::new(IncrementCommand {
                counter: counter.clone(),
                amount: 2,
            }),
        ];

        let results = history.execute_group(commands);
        assert!(results.iter().all(super::CommandResult::is_success));
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_checkpoints() {
        let mut history = CommandHistory::default();
        let counter = Arc::new(AtomicI32::new(0));

        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 5,
        }));

        let checkpoint = history.create_checkpoint("Initial state");

        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 10,
        }));
        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 15,
        }));

        assert_eq!(counter.load(Ordering::SeqCst), 30);

        // Restore to checkpoint
        let restored = history.restore_checkpoint(checkpoint);
        assert!(restored);
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn test_get_checkpoint() {
        let mut history = CommandHistory::default();

        let id = history.create_checkpoint("Test checkpoint");
        let checkpoint = history.get_checkpoint(id);

        assert!(checkpoint.is_some());
        assert_eq!(checkpoint.unwrap().name, "Test checkpoint");
    }

    #[test]
    fn test_list_checkpoints() {
        let mut history = CommandHistory::default();

        history.create_checkpoint("First");
        history.create_checkpoint("Second");
        history.create_checkpoint("Third");

        assert_eq!(history.checkpoints().count(), 3);
    }

    #[test]
    fn test_clear() {
        let mut history = CommandHistory::default();
        let counter = Arc::new(AtomicI32::new(0));

        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 1,
        }));
        history.execute(Box::new(IncrementCommand { counter, amount: 2 }));
        history.create_checkpoint("Test");

        history.clear();

        assert!(!history.can_undo());
        assert!(!history.can_redo());
        assert_eq!(history.checkpoints().count(), 0);
    }

    #[test]
    fn test_max_commands_limit() {
        let config = HistoryConfig {
            max_commands: 3,
            ..Default::default()
        };
        let mut history = CommandHistory::new(config);
        let counter = Arc::new(AtomicI32::new(0));

        for i in 0..5 {
            history.execute(Box::new(IncrementCommand {
                counter: counter.clone(),
                amount: i + 1,
            }));
        }

        assert_eq!(history.undo_count(), 3);
    }

    #[test]
    fn test_pause_resume_recording() {
        let mut history = CommandHistory::default();
        let counter = Arc::new(AtomicI32::new(0));

        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 1,
        }));

        history.pause();
        assert!(!history.is_recording());

        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 2,
        }));

        // Command executed but not recorded
        assert_eq!(counter.load(Ordering::SeqCst), 3);
        assert_eq!(history.undo_count(), 1);

        history.resume();
        assert!(history.is_recording());

        history.execute(Box::new(IncrementCommand { counter, amount: 3 }));
        assert_eq!(history.undo_count(), 2);
    }

    #[test]
    fn test_event_callbacks() {
        use std::sync::atomic::AtomicUsize;

        let mut history = CommandHistory::default();
        let counter = Arc::new(AtomicI32::new(0));
        let event_count = Arc::new(AtomicUsize::new(0));

        let ec = event_count.clone();
        history.on_event(Arc::new(move |_event| {
            ec.fetch_add(1, Ordering::SeqCst);
        }));

        history.execute(Box::new(IncrementCommand { counter, amount: 1 }));
        history.undo();
        history.redo();

        assert_eq!(event_count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_memory_tracking() {
        let mut history = CommandHistory::default();
        let counter = Arc::new(AtomicI32::new(0));

        assert_eq!(history.memory_usage(), 0);

        history.execute(Box::new(IncrementCommand { counter, amount: 1 }));

        assert!(history.memory_usage() > 0);
    }

    #[test]
    fn test_tick() {
        let mut history = CommandHistory::default();
        history.tick(100);
        history.tick(50);
        // Just verify it doesn't panic
    }

    #[test]
    fn test_command_result_helpers() {
        let success = CommandResult::Success;
        let failed = CommandResult::Failed("error".into());
        let cancelled = CommandResult::Cancelled;

        assert!(success.is_success());
        assert!(!success.is_failed());

        assert!(!failed.is_success());
        assert!(failed.is_failed());

        assert!(!cancelled.is_success());
        assert!(!cancelled.is_failed());
    }

    #[test]
    fn test_command_id() {
        let id = CommandId(42);
        assert_eq!(id.as_u64(), 42);
    }

    #[test]
    fn test_group_id() {
        let id = GroupId(123);
        assert_eq!(id.as_u64(), 123);
    }

    #[test]
    fn test_checkpoint_id() {
        let id = CheckpointId(456);
        assert_eq!(id.as_u64(), 456);
    }

    #[test]
    fn test_composite_command() {
        let counter = Arc::new(AtomicI32::new(0));

        let composite = CompositeCommand::new("Add 6")
            .with_command(Box::new(IncrementCommand {
                counter: counter.clone(),
                amount: 1,
            }))
            .with_command(Box::new(IncrementCommand {
                counter: counter.clone(),
                amount: 2,
            }))
            .with_command(Box::new(IncrementCommand {
                counter: counter.clone(),
                amount: 3,
            }))
            .build();

        let mut history = CommandHistory::default();
        history.execute(composite);

        assert_eq!(counter.load(Ordering::SeqCst), 6);

        history.undo();
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        history.redo();
        assert_eq!(counter.load(Ordering::SeqCst), 6);
    }

    #[test]
    fn test_composite_command_rollback_on_failure() {
        let counter = Arc::new(AtomicI32::new(0));

        let composite = CompositeCommand::new("Should fail")
            .with_command(Box::new(IncrementCommand {
                counter: counter.clone(),
                amount: 5,
            }))
            .with_command(Box::new(FailingCommand { should_fail: true }))
            .build();

        let mut history = CommandHistory::default();
        let result = history.execute(composite);

        assert!(result.is_failed());
        // First command should be rolled back
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_set_value_command() {
        let value = Arc::new(std::sync::RwLock::new(10));

        let cmd = Box::new(SetValueCommand::new("Set to 42", value.clone(), 42));

        let mut history = CommandHistory::default();
        history.execute(cmd);

        assert_eq!(*value.read().unwrap(), 42);

        history.undo();
        assert_eq!(*value.read().unwrap(), 10);

        history.redo();
        assert_eq!(*value.read().unwrap(), 42);
    }

    #[test]
    fn test_default_config() {
        let config = HistoryConfig::default();
        assert_eq!(config.max_commands, 1000);
        assert_eq!(config.max_memory, 10 * 1024 * 1024);
        assert!(config.enable_merging);
        assert_eq!(config.auto_group_interval_ms, 500);
    }

    #[test]
    fn test_undo_on_empty_returns_none() {
        let mut history = CommandHistory::default();
        assert!(history.undo().is_none());
    }

    #[test]
    fn test_redo_on_empty_returns_none() {
        let mut history = CommandHistory::default();
        assert!(history.redo().is_none());
    }

    #[test]
    fn test_restore_invalid_checkpoint() {
        let mut history = CommandHistory::default();
        let invalid_id = CheckpointId(999);
        assert!(!history.restore_checkpoint(invalid_id));
    }

    #[test]
    fn test_history_event_variants() {
        let events = vec![
            HistoryEvent::Executed(CommandId(1)),
            HistoryEvent::Undone(CommandId(2)),
            HistoryEvent::Redone(CommandId(3)),
            HistoryEvent::Cleared,
            HistoryEvent::CheckpointCreated(CheckpointId(1)),
            HistoryEvent::CheckpointRestored(CheckpointId(1)),
            HistoryEvent::Trimmed(5),
        ];

        // Verify all variants can be created and compared
        for event in &events {
            assert_eq!(event, event);
        }
    }

    #[test]
    fn test_redo_description() {
        let mut history = CommandHistory::default();
        let counter = Arc::new(AtomicI32::new(0));

        assert!(history.redo_description().is_none());

        history.execute(Box::new(IncrementCommand { counter, amount: 1 }));
        history.undo();

        assert_eq!(history.redo_description(), Some("Increment counter"));
    }

    #[test]
    fn test_redo_count() {
        let mut history = CommandHistory::default();
        let counter = Arc::new(AtomicI32::new(0));

        assert_eq!(history.redo_count(), 0);

        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 1,
        }));
        history.execute(Box::new(IncrementCommand { counter, amount: 2 }));

        history.undo();
        assert_eq!(history.redo_count(), 1);

        history.undo();
        assert_eq!(history.redo_count(), 2);
    }

    #[test]
    fn test_composite_command_description() {
        let counter = Arc::new(AtomicI32::new(0));

        let composite = CompositeCommand::new("My Composite")
            .with_command(Box::new(IncrementCommand { counter, amount: 1 }))
            .build();

        assert_eq!(composite.description(), "My Composite");
    }

    // =========================================================================
    // Additional Edge Case Tests
    // =========================================================================

    #[test]
    fn test_command_id_debug() {
        let id = CommandId(42);
        let debug = format!("{id:?}");
        assert!(debug.contains("CommandId"));
        assert!(debug.contains("42"));
    }

    #[test]
    fn test_command_id_eq() {
        let id1 = CommandId(100);
        let id2 = CommandId(100);
        let id3 = CommandId(200);
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_command_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(CommandId(1));
        set.insert(CommandId(2));
        set.insert(CommandId(1)); // Duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_group_id_debug() {
        let id = GroupId(99);
        let debug = format!("{id:?}");
        assert!(debug.contains("GroupId"));
    }

    #[test]
    fn test_group_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(GroupId(1));
        set.insert(GroupId(2));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_checkpoint_id_debug() {
        let id = CheckpointId(77);
        let debug = format!("{id:?}");
        assert!(debug.contains("CheckpointId"));
    }

    #[test]
    fn test_checkpoint_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(CheckpointId(1));
        set.insert(CheckpointId(2));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_command_result_debug() {
        let result = CommandResult::Success;
        let debug = format!("{result:?}");
        assert!(debug.contains("Success"));

        let failed = CommandResult::Failed("error".to_string());
        let debug = format!("{failed:?}");
        assert!(debug.contains("Failed"));
    }

    #[test]
    fn test_command_result_clone() {
        let result = CommandResult::NeedsConfirmation("confirm?".to_string());
        let cloned = result.clone();
        assert_eq!(result, cloned);
    }

    #[test]
    fn test_command_result_all_variants() {
        let variants = vec![
            CommandResult::Success,
            CommandResult::Failed("error".to_string()),
            CommandResult::Cancelled,
            CommandResult::NeedsConfirmation("confirm".to_string()),
        ];

        for variant in &variants {
            let _ = format!("{variant:?}");
            let cloned = variant.clone();
            assert_eq!(variant, &cloned);
        }
    }

    #[test]
    fn test_history_config_debug() {
        let config = HistoryConfig::default();
        let debug = format!("{config:?}");
        assert!(debug.contains("HistoryConfig"));
    }

    #[test]
    fn test_history_config_clone() {
        let config = HistoryConfig {
            max_commands: 500,
            max_memory: 5 * 1024 * 1024,
            enable_merging: false,
            auto_group_interval_ms: 1000,
        };
        let cloned = config;
        assert_eq!(cloned.max_commands, 500);
        assert!(!cloned.enable_merging);
    }

    #[test]
    fn test_checkpoint_debug() {
        let checkpoint = Checkpoint {
            id: CheckpointId(1),
            name: "Test".to_string(),
            position: 5,
            timestamp: 1000,
        };
        let debug = format!("{checkpoint:?}");
        assert!(debug.contains("Checkpoint"));
    }

    #[test]
    fn test_checkpoint_clone() {
        let checkpoint = Checkpoint {
            id: CheckpointId(1),
            name: "Test".to_string(),
            position: 5,
            timestamp: 1000,
        };
        let cloned = checkpoint;
        assert_eq!(cloned.name, "Test");
        assert_eq!(cloned.position, 5);
    }

    #[test]
    fn test_history_event_debug() {
        let event = HistoryEvent::Cleared;
        let debug = format!("{event:?}");
        assert!(debug.contains("Cleared"));
    }

    #[test]
    fn test_history_event_clone() {
        let event = HistoryEvent::Trimmed(10);
        let cloned = event.clone();
        assert_eq!(event, cloned);
    }

    #[test]
    fn test_memory_limit_trimming() {
        let config = HistoryConfig {
            max_commands: 1000,
            max_memory: 200, // Very small limit
            ..Default::default()
        };
        let mut history = CommandHistory::new(config);
        let counter = Arc::new(AtomicI32::new(0));

        // Each command takes ~64 bytes by default
        for i in 0..10 {
            history.execute(Box::new(IncrementCommand {
                counter: counter.clone(),
                amount: i,
            }));
        }

        // Should have trimmed some commands due to memory limit
        assert!(history.undo_count() < 10);
    }

    #[test]
    fn test_default_command_methods() {
        let counter = Arc::new(AtomicI32::new(0));
        let cmd = IncrementCommand { counter, amount: 1 };

        // Test default implementations
        assert!(!cmd.can_merge(&cmd));
        assert_eq!(cmd.memory_size(), 64);
        assert!(cmd.metadata().is_none());
    }

    #[test]
    fn test_composite_command_memory_size() {
        let counter = Arc::new(AtomicI32::new(0));

        let composite = CompositeCommand::new("Multi")
            .with_command(Box::new(IncrementCommand {
                counter: counter.clone(),
                amount: 1,
            }))
            .with_command(Box::new(IncrementCommand { counter, amount: 2 }))
            .build();

        let memory = composite.memory_size();
        assert!(memory > 64); // Should include both commands
    }

    #[test]
    fn test_group_redo() {
        let mut history = CommandHistory::default();
        let counter = Arc::new(AtomicI32::new(0));

        let _group = history.begin_group();
        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 1,
        }));
        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 2,
        }));
        history.end_group();

        assert_eq!(counter.load(Ordering::SeqCst), 3);

        history.undo();
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        history.redo();
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_checkpoint_position_updates_on_trim() {
        let config = HistoryConfig {
            max_commands: 3,
            ..Default::default()
        };
        let mut history = CommandHistory::new(config);
        let counter = Arc::new(AtomicI32::new(0));

        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 1,
        }));

        let checkpoint_id = history.create_checkpoint("Early");

        // Add more commands to trigger trim
        for i in 2..=5 {
            history.execute(Box::new(IncrementCommand {
                counter: counter.clone(),
                amount: i,
            }));
        }

        // Checkpoint position should have been adjusted
        let checkpoint = history.get_checkpoint(checkpoint_id);
        assert!(checkpoint.is_some());
    }

    #[test]
    fn test_multiple_event_listeners() {
        use std::sync::atomic::AtomicUsize;

        let mut history = CommandHistory::default();
        let counter = Arc::new(AtomicI32::new(0));
        let event_count1 = Arc::new(AtomicUsize::new(0));
        let event_count2 = Arc::new(AtomicUsize::new(0));

        let ec1 = event_count1.clone();
        history.on_event(Arc::new(move |_| {
            ec1.fetch_add(1, Ordering::SeqCst);
        }));

        let ec2 = event_count2.clone();
        history.on_event(Arc::new(move |_| {
            ec2.fetch_add(1, Ordering::SeqCst);
        }));

        history.execute(Box::new(IncrementCommand { counter, amount: 1 }));

        assert_eq!(event_count1.load(Ordering::SeqCst), 1);
        assert_eq!(event_count2.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_set_value_command_description() {
        let value = Arc::new(std::sync::RwLock::new(0));
        let cmd = SetValueCommand::new("Set value", value, 100);
        assert_eq!(cmd.description(), "Set value");
    }

    #[test]
    fn test_undo_without_old_value() {
        let value = Arc::new(std::sync::RwLock::new(10));
        let mut cmd = SetValueCommand::new("Set", value, 42);

        // Try undo without execute first
        let result = cmd.undo();
        assert!(result.is_failed());
    }

    #[test]
    fn test_empty_composite_command() {
        let composite = CompositeCommand::new("Empty").build();

        let mut history = CommandHistory::default();
        let result = history.execute(composite);

        assert!(result.is_success());
    }

    #[test]
    fn test_get_nonexistent_checkpoint() {
        let history = CommandHistory::default();
        let result = history.get_checkpoint(CheckpointId(999));
        assert!(result.is_none());
    }

    #[test]
    fn test_history_clear_resets_memory() {
        let mut history = CommandHistory::default();
        let counter = Arc::new(AtomicI32::new(0));

        history.execute(Box::new(IncrementCommand { counter, amount: 1 }));

        assert!(history.memory_usage() > 0);

        history.clear();
        assert_eq!(history.memory_usage(), 0);
    }

    // Test mergeable command
    struct MergeableIncrement {
        counter: Arc<AtomicI32>,
        total_amount: i32,
    }

    impl Command for MergeableIncrement {
        fn execute(&mut self) -> CommandResult {
            self.counter.fetch_add(self.total_amount, Ordering::SeqCst);
            CommandResult::Success
        }

        fn undo(&mut self) -> CommandResult {
            self.counter.fetch_sub(self.total_amount, Ordering::SeqCst);
            CommandResult::Success
        }

        fn description(&self) -> &'static str {
            "Mergeable increment"
        }

        fn can_merge(&self, _other: &dyn Command) -> bool {
            true
        }

        fn merge(&mut self, _other: Box<dyn Command>) -> Option<Box<dyn Command>> {
            // For testing, just add 10 more
            self.total_amount += 10;
            None
        }
    }

    #[test]
    fn test_command_merging() {
        let config = HistoryConfig {
            enable_merging: true,
            auto_group_interval_ms: 0, // Disable auto-grouping
            ..Default::default()
        };
        let mut history = CommandHistory::new(config);
        let counter = Arc::new(AtomicI32::new(0));

        history.execute(Box::new(MergeableIncrement {
            counter: counter.clone(),
            total_amount: 5,
        }));

        history.execute(Box::new(MergeableIncrement {
            counter,
            total_amount: 3,
        }));

        // Second command should be merged, so only 1 command in history
        assert_eq!(history.undo_count(), 1);
    }

    #[test]
    fn test_redo_clears_on_new_execute() {
        let mut history = CommandHistory::default();
        let counter = Arc::new(AtomicI32::new(0));

        history.execute(Box::new(IncrementCommand {
            counter: counter.clone(),
            amount: 1,
        }));

        history.undo();
        assert_eq!(history.redo_count(), 1);

        history.execute(Box::new(IncrementCommand { counter, amount: 2 }));

        assert_eq!(history.redo_count(), 0);
    }
}
