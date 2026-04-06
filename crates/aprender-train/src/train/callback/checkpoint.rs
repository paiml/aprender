//! Checkpoint callback for saving model state periodically

use std::path::PathBuf;

use super::traits::{CallbackAction, CallbackContext, TrainerCallback};

/// Checkpoint callback to save model state periodically
///
/// Saves model state every N epochs or when a new best loss is achieved.
#[derive(Clone, Debug)]
pub struct CheckpointCallback {
    /// Directory to save checkpoints
    checkpoint_dir: PathBuf,
    /// Save every N epochs (None = only save best)
    save_every: Option<usize>,
    /// Save on best loss
    save_best: bool,
    /// Best loss seen
    best_loss: f32,
    /// Last saved epoch
    pub(crate) last_saved_epoch: Option<usize>,
}

impl CheckpointCallback {
    /// Create checkpoint callback saving to directory
    pub fn new(checkpoint_dir: impl Into<PathBuf>) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.into(),
            save_every: None,
            save_best: true,
            best_loss: f32::INFINITY,
            last_saved_epoch: None,
        }
    }

    /// Configure to save every N epochs
    pub fn save_every(mut self, epochs: usize) -> Self {
        self.save_every = Some(epochs);
        self
    }

    /// Configure to save on best loss
    pub fn save_best(mut self, save: bool) -> Self {
        self.save_best = save;
        self
    }

    /// Get checkpoint path for epoch
    pub fn checkpoint_path(&self, epoch: usize) -> PathBuf {
        self.checkpoint_dir.join(format!("checkpoint_epoch_{epoch}.json"))
    }

    /// Get best checkpoint path
    pub fn best_checkpoint_path(&self) -> PathBuf {
        self.checkpoint_dir.join("checkpoint_best.json")
    }

    /// Save checkpoint (placeholder - actual implementation needs model access)
    fn save_checkpoint(&mut self, epoch: usize, is_best: bool) {
        // Ensure directory exists
        std::fs::create_dir_all(&self.checkpoint_dir).ok();

        // Placeholder: In real implementation, would serialize model state
        let path = if is_best { self.best_checkpoint_path() } else { self.checkpoint_path(epoch) };

        // Write a marker file (real implementation would save model weights)
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let info =
            format!(r#"{{"epoch": {epoch}, "is_best": {is_best}, "timestamp": {timestamp}}}"#);
        std::fs::write(&path, info).ok();

        self.last_saved_epoch = Some(epoch);
    }
}

impl TrainerCallback for CheckpointCallback {
    fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        let mut should_save = false;
        let mut is_best = false;

        // Check if we should save periodically
        if let Some(interval) = self.save_every {
            if (ctx.epoch + 1).is_multiple_of(interval) {
                should_save = true;
            }
        }

        // Check if this is the best model
        let loss = ctx.val_loss.unwrap_or(ctx.loss);
        if self.save_best && loss < self.best_loss {
            self.best_loss = loss;
            should_save = true;
            is_best = true;
        }

        if should_save {
            self.save_checkpoint(ctx.epoch, is_best);
        }

        CallbackAction::Continue
    }

    fn on_train_end(&mut self, ctx: &CallbackContext) {
        // Save final checkpoint
        self.save_checkpoint(ctx.epoch, false);
    }

    fn name(&self) -> &'static str {
        "CheckpointCallback"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_callback_paths() {
        let cb = CheckpointCallback::new("/tmp/checkpoints");
        assert_eq!(
            cb.checkpoint_path(5),
            PathBuf::from("/tmp/checkpoints/checkpoint_epoch_5.json")
        );
        assert_eq!(
            cb.best_checkpoint_path(),
            PathBuf::from("/tmp/checkpoints/checkpoint_best.json")
        );
    }

    #[test]
    fn test_checkpoint_callback_save_every() {
        let temp_dir = tempfile::tempdir().expect("temp file creation should succeed");
        let mut cb = CheckpointCallback::new(temp_dir.path()).save_every(2);

        let mut ctx = CallbackContext::default();
        ctx.loss = 1.0;
        cb.on_epoch_end(&ctx);

        ctx.epoch = 1;
        cb.on_epoch_end(&ctx);
        assert_eq!(cb.last_saved_epoch, Some(1));
    }

    #[test]
    fn test_checkpoint_callback_save_best_disabled() {
        let temp_dir = tempfile::tempdir().expect("temp file creation should succeed");
        let mut cb = CheckpointCallback::new(temp_dir.path()).save_best(false);

        let mut ctx = CallbackContext::default();
        ctx.loss = 0.1;
        cb.on_epoch_end(&ctx);
        assert!(cb.last_saved_epoch.is_none());
    }

    #[test]
    fn test_checkpoint_callback_on_train_end() {
        let temp_dir = tempfile::tempdir().expect("temp file creation should succeed");
        let mut cb = CheckpointCallback::new(temp_dir.path());

        let ctx = CallbackContext { epoch: 5, ..Default::default() };

        cb.on_train_end(&ctx);
        assert_eq!(cb.last_saved_epoch, Some(5));
    }

    #[test]
    fn test_checkpoint_callback_name() {
        let cb = CheckpointCallback::new("/tmp");
        assert_eq!(cb.name(), "CheckpointCallback");
    }

    #[test]
    fn test_checkpoint_callback_val_loss_for_best() {
        let temp_dir = tempfile::tempdir().expect("temp file creation should succeed");
        let mut cb = CheckpointCallback::new(temp_dir.path());

        let mut ctx = CallbackContext::default();
        ctx.loss = 1.0;
        ctx.val_loss = Some(0.5);
        cb.on_epoch_end(&ctx);
        assert_eq!(cb.best_loss, 0.5);
    }

    #[test]
    fn test_checkpoint_callback_clone() {
        let cb = CheckpointCallback::new("/tmp/test");
        let cloned = cb.clone();
        assert_eq!(cb.checkpoint_dir, cloned.checkpoint_dir);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Checkpoint paths should be consistent
        #[test]
        fn checkpoint_paths_are_consistent(
            epoch in 0usize..1000,
        ) {
            let cb = CheckpointCallback::new("/tmp/test");

            // Should generate predictable paths
            let path = cb.checkpoint_path(epoch);
            let expected = format!("/tmp/test/checkpoint_epoch_{epoch}.json");
            prop_assert_eq!(path, PathBuf::from(&expected));

            // Best path should be constant
            let best = cb.best_checkpoint_path();
            prop_assert_eq!(best, PathBuf::from("/tmp/test/checkpoint_best.json"));
        }
    }
}
