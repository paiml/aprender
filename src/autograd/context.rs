//! Execution context for managing computational graphs

/// Context for managing the computational graph
pub struct Context {
    // For now, context is minimal. In future, it could track:
    // - All tensors for memory management
    // - Training vs inference mode
    // - Random state
    training: bool,
}

impl Context {
    /// Create a new context
    pub fn new() -> Self {
        Self { training: true }
    }

    /// Set training mode
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set evaluation mode
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Check if in training mode
    pub fn is_training(&self) -> bool {
        self.training
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_new() {
        let ctx = Context::new();
        assert!(ctx.is_training());
    }

    #[test]
    fn test_context_default() {
        let ctx = Context::default();
        assert!(ctx.is_training());
    }

    #[test]
    fn test_context_train_mode() {
        let mut ctx = Context::new();
        ctx.eval();
        assert!(!ctx.is_training());

        ctx.train();
        assert!(ctx.is_training());
    }

    #[test]
    fn test_context_eval_mode() {
        let mut ctx = Context::new();
        ctx.eval();
        assert!(!ctx.is_training());
    }
}
