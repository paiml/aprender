impl CommandGenerator {
    /// Create generator with common dev command templates
    #[must_use]
    pub fn new() -> Self {
        Self {
            templates: Self::default_templates(),
        }
    }
}
