
/// PMAT-216: Implement TracedForward trait for CPU backend
impl TracedForward for AprTransformer {
    fn forward_traced(&mut self, tokens: &[u32]) -> Result<ForwardTrace> {
        // Delegate to the immutable method (CPU doesn't need mutation)
        AprTransformer::forward_traced(self, tokens)
    }
}

// Tests shattered to tests/ directory (PMAT-803)
#[cfg(test)]
mod tests;
