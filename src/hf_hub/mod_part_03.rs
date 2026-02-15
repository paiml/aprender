
impl Default for HfHubClient {
    fn default() -> Self {
        Self::new().expect("Failed to create HfHubClient")
    }
}

// ============================================================================
// Unit Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod tests;
