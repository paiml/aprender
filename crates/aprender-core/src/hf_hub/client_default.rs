use super::HfHubClient;

impl Default for HfHubClient {
    fn default() -> Self {
        Self::new().expect("Failed to create HfHubClient")
    }
}
