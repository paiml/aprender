use super::HfHubClient;

impl HfHubClient {}

#[path = "client_impl.rs"]
mod client_impl;

#[path = "upload.rs"]
mod upload;

/// Body text of an HTTP response (empty when it cannot be read).
///
/// The upload paths turn `http_status_as_error(false)` on so an error status
/// keeps its body for the message (ureq 2's `Error::Status(code, resp)`).
#[cfg(feature = "hf-hub-integration")]
fn body_text(resp: &mut ureq::http::Response<ureq::Body>) -> String {
    resp.body_mut().read_to_string().unwrap_or_default()
}
