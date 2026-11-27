//! #4600: what the current turn's prompt looked like in the request handed to the driver.
//!
//! Measured at the driver boundary, after the context window has done its truncation. It does
//! not see what a driver or a serve process does with the request afterwards.

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::driver::Message;
use super::result::AgentError;
use crate::serve::context::ContextManager;

/// Input accounting for the current turn's prompt.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct InputRecord {
    /// Bytes of the prompt the caller passed to the turn.
    pub input_bytes_sent: usize,
    /// Leading bytes of that prompt present, unchanged, in the request.
    pub input_bytes_consumed: usize,
    /// SHA-256 (lowercase hex) of the consumed bytes.
    pub prompt_sha256: String,
    /// Tokens of the prompt message, by the counter the context window uses.
    pub prompt_tokens: usize,
}

impl InputRecord {
    /// Measure `prompt` against `request`, whose last `User` message is the current turn's.
    ///
    /// # Errors
    ///
    /// [`AgentError::InputNotConsumed`] when fewer bytes are in the request than were sent.
    pub(super) fn measure(
        prompt: &str,
        request: &[Message],
        context: &ContextManager,
    ) -> Result<Self, AgentError> {
        let message = request.iter().rev().find(|m| matches!(m, Message::User(_)));
        let carried = match message {
            Some(Message::User(text)) => text.as_bytes(),
            _ => &[],
        };
        let sent = prompt.len();
        let consumed = prompt.bytes().zip(carried).take_while(|(a, b)| a == *b).count();
        let prompt_tokens = message.map_or(0, |m| context.estimate_tokens(&[m.to_chat_message()]));
        if consumed < sent {
            return Err(AgentError::InputNotConsumed { sent, consumed });
        }
        Ok(Self {
            input_bytes_sent: sent,
            input_bytes_consumed: consumed,
            prompt_sha256: format!("{:x}", Sha256::digest(&prompt.as_bytes()[..consumed])),
            prompt_tokens,
        })
    }
}

#[cfg(test)]
#[path = "input_record_tests.rs"]
mod tests;
