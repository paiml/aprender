//! LCP-12: Async Compute with Sync Fallback

/// Result of an async operation with fallback capability.
#[derive(Debug, Clone)]
pub enum AsyncResult<T, E> {
    /// Operation completed asynchronously
    Async(T),
    /// Operation completed synchronously (fallback)
    Sync(T),
    /// Operation failed
    Error(E),
}

impl<T, E> AsyncResult<T, E> {
    /// Check if result was obtained asynchronously.
    #[must_use]
    pub fn is_async(&self) -> bool {
        matches!(self, AsyncResult::Async(_))
    }

    /// Check if result was obtained synchronously (fallback).
    #[must_use]
    pub fn is_sync(&self) -> bool {
        matches!(self, AsyncResult::Sync(_))
    }

    /// Check if operation failed.
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self, AsyncResult::Error(_))
    }

    /// Get the result value, regardless of async/sync.
    pub fn into_result(self) -> Result<T, E> {
        match self {
            AsyncResult::Async(v) | AsyncResult::Sync(v) => Ok(v),
            AsyncResult::Error(e) => Err(e),
        }
    }

    /// Map the success value.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> AsyncResult<U, E> {
        match self {
            AsyncResult::Async(v) => AsyncResult::Async(f(v)),
            AsyncResult::Sync(v) => AsyncResult::Sync(f(v)),
            AsyncResult::Error(e) => AsyncResult::Error(e),
        }
    }
}
