//! The cold-start loader (D-18): the artifact goes from S3 straight into ONE pre-sized
//! in-memory buffer.
//!
//! Lambda's scratch disk is 512 MB and cargo-pmcp cannot raise it (RESEARCH Pitfall 3),
//! so the ~0.85 GB artifact never touches a filesystem path. The object is split into
//! disjoint [`PART_BYTES`] slices of that buffer, each filled by its own ranged GET, with
//! at most [`CONCURRENCY`] in flight and up to [`RETRIES`] attempts per part. Every
//! attempt is bounded by [`ATTEMPT_TIMEOUT`] and the whole download by
//! [`DOWNLOAD_DEADLINE`]: attempt counts alone would let one stalled part keep the
//! in-flight load (the `OnceCell` every concurrent first request waits on) pending
//! indefinitely. A cut attempt keeps the bytes it landed and the retry resumes at the
//! first missing byte, so the attempt timeout bounds a stall rather than a transfer: at a
//! fair share of the link a 64 MiB part can need more than one attempt, and re-fetching it
//! from its first byte would throw that progress away on every cold start.
//!
//! The deadline is DERIVED from live cold samples, not guessed (plan 08-30, V4-b): the
//! work that must still fit after the download (sha256 + load ladder with its probe
//! replay + the maximal classify) plus decide-tool-boundary-v1's `margin_ms` is
//! [`POST_DOWNLOAD_RESERVE`], and [`DOWNLOAD_DEADLINE`] is the 30 s gateway cap minus
//! it — see that constant for the samples. A download past it cannot be answered inside
//! the cap anyway (the caller already has its 504), so the only useful outcome is to
//! fail the load, which leaves the cell empty so the next request retries. The handler
//! also bounds the download by the invocation's own remaining time
//! ([`crate::download_budget`]), so a load started late in an invocation is cut before
//! Lambda kills the invocation mid-build.
//!
//! A missing content length is refused (never read as 0), and a length over the
//! decide-apr-v1 cap is refused BEFORE the buffer is allocated. The pin is checked by
//! the caller ([`crate::resolve_from_fetcher`]) before any parse.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::Duration;

use futures::StreamExt as _;

/// One ranged GET's size: 64 MiB (spike 021/026).
pub const PART_BYTES: u64 = 64 * 1024 * 1024;
/// Ranged GETs in flight at once.
pub const CONCURRENCY: usize = 16;
/// Attempts per part (and for the length lookup).
pub const RETRIES: u32 = 5;
/// One attempt's bound; a slower attempt is cut and retried.
pub const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(8);
/// The largest measured post-download cost of one cold maximal request at
/// decide-tool-boundary-v1's 10,240 MB tier: `sha_ms + build_ms + classify_ms` (build_ms
/// is the decide-apr-v1 ladder INCLUDING its 2-row probe replay, plus the model build;
/// classify_ms is Lambda duration minus load_ms). Plan 08-30's four proven-cold samples
/// (08-LIVE-REDEPLOY-EVIDENCE.json `live_attempts[2].cold_samples`, Graviton2, identity
/// 24a44d7e...): #1 CONCENTRATED 3484 + 2824 + 8908 = 15216, #2 DISTRIBUTED
/// 3484 + 2904 + 6588 = 12976, #3 CONCENTRATED 3498 + 2852 + 8911 = **15261**, #4
/// DISTRIBUTED 3484 + 2837 + 6621 = 12942. The max is sample 3.
pub const MEASURED_POST_DOWNLOAD_MS: u64 = 15_261;
/// decide-tool-boundary-v1 `margin_ms` (asserted equal to the contract by a test).
pub const CONTRACT_MARGIN_MS: u64 = 4_000;
/// decide-tool-boundary-v1 `api_gateway_timeout_ms` (asserted equal to the contract).
pub const GATEWAY_CAP_MS: u64 = 30_000;
/// What must still fit after the download: [`MEASURED_POST_DOWNLOAD_MS`] +
/// [`CONTRACT_MARGIN_MS`] = 15261 + 4000 = 19261 ms (R in plan 08-30).
pub const POST_DOWNLOAD_RESERVE: Duration =
    Duration::from_millis(MEASURED_POST_DOWNLOAD_MS + CONTRACT_MARGIN_MS);
/// The whole download's bound: min(25 s, cap - R) = min(25000, 30000 - 19261) = **10739 ms**.
///
/// Derived at 10,240 MB from plan 08-30's four proven-cold samples (see
/// [`MEASURED_POST_DOWNLOAD_MS`]); their download_ms were 8954, 9065, 8927 and 8986, so
/// the largest (sample 2, 9065) clears it by 1674 ms. The 3,008 MB tier's downloads
/// (13494-17831 ms, plans 08-17/08-18) are NOT its reference set: that tier is superseded
/// (decide-tool-boundary-v1 7.0.0) and this constant is priced for the deployed one only.
/// It was 25 s ("the cap minus ~5 s of sha + build") until plan 08-30, which measured
/// sha + build alone at 6.3-6.4 s and the maximal classify at up to 8.9 s after it.
/// A re-measure at another tier or artifact re-derives it; it is never raised to absorb a
/// slow download.
pub const DOWNLOAD_DEADLINE: Duration = Duration::from_millis(10_739);

/// A boxed, `Send` fetch future (object-safe, so `dyn RangeFetcher` works too).
pub type FetchFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, FetchError>> + Send + 'a>>;

/// A transport failure from one fetch attempt (retried).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchError(pub String);

impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A source of byte ranges of one object.
pub trait RangeFetcher: Send + Sync {
    /// The object's length, or `None` when the store did not report one.
    fn content_length(&self) -> FetchFuture<'_, Option<u64>>;

    /// Fill `dest` from the object's bytes `[start, start + dest.len())` (the inclusive
    /// range `start..=start + dest.len() - 1`), returning how many bytes were written.
    /// Writing straight into the caller's slice is what keeps the download to one buffer.
    ///
    /// `written` is published AS BYTES LAND (a prefix of `dest`): an attempt cut by its
    /// timeout, or failing mid-body, keeps what it wrote, and the retry resumes at
    /// `start + written` instead of re-fetching the part from its first byte.
    fn fetch_range<'a>(
        &'a self,
        start: u64,
        dest: &'a mut [u8],
        written: &'a AtomicUsize,
    ) -> FetchFuture<'a, usize>;
}

/// The download's sizes and bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DownloadPolicy {
    /// Bytes per ranged GET.
    pub part_bytes: u64,
    /// Parts in flight at once.
    pub concurrency: usize,
    /// Attempts per part.
    pub retries: u32,
    /// One attempt's bound.
    pub attempt_timeout: Duration,
    /// The whole download's bound.
    pub deadline: Duration,
}

impl DownloadPolicy {
    /// What the deployed Lambda uses.
    pub const DEPLOYED: Self = Self {
        part_bytes: PART_BYTES,
        concurrency: CONCURRENCY,
        retries: RETRIES,
        attempt_timeout: ATTEMPT_TIMEOUT,
        deadline: DOWNLOAD_DEADLINE,
    };
}

/// Why the download was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum S3LoadError {
    /// The length lookup kept failing.
    Head {
        /// Attempts made.
        attempts: u32,
        /// The last failure.
        reason: String,
    },
    /// The store reported no content length (never treated as 0).
    MissingLength,
    /// The length is over the cap; refused before the buffer was allocated.
    TooLarge {
        /// The reported length.
        length: u64,
        /// The cap.
        cap: u64,
    },
    /// A part failed every attempt.
    PartFailed {
        /// The part's first byte.
        offset: u64,
        /// Attempts made.
        attempts: u32,
        /// The last failure.
        last_error: String,
    },
    /// A part's body did not fill its range exactly.
    ShortBody {
        /// The part's first byte.
        offset: u64,
        /// The range length.
        expected: usize,
        /// Bytes received.
        got: usize,
    },
    /// The whole download ran past [`DOWNLOAD_DEADLINE`] and was abandoned.
    DeadlineExceeded {
        /// Time spent before it was abandoned.
        elapsed_ms: u128,
    },
}

impl S3LoadError {
    /// A short stable name for the failure class.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Head { .. } => "s3_head",
            Self::MissingLength => "s3_missing_length",
            Self::TooLarge { .. } => "s3_too_large",
            Self::PartFailed { .. } => "s3_part_failed",
            Self::ShortBody { .. } => "s3_short_body",
            Self::DeadlineExceeded { .. } => "s3_deadline",
        }
    }
}

impl fmt::Display for S3LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Head { attempts, reason } => {
                write!(f, "object length lookup failed after {attempts} attempts: {reason}")
            }
            Self::MissingLength => write!(f, "the object reported no content length"),
            Self::TooLarge { length, cap } => write!(
                f,
                "object is {length} bytes, over the decide-apr-v1 cap {cap}; refused before allocating"
            ),
            Self::PartFailed {
                offset,
                attempts,
                last_error,
            } => write!(
                f,
                "part at byte {offset} failed {attempts} attempts; last: {last_error}"
            ),
            Self::ShortBody {
                offset,
                expected,
                got,
            } => write!(
                f,
                "part at byte {offset} returned {got} bytes for a {expected}-byte range"
            ),
            Self::DeadlineExceeded { elapsed_ms } => write!(
                f,
                "download abandoned at its deadline after {elapsed_ms} ms"
            ),
        }
    }
}

impl std::error::Error for S3LoadError {}

/// Download the object into one in-memory buffer under the DEPLOYED policy.
///
/// # Errors
///
/// An [`S3LoadError`] naming the refusal.
pub async fn download_into_memory<F: RangeFetcher + ?Sized>(
    fetcher: &F,
    cap: u64,
) -> Result<Vec<u8>, S3LoadError> {
    download_into_memory_with(fetcher, cap, &DownloadPolicy::DEPLOYED).await
}

/// Download under an explicit `policy` (tests shrink the part size).
///
/// # Errors
///
/// An [`S3LoadError`] naming the refusal.
pub async fn download_into_memory_with<F: RangeFetcher + ?Sized>(
    fetcher: &F,
    cap: u64,
    policy: &DownloadPolicy,
) -> Result<Vec<u8>, S3LoadError> {
    let started = tokio::time::Instant::now();
    match tokio::time::timeout(policy.deadline, download_within(fetcher, cap, policy)).await {
        Ok(outcome) => outcome,
        // Dropping the inner future cancels every in-flight part and frees the buffer.
        Err(_) => Err(S3LoadError::DeadlineExceeded {
            elapsed_ms: started.elapsed().as_millis(),
        }),
    }
}

async fn download_within<F: RangeFetcher + ?Sized>(
    fetcher: &F,
    cap: u64,
    policy: &DownloadPolicy,
) -> Result<Vec<u8>, S3LoadError> {
    let length = content_length_with_retries(fetcher, policy)
        .await?
        .ok_or(S3LoadError::MissingLength)?;
    // The cap is checked on the REPORTED length, before a byte is allocated.
    if length > cap {
        return Err(S3LoadError::TooLarge { length, cap });
    }
    let len = usize::try_from(length).map_err(|_| S3LoadError::TooLarge { length, cap })?;
    let part = usize::try_from(policy.part_bytes.max(1)).unwrap_or(usize::MAX);
    let mut buffer = vec![0u8; len];
    {
        // A hand-rolled bounded FuturesUnordered rather than
        // `stream::iter(..).map(closure).buffer_unordered(n)`: the closure form makes
        // rustc unable to prove the Lambda handler's future `Send` for every lifetime
        // (the "implementation of Send is not general enough" limitation), which the
        // lib tests alone never exercise.
        let concurrency = policy.concurrency.max(1);
        let mut parts = buffer.chunks_mut(part);
        let mut offset = 0u64;
        let mut in_flight = futures::stream::FuturesUnordered::new();
        loop {
            while in_flight.len() < concurrency {
                let Some(slice) = parts.next() else { break };
                let at = offset;
                offset = offset.saturating_add(slice.len() as u64);
                in_flight.push(fill_part(fetcher, at, slice, policy));
            }
            match in_flight.next().await {
                // The first refusal wins; returning drops (cancels) the parts in flight.
                Some(outcome) => outcome?,
                None => break,
            }
        }
    }
    Ok(buffer)
}

async fn content_length_with_retries<F: RangeFetcher + ?Sized>(
    fetcher: &F,
    policy: &DownloadPolicy,
) -> Result<Option<u64>, S3LoadError> {
    let attempts = policy.retries.max(1);
    let mut last = String::new();
    for attempt in 1..=attempts {
        match tokio::time::timeout(policy.attempt_timeout, fetcher.content_length()).await {
            Ok(Ok(length)) => return Ok(length),
            Ok(Err(e)) => last = e.0,
            Err(_) => last = timed_out(policy),
        }
        if attempt < attempts {
            tracing::warn!(attempt, error = %last, "retrying object length lookup");
        }
    }
    Err(S3LoadError::Head {
        attempts,
        reason: last,
    })
}

fn timed_out(policy: &DownloadPolicy) -> String {
    format!(
        "attempt timed out after {} ms",
        policy.attempt_timeout.as_millis()
    )
}

/// Fill one slice from its ranged GETs: each attempt bounded by the attempt timeout, up
/// to `retries` attempts; a body that does not fill the range exactly is refused.
///
/// A cut or failed attempt KEEPS the bytes it landed (published through the fetcher's
/// `written` counter), and the next attempt resumes at the first missing byte. The attempt
/// timeout therefore bounds a stall, not a transfer: a part that is still making progress
/// when it is cut (13 parts share the link at cold start, so a fair-share 64 MiB part needs
/// longer than one attempt) is continued, never re-fetched from its first byte.
async fn fill_part<F: RangeFetcher + ?Sized>(
    fetcher: &F,
    offset: u64,
    dest: &mut [u8],
    policy: &DownloadPolicy,
) -> Result<(), S3LoadError> {
    let expected = dest.len();
    let attempts = policy.retries.max(1);
    let mut done = 0usize;
    let mut last = String::new();
    for attempt in 1..=attempts {
        let written = AtomicUsize::new(0);
        let at = offset.saturating_add(done as u64);
        let outcome = tokio::time::timeout(
            policy.attempt_timeout,
            fetcher.fetch_range(at, &mut dest[done..], &written),
        )
        .await;
        match outcome {
            Ok(Ok(got)) if done + got == expected => return Ok(()),
            Ok(Ok(got)) => {
                return Err(S3LoadError::ShortBody {
                    offset,
                    expected,
                    got: done + got,
                })
            }
            Ok(Err(e)) => last = e.0,
            Err(_) => last = timed_out(policy),
        }
        // Keep what landed before the cut (never more than the slice it was given).
        done += written.load(AtomicOrdering::Acquire).min(expected - done);
        if done == expected {
            return Ok(());
        }
        if attempt < attempts {
            tracing::warn!(offset, attempt, resume_at = done, error = %last, "retrying part");
        }
    }
    Err(S3LoadError::PartFailed {
        offset,
        attempts,
        last_error: last,
    })
}

/// The production fetcher: `head_object` for the length, ranged `get_object` streamed
/// straight into the caller's slice.
#[derive(Debug, Clone)]
pub struct S3Fetcher {
    client: aws_sdk_s3::Client,
    bucket: String,
    key: String,
}

impl S3Fetcher {
    /// A fetcher over `client` for one object.
    #[must_use]
    pub fn new(client: aws_sdk_s3::Client, bucket: &str, key: &str) -> Self {
        Self {
            client,
            bucket: bucket.to_string(),
            key: key.to_string(),
        }
    }

    /// A fetcher using the default AWS credential and region chain (the Lambda role).
    pub async fn from_default_config(bucket: &str, key: &str) -> Self {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        Self::new(aws_sdk_s3::Client::new(&config), bucket, key)
    }
}

impl RangeFetcher for S3Fetcher {
    fn content_length(&self) -> FetchFuture<'_, Option<u64>> {
        Box::pin(async move {
            let head = self
                .client
                .head_object()
                .bucket(&self.bucket)
                .key(&self.key)
                .send()
                .await
                .map_err(|e| {
                    FetchError(format!(
                        "head_object: {}",
                        aws_sdk_s3::error::DisplayErrorContext(&e)
                    ))
                })?;
            // A negative length is as unusable as none; never read either as 0.
            Ok(head.content_length().and_then(|n| u64::try_from(n).ok()))
        })
    }

    fn fetch_range<'a>(
        &'a self,
        start: u64,
        dest: &'a mut [u8],
        progress: &'a AtomicUsize,
    ) -> FetchFuture<'a, usize> {
        Box::pin(async move {
            if dest.is_empty() {
                return Ok(0);
            }
            let last = start.saturating_add(dest.len() as u64 - 1);
            let object = self
                .client
                .get_object()
                .bucket(&self.bucket)
                .key(&self.key)
                .range(format!("bytes={start}-{last}"))
                .send()
                .await
                .map_err(|e| {
                    FetchError(format!(
                        "get_object: {}",
                        aws_sdk_s3::error::DisplayErrorContext(&e)
                    ))
                })?;
            let mut body = object.body;
            let mut written = 0usize;
            while let Some(chunk) = body
                .try_next()
                .await
                .map_err(|e| FetchError(format!("body: {e}")))?
            {
                let end = written + chunk.len();
                let Some(slot) = dest.get_mut(written..end) else {
                    return Err(FetchError(format!(
                        "body longer than the {}-byte range",
                        dest.len()
                    )));
                };
                slot.copy_from_slice(&chunk);
                written = end;
                progress.store(written, AtomicOrdering::Release);
            }
            Ok(written)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::time::Duration;

    use super::*;
    use crate::tests::tiny_bytes;
    use crate::{
        parse_s3_uri, resolve_from_fetcher, resolve_local, sha256_hex, LoadOnce, Model,
        ResolveError, Sha256Pin, SourceError,
    };

    /// When a fetch attempt hangs (to exercise the timeouts on a paused clock).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Stall {
        Never,
        Always,
        FirstAttemptAt(u64),
        /// The first attempt at this offset lands HALF its slice, then hangs: a part that
        /// was still making progress when its attempt was cut.
        HalfThenHangAt(u64),
    }

    /// An in-memory object with fault injection; records attempts per part.
    struct MemFetcher {
        data: Vec<u8>,
        length: Option<u64>,
        failures_before_success: HashMap<u64, u32>,
        short_at: Option<u64>,
        stall: Stall,
        attempts: Mutex<HashMap<u64, u32>>,
    }

    impl MemFetcher {
        fn new(data: Vec<u8>) -> Self {
            let length = Some(data.len() as u64);
            Self {
                data,
                length,
                failures_before_success: HashMap::new(),
                short_at: None,
                stall: Stall::Never,
                attempts: Mutex::new(HashMap::new()),
            }
        }

        fn attempts_at(&self, offset: u64) -> u32 {
            self.attempts
                .lock()
                .expect("attempts lock")
                .get(&offset)
                .copied()
                .unwrap_or(0)
        }

        fn total_attempts(&self) -> u32 {
            self.attempts.lock().expect("attempts lock").values().sum()
        }
    }

    impl RangeFetcher for MemFetcher {
        fn content_length(&self) -> FetchFuture<'_, Option<u64>> {
            Box::pin(async move { Ok(self.length) })
        }

        fn fetch_range<'a>(
            &'a self,
            start: u64,
            dest: &'a mut [u8],
            written: &'a AtomicUsize,
        ) -> FetchFuture<'a, usize> {
            Box::pin(async move {
                let attempt = {
                    let mut map = self.attempts.lock().expect("attempts lock");
                    let n = map.entry(start).or_insert(0);
                    *n += 1;
                    *n
                };
                let from = usize::try_from(start).expect("offset fits");
                let stalls = match self.stall {
                    Stall::Never => false,
                    Stall::Always => true,
                    Stall::FirstAttemptAt(at) => at == start && attempt == 1,
                    Stall::HalfThenHangAt(at) => {
                        if at == start && attempt == 1 {
                            let half = dest.len() / 2;
                            dest[..half].copy_from_slice(&self.data[from..from + half]);
                            written.store(half, AtomicOrdering::Release);
                            true
                        } else {
                            false
                        }
                    }
                };
                if stalls {
                    tokio::time::sleep(Duration::from_secs(3600)).await;
                }
                if let Some(&fails) = self.failures_before_success.get(&start) {
                    if attempt <= fails {
                        return Err(FetchError(format!("injected failure {attempt}")));
                    }
                }
                let mut n = dest.len();
                if self.short_at == Some(start) {
                    n -= 1;
                }
                dest[..n].copy_from_slice(&self.data[from..from + n]);
                written.store(n, AtomicOrdering::Release);
                Ok(n)
            })
        }
    }

    /// 4-byte parts: a 14-byte object is 3.5 parts (offsets 0, 4, 8, 12).
    const SMALL: DownloadPolicy = DownloadPolicy {
        part_bytes: 4,
        concurrency: 2,
        ..DownloadPolicy::DEPLOYED
    };

    fn object() -> Vec<u8> {
        (0u8..14)
            .map(|b| b.wrapping_mul(17).wrapping_add(3))
            .collect()
    }

    #[tokio::test]
    async fn download_ok_reassembles_an_uneven_object_byte_exact() {
        let fetcher = MemFetcher::new(object());
        let bytes = download_into_memory_with(&fetcher, 1_000, &SMALL)
            .await
            .expect("download");
        assert_eq!(bytes, object());
        for offset in [0, 4, 8, 12] {
            assert_eq!(fetcher.attempts_at(offset), 1, "offset {offset}");
        }
        assert_eq!(fetcher.total_attempts(), 4);
    }

    #[tokio::test]
    async fn transient_part_failure_is_retried_and_recorded() {
        let mut fetcher = MemFetcher::new(object());
        fetcher.failures_before_success.insert(8, 2);
        let bytes = download_into_memory_with(&fetcher, 1_000, &SMALL)
            .await
            .expect("two failures then success");
        assert_eq!(bytes, object());
        assert_eq!(fetcher.attempts_at(8), 3);
    }

    #[tokio::test]
    async fn persistent_part_failure_is_refused_after_five_attempts() {
        let mut fetcher = MemFetcher::new(object());
        fetcher.failures_before_success.insert(4, RETRIES);
        let err = download_into_memory_with(&fetcher, 1_000, &SMALL)
            .await
            .expect_err("a part that keeps failing is refused");
        match err {
            S3LoadError::PartFailed {
                offset,
                attempts,
                last_error,
            } => {
                assert_eq!((offset, attempts), (4, 5));
                assert!(last_error.contains("injected failure 5"), "{last_error}");
            }
            other => panic!("expected PartFailed, got {other:?}"),
        }
        assert_eq!(fetcher.attempts_at(4), RETRIES);
    }

    #[tokio::test]
    async fn short_body_is_refused_with_its_range() {
        let mut fetcher = MemFetcher::new(object());
        fetcher.short_at = Some(8);
        let err = download_into_memory_with(&fetcher, 1_000, &SMALL)
            .await
            .expect_err("short body");
        assert_eq!(
            err,
            S3LoadError::ShortBody {
                offset: 8,
                expected: 4,
                got: 3
            }
        );
    }

    #[tokio::test]
    async fn missing_length_is_refused_never_read_as_zero() {
        let mut fetcher = MemFetcher::new(object());
        fetcher.length = None;
        let err = download_into_memory_with(&fetcher, 1_000, &SMALL)
            .await
            .expect_err("missing length");
        assert_eq!(err, S3LoadError::MissingLength);
        assert_eq!(fetcher.total_attempts(), 0, "no range was fetched");
    }

    #[tokio::test]
    async fn over_cap_length_is_refused_before_allocating() {
        let mut fetcher = MemFetcher::new(object());
        fetcher.length = Some(11);
        let err = download_into_memory_with(&fetcher, 10, &SMALL)
            .await
            .expect_err("over cap");
        assert_eq!(
            err,
            S3LoadError::TooLarge {
                length: 11,
                cap: 10
            }
        );
        assert_eq!(fetcher.total_attempts(), 0);

        // u64::MAX under the contracted cap: allocating first would abort the process.
        fetcher.length = Some(u64::MAX);
        let cap = crate::contracted_cap();
        let err = download_into_memory(&fetcher, cap)
            .await
            .expect_err("over cap");
        assert_eq!(
            err,
            S3LoadError::TooLarge {
                length: u64::MAX,
                cap
            }
        );
        assert_eq!(fetcher.total_attempts(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn deadline_abandons_stalled_parts() {
        let mut fetcher = MemFetcher::new(object());
        fetcher.stall = Stall::Always;
        let policy = DownloadPolicy {
            part_bytes: 4,
            concurrency: 2,
            ..DownloadPolicy::DEPLOYED
        };
        let started = tokio::time::Instant::now();
        let err = download_into_memory_with(&fetcher, 1_000, &policy)
            .await
            .expect_err("stalled parts are abandoned");
        let waited = started.elapsed();
        match err {
            S3LoadError::DeadlineExceeded { elapsed_ms } => {
                assert!(
                    (10_739..11_739).contains(&elapsed_ms),
                    "abandoned at {elapsed_ms} ms"
                );
            }
            other => panic!("expected DeadlineExceeded, got {other:?}"),
        }
        assert!(waited >= DOWNLOAD_DEADLINE && waited < DOWNLOAD_DEADLINE + ATTEMPT_TIMEOUT);
        // Each stalled attempt was cut at ATTEMPT_TIMEOUT and retried: the retry at 8 s is
        // still stalled when the 10.739 s deadline abandons the download -> 2 attempts.
        assert_eq!(fetcher.attempts_at(0), 2);
    }

    /// A part still making progress when its attempt is cut keeps what landed: the retry
    /// asks only for the missing tail (a fresh range starting mid-part), never the part again.
    #[tokio::test(start_paused = true)]
    async fn cut_attempt_resumes_at_the_first_missing_byte() {
        let mut fetcher = MemFetcher::new(object());
        fetcher.stall = Stall::HalfThenHangAt(4);
        let bytes = download_into_memory_with(&fetcher, 1_000, &SMALL)
            .await
            .expect("the resumed part completes");
        assert_eq!(bytes, object());
        assert_eq!(
            fetcher.attempts_at(4),
            1,
            "the part's first byte is fetched once"
        );
        assert_eq!(
            fetcher.attempts_at(6),
            1,
            "the retry resumes at 4 + 2 landed bytes"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn slow_attempt_is_cut_at_attempt_timeout_and_retried() {
        let mut fetcher = MemFetcher::new(object());
        fetcher.stall = Stall::FirstAttemptAt(0);
        let started = tokio::time::Instant::now();
        let bytes = download_into_memory_with(&fetcher, 1_000, &SMALL)
            .await
            .expect("the retry succeeds");
        assert_eq!(bytes, object());
        assert_eq!(fetcher.attempts_at(0), 2);
        let waited = started.elapsed();
        assert!(
            waited >= ATTEMPT_TIMEOUT && waited < ATTEMPT_TIMEOUT + Duration::from_secs(1),
            "{waited:?}"
        );
    }

    #[tokio::test]
    async fn hash_mismatch_is_refused_before_the_ladder_runs() {
        let tiny = tiny_bytes().to_vec();
        let right = Sha256Pin::parse(crate::tests::tiny_golden_sha256()).expect("pin");
        let wrong = Sha256Pin::parse(&"0".repeat(64)).expect("pin");
        let policy = DownloadPolicy {
            part_bytes: 4096,
            ..DownloadPolicy::DEPLOYED
        };

        // The real artifact with the right pin loads, and serves that identity.
        let fetcher = MemFetcher::new(tiny.clone());
        let (model, timeline) =
            resolve_from_fetcher(&fetcher, &right, crate::contracted_cap(), &policy)
                .await
                .expect("pinned tiny artifact resolves from the fetcher");
        assert_eq!(model.identity().artifact_sha256, right.as_str());
        assert_eq!(timeline.source, "s3");
        assert_eq!(timeline.bytes, tiny.len() as u64);

        // The real artifact under a wrong pin is refused.
        let err = resolve_from_fetcher(&fetcher, &wrong, crate::contracted_cap(), &policy)
            .await
            .expect_err("wrong pin");
        assert!(matches!(err, ResolveError::HashMismatch { .. }), "{err:?}");

        // Garbage under a wrong pin is a HASH refusal, not a ladder one: the ladder
        // never ran. The same garbage under its own hash reaches the ladder, which
        // refuses it — so the order is pin first, then parse.
        let garbage = vec![7u8; 100];
        let fetcher = MemFetcher::new(garbage.clone());
        let err = resolve_from_fetcher(&fetcher, &wrong, crate::contracted_cap(), &policy)
            .await
            .expect_err("wrong pin on garbage");
        assert!(matches!(err, ResolveError::HashMismatch { .. }), "{err:?}");
        let own = Sha256Pin::parse(&sha256_hex(&garbage)).expect("pin");
        let err = resolve_from_fetcher(&fetcher, &own, crate::contracted_cap(), &policy)
            .await
            .expect_err("garbage reaches the ladder");
        assert!(matches!(err, ResolveError::Load(_)), "{err:?}");
    }

    #[test]
    fn uri_parse_accepts_s3_and_refuses_the_rest() {
        assert_eq!(
            parse_s3_uri("s3://bucket/decide/srv/abc.apr"),
            Ok(("bucket".to_string(), "decide/srv/abc.apr".to_string()))
        );
        for bad in [
            "https://bucket.s3.amazonaws.com/decide/abc.apr",
            "s3://bucket",
            "s3://bucket/",
            "s3:///key",
            "bucket/key",
        ] {
            assert!(
                matches!(parse_s3_uri(bad), Err(SourceError::BadS3Uri { .. })),
                "{bad} must be refused"
            );
        }
    }

    #[tokio::test]
    async fn failed_load_leaves_the_cell_empty_and_the_next_call_loads() {
        let slot: LoadOnce<Model> = LoadOnce::new();
        let tiny = tiny_bytes().to_vec();
        let pin = Sha256Pin::parse(crate::tests::tiny_golden_sha256()).expect("pin");
        let policy = DownloadPolicy {
            part_bytes: 4096,
            ..DownloadPolicy::DEPLOYED
        };

        let mut failing = MemFetcher::new(tiny.clone());
        failing.failures_before_success.insert(0, RETRIES);
        let first = slot
            .get_or_try_load(|| async {
                resolve_from_fetcher(&failing, &pin, crate::contracted_cap(), &policy)
                    .await
                    .map(|(m, _)| m)
            })
            .await;
        assert!(
            matches!(first, Err(ResolveError::S3(S3LoadError::PartFailed { .. }))),
            "{:?}",
            first.err()
        );
        assert!(slot.get().is_none(), "a failed load leaves the slot empty");

        let good = MemFetcher::new(tiny);
        let (model, performed) = slot
            .get_or_try_load(|| async {
                resolve_from_fetcher(&good, &pin, crate::contracted_cap(), &policy)
                    .await
                    .map(|(m, _)| m)
            })
            .await
            .expect("the second load succeeds");
        assert!(performed);
        assert_eq!(model.identity().artifact_sha256, pin.as_str());

        let (_, performed) = slot
            .get_or_try_load(|| async {
                Err::<Model, ResolveError>(ResolveError::Join("never".into()))
            })
            .await
            .expect("loaded value is reused");
        assert!(!performed, "only the loading call reports performed_load");
    }

    #[tokio::test]
    async fn local_bounded_refuses_by_metadata_before_reading() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("big.apr");
        let cap = 1_024u64;
        let file = std::fs::File::create(&path).expect("create");
        file.set_len(cap + 1).expect("sparse length");
        drop(file);
        let err = resolve_local(&path, None, cap)
            .await
            .expect_err("over the (shrunk) cap");
        match err {
            ResolveError::TooLarge {
                what,
                observed,
                cap: c,
            } => assert_eq!((what, observed, c), ("declared_length", cap + 1, cap)),
            other => panic!("expected TooLarge from metadata, got {other:?}"),
        }
    }
}
