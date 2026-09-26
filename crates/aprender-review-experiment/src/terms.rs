//! Provider-terms tagging for `agent-trace-v1` rows (PRA-001 T12, spec §2.10;
//! contract `agent-trace-terms-v1`).
//!
//! Every captured row carries `provider`, an exact `model_id`, `access_channel`
//! and a `terms_ref` snapshot, so any pool can be rebuilt without a given
//! provider or channel in one call (P-TERMS). This records which terms governed
//! a row; it is not legal advice, and the operator's S-6 ruling governs use.
//!
//! A row whose tags are missing or inconsistent is REFUSED by every rebuild, even
//! one that excludes nothing: an untagged row cannot be shown not to come from an
//! excluded source.

use serde_json::Value;

/// Who served the lane (`provider` in §2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    Google,
    Local,
}

impl Provider {
    pub const ALL: [Provider; 3] = [Provider::Anthropic, Provider::Google, Provider::Local];

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Provider::Anthropic => "anthropic",
            Provider::Google => "google",
            Provider::Local => "local",
        }
    }

    /// The exact wire spelling; anything else is unknown.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.as_str() == s)
    }
}

/// How the lane was reached (`access_channel` in §2.2); terms differ by channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    AnthropicApi,
    ClaudeCodeCli,
    Antigravity,
    GeminiApi,
    Vertex,
    AprServe,
}

impl Channel {
    pub const ALL: [Channel; 6] = [
        Channel::AnthropicApi,
        Channel::ClaudeCodeCli,
        Channel::Antigravity,
        Channel::GeminiApi,
        Channel::Vertex,
        Channel::AprServe,
    ];

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Channel::AnthropicApi => "anthropic-api",
            Channel::ClaudeCodeCli => "claude-code-cli",
            Channel::Antigravity => "antigravity",
            Channel::GeminiApi => "gemini-api",
            Channel::Vertex => "vertex",
            Channel::AprServe => "apr-serve",
        }
    }

    /// The exact wire spelling; anything else is unknown.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.as_str() == s)
    }

    /// The only provider this channel can serve.
    #[must_use]
    pub fn provider(self) -> Provider {
        match self {
            Channel::AnthropicApi | Channel::ClaudeCodeCli => Provider::Anthropic,
            Channel::Antigravity | Channel::GeminiApi | Channel::Vertex => Provider::Google,
            Channel::AprServe => Provider::Local,
        }
    }
}

/// The governing-terms snapshot taken at capture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TermsRef {
    pub url: String,
    pub effective_date: String,
    pub fetched_at: String,
}

/// The four terms fields of one row, validated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tags {
    pub provider: Provider,
    pub model_id: String,
    pub channel: Channel,
    pub terms: TermsRef,
}

/// One reason a row's terms tags do not hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gap {
    /// The line is not a JSON object.
    NotJson,
    /// A required field is absent, null or not a string/object.
    Missing(&'static str),
    UnknownProvider(String),
    UnknownChannel(String),
    /// The channel cannot serve the stated provider.
    ChannelProvider {
        channel: Channel,
        provider: Provider,
    },
    /// A family alias or placeholder, not an exact model id.
    AliasModel(String),
    /// A local model is named by its weights sha256 and nothing else.
    LocalModelNotSha(String),
    TermsUrl(String),
    TermsEffectiveDate(String),
    TermsFetchedAt(String),
}

/// The validated tags of `row`, or every gap found.
///
/// # Errors
/// Returns the gaps when any tag is missing, unknown or inconsistent.
pub fn tags(row: &Value) -> Result<Tags, Vec<Gap>> {
    let mut gaps = Vec::new();
    let text = |key: &'static str, gaps: &mut Vec<Gap>| {
        let s = row.get(key).and_then(Value::as_str);
        if s.is_none() {
            gaps.push(Gap::Missing(key));
        }
        s
    };
    let provider = text("provider", &mut gaps).and_then(|s| {
        Provider::parse(s).or_else(|| {
            gaps.push(Gap::UnknownProvider(s.to_string()));
            None
        })
    });
    let channel = text("access_channel", &mut gaps).and_then(|s| {
        Channel::parse(s).or_else(|| {
            gaps.push(Gap::UnknownChannel(s.to_string()));
            None
        })
    });
    let model_id = text("model_id", &mut gaps);
    if let (Some(p), Some(c)) = (provider, channel) {
        if c.provider() != p {
            gaps.push(Gap::ChannelProvider {
                channel: c,
                provider: p,
            });
        }
    }
    if let Some(m) = model_id {
        if let Some(gap) = model_gap(m, provider) {
            gaps.push(gap);
        }
    }
    let terms = terms_ref(row.get("terms_ref"), &mut gaps);
    match (provider, model_id, channel, terms) {
        (Some(provider), Some(m), Some(channel), Some(terms)) if gaps.is_empty() => Ok(Tags {
            provider,
            model_id: m.to_string(),
            channel,
            terms,
        }),
        _ => Err(gaps),
    }
}

/// An exact id carries a version (a digit); a local id is a weights sha256.
fn model_gap(m: &str, provider: Option<Provider>) -> Option<Gap> {
    let placeholder = m.is_empty() || m.eq_ignore_ascii_case("unknown");
    if provider == Some(Provider::Local) && !placeholder {
        let hex = m.strip_prefix("sha256:").unwrap_or(m);
        let is_sha = hex.len() == 64 && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
        return (!is_sha).then(|| Gap::LocalModelNotSha(m.to_string()));
    }
    let versioned = m.bytes().any(|b| b.is_ascii_digit()) && !m.contains(char::is_whitespace);
    (placeholder || !versioned).then(|| Gap::AliasModel(m.to_string()))
}

fn terms_ref(v: Option<&Value>, gaps: &mut Vec<Gap>) -> Option<TermsRef> {
    let Some(obj) = v.and_then(Value::as_object) else {
        gaps.push(Gap::Missing("terms_ref"));
        return None;
    };
    let field = |key: &'static str, gaps: &mut Vec<Gap>| {
        let s = obj.get(key).and_then(Value::as_str).map(str::to_string);
        if s.is_none() {
            gaps.push(Gap::Missing(key));
        }
        s
    };
    let url = field("url", gaps);
    let effective_date = field("effective_date", gaps);
    let fetched_at = field("fetched_at", gaps);
    let before = gaps.len();
    if let Some(u) = &url {
        if !u.starts_with("https://") || u.len() == "https://".len() {
            gaps.push(Gap::TermsUrl(u.clone()));
        }
    }
    if let Some(d) = &effective_date {
        if !is_date(d) {
            gaps.push(Gap::TermsEffectiveDate(d.clone()));
        }
    }
    if let Some(t) = &fetched_at {
        if !is_utc_timestamp(t) {
            gaps.push(Gap::TermsFetchedAt(t.clone()));
        }
    }
    Some(TermsRef {
        url: url?,
        effective_date: effective_date?,
        fetched_at: fetched_at?,
    })
    .filter(|_| gaps.len() == before)
}

/// `YYYY-MM-DD` with a month 01-12 and a day 01-31.
fn is_date(s: &str) -> bool {
    let b = s.as_bytes();
    let digits = |r: std::ops::Range<usize>| {
        b[r.clone()]
            .iter()
            .all(u8::is_ascii_digit)
            .then(|| num(&b[r]))
    };
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return false;
    }
    matches!(
        (digits(0..4), digits(5..7), digits(8..10)),
        (Some(_), Some(1..=12), Some(1..=31))
    )
}

/// RFC 3339 in UTC: `YYYY-MM-DDTHH:MM:SS[.frac]Z`.
fn is_utc_timestamp(s: &str) -> bool {
    let Some(rest) = s.strip_suffix('Z') else {
        return false;
    };
    let (date, time) = match rest.split_once('T') {
        Some(p) => p,
        None => return false,
    };
    let (hms, frac) = time.split_once('.').unwrap_or((time, "0"));
    let t = hms.as_bytes();
    let two = |i: usize| {
        t[i..i + 2]
            .iter()
            .all(u8::is_ascii_digit)
            .then(|| num(&t[i..i + 2]))
    };
    is_date(date)
        && t.len() == 8
        && t[2] == b':'
        && t[5] == b':'
        && matches!(
            (two(0), two(3), two(6)),
            (Some(0..=23), Some(0..=59), Some(0..=60))
        )
        && !frac.is_empty()
        && frac.bytes().all(|c| c.is_ascii_digit())
}

fn num(digits: &[u8]) -> u32 {
    digits.iter().fold(0, |n, d| n * 10 + u32::from(d - b'0'))
}

/// What a pool rebuild leaves out.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Exclude {
    pub providers: Vec<Provider>,
    pub channels: Vec<Channel>,
}

impl Exclude {
    /// The public-release default (§2.11): gold-and-local only, so every hosted
    /// provider is excluded unless the operator overrides it at a §8 STOP.
    #[must_use]
    pub fn public_release() -> Self {
        Self {
            providers: vec![Provider::Anthropic, Provider::Google],
            channels: Vec::new(),
        }
    }

    /// Parse `--exclude-provider <p>` / `--exclude-channel <c>`, each repeatable.
    ///
    /// # Errors
    /// An unknown flag, a missing value, or an unknown provider or channel.
    pub fn from_args(args: &[&str]) -> Result<Self, String> {
        let mut ex = Self::default();
        let mut it = args.iter();
        while let Some(&flag) = it.next() {
            let value = it.next().ok_or_else(|| format!("{flag}: missing value"))?;
            match flag {
                "--exclude-provider" => ex.providers.push(
                    Provider::parse(value).ok_or_else(|| format!("unknown provider: {value}"))?,
                ),
                "--exclude-channel" => ex.channels.push(
                    Channel::parse(value).ok_or_else(|| format!("unknown channel: {value}"))?,
                ),
                other => return Err(format!("unknown argument: {other}")),
            }
        }
        Ok(ex)
    }

    fn excludes(&self, t: &Tags) -> bool {
        self.providers.contains(&t.provider) || self.channels.contains(&t.channel)
    }
}

/// A rebuilt pool: rows kept, rows left out by the filter, rows refused for gaps.
#[derive(Debug, Default)]
pub struct Rebuild<'a> {
    pub kept: Vec<&'a str>,
    pub excluded: Vec<&'a str>,
    pub refused: Vec<(&'a str, Vec<Gap>)>,
}

/// Rebuild a pool from a JSONL batch, leaving out every excluded source.
/// Rows are passed through byte-for-byte, in input order.
#[must_use]
pub fn rebuild<'a>(rows: &'a str, ex: &Exclude) -> Rebuild<'a> {
    let mut r = Rebuild::default();
    for line in rows.lines().filter(|l| !l.trim().is_empty()) {
        let checked = serde_json::from_str::<Value>(line)
            .ok()
            .filter(Value::is_object)
            .ok_or_else(|| vec![Gap::NotJson])
            .and_then(|v| tags(&v));
        match checked {
            Ok(t) if ex.excludes(&t) => r.excluded.push(line),
            Ok(_) => r.kept.push(line),
            Err(gaps) => r.refused.push((line, gaps)),
        }
    }
    r
}

#[cfg(test)]
#[path = "terms_tests.rs"]
mod tests;
