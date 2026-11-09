use super::*;

#[test]
fn close_posts_the_sha() {
    let a = gh_args(
        12,
        &Outcome::Close {
            sha: "abc123".to_string(),
        },
    );
    assert_eq!(a[..3], ["pr", "close", "12"]);
    assert!(a.last().is_some_and(|b| b.contains("abc123")));
}

#[test]
fn keep_open_never_closes_and_gives_reason() {
    let a = gh_args(
        12,
        &Outcome::KeepOpen {
            reason: "red".to_string(),
        },
    );
    assert_eq!(a[..3], ["pr", "comment", "12"]);
    assert!(!a.iter().any(|s| s == "close"));
    assert!(a.last().is_some_and(|b| b.contains("red")));
}
