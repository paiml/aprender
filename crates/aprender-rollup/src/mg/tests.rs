use super::*;

fn run(status: &str, conclusion: Option<&str>) -> String {
    let c = conclusion.map_or("null".to_string(), |c| format!("\"{c}\""));
    format!("{{\"headSha\":\"H\",\"status\":\"{status}\",\"conclusion\":{c}}}")
}

#[test]
fn verdict_green_needs_every_run_success() {
    let j = format!(
        "[{},{}]",
        run("completed", Some("success")),
        run("completed", Some("success"))
    );
    assert_eq!(parse_verdict(&j, "H"), Ok(Verdict::Green));
}

#[test]
fn verdict_red_if_any_run_failed_even_with_others_pending() {
    let j = format!(
        "[{},{}]",
        run("in_progress", None),
        run("completed", Some("failure"))
    );
    assert_eq!(parse_verdict(&j, "H"), Ok(Verdict::Red));
}

#[test]
fn verdict_pending_when_unfinished_and_none_failed() {
    let j = format!(
        "[{},{}]",
        run("completed", Some("success")),
        run("queued", None)
    );
    assert_eq!(parse_verdict(&j, "H"), Ok(Verdict::Pending));
}

#[test]
fn no_runs_is_pending_not_green() {
    assert_eq!(parse_verdict("[]", "H"), Ok(Verdict::Pending));
}

#[test]
fn cancelled_and_timed_out_are_red() {
    for k in ["cancelled", "timed_out", "startup_failure"] {
        let j = format!("[{}]", run("completed", Some(k)));
        assert_eq!(parse_verdict(&j, "H"), Ok(Verdict::Red), "{k}");
    }
}

#[test]
fn malformed_read_is_not_measured() {
    assert!(parse_verdict("not json", "H").is_err());
    assert!(parse_verdict("{}", "H").is_err());
}

#[test]
fn bisect_all_green_lands_everything() {
    let r = bisect(&[1, 2, 3, 4], &mut |_| Verdict::Green);
    assert_eq!(
        r,
        Bisect {
            good: vec![1, 2, 3, 4],
            culprits: vec![],
            undecided: vec![]
        }
    );
}

#[test]
fn bisect_finds_the_single_culprit() {
    let r = bisect(&[1, 2, 3, 4, 5], &mut |s| {
        if s.contains(&4) {
            Verdict::Red
        } else {
            Verdict::Green
        }
    });
    assert_eq!(r.culprits, vec![4]);
    assert_eq!(r.good, vec![1, 2, 3, 5]);
    assert!(r.undecided.is_empty());
}

#[test]
fn bisect_single_red_entry_is_a_culprit() {
    let r = bisect(&[7], &mut |_| Verdict::Red);
    assert_eq!(r.culprits, vec![7]);
}

#[test]
fn bisect_pending_is_undecided_never_closed() {
    let r = bisect(&[1, 2], &mut |_| Verdict::Pending);
    assert_eq!(r.undecided, vec![1, 2]);
    assert!(r.good.is_empty() && r.culprits.is_empty());
}

#[test]
fn bisect_two_culprits_are_both_found() {
    let r = bisect(&[1, 2, 3, 4], &mut |s| {
        if s.contains(&1) || s.contains(&4) {
            Verdict::Red
        } else {
            Verdict::Green
        }
    });
    assert_eq!(r.culprits, vec![1, 4]);
    assert_eq!(r.good, vec![2, 3]);
}

#[test]
fn close_plan_closes_good_with_sha_and_comments_culprits() {
    let b = Bisect {
        good: vec![1, 2],
        culprits: vec![3],
        undecided: vec![4],
    };
    let p = close_plan(&b, Some("abc"));
    assert_eq!(p.len(), 4);
    assert!(p
        .iter()
        .any(|(n, a)| *n == 1 && a[1] == "close" && a[4].contains("abc")));
    assert!(p.iter().any(|(n, a)| *n == 3 && a[1] == "comment"));
    assert!(p.iter().any(|(n, a)| *n == 4 && a[1] == "comment"));
    assert!(!p
        .iter()
        .any(|(n, a)| (*n == 3 || *n == 4) && a[1] == "close"));
}

#[test]
fn close_plan_without_sha_on_main_closes_nothing() {
    let b = Bisect {
        good: vec![1, 2],
        culprits: vec![],
        undecided: vec![],
    };
    let p = close_plan(&b, None);
    assert!(p.iter().all(|(_, a)| a[1] == "comment"));
}

#[test]
fn run_list_args_filter_merge_group_on_the_branch() {
    let a = run_list_args("rollup/5-3");
    assert!(a.windows(2).any(|w| w == ["--event", "merge_group"]));
    assert!(a.windows(2).any(|w| w == ["--branch", "rollup/5-3"]));
}

#[test]
fn runs_of_another_head_are_ignored() {
    let stale_red = "{\"headSha\":\"OLD\",\"status\":\"completed\",\"conclusion\":\"failure\"}";
    let stale_green = "{\"headSha\":\"OLD\",\"status\":\"completed\",\"conclusion\":\"success\"}";
    assert_eq!(
        parse_verdict(&format!("[{stale_red}]"), "H"),
        Ok(Verdict::Pending)
    );
    assert_eq!(
        parse_verdict(&format!("[{stale_green}]"), "H"),
        Ok(Verdict::Pending)
    );
    let j = format!("[{stale_red},{}]", run("completed", Some("success")));
    assert_eq!(parse_verdict(&j, "H"), Ok(Verdict::Green));
}

#[test]
fn run_list_args_ask_for_head_sha() {
    assert!(run_list_args("b").iter().any(|a| a.contains("headSha")));
}

#[test]
fn ancestor_args_check_against_origin_main() {
    assert_eq!(
        ancestor_args("abc"),
        ["merge-base", "--is-ancestor", "abc", "origin/main"]
    );
}

#[test]
fn a_run_with_only_skipped_jobs_is_not_green() {
    let j = format!("[{}]", run("completed", Some("skipped")));
    assert_eq!(parse_verdict(&j, "H"), Ok(Verdict::Pending));
    let j = format!(
        "[{},{}]",
        run("completed", Some("skipped")),
        run("completed", Some("success"))
    );
    assert_eq!(parse_verdict(&j, "H"), Ok(Verdict::Green));
}
