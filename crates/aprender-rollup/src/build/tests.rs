use super::*;

#[test]
fn branch_name_uses_lowest_number_and_count() {
    assert_eq!(branch_name(&[7, 3, 9]), "rollup/3-3");
}

#[test]
fn message_names_every_constituent() {
    assert_eq!(message(&[3, 7]), "Rollup of 2 PRs: #3 #7");
}

#[test]
fn steps_branch_from_base_then_fetch_and_merge_each() {
    let s = steps("main", &[3, 7]);
    assert_eq!(s[0], ["fetch", "origin", "main"]);
    assert_eq!(s[1], ["checkout", "-B", "rollup/3-2", "origin/main"]);
    assert_eq!(s[2], ["fetch", "origin", "pull/3/head"]);
    assert_eq!(s[3], ["merge", "--no-ff", "-m", "Merge #3", "FETCH_HEAD"]);
    assert_eq!(s[4][2], "pull/7/head");
    assert_eq!(s.len(), 6);
}

#[test]
fn abort_is_merge_abort() {
    assert_eq!(abort_step(), ["merge", "--abort"]);
}

#[test]
fn head_check_step_reads_fetch_head() {
    assert_eq!(head_step(), ["rev-parse", "FETCH_HEAD"]);
}
