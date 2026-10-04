# Runs for the scan rows S1-S8 of scripts/check_ci_unwedge.sh --self-test (jq -n -f).
def r($id; $wf; $br; $s; $ev; $h): {id: $id, workflow: $wf, head_branch: $br, status: $s,
    conclusion: (if $s == "completed" then "success" else null end), event: $ev, head_sha: $h,
    created_at: "2026-10-04T12:00:00Z"};
["queued", "pending", "waiting", "requested", "in_progress"] as $st
| [ range(0; 25) as $i | r(9000 - $i; "nightly"; "main"; "queued"; "schedule"; "n") ]
+ [ r(8000; "CI"; "fa"; "queued"; "pull_request"; "aaaa1") ]
+ [ range(0; 110) as $i | r(7999 - $i; "CI"; "old\($i)"; "completed"; "pull_request"; "o") ]
+ [ r(7000; "CI"; "fb"; "in_progress"; "pull_request"; "bbbb1"),
    r(6999; "CI"; "fc"; "queued"; "pull_request"; "cccc1"),
    r(6998; "CI"; "fd"; "queued"; "pull_request"; "dddd1"),
    r(6997; "CI"; "fe"; "queued"; "pull_request"; "eeee1"),
    r(6996; "CI"; "gh-readonly-queue/main/pr-1-0123456789abcdef0123456789abcdef01234567"; "queued"; "merge_group"; "mmmm1"),
    r(6995; "CI"; "fx"; "completed"; "pull_request"; "xxxx1") ]
+ [ range(0; 23) as $i | r(6000 - $i; "CI"; "f\($i)"; $st[$i % 5]; "pull_request"; "h\($i)") ]
