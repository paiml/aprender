# Open PRs for the scan rows S1-S8 of scripts/check_ci_unwedge.sh --self-test (jq -n -f): 31.
[ {headRefName: "fa", headRefOid: "aaaa2"}, {headRefName: "fb", headRefOid: "bbbb2"},
  {headRefName: "fc", headRefOid: "cccc1"}, {headRefName: "fx", headRefOid: "xxxx2"} ]
+ [ range(0; 23) as $i | {headRefName: "f\($i)", headRefOid: "h\($i)"} ]
+ [ range(0; 4) as $i | {headRefName: "u\($i)", headRefOid: "u\($i)"} ]
