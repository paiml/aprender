# workflow_runs_nightly.awk — does a workflow re-check itself every night? Rule 3 of
# scripts/check_workflow_path_filters.sh (#3676), moved here from workflow_path_filters.py (T44, #4686).
#
# Exit 0 when the workflow's on: block has either
#   - a `schedule:` with at least one cron entry that has a value, or
#   - the chain: `workflow_run:` with `workflows: ["Nightly pick"]` (that one workflow) and `types:` naming
#     `completed`; the nightly producers start on the pick's run instead of their own cron (T44).
# Exit 1 otherwise: no on: block, `schedule: []`, a schedule with no cron, a chain from another workflow.
# Comment lines are ignored, so a commented-out trigger is no trigger. A cron is accepted in the forms the YAML
# reader of the old rule accepted: `- cron: X`, `- {cron: X}` and `schedule: [{cron: X}]`; an empty or quoted-empty
# value is no cron. A sequence item may sit at the key's own indent (`  schedule:` then `  - cron: X`).
#
#   awk -f scripts/lib/workflow_runs_nightly.awk WORKFLOW.yml
function ind(s) { match(s, /^ */); return RLENGTH }
function list1(s) {
    sub(/^[^:]*:[[:space:]]*/, "", s); sub(/[[:space:]]+$/, "", s)
    if (s !~ /^\[.*\]$/) return "\001"
    s = substr(s, 2, length(s) - 2); gsub(/^[[:space:]]+|[[:space:]]+$/, "", s)
    if (s ~ /^".*"$/ || s ~ /^\047.*\047$/) s = substr(s, 2, length(s) - 2)
    return s
}
function hascron(s) { return s ~ /cron:[[:space:]]*["\047]?[^[:space:],}"\047#]/ }
/^on:[[:space:]]*$/ { on = 1; next }
/^[[:space:]]*(#|$)/ { next }
on && ind($0) == 0 { on = 0 }
!on { next }
ind($0) == 2 && !/^  - / { k = $0; sub(/^ +/, "", k); sub(/:.*/, "", k); sc = (k == "schedule"); wr = (k == "workflow_run")
                if (sc && hascron($0)) cron = 1
                next }
sc && /^[[:space:]]+- [{]?[[:space:]]*cron:/ && hascron($0) { cron = 1 }
wr && ind($0) == 4 && /^ +workflows:/ && list1($0) == "Nightly pick" { w = 1 }
wr && ind($0) == 4 && /^ +types:[[:space:]]*\[/ && $0 ~ /[[,[:space:]]completed[],[:space:]]/ { t = 1 }
END { exit !(cron || (w && t)) }
