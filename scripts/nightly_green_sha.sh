#!/usr/bin/env bash
# Which main commit does the nightly build? (#7462, child 2 of #7017)
#
# The nightly used to gate on HEAD alone, so a red HEAD meant no build even when
# the commit just behind it was green. This picks the newest first-parent commit
# whose 'ci / gate' is success, walking at most 20 commits.
#
#   pick           stdin: "<sha> <state>" lines, newest first (line 1 = HEAD),
#                  state = success | failure | pending. Prints one line:
#                    - HEAD when HEAD is success or pending (HEAD unchanged);
#                    - else the first success within the 20-commit window;
#                    - red-ci when none of the 20 is green (caller keeps HEAD so
#                      the gate still reports red-ci).
#   live           reads the 20 first-parent commits of HEAD and their 'ci / gate'
#                  check runs through one GraphQL call (GH_TOKEN must be set),
#                  then prints the same answer as pick.
#   --self-test    runs scripts/tests/nightly_green_sha_test.sh.
set -euo pipefail

WALK=20
here=$(cd "$(dirname "$0")" && pwd)

pick() {
    local n=0 first=1 sha state
    while read -r sha state; do
        n=$((n + 1))
        if [ "$n" -gt "$WALK" ]; then
            break
        fi
        if [ "$first" = 1 ]; then
            first=0
            if [ "$state" = success ] || [ "$state" = pending ]; then
                printf '%s\n' "$sha"
                return 0
            fi
            continue
        fi
        if [ "$state" = success ]; then
            printf '%s\n' "$sha"
            return 0
        fi
    done
    printf 'red-ci\n'
}

live() {
    local tmp
    tmp=$(mktemp -d)
    trap 'rm -rf "${tmp:?}"' EXIT
    git rev-list --first-parent -n "$WALK" HEAD > "$tmp/shas"
    cat > "$tmp/q.graphql" << 'EOF'
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    ref(qualifiedName: "main") {
      target {
        ... on Commit {
          history(first: 100) {
            nodes {
              oid
              statusCheckRollup {
                contexts(first: 100) {
                  nodes { ... on CheckRun { name status conclusion } }
                }
              }
            }
          }
        }
      }
    }
  }
}
EOF
    gh api graphql -F owner=paiml -F name=aprender -F "query=@$tmp/q.graphql" \
        | jq -r '.data.repository.ref.target.history.nodes[]
            | . as $n
            | ([$n.statusCheckRollup.contexts.nodes[]? | select(.name == "ci / gate")][0]) as $c
            | [$n.oid,
               (if $c == null or $c.status != "COMPLETED" then "pending"
                elif $c.conclusion == "SUCCESS" then "success"
                else "failure" end)]
            | @tsv' | tr '\t' ' ' > "$tmp/states"
    # A commit the GraphQL page did not return is pending: never green, never red.
    awk 'NR == FNR { st[$1] = $2; next }
         { print $1, ($1 in st ? st[$1] : "pending") }' "$tmp/states" "$tmp/shas" \
        | pick
}

case "${1:-}" in
    pick) pick ;;
    live) live ;;
    --self-test) exec bash "$here/tests/nightly_green_sha_test.sh" ;;
    *)
        echo "usage: $0 pick | live | --self-test" >&2
        exit 2
        ;;
esac
