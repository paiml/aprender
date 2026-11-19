#!/bin/sh
# test-pr-spot-labels-guard.sh -- the guard must be green on the real tree and RED on each mutant.
# A guard never shown capable of failing is not evidence of anything: every case below was run
# against the guard and the mutant must turn it red (rc=1), a clean file green (rc=0).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
guard="$here/pr-spot-labels-guard.sh"
real="$root/.github/workflows/ci.yml"
command -v yq >/dev/null 2>&1 || { echo "test: yq not found" >&2; exit 2; }
tmp=$(mktemp -d)
[ -n "$tmp" ] && [ -d "$tmp" ] || exit 2
trap 'rm -rf "${tmp:?}"' EXIT
fail=0

expect() {
  name=$1 want=$2 d=$3
  rc=0
  sh "$guard" "$d" > "$tmp/out" 2>&1 || rc=$?
  if [ "$rc" -ne "$want" ]; then
    echo "FAIL $name: want rc=$want got rc=$rc"
    tail -n 3 "$tmp/out"
    fail=$((fail + 1))
  else
    echo "ok   $name"
  fi
}
put() { mkdir -p "$tmp/$1"; cat > "$tmp/$1/$2"; }
# mut <name> <sed-expr> : the real ci.yml with one mutation, which must change the file
mut() {
  mkdir -p "$tmp/$1"
  sed -e "$2" "$real" > "$tmp/$1/ci.yml"
  if cmp -s "$real" "$tmp/$1/ci.yml"; then echo "FAIL $1: the mutation did not apply"; fail=$((fail + 1)); return; fi
  expect "$1" 1 "$tmp/$1"
}
spotjson='["self-hosted","burst-x64","burst-public"]'

mkdir -p "$tmp/real"; cp "$real" "$tmp/real/ci.yml"
expect "real ci.yml alone is green" 0 "$tmp/real"
expect "the real workflow directory is green" 0 "$root/.github/workflows"
mkdir -p "$tmp/empty"; expect "empty directory is not a pass" 2 "$tmp/empty"
printf 'on: [push\n  broken: : :\n' | put unparsable a.yml; expect "unparsable file is red" 1 "$tmp/unparsable"
printf 'on: push\njobs:\n  t:\n    runs-on: [self-hosted, Linux, X64, clean-room]\n    steps: []\n' | put clean a.yml; expect "clean file is green" 0 "$tmp/clean"

# a gate job is only ever the three aggregators, on their one canonical expression
mut gate-fork-label "s/PR_SPOT_OPTIN == '1'/PR_SPOT_OPTIN != '1'/"
mut gate-always-spot "s/(github.event_name == 'pull_request' \&\& vars.PR_SPOT_OPTIN == '1')/true/"
mut gate-or-true "s/vars.PR_SPOT_OPTIN == '1')/vars.PR_SPOT_OPTIN == '1' || true)/"
mut gate-spot-on-merge-group "s/github.event_name == 'pull_request' \&\& vars.PR_SPOT_OPTIN/github.event_name != 'pull_request' \&\& vars.PR_SPOT_OPTIN/"
mut gate-extra-label "s/\"burst-x64\",\"burst-public\"/\"burst-x64\",\"burst-public\",\"gpu\"/"
mut gate-label-dropped "s/fromJSON('\[\"self-hosted\",\"Linux\",\"X64\",\"clean-room\"\]') }}/fromJSON('[\"self-hosted\",\"Linux\",\"clean-room\"]') }}/"
mut gate-other-optin-var "s/vars.PR_SPOT_OPTIN == '1'/vars.PR_SPOT_OPTIN2 == '1'/"
mut permissions-write "s/^      contents: read$/      contents: write/"
mut permissions-wider "s/^      contents: read$/      contents: read\n      id-token: write/"
mut permissions-write-all "s/^      contents: read$/XX/;s/^    permissions:$/    permissions: write-all/;/^XX\$/d"

# nothing else may name a spot label
mut other-job-spot "s/^    runs-on: \[self-hosted, Linux, X64, clean-room\]\$/    runs-on: [self-hosted, burst-x64]/"
mut reusable-spot-with "s/runs_on: '\[\"self-hosted\",\"Linux\",\"X64\",\"clean-room\"\]'/runs_on: '$spotjson'/"
mut reusable-spot-var "s/runs_on: '\[\"self-hosted\",\"Linux\",\"X64\",\"clean-room\"\]'/runs_on: \${{ vars.POOL }}/"
mut workflow-level-env-secret "s/^jobs:\$/env:\n  TOK: \${{ secrets.TOKEN }}\njobs:/"
mut workflow-level-write-all "s/^jobs:\$/permissions: write-all\njobs:/"
mut workflow-level-spot-env "s/^jobs:\$/env:\n  POOL: burst-x64\njobs:/"
mut secret-in-gate-env "s/^  gate:\$/  gate:\n    env:\n      TOK: \${{ secrets.TOKEN }}/"
mut id-token-in-gate "s/^  ci-gate:\$/  ci-gate:\n    env:\n      X: id-token/"
mut checkout-persist-spot-secret "s/^  workspace-test:\$/  workspace-test:\n    env:\n      K: \${{ secrets.K }}/"

# other files: every form of a spot runs-on, a signing or release job, an unmodelled expression
printf 'on: pull_request\njobs:\n  release:\n    runs-on: [self-hosted, burst-x64]\n    steps: []\n' | put new-literal a.yml; expect "new job on a spot literal" 1 "$tmp/new-literal"
printf 'on: pull_request\njobs:\n  sign:\n    permissions:\n      id-token: write\n    runs-on:\n      - self-hosted\n      - burst-x64\n    steps: []\n' | put new-block-list a.yaml; expect "block-list runs-on, .yaml, id-token" 1 "$tmp/new-block-list"
printf 'on: push\njobs:\n  t:\n    runs-on: {group: burst-public}\n    steps: []\n' | put new-group a.yml; expect "group form of runs-on" 1 "$tmp/new-group"
printf 'on: push\njobs:\n  t:\n    runs-on: {labels: [self-hosted, burst-x64]}\n    steps: []\n' | put new-labels a.yml; expect "labels form of runs-on" 1 "$tmp/new-labels"
printf 'on: push\njobs:\n  t:\n    runs-on: ${{ matrix.os }}\n    strategy:\n      matrix:\n        os: [burst-x64]\n    steps: []\n' | put matrix-literal a.yml; expect "matrix value is a spot label" 1 "$tmp/matrix-literal"
printf 'on: push\njobs:\n  t:\n    runs-on: ${{ fromJSON(matrix.labels) }}\n    strategy:\n      matrix:\n        labels: ["[\\"self-hosted\\",\\"burst-x64\\"]"]\n    steps: []\n' | put matrix-labels-json a.yml; expect "matrix.labels JSON names a spot label" 1 "$tmp/matrix-labels-json"
printf 'on: push\njobs:\n  t:\n    runs-on: ${{ fromJSON(matrix.labels) }}\n    strategy:\n      matrix:\n        labels: ${{ fromJSON(vars.LABELS) }}\n    steps: []\n' | put matrix-expr a.yml; expect "matrix fed from a variable cannot be judged" 1 "$tmp/matrix-expr"
printf 'on: push\njobs:\n  t:\n    runs-on: ${{ vars.POOL }}\n    steps: []\n' | put var-expr a.yml; expect "runs-on from a variable cannot be judged" 1 "$tmp/var-expr"
printf 'on: push\njobs:\n  t:\n    runs-on: ${{ inputs.label }}\n    steps: []\n' | put input-expr a.yml; expect "runs-on from an input cannot be judged" 1 "$tmp/input-expr"
printf 'x-pool: &p [self-hosted, burst-x64]\non: push\njobs:\n  t:\n    runs-on: *p\n    steps: []\n' | put anchor a.yml; expect "anchor carries a spot label" 1 "$tmp/anchor"
printf 'on: push\njobs:\n  t:\n    "runs-on": ["self-hosted", "\\x62urst-x64"]\n    steps: []\n' | put escape a.yml; expect "\\x escape spells burst" 1 "$tmp/escape"
printf 'on: push\njobs: {t: {runs-on: [self-hosted, burst-x64], steps: []}}\n' | put flow a.yml; expect "flow-style job" 1 "$tmp/flow"
printf 'base: &b\n  runs-on: [self-hosted, burst-x64]\n  steps: []\non: push\njobs:\n  t:\n    <<: *b\n' | put merge-key a.yml; expect "merge key inherits a spot runs-on" 1 "$tmp/merge-key"
printf 'on: push\njobs:\n  t:\n    uses: o/r/.github/workflows/x.yml@v1\n    with:\n      runs_on: %s\n' "'$spotjson'" | put reusable-literal a.yml; expect "reusable call passes a spot literal" 1 "$tmp/reusable-literal"
printf 'on: push\njobs:\n  t:\n    uses: o/r/.github/workflows/x.yml@v1\n    with:\n      runs_on: ${{ vars.POOL }}\n' | put reusable-expr a.yml; expect "reusable call passes a variable" 1 "$tmp/reusable-expr"
printf 'on: push\nenv:\n  P: burst-public\njobs:\n  t:\n    runs-on: [self-hosted]\n    steps: []\n' | put top-level a.yml; expect "workflow-level spot mention" 1 "$tmp/top-level"
printf 'on: push\njobs:\n  gate:\n    runs-on: [self-hosted, burst-x64]\n    steps: []\n' | put gate-in-other-file b.yml; expect "a job named gate in another file is not allowlisted" 1 "$tmp/gate-in-other-file"
printf 'on: push\njobs:\n  t:\n    runs-on: [self-hosted, Linux, X64, clean-room]\n    steps: []\n' | put crlf a.yml; printf 'on: push\r\njobs:\r\n  t:\r\n    runs-on: [self-hosted, burst-x64]\r\n    steps: []\r\n' > "$tmp/crlf/b.yml"; expect "CRLF file" 1 "$tmp/crlf"
mkdir -p "$tmp/hidden/sub"; printf 'on: push\njobs:\n  t:\n    runs-on: [self-hosted, burst-x64]\n    steps: []\n' > "$tmp/hidden/sub/.h.yml"; expect "hidden file in a subdirectory" 1 "$tmp/hidden"

if [ "$fail" -ne 0 ]; then echo "test-pr-spot-labels-guard: $fail case(s) FAILED"; exit 1; fi
echo "test-pr-spot-labels-guard: all cases ok"
