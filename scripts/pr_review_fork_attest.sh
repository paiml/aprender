#!/usr/bin/env bash
# pr_review_fork_attest.sh - the pr-review:attest path for FORK pull requests (#4462).
#
# WHY. Arm 4 (`present`, pr-review-quorum.yml) requires a signed receipt binding the
# PR's diff. A same-repo PR gets one from ci/sections.yml `pr-review-sign`, which holds
# PR_REVIEW_SIGNING_KEY_B64. A fork PR cannot: GitHub gives a fork's `pull_request` run
# no secrets, and the review itself cannot run server-side without executing fork code
# (the mutation arm) or holding reviewer credentials in CI. So every fork PR was
# unmergeable by construction.
#
# WHAT THIS DOES INSTEAD. A maintainer ATTESTS. Labelling a fork PR `pr-review:attest`
# runs .github/workflows/pr-review-fork-attest.yml (pull_request_target: the BASE
# definition, the BASE secret), which calls this script to:
#
#   authorize  check SERVER-SIDE that the labeler holds admin/maintain/write on the
#              base repository (GET /repos/{repo}/collaborators/{user}/permission),
#              is not the PR author, and that the head really is a fork.
#   build      write an in-toto receipt at attestation_level L2-maintainer-attest,
#              verdict DEGRADED, no consultations, for merge-base(origin/main, head)..head.
#              It READS the head (git objects only: merge-base, and the signer's pinned
#              `git diff --no-ext-diff --no-textconv | git patch-id`); it never checks
#              the head out and never runs anything from it.
#   publish    commit the signed receipt to the BASE-owned branch pr-review-fork-receipts
#              at evidence/pr-review/<pr>/<head>/ and push it (fast-forward only; never
#              force). Arm 4 reads L2 receipts from that branch and from nowhere else.
#
# WHAT THE ATTEST MEANS, SAID PLAINLY. "A person with write permission looked at this
# diff and vouched for it." It is not a review and the receipt says so (DEGRADED, and
# `present` prints "DEGRADED: maintainer-attest by <login>"). It binds ONE diff through
# the signed diff_patch_id the signer stamps, so any new push voids it and the label
# must be applied again.
#
# USAGE
#   pr_review_fork_attest.sh authorize         (env: ATTEST_REPO ATTEST_SENDER
#                                               ATTEST_PR_AUTHOR ATTEST_LABEL ATTEST_HEAD_REPO)
#   pr_review_fork_attest.sh build <dir>       (env: ATTEST_PR ATTEST_HEAD_SHA ATTEST_SENDER
#                                               ATTEST_PR_AUTHOR ATTEST_PERMISSION ATTEST_REPO
#                                               ATTEST_HEAD_REPO ATTEST_RUN_ID)
#   pr_review_fork_attest.sh publish <dir>     (env: ATTEST_PR ATTEST_HEAD_SHA)
#   pr_review_fork_attest.sh --self-test
#
# ENVIRONMENT (optional)
#   PR_REVIEW_GH        the gh binary (default: gh) - the self-test points it at a stub
#   PR_REVIEW_GIT_DIR   repository to read and publish from (default: this one)
#   ATTEST_REMOTE       remote to publish to (default: origin)
#   ATTEST_BRANCH       receipts branch (default: pr-review-fork-receipts)
#
# EXIT  0 done · 1 refused, with the reason named · 2 the box cannot answer
set -uo pipefail

PROG=${0##*/}
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$HERE/.." && pwd)
GD=${PR_REVIEW_GIT_DIR:-$REPO_ROOT}
LABEL=pr-review:attest
LEVEL=L2-maintainer-attest
SKILL_VERSION=2.1.0
PREDICATE_TYPE='https://paiml.dev/attestations/pr-review/v2'

die_env() { echo "$PROG: ENV - $*" >&2; exit 2; }
fail()    { echo "$PROG: FAIL - $*" >&2; exit 1; }
need()    { local v; for v in "$@"; do [ -n "${!v:-}" ] || fail "$v is unset or empty"; done; }

ST_ROOT=''
cleanup() {
    case "${ST_ROOT:-}" in
        */pr-attest-selftest.*) rm -rf -- "${ST_ROOT:?}" ;;
    esac
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
authorize() {
    need ATTEST_REPO ATTEST_SENDER ATTEST_PR_AUTHOR ATTEST_LABEL ATTEST_HEAD_REPO
    [ "$ATTEST_LABEL" = "$LABEL" ] || fail "label '$ATTEST_LABEL' is not $LABEL"
    [ "$ATTEST_HEAD_REPO" != "$ATTEST_REPO" ] \
        || fail "the head is $ATTEST_HEAD_REPO, the base repository itself; a same-repo PR is signed by pr-review-sign, the attest path is for forks only"
    # Case-insensitive: GitHub logins are, and `Alice` labelling alice's PR is still a self-label.
    [ "${ATTEST_SENDER,,}" != "${ATTEST_PR_AUTHOR,,}" ] \
        || fail "$ATTEST_SENDER labelled their own PR; a self-attest is not an attest (S5: reviewer != author)"
    local perm rc=0
    # SERVER-SIDE: the event payload says who clicked, the API says what they may do.
    # `.permission` folds maintain into write and triage into read.
    perm=$("${PR_REVIEW_GH:-gh}" api "repos/$ATTEST_REPO/collaborators/$ATTEST_SENDER/permission" --jq .permission 2>/dev/null) || rc=$?
    [ "$rc" -eq 0 ] || fail "the permission API for $ATTEST_SENDER on $ATTEST_REPO failed (rc $rc); an unanswered question is not a yes"
    case "$perm" in
        admin|maintain|write) ;;
        *) fail "$ATTEST_SENDER holds '${perm:-none}' on $ATTEST_REPO; only admin, maintain or write may attest (a triage role reads as 'read')" ;;
    esac
    echo "AUTHORIZED  $ATTEST_SENDER ($perm) attests a fork PR by $ATTEST_PR_AUTHOR from $ATTEST_HEAD_REPO"
    printf 'permission=%s\n' "$perm"
}

# ---------------------------------------------------------------------------
build() {
    local dir=$1 base sarif_sha
    need ATTEST_PR ATTEST_HEAD_SHA ATTEST_SENDER ATTEST_PR_AUTHOR ATTEST_PERMISSION ATTEST_REPO ATTEST_HEAD_REPO ATTEST_RUN_ID
    command -v jq >/dev/null 2>&1 || die_env "jq is not on PATH"
    case "$ATTEST_PR" in ''|*[!0-9]*) fail "ATTEST_PR '$ATTEST_PR' is not a PR number" ;; esac
    case "$ATTEST_HEAD_SHA" in *[!0-9a-f]*) fail "ATTEST_HEAD_SHA is not a sha" ;; esac
    [ "${#ATTEST_HEAD_SHA}" -eq 40 ] || fail "ATTEST_HEAD_SHA is not a full 40-hex sha"
    git -C "$GD" cat-file -e "${ATTEST_HEAD_SHA}^{commit}" 2>/dev/null \
        || fail "head $ATTEST_HEAD_SHA is not in $GD (fetch refs/pull/$ATTEST_PR/head first)"
    base=$(git -C "$GD" merge-base refs/remotes/origin/main "$ATTEST_HEAD_SHA" 2>/dev/null) \
        || fail "no merge-base(origin/main, $ATTEST_HEAD_SHA) in $GD"
    mkdir -p "$dir" || die_env "cannot create $dir"
    jq -cn '{"$schema":"https://json.schemastore.org/sarif-2.1.0.json","version":"2.1.0",
             "runs":[{"tool":{"driver":{"name":"pr-review-fork-attest",
                      "informationUri":"https://github.com/paiml/aprender/issues/4462"}},"results":[]}]}' \
        > "$dir/findings.sarif" || die_env "cannot write findings.sarif"
    sarif_sha=$(sha256sum < "$dir/findings.sarif" | cut -d' ' -f1)
    jq -cn --arg pt "$PREDICATE_TYPE" --arg sv "$SKILL_VERSION" --arg lv "$LEVEL" \
           --argjson pr "$ATTEST_PR" --arg base "$base" --arg head "$ATTEST_HEAD_SHA" \
           --arg author "$ATTEST_PR_AUTHOR" --arg sender "$ATTEST_SENDER" \
           --arg perm "$ATTEST_PERMISSION" --arg label "$LABEL" \
           --arg hrepo "$ATTEST_HEAD_REPO" --arg brepo "$ATTEST_REPO" --arg run "$ATTEST_RUN_ID" \
           --arg ssha "$sarif_sha" '
      {"_type":"https://in-toto.io/Statement/v1",
       "subject":[{"name":("git+https://github.com/" + $brepo),"digest":{"sha1":$head}}],
       "predicateType":$pt,
       "predicate":{
         "skill_version":$sv, "attestation_level":$lv, "pr":$pr,
         "base_sha":$base, "head_sha":$head,
         "author_actor":{"kind":"human","id":("github:" + $author)},
         "reviewer_actor":{"kind":"human","id":("github:" + $sender),
                           "note":"a maintainer attest, not a review: no consultation ran (#4462)"},
         "verdict":"DEGRADED",
         "verdict_note":("fork PR: attested by " + $sender + " (" + $perm + ") via the " + $label + " label; the pr-review skill did not run"),
         "attestation":{"attester":("github:" + $sender),"permission":$perm,"label":$label,
                        "head_repo":$hrepo,"base_repo":$brepo,"run_id":$run},
         "findings_ref":{"path":"findings.sarif","sha256":$ssha},
         "cost":{"input_tokens":0,"output_tokens":0,"wall_seconds":0}}}' \
        > "$dir/receipt.intoto.jsonl" || die_env "cannot write the receipt"
    echo "BUILT  $dir  (PR $ATTEST_PR, $base..$ATTEST_HEAD_SHA, attester $ATTEST_SENDER)"
}

# ---------------------------------------------------------------------------
# publish <dir> - one commit on the receipts branch, fast-forward push, never force.
# Plumbing only (a private index), so the checkout's working tree is never touched.
publish() {
    local dir=$1 remote=${ATTEST_REMOTE:-origin} branch=${ATTEST_BRANCH:-pr-review-fork-receipts}
    local tracking parent idx f blob tree commit attempt rc path pargs
    need ATTEST_PR ATTEST_HEAD_SHA
    [ -f "$dir/receipt.intoto.jsonl.minisig" ] || fail "$dir holds no signed receipt; publish runs after the signer"
    tracking="refs/remotes/$remote/$branch"
    path="evidence/pr-review/$ATTEST_PR/$ATTEST_HEAD_SHA"
    idx=$(mktemp "${TMPDIR:-/tmp}/pr-attest-idx.XXXXXX") || die_env "mktemp"
    for attempt in 1 2 3; do
        rc=0
        git -C "$GD" ls-remote --exit-code "$remote" "refs/heads/$branch" >/dev/null 2>&1 || rc=$?
        parent=''
        case "$rc" in
            0) git -C "$GD" fetch -q --no-tags "$remote" "+refs/heads/$branch:$tracking" \
                   || { rm -f -- "$idx"; die_env "cannot fetch $remote $branch"; }
               parent=$(git -C "$GD" rev-parse "$tracking") ;;
            2) ;;   # the branch does not exist yet: this is its root commit
            *) rm -f -- "$idx"; die_env "git ls-remote $remote failed (rc $rc)" ;;
        esac
        rm -f -- "$idx"
        if [ -n "$parent" ]; then GIT_INDEX_FILE=$idx git -C "$GD" read-tree "$parent"
        else GIT_INDEX_FILE=$idx git -C "$GD" read-tree --empty; fi || die_env "read-tree"
        for f in receipt.intoto.jsonl receipt.intoto.jsonl.minisig findings.sarif; do
            [ -f "$dir/$f" ] || { rm -f -- "$idx"; fail "$dir/$f is missing"; }
            blob=$(git -C "$GD" hash-object -w -- "$dir/$f") || die_env "hash-object"
            GIT_INDEX_FILE=$idx git -C "$GD" update-index --add --cacheinfo "100644,$blob,$path/$f" \
                || die_env "update-index"
        done
        tree=$(GIT_INDEX_FILE=$idx git -C "$GD" write-tree) || die_env "write-tree"
        pargs=(); [ -z "$parent" ] || pargs=(-p "$parent")
        commit=$(GIT_AUTHOR_NAME=pr-review-fork-attest GIT_AUTHOR_EMAIL=pr-review-fork-attest@users.noreply.github.com \
                 GIT_COMMITTER_NAME=pr-review-fork-attest GIT_COMMITTER_EMAIL=pr-review-fork-attest@users.noreply.github.com \
                 git -C "$GD" commit-tree "${pargs[@]}" \
                   -m "attest PR $ATTEST_PR head $ATTEST_HEAD_SHA (#4462)" "$tree") || die_env "commit-tree"
        if git -C "$GD" push -q "$remote" "$commit:refs/heads/$branch" 2>/dev/null; then
            rm -f -- "$idx"
            echo "PUBLISHED  $path on $branch ($commit)"
            return 0
        fi
        echo "  push of $branch was not a fast-forward (attempt $attempt); refetching" >&2
    done
    rm -f -- "$idx"
    fail "could not fast-forward $branch after 3 attempts; nothing was forced"
}

# ---------------------------------------------------------------------------
# --self-test: every refusal the cop's ruling names goes RED, end to end, through the
# REAL signer, guard and Arm 4 against the deterministic fixture repository.
self_test() {
    local fails=0 pass=0 fix="$REPO_ROOT/tests/fixtures/pr-review"
    command -v minisign >/dev/null 2>&1 || die_env "minisign is not on PATH"
    command -v check-jsonschema >/dev/null 2>&1 || die_env "check-jsonschema is not on PATH"
    ST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/pr-attest-selftest.XXXXXX") || die_env "mktemp"
    local repo="$ST_ROOT/repo" st=$ST_ROOT
    "$fix/make-fixture-repo.sh" "$repo" >/dev/null 2>&1 || die_env "could not build the fixture repository"
    mkdir -p "$repo/.github" "$repo/scripts/lib" "$repo/tests/fixtures/pr-review"
    cp "$fix/keys/pr-review-test.pub" "$repo/.github/pr-review.pub"
    cp "$REPO_ROOT/scripts/check_pr_review_arm4.sh" "$REPO_ROOT/scripts/check_pr_review_receipt.sh" \
       "$REPO_ROOT/scripts/pr_review_sign_receipt.sh" "$REPO_ROOT/scripts/$PROG" "$repo/scripts/"
    cp "$REPO_ROOT/scripts/lib/pr_review_patch_id.sh" "$REPO_ROOT/scripts/lib/git_patch_id.py" "$repo/scripts/lib/"
    cp -a "$REPO_ROOT/schemas" "$repo/schemas"
    cp -a "$fix/positive-control" "$repo/tests/fixtures/pr-review/positive-control"

    local head b64 pr=5001
    head=$(git -C "$repo" rev-parse docs-pr) || die_env "the fixture repo has no docs-pr head"
    b64=$(base64 -w0 < "$fix/keys/pr-review-test-TEST-ONLY.key")

    # gh stub: the permission API, one answer per login; `broken` makes the call fail.
    mkdir -p "$st/bin"
    cat > "$st/bin/gh" <<'STUB'
#!/usr/bin/env bash
case "$2" in
  */collaborators/maint/permission)   echo write ;;
  */collaborators/boss/permission)    echo admin ;;
  */collaborators/author/permission)  echo write ;;
  */collaborators/triager/permission) echo read ;;
  */collaborators/broken/permission)  exit 1 ;;
  *)                                  echo none ;;
esac
STUB
    chmod +x "$st/bin/gh"

    ok()  { printf 'ok    %s\n' "$1"; pass=$((pass + 1)); }
    bad() { printf 'FAIL  %s\n' "$1"; [ -z "${2:-}" ] || printf '      %s\n' "$2"; fails=$((fails + 1)); }
    # expect <label> <want-rc> <reason-substring> <cmd...> - a refusal must be the NAMED
    # one: a row that goes RED on an earlier branch is not evidence for the branch it names.
    expect() {
        local label=$1 want=$2 why=$3 rc=0 out; shift 3
        out=$("$@" 2>&1) || rc=$?
        if [ "$rc" = "$want" ] && grep -qF -- "$why" <<<"$out"; then ok "$label"; else bad "$label (rc=$rc, want $want)" "$(printf '%s' "$out" | tail -2 | tr '\n' ' ')"; fi
    }
    auth() { # <sender> <author> [label] [head_repo]
        env PR_REVIEW_GH="$st/bin/gh" ATTEST_REPO=paiml/aprender ATTEST_SENDER="$1" ATTEST_PR_AUTHOR="$2" \
            ATTEST_LABEL="${3:-$LABEL}" ATTEST_HEAD_REPO="${4:-someone/aprender}" bash "$repo/scripts/$PROG" authorize
    }
    echo "== $PROG --self-test =="
    expect 'authorize: a write labeler on a fork PR'                     0 AUTHORIZED auth maint someone
    expect 'authorize: an admin labeler'                                 0 AUTHORIZED auth boss someone
    expect 'authorize: the AUTHOR labelling their own PR is refused'     1 "labelled their own PR" auth author author
    expect 'authorize: a self-label differing only in case is refused'   1 "labelled their own PR" auth Author author
    expect 'authorize: a TRIAGE-only labeler (reads as read) is refused' 1 "holds 'read'" auth triager someone
    expect 'authorize: a non-collaborator is refused'                    1 "holds 'none'" auth stranger someone
    expect 'authorize: a failing permission API is refused, not a yes'   1 "permission API" auth broken someone
    expect 'authorize: another label is refused'                         1 "is not pr-review:attest" auth maint someone other-label
    expect 'authorize: a same-repo head is refused'                      1 "forks only" auth maint someone "$LABEL" paiml/aprender

    # build + sign (real signer, TEST-ONLY key) + guard.
    mk() { # <dir> [jq mutation]
        local d=$1
        rm -rf -- "${d:?}"
        ( cd "$repo" && env PR_REVIEW_GIT_DIR="$repo" ATTEST_PR=$pr ATTEST_HEAD_SHA="$head" ATTEST_SENDER=maint \
            ATTEST_PR_AUTHOR=someone ATTEST_PERMISSION=write ATTEST_REPO=paiml/aprender \
            ATTEST_HEAD_REPO=someone/aprender ATTEST_RUN_ID=12345 bash scripts/"$PROG" build "$d" ) >/dev/null 2>&1 \
            || return 1
        if [ -n "${2:-}" ]; then
            jq -c "$2" "$d/receipt.intoto.jsonl" > "$d/r.tmp" && mv "$d/r.tmp" "$d/receipt.intoto.jsonl" || return 1
        fi
        ( cd "$repo" && env PR_REVIEW_ALLOW_ATTEST=1 PR_REVIEW_SIGNING_KEY_B64="$b64" PR_REVIEW_GIT_DIR="$repo" \
            bash scripts/pr_review_sign_receipt.sh "$d" ) >/dev/null 2>&1
    }
    guard() { ( cd "$repo" && bash scripts/check_pr_review_receipt.sh "$1" ); }
    local good="$st/attest/evidence/pr-review/$pr/$head"
    if mk "$good"; then ok 'build + sign: the attest receipt is built and signed'; else bad 'build + sign: the attest receipt is built and signed'; fi
    expect 'guard: a well-formed attest is ACCEPTED'                     0 ACCEPT guard "$good"
    local row name why mut
    # <name>|<the reason the guard must NAME>|<jq mutation of the good receipt>
    while IFS='|' read -r name why mut; do
        row="$st/mut/$name"
        mk "$row" "$mut" || { bad "mutant $name could not be built"; continue; }
        expect "guard: mutant $name is REJECTED" 1 "$why" guard "$row"
    done <<'ROWS'
self-label (reviewer = author)|a self-review is not a review|.predicate.reviewer_actor.id = .predicate.author_actor.id | .predicate.attestation.attester = .predicate.author_actor.id
triage permission|attestation.permission "triage"|.predicate.attestation.permission = "triage"
empty permission|attestation.permission ""|.predicate.attestation.permission = ""
a substring of write|attestation.permission "rite"|.predicate.attestation.permission = "rite"
verdict PASS|DEGRADED and never|.predicate.verdict = "PASS"
a consultation claimed|carries consultations|.predicate.consultations = {"pmat":{"status":"consulted"}}
attester is not the reviewer|attester is not reviewer_actor.id|.predicate.attestation.attester = "github:boss"
another label|attestation.label is not|.predicate.attestation.label = "lgtm"
same-repo head|forks only|.predicate.attestation.head_repo = .predicate.attestation.base_repo
no attestation block|predicate.attestation is absent|del(.predicate.attestation)
ROWS

    # Arm 4, end to end: where the receipt sits and what it binds.
    arm4() { # <evidence-root> <attest-root|''> <head>
        ( cd "$repo" && env -u GITHUB_EVENT_NAME PR_NUMBER=$pr PR_HEAD_SHA="$3" \
            PR_REVIEW_EVIDENCE_ROOT="$1" PR_REVIEW_ATTEST_ROOT="$2" bash scripts/check_pr_review_arm4.sh )
    }
    local out rc=0
    out=$(arm4 "$st/empty" "$st/attest/evidence/pr-review" "$head" 2>&1) || rc=$?
    if [ "$rc" -eq 0 ] && grep -q '^  DEGRADED: maintainer-attest by maint$' <<<"$out"; then
        ok 'arm4: the attest on the base-owned branch passes and prints DEGRADED: maintainer-attest by maint'
    else bad 'arm4: the attest on the base-owned branch passes and prints DEGRADED: maintainer-attest by maint' "rc=$rc $(tail -2 <<<"$out" | tr '\n' ' ')"; fi
    # (4) the same signed receipt, but in the PR head's own tree (a non-base branch)
    expect 'arm4: the attest in the head tree (non-base branch) is RED' 1 "pr-review-fork-receipts branch" arm4 "$st/attest/evidence/pr-review" '' "$head"
    # (3) a new push after the attest: the diff moved, the signed patch-id is stale
    local pushed idx="$st/idx" tree blob
    blob=$(printf 'a later push\n' | git -C "$repo" hash-object -w --stdin)
    GIT_INDEX_FILE=$idx git -C "$repo" read-tree "$head"
    GIT_INDEX_FILE=$idx git -C "$repo" update-index --add --cacheinfo "100644,$blob,docs/later.md"
    tree=$(GIT_INDEX_FILE=$idx git -C "$repo" write-tree)
    pushed=$(GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@fixture.invalid GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@fixture.invalid \
        git -C "$repo" commit-tree -p "$head" -m "a push after the attest" "$tree")
    expect 'arm4: a push after the attest (stale patch-id) is RED'      1 "bind a DIFFERENT diff" arm4 "$st/empty" "$st/attest/evidence/pr-review" "$pushed"
    # an L1 receipt planted in the attest root is not read from there
    mkdir -p "$st/l1root/$pr"
    cp -a "$good" "$st/l1root/$pr/"
    jq -c '.predicate.attestation_level = "L1-self"' "$good/receipt.intoto.jsonl" > "$st/l1root/$pr/$head/receipt.intoto.jsonl"
    expect 'arm4: only L2 receipts are read from the attest root'       1 "holds no receipt whose" arm4 "$st/empty" "$st/l1root" "$head"
    # an attest signed for ANOTHER PR, copied under this PR's directory (same diff, so same patch-id)
    local xpr="$st/xpr/$pr/$head"
    mk "$xpr" '.predicate.pr = 5002' || bad 'arm4: the cross-PR attest could not be built'
    expect 'arm4: an attest signed for another PR is RED'             1 "signed for PR 5002, not PR $pr" arm4 "$st/empty" "$st/xpr" "$head"

    # publish: a bare remote, two attests, fast-forward only.
    git init -q --bare "$st/remote.git"
    git -C "$repo" remote add attest-test "$st/remote.git"
    pub() { ( cd "$repo" && env ATTEST_REMOTE=attest-test PR_REVIEW_GIT_DIR="$repo" ATTEST_PR="$1" ATTEST_HEAD_SHA="$2" \
                bash scripts/"$PROG" publish "$3" ); }
    expect 'publish: the first attest creates the branch'               0 PUBLISHED pub $pr "$head" "$good"
    expect 'publish: a second attest fast-forwards onto it'             0 PUBLISHED pub 5002 "$head" "$good"
    local n
    n=$(git -C "$st/remote.git" ls-tree -r --name-only pr-review-fork-receipts 2>/dev/null | grep -c '/receipt.intoto.jsonl.minisig$')
    if [ "$n" = 2 ] && [ "$(git -C "$st/remote.git" rev-list --count pr-review-fork-receipts)" = 2 ]; then
        ok 'publish: both receipts are on the branch, in two linear commits'
    else bad 'publish: both receipts are on the branch, in two linear commits' "n=$n"; fi
    mkdir -p "$st/unsigned"
    expect 'publish: an unsigned directory is refused'                  1 "holds no signed receipt" pub $pr "$head" "$st/unsigned"

    printf '%s --self-test: %d passed, %d failed\n' "$PROG" "$pass" "$fails"
    [ "$fails" -eq 0 ]
}

case "${1:-}" in
    --self-test) self_test; exit $? ;;
    authorize)   authorize ;;
    build)       [ -n "${2:-}" ] || fail "usage: $PROG build <dir>"; build "$2" ;;
    publish)     [ -n "${2:-}" ] || fail "usage: $PROG publish <dir>"; publish "$2" ;;
    -h|--help)   sed -n '2,48p' "$0" ;;
    *)           fail "usage: $PROG authorize | build <dir> | publish <dir> | --self-test" ;;
esac
