#!/usr/bin/env bash
# assets_rehearsal.sh — the nightly rehearsal of the release-asset build (RR-P03/P04, #4806).
#
#   bash scripts/release/assets_rehearsal.sh gen [OUT]       # write the workflow (default: the committed path)
#   bash scripts/release/assets_rehearsal.sh --check [FILE]  # FILE is what gen writes now (default: committed)
#   bash scripts/release/assets_rehearsal.sh lint FILE       # FILE cannot upload
#   bash scripts/release/assets_rehearsal.sh tag             # tag=v<workspace version>-rc.0
#   bash scripts/release/assets_rehearsal.sh verify TAG DIR  # verdict over the built assets (env NEEDS)
#   bash scripts/release/assets_rehearsal.sh ticket CONCLUSION SHA URL  # open/update the one ticket (gh)
#   bash scripts/release/assets_rehearsal.sh --self-test     # the case table
#
# binary-release.yml runs only on `release: published` and on dispatch, so the release's
# asset build had no nightly and could not show three green nights. The rehearsal is that
# nightly: the release's own build jobs, run on a schedule, never uploaded.
#
# The workflow is GENERATED from binary-release.yml (`gen`), never copied by hand: the build
# jobs (pv, all [[bin]]s, apr cuda, apr cpu, apr darwin) keep their own steps, runners and
# checks. Only these differ: the triggers (schedule + dispatch); the tag (below); the
# permissions (contents: read, no `secrets` or `vars`, no environment); each "Upload assets to
# release" step, which becomes a step that checks every archive against its .sha256 and
# keeps the checksums as a run artifact; the release-reading jobs, replaced by `verify`; and
# the credential guard at the start of every job. The rehearsal's first job regenerates and
# compares as canonical JSON (--check), so a binary-release.yml edit without a regen turns
# the rehearsal red instead of letting it rehearse a stale build.
#
# It cannot upload, because nothing in it holds a credential that can write. `lint`
# rests that on two closed properties, not on spotting upload commands:
#   - the token cannot write: workflow permissions exactly contents: read, no job-level
#     permissions, no environment (environment secrets), no reusable-workflow call;
#   - no credential is reachable from an expression: the words `secrets` and `vars`
#     appear nowhere in the file or in its parsed form, in any case or access form
#     (secrets.X, secrets['X'], toJSON(secrets)), and `github` appears only as
#     ${{ github.job }}, ${{ github.sha }} or a github.com URL (so no github.token,
#     github['token'], github.*, toJSON(github) or format() form). A context is an
#     identifier; an expression cannot build its name from a string, so there is no
#     other spelling. The parsed form puts YAML escapes and multi-line expressions in
#     scope.
# A step that POSTs in any form then has nothing to write with. The rest is the
# envelope: triggers exactly schedule + dispatch, every job starting with the
# credential guard, no step `if:` at all (it can run past the guard), workflow and job
# keys and job env names from an allowlist (no workflow env, defaults, container or
# services), actions from an allowlist, and every build job keeping its checksums as a
# run artifact. The upload pattern (gh release, uploads.github.com, -X POST, ...) is a
# tripwire, not the guarantee. The guard covers what lint cannot see, a credential on
# the runner (a flag is not a guard): it refuses to run when GH_TOKEN, GITHUB_TOKEN, a
# cargo registry token, an OIDC request token or a cargo credentials file is present.
# The case table runs lint and the guard over a fixture workflow built from the real
# step bodies.
#
# The verdict: `tag` names the rehearsal's tag, v<workspace version>-rc.0, so
# the rc stamp runs as it does on a real rc. `verify` judges one night:
#   - a build job that failed               -> red;
#   - a build job that did not succeed for any other reason (skipped, cancelled, absent,
#     unreadable needs)                     -> not_measured, never green;
#   - otherwise check_release_assets.sh --assets-from over the names the builds kept:
#     0 green, 1 red, anything else not_measured.
#
# Exit: 0 green / ok, 1 red / not_measured / a failed row, 2 usage or a missing tool.
set -uo pipefail
PROG=assets_rehearsal
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT/.github/workflows/binary-release.yml"
OUT_DEFAULT="$ROOT/.github/workflows/assets-rehearsal-nightly.yml"
YQ="${YQ:-yq}"
GUARD_NAME="Refuse to run where an upload credential exists (the rehearsal cannot upload)"
BUILD_JOBS="build build-all-bins build-apr-cuda build-apr-cpu build-apr-darwin"

die() { printf '%s: %s\n' "$PROG" "$*" >&2; exit 2; }
need_yq() { command -v "$YQ" > /dev/null 2>&1 || die "yq not found (set YQ=<path>)"; }

# The guard step. It runs before the checkout, so it is inline, and it must run under the
# macOS runner's bash 3.2 as well.
guard_step() {
    cat <<'EOF'
name: Refuse to run where an upload credential exists (the rehearsal cannot upload)
shell: bash
run: |
  bad=0
  for v in GH_TOKEN GITHUB_TOKEN CARGO_REGISTRY_TOKEN CARGO_REGISTRIES_CRATES_IO_TOKEN ACTIONS_ID_TOKEN_REQUEST_TOKEN; do
    if [ -n "$(printenv "$v" 2>/dev/null)" ]; then
      echo "::error::refusing to rehearse: $v is set in this job's environment"; bad=1
    fi
  done
  # cargo reads an empty CARGO_HOME as unset, so the guard does too.
  ch=$(printenv CARGO_HOME 2>/dev/null) || true; [ -n "$ch" ] || ch="$HOME/.cargo"
  for f in "$ch/credentials" "$ch/credentials.toml"; do
    if [ -s "$f" ]; then echo "::error::refusing to rehearse: a cargo registry credential file exists: $f"; bad=1; fi
  done
  if [ "$bad" -ne 0 ]; then exit 1; fi
  echo "no upload credential here: no GH_TOKEN, GITHUB_TOKEN, registry token, OIDC request token or cargo credential file"
EOF
}

keep_step() {
    cat <<'EOF'
name: Check every archive against its .sha256 and keep the checksums (the rehearsal never uploads)
shell: bash
run: |
  set -euo pipefail
  KEEP="$RUNNER_TEMP/rehearsal-keep"
  rm -rf "${KEEP:?}"; mkdir -p "$KEEP"; n=0
  for s in *.tar.gz.sha256 dist/*.tar.gz.sha256; do
    [ -f "$s" ] || continue
    (cd "$(dirname "$s")" && shasum -a 256 -c "$(basename "$s")")
    cp "$s" "$KEEP/"; n=$((n + 1))
  done
  if [ "$n" -eq 0 ]; then echo "::error::the package step left no archive"; exit 1; fi
  echo "kept $n checksums" | tee -a "$GITHUB_STEP_SUMMARY"
EOF
}

artifact_step() {
    cat <<'EOF'
name: Keep the checksums as a run artifact
uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a  # v7.0.1
with:
  name: rehearsal-${{ github.job }}-${{ strategy.job-index }}
  path: ${{ runner.temp }}/rehearsal-keep/
  if-no-files-found: error
  retention-days: 5
EOF
}

# lint_q ARGS...: one yq query over lint's exploded document $x (read by dynamic scope).
# Its status is yq's, so a caller can tell a query that failed from one that found nothing.
lint_q() { "$YQ" "$@" - <<< "$x"; }

# lint_unmeasured NAME: a query yq fails on is red. Its empty answer is never
# read as "nothing found": a job `env:` written as a string makes `keys` error, and an
# empty list of disallowed names is what every allowlist check reads as a pass.
lint_unmeasured() { echo "lint: yq failed on the $1 check (a check that cannot be measured is red)"; bad=1; }

# lint FILE: every property that keeps the rehearsal from uploading. One line per failure.
lint() {
    local f=$1 bad=0 v g t x
    need_yq
    [ -f "$f" ] || die "lint: no such file $f"
    # Every structural query reads the parsed document with aliases expanded, so a step,
    # job or key reached through an anchor (`- *a`, `<<: *a`) is checked as Actions sees it.
    # Every yq call fails closed: a parse error, a failed query (lint_unmeasured) and a
    # `-e` query that errors or matches nothing are each red.
    x=$("$YQ" --yaml-fix-merge-anchor-to-spec=true 'explode(.)' "$f" 2>/dev/null) || { echo "lint: yq cannot parse $f"; bad=1; }
    v=$(lint_q -o=json -I=0 '.permissions') || lint_unmeasured "workflow permissions"
    [ "$v" = '{"contents":"read"}' ] || { echo "lint: workflow permissions are $v, not exactly contents: read"; bad=1; }
    v=$(lint_q '[.jobs[] | select(has("permissions"))] | length') || lint_unmeasured "job permissions"
    [ "$v" = 0 ] || { echo "lint: $v job(s) set their own permissions"; bad=1; }
    v=$(lint_q '[.jobs[] | select(has("environment"))] | length') || lint_unmeasured "job environment"
    [ "$v" = 0 ] || { echo "lint: $v job(s) name an environment (environment secrets)"; bad=1; }
    v=$(lint_q '.on | keys | sort | join(",")') || lint_unmeasured "trigger"
    [ "$v" = "schedule,workflow_dispatch" ] || { echo "lint: triggers are '$v', not schedule,workflow_dispatch"; bad=1; }
    # The guard is matched on its name AND its body, so a step that only borrows the name
    # (a hollow guard) does not count. Its keys must be exactly the guard's keys: an `if:` can
    # skip it, `continue-on-error:` lets the job run on past its refusal, and a `shell:` or
    # `env:` changes what its body does, so any extra key is a guard that may not hold.
    g=$(mktemp) || die "mktemp failed"
    guard_step > "$g"
    v=$(G=$g lint_q '[.jobs[] | select((.steps[0] // {} | keys | sort | join(",")) != (load(strenv(G)) | keys | sort | join(",")) or (.steps[0].name // "") != load(strenv(G)).name or (.steps[0].shell // "") != load(strenv(G)).shell or (.steps[0].run // "") != load(strenv(G)).run)] | length') || lint_unmeasured "credential guard"
    rm -f "${g:?}"
    [ "$v" = 0 ] || { echo "lint: $v job(s) do not start with the credential guard"; bad=1; }
    # No step carries an `if:` at all. A step `if:` loses its implicit success() as soon as
    # the expression mentions any status function, so `success() || true`, `!success()`,
    # always() or failure() all run on past the guard's refusal, and a pattern over the
    # expression cannot list every form. The release's build steps use none.
    v=$(lint_q '[.jobs[].steps[] | select(has("if"))] | length') || lint_unmeasured "step if"
    [ "$v" = 0 ] || { echo "lint: $v step(s) carry an if: (a step if: can run past the credential guard)"; bad=1; }
    # What acts before or around the guard is allowlisted: the workflow's and every job's keys,
    # and job env names. A workflow or job env (BASH_ENV runs before the guard's bash),
    # defaults, container or services is outside the list.
    v=$(lint_q '[keys[] | select(test("^(name|on|permissions|concurrency|jobs)$") | not)] | join(" ")') || lint_unmeasured "workflow key"
    [ -z "$v" ] || { echo "lint: workflow key(s) outside the allowlist (they act before the credential guard): $v"; bad=1; }
    v=$(lint_q '[.jobs[] | keys[] | select(test("^(name|runs-on|timeout-minutes|outputs|steps|needs|strategy|env|if)$") | not)] | unique | join(" ")') || lint_unmeasured "job key"
    [ -z "$v" ] || { echo "lint: job key(s) outside the allowlist (they act before the credential guard): $v"; bad=1; }
    v=$(lint_q '[.jobs[] | (.env // {}) | keys[] | select(test("^(MACOSX_DEPLOYMENT_TARGET)$") | not)] | unique | join(" ")') || lint_unmeasured "job env"
    [ -z "$v" ] || { echo "lint: job env name(s) outside the allowlist (they act before the credential guard): $v"; bad=1; }
    # No credential is reachable from an expression. The text checked is the file plus
    # the parsed document (aliases expanded, one-line JSON), so a `$` or a letter written
    # as a YAML double-quoted escape, and an expression split across lines, are both in
    # it. `secrets` and `vars` are banned there as words, in any case, so every access
    # form is caught.
    # `github` is banned the same way outside three exact forms, `${{ github.job }}`,
    # `${{ github.sha }}` and an https://github.com/ URL, each blanked to a space first:
    # github.token, github['token'], github.*, toJSON(github) and format() all leave a
    # `github` word behind. No expression is parsed, so there is no parse to get wrong.
    t=$(cat "$f" && "$YQ" --yaml-fix-merge-anchor-to-spec=true -o=json -I=0 'explode(.)' "$f") || { echo "lint: yq cannot parse $f"; bad=1; }
    if printf '%s\n' "$t" | grep -n -i -E '(^|[^A-Za-z0-9_])(secrets|vars)([^A-Za-z0-9_]|$)'; then
        echo "lint: a secrets or vars reference (a credential reachable from an expression)"; bad=1
    fi
    if printf '%s\n' "$t" | sed -e 's/\${{ github\.job }}/ /g' -e 's/\${{ github\.sha }}/ /g' -e 's#https://github\.com/# #g' | grep -n -i -E '(^|[^A-Za-z0-9_])github([^A-Za-z0-9_]|$)'; then
        echo "lint: the job token reachable from an expression (a github reference outside \${{ github.job }}, \${{ github.sha }} and a github.com URL)"; bad=1
    fi
    # The upload tripwire reads the same decoded text, so `run: "\x63argo publish"` is caught.
    if printf '%s\n' "$t" | grep -n -E 'uploads\.github\.com|-X ?.?(POST|PUT|PATCH|DELETE)|--request[ =].?(POST|PUT|PATCH|DELETE)|gh (release|api)|cargo publish|github\.event\.release|inputs\.tag'; then
        echo "lint: an upload, a release write or a release-event input"; bad=1
    fi
    # Actions are allowlisted, not denylisted: a release-upload action under any name is
    # outside the list. These three are all the release workflow and the rehearsal use.
    v=$(lint_q '[.jobs[].steps[] | select(has("uses")) | .uses | sub("@.*"; "") | select(test("^actions/(checkout|upload-artifact|download-artifact)$") | not)] | unique | join(" ")') || lint_unmeasured "action allowlist"
    [ -z "$v" ] || { echo "lint: an action outside the allowlist (an upload path): $v"; bad=1; }
    for v in $BUILD_JOBS; do
        "$YQ" -e ".jobs[\"$v\"].steps[] | select(.uses // \"\" | test(\"^actions/upload-artifact@\"))" - <<< "$x" > /dev/null 2>&1 \
            || { echo "lint: build job $v keeps no checksums"; bad=1; }
    done
    [ "$bad" -eq 0 ] && echo "ok: ${f#"$ROOT"/} cannot upload"
    return "$bad"
}

assets_job() {
    cat <<'EOF'
name: Rehearsal tag, and the rehearsal is still the release's build
runs-on: [self-hosted, Linux, X64, clean-room]
timeout-minutes: 10
outputs:
  tag: ${{ steps.tag.outputs.tag }}
steps:
  - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1
  - name: The workflow is binary-release.yml's build, regenerated, and cannot upload
    run: |
      set -euo pipefail
      curl -sSfL -o "$RUNNER_TEMP/yq" https://github.com/mikefarah/yq/releases/download/v4.52.2/yq_linux_amd64
      echo "a74bd266990339e0c48a2103534aef692abf99f19390d12c2b0ce6830385c459  $RUNNER_TEMP/yq" | sha256sum -c -
      chmod +x "$RUNNER_TEMP/yq"
      export YQ="$RUNNER_TEMP/yq"
      bash scripts/release/assets_rehearsal.sh --self-test
      bash scripts/release/assets_rehearsal.sh --check
  - name: Rehearsal tag
    id: tag
    run: bash scripts/release/assets_rehearsal.sh tag | tee -a "$GITHUB_OUTPUT"
EOF
}

verify_job() {
    cat <<'EOF'
name: The rehearsal built every asset the release owes
needs: [assets, build, build-all-bins, build-apr-cuda, build-apr-cpu, build-apr-darwin]
if: ${{ !cancelled() && needs.assets.result == 'success' }}
runs-on: [self-hosted, Linux, X64, clean-room]
timeout-minutes: 10
steps:
  - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1  # v7.0.1
  - name: Fetch the kept checksums
    continue-on-error: true
    uses: actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c  # v8.0.1
    with:
      pattern: rehearsal-*
      merge-multiple: true
      path: ${{ runner.temp }}/rehearsal-got
  - name: Verdict (green only when every build ran and the asset set is complete)
    env:
      NEEDS: ${{ toJSON(needs) }}
      TAG: ${{ needs.assets.outputs.tag }}
    run: bash scripts/release/assets_rehearsal.sh verify "$TAG" "$RUNNER_TEMP/rehearsal-got" | tee -a "$GITHUB_STEP_SUMMARY"; exit "${PIPESTATUS[0]}"
EOF
}

# gen: binary-release.yml -> the rehearsal workflow, on stdout.
generate() {
    local t
    need_yq
    [ -f "$SRC" ] || die "missing $SRC"
    t=$(mktemp -d) || die "mktemp failed"
    guard_step > "$t/guard.yml"; keep_step > "$t/keep.yml"; artifact_step > "$t/art.yml"
    assets_job > "$t/assets.yml"; verify_job > "$t/verify.yml"
    cat > "$t/prog.yq" <<'EOF'
... comments = ""
| .name = "Assets rehearsal (nightly)"
| .on = {"schedule": [{"cron": "37 0 * * *"}], "workflow_dispatch": {}}
| .permissions = {"contents": "read"}
| .concurrency = {"group": "assets-rehearsal-nightly", "cancel-in-progress": false}
| del(.jobs["verify-apr-assets"], .jobs["smoke-cuda"], .jobs["smoke-cpu"], .jobs["summary"])
| .jobs.assets = load(strenv(T) + "/assets.yml")
| del(.jobs[] | select(.if == "needs.assets.outputs.present == 'false'") | .if)
| (.jobs[].steps[] | select(has("run")) | .run) |= sub("\$\{\{ github\.event\.release\.tag_name \|\| inputs\.tag \}\}"; "${{ needs.assets.outputs.tag }}")
| (.jobs[].steps[] | select(.with.ref == "${{ steps.tag.outputs.tag }}") | .with.ref) = "${{ github.sha }}"
| (.jobs[] | select(.steps | any_c((.name // "") | test("^Upload assets to release")))) |= (
    .steps = ([.steps[] | select((.name // "") | test("^Upload assets to release") | not)]
              + [load(strenv(T) + "/keep.yml"), load(strenv(T) + "/art.yml")]))
| .jobs.verify = load(strenv(T) + "/verify.yml")
| .jobs[].steps |= [load(strenv(T) + "/guard.yml")] + .
| . head_comment = "GENERATED by scripts/release/assets_rehearsal.sh gen from binary-release.yml. Do not edit:\nedit binary-release.yml or the generator, then regenerate. The rehearsal checks this file\nagainst a fresh generation (--check) and goes red on drift.\n\nNightly rehearsal of the release-asset build (RR-P03/P04, #4806): every asset a release owes,\nbuilt at the run commit by the release workflow's own steps on its own runners, never uploaded."
EOF
    T="$t" "$YQ" --from-file "$t/prog.yq" "$SRC"
    local rc=$?
    rm -rf "${t:?}"
    return "$rc"
}

canon() { "$YQ" -o=json -I=0 'sort_keys(..)' "$1"; }

# --check [FILE]: FILE (default the committed workflow) equals a fresh generation.
check() {
    local f=${1:-$OUT_DEFAULT} t rc=0
    need_yq
    [ -f "$f" ] || { echo "drift: $f does not exist; run: bash scripts/release/assets_rehearsal.sh gen"; return 1; }
    t=$(mktemp) || die "mktemp failed"
    generate > "$t" || { rm -f "${t:?}"; die "generation failed"; }
    if [ "$(canon "$t")" != "$(canon "$f")" ]; then
        echo "drift: ${f#"$ROOT"/} is not what binary-release.yml generates; run: bash scripts/release/assets_rehearsal.sh gen"
        rc=1
    else
        echo "ok: ${f#"$ROOT"/} is binary-release.yml's build, freshly generated"
    fi
    rm -f "${t:?}"
    return "$rc"
}

# fixture DIR: DIR/fx.yml, a workflow whose build jobs are the guard, a build, the keep step
# and the artifact step — the real step bodies, so lint and the guard are tested on them.
fixture() {
    local d=$1 j
    guard_step > "$d/guard.yml"; keep_step > "$d/keep.yml"; artifact_step > "$d/art.yml"
    "$YQ" -n '.name = "fixture" | .on = {"schedule": [{"cron": "37 0 * * *"}], "workflow_dispatch": {}} | .permissions = {"contents": "read"}' > "$d/fx.yml" || return 1
    for j in $BUILD_JOBS; do
        J=$j T=$d "$YQ" -i '.jobs[strenv(J)] = {"runs-on": "ubuntu-latest", "steps": [load(strenv(T) + "/guard.yml"), {"name": "build", "run": "true"}, load(strenv(T) + "/keep.yml"), load(strenv(T) + "/art.yml")]}' "$d/fx.yml" || return 1
    done
}

tag() {
    local v
    v=$(awk '/^\[workspace.package\]/{p=1;next} /^\[/{p=0} p&&/^version *=/{gsub(/"/,"",$3);print $3;exit}' "$ROOT/Cargo.toml")
    [[ $v =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "tag: workspace version '$v' is not X.Y.Z" >&2; return 1; }
    echo "tag=v$v-rc.0"
}

# verify TAG DIR, with NEEDS = toJSON(needs). Prints `verdict: green|red|not_measured`.
verify() {
    local tagv=$1 dir=$2 j r red="" nm="" list rc
    command -v jq > /dev/null 2>&1 || die "jq not found"
    for j in $BUILD_JOBS; do
        r=$(printf '%s' "${NEEDS:-}" | jq -r --arg j "$j" '.[$j].result // "absent"' 2>/dev/null) || r=unreadable
        case "$r" in
            success) ;;
            failure) red="$red $j" ;;
            *) nm="$nm $j($r)" ;;
        esac
    done
    if [ -n "$red" ]; then echo "verdict: red — build job(s) failed:$red"; return 1; fi
    if [ -n "$nm" ]; then echo "verdict: not_measured — build job(s) did not run to a result:$nm"; return 1; fi
    # A build job keeps an archive's .sha256 only after checking the archive against it
    # (`shasum -c` in its keep step), so a kept .sha256 stands for its archive too.
    list=$(mktemp) || die "mktemp failed"
    for s in "$dir"/*.sha256; do
        [ -f "$s" ] || continue
        s=${s##*/}; printf '%s\n%s\n' "${s%.sha256}" "$s"
    done | sort -u > "$list"
    echo "built: $(grep -c . "$list") asset names at $tagv"
    bash "$ROOT/scripts/check_release_assets.sh" "$tagv" --assets-from "$list"; rc=$?
    rm -f "${list:?}"
    case "$rc" in
        0) echo "verdict: green — every asset the release owes was built"; return 0 ;;
        1) echo "verdict: red — the build left an owed asset out"; return 1 ;;
        *) echo "verdict: not_measured — check_release_assets.sh exit $rc"; return 1 ;;
    esac
}


# ticket CONCLUSION SHA URL: the standing release policy (contracts/model-capability-ladder-v1.yaml,
# ladder.release_policy) makes this nightly one that cannot stop a release, and a night that is not
# green opens or updates its ONE ticket, owned by the policy's ticket_owner. Run by
# assets-rehearsal-tickets.yml on the rehearsal's workflow_run, which holds the issue credential
# this workflow may not. failure -> red; any other conclusion that is not success, skipped or
# neutral (cancelled, timed out, a startup failure) -> not_measured, which a ticket also records.
# One comment per commit (the marker assets-rehearsal@<SHA>). No readable owner: no ticket, exit 1.
# A read or write that fails is a failed run (exit 1), never a ticket. ASSETS_REHEARSAL_GH,
# ASSETS_REHEARSAL_LADDER and ASSETS_REHEARSAL_POLICY_LIB point elsewhere (the case table only).
TICKET_TITLE="assets-rehearsal nightly: not green"
ticket() {
    local concl=$1 c=$2 url=$3 gh=${ASSETS_REHEARSAL_GH:-gh} state owner mark j n len body page=50
    local lib=${ASSETS_REHEARSAL_POLICY_LIB:-$ROOT/scripts/lib/release_policy.sh}
    local ladder=${ASSETS_REHEARSAL_LADDER:-$ROOT/contracts/model-capability-ladder-v1.yaml}
    command -v jq > /dev/null 2>&1 || die "jq not found"
    [[ $c =~ ^[0-9a-f]{40}$ ]] || die "ticket: '$c' is not a commit sha"
    case "$concl" in
        success | skipped | neutral) echo "TICKET none: the rehearsal at ${c:0:9} concluded $concl"; return 0 ;;
        failure) state=red ;;
        ?*) state=not_measured ;;
        *) die "ticket: empty conclusion" ;;
    esac
    . "$lib" || { echo "NO-TICKET: cannot load the release policy library $lib"; return 1; }
    owner=$(rp_ticket_owner "$ladder") || { rp_ticket_owner "$ladder" > /dev/null; echo "NO-TICKET: no ticket owner, so no ticket is opened: $RP_WHY"; return 1; }
    mark="assets-rehearsal@$c"
    if ! j=$("$gh" issue list --state open --limit "$page" --search "\"$TICKET_TITLE\" in:title" --json number,title); then
        echo "NOT-MEASURED: the issue search for '$TICKET_TITLE' failed"; return 1
    fi
    n=$(printf '%s' "$j" | jq -r --arg t "$TICKET_TITLE" '[.[] | select(.title == $t) | .number] | min // empty' 2> /dev/null)
    # A search that fills its page may have cut the ticket off, and output that is not a JSON list
    # proves nothing: either way a missing title is not "no ticket", so nothing is opened.
    if [ -z "$n" ]; then
        len=$(jq 'if type == "array" then length else error end' <<< "$j" 2> /dev/null) || len=""
        if ! [[ "$len" =~ ^[0-9]+$ ]] || [ "$len" -ge "$page" ]; then
            echo "NOT-MEASURED: the issue search for '$TICKET_TITLE' returned a full page or no readable list"; return 1
        fi
    fi
    body="The nightly assets rehearsal at ${c:0:9} is $state ($concl): $url. Under the standing release policy this nightly cannot stop a release; the release notes list it as a known failure with this ticket until it is green. Owner: $owner. $mark"
    if [ -n "$n" ]; then
        if ! j=$("$gh" issue view "$n" --json body,comments); then echo "NOT-MEASURED: reading #$n failed"; return 1; fi
        if grep -qF -- "$mark" <<< "$j"; then echo "TICKET kept #$n (already names ${c:0:9})"
        elif "$gh" issue comment "$n" --body "$body" > /dev/null; then echo "TICKET updated #$n: $state"
        else echo "NOT-MEASURED: commenting on #$n failed"; return 1; fi
    else
        if ! n=$("$gh" issue create --title "$TICKET_TITLE" --body "$body"); then echo "NOT-MEASURED: opening the ticket failed"; return 1; fi
        n=${n##*/}
        [[ $n =~ ^[0-9]+$ ]] || { echo "NOT-MEASURED: opening the ticket printed no issue url"; return 1; }
        echo "TICKET opened #$n: $state"
    fi
}
# ---------------------------------------------------------------------------------------
PASS=0; FAIL=0
row() { # row NAME WANT_RC GOT_RC WANT_PATTERN OUTPUT
    if [ "$2" = "$3" ] && grep -q -E -- "$4" <<< "$5"; then PASS=$((PASS + 1)); printf 'PASS  %s\n' "$1"
    else FAIL=$((FAIL + 1)); printf 'FAIL  %s (want rc=%s /%s/, got rc=%s)\n%s\n' "$1" "$2" "$4" "$3" "$5"; fi
}

self_test() {
    local d out g full
    need_yq
    d=$(mktemp -d) || die "mktemp failed"
    fixture "$d" || die "fixture failed"

    # The generated workflow: no drift, cannot upload, and its guard is the guard step.
    generate > "$d/gen.yml" || die "generation failed"
    out=$(check "$d/gen.yml"); row "fresh generation -> no drift" 0 $? '^ok' "$out"
    out=$(lint "$d/gen.yml"); row "fresh generation -> cannot upload" 0 $? 'cannot upload' "$out"
    if [ -f "$OUT_DEFAULT" ]; then
        out=$(check); row "committed workflow -> no drift" 0 $? '^ok' "$out"
    fi
    "$YQ" '.jobs.build.steps += [{"name": "extra", "run": "true"}]' "$d/gen.yml" > "$d/m.yml"
    out=$(check "$d/m.yml"); row "hand edit of a build job -> drift" 1 $? '^drift' "$out"
    guard_step > "$d/want.yml"
    out=$(W="$d/want.yml" "$YQ" '[.jobs[] | select(.steps[0].run != load(strenv(W)).run)] | length' "$d/gen.yml")
    [ "$out" = 0 ]; row "every generated job's guard is the guard step" 0 $? '^0$' "$out"

    out=$(lint "$d/fx.yml"); row "fixture of the real steps -> cannot upload" 0 $? 'cannot upload' "$out"
    mut() { "$YQ" "$1" "$d/fx.yml" > "$d/m.yml"; }
    mut '.permissions.contents = "write"'
    out=$(lint "$d/m.yml"); row "contents: write -> lint red" 1 $? 'permissions' "$out"
    mut '.jobs.build.permissions = {"contents": "write"}'
    out=$(lint "$d/m.yml"); row "job-level permissions -> lint red" 1 $? 'set their own permissions' "$out"
    # The token every action takes implicitly (default inputs) is the workflow's: exactly
    # contents: read. A missing block falls back to the repository default, which can be
    # write; a job-level block, even a read one, replaces the workflow's for that job.
    mut 'del(.permissions)'
    out=$(lint "$d/m.yml"); row "no workflow permissions (repo default token) -> lint red" 1 $? 'workflow permissions' "$out"
    mut '.permissions = "write-all"'
    out=$(lint "$d/m.yml"); row "permissions: write-all -> lint red" 1 $? 'workflow permissions' "$out"
    mut '.permissions = "read-all"'
    out=$(lint "$d/m.yml"); row "permissions: read-all (not exactly contents: read) -> lint red" 1 $? 'workflow permissions' "$out"
    mut '.permissions.id-token = "write"'
    out=$(lint "$d/m.yml"); row "id-token write added to the read-only block -> lint red" 1 $? 'workflow permissions' "$out"
    mut '.jobs.build.permissions.contents = "read"'
    out=$(lint "$d/m.yml"); row "a job-level block, even a read-only one (replaces the workflow's) -> lint red" 1 $? 'set their own permissions' "$out"
    mut '.jobs.build.steps[1] = {"name": "co", "uses": "actions/checkout@v4"}'
    out=$(lint "$d/m.yml"); row "an allowlisted action taking the implicit token -> still cannot upload (contents: read)" 0 $? 'cannot upload' "$out"
    mut '.jobs.build.steps[1] = {"name": "co", "uses": "actions/checkout@v4", "with": {"token": "${{ github.token }}"}}'
    out=$(lint "$d/m.yml"); row "an action handed github.token explicitly -> lint red" 1 $? 'job token reachable' "$out"
    mut '.jobs.build.steps[1] = {"name": "rel", "uses": "softprops/action-gh-release@v2"}'
    out=$(lint "$d/m.yml"); row "a release action taking the implicit token -> lint red" 1 $? 'outside the allowlist' "$out"
    mut '.jobs.build.environment = "release"'
    out=$(lint "$d/m.yml"); row "an environment -> lint red" 1 $? 'environment' "$out"
    mut '.on.release = {"types": ["published"]}'
    out=$(lint "$d/m.yml"); row "release trigger -> lint red" 1 $? 'triggers' "$out"
    # Every way to reach a credential from an expression is red. The planted step is the
    # one a review found: a later step that maps the credential into its env and POSTs
    # with curl -d, which no upload pattern names.
    credrow() {
        V="$1" "$YQ" '.jobs["build-apr-cpu"].steps[1].env.T = strenv(V) | .jobs["build-apr-cpu"].steps[1].run = "curl -H \"authorization: token $T\" -d @x https://example.invalid/repos/o/r/releases"' "$d/fx.yml" > "$d/m.yml"
        out=$(lint "$d/m.yml"); row "a later step with env T: $1 and curl -d -> lint red" 1 $? "$2" "$out"
    }
    credrow '${{ secrets.GITHUB_TOKEN }}' 'secrets or vars'
    credrow "\${{ secrets['GH_TOKEN'] }}" 'secrets or vars'
    credrow '${{ toJSON(secrets) }}' 'secrets or vars'
    credrow '${{ Secrets.GH_TOKEN }}' 'secrets or vars'
    credrow '${{ vars.RELEASE_TOKEN }}' 'secrets or vars'
    credrow '${{ github.token }}' 'job token reachable'
    credrow '${{ GitHub . Token }}' 'job token reachable'
    credrow "\${{ github['token'] }}" 'job token reachable'
    credrow '${{ toJSON(github) }}' 'job token reachable'
    credrow '${{ fromJSON(toJSON(github)).token }}' 'job token reachable'
    credrow "\${{ format('{0}', github) }}" 'job token reachable'
    credrow '${{ toJSON(github.*) }}' 'job token reachable'
    credrow "\${{ github
      .token }}" 'job token reachable'
    # A YAML double-quoted escape is decoded before Actions reads the expression, so the
    # raw text never shows the word; the parsed form does. The escapes are built here,
    # not written out, so this file never holds one either.
    escrow() { # escrow VALUE PATTERN: VALUE goes in as a double-quoted YAML scalar
        PV=PLANTED "$YQ" '.jobs["build-apr-cpu"].steps[1].env.T = strenv(PV) | .jobs["build-apr-cpu"].steps[1].run = "curl -d @x https://example.invalid/"' "$d/fx.yml" > "$d/m0.yml"
        local y; y=$(< "$d/m0.yml"); printf '%s\n' "${y/PLANTED/"\"$1\""}" > "$d/m.yml"
        out=$(lint "$d/m.yml"); row "a later step with env T: \"$1\" (escaped) and curl -d -> lint red" 1 $? "$2" "$out"
    }
    esc_dollar=$(printf '\\%s' u0024); esc_g=$(printf '\\%s' u0067); esc_s=$(printf '\\%s' u0073)
    escrow "${esc_dollar}{{ ${esc_g}ithub.token }}" 'job token reachable'
    escrow "${esc_dollar}{{ ${esc_s}ecrets.GH_TOKEN }}" 'secrets or vars'
    # The upload tripwire reads the decoded text too: a run written with an escaped letter
    # is the same command once Actions has read the YAML.
    esc_c=$(printf '\\%s' x63)
    PV=PLANTED "$YQ" '.jobs["build-apr-cuda"].steps[4].name = "up" | .jobs["build-apr-cuda"].steps[4].run = strenv(PV)' "$d/fx.yml" > "$d/m0.yml"
    y=$(< "$d/m0.yml"); printf '%s\n' "${y/PLANTED/"\"${esc_c}argo publish\""}" > "$d/m.yml"
    out=$(lint "$d/m.yml"); row "a run of cargo publish with an escaped first letter -> lint red" 1 $? 'an upload' "$out"
    # Without a credential the same POST has nothing to write with: lint stays green, and
    # the reading of github that the release's steps use stays green with it.
    credrow_green() {
        V="$1" "$YQ" '.jobs["build-apr-cpu"].steps[1].env.T = strenv(V) | .jobs["build-apr-cpu"].steps[1].run = "curl -d @x https://example.invalid/"' "$d/fx.yml" > "$d/m.yml"
        out=$(lint "$d/m.yml"); row "a later step with env T: $1 and curl -d, no credential -> still cannot upload" 0 $? 'cannot upload' "$out"
    }
    credrow_green '${{ github.sha }}'
    credrow_green '${{ github.job }}-${{ strategy.job-index }}'
    mut '.jobs["build-apr-cuda"].steps += [{"name": "up", "run": "curl -sSf -X POST --data-binary @a https://uploads.github.com/x"}]'
    out=$(lint "$d/m.yml"); row "a planted upload step -> lint red" 1 $? 'upload' "$out"
    mut '.jobs["build-apr-cuda"].steps += [{"name": "up", "run": "gh release upload v1 a.tar.gz"}]'
    out=$(lint "$d/m.yml"); row "a planted gh release upload -> lint red" 1 $? 'upload' "$out"
    mut '.jobs["build-apr-cuda"].steps += [{"name": "up", "run": "gh api repos/o/r/releases/1/assets --input a.tar.gz"}]'
    out=$(lint "$d/m.yml"); row "a planted gh api call -> lint red" 1 $? 'upload' "$out"
    mut '.jobs["build-apr-cuda"].steps += [{"name": "up", "uses": "softprops/action-gh-release@v2", "with": {"files": "a.tar.gz"}}]'
    out=$(lint "$d/m.yml"); row "a planted release-upload action -> lint red" 1 $? 'outside the allowlist' "$out"
    # A whole step reached through an alias (`- *a`, the anchor on another step's env) is
    # the step Actions runs: lint reads the expanded document, so it is checked like any other.
    mut 'with(.jobs["build-apr-cuda"].steps; . = . + [{"name": "h", "run": "true", "env": {"if": "always()", "name": "z", "run": "echo past the guard"}}] | .[-1].env anchor = "a" | . = . + [null] | .[-1] alias = "a")'
    out=$(lint "$d/m.yml"); row "a whole-step alias whose anchor carries if: always() -> lint red" 1 $? 'carry an if:' "$out"
    mut 'with(.jobs["build-apr-cuda"].steps; . = . + [{"name": "h", "run": "true", "env": {"name": "up", "uses": "softprops/action-gh-release@v2"}}] | .[-1].env anchor = "a" | . = . + [null] | .[-1] alias = "a")'
    out=$(lint "$d/m.yml"); row "a whole-step alias whose anchor carries an unlisted uses: -> lint red" 1 $? 'outside the allowlist' "$out"
    # A merge key (`<<: *a`) after the step's own key: the step's own value wins under the
    # YAML spec, and lint must read it that way (yq's default lets the anchor win).
    mut 'with(.jobs["build-apr-cuda"].steps; . = . + [{"name": "c", "uses": "actions/checkout@v4"}] | .[-1] anchor = "a" | . = . + [{"name": "m", "uses": "softprops/action-gh-release@v2", "<<": "x"}] | .[-1]["<<"] alias = "a")'
    out=$(lint "$d/m.yml"); row "an unlisted uses: written before a merge key whose anchor is allowlisted -> lint red" 1 $? 'outside the allowlist' "$out"
    # A yq without the spec flag refuses the call, and lint reads that as red, never green.
    printf '%s\n' '#!/bin/sh' 'case " $* " in *" --yaml-fix-merge-anchor-to-spec"*) echo "unknown flag" >&2; exit 1;; esac' '"$REAL_YQ" "$@"' > "$d/oldyq"
    chmod +x "$d/oldyq"
    out=$(REAL_YQ="$YQ" "$d/oldyq" '.name' "$d/fx.yml"); row "control: that yq still answers a call without the flag" 0 $? '.' "$out"
    out=$(REAL_YQ="$YQ" YQ="$d/oldyq" lint "$d/fx.yml"); row "a yq that does not know the merge-to-spec flag -> lint red" 1 $? 'cannot parse' "$out"
    # Every yq call in lint fails closed. A yq that fails one query (picked by a substring
    # only that call's arguments hold) must turn lint red with that check's message.
    printf '%s\n' '#!/bin/sh' 'case " $* " in *"$FAILQ"*) echo "planted yq failure" >&2; exit 1;; esac' '"$REAL_YQ" "$@"' > "$d/failyq"
    chmod +x "$d/failyq"
    out=$(REAL_YQ="$YQ" YQ="$d/failyq" FAILQ='no such query' lint "$d/fx.yml"); row "control: the failing-yq stub with no match -> still cannot upload" 0 $? 'cannot upload' "$out"
    failrow() { # failrow SUBSTRING PATTERN
        out=$(REAL_YQ="$YQ" YQ="$d/failyq" FAILQ="$1" lint "$d/fx.yml" 2>/dev/null); row "yq fails the query holding '$1' -> lint red" 1 $? "$2" "$out"
    }
    failrow 'spec=true explode(.)' 'cannot parse'
    failrow '-I=0 explode(.)' 'cannot parse'
    failrow '-I=0 .permissions' 'failed on the workflow permissions check'
    failrow 'has("permissions")' 'failed on the job permissions check'
    failrow 'has("environment")' 'failed on the job environment check'
    failrow '.on | keys' 'failed on the trigger check'
    failrow 'load(strenv(G))' 'failed on the credential guard check'
    failrow 'has("if")' 'failed on the step if check'
    failrow 'concurrency|jobs)$' 'failed on the workflow key check'
    failrow 'strategy|env|if)$' 'failed on the job key check'
    failrow 'MACOSX_DEPLOYMENT_TARGET)$' 'failed on the job env check'
    failrow 'download-artifact)$' 'failed on the action allowlist check'
    failrow 'upload-artifact@' 'keeps no checksums'
    mut '.jobs["build-apr-darwin"].steps[0].run = "true"'
    out=$(lint "$d/m.yml"); row "a hollow guard (right name, empty body) -> lint red" 1 $? 'credential guard' "$out"
    mut 'del(.jobs["build-apr-darwin"].steps[0])'
    out=$(lint "$d/m.yml"); row "a job without the guard -> lint red" 1 $? 'credential guard' "$out"
    mut '.jobs[].steps[0].if = "${{ false }}"'
    out=$(lint "$d/m.yml"); row "a guard skipped by if: false (every job) -> lint red" 1 $? 'credential guard' "$out"
    mut '.jobs[].steps[0].continue-on-error = true'
    out=$(lint "$d/m.yml"); row "a guard with continue-on-error (every job) -> lint red" 1 $? 'credential guard' "$out"
    mut '.jobs.build.steps[0].shell = "sh"'
    out=$(lint "$d/m.yml"); row "a guard run by another shell -> lint red" 1 $? 'credential guard' "$out"
    mut '.jobs["build-apr-cuda"].steps += [{"name": "up", "run": "curl -sSf -XDELETE https://api.github.com/x"}]'
    out=$(lint "$d/m.yml"); row "a planted curl -XDELETE -> lint red" 1 $? 'upload' "$out"
    mut '.jobs["build-apr-cuda"].steps += [{"name": "up", "run": "curl -sSf --request POST https://api.github.com/x"}]'
    out=$(lint "$d/m.yml"); row "a planted curl --request POST -> lint red" 1 $? 'upload' "$out"
    # Every form of a step if: is red, the ones a pattern would miss among them.
    ifrow() {
        V="$1" "$YQ" '.jobs.build.steps[1].if = strenv(V)' "$d/fx.yml" > "$d/m.yml"
        out=$(lint "$d/m.yml"); row "a later step under if: $1 -> lint red" 1 $? 'carry an if:' "$out"
    }
    ifrow '${{ always() }}'
    ifrow 'failure()'
    ifrow '${{ !Cancelled() }}'
    ifrow '${{ success() || true }}'
    ifrow '${{ !success() }}'
    ifrow "\${{ format('{0}', 'true') }}"
    ifrow "\${{ success() && github.event_name == 'schedule' }}"
    ifrow 'success()'
    ifrow 'true'
    mut '.jobs.build.env.BASH_ENV = "x.sh"'
    out=$(lint "$d/m.yml"); row "a job env BASH_ENV -> lint red" 1 $? 'job env name.*BASH_ENV' "$out"
    # A job env written as an expression string has no keys for yq to read; the query fails,
    # and a failed query is red, never an empty list of names.
    V="\${{ fromJSON('{\"BASH_ENV\":\"x.sh\"}') }}" "$YQ" '.jobs["build-apr-cuda"].env = strenv(V)' "$d/fx.yml" > "$d/m.yml"
    out=$(lint "$d/m.yml"); row "a job env written as an expression string -> lint red" 1 $? 'failed on the job env check' "$out"
    mut '.jobs["build-apr-darwin"].env.MACOSX_DEPLOYMENT_TARGET = "11.0"'
    out=$(lint "$d/m.yml"); row "the release's own job env (MACOSX_DEPLOYMENT_TARGET) -> still cannot upload" 0 $? 'cannot upload' "$out"
    mut '.env.BASH_ENV = "x.sh"'
    out=$(lint "$d/m.yml"); row "a workflow env -> lint red" 1 $? 'workflow key.*env' "$out"
    mut '.defaults.run.shell = "sh"'
    out=$(lint "$d/m.yml"); row "workflow defaults -> lint red" 1 $? 'workflow key.*defaults' "$out"
    mut '.jobs.build.container = "ubuntu:24.04"'
    out=$(lint "$d/m.yml"); row "a job container -> lint red" 1 $? 'job key.*container' "$out"
    mut '.jobs.build.services.s.image = "x"'
    out=$(lint "$d/m.yml"); row "job services -> lint red" 1 $? 'job key.*services' "$out"
    mut '.jobs.build.defaults.run.working-directory = "x"'
    out=$(lint "$d/m.yml"); row "job defaults -> lint red" 1 $? 'job key.*defaults' "$out"
    mut 'del(.jobs.build.steps[] | select(.uses // "" | test("^actions/upload-artifact@")))'
    out=$(lint "$d/m.yml"); row "a build job that keeps nothing -> lint red" 1 $? 'keeps no checksums' "$out"

    # The guard, extracted from the fixture and run as the runner would.
    g="$d/guard.sh"; "$YQ" '.jobs.build.steps[0].run' "$d/fx.yml" > "$g"
    mkdir -p "$d/home/.cargo" "$d/credhome/.cargo"; echo 'token = "x"' > "$d/credhome/.cargo/credentials.toml"
    gr() { env -i PATH="$PATH" HOME="$d/home" "$@" bash "$g" 2>&1; }
    out=$(gr); row "guard: no credential -> runs" 0 $? 'no upload credential' "$out"
    out=$(gr CARGO_REGISTRY_TOKEN=x); row "guard: planted CARGO_REGISTRY_TOKEN -> refuses" 1 $? 'CARGO_REGISTRY_TOKEN is set' "$out"
    out=$(gr CARGO_REGISTRIES_CRATES_IO_TOKEN=x); row "guard: planted CARGO_REGISTRIES_CRATES_IO_TOKEN -> refuses" 1 $? 'CARGO_REGISTRIES_CRATES_IO_TOKEN is set' "$out"
    out=$(gr GH_TOKEN=x); row "guard: planted GH_TOKEN -> refuses" 1 $? 'GH_TOKEN is set' "$out"
    out=$(gr GITHUB_TOKEN=x); row "guard: planted GITHUB_TOKEN -> refuses" 1 $? 'GITHUB_TOKEN is set' "$out"
    out=$(gr ACTIONS_ID_TOKEN_REQUEST_TOKEN=x); row "guard: planted OIDC request token -> refuses" 1 $? 'ACTIONS_ID_TOKEN_REQUEST_TOKEN is set' "$out"
    out=$(gr HOME="$d/credhome"); row "guard: planted credentials.toml -> refuses" 1 $? 'credential file' "$out"
    out=$(gr CARGO_HOME="$d/credhome/.cargo"); row "guard: CARGO_HOME credentials -> refuses" 1 $? 'credential file' "$out"
    out=$(gr HOME="$d/credhome" CARGO_HOME=); row "guard: CARGO_HOME set but empty, home credentials.toml -> refuses" 1 $? 'credential file' "$out"

    # The verdict, over a fixture of the release's own list.
    full='{"assets":{"result":"success"},"build":{"result":"success"},"build-all-bins":{"result":"success"},"build-apr-cuda":{"result":"success"},"build-apr-cpu":{"result":"success"},"build-apr-darwin":{"result":"success"}}'
    mkdir -p "$d/got"
    bash "$ROOT/scripts/check_release_assets.sh" --list v9.9.9-rc.0 | grep '\.sha256$' | while read -r n; do : > "$d/got/$n"; done
    : > "$d/got/aprender-shell-v9.9.9-rc.0-x86_64-unknown-linux-gnu.tar.gz.sha256"
    out=$(NEEDS=$full verify v9.9.9-rc.0 "$d/got"); row "verify: every asset built -> green" 0 $? 'verdict: green' "$out"
    # The job-result rows run over the COMPLETE asset set, so only the job result can turn
    # them: a result that slipped through to the asset check would read green and fail the row.
    out=$(NEEDS=${full/'"build-apr-cpu":{"result":"success"}'/'"build-apr-cpu":{"result":"failure"}'} verify v9.9.9-rc.0 "$d/got")
    row "verify: a failed build job, assets complete -> red" 1 $? 'verdict: red — build job' "$out"
    out=$(NEEDS=${full/'"build-apr-cuda":{"result":"success"}'/'"build-apr-cuda":{"result":"skipped"}'} verify v9.9.9-rc.0 "$d/got")
    row "verify: a skipped build job, assets complete -> not_measured, never green" 1 $? 'verdict: not_measured' "$out"
    out=$(NEEDS=${full/'"build-apr-darwin":{"result":"success"}'/'"build-apr-darwin":{"result":"cancelled"}'} verify v9.9.9-rc.0 "$d/got")
    row "verify: a cancelled build job, assets complete -> not_measured" 1 $? 'verdict: not_measured' "$out"
    out=$(NEEDS='' verify v9.9.9-rc.0 "$d/got"); row "verify: unreadable needs, assets complete -> not_measured" 1 $? 'verdict: not_measured' "$out"
    rm -f "$d/got/apr-v9.9.9-rc.0-aarch64-apple-darwin-cpu.tar.gz.sha256"
    out=$(NEEDS=$full verify v9.9.9-rc.0 "$d/got"); row "verify: darwin asset missing -> red" 1 $? 'verdict: red — the build left' "$out"

    # The ticket of a night that is not green, through a gh stub that logs each call.
    cat > "$d/gh" <<'GH'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FXGH_LOG"
case "$1 $2" in
    "issue list") [ "${FXGH_FAIL:-}" != list ] || exit 1; cat -- "$FXGH_LIST" ;;
    "issue view") [ "${FXGH_FAIL:-}" != view ] || exit 1; cat -- "$FXGH_VIEW" ;;
    "issue comment") [ "${FXGH_FAIL:-}" != comment ] || exit 1 ;;
    "issue create") [ "${FXGH_FAIL:-}" != create ] || exit 1; echo "${FXGH_CREATED:-https://github.invalid/o/r/issues/77}" ;;
    *) exit 9 ;;
esac
GH
    chmod +x "$d/gh"
    { printf 'ladder:\n  release_policy:\n    name: crux-smoke\n    since: "0.0.0"\n    date: "d"\n    quote: "q"\n'
      printf '    hosts: [lambda, gx10]\n    thinking: ["off", "on"]\n    larger_rows: nightly\n    red_row_needs: ticket\n'
      printf '    ticket_owner: "#3598"\n    release_notes: known_failures\n'; } > "$d/ladder.yaml"
    grep -v ticket_owner "$d/ladder.yaml" > "$d/ladder-noowner.yaml"
    local sha=0123456789abcdef0123456789abcdef01234567 t='assets-rehearsal nightly: not green'
    tk() { # tk CONCLUSION LIST-JSON VIEW-JSON [FAIL] -> ticket's output, then the gh calls
        printf '%s\n' "$2" > "$d/tk-list"; printf '%s\n' "$3" > "$d/tk-view"; : > "$d/tk-log"
        ( export ASSETS_REHEARSAL_GH="$d/gh" FXGH_LOG="$d/tk-log" FXGH_LIST="$d/tk-list" FXGH_VIEW="$d/tk-view" FXGH_FAIL="${4:-}" \
              ASSETS_REHEARSAL_LADDER="${TK_LADDER:-$d/ladder.yaml}"
          ticket "$1" "$sha" https://github.invalid/run/1 ); local rc=$?
        printf 'GH %s\n' "$(tr '\n' ';' < "$d/tk-log")"; return "$rc"
    }
    out=$(tk success '[]' '{}'); row "ticket: a green night opens nothing" 0 $? '^TICKET none.*GH *$' "$(tr '\n' ' ' <<< "$out")"
    out=$(tk failure '[]' '{}'); row "ticket: a red night, no ticket -> opens one" 0 $? "TICKET opened #77: red.*issue create --title $t --body .*is red \(failure\).*Owner: #3598" "$(tr '\n' ' ' <<< "$out")"
    out=$(tk cancelled '[]' '{}'); row "ticket: a cancelled night -> not_measured ticket" 0 $? 'TICKET opened #77: not_measured' "$out"
    out=$(tk failure "[{\"number\":12,\"title\":\"$t\"},{\"number\":9,\"title\":\"$t\"}]" '{"body":"x"}')
    row "ticket: an open ticket -> one comment on the lowest" 0 $? 'TICKET updated #9: red.*issue view 9 .*issue comment 9 ' "$(tr '\n' ' ' <<< "$out")"
    out=$(tk failure "[{\"number\":9,\"title\":\"$t\"}]" "{\"body\":\"assets-rehearsal@$sha\"}")
    row "ticket: the commit already named -> kept, no comment" 0 $? 'TICKET kept #9' "$out"
    grep -q 'issue comment' <<< "$out"; row "ticket: kept means no comment call" 1 $? '.' "$out"
    out=$(tk failure "[{\"number\":9,\"title\":\"$t (old)\"}]" '{}'); row "ticket: a title that only contains it -> opens its own" 0 $? 'TICKET opened #77' "$out"
    out=$(TK_LADDER="$d/ladder-noowner.yaml" tk failure '[]' '{}'); row "ticket: no ticket owner -> no gh call, fails" 1 $? 'NO-TICKET: no ticket owner.*GH *$' "$(tr '\n' ' ' <<< "$out")"
    out=$(tk failure '[]' '{}' list); row "ticket: the search fails -> not_measured, nothing opened" 1 $? 'NOT-MEASURED: the issue search' "$out"
    grep -q 'issue create' <<< "$out"; row "ticket: a failed search opens nothing" 1 $? '.' "$out"
    out=$(tk failure "$(jq -cn '[range(50) | {number: (. + 100), title: "other"}]')" '{}'); row "ticket: a full page without the title -> opens nothing" 1 $? 'full page or no readable list' "$out"
    out=$(tk failure '{"not":"a list"}' '{}'); row "ticket: a search that is not a list -> opens nothing" 1 $? 'full page or no readable list' "$out"
    out=$(tk failure "[{\"number\":9,\"title\":\"$t\"}]" '{}' view); row "ticket: reading the ticket fails -> fails" 1 $? 'NOT-MEASURED: reading #9' "$out"
    out=$(tk failure "[{\"number\":9,\"title\":\"$t\"}]" '{"body":"x"}' comment); row "ticket: the comment fails -> fails" 1 $? 'NOT-MEASURED: commenting on #9' "$out"
    out=$(tk failure '[]' '{}' create); row "ticket: opening fails -> fails" 1 $? 'NOT-MEASURED: opening the ticket failed' "$out"
    out=$(FXGH_CREATED=oops tk failure '[]' '{}'); row "ticket: opening prints no url -> fails" 1 $? 'printed no issue url' "$out"

    out=$(tag); row "tag: workspace version -> vX.Y.Z-rc.0" 0 $? '^tag=v[0-9]+\.[0-9]+\.[0-9]+-rc\.0$' "$out"

    rm -rf "${d:?}"
    printf '%s self-test: %d pass, %d fail\n' "$PROG" "$PASS" "$FAIL"
    [ "$FAIL" -eq 0 ]
}

case "${1:-}" in
    gen) need_yq; o=${2:-$OUT_DEFAULT}; generate > "$o.tmp" && mv "$o.tmp" "$o" && echo "wrote ${o#"$ROOT"/}" ;;
    --check) check "${2:-}" ;;
    lint) [ $# -eq 2 ] || die "usage: lint FILE"; lint "$2" ;;
    tag) tag ;;
    verify) [ $# -eq 3 ] || die "usage: verify TAG DIR"; verify "$2" "$3" ;;
    ticket) [ $# -eq 4 ] || die "usage: ticket CONCLUSION SHA URL"; ticket "$2" "$3" "$4" ;;
    --self-test) self_test ;;
    *) sed -n '2,10p' "${BASH_SOURCE[0]}" >&2; exit 2 ;;
esac
