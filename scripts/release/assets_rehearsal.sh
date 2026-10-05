#!/usr/bin/env bash
# assets_rehearsal.sh — the nightly rehearsal of the release-asset build (RR-P03/P04, #4806).
#
#   bash scripts/release/assets_rehearsal.sh lint FILE       # FILE cannot upload
#   bash scripts/release/assets_rehearsal.sh tag             # tag=v<workspace version>-rc.0
#   bash scripts/release/assets_rehearsal.sh verify TAG DIR  # verdict over the built assets (env NEEDS)
#   bash scripts/release/assets_rehearsal.sh --self-test     # the case table
#
# binary-release.yml runs only on `release: published` and on dispatch, so the release's
# asset build had no nightly and could not show three green nights. The rehearsal is that
# nightly: the release's own build jobs, run on a schedule, never uploaded.
#
# It cannot upload. `lint` checks every property that keeps it so: workflow permissions
# exactly contents: read, no job-level permissions, no environment, triggers exactly
# schedule + dispatch, no `secrets.` reference, no upload, release write or release-event
# input, every job starting with the credential guard, and every build job keeping its
# checksums as a run artifact. The guard looks for the credential itself (a flag is not a
# guard): it refuses to run when GH_TOKEN, GITHUB_TOKEN, a cargo registry token, an OIDC
# request token or a cargo credentials file is present. The case table runs lint and the
# guard over a fixture workflow built from the real step bodies.
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
  ch=$(printenv CARGO_HOME 2>/dev/null) || ch="$HOME/.cargo"
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

# lint FILE: every property that keeps the rehearsal from uploading. One line per failure.
lint() {
    local f=$1 bad=0 v
    need_yq
    [ -f "$f" ] || die "lint: no such file $f"
    v=$("$YQ" -o=json -I=0 '.permissions' "$f")
    [ "$v" = '{"contents":"read"}' ] || { echo "lint: workflow permissions are $v, not exactly contents: read"; bad=1; }
    v=$("$YQ" '[.jobs[] | select(has("permissions"))] | length' "$f")
    [ "$v" = 0 ] || { echo "lint: $v job(s) set their own permissions"; bad=1; }
    v=$("$YQ" '[.jobs[] | select(has("environment"))] | length' "$f")
    [ "$v" = 0 ] || { echo "lint: $v job(s) name an environment (environment secrets)"; bad=1; }
    v=$("$YQ" '.on | keys | sort | join(",")' "$f")
    [ "$v" = "schedule,workflow_dispatch" ] || { echo "lint: triggers are '$v', not schedule,workflow_dispatch"; bad=1; }
    v=$("$YQ" '[.jobs[] | select((.steps[0].name // "") != "'"$GUARD_NAME"'")] | length' "$f")
    [ "$v" = 0 ] || { echo "lint: $v job(s) do not start with the credential guard"; bad=1; }
    if grep -n -E 'secrets\.' "$f"; then echo "lint: a secrets. reference"; bad=1; fi
    if grep -n -E 'uploads\.github\.com|-X (POST|PUT|PATCH|DELETE)|gh release|cargo publish|github\.event\.release|inputs\.tag' "$f"; then
        echo "lint: an upload, a release write or a release-event input"; bad=1
    fi
    for v in $BUILD_JOBS; do
        "$YQ" -e ".jobs[\"$v\"].steps[] | select(.uses // \"\" | test(\"^actions/upload-artifact@\"))" "$f" > /dev/null 2>&1 \
            || { echo "lint: build job $v keeps no checksums"; bad=1; }
    done
    [ "$bad" -eq 0 ] && echo "ok: ${f#"$ROOT"/} cannot upload"
    return "$bad"
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

    out=$(lint "$d/fx.yml"); row "fixture of the real steps -> cannot upload" 0 $? 'cannot upload' "$out"
    mut() { "$YQ" "$1" "$d/fx.yml" > "$d/m.yml"; }
    mut '.permissions.contents = "write"'
    out=$(lint "$d/m.yml"); row "contents: write -> lint red" 1 $? 'permissions' "$out"
    mut '.jobs.build.permissions = {"contents": "write"}'
    out=$(lint "$d/m.yml"); row "job-level permissions -> lint red" 1 $? 'set their own permissions' "$out"
    mut '.jobs.build.environment = "release"'
    out=$(lint "$d/m.yml"); row "an environment -> lint red" 1 $? 'environment' "$out"
    mut '.on.release = {"types": ["published"]}'
    out=$(lint "$d/m.yml"); row "release trigger -> lint red" 1 $? 'triggers' "$out"
    mut '.jobs["build-apr-cpu"].steps[-1].env = {"GH_TOKEN": "${{ secrets.GITHUB_TOKEN }}"}'
    out=$(lint "$d/m.yml"); row "a secrets. reference -> lint red" 1 $? 'secrets' "$out"
    mut '.jobs["build-apr-cuda"].steps += [{"name": "up", "run": "curl -sSf -X POST --data-binary @a https://uploads.github.com/x"}]'
    out=$(lint "$d/m.yml"); row "a planted upload step -> lint red" 1 $? 'upload' "$out"
    mut '.jobs["build-apr-cuda"].steps += [{"name": "up", "run": "gh release upload v1 a.tar.gz"}]'
    out=$(lint "$d/m.yml"); row "a planted gh release upload -> lint red" 1 $? 'upload' "$out"
    mut 'del(.jobs["build-apr-darwin"].steps[0])'
    out=$(lint "$d/m.yml"); row "a job without the guard -> lint red" 1 $? 'credential guard' "$out"
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

    out=$(tag); row "tag: workspace version -> vX.Y.Z-rc.0" 0 $? '^tag=v[0-9]+\.[0-9]+\.[0-9]+-rc\.0$' "$out"

    rm -rf "${d:?}"
    printf '%s self-test: %d pass, %d fail\n' "$PROG" "$PASS" "$FAIL"
    [ "$FAIL" -eq 0 ]
}

case "${1:-}" in
    lint) [ $# -eq 2 ] || die "usage: lint FILE"; lint "$2" ;;
    tag) tag ;;
    verify) [ $# -eq 3 ] || die "usage: verify TAG DIR"; verify "$2" "$3" ;;
    --self-test) self_test ;;
    *) sed -n '2,7p' "${BASH_SOURCE[0]}" >&2; exit 2 ;;
esac
