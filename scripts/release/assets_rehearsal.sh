#!/usr/bin/env bash
# assets_rehearsal.sh — the nightly rehearsal of the release-asset build (RR-P03/P04, #4806).
#
#   bash scripts/release/assets_rehearsal.sh gen [OUT]       # write the workflow (default: the committed path)
#   bash scripts/release/assets_rehearsal.sh --check [FILE]  # FILE is what gen writes now (default: committed)
#   bash scripts/release/assets_rehearsal.sh lint FILE       # FILE cannot upload
#   bash scripts/release/assets_rehearsal.sh tag             # tag=v<workspace version>-rc.0
#   bash scripts/release/assets_rehearsal.sh verify TAG DIR  # verdict over the built assets (env NEEDS)
#   bash scripts/release/assets_rehearsal.sh --self-test     # the case table
#
# binary-release.yml runs only on `release: published` and on dispatch, so the release's
# asset build had no nightly and could not show three green nights. The rehearsal is that
# nightly: the release's own build jobs, run on a schedule, never uploaded.
#
# The workflow is GENERATED from binary-release.yml (`gen`), never copied by hand: the build
# jobs (pv, all [[bin]]s, apr cuda, apr cpu, apr darwin) keep their own steps, runners and
# checks. Only these differ: the triggers (schedule + dispatch); the tag (below); the
# permissions (contents: read, no `secrets.`, no environment); each "Upload assets to
# release" step, which becomes a step that checks every archive against its .sha256 and
# keeps the checksums as a run artifact; the release-reading jobs, replaced by `verify`; and
# the credential guard at the start of every job. The rehearsal's first job regenerates and
# compares as canonical JSON (--check), so a binary-release.yml edit without a regen turns
# the rehearsal red instead of letting it rehearse a stale build.
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
    local f=$1 bad=0 v g
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
    # The guard is matched on its name AND its body, so a step that only borrows the name
    # (a hollow guard) does not count.
    g=$(mktemp) || die "mktemp failed"
    guard_step > "$g"
    v=$(G=$g "$YQ" '[.jobs[] | select((.steps[0].name // "") != load(strenv(G)).name or (.steps[0].run // "") != load(strenv(G)).run)] | length' "$f")
    rm -f "${g:?}"
    [ "$v" = 0 ] || { echo "lint: $v job(s) do not start with the credential guard"; bad=1; }
    if grep -n -E 'secrets\.' "$f"; then echo "lint: a secrets. reference"; bad=1; fi
    if grep -n -E 'uploads\.github\.com|-X (POST|PUT|PATCH|DELETE)|gh (release|api)|cargo publish|github\.event\.release|inputs\.tag' "$f"; then
        echo "lint: an upload, a release write or a release-event input"; bad=1
    fi
    # Actions are allowlisted, not denylisted: a release-upload action under any name is
    # outside the list. These three are all the release workflow and the rehearsal use.
    v=$("$YQ" '[.jobs[].steps[] | select(has("uses")) | .uses | sub("@.*"; "") | select(test("^actions/(checkout|upload-artifact|download-artifact)$") | not)] | unique | join(" ")' "$f")
    [ -z "$v" ] || { echo "lint: an action outside the allowlist (an upload path): $v"; bad=1; }
    for v in $BUILD_JOBS; do
        "$YQ" -e ".jobs[\"$v\"].steps[] | select(.uses // \"\" | test(\"^actions/upload-artifact@\"))" "$f" > /dev/null 2>&1 \
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
    mut '.jobs["build-apr-cuda"].steps += [{"name": "up", "run": "gh api repos/o/r/releases/1/assets --input a.tar.gz"}]'
    out=$(lint "$d/m.yml"); row "a planted gh api call -> lint red" 1 $? 'upload' "$out"
    mut '.jobs["build-apr-cuda"].steps += [{"name": "up", "uses": "softprops/action-gh-release@v2", "with": {"files": "a.tar.gz"}}]'
    out=$(lint "$d/m.yml"); row "a planted release-upload action -> lint red" 1 $? 'outside the allowlist' "$out"
    mut '.jobs["build-apr-darwin"].steps[0].run = "true"'
    out=$(lint "$d/m.yml"); row "a hollow guard (right name, empty body) -> lint red" 1 $? 'credential guard' "$out"
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
    gen) need_yq; o=${2:-$OUT_DEFAULT}; generate > "$o.tmp" && mv "$o.tmp" "$o" && echo "wrote ${o#"$ROOT"/}" ;;
    --check) check "${2:-}" ;;
    lint) [ $# -eq 2 ] || die "usage: lint FILE"; lint "$2" ;;
    tag) tag ;;
    verify) [ $# -eq 3 ] || die "usage: verify TAG DIR"; verify "$2" "$3" ;;
    --self-test) self_test ;;
    *) sed -n '2,9p' "${BASH_SOURCE[0]}" >&2; exit 2 ;;
esac
