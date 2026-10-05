#!/usr/bin/env bash
# nightly_prune_release.sh DIST -- make the nightly release fit what publish sees.
#
# scripts/nightly_manifest.py publish lists the release's assets ONE page deep
# (per_page=100). The release held 234: 117 staged.* leftovers from runs that
# died mid-publish, and 117 real assets. The leftovers past page one were never
# deleted, so staging the same name again failed HTTP 422 already_exists (run
# 37171118510), every night. Purging the leftovers alone is not enough: 117 real
# assets still overflow page one, publish never sees the old copy of the names
# past it, and renaming its staged upload onto such a name is the same 422.
#
# So, reading EVERY page, before publish runs:
#   1. delete every staged.* asset (a leftover; publish stages afresh), and
#   2. delete the old copy of each name this run uploads that sits past page one
#      of what publish will list. "Uploads" is publish's own plan (publish_plan in
#      scripts/nightly_manifest.py), read from MANIFEST in jq: the tools of targets
#      green at SHA and built by this run, each asset and its .sha256, present in
#      DIST -- plus nightly-manifest.json. A file merely in DIST is not enough.
# A name past page one that this run does not upload is kept: it is some arch's
# last green build, and publish never touches it, so it cannot collide.
# Cost of 2: if publish then fails while staging, those (at most overflow-many)
# names are absent until the next green night -- the same window publish's own
# delete-then-rename opens for every name.
#
# Any API or tool failure exits nonzero: a listing that could not be read is not
# an empty release. HTTP 404 on the release itself is the first run: nothing to do.
#
#   bash scripts/nightly_prune_release.sh DIST MANIFEST SHA   (needs GH_TOKEN, GITHUB_REPOSITORY)
#   bash scripts/nightly_prune_release.sh --self-test
#
# Test seam: NIGHTLY_PRUNE_API=<cmd> is called as `<cmd> METHOD URL` in place of
# curl; it prints the body and exits 0, or exits 4 for a 404, else nonzero.
set -euo pipefail

PAGE=100
STAGED=staged.
MANIFEST_ASSET=nightly-manifest.json

die() { echo "::error::nightly_prune_release: $*" >&2; exit 1; }

api() { # METHOD URL -> body on stdout; rc 4 = HTTP 404
  if [ -n "${NIGHTLY_PRUNE_API:-}" ]; then
    "$NIGHTLY_PRUNE_API" "$1" "$2"
    return
  fi
  local body code
  body=$(mktemp)
  code=$(curl -sS --retry 3 --retry-all-errors -o "$body" -w '%{http_code}' -X "$1" \
    -H "Authorization: Bearer ${GH_TOKEN:?GH_TOKEN unset}" \
    -H "Accept: application/vnd.github+json" "$2") || { rm -f "${body:?}"; return 1; }
  case "$code" in
    2??) cat "$body"; rm -f "${body:?}" ;;
    404) rm -f "${body:?}"; return 4 ;;
    *) echo "::error::$1 $2: HTTP $code" >&2; rm -f "${body:?}"; return 1 ;;
  esac
}

prune() { # DIST MANIFEST SHA
  local dist="$1" manifest="$2" sha="$3" A rel rc page batch n all name id i plan
  local up=$'\n'  # newline-delimited set: the names publish uploads this run
  [ -d "$dist" ] || die "dist dir $dist does not exist"
  command -v jq >/dev/null || die "jq not found on this runner"
  [ -n "$sha" ] || die "no sha given"
  # publish_plan, mirrored: green at SHA, built by this run, each asset and its .sha256
  plan=$(jq -r --arg sha "$sha" '.run_id as $r | .targets[]
      | select(.status == "green" and .green_sha == $sha and .built_run_id == $r)
      | .tools[].asset | ., . + ".sha256"' "$manifest") || die "reading the plan from $manifest failed"
  while IFS= read -r name; do
    [ -z "$name" ] || [ ! -f "$dist/$name" ] || up+="$name"$'\n'
  done <<< "$plan"
  up+="$MANIFEST_ASSET"$'\n'
  A="https://api.github.com/repos/${GITHUB_REPOSITORY:?GITHUB_REPOSITORY unset}"
  rc=0
  rel=$(api GET "$A/releases/tags/nightly") || rc=$?
  if [ "$rc" -eq 4 ]; then
    echo "no nightly release yet (HTTP 404): nothing to prune"
    return 0
  fi
  [ "$rc" -eq 0 ] || die "reading the nightly release failed"
  rel=$(printf '%s' "$rel" | jq -er .id) || die "the nightly release has no id"

  all=""  # "id<TAB>name" per asset, in the order the API lists them
  page=1
  while :; do
    batch=$(api GET "$A/releases/$rel/assets?per_page=$PAGE&page=$page") || die "listing assets page $page failed"
    # a 200 whose body is not an array (a proxy's HTML, an error object) is a listing
    # that could not be read, never the empty page that ends it
    jq -e 'type == "array"' <<< "$batch" >/dev/null 2>&1 || die "assets page $page is not a JSON array"
    batch=$(printf '%s' "$batch" | jq -r '.[] | "\(.id)\t\(.name)"') || die "reading assets page $page failed"
    n=0
    [ -z "$batch" ] || n=$(printf '%s\n' "$batch" | wc -l)
    [ -z "$batch" ] || all+="$batch"$'\n'
    [ "$n" -ge "$PAGE" ] || break
    page=$((page + 1))
  done

  local staged=0 dropped=0 total=0
  i=0  # position among the non-staged assets: what publish's one page will hold
  while IFS=$'\t' read -r id name; do
    [ -n "$id" ] || continue
    total=$((total + 1))
    case "$name" in
      "$STAGED"*)
        api DELETE "$A/releases/assets/$id" >/dev/null || die "deleting $name ($id) failed"
        staged=$((staged + 1))
        continue ;;
    esac
    i=$((i + 1))
    [ "$i" -gt "$PAGE" ] || continue
    if [[ "$up" == *$'\n'"$name"$'\n'* ]]; then
      api DELETE "$A/releases/assets/$id" >/dev/null || die "deleting $name ($id) failed"
      dropped=$((dropped + 1))
      echo "past page one, re-uploaded this run: deleted old $name"
    fi
  done <<< "$all"
  echo "nightly release $rel: $total asset(s) over $page page(s); deleted $staged staged leftover(s) and $dropped old copy(ies) past page one"
}

# ---------------------------------------------------------------- self-test

self_test() {
  local tmp fails=0 out rc
  tmp=$(mktemp -d)
  # stub API: assets come from $tmp/assets (id<TAB>name), deletes are logged
  cat > "$tmp/api" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
d="$(dirname "$0")"
[ ! -e "$d/fail" ] || exit 1
case "$1 $2" in
  "GET "*/releases/tags/nightly) [ ! -e "$d/norel" ] || exit 4; echo '{"id": 7}' ;;
  "GET "*/assets\?per_page=*)
    p=${2##*page=}
    [ ! -e "$d/badpage$p" ] || { echo '{}'; exit 0; }
    s=$(( (p - 1) * 100 + 1 )); e=$(( p * 100 ))
    sed -n "${s},${e}p" "$d/assets" | jq -Rn '[inputs | split("\t") | {id: (.[0]|tonumber), name: .[1]}]' ;;
  "DELETE "*) echo "${2##*/}" >> "$d/deleted" ;;
  *) exit 1 ;;
esac
STUB
  chmod +x "$tmp/api"
  check() { # NAME GOT WANT
    if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: got '$2', want '$3'"; fails=$((fails + 1)); fi
  }
  mkplan() { # manifest: every file in dist is a tool of target g, green at S and built by run 7 --
             # except a name listed in $tmp/off as "NAME<TAB>STATUS<TAB>SHA<TAB>RUN", which gets a target of its own
    find "$tmp/dist" -type f -printf '%f\n' | jq -Rn --rawfile off "$tmp/off" '
      [inputs] as $all
      | ($off | split("\n") | map(select(. != "") | split("\t"))) as $o
      | ($o | map(.[0])) as $skip
      | {run_id: 7, targets: ({
          g: {status: "green", green_sha: "S", built_run_id: 7,
              tools: ([$all | .[] | select(. as $n | $skip | index($n) | not)] | map({key: ., value: {asset: .}}) | from_entries)}}
        + ($o | map({key: ("o-" + .[0]), value: {status: .[1], green_sha: .[2], built_run_id: (.[3] | tonumber),
                                                 tools: {x: {asset: .[0]}}}}) | from_entries))}' > "$tmp/manifest"
  }
  run() { # -> rc in $rc, output in $out; fresh delete log and manifest
    : > "$tmp/deleted"
    mkplan
    rc=0
    out=$(GITHUB_REPOSITORY=o/r NIGHTLY_PRUNE_API="$tmp/api" prune "$tmp/dist" "$tmp/manifest" S 2>&1) || rc=$?
  }
  mkdir -p "$tmp/dist"
  : > "$tmp/off"

  # the measured release: 234 assets, staged and real interleaved, every real one rebuilt
  : > "$tmp/assets"
  for i in $(seq 1 234); do
    if [ $((i % 2)) -eq 1 ]; then printf '%s\tstaged.a%s\n' "$i" "$i"; else printf '%s\ta%s\n' "$i" "$i"; : > "$tmp/dist/a$i"; fi
  done >> "$tmp/assets"
  run
  check "234 assets over 3 pages: exit 0" "$rc" 0
  check "every staged leftover is deleted, past page one too" "$(grep -c . "$tmp/deleted")" 134
  check "the 117 staged ones are all among the deletes" \
    "$(awk 'NR%2==1{print $1}' "$tmp/assets" | grep -cxFf - "$tmp/deleted")" 117
  check "real assets past publish's page one that are re-uploaded: the 17 old copies go" \
    "$(awk 'NR%2==0{print $1}' "$tmp/assets" | grep -cxFf - "$tmp/deleted")" 17
  check "the first 100 real assets are never deleted (publish swaps them itself)" \
    "$(awk 'NR%2==0{print $1}' "$tmp/assets" | head -100 | grep -cxFf - "$tmp/deleted" || true)" 0

  # a real asset past page one that this run does NOT rebuild is some arch's last green: kept
  rm -f "$tmp/dist/a234"; : > "$tmp/dist/a234.old"  # a planned name that CONTAINS a234 must not match it
  run
  check "a past-page-one asset this run does not upload is kept" "$(grep -cx 234 "$tmp/deleted" || true)" 0
  check "... and the other 16 old copies still go" \
    "$(awk 'NR%2==0{print $1}' "$tmp/assets" | grep -cxFf - "$tmp/deleted")" 16

  # a file in dist that publish's plan leaves out is not uploaded, so its old copy is kept -- one row per
  # filter of publish_plan: a red target, a target green only at an older sha, a night reused from run 6
  for o in 'red	S	7' 'green	S0	7' 'green	S	6'; do
    printf 'a232\t%s\n' "$o" > "$tmp/off"
    run
    check "a past-page-one asset in dist outside the plan (${o//	/ }) is kept" "$(grep -cx 232 "$tmp/deleted" || true)" 0
    check "... and the other 15 old copies still go (${o//	/ })" \
      "$(awk 'NR%2==0{print $1}' "$tmp/assets" | grep -cxFf - "$tmp/deleted")" 15
  done
  : > "$tmp/off"

  # an exact multiple of the page size ends on the empty page, and the manifest counts as uploaded
  : > "$tmp/assets"
  for i in $(seq 1 200); do printf '%s\tb%s\n' "$i" "$i"; done >> "$tmp/assets"
  printf '201\t%s\n' "$MANIFEST_ASSET" >> "$tmp/assets"
  run
  check "201 assets: page 3 is read and the manifest past page one is deleted" "$(cat "$tmp/deleted")" 201
  sed -i '$d' "$tmp/assets"
  run
  check "200 assets, none rebuilt: nothing deleted" "$(grep -c . "$tmp/deleted" || true)" 0
  check "... and the run says it read 200 assets" "$(printf '%s' "$out" | grep -c '200 asset(s)')" 1

  # first run, and fail-closed
  touch "$tmp/norel"; run
  check "no release yet (404): exit 0, nothing deleted" "$rc/$(grep -c . "$tmp/deleted" || true)" 0/0
  rm -f "$tmp/norel"; touch "$tmp/fail"; run
  check "an API failure is never an empty release: exit nonzero" "$rc" 1
  rm -f "$tmp/fail"
  touch "$tmp/badpage2"; run
  check "a 200 page that is an empty JSON object, not an array, is never the empty last page: exit nonzero" "$rc" 1
  rm -f "$tmp/badpage2"
  check "a missing dist dir: exit nonzero" \
    "$( (GITHUB_REPOSITORY=o/r NIGHTLY_PRUNE_API="$tmp/api" prune "$tmp/nope" "$tmp/manifest" S) >/dev/null 2>&1; echo $?)" 1
  echo '{}' > "$tmp/nomanifest"
  check "a manifest with no targets is an unreadable plan, never an empty one: exit nonzero" \
    "$( (GITHUB_REPOSITORY=o/r NIGHTLY_PRUNE_API="$tmp/api" prune "$tmp/dist" "$tmp/nomanifest" S) >/dev/null 2>&1; echo $?)" 1

  rm -rf "${tmp:?}"
  if [ "$fails" -eq 0 ]; then echo "nightly_prune_release self-test: PASS"; else echo "nightly_prune_release self-test: FAIL ($fails)"; fi
  [ "$fails" -eq 0 ]
}

usage() { echo "usage: nightly_prune_release.sh DIST MANIFEST SHA, or --self-test" >&2; exit 2; }

case "${1:-}" in
  --self-test) self_test ;;
  ""|-h|--help) usage ;;
  *) [ "$#" -eq 3 ] || usage
     prune "$1" "$2" "$3" ;;
esac
