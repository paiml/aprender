#!/usr/bin/env bash
# sign_artifacts.sh -- sign and verify the nightly's release binaries, bound to ONE commit sha (APR 0.73.0 plan N2, W3)
#
#   sign:    bash scripts/release/sign_artifacts.sh --sign   --sha <40-hex> --dir DIR --pubkey PUB [--key SECRET_FILE]
#   verify:  bash scripts/release/sign_artifacts.sh --verify --sha <40-hex> --dir DIR --pubkey PUB
#
# DIR is the evidence directory holding the binaries (any depth). Sign writes two files into it:
#   binaries.manifest          header, `sha=<40-hex>`, then one `<sha256>  <relpath>` line per binary, sorted
#   binaries.manifest.minisig  minisign signature over the manifest; trusted comment `aprender-binaries sha=<sha> n=<count>`
# Verify (the form the tag-CI job W4 calls, on a GitHub-hosted runner, with nothing built) refuses unless ALL hold:
#   the signature verifies under PUB; the signed trusted comment AND the manifest header name exactly --sha and the count;
#   every listed file exists with the listed sha256; no file in DIR is unlisted. Yesterday's evidence (another sha), a
#   tampered binary, an added or removed binary, an edited manifest and a signature from another key are each a refusal.
#
# Signing primitive: minisign, the one the repo already uses (scripts/pr_review_sign_receipt.sh). NO KEY LIVES IN THE REPO.
# The secret half comes from --key FILE or from env RELEASE_SIGNING_KEY_B64 (base64 of the minisign secret-key FILE; it is
# materialised under a 0700 temp dir and removed on every exit path); passphrase from env RELEASE_SIGNING_PASSWORD (may be
# empty for a -W key). The public half is always an argument: it is never defaulted to the PR-review key, which is a
# different trust root. Sign verifies its own output under PUB and deletes it if it does not verify.
#
# exit: 0 signed/verified - 1 refused (reason printed) - 2 usage or the box cannot answer (no minisign, jq-free, no sha tool)
# Every refusal is a `|| refuse "..." # M:<name>` line; falsify_sign_artifacts.sh --mutants disables each one in turn.
set -uo pipefail

PROG=sign_artifacts
MANIFEST=binaries.manifest
SIGFILE=binaries.manifest.minisig
HDR='aprender-binaries-manifest v1'
mode="" sha="" dir="" pub="" keyfile=""
WORK=""

usage() { printf '%s: %s\n' "$PROG" "$*" >&2; exit 2; }
refuse() { printf '%s: REFUSE - %s\n' "$PROG" "$*" >&2; exit 1; }
cleanup() { if [ -n "$WORK" ] && [ -d "$WORK" ]; then rm -rf -- "${WORK:?}"; fi; }
trap cleanup EXIT

while [ $# -gt 0 ]; do
    case "$1" in
        --sign) mode=sign; shift ;;
        --verify) mode=verify; shift ;;
        --sha | --dir | --pubkey | --key)
            [ $# -ge 2 ] || usage "$1 needs a value"
            case "$1" in --sha) sha=$2 ;; --dir) dir=$2 ;; --pubkey) pub=$2 ;; *) keyfile=$2 ;; esac
            shift 2 ;;
        *) usage "unknown argument: $1" ;;
    esac
done
[ -n "$mode" ] || usage "one of --sign / --verify is required"
[[ $sha =~ ^[0-9a-f]{40}$ ]] || usage "--sha must be a full 40-hex sha"
[ -n "$dir" ] && [ -d "$dir" ] || usage "--dir must be an existing directory"
[ -n "$pub" ] || usage "--pubkey is required (there is no default trust root)"
command -v minisign > /dev/null 2>&1 || usage "minisign is not installed"

if command -v sha256sum > /dev/null 2>&1; then hash_file() { sha256sum -- "$1" | cut -d' ' -f1; }
elif command -v shasum > /dev/null 2>&1; then hash_file() { shasum -a 256 -- "$1" | cut -d' ' -f1; }
else usage "neither sha256sum nor shasum is installed"; fi

# list_files: relpaths of every file under DIR except our own two outputs, C-sorted. A name outside [A-Za-z0-9._/+-]
# (space, newline, ...) cannot be written into a line-oriented manifest safely, so it is refused rather than escaped.
list_files() {
    local f rel
    while IFS= read -r -d '' f; do
        rel=${f#"$dir"/}
        [ "$rel" = "$MANIFEST" ] || [ "$rel" = "$SIGFILE" ] || printf '%s\0' "$rel"
    done < <(find "$dir" -type f -print0) | LC_ALL=C sort -z
}

build_manifest() {
    local out=$1 rel n=0 h
    printf '%s\nsha=%s\n' "$HDR" "$sha" > "$out"
    while IFS= read -r -d '' rel; do
        [[ $rel =~ ^[A-Za-z0-9._/+-]+$ ]] || refuse "file name '$rel' is outside [A-Za-z0-9._/+-]; not signed" # M:name-charset
        h=$(hash_file "$dir/$rel") && [ -n "$h" ] || refuse "cannot hash $rel" # M:hash-readable
        printf '%s  %s\n' "$h" "$rel" >> "$out"
        n=$((n + 1))
    done < <(list_files)
    [ "$n" -gt 0 ] || refuse "no binaries under $dir; an empty manifest signs nothing" # M:nonempty
    echo "$n"
}

materialise_key() {
    if [ -n "$keyfile" ]; then
        [ -s "$keyfile" ] || refuse "--key $keyfile is missing or empty" # M:key-file
        return 0
    fi
    [ -n "${RELEASE_SIGNING_KEY_B64:-}" ] || refuse "no signing key: pass --key FILE or set RELEASE_SIGNING_KEY_B64 (the key is never in the repo)" # M:key-source
    WORK=$(mktemp -d -t signart.XXXXXX) || usage "cannot create a temp dir"
    chmod 700 "$WORK"
    printf '%s' "$RELEASE_SIGNING_KEY_B64" | base64 -d > "$WORK/key" 2> /dev/null
    [ -s "$WORK/key" ] || refuse "RELEASE_SIGNING_KEY_B64 did not decode to a non-empty minisign key file" # M:key-decode
    chmod 600 "$WORK/key"
    keyfile=$WORK/key
}

do_verify() {
    local out tc n=0 line h rel seen="" listed actual
    [ -s "$pub" ] || refuse "public key $pub is missing or empty" # M:pub-present
    [ -f "$dir/$MANIFEST" ] || refuse "no $MANIFEST in $dir" # M:manifest-present
    [ -f "$dir/$SIGFILE" ] || refuse "no $SIGFILE in $dir; unsigned evidence is not accepted" # M:sig-present
    out=$(minisign -V -p "$pub" -m "$dir/$MANIFEST" -x "$dir/$SIGFILE" 2> /dev/null) \
        || refuse "signature does not verify under $pub (tampered manifest, tampered signature or another key)" # M:sig-valid
    tc=$(printf '%s\n' "$out" | sed -n 's/^Trusted comment: //p')
    [ "$(head -n 1 "$dir/$MANIFEST")" = "$HDR" ] || refuse "manifest header is not '$HDR'" # M:header
    [ "$(sed -n 2p "$dir/$MANIFEST")" = "sha=$sha" ] || refuse "manifest is bound to '$(sed -n 2p "$dir/$MANIFEST")', not sha=$sha" # M:manifest-sha
    while IFS= read -r line; do
        [[ $line =~ ^([0-9a-f]{64})\ \ ([A-Za-z0-9._/+-]+)$ ]] || refuse "malformed manifest line: $line" # M:line-format
        h=${BASH_REMATCH[1]}
        rel=${BASH_REMATCH[2]}
        case "
$seen
" in *"
$rel
"*) refuse "$rel is listed twice" ;; esac # M:no-dup
        seen="$seen
$rel"
        [ -f "$dir/$rel" ] || refuse "listed binary $rel is missing" # M:file-present
        [ "$(hash_file "$dir/$rel")" = "$h" ] || refuse "$rel does not match its signed sha256 (tampered)" # M:hash-match
        n=$((n + 1))
    done < <(tail -n +3 "$dir/$MANIFEST")
    [ "$n" -gt 0 ] || refuse "manifest lists no binaries" # M:verify-nonempty
    [ "$tc" = "aprender-binaries sha=$sha n=$n" ] || refuse "signed comment is '$tc', expected 'aprender-binaries sha=$sha n=$n'" # M:trusted-comment
    listed=$(tail -n +3 "$dir/$MANIFEST" | cut -c 67- | LC_ALL=C sort)
    actual=$(list_files | tr '\0' '\n')
    [ "$listed" = "$actual" ] || refuse "files in $dir differ from the signed set (unsigned extra file or missing one)" # M:no-extra
    printf '%s: OK %s binaries verified for %s under %s\n' "$PROG" "$n" "${sha:0:9}" "$pub"
}

do_sign() {
    local n pw
    [ -s "$pub" ] || refuse "public key $pub is missing or empty; signing without being able to verify the result ships an unverifiable signature" # M:sign-pub
    materialise_key
    n=$(build_manifest "$dir/$MANIFEST") || exit 1
    rm -f -- "$dir/$SIGFILE"
    pw=${RELEASE_SIGNING_PASSWORD:-}
    # a here-string, not a pipe: an unencrypted key never reads stdin, and under pipefail the writer dies of SIGPIPE (a measured 10% flake)
    minisign -S -s "$keyfile" -m "$dir/$MANIFEST" -x "$dir/$SIGFILE" \
        -t "aprender-binaries sha=$sha n=$n" -c "nightly binaries signed by the release signer" > /dev/null 2>&1 <<< "$pw" \
        || { rm -f -- "$dir/$MANIFEST" "$dir/$SIGFILE"; refuse "minisign -S failed (wrong passphrase or unreadable key)"; } # M:sign-rc
    [ -f "$dir/$SIGFILE" ] || { rm -f -- "$dir/$MANIFEST"; refuse "minisign reported success but wrote no signature"; } # M:sig-written
    # verify our own output in a subshell so a refusal removes the artifacts instead of leaving a bad signature behind
    ( do_verify ) > /dev/null 2>&1 \
        || { rm -f -- "$dir/$MANIFEST" "$dir/$SIGFILE"; refuse "the signature just produced does not verify under $pub; the key and the public half are not a pair. Removed."; } # M:self-verify
    printf '%s: SIGNED %s binaries for %s in %s\n' "$PROG" "$n" "${sha:0:9}" "$dir"
}

if [ "$mode" = sign ]; then do_sign; else do_verify; fi
