#!/usr/bin/env bash
# falsify_sign_artifacts.sh -- case table and mutants for sign_artifacts.sh (APR 0.73.0 plan N2/A2, W3)
#
#   bash scripts/release/falsify_sign_artifacts.sh              # every refusal row + the green rows
#   bash scripts/release/falsify_sign_artifacts.sh --mutants    # the table plus one mutant per `# M:<name>` check
#   SIGN_SCRIPT=/path/to/copy.sh bash ... falsify_sign_artifacts.sh   # run the table on another copy
#
# Every row builds a throwaway evidence dir and throwaway minisign keys under mktemp; no real key, no network, no repo
# state. Setup steps always use the PRISTINE script, so a mutant changes only the invocation under test. A row passes
# when the exit status AND a substring of the message match: a different refusal is not credit for this one.
# A mutant is caught when the table fails with the check neutralised (`refuse` -> `:` on the tagged line).
#
# Exit 0 = every row held (with --mutants: and every mutant was caught). 1 = a row failed or a mutant survived.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
REAL="$HERE/sign_artifacts.sh"
SUBJECT="${SIGN_SCRIPT:-$REAL}"
command -v minisign > /dev/null 2>&1 || { echo "falsify_sign_artifacts: minisign is not installed" >&2; exit 2; }
TMP=$(mktemp -d -t falsign.XXXXXX) || exit 2
trap 'rm -rf -- "${TMP:?}"' EXIT

S=$(printf 'a%.0s' {1..40})        # today's sha
Y=$(printf 'b%.0s' {1..40})        # yesterday's sha
printf '\n\n' | minisign -G -W -p "$TMP/a.pub" -s "$TMP/a.key" > /dev/null 2>&1
printf '\n\n' | minisign -G -W -p "$TMP/b.pub" -s "$TMP/b.key" > /dev/null 2>&1
[ -f "$TMP/a.key" ] && [ -f "$TMP/b.pub" ] || { echo "falsify_sign_artifacts: cannot generate throwaway keys" >&2; exit 2; }
A_PUB="$TMP/a.pub" A_KEY="$TMP/a.key" B_PUB="$TMP/b.pub" B_KEY="$TMP/b.key"

# a minisign that exits 0 on -S without writing anything (the "reported success but wrote no signature" branch)
mkdir -p "$TMP/stub"
REAL_MS=$(command -v minisign)
printf '#!/bin/sh\nfor a in "$@"; do [ "$a" = "-S" ] && exit 0; done\nexec %s "$@"\n' "$REAL_MS" > "$TMP/stub/minisign"
chmod +x "$TMP/stub/minisign"

pass=0
fail=0
n=0
D=""
sign_ok() { bash "$REAL" --sign --sha "${1:-$S}" --dir "$D" --pubkey "$A_PUB" --key "$A_KEY" > /dev/null 2>&1; }
# mksigned <manifest-text> <trusted-comment>: a manifest the REAL key signed, so only the content checks can refuse it
mksigned() {
    printf '%s' "$1" > "$D/binaries.manifest"
    rm -f "$D/binaries.manifest.minisig"
    minisign -S -s "$A_KEY" -m "$D/binaries.manifest" -x "$D/binaries.manifest.minisig" -t "$2" > /dev/null 2>&1 < /dev/null
}
hsum() { sha256sum "$D/$1" | cut -d' ' -f1; }
H="aprender-binaries-manifest v1"

# row <name> <expected-rc> <expected-substring> <setup> <args>
row() {
    local name=$1 erc=$2 esub=$3 setup=$4 args=$5 out rc
    n=$((n + 1))
    D="$TMP/r$RUN.$n/d"
    mkdir -p "$D/sub"
    echo bin1 > "$D/apr"
    echo bin2 > "$D/sub/pv"
    # setup and the call share one subshell: a row's exports (PATH stub, key env) must not leak into the next row
    out=$(cd "$TMP" && { eval "$setup" > /dev/null 2>&1; eval "set -- $args"; bash "$SUBJECT" "$@" 2>&1; })
    rc=$?
    if [ "$rc" = "$erc" ] && case "$out" in *"$esub"*) true ;; *) false ;; esac; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        printf 'ROW FAIL %-34s want rc=%s ~"%s" got rc=%s: %s\n' "$name" "$erc" "$esub" "$rc" "$(printf '%s' "$out" | head -n 2 | tr '\n' ' ')" >&2
    fi
}

SIGN='--sign --sha $S --dir $D --pubkey $A_PUB --key $A_KEY'
VER='--verify --sha $S --dir $D --pubkey $A_PUB'

table() {
    # green rows
    row sign-green              0 "SIGNED 2 binaries"      ''                          "$SIGN"
    row sign-then-verify        0 "OK 2 binaries verified" 'sign_ok'                   "$VER"
    row sign-env-key-green      0 "SIGNED 2 binaries"      'export RELEASE_SIGNING_KEY_B64=$(base64 -w0 "$A_KEY")' '--sign --sha $S --dir $D --pubkey $A_PUB'
    row resign-overwrites       0 "SIGNED 2 binaries"      'sign_ok; echo new > "$D/apr"' "$SIGN"
    # A2: yesterday's receipt, tampering
    row verify-other-sha        1 "bound to"               'sign_ok $Y'                '--verify --sha $S --dir $D --pubkey $A_PUB'
    row verify-tampered-binary  1 "tampered"               'sign_ok; echo x >> "$D/apr"' "$VER"
    row verify-tampered-nested  1 "tampered"               'sign_ok; echo x >> "$D/sub/pv"' "$VER"
    row verify-edited-manifest  1 "does not verify"        'sign_ok; echo x >> "$D/apr"; sed -i "s/^[0-9a-f]\{64\}  apr$/$(hsum apr)  apr/" "$D/binaries.manifest"' "$VER"
    row verify-extra-file       1 "differ from the signed" 'sign_ok; echo e > "$D/extra"' "$VER"
    row verify-missing-file     1 "is missing"             'sign_ok; rm "$D/sub/pv"'   "$VER"
    row verify-wrong-key        1 "does not verify"        'sign_ok'                   '--verify --sha $S --dir $D --pubkey $B_PUB'
    row verify-no-signature     1 "no binaries.manifest.minisig" 'sign_ok; rm "$D/binaries.manifest.minisig"' "$VER"
    row verify-no-manifest      1 "no binaries.manifest in" 'sign_ok; rm "$D/binaries.manifest"' "$VER"
    row verify-empty-pubkey     1 "is missing or empty"    'sign_ok; : > "$TMP/empty.pub"' '--verify --sha $S --dir $D --pubkey $TMP/empty.pub'
    row verify-tampered-comment 1 "does not verify"        'sign_ok; sed -i "s/^trusted comment: .*/trusted comment: aprender-binaries sha=$S n=9/" "$D/binaries.manifest.minisig"' "$VER"
    # correctly signed but wrong content: only the content checks can refuse these
    row verify-bad-header       1 "header is not"          'mksigned "$(printf "bogus\nsha=%s\n%s  apr\n" $S $(hsum apr))
" "aprender-binaries sha=$S n=1"' "$VER"
    row verify-manifest-sha     1 "bound to"               'mksigned "$(printf "%s\nsha=%s\n%s  apr\n%s  sub/pv\n" "$H" $Y $(hsum apr) $(hsum sub/pv))
" "aprender-binaries sha=$S n=2"' "$VER"
    row verify-comment-sha      1 "signed comment is"      'mksigned "$(printf "%s\nsha=%s\n%s  apr\n%s  sub/pv\n" "$H" $S $(hsum apr) $(hsum sub/pv))
" "aprender-binaries sha=$Y n=2"' "$VER"
    row verify-comment-count    1 "signed comment is"      'mksigned "$(printf "%s\nsha=%s\n%s  apr\n%s  sub/pv\n" "$H" $S $(hsum apr) $(hsum sub/pv))
" "aprender-binaries sha=$S n=7"' "$VER"
    row verify-malformed-line   1 "malformed manifest line" 'mksigned "$(printf "%s\nsha=%s\nnot-a-hash apr\n" "$H" $S)
" "aprender-binaries sha=$S n=1"' "$VER"
    row verify-duplicate-entry  1 "is listed twice"        'mksigned "$(printf "%s\nsha=%s\n%s  apr\n%s  apr\n" "$H" $S $(hsum apr) $(hsum apr))
" "aprender-binaries sha=$S n=2"' "$VER"
    row verify-empty-manifest   1 "lists no binaries"      'mksigned "$(printf "%s\nsha=%s\n" "$H" $S)
" "aprender-binaries sha=$S n=0"' "$VER"
    # sign refusals
    row sign-no-key-source      1 "no signing key"         'unset RELEASE_SIGNING_KEY_B64' '--sign --sha $S --dir $D --pubkey $A_PUB'
    row sign-bad-b64-key        1 "did not decode"         'export RELEASE_SIGNING_KEY_B64="!!!"' '--sign --sha $S --dir $D --pubkey $A_PUB'
    row sign-empty-key-file     1 "missing or empty"       ': > "$TMP/empty.key"'      '--sign --sha $S --dir $D --pubkey $A_PUB --key $TMP/empty.key'
    row sign-garbage-key        1 "minisign -S failed"     'echo junk > "$TMP/junk.key"' '--sign --sha $S --dir $D --pubkey $A_PUB --key $TMP/junk.key'
    row sign-empty-dir          1 "no binaries under"      'find "$D" -type f -delete'             "$SIGN"
    row sign-bad-filename       1 "outside"                'echo z > "$D/has space"'   "$SIGN"
    row sign-unreadable-file    1 "cannot hash"            'chmod 000 "$D/apr"'        "$SIGN"
    row sign-pub-missing        1 "is missing or empty"    ''                          '--sign --sha $S --dir $D --pubkey $TMP/nope.pub --key $A_KEY'
    row sign-pubkey-mismatch    1 "does not verify under"  ''                          '--sign --sha $S --dir $D --pubkey $B_PUB --key $A_KEY'
    row sign-mismatch-cleans-up 1 "Removed"                ''                          '--sign --sha $S --dir $D --pubkey $B_PUB --key $A_KEY'
    row sign-no-sig-written     1 "wrote no signature"     'export PATH="$TMP/stub:$PATH"' "$SIGN"
    # usage (rc 2)
    row usage-no-mode           2 "one of --sign"          ''                          '--sha $S --dir $D --pubkey $A_PUB'
    row usage-short-sha         2 "40-hex"                 ''                          '--sign --sha abc --dir $D --pubkey $A_PUB'
    row usage-missing-dir       2 "existing directory"     ''                          '--sign --sha $S --dir $TMP/none --pubkey $A_PUB'
    row usage-no-pubkey         2 "--pubkey is required"   ''                          '--verify --sha $S --dir $D'
    row usage-unknown-arg       2 "unknown argument"       ''                          '--frob'
    row usage-flag-no-value     2 "needs a value"          ''                          '--sha'
}

# post-conditions that a message match cannot show
post_checks() {
    local out
    D="$TMP/p1.$RUN/d"; mkdir -p "$D"; echo b > "$D/apr"
    bash "$SUBJECT" --sign --sha "$S" --dir "$D" --pubkey "$B_PUB" --key "$A_KEY" > /dev/null 2>&1
    if [ -e "$D/binaries.manifest" ] || [ -e "$D/binaries.manifest.minisig" ]; then
        fail=$((fail + 1)); echo "ROW FAIL post-bad-pair-leaves-no-artifacts" >&2
    else pass=$((pass + 1)); fi
    D="$TMP/p2.$RUN/d"; mkdir -p "$D"; echo b > "$D/apr"
    # the key materialised from the env var must not outlive the process (private TMPDIR, so only our dirs count)
    mkdir -p "$TMP/tmpd"
    TMPDIR="$TMP/tmpd" RELEASE_SIGNING_KEY_B64=$(base64 -w0 "$A_KEY") bash "$SUBJECT" --sign --sha "$S" --dir "$D" --pubkey "$A_PUB" > /dev/null 2>&1
    out=$(find "$TMP/tmpd" -mindepth 1 | wc -l)
    if [ "$out" -ne 0 ]; then fail=$((fail + 1)); echo "ROW FAIL post-key-materialisation-removed" >&2
    else pass=$((pass + 1)); fi
}

RUN=0
run_table() { RUN=$((RUN + 1)); pass=0; fail=0; n=0; table; post_checks; }

run_table
printf 'falsify_sign_artifacts: table %s passed, %s failed\n' "$pass" "$fail"
[ "$fail" = 0 ] || exit 1

if [ "${1:-}" = "--mutants" ]; then
    caught=0
    survived=0
    tags=$(sed -n 's/.*# M:\([a-z0-9-]*\)$/\1/p' "$REAL")
    for t in $tags; do
        m="$TMP/mutant-$t.sh"
        sed "/# M:$t\$/ s/\\brefuse /: /" "$REAL" > "$m"
        if cmp -s "$REAL" "$m"; then echo "MUTANT $t: sed changed nothing" >&2; survived=$((survived + 1)); continue; fi
        # the mutant must still parse
        bash -n "$m" 2> /dev/null || { echo "MUTANT $t: does not parse" >&2; survived=$((survived + 1)); continue; }
        SUBJECT="$m"
        run_table 2> /dev/null
        if [ "$fail" -gt 0 ]; then caught=$((caught + 1)); printf 'MUTANT %-22s caught (%s rows failed)\n' "$t" "$fail"
        else survived=$((survived + 1)); printf 'MUTANT %-22s SURVIVED\n' "$t" >&2; fi
    done
    printf 'falsify_sign_artifacts: mutants %s caught, %s survived\n' "$caught" "$survived"
    [ "$survived" = 0 ] || exit 1
fi
exit 0
