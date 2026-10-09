#!/usr/bin/env bash
# check_crate_contents.sh -- nothing secret in the published crates (release entry RR-P38, publish stage).
#
# For every publishable workspace member (cargo metadata --no-deps, `publish` not false) it reads the
# file list `cargo package --list` would upload (never --allow-dirty) and refuses on:
#   names    .env and .env.<x> (except the templates .env.example / .env.sample / .env.template),
#            credentials, credentials.toml, .git-credentials, .netrc, .npmrc, .pypirc,
#            id_rsa / id_dsa / id_ecdsa / id_ed25519, and *.pem *.key *.p12 *.pfx *.jks *.keystore
#   content  a PEM private-key block: the BEGIN line AND at least two base64 body lines after it (a bare
#            BEGIN line in a doc or a vulnerability template is not a key), and the registry, forge, cloud,
#            chat and model-hub token shapes at their full documented length and charset.
# No allowlist and no exemption: a fixture key moves out of the package (under an excluded tests/).
# A finding prints the crate, the file, the line and the rule -- never the matched text.
#
# Usage: check_crate_contents.sh            scan the workspace at the current directory
#        check_crate_contents.sh --selftest  planted cases against a stub cargo
#        check_crate_contents.sh --mutants   each mutant of this file must turn a self-test case red
# rc 0  every published crate was read and none carries a finding
# rc 1  a finding (FOUND lines)
# rc 2  not measured: cargo metadata or a package list failed, no crate or no file was read, or a listed
#       file is missing. Never a pass.
# cargo is ${CARGO:-cargo}.
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/${BASH_SOURCE[0]##*/}"

# Token shapes, at their documented length. Each is bounded on both sides so a longer identifier or a
# base64 run that happens to contain the prefix is not read as a token.
B='(^|[^A-Za-z0-9_])'
E='([^A-Za-z0-9_]|$)'
TOKEN_RULES=(
    "crates-io-token ${B}cio[A-Za-z0-9]{32}${E}"
    "github-token ${B}gh[pousr]_[A-Za-z0-9]{36}${E}"
    "github-fine-grained-token ${B}github_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{59}${E}"
    "aws-access-key-id ${B}(AKIA|ASIA)[0-9A-Z]{16}${E}"
    "slack-token ${B}xox[abposr]-[0-9]{10,13}-[0-9]{10,13}-[A-Za-z0-9-]{24,}${E}"
    "anthropic-key ${B}sk-ant-(api|admin)[0-9]{2}-[A-Za-z0-9_-]{80,}${E}"
    "huggingface-token ${B}hf_[A-Za-z0-9]{34}${E}"
)
PEM_BEGIN='-----BEGIN [A-Z0-9 ]*PRIVATE KEY( BLOCK)?-----'

nm() { echo "NOT MEASURED: $*" >&2; exit 2; }

# name_rule BASENAME -> prints the rule a file name breaks, nothing when it breaks none
name_rule() {
    case "$1" in
        .env.example | .env.sample | .env.template) ;;
        .env | .env.*) echo "env-file" ;;
        credentials | credentials.toml | .git-credentials | .netrc | .npmrc | .pypirc) echo "credentials-file" ;;
        id_rsa | id_dsa | id_ecdsa | id_ed25519) echo "ssh-private-key-file" ;;
        *.pem | *.key | *.p12 | *.pfx | *.jks | *.keystore) echo "key-file" ;;
    esac
}

# pem_blocks FILE... -> "file:line" per BEGIN line followed by >= 2 base64 body lines. A `\n` escape inside
# a string literal is read as a line break, so a key written on one line is still a block.
pem_blocks() {
    awk -v begin="$PEM_BEGIN" '
        FNR == 1 { st = 0 }
        {
            n = split($0, part, /\\n/)
            for (j = 1; j <= n; j++) {
                s = part[j]
                if (s ~ begin) { st = 1; body = 0; at = FNR; continue }
                if (!st) continue
                gsub(/^[ \t"\047,]+|[ \t"\047,;)]+$/, "", s)
                if (s ~ /^[A-Za-z0-9+\/]{16,}={0,2}$/) { if (++body == 2) { print FILENAME ":" at; st = 0 } }
                else if (s != "") st = 0
            }
        }' "$@"
}

# scan_crate NAME DIR LIST -> FOUND lines on stdout; adds to FILES and FINDINGS
scan_crate() {
    local name="$1" dir="$2" list="$3" f r rule re out rc hit
    local -a files=()
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        # written by cargo into the archive, not read from the tree
        case "$f" in Cargo.toml.orig | .cargo_vcs_info.json) continue ;; esac
        if [ ! -f "$dir/$f" ]; then
            [ "$f" = Cargo.lock ] && continue
            nm "$name: listed file $f is not in the tree"
        fi
        r=$(name_rule "${f##*/}")
        if [ -n "$r" ]; then echo "FOUND $name $f $r"; FINDINGS=$((FINDINGS + 1)); fi
        files+=("$f")
    done < "$list"
    [ "${#files[@]}" -gt 0 ] || nm "$name: the package lists no file"
    FILES=$((FILES + ${#files[@]}))
    for rule in "${TOKEN_RULES[@]}"; do
        re="${rule#* }"
        rc=0
        out=$(cd "$dir" && grep -nIHE -e "$re" -- "${files[@]}") || rc=$?
        [ "$rc" -le 1 ] || nm "$name: grep failed (rc=$rc)"
        while IFS= read -r hit; do
            [ -n "$hit" ] || continue
            echo "FOUND $name $(printf '%s' "$hit" | cut -d: -f1,2) ${rule%% *}"; FINDINGS=$((FINDINGS + 1))
        done <<< "$out"
    done
    rc=0
    out=$(cd "$dir" && pem_blocks "${files[@]}") || rc=$?
    [ "$rc" -eq 0 ] || nm "$name: the PEM reader failed (rc=$rc)"
    while IFS= read -r hit; do
        [ -n "$hit" ] || continue
        echo "FOUND $name $hit pem-private-key"; FINDINGS=$((FINDINGS + 1))
    done <<< "$out"
}

scan() {
    local cargo="${CARGO:-cargo}" meta crates name dir list
    CRATES=0 FILES=0 FINDINGS=0
    meta=$("$cargo" metadata --no-deps --format-version 1) || nm "cargo metadata failed"
    crates=$(printf '%s' "$meta" | jq -r '.packages[] | select(.publish != []) | .name + "\t" + (.manifest_path | rtrimstr("/Cargo.toml"))') \
        || nm "cargo metadata could not be read"
    tmp=$(mktemp -d "${TMPDIR:-/tmp}/crate-contents.XXXXXX") || nm "mktemp failed"
    trap 'rm -rf "${tmp:?}"' EXIT
    while IFS=$'\t' read -r name dir; do
        [ -n "$name" ] || continue
        list="$tmp/$name.list"
        "$cargo" package -p "$name" --list > "$list" || nm "cargo package -p $name --list failed"
        scan_crate "$name" "$dir" "$list"
        CRATES=$((CRATES + 1))
    done <<< "$crates"
    [ "$CRATES" -gt 0 ] || nm "no publishable crate was read"
    echo "crate-contents: crates=$CRATES files=$FILES findings=$FINDINGS"
    [ "$FINDINGS" -eq 0 ]
}

# ---- self-test -------------------------------------------------------------------------------------------
# A stub cargo answers `metadata` from $ST/meta.json and `package -p X --list` from $ST/X.list (rc from
# $ST/X.rc when present). Planted secrets are built by concatenation, so this file carries none.
stub_cargo() {
    cat > "$1/cargo" << 'EOF'
#!/usr/bin/env bash
st="${0%/*}"
case "$1" in
    metadata) [ -f "$st/meta.fail" ] && exit 101; cat "$st/meta.json" ;;
    package) [ -f "$st/$3.rc" ] && exit "$(cat "$st/$3.rc")"; cat "$st/$3.list" ;;
    *) exit 64 ;;
esac
EOF
    chmod +x "$1/cargo"
}

# crate ST NAME PUBLISH(true|false) FILE... -> a crate dir with the files, listed for `package --list`
crate() {
    local st="$1" name="$2" pub="$3" f
    shift 3
    mkdir -p "$st/ws/$name"
    : > "$st/$name.list"
    for f in "$@"; do
        mkdir -p "$st/ws/$name/$(dirname "$f")"
        [ -f "$st/ws/$name/$f" ] || printf 'fn main() {}\n' > "$st/ws/$name/$f"
        printf '%s\n' "$f" >> "$st/$name.list"
    done
    printf '%s\t%s\t%s\n' "$name" "$pub" "$st/ws/$name/Cargo.toml" >> "$st/crates.tsv"
}

meta() {
    jq -Rn '{packages: [inputs | split("\t") | {name: .[0], manifest_path: .[2], publish: (if .[1] == "false" then [] else null end)}]}' \
        < "$1/crates.tsv" > "$1/meta.json"
}

rep() { local i s=""; for ((i = 0; i < $2; i++)); do s="$s$1"; done; printf '%s' "$s"; }

# plant CASE ST -> builds the planted workspace for one case
plant() {
    local c="$1" st="$2" b64 begin
    begin="-----BEGIN ""PRIVATE KEY-----"
    b64="$(rep MIIEvQIBADANBgkq 4)"
    crate "$st" good true Cargo.toml src/lib.rs .env.example README.md
    # the clean shapes: a bare BEGIN line, a short chat-token-like string, a cert, a longer identifier
    printf 'let t = "%s";\nlet s = "%s";\n' "$begin" "xox""b-123456789012" >> "$st/ws/good/src/lib.rs"
    printf -- '-----BEGIN CERTIFICATE-----\n%s\n%s\n-----END CERTIFICATE-----\n' "$b64" "$b64" >> "$st/ws/good/README.md"
    printf 'let id = "x%s";\n' "hf_$(rep a 34)" >> "$st/ws/good/src/lib.rs"
    case "$c" in
        clean) ;;
        pem-block) printf '%s\n%s\n%s\n' "$begin" "$b64" "$b64" >> "$st/ws/good/src/lib.rs" ;;
        pem-escaped) printf 'const K: &str = "%s\\n%s\\n%s\\n";\n' "$begin" "$b64" "$b64" >> "$st/ws/good/src/lib.rs" ;;
        pem-literal) printf '    const K: &str = r"%s\n    %s\n    %s\n";\n' "$begin" "$b64" "$b64" >> "$st/ws/good/src/lib.rs" ;;
        pem-one-body-line) printf '%s\n%s\n\nfn x() {}\n%s\n' "$begin" "$b64" "$b64" >> "$st/ws/good/src/lib.rs" ;;
        crates-io) printf 'token = "%s"\n' "ci""o$(rep aB3 10)xy" >> "$st/ws/good/src/lib.rs" ;;
        github) printf 'let t = "%s";\n' "gh""p_$(rep aB3d 9)" >> "$st/ws/good/src/lib.rs" ;;
        github-pat) printf 'let t = "%s";\n' "github""_pat_$(rep A 22)_$(rep b 59)" >> "$st/ws/good/src/lib.rs" ;;
        aws) printf 'key = %s\n' "AKI""A$(rep Q7 8)" >> "$st/ws/good/src/lib.rs" ;;
        slack) printf 'let s = "%s";\n' "xox""b-1234567890-1234567890-$(rep aZ 12)" >> "$st/ws/good/src/lib.rs" ;;
        anthropic) printf 'let s = "%s";\n' "s""k""-a""nt-api03-$(rep aZ_9 21)" >> "$st/ws/good/src/lib.rs" ;;
        huggingface) printf 'let s = "%s";\n' "h""f_$(rep a 34)" >> "$st/ws/good/src/lib.rs" ;;
        name-env) crate "$st" bad true Cargo.toml src/lib.rs .env ;;
        name-env-local) crate "$st" bad true Cargo.toml src/lib.rs .env.local ;;
        name-ssh) crate "$st" bad true Cargo.toml src/lib.rs keys/id_ed25519 ;;
        name-pem) crate "$st" bad true Cargo.toml src/lib.rs certs/server.pem ;;
        name-credentials) crate "$st" bad true Cargo.toml src/lib.rs credentials.toml ;;
        unpublished-secret) crate "$st" priv false Cargo.toml src/lib.rs .env ;;
        no-crate) : > "$st/crates.tsv"; : > "$st/good.list" ;;
        all-unpublished) : > "$st/crates.tsv"; crate "$st" priv false Cargo.toml src/lib.rs ;;
        empty-list) crate "$st" bare true ; : > "$st/bare.list" ;;
        missing-file) printf 'src/gone.rs\n' >> "$st/good.list" ;;
        unreadable-file) chmod 000 "$st/ws/good/README.md" ;;
        generated-files) printf 'Cargo.toml.orig\n.cargo_vcs_info.json\nCargo.lock\n' >> "$st/good.list" ;;
        package-fails) echo 101 > "$st/good.rc" ;;
        metadata-fails) : > "$st/meta.fail" ;;
        *) echo "unknown case $c" >&2; return 1 ;;
    esac
    meta "$st"
}

# CASES: name rc pattern-the-output-must-hold
CASES=(
    "clean 0 crates=1 files=4 findings=0"
    "pem-block 1 FOUND good src/lib.rs:5 pem-private-key"
    "pem-escaped 1 FOUND good src/lib.rs:5 pem-private-key"
    "pem-literal 1 FOUND good src/lib.rs:5 pem-private-key"
    "pem-one-body-line 0 findings=0"
    "crates-io 1 FOUND good src/lib.rs:5 crates-io-token"
    "github 1 FOUND good src/lib.rs:5 github-token"
    "github-pat 1 FOUND good src/lib.rs:5 github-fine-grained-token"
    "aws 1 FOUND good src/lib.rs:5 aws-access-key-id"
    "slack 1 FOUND good src/lib.rs:5 slack-token"
    "anthropic 1 FOUND good src/lib.rs:5 anthropic-key"
    "huggingface 1 FOUND good src/lib.rs:5 huggingface-token"
    "name-env 1 FOUND bad .env env-file"
    "name-env-local 1 FOUND bad .env.local env-file"
    "name-ssh 1 FOUND bad keys/id_ed25519 ssh-private-key-file"
    "name-pem 1 FOUND bad certs/server.pem key-file"
    "name-credentials 1 FOUND bad credentials.toml credentials-file"
    "unpublished-secret 0 crates=1 files=4 findings=0"
    "no-crate 2 NOT MEASURED: no publishable crate was read"
    "all-unpublished 2 NOT MEASURED: no publishable crate was read"
    "empty-list 2 NOT MEASURED: bare: the package lists no file"
    "missing-file 2 NOT MEASURED: good: listed file src/gone.rs is not in the tree"
    "unreadable-file 2 NOT MEASURED: good: grep failed (rc=2)"
    "generated-files 0 crates=1 files=4 findings=0"
    "package-fails 2 NOT MEASURED: cargo package -p good --list failed"
    "metadata-fails 2 NOT MEASURED: cargo metadata failed"
)

# selftest [SCRIPT] -> runs every case against SCRIPT (this file by default); rc 1 on any wrong case
selftest() {
    local script="${1:-$SCRIPT_PATH}" root c name want pat st out rc fail=0 n=0
    root=$(mktemp -d "${TMPDIR:-/tmp}/crate-contents-st.XXXXXX") || { echo "selftest: mktemp failed" >&2; return 2; }
    for c in "${CASES[@]}"; do
        read -r name want pat <<< "$c"
        st="$root/$name"; mkdir -p "$st"; : > "$st/crates.tsv"
        stub_cargo "$st"
        plant "$name" "$st" || { fail=1; continue; }
        rc=0
        out=$(cd "$st/ws" && CARGO="$st/cargo" bash "$script" 2>&1) || rc=$?
        n=$((n + 1))
        if [ "$rc" != "$want" ] || [[ "$out" != *"$pat"* ]] || [[ "$out" == *"MIIEvQIBADANBgkq"* ]]; then
            echo "FAIL $name: rc=$rc (want $want), output lacks '$pat' or prints the secret:"; printf '%s\n' "$out" | sed 's/^/    /'
            fail=1
        else
            echo "ok   $name"
        fi
    done
    chmod -R u+rwX "${root:?}"; rm -rf "${root:?}"
    [ "$n" -eq "${#CASES[@]}" ] || { echo "selftest: ran $n of ${#CASES[@]} cases"; return 1; }
    [ "$fail" -eq 0 ] && echo "selftest: $n/${#CASES[@]} cases pass"
}

# MUTANTS: name@@from@@to, applied to a copy of this file; the copy's self-test must fail
MUTANTS=(
    "M01 env-template-widened@@.env.example | .env.sample | .env.template) ;;@@.env.example | .env.sample | .env.template | .env.*) ;;"
    "M02 pem-body-one-line@@if (++body == 2)@@if (++body == 3)"
    "M03 pem-escape-ignored@@n = split(\$0, part, /\\\\n/)@@n = split(\$0, part, /\\\\x/)"
    "M04 grep-error-passes@@[ \"\$rc\" -le 1 ] || nm@@[ \"\$rc\" -le 2 ] || nm"
    "M05 no-crate-passes@@[ \"\$CRATES\" -gt 0 ] || nm@@[ \"\$CRATES\" -ge 0 ] || nm"
    "M06 empty-list-passes@@[ \"\${#files[@]}\" -gt 0 ] || nm@@[ \"\${#files[@]}\" -ge 0 ] || nm"
    "M07 missing-file-skipped@@nm \"\$name: listed file@@continue; nm \"\$name: listed file"
    "M08 unpublished-scanned@@select(.publish != [])@@select(true)"
    "M09 findings-not-red@@[ \"\$FINDINGS\" -eq 0 ]@@[ \"\$FINDINGS\" -ge 0 ]"
    "M10 package-fail-passes@@--list > \"\$list\" || nm@@--list > \"\$list\" || true || nm"
    "M11 ssh-name-dropped@@id_rsa | id_dsa | id_ecdsa | id_ed25519)@@id_rsa | id_dsa | id_ecdsa)"
    "M12 slack-dropped@@\"slack-token \${B}xox@@\"slack-token \${B}NEVERxox"
    "M13 boundary-dropped@@B='(^|[^A-Za-z0-9_])'@@B=''"
    "M14 generated-not-skipped@@Cargo.toml.orig | .cargo_vcs_info.json) continue ;;@@Cargo.toml.orig) continue ;;"
)

mutants() {
    local m name from to tmp killed=0 n=0 out
    tmp=$(mktemp -d "${TMPDIR:-/tmp}/crate-contents-mut.XXXXXX") || { echo "mutants: mktemp failed" >&2; return 2; }
    selftest > "$tmp/base.out" 2>&1 || { cat "$tmp/base.out"; echo "mutants: the unmutated self-test is red"; rm -rf "${tmp:?}"; return 1; }
    for m in "${MUTANTS[@]}"; do
        name="${m%%@@*}"; m="${m#*@@}"; from="${m%%@@*}"; to="${m#*@@}"
        n=$((n + 1))
        out=$(FROM="$from" TO="$to" awk 'BEGIN { f = ENVIRON["FROM"]; t = ENVIRON["TO"] }
            { i = index($0, f); if (i && !done) { $0 = substr($0, 1, i - 1) t substr($0, i + length(f)); done = 1 } print }
            END { if (!done) exit 3 }' "$SCRIPT_PATH" > "$tmp/m.sh"; echo $?)
        if [ "$out" != 0 ]; then echo "ERROR $name: the mutation does not apply (a mutant that changes nothing is not a kill)"; rm -rf "${tmp:?}"; return 1; fi
        if (SCRIPT_PATH="$tmp/m.sh"; selftest "$tmp/m.sh") > "$tmp/m.out" 2>&1; then
            echo "SURVIVED $name"
        else
            echo "killed   $name"; killed=$((killed + 1))
        fi
    done
    rm -rf "${tmp:?}"
    echo "mutants: $killed/$n killed"
    [ "$killed" -eq "$n" ]
}

case "${1:-}" in
    "") scan ;;
    --selftest) selftest ;;
    --mutants) mutants ;;
    *) echo "usage: $0 [--selftest|--mutants]" >&2; exit 2 ;;
esac
