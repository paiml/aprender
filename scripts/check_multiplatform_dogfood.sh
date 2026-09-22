#!/usr/bin/env bash
# Every release must be dogfooded on EVERY supported platform, not just the one
# the release engineer happens to be sitting at.
#
# WHY THIS EXISTS. The 0.64.0 pre-release sweep ran `cargo install aprender` on
# four hosts for the first time. The published crate had NEVER been verified on
# either arm64 platform. What that bought, in one afternoon:
#
#   aprender#2567  Q4_K GEMV -- the hottest kernel in quantized inference -- has
#                  ZERO aarch64 SIMD, and `matmul_q4k_f32_parallel` on non-x86 is
#                  a direct call to the SERIAL scalar routine. Numbers correct,
#                  speed wrong, so no correctness gate could ever have caught it.
#   aprender#2568  The OOM guard reads /proc/meminfo and `.unwrap_or(u64::MAX)`,
#                  so on macOS the threshold is ~12.8 EXABYTES and the guard can
#                  never fire. Its only test self-skips with
#                  `cfg!(target_os = "linux")` -- the platform where it is broken.
#   aprender#2572  `block v0.1.6` faces future-rustc rejection and sits under
#                  wgpu->metal, the ONLY GPU backend macOS has. Entirely absent
#                  from the Linux dependency graph.
#
# Every one of those is invisible from a single host BY CONSTRUCTION. That is the
# argument: this is not diligence theatre, it is the only way to see this class.
#
# WHAT THIS GATE CHECKS. Not that someone ran a sweep -- that a dated RECEIPT
# exists for each host, for the version being cut. A receipt is evidence; a
# checklist tick is not.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

# The supported platform matrix. A host is here because it is a DISTINCT
# combination of ISA, OS and accelerator -- not because we happen to own it.
#   lambda      x86_64 Linux  + RTX 4090 (sm_89)     consumer x86, AVX2
#   intel       x86_64 Linux  + no discrete GPU      Xeon, AVX-512 + VNNI
#   gx10        aarch64 Linux + GB10 (sm_121)        ARM server, unified memory
#   mini        arm64 macOS   + Metal                Apple silicon, no /proc, APFS
HOSTS="lambda intel gx10 mini"

# The version comes from cargo, not from a grep of Cargo.toml: under
# `version.workspace = true` the grep is empty and every receipt reads as STALE
# (review quorum on #2859, lane 3, measured). The grep stays as the fallback
# for a tree whose metadata cannot be resolved; an empty result is a refusal.
VERSION="$(cargo metadata --no-deps --offline --format-version 1 --manifest-path Cargo.toml 2>/dev/null \
  | python3 -c '
import json, os, sys
try:
    m = json.load(sys.stdin)
except ValueError:
    sys.exit(1)
root = os.path.realpath("Cargo.toml")
for p in m.get("packages", []):
    if os.path.realpath(p["manifest_path"]) == root:
        print(p["version"]); sys.exit(0)
sys.exit(1)' 2>/dev/null)" || VERSION=""
[ -n "$VERSION" ] || VERSION="$(awk -F'"' '/^version *=/{print $2; exit}' Cargo.toml)"
if [ -z "$VERSION" ]; then
    printf 'FAIL  the version being cut cannot be resolved (cargo metadata names no root package version; Cargo.toml has no literal)\n'
    exit 1
fi
# The receipts are read from the committed tree unless DOGFOOD_RECEIPTS_DIR names another dir
# (#3731): the release train's post-publish dogfood runs in the TAG's worktree AFTER its hosts
# step has produced the receipts in $AP/receipts, and the tag's tree cannot contain files made
# after it. The same receipts are committed under evidence/dogfood/<version>/ by the ledger PR.
DIR="${DOGFOOD_RECEIPTS_DIR:-evidence/dogfood/$VERSION}"
# ONE validator, shared with scripts/check_bench_receipt.sh. Two readers of one
# schema is the divergence class #2640 exists to close.
REPO_BENCH_VALIDATOR="scripts/lib/bench_receipt.py"
# Versions whose receipts were cut BEFORE parity lanes existed (#2696, added
# 2026-08-24). Failing a release for a gate that postdates its receipts is a
# gate nobody can satisfy. This list is closed: every version after 0.64.0
# requires parity lanes, and adding an entry here is a visible diff that has to
# be argued for.
PARITY_GRANDFATHERED="0.63.0 0.64.0"
# Blocks that run as REPORT for ONE named version (cop ruling on #3731, 2026-09-21). The train's own
# host_receipt.sh was run against the PUBLISHED 0.68.2 on all four hosts first; a block that went
# green there BLOCKS from the next release on, and a (host, block) that could NOT go green is listed
# here as VERSION:HOST:BLOCK:#ISSUE (BLOCK is bench or parity), the issue carrying the measured
# refusal. The row is still printed, as REPORT; host_receipt.sh names it in the receipt's
# unmeasured[], and autopilot's postpub step puts the REPORT rows into the release notes. An entry
# applies only to its own version, and only to a block that is ABSENT or VALID-BUT-NOT-GREEN (a
# required lane missing, or a lane below its floor): what the published 0.68.2 measured on lambda,
# whose block was complete and valid and still 0.40x. A block that is INVALID is judged as always:
# an entry never excuses a producer that wrote a malformed receipt. The list is closed.
REPORT_ONLY="0.69.1:lambda:parity:#3805 0.69.1:gx10:parity:#3805 0.69.1:intel:parity:#2855 0.69.1:mini:parity:#2855 0.69.1:lambda:bench:#3758 0.69.1:intel:bench:#3758 0.69.1:gx10:bench:#3758 0.69.1:mini:bench:#3758"
# report_only_for BLOCK -> the issue that owes BLOCK on host $h at $VERSION, or nothing
report_only_for() {
    local e
    for e in $REPORT_ONLY; do
        case "$e" in "$VERSION:$h:$1:"*) printf '%s' "${e##*:}"; return 0 ;; esac
    done
}
# The blocks the release train's host_receipt.sh emits beyond install (#3731): a sane `generate`,
# the .crate's sha256 as crates.io published it AND as the host downloaded it, equal, and a
# non-empty unmeasured[]. All three went green on the published 0.68.2 on all four hosts, so they
# BLOCK (the cop's first-green ruling). These versions' receipts were made by hand before the train
# produced any; they carry none of the three. The list is closed.
TRAIN_RECEIPT_GRANDFATHERED="0.63.0 0.64.0 0.65.2"
rc=0

printf -- '--- multi-platform dogfood receipts for %s -------------------------\n' "$VERSION"

# ANTI-VACUITY, TWO LAYERS. The first layer alone was UNSOUND and the hole was
# found by adversarial review before this gate ever landed:
#
#   `HOSTS=...` and the floor `-lt 4` were BOTH literals in THIS file, so ONE
#   commit editing both to two exited 0 -- measured, rc=0 with the matrix
#   silently halved. The original mutation evidence proved the floor FIRES at 2;
#   it never proved the floor could not be LOWERED. That is the same defect the
#   competitive-parity ledger spent five rounds on:
#
#       ANY STATE THE AUTHOR WRITES AND THE GATE READS CAN BE MOVED IN THE SAME
#       COMMIT.
#
# Layer 2 is the fix, and it is the parity resolution: the comparand lives on
# PROTECTED main. `main` requires a PR, review and passing CI to change, so the
# host list as it exists at origin/main is the one prior state this branch
# cannot rewrite in its own PR. The matrix may GROW freely and may never SHRINK.
n_hosts=$(printf '%s\n' $HOSTS | grep -c .)

# Layer 1: an absolute floor. Weak on its own (editable here), kept because it
# is the only thing that works during bootstrap, when main has no copy yet.
if [ "$n_hosts" -lt 4 ]; then
    printf 'FAIL  the platform matrix has %s host(s); at least 4 are required.\n' "$n_hosts"
    printf '      A shrinking matrix silently narrows what "verified" means.\n'
    exit 1
fi

# Layer 2: the matrix at origin/main is the floor, and this file cannot move it.
main_hosts=$(git show origin/main:scripts/check_multiplatform_dogfood.sh 2>/dev/null \
             | sed -n 's/^HOSTS="\(.*\)"$/\1/p' | head -1)
if [ -z "$main_hosts" ]; then
    # BOOTSTRAP, and it is self-limiting rather than renewable: reachable only
    # while this script does not exist at the protected ref. Once it lands, this
    # branch is unreachable forever. A renewable bootstrap is the fifth hat of
    # `registry: true` and is exactly what parity round 5 was caught on.
    printf '!     BOOTSTRAP: this gate is not yet on origin/main, so the matrix\n'
    printf '      has no protected floor this run. It gains one the moment this\n'
    printf '      script lands.\n'
else
    missing=""
    for h in $main_hosts; do
        case " $HOSTS " in *" $h "*) ;; *) missing="$missing $h" ;; esac
    done
    if [ -n "$missing" ]; then
        printf 'FAIL  the matrix DROPPED host(s) present at origin/main:%s\n' "$missing"
        printf '      The host list at the protected ref is the floor. Editing it in\n'
        printf '      the same commit that reads it is the defect this layer exists\n'
        printf '      to close -- growing the matrix is free, shrinking it is not.\n'
        exit 1
    fi
    printf 'ok    matrix covers every host present at origin/main (%s)\n' \
        "$(printf '%s\n' $main_hosts | grep -c .)"
fi

# PRE-PUBLISH PHASE (scripts/dogfood.sh --phase pre-publish, PMAT-745). Every
# receipt this gate reads comes from `cargo install aprender --version X` on a
# host, so before X is on crates.io there is nothing any host could have
# installed: the gate is DEFERRED with the obligation named, after the matrix
# floor above has been checked. In every other phase a missing receipt is the
# FAIL it always was. dogfood.sh records a DEFERRED line as DEFER only in the
# pre-publish phase and as FAIL anywhere else.
if [ "${DOGFOOD_PHASE:-full}" = pre-publish ]; then
    printf 'DEFERRED: %s is not on crates.io yet, so no host can have run `cargo install aprender --version %s`; owed by the post-publish dogfood as %s/{%s}.json\n' \
        "$VERSION" "$VERSION" "$DIR" "$(printf '%s' "$HOSTS" | tr ' ' ',')"
    exit 0
fi
for h in $HOSTS; do
    f="$DIR/$h.json"
    if [ ! -f "$f" ]; then
        printf 'FAIL  %-7s no receipt at %s\n' "$h" "$f"
        printf '        run the dogfood on %s and record its receipt before cutting.\n' "$h"
        rc=1
        continue
    fi

    got_ver=$(python3 -c "import json,sys;print(json.load(open('$f')).get('version_tested',''))" 2>/dev/null)
    inst=$(python3 -c "import json,sys;print(json.load(open('$f')).get('install_rc','MISSING'))" 2>/dev/null)
    when=$(python3 -c "import json,sys;print(json.load(open('$f')).get('date',''))" 2>/dev/null)

    if [ "$got_ver" != "$VERSION" ]; then
        printf 'FAIL  %-7s receipt is for %s, this cut is %s -- STALE\n' "$h" "${got_ver:-<none>}" "$VERSION"
        printf '        a receipt from a previous release says nothing about this one.\n'
        rc=1
    elif [ "$inst" != "0" ]; then
        printf 'FAIL  %-7s install_rc=%s -- `cargo install aprender` did not succeed\n' "$h" "$inst"
        rc=1
    else
        printf 'ok    %-7s %s verified %s (install rc=0)\n' "$h" "$VERSION" "${when:-undated}"
    fi

    # ── the train's receipt blocks (#3731) ─────────────────────────────────
    # "OK" is printed only when every block holds: a receipt this cannot read is a FAIL, never a pass.
    case " $TRAIN_RECEIPT_GRANDFATHERED " in
        *" $VERSION "*)
            printf 'REPORT %-6s generate/sha256/unmeasured not required for %s (hand-made receipt, pre-#3731)\n' "$h" "$VERSION" ;;
        *)
            why=$(python3 - "$f" <<'PY'
import json, re, sys
r = json.load(open(sys.argv[1]))
bad, hexre = [], re.compile(r"[0-9a-f]{64}")
g = r.get("generate")
if not isinstance(g, dict):
    bad.append("generate is null: " + next((u for u in r.get("unmeasured") or [] if isinstance(u, str) and u.startswith("generate:")), "and no reason is recorded"))
elif g.get("output_sane") is not True:
    bad.append("generate.output_sane is not true: the answer to 2+2 did not contain 4")
pub, mea = r.get("sha256_published"), r.get("sha256_measured")
if not (isinstance(pub, str) and hexre.fullmatch(pub) and isinstance(mea, str) and hexre.fullmatch(mea)):
    bad.append(f"sha256_published and sha256_measured are not both recorded ({pub!r}, {mea!r})")
elif pub != mea:
    bad.append(f"the .crate the host downloaded ({mea[:12]}) is not the one crates.io published ({pub[:12]})")
u = r.get("unmeasured")
if not (isinstance(u, list) and u and all(isinstance(x, str) and x for x in u)):
    bad.append("unmeasured[] is empty or absent: a receipt names what its host could not prove")
print(" | ".join(bad) if bad else "OK")
PY
)
            if [ "$why" = OK ]; then
                printf 'ok    %-7s generate sane, .crate sha256 published = measured, unmeasured[] named\n' "$h"
            else
                printf 'FAIL  %-7s %s\n' "$h" "${why:-the receipt could not be read}"
                rc=1
            fi ;;
    esac

    # ── the bench block (PARITY-003, aprender#2670) ────────────────────────
    #
    # CPU-CLASS, apr-vs-apr, NO COMPARATOR — and that is a decision, not an
    # omission. `cargo install aprender` builds CPU-only on every host in this
    # matrix (crates/apr-cli/Cargo.toml `default` carries no cuda, no wgpu),
    # while llama.cpp runs CUDA on lambda/gx10 and Metal on mini. A ratio here
    # would read ~0.05-0.10, nobody would red a release over it (correctly),
    # the row would go EXISTENCE-ONLY and the threshold would never arm — the
    # exact shape this gate itself had before #2658. The tree already documents
    # that collapse at crates/apr-cli/src/dispatch.rs:165.
    #
    # An apr-vs-apr self-ratchet catches OUR regressions, which is the goal,
    # without inventing a number nobody will act on. The comparator ratio lives
    # in the pre-publish phase where --features cuda exists (#2677).
    #
    # ABSENT was REPORT until the train produced the block (#3731). The hand-made
    # receipts (TRAIN_RECEIPT_GRANDFATHERED) still REPORT it. From then on the block
    # went green on the published 0.68.2 (lambda), so an absent block FAILs unless
    # REPORT_ONLY names this (version, host) with the issue that owes it. Every row
    # says WHY, from the producer's own refusal recorded in the train's receipt --
    # REPORT rows reach the release notes (autopilot postpub).
    if ! python3 "$REPO_BENCH_VALIDATOR" --has-bench "$f" >/dev/null 2>&1; then
        brefused=$(python3 -c "import json,sys;a=json.load(open(sys.argv[1])).get('bench_attempt') or {};print(str(a.get('reason',''))[:200])" "$f" 2>/dev/null)
        bwhy="${brefused:+the producer refused on this host: $brefused}"
        bowed=$(report_only_for bench)
        case " $TRAIN_RECEIPT_GRANDFATHERED " in
            *" $VERSION "*)
                printf 'REPORT %-6s no bench block (hand-made receipt, pre-#3731)%s\n' "$h" "${bwhy:+: $bwhy}" ;;
            *)
                if [ -n "$bowed" ]; then
                    printf 'REPORT %-6s no bench block: runs as REPORT for %s, owed by %s; %s\n' "$h" "$VERSION" "$bowed" "${bwhy:-no bench_attempt recorded}"
                else
                    printf 'FAIL  %-7s no bench block: %s\n' "$h" "${bwhy:-no bench_attempt recorded}"
                    rc=1
                fi ;;
        esac
    elif python3 "$REPO_BENCH_VALIDATOR" --bench "$f" >/dev/null 2>&1; then
        bmed=$(python3 "$REPO_BENCH_VALIDATOR" --bench-median "$f" 2>/dev/null)
        printf 'ok    %-7s bench: median %s ms, CPU-class self-ratchet\n' "$h" "${bmed:-?}"
    else
        printf 'FAIL  %-7s bench block present but INVALID:\n' "$h"
        python3 "$REPO_BENCH_VALIDATOR" --bench "$f" 2>&1 | sed 's/^/          /'
        rc=1
    fi

    # ── the parity lanes (#2696) ───────────────────────────────────────────
    #
    # The bench block above is an apr-vs-apr self-ratchet, and the comment
    # explaining why it has no comparator was RIGHT about the failure mode and
    # WRONG about the remedy. It said: `cargo install aprender` is CPU-only on
    # every host here, llama.cpp is CUDA/Metal, so a ratio would read 0.05-0.10,
    # nobody would red a release over it, and the row would go EXISTENCE-ONLY
    # forever. All true. The conclusion drawn was "do not compare".
    #
    # What that cost: the published binary's CPU-only-ness sat in this comment
    # as a known fact while NOTHING measured what it costs a user. On
    # 2026-08-24 it was measured for the first time — 15.7 tok/s decode against
    # llama.cpp's 158.9, and 7.5 SECONDS to first token, on a machine with an
    # idle RTX 4090 — because `--gpu` is accepted and silently ignored (#2696).
    #
    # The remedy is not to skip the comparison. It is to COMPARE WITHIN THE
    # CLASS. The published apr takes the cpu path, so its comparator is
    # llama.cpp `-ngl 0`, which also takes the cpu path. That ratio is
    # meaningful, it arms a floor, and no cross-class row is created. The
    # accelerated lanes belong to the pre-publish phase where --features cuda
    # exists; this phase owns the artifact users receive.
    #
    # bench_receipt.py --parity enforces the rest: same-class or no verdict,
    # ratio DERIVED from the samples rather than asserted, comparator pinned,
    # and the subject naming which artifact it was.
    case " $PARITY_GRANDFATHERED " in
        *" $VERSION "*)
            # Receipts for these versions were cut before parity lanes existed.
            # The list is short, dated, and cannot grow without a visible diff.
            printf 'REPORT %-6s parity lanes not required for %s (pre-#2696 cut)\n' "$h" "$VERSION"
            ;;
        *)
            report_only=$(report_only_for parity)
            if [ -n "$report_only" ] && ! python3 "$REPO_BENCH_VALIDATOR" --has-parity "$f" >/dev/null 2>&1; then
                printf 'REPORT %-6s no parity block: runs as REPORT for %s, owed by %s (it could not go green on the published 0.68.2)\n' \
                    "$h" "$VERSION" "$report_only"
            elif ! python3 "$REPO_BENCH_VALIDATOR" --has-parity "$f" >/dev/null 2>&1; then
                printf 'FAIL  %-7s no parity block. A release with no measured ratio\n' "$h"
                printf '        against a pinned comparator is a release whose speed\n'
                printf '        claim nothing checked (#2696).\n'
                rc=1
            elif ! python3 "$REPO_BENCH_VALIDATOR" --parity "$f" >/dev/null 2>&1; then
                printf 'FAIL  %-7s parity block present but INVALID:\n' "$h"
                python3 "$REPO_BENCH_VALIDATOR" --parity "$f" 2>&1 | sed 's/^/          /'
                rc=1
            else
                # Required lanes are derived from the host's OWN declared
                # accelerator, not from a list maintained beside it. A host that
                # gains a GPU gains a required lane without anyone remembering.
                accel=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('accelerator',''))" "$f")
                want="cpu"
                case "$accel" in
                    *sm_*|*NVIDIA*|*CUDA*) want="cpu cuda" ;;
                    *Metal*|*M1*|*M2*|*M3*|*M4*) want="cpu metal" ;;
                esac
                have=$(python3 "$REPO_BENCH_VALIDATOR" --parity-ratio "$f" 2>/dev/null | awk '{print $1}' | tr '\n' ' ')
                missing=""
                for w in $want; do
                    case " $have " in *" $w "*) : ;; *) missing="$missing $w" ;; esac
                done
                if [ -n "$missing" ] && [ -n "$report_only" ]; then
                    printf 'REPORT %-6s parity: accelerator '"'"'%s'"'"' needs lane(s)%s, receipt has: %s -- runs as REPORT for %s, owed by %s\n' \
                        "$h" "$accel" "$missing" "${have:-<none>}" "$VERSION" "$report_only"
                elif [ -n "$missing" ]; then
                    printf 'FAIL  %-7s accelerator '%s' needs lane(s)%s, receipt has: %s\n' \
                        "$h" "$accel" "$missing" "${have:-<none>}"
                    rc=1
                else
                    python3 "$REPO_BENCH_VALIDATOR" --parity-ratio "$f" 2>/dev/null | \
                    while read -r lane ratio verdict; do
                        printf '      %-7s parity %-6s %sx vs llama.cpp  %s\n' "$h" "$lane" "$ratio" "$verdict"
                    done
                    if grep -q ' FAIL$' <<< "$(python3 "$REPO_BENCH_VALIDATOR" --parity-ratio "$f" 2>/dev/null)" && [ -n "$report_only" ]; then
                        printf 'REPORT %-6s parity: a lane is below its declared floor -- runs as REPORT for %s, owed by %s\n' "$h" "$VERSION" "$report_only"
                    elif grep -q ' FAIL$' <<< "$(python3 "$REPO_BENCH_VALIDATOR" --parity-ratio "$f" 2>/dev/null)" ; then
                        printf 'FAIL  %-7s a parity lane is below its declared floor\n' "$h"
                        rc=1
                    else
                        printf 'ok    %-7s all required parity lanes at or above floor\n' "$h"
                    fi
                fi
            fi
            ;;
    esac
done

if [ "$rc" -eq 0 ]; then
    printf '\nPASS  all %s platforms carry a receipt for %s.\n' "$n_hosts" "$VERSION"
else
    printf '\nFAIL  see rows above. Record receipts under %s/ as <host>.json with at\n' "$DIR"
    printf '      least: {"host","arch","version_tested","date","install_rc"}.\n'
fi
exit "$rc"
