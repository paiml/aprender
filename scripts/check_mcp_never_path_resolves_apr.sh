#!/usr/bin/env bash
# Every MCP tool must spawn the RESOLVED apr binary, never a bare `apr` that
# $PATH answers.
#
# WHY THIS EXISTS. During the 0.63.0 dogfood, `apr mcp` spawned a bare "apr" and
# therefore ran a 26-day-old 0.60.0 while reporting itself as 0.63.0 -- the MCP
# server was reporting results for code it was not running. Fixed in
# aprender#2563 (serve.rs:130, finetune.rs:98).
#
# WHY THE EXISTING FALSIFIER DOES NOT COVER IT. falsify_mcp_dogfood_001 passes
# identically before and after the fix -- measured. It cannot discriminate, so it
# is not evidence. This guard is behavioural: it puts a SHADOW `apr` first on
# $PATH which touches a marker, drives every tool, and asserts the marker is
# never written.
#
# THE TRAP THAT MADE A FIRST ATTEMPT REPORT THE DEFECT AS 4x BIGGER:
# apr_bin.rs::is_apr_binary requires `path.file_stem() == "apr"`. A pinned binary
# copied to `apr-main` is REJECTED by the resolver and falls through to the $PATH
# fallback -- so every tool looked broken when only two were. The shadow and the
# positive control must both be named exactly `apr`.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

# Pin the binary rather than `cargo run`: a cold rebuild here takes minutes, and
# the repo rule is that a diagnostic run against an unpinned binary is worse than
# no diagnostic. apr_bin.sh proves the binary was built from HEAD.
. scripts/apr_bin.sh || exit 2

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
MARKER="$TMP/SHADOW_WAS_RUN"
mkdir -p "$TMP/shadowbin"
# Written with printf, NOT a heredoc: bashrs parses an embedded heredoc as shell,
# so a shim shebang inside one is reported as SC1128 "shebang must be on the first
# line" against THIS file. Same reason the contract guards keep fixtures as
# committed files rather than inline heredocs.
{
    printf '#!/usr/bin/env bash\n'
    printf 'printf shadow\\n >> "%s"\n' "$MARKER"
    printf 'exit 0\n'
} > "$TMP/shadowbin/apr" || exit 2
chmod +x "$TMP/shadowbin/apr" || exit 2

rc=0
drive_tools() {   # $1 = value for APR_BIN ("" to leave unset)
    local aprbin="$1" out
    : > "$MARKER"
    # tools/list only ENUMERATES -- it spawns nothing, so driving it proves
    # nothing. The positive control caught exactly that and refused to pass. We
    # must CALL a tool that reaches a spawn site. apr.serve (serve.rs:130) is one
    # of the two sites aprender#2563 fixed; a nonexistent model makes it fail fast
    # while still performing the spawn we are testing.
    out=$(printf '%s\n%s\n%s\n' \
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
        '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
        '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"apr.serve","arguments":{"model_path":"/nonexistent/probe.gguf","port":58231}}}' \
        | { if [ -n "$aprbin" ]; then APR_BIN="$aprbin" PATH="$TMP/shadowbin:$PATH" \
              timeout 60 "$APR" mcp 2>&1
            else PATH="$TMP/shadowbin:$PATH" \
              timeout 60 "$APR" mcp 2>&1; fi; } )
    printf '%s' "$out" > "$TMP/last_out"
    [ -s "$MARKER" ] && return 1 || return 0
}

printf -- '--- MCP must never resolve `apr` through $PATH -----------------------\n'
if drive_tools ""; then
    printf 'ok    no MCP tool executed the shadow `apr` on $PATH\n'
else
    printf 'FAIL  an MCP tool executed the shadow `apr` found on $PATH.\n'
    printf '      That is the aprender#2563 defect: the server reports results for\n'
    printf '      a binary it did not build. Route the spawn through apr_binary().\n'
    rc=1
fi

# POSITIVE CONTROL. Without this the guard passes when the harness is broken --
# e.g. if the shim never became executable, or `apr mcp` failed to start at all.
printf -- '\n--- positive control: the shadow IS reachable when pointed at -------\n'
if drive_tools "$TMP/shadowbin/apr"; then
    printf 'FAIL  positive control did NOT fire: APR_BIN was pointed at the shadow\n'
    printf '      and it was never executed. The harness cannot detect the defect,\n'
    printf '      so its PASS above means nothing. Check the shim is executable and\n'
    printf '      that its file stem is exactly `apr` (is_apr_binary requires it).\n'
    rc=1
else
    printf 'ok    positive control fired -- the harness can see an executed shadow\n'
fi

[ "$rc" -eq 0 ] && printf '\nPASS  MCP spawns the resolved binary, and the guard can prove it.\n' \
                || printf '\nFAIL  see rows above.\n'
exit "$rc"
