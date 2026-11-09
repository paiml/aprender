# lib_write_guard.sh — the release rehearsal's write guard (APR-071 spec H10, row B1; E1 #3998).
#
# H10: "A rehearsal makes no write outside its own state directory: no `git tag`, no `git push` of
# any ref, no `gh release create` or `edit`, no workflow dispatch on a tag, no `gh issue` or `gh pr`
# write, no milestone change, no `cargo publish`."
#
# scripts/release/rehearse.sh puts one stub per guarded tool (git gh cargo ssh scp sftp curl) first
# on PATH -- inside the redirected CARGO_HOME/bin too, because the release scripts prepend
# "$CARGO_HOME/bin" and prepare_bump.sh calls "$CARGO_HOME/bin/cargo" by path. Every stub sources
# this file and calls wg_stub. Each call is recorded; a read runs the real tool; a write is refused
# (exit 97) and recorded as a WRITE row, and one WRITE row makes the night red.
#
# Unknown is a write: a gh command or a git subcommand outside the state dir that is not on a read
# list is refused, never let through. The case table is `rehearse.sh --selftest`.
#
# SOURCED, so option-neutral: no `set` here (CLAUDE.md, scripts/check_sourced_libs_option_neutral.sh).

# wg_inside STATE PATH -> rc 0 when PATH resolves to STATE or below it
wg_inside() {
    local s p
    s=$(realpath -m -- "$1") || return 1
    p=$(realpath -m -- "$2") || return 1
    [ "$p" = "$s" ] || [ "${p#"$s"/}" != "$p" ]
}

# wg_git_read SUB [ARG...] -> rc 0 when the git subcommand only reads the repository
wg_git_read() {
    local sub=$1 a
    shift
    case $sub in
        status|log|show|rev-parse|rev-list|diff|diff-tree|diff-index|diff-files|ls-files|ls-tree|ls-remote \
        |cat-file|merge-base|describe|for-each-ref|show-ref|name-rev|blame|grep|shortlog|cherry|var|version \
        |help|check-ignore|check-attr|check-ref-format|count-objects|verify-commit|verify-tag|whatchanged \
        |archive|get-tar-commit-id|show-branch|range-diff|--version|--help)
            return 0 ;;
        config)
            for a in "$@"; do
                case $a in --get|--get-all|--get-regexp|--get-urlmatch|-l|--list|--show-origin|--show-scope) return 0 ;; esac
            done
            return 1 ;;
        branch)
            for a in "$@"; do
                case $a in --show-current|-l|--list) return 0 ;; -*) ;; *) return 1 ;; esac
            done
            return 0 ;;
        worktree|remote|stash|notes)
            [ $# -eq 0 ] && return 0
            case $1 in list|show|get-url|-v|--verbose) return 0 ;; esac
            return 1 ;;
        symbolic-ref)
            for a in "$@"; do case $a in -d|--delete) return 1 ;; esac; done
            local n=0
            for a in "$@"; do case $a in -*) ;; *) n=$((n + 1)) ;; esac; done
            [ "$n" -le 1 ] ;;
        *) return 1 ;;
    esac
}

# wg_git_tag_writes [ARG...] -> rc 0 when `git tag ARG...` creates, moves, signs or deletes a tag
wg_git_tag_writes() {
    local a skip=0 list=0 pos=0
    for a in "$@"; do
        if [ "$skip" = 1 ]; then skip=0; continue; fi
        case $a in
            -l|--list|-v|--verify) list=1 ;;
            -a|--annotate|-s|--sign|-f|--force|-d|--delete|-m*|--message*|-F*|--file*|-u*|--local-user*|-e|--edit) return 0 ;;
            --contains|--no-contains|--points-at|--merged|--no-merged|--sort|--format|--color|--column) skip=1 ;;
            -*) ;;
            *) pos=1 ;;
        esac
    done
    [ "$list" = 1 ] && return 1
    [ "$pos" = 1 ]
}

# wg_git STATE CWD [ARG...] -> "READ" | "WRITE <why>"
wg_git() {
    local state=$1 dir=$2 gitdir="" sub="" a
    shift 2
    while [ $# -gt 0 ]; do
        a=$1
        case $a in
            -C) dir=$(cd "$dir" 2>/dev/null && cd "${2:-.}" 2>/dev/null && pwd) || dir="${2:-?}"; shift 2 ;;
            -c|--namespace|--exec-path) shift 2 ;;
            --git-dir) gitdir=${2:-}; shift 2 ;;
            --git-dir=*) gitdir=${a#--git-dir=}; shift ;;
            --work-tree) shift 2 ;;
            -*) shift ;;
            *) sub=$a; shift; break ;;
        esac
    done
    case $sub in
        "") echo READ; return ;;
        push|send-email|request-pull|svn|p4) echo "WRITE git $sub"; return ;;
        tag) if wg_git_tag_writes "$@"; then echo "WRITE git tag (a tag is a release write, H10)"; else echo READ; fi; return ;;
        config)
            for a in "$@"; do
                case $a in --global|--system)
                    if wg_git_read config "$@"; then echo READ; else echo "WRITE git config $a (outside the state dir)"; fi
                    return ;;
                esac
            done ;;
    esac
    if wg_git_read "$sub" "$@"; then echo READ; return; fi
    [ -n "$gitdir" ] && case $gitdir in /*) dir=$gitdir ;; *) dir=$dir/$gitdir ;; esac
    if wg_inside "$state" "$dir"; then echo READ; return; fi
    echo "WRITE git $sub in $dir (outside the state dir)"
}

# wg_gh_api [ARG...] -> "READ" | "WRITE <why>" for `gh api ARG...`
wg_gh_api() {
    local method="" body=0 path="" a skip=0 mut=0
    for a in "$@"; do
        if [ "$skip" = m ]; then method=$a; skip=0; continue; fi
        if [ "$skip" = 1 ]; then skip=0; case $a in *mutation*) mut=1 ;; esac; continue; fi
        case $a in
            -X|--method) skip=m ;;
            -X*) method=${a#-X} ;;
            --method=*) method=${a#--method=} ;;
            -f|-F|--field|--raw-field|--input) body=1; skip=1 ;;
            -f*|-F*|--field=*|--raw-field=*|--input=*) body=1; case $a in *mutation*) mut=1 ;; esac ;;
            -H|--header|-q|--jq|-t|--template|--cache|--hostname|-p|--preview) skip=1 ;;
            -*) ;;
            *) [ -z "$path" ] && path=$a ;;
        esac
    done
    if [ "$path" = graphql ]; then
        if [ "$mut" = 1 ]; then echo "WRITE gh api graphql mutation"; else echo READ; fi
        return
    fi
    [ -z "$method" ] && { if [ "$body" = 1 ]; then method=POST; else method=GET; fi; }
    method=$(printf '%s' "$method" | tr '[:lower:]' '[:upper:]')
    case $method in GET|HEAD) echo READ ;; *) echo "WRITE gh api $method ${path:-?}" ;; esac
}

# wg_gh [ARG...] -> "READ" | "WRITE <why>". A command not named here is a write (fail closed).
wg_gh() {
    local cmd=${1:-} sub=${2:-}
    case $cmd in
        api) shift; wg_gh_api "$@"; return ;;
        ""|--version|version|help|--help|status|search|browse|completion) echo READ; return ;;
    esac
    case $cmd:$sub in
        release:view|release:list|release:download|run:view|run:list|run:download|run:watch \
        |workflow:view|workflow:list|pr:view|pr:list|pr:status|pr:checks|pr:diff \
        |issue:view|issue:list|issue:status|repo:view|repo:list|repo:clone|label:list \
        |secret:list|variable:list|variable:get|cache:list|auth:status|auth:token|config:get|config:list \
        |ruleset:list|ruleset:view|ruleset:check|project:list|project:view|gist:list|gist:view)
            echo READ ;;
        *) echo "WRITE gh $cmd${sub:+ $sub}" ;;
    esac
}

# wg_cargo STATE [ARG...] -> "READ" | "WRITE <why>"
wg_cargo() {
    local state=$1 sub="" a root=""
    shift
    while [ $# -gt 0 ]; do
        case $1 in
            +*|-q|--quiet|-v|-vv|--verbose|--locked|--frozen|--offline) shift ;;
            -Z|--config|-C|--color) shift 2 ;;
            -*) shift ;;
            *) sub=$1; shift; break ;;
        esac
    done
    case $sub in
        publish)
            for a in "$@"; do case $a in --dry-run|-n) echo READ; return ;; esac; done
            echo "WRITE cargo publish (no --dry-run)" ;;
        yank|owner|login|logout) echo "WRITE cargo $sub" ;;
        install)
            local skip=0
            for a in "$@"; do
                if [ "$skip" = 1 ]; then root=$a; skip=0; continue; fi
                case $a in --root) skip=1 ;; --root=*) root=${a#--root=} ;; esac
            done
            [ -z "$root" ] && root=${CARGO_INSTALL_ROOT:-${CARGO_HOME:-$HOME/.cargo}}
            if wg_inside "$state" "$root"; then echo READ; else echo "WRITE cargo install into $root (outside the state dir)"; fi ;;
        *) echo READ ;;
    esac
}

# wg_curl [ARG...] -> "READ" | "WRITE <why>": a body, an upload or a non-GET method writes
wg_curl() {
    local a skip=0
    for a in "$@"; do
        if [ "$skip" = 1 ]; then
            skip=0
            case $(printf '%s' "$a" | tr '[:lower:]' '[:upper:]') in GET|HEAD) ;; *) echo "WRITE curl -X $a"; return ;; esac
            continue
        fi
        case $a in
            -X|--request) skip=1 ;;
            -X*) case $(printf '%s' "${a#-X}" | tr '[:lower:]' '[:upper:]') in GET|HEAD) ;; *) echo "WRITE curl $a"; return ;; esac ;;
            --request=*) case $(printf '%s' "${a#--request=}" | tr '[:lower:]' '[:upper:]') in GET|HEAD) ;; *) echo "WRITE curl $a"; return ;; esac ;;
            -d|-d?*|--data*|--json*|-F|-F?*|--form*|-T|-T?*|--upload-file*) echo "WRITE curl ${a%%=*} (a body or an upload)"; return ;;
        esac
    done
    echo READ
}

# wg_classify TOOL STATE CWD [ARG...] -> prints "READ" or "WRITE <why>"; rc 0 READ, 1 WRITE
wg_classify() {
    local tool=$1 state=$2 cwd=$3 v
    shift 3
    case $tool in
        git) v=$(wg_git "$state" "$cwd" "$@") ;;
        gh) v=$(wg_gh "$@") ;;
        cargo) v=$(wg_cargo "$state" "$@") ;;
        curl) v=$(wg_curl "$@") ;;
        ssh|scp|sftp) v="WRITE $tool (a remote host is outside the state dir)" ;;
        *) v="WRITE $tool (not a guarded tool)" ;;
    esac
    printf '%s\n' "$v"
    [ "$v" = READ ]
}

# wg_stub TOOL [ARG...] -- the body of every stub. Needs WG_STATE, WG_CALLS and WG_REAL_<TOOL>.
wg_stub() {
    local tool=$1 v real caller argv var why
    shift
    var="WG_REAL_$(printf '%s' "$tool" | tr '[:lower:]' '[:upper:]')"
    real=${!var:-}
    v=$(wg_classify "$tool" "${WG_STATE:?}" "$PWD" "$@")
    caller=$(ps -o args= -p "$PPID" 2>/dev/null | cut -c1-160)
    argv=$(printf '%q ' "$@")
    case $v in
        READ) [ -x "$real" ] || v="MISSING no real $tool" ;;
        *) v="WRITE${v#WRITE}" ;;
    esac
    why=${v#* }; [ "$why" = "$v" ] && why=-
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${WG_STAGE:--}" "$tool" "${v%% *}" "$why" "${caller//$'\t'/ }" "${argv:0:400}" >> "${WG_CALLS:?}"
    case $v in
        READ) exec "$real" "$@" ;;
        MISSING*) printf 'rehearsal guard: %s\n' "$v" >&2; exit 127 ;;
    esac
    printf 'REHEARSAL WRITE REFUSED (H10): %s\n' "${v#WRITE }" >&2
    exit 97
}
