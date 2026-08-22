# workflow_env_scan.awk -- the scanner behind check_workflow_env_defined.sh.
#
# Reads ONE GitHub Actions workflow and prints a row for every `$VAR` a `run:`
# block interpolates that its job does not define. Exits 1 if it printed any.
#
#   awk -v src_root=<repo root> -v wf=<label> -f workflow_env_scan.awk <file.yml>
#
# `src_root` resolves the relative paths a run: block `source`s, so a variable a
# sourced library exports counts as defined.
#
# It lives in its own file rather than inside the guard's single quotes because
# bashrs lints scripts/*.sh and parses an embedded awk program as shell: this
# program produced 122 diagnostics (90 of them SC1028) purely as a quoting
# artifact, against a shrink-only repo-wide error ratchet.


    function indent(s,   i) { i = match(s, /[^ ]/); return (i == 0) ? -1 : i - 1 }

    # Record every NAME= / export NAME= / local NAME= assignment on a line,
    # wherever it sits: a `case` branch puts one after `)`, a pipeline after
    # `|`. Anchoring to start-of-line missed STRIP= in binary-release.yml.
    function harvest_assignments(s, scope,   rest, pre, name) {
        rest = s
        while (match(rest, /[A-Za-z_][A-Za-z0-9_]*\+?=/)) {
            name = substr(rest, RSTART, RLENGTH)
            sub(/\+?=$/, "", name)
            pre = (RSTART == 1) ? "" : substr(rest, RSTART - 1, 1)
            # A token start, not the tail of `--flag=x` or `$FOO=`.
            if (RSTART == 1 || pre ~ /[ \t;&|(){}]/) defined[scope "\t" name] = 1
            rest = substr(rest, RSTART + RLENGTH)
        }
        # `for NAME in ...`
        if (match(s, /(^|[ \t;])for[ \t]+[A-Za-z_][A-Za-z0-9_]*[ \t]+in([ \t]|$)/)) {
            name = substr(s, RSTART, RLENGTH)
            sub(/^[ \t;]*for[ \t]+/, "", name); sub(/[ \t]+in[ \t]*$/, "", name)
            defined[scope "\t" name] = 1
        }
        # `read [-r] [-d x] NAME...`, with or without a leading `while IFS= `.
        if (match(s, /(^|[ \t;|])read[ \t]+/)) {
            rest = substr(s, RSTART + RLENGTH)
            n = split(rest, parts, /[ \t]+/)
            for (i = 1; i <= n; i++) {
                tok = parts[i]
                sub(/[;&|<>)].*$/, "", tok)      # `read -r line; do` -> `line`
                if (tok == "") break
                if (tok ~ /^-/) continue
                if (tok ~ /^[A-Za-z_][A-Za-z0-9_]*$/) defined[scope "\t" tok] = 1
                else break
            }
        }
    }

    # One physical line of shell inside a run: block.
    function process_run_line(body, scope,   scan, rest, v) {
        if (skip_block) return
        harvest_assignments(body, scope)
        harvest_source(body, scope)
        # `${{ ... }}` is a GitHub expression, substituted before bash ever sees
        # the line. Blank it out so its innards are not read as shell variables.
        scan = body
        while (match(scan, /\$\{\{[^}]*\}\}/)) {
            scan = substr(scan, 1, RSTART - 1) "X" substr(scan, RSTART + RLENGTH)
        }
        rest = scan
        while (match(rest, /\$\{[A-Za-z_][A-Za-z0-9_]*[}:#%\/]/)) {
            v = substr(rest, RSTART + 2, RLENGTH - 3)
            refs[++nref] = scope "\t" v "\t" NR
            rest = substr(rest, RSTART + RLENGTH)
        }
        rest = scan
        while (match(rest, /\$[A-Za-z_][A-Za-z0-9_]*/)) {
            v = substr(rest, RSTART + 1, RLENGTH - 1)
            refs[++nref] = scope "\t" v "\t" NR
            rest = substr(rest, RSTART + RLENGTH)
        }
    }

    # `. path` / `source path` -- take what the file exports as defined here.
    function harvest_source(s, scope,   rest, path, l) {
        if (!match(s, /(^|[ \t;&|])(\.|source)[ \t]+[^ \t;&|)]+/)) return
        rest = substr(s, RSTART, RLENGTH)
        sub(/^[ \t;&|]*(\.|source)[ \t]+/, "", rest)
        gsub(/["']/, "", rest)
        if (rest ~ /\$/) return          # dynamic path: cannot resolve, do not guess
        path = (rest ~ /^\//) ? rest : src_root "/" rest
        while ((getline l < path) > 0) harvest_assignments(l, scope)
        close(path)
    }

    BEGIN { job = ""; in_jobs = 0; env_ind = -1; run_ind = -1; step_shell = "" }
    {
        line = $0; sub(/\r$/, "", line)
        if (line ~ /^[ \t]*$/) next
        ind = indent(line)
        body = line; sub(/^[ ]*/, "", body)

        # ---- inside a run: block scalar ----------------------------------
        if (run_ind >= 0) {
            if (ind > run_ind) { process_run_line(body, job); next }
            run_ind = -1
        }

        # ---- inside an env: mapping --------------------------------------
        if (env_ind >= 0) {
            if (ind > env_ind) {
                if (match(body, /^[A-Za-z_][A-Za-z0-9_]*[ ]*:/)) {
                    v = substr(body, RSTART, RLENGTH); sub(/[ ]*:$/, "", v)
                    defined[env_scope "\t" v] = 1
                    if (env_scope == "\001WF") wf_env[v] = 1
                }
                next
            }
            env_ind = -1
        }

        if (ind == 0 && body ~ /^jobs[ ]*:/) { in_jobs = 1; job = ""; next }
        if (ind == 0 && body ~ /^env[ ]*:/)  { env_ind = 0; env_scope = "\001WF"; next }
        if (ind == 0) { in_jobs = 0; next }
        if (!in_jobs) next

        if (ind == 2 && match(body, /^[A-Za-z0-9_-]+[ ]*:[ ]*$/)) {
            job = substr(body, 1, index(body, ":") - 1); sub(/[ ]+$/, "", job)
            step_shell = ""
            next
        }
        if (job == "") next

        # A new step resets the declared shell.
        if (body ~ /^-[ ]+/) step_shell = ""
        if (match(body, /^-?[ ]*shell[ ]*:[ ]*/)) {
            step_shell = substr(body, RSTART + RLENGTH)
            gsub(/["']/, "", step_shell)
        }
        if (body ~ /^env[ ]*:/)      { env_ind = ind;     env_scope = job; next }
        if (body ~ /^-[ ]+env[ ]*:/) { env_ind = ind + 2; env_scope = job; next }
        if (match(body, /^-?[ ]*run[ ]*:/)) {
            run_ind = ind
            # pwsh/powershell/python/cmd have their own variable scoping; a
            # `$archive` there is not a shell variable at all.
            skip_block = (step_shell != "" && step_shell !~ /^(bash|sh)([ \t]|$)/)
            # `run: echo "$X"` puts the whole command on this line. Scanning
            # only the indented continuation missed every single-line step --
            # which is most of them.
            inline = substr(body, RSTART + RLENGTH)
            sub(/^[ \t]*/, "", inline)
            if (inline != "" && inline !~ /^[|>][-+0-9]*[ \t]*(#.*)?$/) {
                process_run_line(inline, job)
            }
            next
        }
    }

    END {
        allow = "^(GITHUB_|RUNNER_|ACTIONS_|INPUT_|CI$|HOME$|PATH$|PWD$|OLDPWD$|USER$|LOGNAME$|SHELL$|TERM$|TMPDIR$|LANG$|LC_|HOSTNAME$|IFS$|PIPESTATUS$|BASH|FUNCNAME$|LINENO$|RANDOM$|SECONDS$|REPLY$|OSTYPE$|UID$|EUID$|PS[0-9]$)"
        bad = 0
        for (k = 1; k <= nref; k++) {
            split(refs[k], p, "\t"); j = p[1]; v = p[2]; ln = p[3]
            if (v ~ allow)        continue
            if (wf_env[v])        continue
            if (defined[j "\t" v]) continue
            printf "%s:%s: job `%s` interpolates $%s, which nothing in that job defines\n", wf, ln, j, v
            bad++
        }
        exit (bad > 0) ? 1 : 0
    }
