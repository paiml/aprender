
/// Output filtered suggestions.
fn output_suggestions(suggestions: Vec<(String, f32)>) {
    let filtered = filter_sensitive_suggestions(suggestions);
    for (suggestion, score) in filtered {
        println!("{}\t{:.3}", suggestion, score);
    }
}

/// Handle paged model suggestion.
fn suggest_paged(path: &std::path::Path, prefix: &str, count: usize, mem_mb: usize) {
    let paged_path = path.with_extension("apbundle");
    let mut model = match PagedMarkovModel::load(&paged_path, mem_mb) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("# aprender: {e}");
            return;
        }
    };
    output_suggestions(model.suggest(prefix, count));
}

/// Handle standard model suggestion.
fn suggest_standard(path: &std::path::Path, prefix: &str, count: usize, password: Option<&str>) {
    let model = if let Some(pwd) = password {
        match MarkovModel::load_encrypted(path, pwd) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("# aprender: {e}");
                return;
            }
        }
    } else {
        match load_model_graceful(path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{e}");
                return;
            }
        }
    };
    output_suggestions(model.suggest(prefix, count));
}

fn cmd_suggest(
    prefix: &str,
    model_path: &str,
    count: usize,
    memory_limit: Option<usize>,
    use_password: bool,
) {
    // Phase 1: Input validation (Poka-yoke)
    let validated_prefix = match sanitize_prefix(prefix) {
        Ok(p) => p,
        Err(ShellError::InvalidInput { message }) => {
            eprintln!("# aprender: {message}");
            return;
        }
        Err(_) => return,
    };

    let path = expand_path(model_path);
    let password = get_password_or_prompt(use_password, "# aprender: ");

    if let Some(mem_mb) = memory_limit {
        suggest_paged(&path, &validated_prefix, count, mem_mb);
    } else {
        suggest_standard(&path, &validated_prefix, count, password.as_deref());
    }
}

/// Print paged model error hints.
fn print_paged_model_error_hint(e: &std::io::Error, paged_path: &std::path::Path, mem_mb: usize) {
    let err_str = e.to_string();
    if err_str.contains("Checksum") || err_str.contains("corrupt") {
        eprintln!("   Hint: The model may be corrupted. Run 'aprender-shell train --memory-limit {mem_mb}' to rebuild.");
    } else if !paged_path.exists() {
        eprintln!("   Hint: Model file not found. Train a model first with 'aprender-shell train --memory-limit {mem_mb}'");
    }
}

/// Print standard model error hints.
fn print_standard_model_error_hint(e: &std::io::Error, path: &std::path::Path) {
    let err_str = e.to_string();
    if err_str.contains("Checksum mismatch") {
        eprintln!(
            "   Hint: The model file may be corrupted. Run 'aprender-shell train' to rebuild."
        );
    } else if !path.exists() {
        eprintln!("   Hint: Model file not found. Train a model first with 'aprender-shell train'");
    } else if MarkovModel::is_encrypted(path).unwrap_or(false) {
        eprintln!("   Hint: This model is encrypted. Use --password flag.");
    }
}

/// Print paging statistics if available.
fn print_paging_stats(model: &PagedMarkovModel) {
    if let Some(paging_stats) = model.paging_stats() {
        println!("\n📈 Paging Statistics:");
        println!("   Page hits:       {}", paging_stats.hits);
        println!("   Page misses:     {}", paging_stats.misses);
        println!("   Evictions:       {}", paging_stats.evictions);
        let total = paging_stats.hits + paging_stats.misses;
        if total > 0 {
            let hit_rate = paging_stats.hits as f64 / total as f64 * 100.0;
            println!("   Hit rate:        {:.1}%", hit_rate);
        }
    }
}

/// Print top commands from a model.
fn print_top_commands(commands: Vec<(String, u32)>) {
    println!("\n🔝 Top commands:");
    for (cmd, count) in commands {
        println!("   {:>6}x  {}", count, cmd);
    }
}

/// Handle paged model stats display.
fn stats_paged(path: &std::path::Path, mem_mb: usize) {
    let paged_path = path.with_extension("apbundle");
    let model = match PagedMarkovModel::load(&paged_path, mem_mb) {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "❌ Failed to load paged model '{}': {e}",
                paged_path.display()
            );
            print_paged_model_error_hint(&e, &paged_path, mem_mb);
            std::process::exit(1);
        }
    };

    let stats = model.stats();
    println!("📊 Paged Model Statistics:");
    println!("   N-gram size:     {}", stats.n);
    println!("   Total commands:  {}", stats.total_commands);
    println!("   Vocabulary size: {}", stats.vocab_size);
    println!("   Total segments:  {}", stats.total_segments);
    println!("   Loaded segments: {}", stats.loaded_segments);
    println!(
        "   Memory limit:    {:.1} MB",
        stats.memory_limit as f64 / 1024.0 / 1024.0
    );
    println!(
        "   Loaded bytes:    {:.1} KB",
        stats.loaded_bytes as f64 / 1024.0
    );

    print_paging_stats(&model);
    print_top_commands(model.top_commands(10));
}

/// Handle standard model stats display.
fn stats_standard(path: &std::path::Path, password: Option<&str>) {
    let model = if let Some(pwd) = password {
        MarkovModel::load_encrypted(path, pwd).unwrap_or_else(|e| {
            eprintln!("❌ Failed to load encrypted model: {e}");
            std::process::exit(1);
        })
    } else {
        match MarkovModel::load(path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("❌ Failed to load model '{}': {e}", path.display());
                print_standard_model_error_hint(&e, path);
                std::process::exit(1);
            }
        }
    };

    let encrypted = MarkovModel::is_encrypted(path).unwrap_or(false);

    println!("📊 Model Statistics:");
    println!("   N-gram size: {}", model.ngram_size());
    println!("   Unique n-grams: {}", model.ngram_count());
    println!("   Vocabulary size: {}", model.vocab_size());
    println!(
        "   Model size: {:.1} KB",
        model.size_bytes() as f64 / 1024.0
    );
    if encrypted {
        println!("   🔒 Encryption: AES-256-GCM");
    }
    print_top_commands(model.top_commands(10));
}

fn cmd_stats(model_path: &str, memory_limit: Option<usize>, use_password: bool) {
    let path = expand_path(model_path);
    let password = get_password_or_prompt(use_password, "❌ ");

    if let Some(mem_mb) = memory_limit {
        stats_paged(&path, mem_mb);
    } else {
        stats_standard(&path, password.as_deref());
    }
}

fn cmd_export(model_path: &str, output: &PathBuf) {
    let path = expand_path(model_path);

    if !path.exists() {
        eprintln!("❌ Model file not found: {}", path.display());
        eprintln!("   Hint: Train a model first with 'aprender-shell train'");
        std::process::exit(1);
    }

    if let Err(e) = std::fs::copy(&path, output) {
        eprintln!("❌ Failed to export model: {e}");
        if e.kind() == std::io::ErrorKind::PermissionDenied {
            eprintln!(
                "   Hint: Check write permissions for '{}'",
                output.display()
            );
        } else if e.kind() == std::io::ErrorKind::NotFound {
            eprintln!("   Hint: Destination directory may not exist");
        }
        std::process::exit(1);
    }

    println!("✅ Model exported to: {}", output.display());
}

fn cmd_import(input: &PathBuf, output: &str) {
    if !input.exists() {
        eprintln!("❌ Input file not found: {}", input.display());
        std::process::exit(1);
    }

    let output_path = expand_path(output);

    if let Err(e) = std::fs::copy(input, &output_path) {
        eprintln!("❌ Failed to import model: {e}");
        if e.kind() == std::io::ErrorKind::PermissionDenied {
            eprintln!(
                "   Hint: Check write permissions for '{}'",
                output_path.display()
            );
        }
        std::process::exit(1);
    }

    println!("✅ Model imported to: {}", output_path.display());
}

fn cmd_zsh_widget() {
    print!(
        r#"# >>> aprender-shell widget >>>
# shellcheck shell=zsh disable=SC2154,SC2086,SC2089,SC2227,SC2201,SC2067
# aprender-shell ZSH widget v5 (with daemon support)
# This script is meant to be sourced, not executed directly
# Add this to your ~/.zshrc
# Toggle: export APRENDER_DISABLED=1 to disable
# Daemon: export APRENDER_USE_DAEMON=1 for sub-ms latency
# Uninstall: aprender-shell uninstall --zsh
# Hardened per: docs/specifications/aprender-shell-harden-plan.md
# Validated with bashrs lint

# Environment variables (set externally by user, defaults provided)
APRENDER_DISABLED="${{APRENDER_DISABLED:-0}}"
APRENDER_USE_DAEMON="${{APRENDER_USE_DAEMON:-0}}"
APRENDER_AUTO_DAEMON="${{APRENDER_AUTO_DAEMON:-0}}"
APRENDER_SOCKET="${{APRENDER_SOCKET:-/tmp/aprender-shell.sock}}"
# Debugging (issue #84)
APRENDER_TIMING="${{APRENDER_TIMING:-0}}"
APRENDER_TRACE="${{APRENDER_TRACE:-0}}"
APRENDER_TRACE_FILE="${{APRENDER_TRACE_FILE:-/tmp/aprender-trace.log}}"
# Renacer syscall tracing (issue #89)
APRENDER_RENACER="${{APRENDER_RENACER:-0}}"
APRENDER_RENACER_OPTS="${{APRENDER_RENACER_OPTS:--c --stats}}"
APRENDER_RENACER_LOG="${{APRENDER_RENACER_LOG:-/tmp/aprender-renacer.log}}"

# Check if daemon is running
_aprender_daemon_available() {{
    [[ -S "$APRENDER_SOCKET" ]] && nc -z -U "$APRENDER_SOCKET" 2>/dev/null
}}

# Get suggestion from daemon (sub-ms latency)
_aprender_suggest_daemon() {{
    local prefix="$1"
    echo "$prefix" | nc -U "$APRENDER_SOCKET" 2>/dev/null | head -1
}}

# Get suggestion via command (fallback, ~10ms)
_aprender_suggest_cmd() {{
    local prefix="$1"
    if [[ "$APRENDER_RENACER" == '1' ]] && command -v renacer &>/dev/null; then
        # Wrap with renacer for syscall tracing (issue #89)
        renacer $APRENDER_RENACER_OPTS -- aprender-shell suggest "$prefix" 2>>"$APRENDER_RENACER_LOG" | head -1 | cut -f1
    else
        timeout 0.1 aprender-shell suggest "$prefix" 2>/dev/null | head -1 | cut -f1
    fi
}}

_aprender_suggest() {{
    # Skip if disabled or buffer too short
    [[ "$APRENDER_DISABLED" == '1' ]] && return
    [[ "${{#BUFFER}}" -lt 2 ]] && {{ POSTDISPLAY=''; return; }}

    local suggestion start_ms end_ms elapsed_ms

    # Timing start (issue #84)
    [[ "$APRENDER_TIMING" == '1' ]] && start_ms=$(($(date +%s%N 2>/dev/null || echo 0)/1000000))

    # Use daemon if available and enabled, otherwise fall back to command
    if [[ "$APRENDER_USE_DAEMON" == '1' ]] && _aprender_daemon_available; then
        suggestion="$(_aprender_suggest_daemon "$BUFFER")"
    else
        suggestion="$(_aprender_suggest_cmd "$BUFFER")"
    fi

    # Timing end (issue #84)
    if [[ "$APRENDER_TIMING" == '1' ]]; then
        end_ms=$(($(date +%s%N 2>/dev/null || echo 0)/1000000))
        elapsed_ms=$((end_ms - start_ms))
        print -u2 "[aprender] ${{elapsed_ms}}ms: '$BUFFER' -> '$suggestion'"
    fi

    # Trace logging (issue #84)
    if [[ "$APRENDER_TRACE" == '1' ]]; then
        echo "$(date +%Y-%m-%dT%H:%M:%S) prefix='$BUFFER' suggestion='$suggestion'" >> "$APRENDER_TRACE_FILE"
    fi

    if [[ -n "$suggestion" && "$suggestion" != "$BUFFER" ]]; then
        local suffix="${{suggestion#"$BUFFER"}}"

        # Robust ANSI handling with terminal capability check
        # terminfo is a ZSH builtin associative array
        if [[ -n "$TERM" && "$TERM" != 'dumb' ]] && (( ${{+terminfo[colors]}} )) && (( ${{terminfo[colors]:-0}} >= 8 )); then
            # Use ZSH prompt expansion for portable color codes
            POSTDISPLAY="$(print -P " %F{{240}}${{suffix}}%f")"
        else
            # Fallback: no colors for unsupported terminals
            POSTDISPLAY=" ${{suffix}}"
        fi
    else
        POSTDISPLAY=''
    fi
}}

_aprender_accept() {{
    if [[ -n "$POSTDISPLAY" ]]; then
        # Strip color codes and leading space when accepting
        local clean_suffix
        clean_suffix="${{POSTDISPLAY# }}"
        # Remove ANSI escape sequences (both $'\e[...m' and %F/%f)
        clean_suffix="${{clean_suffix//\\e\[*m/}}"
        clean_suffix="${{clean_suffix//%F\{{*\}}/}}"
        clean_suffix="${{clean_suffix//%f/}}"

        BUFFER="${{BUFFER}}${{clean_suffix}}"
        POSTDISPLAY=''
        CURSOR="${{#BUFFER}}"
        zle redisplay
    else
        # No suggestion: fall back to default Tab completion (issue #83)
        zle expand-or-complete
    fi
}}

_aprender_accept_word() {{
    # Accept next word of suggestion only (issue #85)
    if [[ -n "$POSTDISPLAY" ]]; then
        local clean_suffix
        clean_suffix="${{POSTDISPLAY# }}"
        # Remove color codes
        clean_suffix="${{clean_suffix//\\e\[*m/}}"
        clean_suffix="${{clean_suffix//%F\{{*\}}/}}"
        clean_suffix="${{clean_suffix//%f/}}"

        # Extract next word (up to first space)
        local next_word="${{clean_suffix%% *}}"
        if [[ "$next_word" == "$clean_suffix" ]]; then
            # No space found - accept entire remaining suggestion
            BUFFER="${{BUFFER}}${{next_word}}"
            POSTDISPLAY=''
        else
            # Accept word + space, update suggestion
            BUFFER="${{BUFFER}}${{next_word}} "
            local remaining="${{clean_suffix#* }}"
            if [[ -n "$remaining" ]]; then
                POSTDISPLAY=" ${{remaining}}"
            else
                POSTDISPLAY=''
            fi
        fi
        CURSOR="${{#BUFFER}}"
    fi
    zle redisplay
}}

zle -N _aprender_suggest
zle -N _aprender_accept
zle -N _aprender_accept_word

# Trigger on each keystroke
autoload -Uz add-zle-hook-widget
add-zle-hook-widget line-pre-redraw _aprender_suggest

# Accept with Tab or Right Arrow (full suggestion)
bindkey '^I' _aprender_accept      # Tab
bindkey '^[[C' _aprender_accept    # Right arrow
# Accept next word only (issue #85)
bindkey '^[[1;5C' _aprender_accept_word  # Ctrl+Right

# Start daemon automatically if requested
if [[ "$APRENDER_AUTO_DAEMON" == '1' ]] && ! _aprender_daemon_available; then
    aprender-shell daemon &>/dev/null &
    disown
fi
# <<< aprender-shell widget <<<
"#
    );
}
