
fn cmd_bash_widget() {
    print!(
        r#"# >>> aprender-shell widget >>>
# aprender-shell Bash widget v1 (issue #82)
# Add this to your ~/.bashrc
# Toggle: export APRENDER_DISABLED=1 to disable
# Uninstall: aprender-shell uninstall --bash

# Environment variables
APRENDER_DISABLED="${{APRENDER_DISABLED:-0}}"
_APRENDER_SUGGESTION=""

_aprender_get_suggestion() {{
    local prefix="$1"
    timeout 0.1 aprender-shell suggest "$prefix" 2>/dev/null | head -1 | cut -f1
}}

_aprender_suggest() {{
    [[ "$APRENDER_DISABLED" == "1" ]] && return
    [[ "${{#READLINE_LINE}}" -lt 2 ]] && return

    local suggestion
    suggestion=$(_aprender_get_suggestion "$READLINE_LINE")

    if [[ -n "$suggestion" && "$suggestion" != "$READLINE_LINE" ]]; then
        _APRENDER_SUGGESTION="${{suggestion#$READLINE_LINE}}"
        # Display ghost text in gray (after cursor position)
        echo -ne "\e[90m${{_APRENDER_SUGGESTION}}\e[0m\e[${{#_APRENDER_SUGGESTION}}D"
    else
        _APRENDER_SUGGESTION=""
    fi
}}

_aprender_accept() {{
    if [[ -n "$_APRENDER_SUGGESTION" ]]; then
        READLINE_LINE="${{READLINE_LINE}}${{_APRENDER_SUGGESTION}}"
        READLINE_POINT=${{#READLINE_LINE}}
        _APRENDER_SUGGESTION=""
    fi
}}

_aprender_clear() {{
    _APRENDER_SUGGESTION=""
}}

# Bind to readline hooks
# Note: Bash readline integration is limited compared to ZSH
# This provides basic suggestion on Tab key
bind -x '"\C-i": _aprender_accept'  # Tab to accept
bind -x '"\e[C": _aprender_accept'  # Right arrow to accept

# Show suggestions after each command edit (Bash 4.0+)
if [[ "${{BASH_VERSINFO[0]}}" -ge 4 ]]; then
    PROMPT_COMMAND="_aprender_suggest; ${{PROMPT_COMMAND}}"
fi
# <<< aprender-shell widget <<<
"#
    );
}

fn cmd_fish_widget() {
    print!(
        r#"# >>> aprender-shell widget >>>
# aprender-shell Fish widget v1
# Add this to your ~/.config/fish/config.fish
# Toggle: set -gx APRENDER_DISABLED 1 to disable
# Uninstall: aprender-shell uninstall --fish

function __aprender_suggest --description "Get AI-powered command suggestions"
    # Skip if disabled
    if test "$APRENDER_DISABLED" = "1"
        return
    end

    set -l cmd (commandline -b)
    # Skip if command too short
    if test (string length "$cmd") -lt 2
        return
    end

    # Get suggestion (with timeout for responsiveness)
    set -l suggestion (timeout 0.1 aprender-shell suggest "$cmd" 2>/dev/null | head -1 | cut -f1)

    if test -n "$suggestion" -a "$suggestion" != "$cmd"
        echo "$suggestion"
    end
end

function __aprender_complete --description "Complete commands for aprender suggestions"
    set -l cmd (commandline -cp)
    if test (string length "$cmd") -ge 2
        aprender-shell suggest "$cmd" 2>/dev/null | while read -l line
            set -l suggestion (echo "$line" | cut -f1)
            set -l score (echo "$line" | cut -f2)
            printf "%s\t%s\n" "$suggestion" "score: $score"
        end
    end
end

# Register completions for all commands
complete -f -c '*' -a '(__aprender_complete)'

# Fish autosuggestion hook (runs on each keystroke)
function __aprender_autosuggest --on-event fish_preexec
    # Optional: Update model incrementally (runs on command execution)
    # aprender-shell update --quiet 2>/dev/null &
end
# <<< aprender-shell widget <<<
"#
    );
}

fn cmd_uninstall(zsh: bool, bash: bool, fish: bool, keep_model: bool, dry_run: bool) {
    let detect_all = !zsh && !bash && !fish;

    let home = match dirs::home_dir() {
        Some(h) => h,
        None => {
            eprintln!("❌ Could not determine home directory");
            eprintln!("   Hint: Set HOME environment variable");
            std::process::exit(1);
        }
    };

    let action = if dry_run { "Would remove" } else { "Removed" };
    let mut removed_any = false;

    // Uninstall from each shell config
    let shells = [
        (zsh, ".zshrc", "source ~/.zshrc"),
        (bash, ".bashrc", "source ~/.bashrc"),
        (
            fish,
            ".config/fish/config.fish",
            "source ~/.config/fish/config.fish",
        ),
    ];

    for (requested, config_path, _reload_cmd) in &shells {
        if *requested || detect_all {
            removed_any |= uninstall_shell_widget(&home, config_path, *requested, action, dry_run);
        }
    }

    // Remove model files
    if !keep_model {
        removed_any |= remove_model_files(&home, dry_run);
    }

    print_uninstall_summary(removed_any, dry_run, zsh, bash, fish, detect_all);
}

/// Uninstall widget from a shell config file.
fn uninstall_shell_widget(
    home: &std::path::Path,
    config_path: &str,
    explicitly_requested: bool,
    action: &str,
    dry_run: bool,
) -> bool {
    let config_file = home.join(config_path);

    if !config_file.exists() {
        if explicitly_requested {
            println!("ℹ {} does not exist", config_file.display());
        }
        return false;
    }

    match remove_widget_block(&config_file, dry_run) {
        Ok(true) => {
            println!("✓ {} widget from {}", action, config_file.display());
            true
        }
        Ok(false) => {
            if explicitly_requested {
                println!("ℹ No widget found in {}", config_file.display());
            }
            false
        }
        Err(e) => {
            eprintln!("✗ Error processing {}: {}", config_file.display(), e);
            false
        }
    }
}

/// Remove model and bundle files.
fn remove_model_files(home: &std::path::Path, dry_run: bool) -> bool {
    let mut removed = false;

    // Standard model file
    let model_file = home.join(".aprender-shell.model");
    if model_file.exists() {
        removed |= remove_single_file(&model_file, "model file", dry_run);
    }

    // Paged model bundle
    let bundle_file = home.join(".aprender-shell.apbundle");
    if bundle_file.exists() {
        removed |= remove_directory(&bundle_file, "model bundle", dry_run);
    }

    removed
}

/// Remove a single file with appropriate messaging.
fn remove_single_file(path: &std::path::Path, description: &str, dry_run: bool) -> bool {
    if dry_run {
        println!("✓ Would remove {} {}", description, path.display());
        return true;
    }

    match std::fs::remove_file(path) {
        Ok(()) => {
            println!("✓ Removed {} {}", description, path.display());
            true
        }
        Err(e) => {
            eprintln!("✗ Error removing {}: {}", description, e);
            false
        }
    }
}

/// Remove a directory with appropriate messaging.
fn remove_directory(path: &std::path::Path, description: &str, dry_run: bool) -> bool {
    if dry_run {
        println!("✓ Would remove {} {}", description, path.display());
        return true;
    }

    match std::fs::remove_dir_all(path) {
        Ok(()) => {
            println!("✓ Removed {} {}", description, path.display());
            true
        }
        Err(e) => {
            eprintln!("✗ Error removing {}: {}", description, e);
            false
        }
    }
}

/// Print summary after uninstall.
fn print_uninstall_summary(
    removed_any: bool,
    dry_run: bool,
    zsh: bool,
    bash: bool,
    fish: bool,
    detect_all: bool,
) {
    if !removed_any {
        println!("ℹ No aprender-shell installation found");
        return;
    }

    if dry_run {
        println!("\n💡 Run without --dry-run to apply changes");
        return;
    }

    println!("\n✅ Done! Restart your shell or run:");
    if zsh || detect_all {
        println!("   source ~/.zshrc");
    }
    if bash || detect_all {
        println!("   source ~/.bashrc");
    }
    if fish || detect_all {
        println!("   source ~/.config/fish/config.fish");
    }
}

/// Remove widget block between marker comments from a file
fn remove_widget_block(path: &std::path::Path, dry_run: bool) -> std::io::Result<bool> {
    let content = std::fs::read_to_string(path)?;

    let start_marker = "# >>> aprender-shell widget >>>";
    let end_marker = "# <<< aprender-shell widget <<<";

    if let Some(start_idx) = content.find(start_marker) {
        if let Some(end_idx) = content.find(end_marker) {
            let end_idx = end_idx + end_marker.len();

            // Find the newline after end marker
            let end_idx = content[end_idx..]
                .find('\n')
                .map(|i| end_idx + i + 1)
                .unwrap_or(end_idx);

            // Find newline before start marker (to remove blank line)
            let start_idx = if start_idx > 0 && content.as_bytes()[start_idx - 1] == b'\n' {
                start_idx - 1
            } else {
                start_idx
            };

            if !dry_run {
                let new_content = format!("{}{}", &content[..start_idx], &content[end_idx..]);
                std::fs::write(path, new_content)?;
            }
            return Ok(true);
        }
    }

    Ok(false)
}

fn cmd_validate(history_path: Option<PathBuf>, ngram: usize, ratio: f32) {
    validate_ngram(ngram);
    println!("🔬 aprender-shell: Model Validation\n");

    // Find and parse history with graceful error handling (QA 2.4, 8.3)
    let history_file = find_history_file_graceful(history_path);
    println!("📂 History file: {}", history_file.display());

    let commands = parse_history_graceful(&history_file);
    println!("📊 Total commands: {}", commands.len());
    println!("⚙️  N-gram size: {}", ngram);
    println!(
        "📈 Train/test split: {:.0}% / {:.0}%\n",
        ratio * 100.0,
        (1.0 - ratio) * 100.0
    );

    print!("🧪 Running holdout validation... ");
    let result = MarkovModel::validate(&commands, ngram, ratio);
    println!("done!\n");

    println!("═══════════════════════════════════════════");
    println!("           VALIDATION RESULTS              ");
    println!("═══════════════════════════════════════════");
    println!("  Training set:     {:>6} commands", result.train_size);
    println!("  Test set:         {:>6} commands", result.test_size);
    println!("  Evaluated:        {:>6} commands", result.evaluated);
    println!("───────────────────────────────────────────");
    // Use aprender's ranking metrics
    println!(
        "  Hit@1  (top 1):   {:>6.1}%",
        result.metrics.hit_at_1 * 100.0
    );
    println!(
        "  Hit@5  (top 5):   {:>6.1}%",
        result.metrics.hit_at_5 * 100.0
    );
    println!(
        "  Hit@10 (top 10):  {:>6.1}%",
        result.metrics.hit_at_10 * 100.0
    );
    println!("  MRR (Mean Recip): {:>6.3}", result.metrics.mrr);
    println!("═══════════════════════════════════════════");

    // Interpretation
    println!("\n📊 Interpretation:");
    if result.metrics.hit_at_5 >= 0.5 {
        println!("   ✅ Excellent: Model finds correct command in top 5 >50% of the time");
    } else if result.metrics.hit_at_5 >= 0.3 {
        println!("   ✓ Good: Model provides useful suggestions");
    } else {
        println!("   ⚠️  Consider more training data or adjusting n-gram size");
        println!("   💡 Try: aprender-shell augment --count 5000");
    }
}
