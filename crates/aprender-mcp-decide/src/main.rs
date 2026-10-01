//! Local runner for the thin decision MCP server — stdio transport.
//!
//! Stdio is the transport MCP clients (Claude Desktop, Claude Code, Cursor) spawn
//! directly, and what the E2E test drives. The pmcp.run deployment does NOT run this
//! binary — it runs `aprender-mcp-decide-lambda`, which fetches the `.apr` from S3 and
//! serves streamable-http. This runner reads a LOCAL path only. Everything
//! human-readable goes to stderr: stdout belongs to the protocol.
//!
//! ```bash
//! aprender-mcp-decide --model models/decide/tweet-stance-16.apr
//! APRENDER_DECIDE_MODEL=... aprender-mcp-decide
//! ```

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

/// Server name advertised in `initialize`.
const SERVER_NAME: &str = "aprender-decide-classify";

/// The model-path environment fallback.
const ENV_MODEL: &str = "APRENDER_DECIDE_MODEL";

/// The runner's argv, declared (clap derive) rather than hand-parsed: an unknown flag or a
/// `--model` without a value is a usage error (exit 2), never silently ignored.
#[derive(Debug, clap::Parser)]
#[command(
    name = "aprender-mcp-decide",
    version,
    about = "Thin single-model MCP server (stdio): one task-bound `classify` tool over a \
             verified decide-apr-v1 artifact"
)]
struct Cli {
    /// The decide-apr-v1 `.apr` to serve. Falls back to $APRENDER_DECIDE_MODEL.
    #[arg(long, value_name = "FILE")]
    model: Option<PathBuf>,
}

/// `--model`, else the environment fallback, else the "no model" usage error.
fn model_path(cli: Cli) -> Result<PathBuf, String> {
    cli.model
        .or_else(|| std::env::var_os(ENV_MODEL).map(PathBuf::from))
        .ok_or_else(|| {
            format!("no model: pass --model <FILE> or set {ENV_MODEL} to a decide-apr-v1 artifact")
        })
}

#[tokio::main]
async fn main() -> ExitCode {
    // clap prints its own usage error and exits 2 (0 for --help / --version).
    let cli = <Cli as clap::Parser>::parse();
    let model_path = match model_path(cli) {
        Ok(path) => path,
        Err(message) => {
            eprintln!("error: {message}");
            return ExitCode::from(2);
        }
    };

    let started = Instant::now();
    let model = match aprender_mcp_decide::load_model_from_path(&model_path) {
        Ok(model) => Arc::new(model),
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };
    let id = model.identity();
    eprintln!(
        "{SERVER_NAME}: model loaded from {} in {} ms (artifact_sha256 {}, recipe_id {}, base {}) \
         — serving `{}` on stdio",
        model_path.display(),
        started.elapsed().as_millis(),
        id.artifact_sha256,
        id.recipe_id,
        id.base,
        aprender_mcp_decide::TOOL_NAME
    );

    let server =
        match aprender_mcp_decide::build_server(model, SERVER_NAME, env!("CARGO_PKG_VERSION")) {
            Ok(server) => server,
            Err(error) => {
                eprintln!("error: server construction refused: {error}");
                return ExitCode::FAILURE;
            }
        };

    if let Err(error) = server.run_stdio().await {
        eprintln!("error: stdio server terminated: {error}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
