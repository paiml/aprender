//! Local runner for the thin SetFit MCP server — stdio transport.
//!
//! Stdio is the transport MCP clients (Claude Desktop, Claude Code, Cursor)
//! spawn directly, and what the E2E test drives. The pmcp.run deployment does
//! NOT run this binary — it runs the Lambda loopback wrapper, which embeds the
//! model and serves streamable-http. Everything human-readable goes to stderr:
//! stdout belongs to the protocol.
//!
//! ```bash
//! aprender-mcp-setfit --model models/setfit-abortion-s17x8.apr
//! APRENDER_SETFIT_MODEL=... aprender-mcp-setfit
//! ```

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

/// Server name advertised in `initialize`.
const SERVER_NAME: &str = "aprender-setfit-predict";

fn model_path_from(mut args: std::env::Args) -> Result<PathBuf, String> {
    let mut model: Option<PathBuf> = None;
    let _argv0 = args.next();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => {
                let value = args
                    .next()
                    .ok_or_else(|| String::from("--model requires a path"))?;
                model = Some(PathBuf::from(value));
            }
            other => {
                return Err(format!(
                    "unknown argument {other}; usage: aprender-mcp-setfit --model <FILE>"
                ));
            }
        }
    }
    model
        .or_else(|| std::env::var_os("APRENDER_SETFIT_MODEL").map(PathBuf::from))
        .ok_or_else(|| {
            String::from(
                "no model: pass --model <FILE> or set APRENDER_SETFIT_MODEL to a \
                 setfit-apr-v1 artifact",
            )
        })
}

#[tokio::main]
async fn main() -> ExitCode {
    let model_path = match model_path_from(std::env::args()) {
        Ok(path) => path,
        Err(message) => {
            eprintln!("error: {message}");
            return ExitCode::from(2);
        }
    };

    let model = match aprender_mcp_setfit::load_model_from_path(&model_path) {
        Ok(model) => Arc::new(model),
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "{SERVER_NAME}: model loaded from {} — serving `{}` on stdio",
        model_path.display(),
        aprender_mcp_setfit::TOOL_NAME
    );

    let server =
        match aprender_mcp_setfit::build_server(model, SERVER_NAME, env!("CARGO_PKG_VERSION")) {
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
