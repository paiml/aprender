//! Oracle subcommand dispatch for the batuta CLI.
//!
//! Handles the many Oracle flags: RAG, PMAT query, cookbook, local
//! workspace, Coursera assets, and classic oracle queries.

use tracing::info;

use crate::cli;
use crate::main_oracle_args::OracleArgs;

/// RAG-specific args extracted from `OracleArgs` for dispatch.
struct RagDispatchArgs<'a> {
    query: &'a Option<String>,
    rag: bool,
    rag_index: bool,
    rag_index_force: bool,
    rag_stats: bool,
    rag_profile: bool,
    rag_trace: bool,
    answer: bool,
    answer_model: &'a str,
    #[cfg(feature = "native")]
    rag_dashboard: bool,
    format: cli::oracle::OracleOutputFormat,
}

/// Try dispatching an Oracle RAG subcommand.
fn try_oracle_rag(args: &RagDispatchArgs<'_>) -> Option<anyhow::Result<()>> {
    #[cfg(feature = "native")]
    if args.rag_dashboard {
        return Some(cli::oracle::cmd_oracle_rag_dashboard());
    }
    if args.rag_stats {
        return Some(cli::oracle::cmd_oracle_rag_stats(args.format));
    }
    if args.rag_index || args.rag_index_force {
        return Some(cli::oracle::cmd_oracle_rag_index(args.rag_index_force));
    }
    if args.answer {
        return Some(cli::oracle::cmd_oracle_rag_answer(
            args.query.clone(),
            args.answer_model,
            args.format,
        ));
    }
    if args.rag {
        return Some(cli::oracle::cmd_oracle_rag_with_profile(
            args.query.clone(),
            args.format,
            args.rag_profile,
            args.rag_trace,
        ));
    }
    None
}

/// Try dispatching a specialized Oracle subcommand (local/RAG/pmat-query/cookbook).
/// Returns `Some(result)` if a subcommand matched, `None` for default classic oracle.
#[allow(clippy::too_many_arguments)]
fn try_oracle_subcommand(
    query: &Option<String>,
    local: bool,
    dirty: bool,
    publish_order: bool,
    rag: bool,
    rag_index: bool,
    rag_index_force: bool,
    rag_stats: bool,
    rag_profile: bool,
    rag_trace: bool,
    answer: bool,
    answer_model: &str,
    #[cfg(feature = "native")] rag_dashboard: bool,
    pmat_query: bool,
    pmat_project_path: &Option<String>,
    pmat_limit: usize,
    pmat_min_grade: &Option<String>,
    pmat_max_complexity: Option<u32>,
    pmat_include_source: bool,
    pmat_all_local: bool,
    cookbook: bool,
    recipe: &Option<String>,
    recipes_by_tag: &Option<String>,
    recipes_by_component: &Option<String>,
    search_recipes: &Option<String>,
    format: cli::oracle::OracleOutputFormat,
) -> Option<anyhow::Result<()>> {
    if local || dirty || publish_order {
        return Some(cli::oracle::cmd_oracle_local(local, dirty, publish_order, format));
    }

    let rag_result = try_oracle_rag(&RagDispatchArgs {
        query,
        rag,
        rag_index,
        rag_index_force,
        rag_stats,
        rag_profile,
        rag_trace,
        answer,
        answer_model,
        #[cfg(feature = "native")]
        rag_dashboard,
        format,
    });
    if rag_result.is_some() {
        return rag_result;
    }

    if pmat_query {
        return Some(cli::oracle::cmd_oracle_pmat_query(
            query.clone(),
            pmat_project_path.clone(),
            pmat_limit,
            pmat_min_grade.clone(),
            pmat_max_complexity,
            pmat_include_source,
            rag,
            pmat_all_local,
            format,
        ));
    }

    if cookbook
        || recipe.is_some()
        || recipes_by_tag.is_some()
        || recipes_by_component.is_some()
        || search_recipes.is_some()
    {
        return Some(cli::oracle::cmd_oracle_cookbook(
            cookbook,
            recipe.clone(),
            recipes_by_tag.clone(),
            recipes_by_component.clone(),
            search_recipes.clone(),
            format,
        ));
    }
    None
}

/// Handle Oracle subcommand dispatch from flattened `OracleArgs`.
pub(crate) fn dispatch_oracle(args: OracleArgs) -> anyhow::Result<()> {
    info!("Oracle Mode");

    let OracleArgs {
        query,
        recommend,
        problem,
        data_size,
        integrate,
        capabilities,
        list,
        show,
        interactive,
        rag,
        rag_index,
        rag_index_force,
        rag_stats,
        rag_profile,
        rag_trace,
        answer,
        answer_model,
        #[cfg(feature = "native")]
        rag_dashboard,
        cookbook,
        recipe,
        recipes_by_tag,
        recipes_by_component,
        search_recipes,
        local,
        dirty,
        publish_order,
        pmat_query,
        pmat_project_path,
        pmat_limit,
        pmat_min_grade,
        pmat_max_complexity,
        pmat_include_source,
        pmat_all_local,
        asset,
        transcript,
        output,
        topic,
        course_title,
        arxiv,
        arxiv_live,
        arxiv_max,
        format,
    } = args;

    // Coursera asset generation (checked before subcommands)
    if let Some(asset_type) = asset {
        let transcript_path = transcript
            .ok_or_else(|| anyhow::anyhow!("--transcript <path> is required for --asset"))?;
        return cli::oracle::cmd_oracle_asset(
            asset_type,
            transcript_path,
            output,
            topic,
            course_title,
            format,
        );
    }

    if let Some(result) = try_oracle_subcommand(
        &query,
        local,
        dirty,
        publish_order,
        rag,
        rag_index,
        rag_index_force,
        rag_stats,
        rag_profile,
        rag_trace,
        answer,
        &answer_model,
        #[cfg(feature = "native")]
        rag_dashboard,
        pmat_query,
        &pmat_project_path,
        pmat_limit,
        &pmat_min_grade,
        pmat_max_complexity,
        pmat_include_source,
        pmat_all_local,
        cookbook,
        &recipe,
        &recipes_by_tag,
        &recipes_by_component,
        &search_recipes,
        format,
    ) {
        return result;
    }

    cli::oracle::cmd_oracle(cli::oracle::OracleOptions {
        query,
        recommend,
        problem,
        data_size,
        integrate,
        capabilities,
        list,
        show,
        interactive,
        arxiv,
        arxiv_live,
        arxiv_max,
        format,
    })
}
