//! #3979: a router that ADVERTISES exactly what it MOUNTS.
//!
//! The APR-CPU fallback router answered `GET /` with a hand-written sentence naming four
//! routes, and its 404 body with a hand-written list of six, while `add_ollama_stubs`
//! mounted five more that neither named. The SafeTensors routers served no `GET /` at
//! all. The GGUF router (`realizar::api::router`) already derives its index from the
//! table it mounts, so the route surface a client could discover depended on the FORMAT
//! of the file passed to `apr serve run`. The CRUX serve verb reads `GET /` to learn what
//! to test, so those cells were RED `no_route_index`, correctly.
//!
//! [`Indexed::route`] mounts a route and records it in one call, so advertising and
//! mounting cannot disagree. [`Indexed::finish`] mounts `GET /` and the 404 fallback
//! from that record, in the GGUF router's shape:
//! `{"service": "apr serve", "version": …, "routes": ["GET /", "METHOD /path", …]}`.

use axum::{routing::MethodRouter, Json, Router};

/// A router plus the `"METHOD /path"` of every route mounted on it.
pub(crate) struct Indexed<S = ()> {
    router: Router<S>,
    routes: Vec<String>,
}

impl<S: Clone + Send + Sync + 'static> Indexed<S> {
    pub(crate) fn new() -> Self {
        Self {
            router: Router::new(),
            routes: Vec::new(),
        }
    }

    /// Mount `handler` at `path` and record it under `method`.
    pub(crate) fn route(
        mut self,
        method: &'static str,
        path: &'static str,
        handler: MethodRouter<S>,
    ) -> Self {
        self.routes.push(format!("{method} {path}"));
        self.router = self.router.route(path, handler);
        self
    }

    /// Mount a table of routes, recording each.
    pub(crate) fn routes(self, table: Vec<(&'static str, &'static str, MethodRouter<S>)>) -> Self {
        table
            .into_iter()
            .fold(self, |acc, (m, p, h)| acc.route(m, p, h))
    }

    /// Apply layers or other transforms to the router without losing the record.
    pub(crate) fn map(mut self, f: impl FnOnce(Router<S>) -> Router<S>) -> Self {
        self.router = f(self.router);
        self
    }

    /// Supply the state; the record travels with the router.
    pub(crate) fn with_state<S2>(self, state: S) -> Indexed<S2> {
        Indexed {
            router: self.router.with_state(state),
            routes: self.routes,
        }
    }
}

impl Indexed<()> {
    /// Merge another stateless indexed router; both records are kept.
    pub(crate) fn merge(mut self, other: Indexed<()>) -> Self {
        self.router = self.router.merge(other.router);
        self.routes.extend(other.routes);
        self
    }

    /// The `"METHOD /path"` list `GET /` will serve, `GET /` first.
    pub(crate) fn index(&self) -> Vec<String> {
        std::iter::once("GET /".to_string())
            .chain(self.routes.iter().cloned())
            .collect()
    }

    /// Mount `GET /` and the 404 fallback from the record, and return the router.
    pub(crate) fn finish(self) -> Router {
        let index = self.index();
        let root = index.clone();
        let not_found = index;
        self.router
            .route(
                "/",
                axum::routing::get(move || {
                    let routes = root.clone();
                    async move {
                        Json(serde_json::json!({
                            "service": "apr serve",
                            "version": env!("CARGO_PKG_VERSION"),
                            "routes": routes,
                        }))
                    }
                }),
            )
            .fallback(move || {
                let routes = not_found.clone();
                async move {
                    (
                        axum::http::StatusCode::NOT_FOUND,
                        Json(serde_json::json!({
                            "error": "not_found",
                            "message": "Route not found. Available routes are listed in `routes`.",
                            "routes": routes,
                        })),
                    )
                }
            })
    }
}
