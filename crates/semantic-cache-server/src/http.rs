//! API HTTP de depuração, só leitura: saúde, estatísticas, métricas do
//! Prometheus e inspeção das entradas persistidas, para olhar o cache com
//! `curl`. As rotas de consulta e gravação são do PR 54/100: entram com `routes(..).merge(..)` antes do
//! [`with_limits`], que vale para todas.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::{DefaultBodyLimit, Query, Request, State};
use axum::http::{header, header::HOST, uri::Authority, StatusCode};
use axum::middleware::{from_fn, from_fn_with_state, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use semantic_cache_rs::CacheStore;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::config::HttpConfig;
use crate::memory::MemoryBudget;
use crate::persistence::SledStore;
use crate::{alloc, build_info, metrics};

/// Teto de entradas numa resposta do `/cache/entries`.
pub const MAX_ENTRIES: usize = 1000;

#[derive(Debug, Clone)]
struct Inspected {
    store: Option<Arc<SledStore>>,
    budget: MemoryBudget,
    started: Instant,
}

/// `/health`, `/stats` e `/metrics` (`budget` é o orçamento do store); com
/// persistência, também `/cache/entries`.
pub fn routes(store: Option<Arc<SledStore>>, budget: MemoryBudget) -> Router {
    let mut router = Router::new()
        .route("/health", get(health))
        .route("/stats", get(stats))
        .route("/metrics", get(prometheus));
    if store.is_some() {
        router = router.route("/cache/entries", get(entries));
    }
    let started = Instant::now();
    router.with_state(Inspected {
        store,
        budget,
        started,
    })
}

/// Teto do corpo e prazo por pedido em todas as rotas de `router`. O corpo é
/// lido dentro do handler, então o prazo cobre também quem o manda devagar.
/// No loopback, só atende `Host` de loopback (ver [`loopback_host`]).
pub fn with_limits(router: Router, config: &HttpConfig) -> Router {
    let router = router
        .layer(DefaultBodyLimit::max(config.max_body_bytes))
        .layer(from_fn_with_state(config.timeout(), deadline));
    if config.listen.ip().is_loopback() {
        return router.layer(from_fn(loopback_host));
    }
    router
}

/// Contra DNS rebinding: um site que faça o próprio domínio apontar para
/// 127.0.0.1 leria o cache com o navegador da vítima. Fora do loopback (só por
/// configuração explícita) a checagem não se aplica.
async fn loopback_host(request: Request, next: Next) -> Response {
    let host = request.headers().get(HOST).and_then(|h| h.to_str().ok());
    let host = host.and_then(|h| h.parse::<Authority>().ok());
    match host.as_ref().map(Authority::host) {
        Some("localhost" | "127.0.0.1" | "[::1]") => next.run(request).await,
        _ => (StatusCode::FORBIDDEN, "Host não é loopback").into_response(),
    }
}

async fn deadline(State(limit): State<Duration>, request: Request, next: Next) -> Response {
    let response = tokio::time::timeout(limit, next.run(request)).await;
    response.unwrap_or_else(|_| StatusCode::REQUEST_TIMEOUT.into_response())
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok", "version": build_info::version_line() }))
}

async fn stats(State(state): State<Inspected>) -> Json<Value> {
    let allocator = alloc::stats().map(|s| {
        json!({
            "name": alloc::ALLOCATOR,
            "allocated_bytes": s.allocated,
            "resident_bytes": s.resident,
            "fragmentation": s.fragmentation(),
        })
    });
    Json(json!({
        "uptime_secs": state.started.elapsed().as_secs(),
        "allocator": allocator,
        "store_entries": state.store.as_ref().map(|store| store.len()),
        "ipc_discarded_frames": semantic_cache_ipc::frame::discarded_frames(),
    }))
}

/// Só lê contadores atômicos: nada de disco, cabe na thread do runtime.
async fn prometheus(State(state): State<Inspected>) -> impl IntoResponse {
    let text = metrics::render(&state.budget, state.store.as_deref());
    ([(header::CONTENT_TYPE, metrics::CONTENT_TYPE)], text)
}

#[derive(Debug, Deserialize)]
struct Page {
    limit: Option<usize>,
}

/// Entradas na ordem das chaves, até `limit` (padrão 20, teto
/// [`MAX_ENTRIES`]), sem o embedding.
async fn entries(State(state): State<Inspected>, Query(page): Query<Page>) -> Response {
    let Some(store) = state.store else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let limit = page.limit.unwrap_or(20).min(MAX_ENTRIES);
    // O sled lê do disco de forma síncrona: fora das threads do runtime.
    let read = tokio::task::spawn_blocking(move || {
        let summary = |(key, e): (String, semantic_cache_rs::CacheEntry)| {
            json!({
                "key": key,
                "id": e.id,
                "prompt": e.prompt,
                "response_bytes": e.response.len(),
                "dimension": e.embedding.dim(),
                "hit_count": e.hit_count,
                "created_at": e.created_at,
                "ttl_seconds": e.ttl_seconds,
            })
        };
        let entries = store.entries().take(limit).map(|item| item.map(summary));
        entries.collect::<Result<Vec<_>, _>>()
    });
    match read.await {
        Ok(Ok(entries)) => Json(json!({ "entries": entries })).into_response(),
        Ok(Err(err)) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use axum::body::{to_bytes, Body};
    use axum::routing::post;
    use semantic_cache_rs::{CacheEntry, Vector};
    use tower::ServiceExt;

    use super::*;
    use crate::config::{MemoryConfig, PersistenceConfig};

    fn budget() -> MemoryBudget {
        MemoryBudget::new(&MemoryConfig::default()).unwrap()
    }

    /// O texto do `GET /metrics` como o Prometheus pede (`Host` com a porta),
    /// conferidos o status e o `Content-Type`.
    async fn scrape(app: &Router) -> String {
        let request = Request::get("/metrics").header(HOST, "127.0.0.1:8099");
        let response = app.clone().oneshot(request.body(Body::empty()).unwrap());
        let response = response.await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let kind = &response.headers()[header::CONTENT_TYPE];
        assert_eq!(kind, metrics::CONTENT_TYPE);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        String::from_utf8(body.to_vec()).unwrap()
    }

    /// `(status, corpo em JSON)`, com `Host: localhost`; corpo que não é JSON
    /// vira `null`.
    async fn call(router: &Router, method: &str, uri: &str, body: &str) -> (StatusCode, Value) {
        let request = Request::builder()
            .method(method)
            .uri(uri)
            .header(HOST, "localhost");
        let request = request.body(Body::from(body.to_owned())).unwrap();
        let response = router.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    #[tokio::test]
    async fn health_and_stats_answer_without_persistence() {
        let router = routes(None, budget());
        let (status, body) = call(&router, "GET", "/health", "").await;
        assert_eq!((status, &body["status"]), (StatusCode::OK, &json!("ok")));
        assert_eq!(body["version"], build_info::version_line());
        let (status, body) = call(&router, "GET", "/stats", "").await;
        assert_eq!(
            (status, &body["store_entries"]),
            (StatusCode::OK, &Value::Null)
        );
        assert_eq!(body["allocator"]["name"], alloc::ALLOCATOR);
        assert!(body["ipc_discarded_frames"].is_u64());
        // Sem persistência a inspeção não existe; e nada aqui aceita escrita.
        assert_eq!(
            call(&router, "GET", "/cache/entries", "").await.0,
            StatusCode::NOT_FOUND
        );
        let post = call(&router, "POST", "/health", "").await.0;
        assert_eq!(post, StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn entries_are_listed_in_key_order_without_the_embedding() {
        let dir = std::env::temp_dir().join(format!("semcache-http-{}", std::process::id()));
        let config = PersistenceConfig {
            path: Some(dir.clone()),
            ..PersistenceConfig::default()
        };
        let store = Arc::new(SledStore::open(&dir, &config).unwrap());
        for (key, prompt) in [("b", "olá"), ("a", "oi"), ("c", "e aí")] {
            let embedding = Vector { data: vec![0.5; 3] };
            let (id, response) = (key.to_owned(), "resposta".to_owned());
            let entry = CacheEntry {
                id,
                prompt: prompt.into(),
                response,
                embedding,
                hit_count: 2,
                created_at: 9,
                ttl_seconds: 60,
            };
            store.put(key, &entry).unwrap();
        }
        let router = routes(Some(store), budget());
        let (status, body) = call(&router, "GET", "/cache/entries?limit=2", "").await;
        let expected = |key: &str, prompt: &str| {
            json!({ "key": key, "id": key, "prompt": prompt, "response_bytes": 8, "dimension": 3,
                "hit_count": 2, "created_at": 9, "ttl_seconds": 60 })
        };
        let listed = json!({ "entries": [expected("a", "oi"), expected("b", "olá")] });
        assert_eq!((status, body), (StatusCode::OK, listed));
        let empty = call(&router, "GET", "/cache/entries?limit=0", "").await;
        assert_eq!(empty, (StatusCode::OK, json!({ "entries": [] })));
        let bad = call(&router, "GET", "/cache/entries?limit=-1", "").await.0;
        assert_eq!(bad, StatusCode::BAD_REQUEST);
        assert_eq!(
            call(&router, "GET", "/stats", "").await.1["store_entries"],
            3
        );
        let text = scrape(&router).await;
        assert!(text.contains("\nsemantic_cache_entries 3\n"), "{text}");
        drop(router);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn metrics_answer_the_scrape_through_the_loopback_guard() {
        let app = with_limits(routes(None, budget()), &HttpConfig::default());
        let text = scrape(&app).await;
        let limit = "\nsemantic_cache_memory_limit_bytes 1073741824\n";
        assert!(text.contains(limit), "{text}");
    }

    #[tokio::test]
    async fn body_limit_and_deadline_cover_merged_routes() {
        let config = HttpConfig {
            max_body_bytes: 8,
            timeout_ms: 50,
            ..HttpConfig::default()
        };
        let extra = Router::new()
            .route("/echo", post(|body: String| async move { body }))
            .route("/slow", get(std::future::pending::<String>));
        let app = with_limits(routes(None, budget()).merge(extra), &config);
        assert_eq!(
            call(&app, "POST", "/echo", "12345678").await.0,
            StatusCode::OK
        );
        let big = call(&app, "POST", "/echo", "123456789").await.0;
        assert_eq!(big, StatusCode::PAYLOAD_TOO_LARGE);
        let slow = tokio::time::timeout(Duration::from_secs(10), call(&app, "GET", "/slow", ""));
        assert_eq!(
            slow.await.expect("o prazo não cortou").0,
            StatusCode::REQUEST_TIMEOUT
        );
    }

    #[tokio::test]
    async fn loopback_listener_only_answers_loopback_hosts() {
        let app = with_limits(routes(None, budget()), &HttpConfig::default());
        let host = |host: &'static str| {
            let request = Request::get("/health")
                .header(HOST, host)
                .body(Body::empty());
            app.clone().oneshot(request.unwrap())
        };
        for ok in ["localhost", "127.0.0.1:8099", "[::1]:8099"] {
            assert_eq!(host(ok).await.unwrap().status(), StatusCode::OK, "{ok}");
        }
        for rebound in ["evil.example:8099", "127.0.0.1.evil.example", ""] {
            assert_eq!(
                host(rebound).await.unwrap().status(),
                StatusCode::FORBIDDEN,
                "{rebound}"
            );
        }
        // Exposto de propósito (fora do loopback), o Host não é conferido.
        let open = HttpConfig {
            listen: "0.0.0.0:8099".parse().unwrap(),
            ..HttpConfig::default()
        };
        let request = Request::get("/health")
            .header(HOST, "cache.interno")
            .body(Body::empty());
        let response = with_limits(routes(None, budget()), &open)
            .oneshot(request.unwrap())
            .await;
        assert_eq!(response.unwrap().status(), StatusCode::OK);
    }
}
