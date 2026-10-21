# Observabilidade do semantic-cache-rs

A parte do cache nas convenções da Fase 4, fixas entre os quatro repositórios:
os nomes, tipos, labels e buckets daqui são os que os dashboards, os alertas e
os testes usam. Mudou um nome, muda em todos, ou o painel fica vazio sem erro.

## Métricas (Prometheus)

`GET /metrics` na API HTTP de depuração (`[http]`, PR 53/100), em formato texto
0.0.4. A API vem desligada; ligada, escuta em `127.0.0.1:8099` e só atende
`Host` de loopback (contra DNS rebinding). Não há senha: quem protege é o
loopback. Fora dele (`SEMCACHE_HTTP_LISTEN`), a API inteira fica exposta,
inclusive o `/cache/entries`, que mostra prompts: só em rede interna. Toda
série existe desde a partida, zerada, para o `rate()` ver o primeiro evento.

| Métrica | Tipo | Labels | De onde vem |
|---|---|---|---|
| `semantic_cache_ipc_requests_total` | counter | kind, outcome | `semantic_cache_ipc::metrics::request`, chamado pelo handler a cada pedido: hoje o `Basic`, depois o despachante do PR 52/100 |
| `semantic_cache_ipc_request_duration_seconds` | histogram | kind | o mesmo `request`: do frame lido à resposta escrita (ou à falha da escrita) |
| `semantic_cache_lookups_total` | counter | result | `semantic_cache_ipc::metrics::lookup`, chamado por quem busca; o binário ainda não busca (LOOKUP é `unsupported` até o PR 52/100) e fica em zero |
| `semantic_cache_search_duration_seconds` | histogram | index | `semantic_cache_ipc::metrics::search`, chamado por quem busca, com o tempo de uma busca top-k; zero pelo mesmo motivo |
| `semantic_cache_entries` | gauge | — | tamanho do store persistido (contador atômico do `SledStore`); 0 sem persistência |
| `semantic_cache_memory_bytes` | gauge | — | bytes reservados no `MemoryBudget` (PR 33/100) |
| `semantic_cache_memory_limit_bytes` | gauge | — | teto do orçamento (`memory.max_bytes`) |
| `semantic_cache_ipc_connections` | gauge | — | conexões em atendimento no `serve` (as recusadas pelo teto não contam) |
| `semantic_cache_ipc_frames_discarded_total` | counter | — | frames recusados pelo codec: magic, versão, flags, CRC ou tamanho (PR 63/100) |

Labels, todos de valores fixos:

- `kind`: `lookup`, `store`, `stats`, `lookup_stream`, `ping`; kind fora do v1
  vira `unknown`, para a cardinalidade não crescer com o que o cliente manda.
- `outcome`: `ok`, `error` (respondido com ERROR), `timeout` (a resposta parou
  no prazo de ociosidade) e `canceled` (o par saiu antes da resposta).
- `result`: `hit`, `miss`, `error`.
- `index`: `flat_f32` e `flat_i8` (varredura exata em f32 e nos códigos int8)
  e `hnsw` (o índice do PR 32/100).

Os dois histogramas usam os buckets do contrato, em segundos: `0.0001 0.00025
0.0005 0.001 0.002 0.005 0.01 0.025` e `+Inf`, com `_sum` e `_count`
(`_count` = o `+Inf`). Uma observação conta no primeiro `le` que a cabe
(`le` é "menor ou igual": 100 µs cai em `le="0.0001"`). O p99 do LOOKUP:
`histogram_quantile(0.99, sum by (le) (rate(semantic_cache_ipc_request_duration_seconds_bucket{kind="lookup"}[5m])))`.

Nenhum label leva tenant, prompt, resposta, embedding ou request id. Frame
inválido não é pedido: só conta em `semantic_cache_ipc_frames_discarded_total`.
Não há `process_*` (exportador escrito à mão, sem crate); o heap do alocador
segue no `/stats`.

Coleta com o Prometheus no mesmo host:

```yaml
scrape_configs:
  - job_name: semantic-cache
    static_configs:
      - targets: ["127.0.0.1:8099"]
```
