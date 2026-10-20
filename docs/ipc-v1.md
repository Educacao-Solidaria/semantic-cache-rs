## 1. IPC v1 — gateway (Go) ⇄ semantic-cache (Rust)

**Transporte:** Unix domain socket, modo stream. Caminho padrão `/tmp/semantic-cache.sock`
(configurável nos dois lados). Windows (CACHRS PR 61): named pipe `\\.\pipe\semantic-cache`,
mesmo framing.

**Frame** (inteiros em big-endian):

| offset | bytes | campo |
|---|---|---|
| 0 | 4 | magic `53 43 49 50` ("SCIP") |
| 4 | 1 | version = `1` |
| 5 | 1 | kind |
| 6 | 2 | flags — v1: sempre 0; bit desconhecido → `bad_frame` |
| 8 | 4 | request_id (u32). A resposta ecoa o do pedido. |
| 12 | 4 | payload_len (u32). Máximo 16 MiB (16 777 216); acima → `too_large` e fecha a conexão |
| 16 | N | payload: JSON UTF-8 (N = payload_len; pode ser 0) |
| 16+N | 4 | crc32 **IEEE** (`hash/crc32.ChecksumIEEE` / `crc32fast`) sobre os bytes `[0, 16+N)` |

**Kinds e payloads** (os campos JSON seguem as tags do contrato Go do Luan em `pkg/client/cache`):

| kind | nome | payload |
|---|---|---|
| `0x01` | LOOKUP_REQ | `{"prompt": str, "embedding": [f32], "threshold": f32}` |
| `0x02` | LOOKUP_RESP | `{"hit": bool, "score": f32, "cached_response": str (omitido se miss), "latency_micros": i64}` = `cache.LookupResult` |
| `0x03` | STORE_REQ | `{"prompt": str, "response": str, "embedding": [f32], "ttl_seconds": int}` (0 = sem TTL) |
| `0x04` | STORE_RESP | `{}` |
| `0x05` | STATS_REQ | `{}` |
| `0x06` | STATS_RESP | `{"total_entries", "hit_count", "miss_count", "hit_ratio"}` = `cache.StatsResult` |
| `0x07` | LOOKUP_STREAM_REQ | igual ao LOOKUP_REQ + `"chunk_bytes": int` opcional (CACHRS PR 57) |
| `0x08` | STREAM_CHUNK | `{"delta": str}` — zero ou mais, mesmo request_id |
| `0x09` | STREAM_END | `{"hit": bool, "score": f32, "latency_micros": i64}` — encerra o stream; miss = STREAM_END com `hit:false` e nenhum chunk |
| `0x0E` | PING | `{}` ou vazio |
| `0x0F` | PONG | `{}` ou vazio |
| `0x7F` | ERROR | `{"code": "bad_frame"\|"bad_request"\|"too_large"\|"overloaded"\|"unsupported"\|"internal", "message": str}` |

**Segurança do socket (adendo de 20/10, só acrescenta — o fio não muda):** `/tmp` é de todos, e
quem criar `/tmp/semantic-cache.sock` antes do cache recebe os prompts e pode devolver resposta falsa.
- Servidor (Rust): o socket fica com permissão `0600` antes do primeiro `listen` (chmod entre `bind` e
  `listen`: antes do `listen` o kernel recusa todo `connect`) e só remove na partida um caminho que seja
  socket morto — nunca arquivo comum nem symlink (CACHRS 51/55).
- Cliente (Go): antes de conectar, `Lstat` do caminho — tem de ser socket, não symlink, e o dono tem
  de ser o uid esperado (padrão: o próprio uid do gateway; configurável para produção com usuários
  separados). Não confere → não conecta e o cache vira miss, com log (GATEGO 69).
- Produção: socket num diretório privado do serviço (ex.: `/run/semantic-cache/`, `0750`) pela config
  dos dois lados; `/tmp/semantic-cache.sock` fica como padrão de desenvolvimento.
- Windows: o servidor fala named pipe (CACHRS 61) e o cliente Go fala AF_UNIX — os dois só se
  encontram no Windows se um dos lados ganhar o outro transporte; fora do escopo do v1.

**Regras:**
- Magic, versão ou CRC errados: o receptor manda ERROR `bad_frame` (se o cabeçalho foi lido) e
  **fecha a conexão** — não tenta ressincronizar o stream.
- Kind desconhecido num frame íntegro: ERROR `unsupported`, conexão segue.
- Pedidos podem ir em pipeline numa mesma conexão; respostas podem chegar fora de ordem —
  correlacione por request_id. O cliente nunca reutiliza um request_id ainda pendente.
- Dimensão do embedding diferente da do índice: ERROR `bad_request`.
- Servidor no limite de conexões (CACHRS PR 65): ERROR `overloaded` com request_id 0 e fecha.
- Mensagem de erro não carrega prompt, resposta nem embedding (podem ser dado do usuário).
- Receptor não depende de ordem de chave nem de espaço no JSON; float pode vir `1` ou `1.0`.

**Frames de referência** (conferidos com `zlib.crc32` e com `hash/crc32` do Go). Os dois repos têm
teste da camada de framing contra estes bytes: `encode(kind, flags, request_id, payload_bytes)`
produz exatamente o hex; `decode(hex)` devolve os mesmos campos e bytes de payload.

```text
PING, request_id=1, payload vazio (20 bytes):
53434950010e00000000000100000000c47a57de

LOOKUP_REQ, request_id=7, payload {"prompt":"olá","embedding":[0.5,-0.25],"threshold":0.75} (58 bytes; frame 78):
5343495001010000000000070000003a7b2270726f6d7074223a226f6cc3a1222c22656d62656464696e67223a5b302e352c2d302e32355d2c227468726573686f6c64223a302e37357d9cad1a2b

ERROR, request_id=7, payload {"code":"bad_request","message":"dimensão 2 ≠ 1536"} (55 bytes; frame 75):
53434950017f000000000007000000377b22636f6465223a226261645f72657175657374222c226d657373616765223a2264696d656e73c3a36f203220e289a02031353336227d0777f547

O mesmo LOOKUP_REQ com 1 bit trocado no offset 20 — TEM de ser rejeitado por CRC:
5343495001010000000000070000003a7b2270726e6d7074223a226f6cc3a1222c22656d62656464696e67223a5b302e352c2d302e32355d2c227468726573686f6c64223a302e37357d9cad1a2b
```

**Quem faz o quê:**
- Go (Victor): GATEGO 51 cliente UDS que implementa `cache.Client` do Luan; 53 framing; 61
  gravação assíncrona (STORE_REQ); 69 validação de integridade; 73 bench UDS × TCP; 75 e2e com mocks.
- Rust (Victor): CACHRS 51 servidor UDS (accept loop + entrega da conexão a um handler por trait —
  o despachante é o PR 52 do Luan); 55 limpeza do `.sock`; 57 LOOKUP_STREAM; 61 named pipes;
  63 crc32; 65 limite de conexões; 69 desconexões abruptas; 71 bench e2e < 1,5 ms; 73 carga 50k;
  75 e2e Rust ⇄ Go (o CI do Rust pode fazer checkout do flow-gateway-go — repos públicos).
- O despachante (CACHRS 52) é do Luan: até ele existir, o servidor do Victor responde com um
  handler mínimo próprio (PING/STATS e, se couber, LOOKUP/STORE sobre o store que já existe) —
  contract-first, sem esperar.
