#!/usr/bin/env bash
# Captura embeddings reais pelo OpenRouter e grava uma fixture `captured`.
#
#   (em crates/semantic-cache-testkit)
#   OPENROUTER_API_KEY=... scripts/capture-embeddings.sh \
#     openai/text-embedding-3-small fixtures/embeddings/texts.tsv \
#     fixtures/embeddings/openai-text-embedding-3-small.jsonl
#
# `OPENROUTER_BASE_URL` troca o endpoint (padrão https://openrouter.ai/api/v1).
# Chamada paga: confira o modelo e o número de textos antes. A chave só vai no
# cabeçalho HTTP, lida de um descritor (não aparece em `ps`) e nunca é gravada.
set -euo pipefail
[ $# -eq 3 ] || { echo "uso: $0 <modelo> <textos.tsv> <saida.jsonl>" >&2; exit 2; }
: "${OPENROUTER_API_KEY:?defina OPENROUTER_API_KEY}"
model=$1 texts=$2 out=$3
bin=(cargo run --quiet --locked --manifest-path "$(dirname "$0")/../Cargo.toml"
  --bin capture-embeddings --)
trap 'rm -f "$out.tmp"' EXIT
"${bin[@]}" request "$model" "$texts" |
  curl --silent --show-error --fail-with-body "${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}/embeddings" \
    -H @<(printf 'Authorization: Bearer %s\n' "$OPENROUTER_API_KEY") \
    -H 'Content-Type: application/json' --data-binary @- |
  "${bin[@]}" fixture "$model" "$texts" >"$out.tmp"
mv "$out.tmp" "$out"
echo "gravado: $out" >&2
