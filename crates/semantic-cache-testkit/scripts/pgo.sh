#!/usr/bin/env bash
# PGO (profile-guided optimization) do caminho quente do cache, sobre o bench
# `ipc` do PR 71/100: LOOKUP em JSON → frame → `serve` de produção → busca com o
# kernel SIMD ativo → resposta. Reprodutível e fora do build padrão:
#   1. compila o bench instrumentado (-Cprofile-generate) e roda a carga dele;
#   2. funde o perfil com o llvm-profdata do próprio rustc (componente llvm-tools);
#   3. recompila com o perfil (-Cprofile-use) e sem ele (a base): mesmas flags e o
#      profile do binário de produção (LTO gordo, uma unidade de codegen);
#   4. mede base e PGO na mesma máquina, em rodadas intercaladas, e compara
#      (scripts/pgo-report.awk).
#
#   (em crates/semantic-cache-testkit)
#   rustup component add llvm-tools   # uma vez
#   scripts/pgo.sh [rodadas]          # padrão 5; cada rodada roda as duas versões
#
# Sai em target/pgo/: ipc.profdata, os binários ipc-base e ipc-pgo, a saída de cada
# rodada (runs/) e relatorio.md. Treino e medição usam a mesma carga
# (SEMCACHE_BENCH_QUICK): em produção, o ganho tende a ser menor.
# ponytail: o perfil só vale para o binário do bench. O de produção ainda não faz
# LOOKUP (o despachante é o PR 52/100); quando fizer, o treino passa a ser uma
# carga de cliente no socket dele, com estes mesmos passos.
set -euo pipefail
cd "$(dirname "$0")/.."
rounds=${1:-5}
[[ $rounds =~ ^[1-9][0-9]?$ ]] || { echo "uso: $0 [rodadas, de 1 a 99]" >&2; exit 2; }
host=$(rustc -vV | sed -n 's/^host: //p')
profdata=$(rustc --print sysroot)/lib/rustlib/$host/bin/llvm-profdata
[ -x "$profdata" ] || { echo "falta o llvm-profdata: rustup component add llvm-tools" >&2; exit 1; }
out=$PWD/target/pgo
rm -rf "$out/raw" "$out/runs" "$out/relatorio.md"
mkdir -p "$out/runs"
export CARGO_PROFILE_BENCH_LTO=fat CARGO_PROFILE_BENCH_CODEGEN_UNITS=1 SEMCACHE_BENCH_QUICK=1

# build <nome> [flag do rustc...]: compila o bench `ipc` só com essas flags (as
# RUSTFLAGS do ambiente ficam de fora) e copia o executável para $out/<nome>. O
# `--target` explícito impede que as flags cheguem aos build scripts e às macros.
build() {
  local name=$1 json
  shift
  json=$(CARGO_ENCODED_RUSTFLAGS=$(IFS=$'\x1f'; echo "$*") cargo bench --locked --no-run \
    --bench ipc --target "$host" --target-dir "$out/target" --message-format=json-render-diagnostics)
  cp "$(sed -n '/"kind":\["bench"\]/s/.*"executable":"\([^"]*\)".*/\1/p' <<<"$json")" "$out/$name"
}

build ipc-gen "-Cprofile-generate=$out/raw"
LLVM_PROFILE_FILE="$out/raw/%p.profraw" "$out/ipc-gen" --bench >"$out/treino.txt"
"$profdata" merge -o "$out/ipc.profdata" "$out/raw"
build ipc-base
build ipc-pgo "-Cprofile-use=$out/ipc.profdata"
for r in $(seq "$rounds"); do
  # A ordem alterna (ABBA): a deriva da máquina não pesa sempre do mesmo lado.
  order=(base pgo)
  ((r % 2)) || order=(pgo base)
  for v in "${order[@]}"; do "$out/ipc-$v" --bench >"$out/runs/$v-$r.txt"; done
done
report=$(LC_ALL=C awk -v expect="$rounds" -f scripts/pgo-report.awk "$out"/runs/*.txt)
{
  echo "PGO do bench \`ipc\`, $rounds rodadas intercaladas: $(head -1 "$out/runs/base-1.txt")."
  echo "Δ = (PGO − base) / base em cada rodada (negativo = PGO mais rápido): a mediana e, entre"
  echo "parênteses, em quantas rodadas o PGO foi mais rápido."
  echo
  echo "$report"
} | tee "$out/relatorio.md"
