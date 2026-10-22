# check=error=true

# Imagem do semantic-cache-server: binário estático (musl) numa `scratch`, sem
# shell, rodando como 65532:65532 (contrato da Fase 5, §2 e §3). Contexto = raiz:
#   docker build -t educacao-solidaria/semantic-cache:dev .
# O CI (.github/workflows/docker.yml) confere tamanho, conteúdo e o container.

# Rust 1.99.1 + cargo-chef 0.1.78 sobre Alpine 3.24: o alvo nativo já é musl e o
# binário sai estático, sem compilação cruzada. O C do jemalloc compila com o
# gcc e o musl-dev da imagem; falta só o make.
# lukemathwalker/cargo-chef:0.1.78-rust-1.99.1-alpine3.24
FROM lukemathwalker/cargo-chef@sha256:2c27e67049f17e8b67d882a2cf2f5e1a1d4d71c206ec97f91edf278254688e95 AS chef
# Sem versão fixa: o repositório da Alpine só guarda a última de cada pacote.
# hadolint ignore=DL3018
RUN apk add --no-cache make
WORKDIR /src

# A receita do cargo-chef: manifests e lock do workspace do servidor, sem código.
FROM chef AS planner
COPY . /src
WORKDIR /src/crates/semantic-cache-server
RUN cargo chef prepare --recipe-path /recipe.json

FROM chef AS build
# Os path deps do servidor (crate da raiz, ipc e o simd dos testes) ficam fora
# do workspace dele, e o cargo-chef só põe membros na receita: entram aqui os
# manifests deles com uma lib vazia, e a camada das dependências só muda quando
# muda dependência, não código.
COPY Cargo.toml /src/
COPY crates/semantic-cache-ipc/Cargo.toml /src/crates/semantic-cache-ipc/
COPY crates/semantic-cache-simd/Cargo.toml /src/crates/semantic-cache-simd/
RUN for crate in /src /src/crates/semantic-cache-ipc /src/crates/semantic-cache-simd; do \
        mkdir -p "$crate/src" && touch "$crate/src/lib.rs"; \
    done
COPY --from=planner /recipe.json /recipe.json
WORKDIR /src/crates/semantic-cache-server
# Dependências em release (LTO, como o binário). As libs vazias saem do target,
# como o cargo-chef faz com os membros: o build abaixo compila o código real.
RUN cargo chef cook --release --locked --recipe-path /recipe.json \
    && cargo clean --release --locked -p semantic-cache-rs -p semantic-cache-ipc
COPY . /src
# A imagem final é esta árvore: o binário (root, 0755) e as pastas do socket e
# da persistência, do uid do serviço e só dele (0700).
RUN cargo build --release --locked \
    && mkdir -p /rootfs/usr/local/bin /rootfs/run/semantic-cache /rootfs/var/lib/semantic-cache \
    && cp target/release/semantic-cache-server /rootfs/usr/local/bin/ \
    && chown 65532:65532 /rootfs/run/semantic-cache /rootfs/var/lib/semantic-cache \
    && chmod 0777 /rootfs/run/semantic-cache /rootfs/var/lib/semantic-cache

FROM scratch
# Sem --chown: o COPY de outro estágio mantém o dono e o modo de lá.
COPY --from=build /rootfs/ /
USER 65532:65532
ENV SEMCACHE_IPC_SOCKET_PATH=/run/semantic-cache/semantic-cache.sock
ENV SEMCACHE_PERSISTENCE_PATH=/var/lib/semantic-cache
# API HTTP de depuração: desligada por padrão; o compose a liga em 0.0.0.0:8099,
# só na rede interna.
EXPOSE 8099
# Prontidão do contrato: um PING respondido pelo socket, pelo uid do serviço.
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --start-interval=1s --retries=3 \
    CMD ["/usr/local/bin/semantic-cache-server", "--ping"]
ENTRYPOINT ["/usr/local/bin/semantic-cache-server"]
