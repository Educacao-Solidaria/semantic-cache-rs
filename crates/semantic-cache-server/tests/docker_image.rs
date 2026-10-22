//! A imagem do servidor (`Dockerfile` e `.dockerignore` na raiz) contra o
//! contrato da Fase 5, §2 e §3: bases fixadas por digest, final `scratch`, uid
//! 65532, socket e persistência nos caminhos dos volumes, a porta da API, o
//! HEALTHCHECK pelo próprio binário e o contexto de build sem segredo nem
//! `target/`. O que só o build mostra (tamanho, dono das pastas, o binário
//! rodando) fica com o CI, em `.github/workflows/docker.yml`.

use std::collections::HashMap;
use std::path::Path;

use semantic_cache_server::config::Config;

const DOCKERFILE: &str = include_str!("../../../Dockerfile");
const DOCKERIGNORE: &str = include_str!("../../../.dockerignore");
const SOCKET: &str = "/run/semantic-cache/semantic-cache.sock";
const DATA: &str = "/var/lib/semantic-cache";
const SECRETS: [&str; 4] = ["KEY", "TOKEN", "SECRET", "PASSWORD"];

/// Palavra-chave e argumentos.
type Instruction = (String, String);

/// As instruções do Dockerfile por estágio (cada `FROM` abre um), com as
/// continuações juntadas e sem comentários.
fn stages() -> Vec<Vec<Instruction>> {
    let mut stages: Vec<Vec<Instruction>> = Vec::new();
    let mut joined = String::new();
    for line in DOCKERFILE.lines().map(str::trim) {
        if line.starts_with('#') || (joined.is_empty() && line.is_empty()) {
            continue;
        }
        if let Some(head) = line.strip_suffix('\\') {
            joined.push_str(head);
            joined.push(' ');
            continue;
        }
        joined.push_str(line);
        let (word, args) = joined.split_once(' ').unwrap_or((joined.as_str(), ""));
        let instruction = (word.to_ascii_uppercase(), args.trim().to_owned());
        if instruction.0 == "FROM" {
            stages.push(Vec::new());
        }
        let stage = stages.last_mut().expect("instrução antes do FROM");
        stage.push(instruction);
        joined.clear();
    }
    stages
}

/// Os argumentos das instruções `word` do estágio, na ordem.
fn all<'a>(stage: &'a [Instruction], word: &str) -> Vec<&'a str> {
    let found = stage.iter().filter(|(w, _)| w == word);
    found.map(|(_, args)| args.as_str()).collect()
}

/// A instrução `word` do estágio, que tem de ser única.
fn one<'a>(stage: &'a [Instruction], word: &str) -> &'a str {
    match all(stage, word)[..] {
        [args] => args,
        ref found => panic!("{word}: esperava uma, achei {found:?}"),
    }
}

#[test]
fn every_base_is_pinned_by_digest_and_the_final_stage_is_scratch() {
    let stages = stages();
    let mut named = Vec::new();
    for stage in &stages {
        let from: Vec<&str> = one(stage, "FROM").split_whitespace().collect();
        let pinned = from[0].split_once("@sha256:").is_some_and(|(_, digest)| {
            digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit())
        });
        let local = from[0] == "scratch" || named.contains(&from[0]);
        assert!(pinned || local, "base sem digest: {}", from[0]);
        if let [_, alias, name] = from[..] {
            assert_eq!(alias, "AS");
            named.push(name);
        }
    }
    assert_eq!(one(stages.last().unwrap(), "FROM"), "scratch");
}

#[test]
fn the_server_runs_as_65532_with_the_contract_paths_and_port() {
    let stages = stages();
    let last = stages.last().unwrap();
    assert_eq!(one(last, "USER"), "65532:65532");
    let env: HashMap<&str, &str> = all(last, "ENV")
        .iter()
        .flat_map(|args| args.split_whitespace())
        .map(|pair| pair.split_once('=').expect("ENV chave=valor"))
        .collect();
    // O binário lê essas variáveis: um nome errado cairia no padrão (/tmp).
    let config = Config::load_with(|var| env.get(var).copied().map(str::to_owned)).unwrap();
    assert_eq!(config.ipc.socket_path, Path::new(SOCKET));
    assert_eq!(config.persistence.path.as_deref(), Some(Path::new(DATA)));
    assert_eq!(config.http.listen.port(), 8099);
    assert_eq!(one(last, "EXPOSE"), "8099");
}

#[test]
fn the_final_image_is_the_build_tree_with_the_service_dirs_owned_by_65532() {
    let stages = stages();
    // Nada entra na imagem final fora da árvore montada no estágio `build`.
    let copies = all(stages.last().unwrap(), "COPY");
    assert_eq!(copies, ["--from=build /rootfs/ /"]);
    let build = stages
        .iter()
        .find(|s| one(s, "FROM").ends_with(" AS build"));
    let runs = all(build.unwrap(), "RUN").join(" && ");
    let chown = runs.split("&&").map(str::trim).find_map(|command| {
        let owned = command.strip_prefix("chown 65532:65532 ")?;
        Some(owned.split_whitespace().collect::<Vec<_>>())
    });
    let owned = chown.expect("chown 65532:65532 no build");
    // As pastas em que o servidor escreve: a do socket e a da persistência.
    let socket_dir = Path::new(SOCKET).parent().unwrap().to_str().unwrap();
    for dir in [socket_dir, DATA] {
        assert!(owned.contains(&format!("/rootfs{dir}").as_str()), "{dir}");
    }
}

#[test]
fn the_healthcheck_is_the_binary_pinging_the_socket() {
    let stages = stages();
    let last = stages.last().unwrap();
    let entrypoint: Vec<String> = serde_json::from_str(one(last, "ENTRYPOINT")).unwrap();
    let (_, cmd) = one(last, "HEALTHCHECK").split_once(" CMD ").unwrap();
    // Forma exec (JSON), como o ENTRYPOINT: a imagem não tem shell.
    let cmd: Vec<String> = serde_json::from_str(cmd).expect("CMD em forma exec");
    assert_eq!(cmd, [entrypoint[0].as_str(), "--ping"]);
}

#[test]
fn no_secret_enters_the_image_or_the_build_context() {
    for (word, args) in stages().iter().flatten() {
        let upper = args.to_ascii_uppercase();
        let secret = SECRETS.iter().any(|s| upper.contains(s));
        let declared = matches!(word.as_str(), "ARG" | "ENV");
        assert!(!(declared && secret), "{word} {args}");
    }
    let rules: Vec<&str> = DOCKERIGNORE
        .lines()
        .map(str::trim)
        .filter(|rule| !rule.is_empty() && !rule.starts_with('#'))
        .collect();
    // Vale a última regra que casa: as exclusões vêm depois de toda exceção (`!`).
    let last_exception = rules.iter().rposition(|r| r.starts_with('!'));
    let after_exceptions = last_exception.map_or(0, |i| i + 1);
    for kept_out in [
        "**/.git",
        "**/target",
        "**/.env",
        "**/.env.*",
        "**/node_modules",
        "**/.venv",
        "**/rust-toolchain.toml",
    ] {
        let at = rules.iter().position(|rule| *rule == kept_out);
        assert!(at.is_some_and(|i| i >= after_exceptions), "{kept_out}");
    }
}
