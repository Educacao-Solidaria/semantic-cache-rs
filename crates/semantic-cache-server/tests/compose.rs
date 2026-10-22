//! O compose do cache (`deploy/compose/compose.yaml`) contra o contrato da
//! Fase 5, §3 e §4: a imagem daqui com o socket e a persistência nos volumes
//! nomeados, a API só na rede interna, o lado do gateway com o uid 65532 e o
//! volume do socket só para leitura, e serviço nenhum com privilégio. Quem
//! conversa pelo socket (e quem o kernel recusa) só o CI vê: o job da imagem,
//! em `.github/workflows/docker.yml`, sobe este compose.

use std::collections::BTreeSet;
use std::path::Path;

use semantic_cache_server::config::Config;

const COMPOSE: &str = include_str!("../../../deploy/compose/compose.yaml");
const DOCKERFILE: &str = include_str!("../../../Dockerfile");
const IMAGE: &str = "educacao-solidaria/semantic-cache:dev";
const SOCKET: &str = "/run/semantic-cache/semantic-cache.sock";
const DATA: &str = "/var/lib/semantic-cache";
const SECRETS: [&str; 4] = ["KEY", "TOKEN", "SECRET", "PASSWORD"];
/// O que um serviço pode declarar: chave nova (`ports`, `privileged`,
/// `cap_add`, `healthcheck`...) derruba o teste até ser revista aqui.
const KNOWN: &str = "build image environment volumes networks read_only cap_drop \
                     security_opt profiles user entrypoint network_mode depends_on";
const UNREAD: &str = "YAML fora do que o teste lê";

/// Cada linha do YAML: o caminho das chaves até ela (`services.cache.image`)
/// e o valor sem aspas. Chave que abre bloco entra sem valor; item de lista
/// repete o caminho da lista.
type Flat = Vec<(String, Option<String>)>;

/// Lê só o YAML que o compose usa (mapa e lista em bloco, lista JSON numa
/// linha). Âncora, mapa numa linha, bloco de texto, mapa dentro de lista e
/// `chave:valor` sem espaço derrubam o teste em vez de serem lidos errado.
fn flat(yaml: &str) -> Flat {
    let odd = |value: &str| value.starts_with(['&', '*', '{', '|', '>', '!']);
    let unquote = |value: &str| {
        let inner = value.strip_prefix('"').and_then(|v| v.strip_suffix('"'));
        inner.unwrap_or(value).to_owned()
    };
    let mut open: Vec<(usize, &str)> = Vec::new();
    let mut flat = Vec::new();
    for line in yaml.lines() {
        let text = line.trim_start();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let indent = line.len() - text.len();
        while open.last().is_some_and(|&(at, _)| at >= indent) {
            open.pop();
        }
        let mut path: Vec<&str> = open.iter().map(|&(_, key)| key).collect();
        if let Some(item) = text.strip_prefix("- ") {
            assert!(!odd(item) && !item.contains(": "), "{UNREAD}: {line}");
            flat.push((path.join("."), Some(unquote(item))));
            continue;
        }
        let pair = text.split_once(": ");
        let pair = pair.or_else(|| Some((text.strip_suffix(':')?, "")));
        let (key, value) = pair.unwrap_or_else(|| panic!("{UNREAD}: {line}"));
        let value = value.trim();
        assert!(key != "<<" && !odd(value), "{UNREAD}: {line}");
        path.push(key);
        if value.is_empty() {
            open.push((indent, key));
        }
        flat.push((path.join("."), (!value.is_empty()).then(|| unquote(value))));
    }
    flat
}

/// Um volume do compose, `volume:destino[:ro]`: (volume, destino, só leitura).
fn mount(spec: &str) -> (&str, &str, bool) {
    match spec.split(':').collect::<Vec<_>>()[..] {
        [volume, target] => (volume, target, false),
        [volume, target, "ro"] => (volume, target, true),
        _ => panic!("volume fora de `volume:destino[:ro]`: {spec}"),
    }
}

/// O binário do ENTRYPOINT da imagem (forma exec).
fn binary() -> String {
    let mut lines = DOCKERFILE.lines();
    let exec = lines.find_map(|l| l.strip_prefix("ENTRYPOINT "));
    let mut entrypoint: Vec<String> = serde_json::from_str(exec.expect("ENTRYPOINT")).unwrap();
    entrypoint.remove(0)
}

struct Compose(Flat);

impl Compose {
    fn load() -> Self {
        Self(flat(COMPOSE))
    }

    /// Os valores em `path`, na ordem (uma lista dá vários).
    fn all(&self, path: &str) -> Vec<&str> {
        let found = self.0.iter().filter(|(at, _)| at == path);
        found.filter_map(|(_, value)| value.as_deref()).collect()
    }

    /// O valor em `path`, se houver; mais de um derruba o teste.
    fn get(&self, path: &str) -> Option<&str> {
        match self.all(path)[..] {
            [] => None,
            [value] => Some(value),
            ref many => panic!("{path}: esperava um valor, achei {many:?}"),
        }
    }

    fn one(&self, path: &str) -> &str {
        self.get(path).unwrap_or_else(|| panic!("{path}: ausente"))
    }

    /// As chaves logo abaixo de `path`.
    fn keys(&self, path: &str) -> BTreeSet<&str> {
        let prefix = format!("{path}.");
        let paths = self.0.iter().map(|(at, _)| at.as_str());
        let below = paths.filter_map(|at| at.strip_prefix(prefix.as_str()));
        below.map(|rest| rest.split('.').next().unwrap()).collect()
    }

    fn mounts(&self, service: &str) -> Vec<(&str, &str, bool)> {
        let specs = self.all(&format!("services.{service}.volumes"));
        specs.into_iter().map(mount).collect()
    }

    /// A config que o binário carrega no container do serviço: o `ENV` do
    /// estágio final do Dockerfile com o `environment` do serviço por cima.
    fn config(&self, service: &str) -> Config {
        let (_, last_stage) = DOCKERFILE.rsplit_once("\nFROM ").expect("FROM");
        let image: Vec<(&str, &str)> = last_stage
            .lines()
            .filter_map(|line| line.strip_prefix("ENV "))
            .flat_map(str::split_whitespace)
            .filter_map(|pair| pair.split_once('='))
            .collect();
        let env = |var: &str| {
            let set = self.get(&format!("services.{service}.environment.{var}"));
            let baked = image.iter().find(|(name, _)| *name == var);
            set.or(baked.map(|&(_, value)| value)).map(str::to_owned)
        };
        Config::load_with(env).expect("config do serviço")
    }
}

#[test]
fn the_cache_runs_this_image_with_the_socket_and_the_data_on_named_volumes() {
    let compose = Compose::load();
    // A imagem do Dockerfile da raiz, com o uid (65532), o ENTRYPOINT e o
    // HEALTHCHECK dela (que nem está entre as chaves conhecidas).
    assert_eq!(compose.one("services.cache.build.context"), "../..");
    assert_eq!(compose.one("services.cache.image"), IMAGE);
    let own = compose.keys("services.cache");
    let overridden = own.contains("user") || own.contains("entrypoint");
    assert!(!overridden, "{own:?}");
    // Um nome de variável errado cairia no padrão (/tmp, sem API): a config
    // que o binário carrega é a medida.
    let config = compose.config("cache");
    assert_eq!(config.ipc.socket_path, Path::new(SOCKET));
    assert_eq!(config.persistence.path.as_deref(), Some(Path::new(DATA)));
    assert!(config.http.enabled);
    assert_eq!(config.http.listen.to_string(), "0.0.0.0:8099");
    let socket_dir = Path::new(SOCKET).parent().unwrap().to_str().unwrap();
    let ipc = ("cache-ipc", socket_dir, false);
    assert_eq!(compose.mounts("cache"), [ipc, ("cache-data", DATA, false)]);
}

#[test]
fn whoever_shares_the_socket_volume_reads_it_only_and_runs_as_65532() {
    let compose = Compose::load();
    let socket_dir = Path::new(SOCKET).parent().unwrap().to_str().unwrap();
    let mut clients = Vec::new();
    for service in compose.keys("services") {
        let mut ipc = compose.mounts(service);
        ipc.retain(|&(volume, _, _)| volume == "cache-ipc");
        if service == "cache" || ipc.is_empty() {
            continue;
        }
        // No mesmo lugar, sem trocar nem apagar o socket, com o uid do dono.
        assert_eq!(ipc, [("cache-ipc", socket_dir, true)], "{service}");
        let user = compose.one(&format!("services.{service}.user"));
        assert_eq!(user, "65532:65532", "{service}");
        clients.push(service);
    }
    assert_eq!(clients, ["ping"]);
    // O PING do HEALTHCHECK, de outro container e sem rede: só o volume liga.
    assert_eq!(compose.one("services.ping.image"), IMAGE);
    assert_eq!(compose.config("ping").ipc.socket_path, Path::new(SOCKET));
    let entrypoint = compose.one("services.ping.entrypoint");
    let entrypoint: Vec<String> = serde_json::from_str(entrypoint).unwrap();
    assert_eq!(entrypoint, [binary(), "--ping".to_owned()]);
    assert_eq!(compose.one("services.ping.network_mode"), "none");
    let condition = compose.one("services.ping.depends_on.cache.condition");
    assert_eq!(condition, "service_healthy");
}

#[test]
fn the_volumes_are_named_and_the_cache_network_stays_internal() {
    let compose = Compose::load();
    // O CI acha o volume do socket pelo projeto: `semantic-cache_cache-ipc`.
    assert_eq!(compose.one("name"), "semantic-cache");
    let volumes = BTreeSet::from(["cache-ipc", "cache-data"]);
    assert_eq!(compose.keys("volumes"), volumes);
    for volume in volumes {
        // Sem `driver_opts`, que trocariam o volume por uma pasta do host.
        let options = compose.keys(&format!("volumes.{volume}"));
        assert!(options.is_empty(), "{volume}: {options:?}");
    }
    assert_eq!(compose.all("services.cache.networks"), ["internal"]);
    assert_eq!(compose.one("networks.internal.internal"), "true");
}

#[test]
fn no_service_keeps_privileges_publishes_ports_or_carries_secrets() {
    let compose = Compose::load();
    for service in compose.keys("services") {
        let at = |key: &str| format!("services.{service}.{key}");
        let mut unknown = compose.keys(&format!("services.{service}"));
        unknown.retain(|key| !KNOWN.split_whitespace().any(|known| known == *key));
        assert!(unknown.is_empty(), "{service}: {unknown:?}");
        assert_eq!(compose.one(&at("read_only")), "true", "{service}");
        assert_eq!(compose.all(&at("cap_drop")), ["ALL"], "{service}");
        let options = compose.all(&at("security_opt"));
        assert_eq!(options, ["no-new-privileges:true"], "{service}");
        // Nem a rede do host, onde o 0.0.0.0:8099 da API cairia.
        let network = compose.get(&at("network_mode"));
        let off_host = matches!(network, None | Some("none"));
        assert!(off_host, "{service}: {network:?}");
        // Segredo só pelo `.env` do compose do ecossistema, nunca escrito aqui.
        for var in compose.keys(&at("environment")) {
            let upper = var.to_ascii_uppercase();
            let secret = SECRETS.iter().any(|s| upper.contains(s));
            assert!(!secret, "{service}: {var}");
        }
    }
}

#[test]
fn the_reader_refuses_the_yaml_it_would_misread() {
    let odd = ["a: &x 1", "a: *x", "<<:\n  a: 1", "a: {b: 1}", "a: |"];
    let also = ["a: >-", "- b: 1", "---", "a:b"];
    for yaml in odd.into_iter().chain(also) {
        assert!(std::panic::catch_unwind(|| flat(yaml)).is_err(), "{yaml}");
    }
    let read = Compose(flat("a:\n  b: \"1:2\"\n  c:\n    - x:/y:ro\n  # d: 1\n"));
    assert_eq!((read.one("a.b"), read.all("a.c")), ("1:2", vec!["x:/y:ro"]));
    assert_eq!(read.keys("a"), BTreeSet::from(["b", "c"]));
}
