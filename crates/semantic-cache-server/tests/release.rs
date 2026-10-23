//! O workflow de release (`.github/workflows/release.yml`) contra o contrato da
//! Fase 5, §6: publica só no push de uma tag `v*`, e o único job com escrita é
//! o que publica, sem checkout; no PR, todo o resto roda (os dois alvos musl,
//! cada binário de pé no runner da arquitetura dele). Estático, página do
//! jemalloc, PING e SIGTERM só o runner mostra: o próprio workflow confere.

use std::collections::{BTreeSet, HashMap, HashSet};

const RELEASE: &str = include_str!("../../../.github/workflows/release.yml");
/// O job que publica: o único com escrita, e só no push de uma tag `v*`.
const PUBLISH: &str = "publish";
const ON_TAG: &str = "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')";
/// As actions que o repositório já usa; outra entra revista aqui.
const ACTIONS: [&str; 4] = [
    "actions/checkout@v4",
    "actions/download-artifact@v4",
    "actions/upload-artifact@v4",
    "dtolnay/rust-toolchain@stable",
];
const UNREAD: &str = "YAML fora do que o teste lê";

/// Cada chave com o caminho até ela, em que item de lista vira o índice
/// (`jobs.build.steps.3.run`), e o valor cru: com as aspas, e o texto inteiro
/// num `run: |`. Chave que abre bloco entra sem valor.
type Flat = Vec<(String, Option<String>)>;

/// Lê só o YAML que o workflow usa: mapa e lista de mapas em bloco, texto em
/// bloco `|` e lista em fluxo como valor cru (`['v*']`). Âncora, alias, tag,
/// mapa em fluxo, texto dobrado, item escalar, escalar em mais de uma linha e
/// linha sem `chave:` derrubam o teste em vez de serem lidos errado.
fn flat(yaml: &str) -> Flat {
    let mut open: Vec<(usize, String)> = Vec::new();
    let mut items: HashMap<String, usize> = HashMap::new();
    let mut flat = Vec::new();
    // Depois de um escalar, uma linha mais funda seria a continuação dele.
    let mut deepest = usize::MAX;
    let mut lines = yaml.lines().peekable();
    while let Some(line) = lines.next() {
        let mut text = line.trim_start();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let mut indent = depth(line);
        assert!(indent <= deepest, "{UNREAD}: {line}");
        open.retain(|&(at, _)| at < indent);
        if let Some(item) = text.strip_prefix("- ") {
            let n = items.entry(join(&open)).or_default();
            open.push((indent, n.to_string()));
            *n += 1;
            (indent, text) = (indent + 2, item);
        }
        let pair = text.split_once(": ");
        let pair = pair.or_else(|| Some((text.strip_suffix(':')?, "")));
        let (key, value) = pair.unwrap_or_else(|| panic!("{UNREAD}: {line}"));
        let value = value.trim();
        let odd = value.starts_with(['&', '*', '!', '{', '>']) || key == "<<";
        assert!(!odd, "{UNREAD}: {line}");
        open.push((indent, key.to_owned()));
        let at = join(&open);
        deepest = indent;
        let value = match value {
            "" => {
                deepest = usize::MAX;
                flat.push((at, None));
                continue;
            }
            "|" | "|-" => {
                let inside = |l: &&str| l.trim().is_empty() || depth(l) > indent;
                let block: Vec<&str> = std::iter::from_fn(|| lines.next_if(inside)).collect();
                block.join("\n")
            }
            value => value.to_owned(),
        };
        open.pop();
        flat.push((at, Some(value)));
    }
    flat
}

fn depth(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

fn join(open: &[(usize, String)]) -> String {
    let keys: Vec<&str> = open.iter().map(|(_, key)| key.as_str()).collect();
    keys.join(".")
}

struct Workflow(Flat);

impl Workflow {
    fn load() -> Self {
        let flat = flat(RELEASE);
        // Chave repetida: o Actions leria uma, o teste a outra.
        let mut seen = HashSet::new();
        for (at, _) in &flat {
            assert!(seen.insert(at.as_str()), "{at}: repetida");
        }
        Self(flat)
    }

    /// Os valores dos caminhos que `pick` aceita, com o caminho, na ordem.
    fn values(&self, pick: impl Fn(&str) -> bool) -> Vec<(&str, &str)> {
        let found = self.0.iter().filter(|(at, _)| pick(at));
        let found = found.filter_map(|(at, value)| Some((at, value.as_deref()?)));
        found.map(|(at, value)| (at.as_str(), value)).collect()
    }

    /// Os valores da chave `key` em qualquer ponto do arquivo.
    fn every(&self, key: &str) -> Vec<(&str, &str)> {
        self.values(|at| at.rsplit('.').next() == Some(key))
    }

    fn get(&self, path: &str) -> Option<&str> {
        let found = self.values(|at| at == path);
        found.first().map(|&(_, value)| value)
    }

    fn has(&self, path: &str) -> bool {
        self.0.iter().any(|(at, _)| at == path)
    }

    /// As chaves logo abaixo de `path`.
    fn keys(&self, path: &str) -> BTreeSet<&str> {
        let prefix = format!("{path}.");
        let paths = self.0.iter().map(|(at, _)| at.as_str());
        let below = paths.filter_map(|at| at.strip_prefix(prefix.as_str()));
        below.map(|rest| rest.split('.').next().unwrap()).collect()
    }
}

#[test]
fn it_runs_on_v_tags_and_on_every_pull_request() {
    let w = Workflow::load();
    assert_eq!(w.keys("on"), BTreeSet::from(["pull_request", "push"]));
    assert_eq!(w.keys("on.push"), BTreeSet::from(["tags"]));
    assert_eq!(w.get("on.push.tags"), Some("['v*']"));
    // Filtro (`paths`, `types`) deixaria PR sem a prova do pipeline.
    assert!(w.has("on.pull_request"));
    assert_eq!(w.keys("on.pull_request"), BTreeSet::new());
}

#[test]
fn only_the_publish_job_writes_and_only_on_a_tag_push() {
    let w = Workflow::load();
    assert_eq!(w.keys("permissions"), BTreeSet::from(["contents"]));
    assert_eq!(w.get("permissions.contents"), Some("read"));
    for job in w.keys("jobs") {
        let permissions = format!("jobs.{job}.permissions");
        if job != PUBLISH {
            assert!(!w.has(&permissions), "{job}: permissões só no {PUBLISH}");
            continue;
        }
        assert_eq!(w.keys(&permissions), BTreeSet::from(["contents"]));
        assert_eq!(w.get(&format!("{permissions}.contents")), Some("write"));
    }
    // O único `if` é o dele: o resto roda no PR, e nenhum passo fica de fora
    // nem tem a falha engolida.
    assert_eq!(w.every("if"), [("jobs.publish.if", ON_TAG)]);
    let swallowed = w.every("continue-on-error");
    assert_eq!(swallowed, Vec::<(&str, &str)>::new());
}

#[test]
fn the_publish_job_ships_what_the_other_jobs_checked_without_repo_code() {
    let w = Workflow::load();
    assert_eq!(w.get("jobs.publish.needs"), Some("checksums"));
    assert_eq!(w.get("jobs.checksums.needs"), Some("build"));
    let steps = w.every("uses").into_iter();
    let publish = steps.filter(|(at, _)| at.starts_with("jobs.publish."));
    let actions: Vec<&str> = publish.map(|(_, action)| action).collect();
    assert_eq!(actions, ["actions/download-artifact@v4"]);
}

#[test]
fn the_build_covers_both_musl_targets_with_the_jemalloc_page_of_each() {
    let w = Workflow::load();
    let matrix = "jobs.build.strategy.matrix";
    // Outro eixo na matriz multiplicaria os alvos.
    assert_eq!(w.keys(matrix), BTreeSet::from(["include"]));
    let item = |i: &str, key: &str| w.get(&format!("{matrix}.include.{i}.{key}"));
    let mut targets: Vec<_> = (w.keys(&format!("{matrix}.include")).into_iter())
        .map(|i| (item(i, "target"), item(i, "lg-page")))
        .collect();
    targets.sort_unstable();
    let aarch64 = (Some("aarch64-unknown-linux-musl"), Some("16"));
    let x86_64 = (Some("x86_64-unknown-linux-musl"), Some("12"));
    assert_eq!(targets, [aarch64, x86_64]);
}

#[test]
fn every_job_has_a_deadline_and_no_script_expands_an_expression() {
    let w = Workflow::load();
    for job in w.keys("jobs") {
        let minutes = w.get(&format!("jobs.{job}.timeout-minutes"));
        let minutes: u32 = minutes.and_then(|m| m.parse().ok()).expect(job);
        assert!((1..=30).contains(&minutes), "{job}: {minutes} min");
    }
    // `${{ }}` num `run:` vira texto do script antes do bash: o nome da tag,
    // texto de quem a criou, entraria cru. Só por `env:`.
    for (at, script) in w.every("run") {
        assert!(!script.contains("${{"), "{at}");
    }
}

#[test]
fn the_actions_are_the_ones_the_repo_already_uses_and_no_secret_is_read() {
    let w = Workflow::load();
    for (at, action) in w.every("uses") {
        assert!(ACTIONS.contains(&action), "{at}: {action}");
    }
    // O publish escreve com o token do próprio job (`github.token`).
    assert!(!RELEASE.contains("secrets"));
}
