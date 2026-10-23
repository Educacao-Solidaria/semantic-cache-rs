//! A auditoria de dependências (`deny.toml` na raiz) contra o contrato da Fase
//! 5, §6: reprova vulnerabilidade, crate sem manutenção, "unsound", versão
//! retirada, licença fora da lista e origem fora do crates.io, nenhuma regra
//! desligada inteira, e exceção só no `deny.toml`, uma por achado, com motivo
//! e data de revisão. O workflow (`.github/workflows/audit.yml`) passa por todo
//! Cargo.lock do repositório. Os achados mesmo (o banco de avisos muda todo
//! dia) e o prazo das exceções só o workflow vê.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use toml::{Table, Value};

const DENY: &str = include_str!("../../../deny.toml");
const WORKFLOW: &str = include_str!("../../../.github/workflows/audit.yml");
/// As chaves de cada seção. Outra (`exclude`, `targets`, `skip`, `allow-git`,
/// `db-urls`...) muda o grafo ou a checagem: entra revista aqui.
const SECTIONS: [(&str, &[&str]); 4] = [
    ("graph", &["all-features"]),
    (
        "advisories",
        &[
            "ignore",
            "unmaintained",
            "unsound",
            "unused-ignored-advisory",
            "yanked",
        ],
    ),
    (
        "licenses",
        &[
            "allow",
            "clarify",
            "exceptions",
            "include-dev",
            "private",
            "unused-allowed-license",
            "unused-license-exception",
        ],
    ),
    ("sources", &["unknown-git", "unknown-registry"]),
];
/// As licenças permissivas que a lista global pode ter. Outra (copyleft, como
/// GPL e MPL, ou que não seja de código aberto) só como exceção de um crate.
const PERMISSIVE: [&str; 13] = [
    "0BSD",
    "Apache-2.0",
    "Apache-2.0 WITH LLVM-exception",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "BSL-1.0",
    "CC0-1.0",
    "ISC",
    "MIT",
    "MIT-0",
    "Unicode-3.0",
    "Unlicense",
    "Zlib",
];

fn deny() -> Table {
    DENY.parse().expect("deny.toml")
}

/// `n` algarismos ASCII, nem mais nem menos.
fn digits(s: &str, n: usize) -> bool {
    s.len() == n && s.bytes().all(|b| b.is_ascii_digit())
}

/// Se o texto marca a revisão como "revisar em AAAA-MM-DD", numa linha só e
/// com mês e dia possíveis: o que o passo "exceções no prazo" do workflow lê.
fn review_date(text: &str) -> bool {
    text.split("revisar em ").skip(1).any(|after| {
        let date = after.get(..10).unwrap_or_default();
        match date.split('-').collect::<Vec<_>>()[..] {
            [y, m, d] if digits(y, 4) && digits(m, 2) && digits(d, 2) => {
                (1..=12).contains(&m.parse::<u8>().unwrap())
                    && (1..=31).contains(&d.parse::<u8>().unwrap())
            }
            _ => false,
        }
    })
}

/// As linhas de comentário logo acima de cada linha `header` do deny.toml.
fn comments_above(header: &str) -> Vec<Vec<&'static str>> {
    let lines: Vec<&str> = DENY.lines().collect();
    let at = (0..lines.len()).filter(|&i| lines[i] == header);
    let above = |i: usize| lines[..i].iter().rev().take_while(|l| l.starts_with('#'));
    at.map(|i| above(i).copied().collect()).collect()
}

/// Os workspaces do `WORKSPACES: >-` do workflow, um por linha.
fn workspaces() -> BTreeSet<String> {
    let indent = |l: &str| l.len() - l.trim_start().len();
    let mut lines = WORKFLOW
        .lines()
        .skip_while(|l| l.trim() != "WORKSPACES: >-");
    let head = indent(lines.next().expect("WORKSPACES no workflow"));
    let block = lines.take_while(|l| indent(l) > head && !l.trim().is_empty());
    block.map(|l| l.trim().to_owned()).collect()
}

#[test]
fn no_check_is_switched_off() {
    let deny = deny();
    let sections: Vec<&str> = deny.keys().map(String::as_str).collect();
    assert_eq!(sections, ["advisories", "graph", "licenses", "sources"]);
    for (section, known) in SECTIONS {
        for key in deny[section].as_table().expect(section).keys() {
            let known = known.contains(&key.as_str());
            assert!(known, "[{section}] {key}: chave nova entra revista aqui");
        }
    }
    let strict: [(&str, &str, Value); 7] = [
        ("graph", "all-features", true.into()),
        ("advisories", "unmaintained", "all".into()),
        ("advisories", "unsound", "all".into()),
        ("advisories", "yanked", "deny".into()),
        ("licenses", "include-dev", true.into()),
        ("sources", "unknown-registry", "deny".into()),
        ("sources", "unknown-git", "deny".into()),
    ];
    for (section, key, value) in strict {
        assert_eq!(deny[section].get(key), Some(&value), "[{section}] {key}");
    }
    // `registries`/`ignore-sources` pulariam a licença de crates de terceiros.
    let private: Table = "ignore = true".parse().unwrap();
    let only_ours = deny["licenses"]["private"].as_table() == Some(&private);
    assert!(
        only_ours,
        "licenses.private: só os crates com publish = false"
    );
}

#[test]
fn only_permissive_licenses_are_allowed_for_every_crate() {
    let deny = deny();
    let allow = deny["licenses"]["allow"]
        .as_array()
        .expect("licenses.allow");
    for license in allow.iter().map(|l| l.as_str().expect("licença em texto")) {
        let permissive = PERMISSIVE.contains(&license);
        assert!(permissive, "{license}: só como exceção de um crate");
    }
}

#[test]
fn every_exception_has_a_reason_and_a_review_date() {
    let deny = deny();
    let ignore = deny["advisories"]["ignore"].as_array().expect("ignore");
    for entry in ignore {
        let entry = entry.as_table().expect("exceção em tabela, com motivo");
        let keys: Vec<&str> = entry.keys().map(String::as_str).collect();
        assert_eq!(keys, ["id", "reason"], "{entry}: outra forma entra revista");
        let id = entry["id"].as_str().expect("id");
        let number = id.strip_prefix("RUSTSEC-").and_then(|n| n.split_once('-'));
        assert!(
            number.is_some_and(|(y, n)| digits(y, 4) && digits(n, 4)),
            "{id}"
        );
        let reason = entry["reason"].as_str().expect("reason");
        assert!(
            review_date(reason),
            "{id}: o motivo sem \"revisar em AAAA-MM-DD\""
        );
    }
    // Licença não tem `reason` no cargo-deny: o motivo e a data vão no
    // comentário logo acima do `[[licenses.*]]`, a única forma aceita.
    for key in ["clarify", "exceptions"] {
        let header = format!("[[licenses.{key}]]");
        let entries = deny["licenses"].get(key).and_then(Value::as_array);
        let blocks = comments_above(&header);
        assert_eq!(blocks.len(), entries.map_or(0, Vec::len), "{header}");
        for comment in blocks {
            assert!(comment.iter().any(|l| review_date(l)), "{header} sem data");
        }
    }
}

#[test]
fn the_review_date_is_a_whole_date_on_one_line() {
    assert!(review_date(
        "bincode 1.3.3 sem manutenção; revisar em 2027-01-22"
    ));
    for wrong in [
        "revisar em 2027-1-22",
        "revisar em 2027-13-01",
        "revisar em 2027-01-00",
        "revisar em 22/01/2027",
        "revisar em\n2027-01-22",
        "Revisar em 2027-01-22",
        "revisar em ",
        "",
    ] {
        assert!(!review_date(wrong), "{wrong:?}");
    }
}

#[test]
fn every_lockfile_of_the_repository_is_audited() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut locked = BTreeSet::new();
    for dir in crates.read_dir().unwrap().map(|d| d.unwrap().path()) {
        if dir.join("Cargo.lock").is_file() {
            let name = dir.file_name().unwrap().to_str().unwrap();
            locked.insert(format!("crates/{name}"));
        }
    }
    assert_eq!(workspaces(), locked);
    // O crate da raiz não tem lock versionado: as dependências dele entram
    // pelos locks dos workspaces que dependem dele por path.
    let root = locked.iter().any(|w| {
        let lock = fs::read_to_string(crates.join("..").join(w).join("Cargo.lock"));
        lock.unwrap().contains("\nname = \"semantic-cache-rs\"\n")
    });
    assert!(root, "nenhum lock auditado traz o crate da raiz");
}
