//! Monta fixtures de embeddings no formato de `semantic_cache_testkit::datasets`.
//!
//! Não faz rede nem lê chave: o `scripts/capture-embeddings.sh` chama o
//! `OpenRouter` com `curl` e passa a resposta por aqui.
//!
//! Subcomandos: `request` (corpo da requisição), `fixture` (resposta no stdin
//! → fixture capturada) e `synthetic` (placeholder sintético).
//! `textos.tsv`: uma entrada por linha, `grupo<TAB>texto`.

use semantic_cache_testkit::datasets::{
    EmbeddingFixture, FixtureEntry, FixtureHeader, Provenance, SCHEMA_VERSION,
};
use semantic_cache_testkit::fixtures::VectorGen;
use serde::Deserialize;
use serde_json::json;

/// Similaridade entre paráfrases do placeholder: acima do limiar de 0,92.
const SYNTHETIC_PARAPHRASE_COSINE: f32 = 0.95;

/// Resposta de `POST /api/v1/embeddings`.
#[derive(Deserialize)]
struct Response {
    data: Vec<Item>,
}

#[derive(Deserialize)]
struct Item {
    index: usize,
    embedding: Vec<f32>,
}

type Texts = Vec<(String, String)>;

fn read_texts(path: &str) -> Result<Texts, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split_once('\t')
                .map(|(g, t)| (g.to_owned(), t.to_owned()))
                .ok_or_else(|| format!("{path}: linha sem TAB: {l}"))
        })
        .collect()
}

fn build(header: FixtureHeader, texts: Texts, embeddings: Vec<Vec<f32>>) -> Result<String, String> {
    let entries = texts
        .into_iter()
        .zip(embeddings)
        .enumerate()
        .map(|(i, ((group, text), embedding))| FixtureEntry {
            id: format!("{group}-{i}"),
            group,
            text,
            embedding,
        })
        .collect();
    let fixture = EmbeddingFixture { header, entries };
    // Passa pelo loader: o que sai daqui sempre carrega.
    EmbeddingFixture::parse(&fixture.to_jsonl()).map_err(|e| e.to_string())?;
    Ok(fixture.to_jsonl())
}

fn captured(model: &str, texts: Texts) -> Result<String, String> {
    let body = std::io::read_to_string(std::io::stdin()).map_err(|e| e.to_string())?;
    let mut resp: Response =
        serde_json::from_str(&body).map_err(|e| format!("resposta do OpenRouter: {e}"))?;
    if resp.data.len() != texts.len() {
        return Err(format!(
            "{} embeddings para {} textos",
            resp.data.len(),
            texts.len()
        ));
    }
    resp.data.sort_by_key(|item| item.index);
    if resp
        .data
        .iter()
        .enumerate()
        .any(|(i, item)| item.index != i)
    {
        return Err("índices da resposta não são 0..n".to_owned());
    }
    // Vazio vira dimensions = 0, que o loader rejeita em `build`.
    let dimensions = resp.data.first().map_or(0, |item| item.embedding.len());
    let header = FixtureHeader {
        schema_version: SCHEMA_VERSION,
        provenance: Provenance::Captured,
        model: model.to_owned(),
        dimensions,
        source: "OpenRouter POST /api/v1/embeddings".to_owned(),
    };
    build(
        header,
        texts,
        resp.data.into_iter().map(|i| i.embedding).collect(),
    )
}

fn synthetic(dim: usize, seed: u64, texts: Texts) -> Result<String, String> {
    let mut g = VectorGen::new(seed);
    let mut bases: Vec<(String, semantic_cache_rs::Vector)> = Vec::new();
    let mut embeddings = Vec::new();
    for (group, _) in &texts {
        let v = if let Some((_, base)) = bases.iter().find(|(g2, _)| g2 == group) {
            g.with_cosine(base, SYNTHETIC_PARAPHRASE_COSINE)
        } else {
            let base = g.unit(dim);
            bases.push((group.clone(), base.clone()));
            base
        };
        embeddings.push(v.data);
    }
    let header = FixtureHeader {
        schema_version: SCHEMA_VERSION,
        provenance: Provenance::Synthetic,
        model: "synthetic".to_owned(),
        dimensions: dim,
        source: format!(
            "SINTÉTICO, não é embedding de modelo: VectorGen semente {seed}, \
             paráfrases com cosseno {SYNTHETIC_PARAPHRASE_COSINE}"
        ),
    };
    build(header, texts, embeddings)
}

fn parse<T: std::str::FromStr>(s: &str) -> Result<T, String> {
    s.parse().map_err(|_| format!("número inválido: {s}"))
}

fn run(args: &[String]) -> Result<String, String> {
    match args {
        [cmd, model, texts] if cmd == "request" => {
            let input: Vec<String> = read_texts(texts)?.into_iter().map(|(_, t)| t).collect();
            // Só provedores que não retêm nem treinam com os textos enviados.
            let provider = json!({ "data_collection": "deny" });
            Ok(json!({ "model": model, "input": input, "provider": provider }).to_string())
        }
        [cmd, model, texts] if cmd == "fixture" => captured(model, read_texts(texts)?),
        [cmd, dim, seed, texts] if cmd == "synthetic" => {
            synthetic(parse(dim)?, parse(seed)?, read_texts(texts)?)
        }
        _ => Err(
            "uso: capture-embeddings request|fixture <modelo> <textos.tsv> \
                  | synthetic <dim> <semente> <textos.tsv>"
                .to_owned(),
        ),
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    print!("{}", run(&args)?);
    Ok(())
}
