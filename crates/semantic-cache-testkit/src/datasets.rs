//! Fixtures de embeddings de modelos (via `OpenRouter`) para validação cruzada.
//!
//! Formato JSON Lines, um arquivo por modelo: a 1ª linha é o cabeçalho
//! ([`FixtureHeader`]); cada linha seguinte é uma entrada ([`FixtureEntry`]),
//! o que mantém o diff legível quando um embedding muda.
//! Entradas do mesmo `group` são paráfrases: num embedding real, a
//! similaridade entre elas deve passar do limiar de hit do cache.
//!
//! O campo `provenance` separa o que veio de um modelo (`captured`) do que foi
//! gerado (`synthetic`). Teste que afirma algo sobre o comportamento de um
//! modelo precisa checar [`EmbeddingFixture::is_captured`].
//!
//! ponytail: a única fixture versionada hoje é sintética
//! ([`synthetic_placeholder`], 64 dimensões, gerada pelo `VectorGen`) — o teto
//! é que ela valida formato e loader, não o comportamento de modelo nenhum.
//! Upgrade: rodar `scripts/capture-embeddings.sh` com uma chave do `OpenRouter`
//! (ex. `openai/text-embedding-3-small`) e versionar o `.jsonl` gerado, que
//! sai com `provenance: captured`.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

/// Versão do formato que este loader entende.
pub const SCHEMA_VERSION: u32 = 1;

/// De onde vieram os vetores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provenance {
    /// Resposta real de um modelo de embedding.
    Captured,
    /// Gerado localmente; não representa modelo nenhum.
    Synthetic,
}

/// Primeira linha do arquivo.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureHeader {
    pub schema_version: u32,
    pub provenance: Provenance,
    /// Id do modelo no `OpenRouter` (`openai/text-embedding-3-small`) ou
    /// `synthetic` para fixtures geradas.
    pub model: String,
    pub dimensions: usize,
    /// Como os vetores foram obtidos (endpoint ou gerador e semente).
    pub source: String,
}

/// Um texto e o embedding dele.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureEntry {
    pub id: String,
    /// Entradas com o mesmo grupo são paráfrases umas das outras.
    pub group: String,
    pub text: String,
    pub embedding: Vec<f32>,
}

/// Fixture carregada e validada.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingFixture {
    pub header: FixtureHeader,
    pub entries: Vec<FixtureEntry>,
}

/// Falha ao ler ou validar uma fixture. `line` conta a partir de 1.
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("linha {line}: JSON inválido: {source}")]
    Json {
        line: usize,
        source: serde_json::Error,
    },
    #[error("linha {line}: {reason}")]
    Invalid { line: usize, reason: String },
}

fn invalid<T>(line: usize, reason: impl Into<String>) -> Result<T, DatasetError> {
    Err(DatasetError::Invalid {
        line,
        reason: reason.into(),
    })
}

impl EmbeddingFixture {
    /// Lê e valida o conteúdo de um arquivo `.jsonl`. Linhas em branco são
    /// ignoradas.
    ///
    /// # Errors
    ///
    /// [`DatasetError`] na primeira linha inválida.
    pub fn parse(jsonl: &str) -> Result<Self, DatasetError> {
        let mut lines = jsonl
            .lines()
            .enumerate()
            .map(|(i, l)| (i + 1, l))
            .filter(|(_, l)| !l.trim().is_empty());
        let Some((line, first)) = lines.next() else {
            return invalid(1, "fixture vazia");
        };
        let header: FixtureHeader =
            serde_json::from_str(first).map_err(|source| DatasetError::Json { line, source })?;
        if header.schema_version != SCHEMA_VERSION {
            return invalid(
                line,
                format!("schema_version {} não suportada", header.schema_version),
            );
        }
        if header.dimensions == 0 {
            return invalid(line, "dimensions = 0");
        }
        let mut ids = HashSet::new();
        let mut entries = Vec::new();
        for (line, text) in lines {
            let entry: FixtureEntry =
                serde_json::from_str(text).map_err(|source| DatasetError::Json { line, source })?;
            let dims = entry.embedding.len();
            if dims != header.dimensions {
                return invalid(
                    line,
                    format!("{dims} dimensões, cabeçalho declara {}", header.dimensions),
                );
            }
            if !entry.embedding.iter().all(|x| x.is_finite()) {
                return invalid(line, "valor não finito");
            }
            if !ids.insert(entry.id.clone()) {
                return invalid(line, format!("id repetido `{}`", entry.id));
            }
            entries.push(entry);
        }
        if entries.is_empty() {
            return invalid(line, "fixture sem entradas");
        }
        Ok(Self { header, entries })
    }

    /// Serializa no formato de [`EmbeddingFixture::parse`].
    #[must_use]
    pub fn to_jsonl(&self) -> String {
        let mut out = to_line(&self.header);
        for entry in &self.entries {
            out.push_str(&to_line(entry));
        }
        out
    }

    /// `true` só para vetores vindos de um modelo real.
    #[must_use]
    pub fn is_captured(&self) -> bool {
        self.header.provenance == Provenance::Captured
    }
}

fn to_line(value: &impl Serialize) -> String {
    let mut line = serde_json::to_string(value).expect("tipos da fixture sempre serializam");
    line.push('\n');
    line
}

/// Fixture sintética versionada em `fixtures/embeddings/`. **Não** contém
/// embeddings de modelo: serve para testar formato e loader.
///
/// # Panics
///
/// Se o arquivo versionado estiver inválido (o teste deste módulo pega antes).
#[must_use]
pub fn synthetic_placeholder() -> EmbeddingFixture {
    EmbeddingFixture::parse(include_str!(
        "../fixtures/embeddings/synthetic-placeholder.jsonl"
    ))
    .expect("fixture versionada válida")
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_cache_rs::Vector;

    const HEADER: &str = r#"{"schema_version":1,"provenance":"captured","model":"m","dimensions":2,"source":"teste"}"#;

    fn fixture(entries: &[&str]) -> String {
        format!("{HEADER}\n{}", entries.join("\n"))
    }

    #[test]
    fn placeholder_is_labeled_synthetic_and_grouped() {
        let f = synthetic_placeholder();
        assert!(!f.is_captured());
        assert_eq!(f.header.model, "synthetic");
        // Paráfrases (mesmo grupo) acima do limiar de hit; grupos distintos abaixo.
        let v = |e: &FixtureEntry| Vector::new(e.embedding.clone()).unwrap();
        for (i, a) in f.entries.iter().enumerate() {
            for b in &f.entries[i + 1..] {
                let cos = v(a).cosine_similarity(&v(b)).unwrap();
                assert!((a.group == b.group) == (cos >= 0.92), "{}: {cos}", b.id);
            }
        }
    }

    #[test]
    fn roundtrip_preserves_everything() {
        let f = synthetic_placeholder();
        assert_eq!(EmbeddingFixture::parse(&f.to_jsonl()).unwrap(), f);
    }

    #[test]
    fn rejects_invalid_files() {
        let e = |s: &str| EmbeddingFixture::parse(s).unwrap_err().to_string();
        let ok = r#"{"id":"a","group":"g","text":"t","embedding":[1.0,0.0]}"#;
        assert_eq!(e(""), "linha 1: fixture vazia");
        assert_eq!(e(HEADER), "linha 1: fixture sem entradas");
        assert!(e(&HEADER.replace(":1,", ":2,")).contains("schema_version 2"));
        assert!(e(&HEADER.replace(":2,", ":0,")).contains("dimensions = 0"));
        let short = fixture(&[&ok.replace("[1.0,0.0]", "[1.0]")]);
        assert_eq!(e(&short), "linha 2: 1 dimensões, cabeçalho declara 2");
        // 1e39 estoura o f32 e vira infinito.
        let inf = ok.replace("[1.0,", "[1e39,");
        assert_eq!(e(&fixture(&[&inf])), "linha 2: valor não finito");
        assert_eq!(e(&fixture(&[ok, ok])), "linha 3: id repetido `a`");
        assert!(e(&fixture(&[ok, "{"])).starts_with("linha 3: JSON inválido"));
        assert!(e(&fixture(&[&ok.replace('}', r#","extra":1}"#)])).contains("unknown field"));
    }
}
