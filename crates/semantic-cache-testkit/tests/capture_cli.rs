//! O binário `capture-embeddings` de ponta a ponta, sem rede: a resposta do
//! `OpenRouter` é simulada no stdin.

use std::io::Write;
use std::process::{Command, Output, Stdio};

use semantic_cache_testkit::assert_vec_near;
use semantic_cache_testkit::datasets::{synthetic_placeholder, EmbeddingFixture};

const TEXTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/embeddings/texts.tsv");

fn run(args: &[&str], stdin: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_capture-embeddings"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn request_body_carries_model_and_texts() {
    let out = run(&["request", "openai/text-embedding-3-small", TEXTS], "");
    assert!(out.status.success());
    let body: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(body["model"], "openai/text-embedding-3-small");
    assert_eq!(body["input"].as_array().unwrap().len(), 6);
    assert_eq!(body["provider"]["data_collection"], "deny");
    assert_eq!(body["input"][0], "Qual é a capital da França?");
}

#[test]
fn fixture_from_response_is_captured_and_ordered_by_index() {
    // Resposta fora de ordem: a fixture segue `index`, não a ordem do array.
    let data: Vec<_> = (0..6u8)
        .rev()
        .map(|i| serde_json::json!({ "index": i, "embedding": [f32::from(i) + 1.0, 0.5, 0.25] }))
        .collect();
    let resp = serde_json::json!({ "data": data }).to_string();
    let out = run(&["fixture", "m/x", TEXTS], &resp);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let f = EmbeddingFixture::parse(std::str::from_utf8(&out.stdout).unwrap()).unwrap();
    assert!(f.is_captured());
    assert_eq!((f.header.model.as_str(), f.header.dimensions), ("m/x", 3));
    assert_eq!(f.entries[0].embedding[0], 1.0);
    assert_eq!(f.entries[5].id, "bolo-cenoura-5");
}

#[test]
fn versioned_placeholder_is_reproducible() {
    let out = run(&["synthetic", "64", "1501", TEXTS], "");
    let regenerated = EmbeddingFixture::parse(std::str::from_utf8(&out.stdout).unwrap()).unwrap();
    let versioned = synthetic_placeholder();
    assert_eq!(regenerated.header, versioned.header);
    // Tolerância e não igualdade de bits: `ln`/`cos` da libm variam entre plataformas.
    for (a, b) in regenerated.entries.iter().zip(&versioned.entries) {
        assert_eq!((&a.id, &a.text), (&b.id, &b.text));
        assert_vec_near!(a.embedding, b.embedding, abs = 1e-6);
    }
}
