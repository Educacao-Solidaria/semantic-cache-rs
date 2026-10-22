//! O relatório do `scripts/pgo.sh` (`scripts/pgo-report.awk`) sobre saídas
//! sintéticas do bench `ipc`: mediana de cada versão, Δ pareado por rodada com
//! a contagem das rodadas em que o PGO ganhou, e erro (sem tabela pela metade)
//! quando falta linha, rodada ou número.
#![cfg(unix)]

use std::path::PathBuf;
use std::process::Command;

/// O começo da saída do bench, que o relatório ignora.
const HEADER: &str = "300 consultas por linha, kernel avx2\n\n\
    | índice | vetores | ponta a ponta p50 / p99 / máx | no servidor p50 / p99 | ok |\n\
    |---|---|---|---|---|\n";

/// Uma linha da tabela do bench: `key` = "índice | vetores" e os p50 de ponta
/// a ponta e no servidor, em ms.
fn row(key: &str, (e2e, srv): (f64, f64)) -> String {
    format!("| {key} | {e2e:.3} ms / 9.000 ms / 9.900 ms | {srv:.3} ms / 9.000 ms | sim |\n")
}

/// Um arquivo de rodada com o cabeçalho do bench e as linhas `rows`.
fn file(name: &str, rows: &str) -> (String, String) {
    (name.to_owned(), format!("{HEADER}{rows}"))
}

/// Os dois arquivos da rodada `r`: `base-<r>.txt` e `pgo-<r>.txt`.
fn round(r: u32, base: &str, pgo: &str) -> Vec<(String, String)> {
    vec![
        file(&format!("base-{r}.txt"), base),
        file(&format!("pgo-{r}.txt"), pgo),
    ]
}

/// O awk sobre `files`, gravados num diretório só deste caso: `Ok(stdout)` ou
/// `Err(stderr)`, e a falha não deixa nada no stdout.
fn report(case: &str, expect: u32, files: &[(String, String)]) -> Result<String, String> {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("pgo-report-{case}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/pgo-report.awk");
    let expect = format!("expect={expect}");
    let mut awk = Command::new("awk");
    awk.env("LC_ALL", "C").args(["-v", &expect, "-f", script]);
    for (name, text) in files {
        std::fs::write(dir.join(name), text).unwrap();
        awk.arg(dir.join(name));
    }
    let out = awk.output().expect("awk no PATH");
    let stdout = String::from_utf8(out.stdout).unwrap();
    if out.status.success() {
        return Ok(stdout);
    }
    assert_eq!(stdout, "", "{case}: tabela pela metade");
    Err(String::from_utf8(out.stderr).unwrap())
}

/// As linhas da tabela, sem o cabeçalho.
fn rows(table: &str) -> Vec<&str> {
    table.lines().skip(2).collect()
}

#[test]
fn each_round_is_paired_and_the_delta_is_not_the_ratio_of_the_medians() {
    // Δ por rodada: −10%, +10% e −20% → mediana −10%, mais rápido em 2 de 3; a
    // razão das medianas daria +10%.
    let key = "flat_f32 | 1000";
    let files = [
        round(1, &row(key, (0.1, 0.05)), &row(key, (0.09, 0.04))),
        round(2, &row(key, (0.2, 0.05)), &row(key, (0.22, 0.04))),
        round(3, &row(key, (0.3, 0.05)), &row(key, (0.24, 0.04))),
    ]
    .concat();
    assert_eq!(
        rows(&report("paired", 3, &files).unwrap()),
        ["| flat_f32 | 1000 | 0.200 → 0.220 ms | -10.0% (2/3) | 0.050 → 0.040 ms | -20.0% (3/3) |"]
    );
}

#[test]
fn an_even_count_takes_the_middle_two_and_a_zero_base_has_no_delta() {
    // De ponta a ponta, Δ 0 (empate não conta como mais rápido) e −50%.
    let key = "flat_i8 | 0";
    let files = [
        round(1, &row(key, (0.1, 0.0)), &row(key, (0.1, 0.01))),
        round(2, &row(key, (0.3, 0.0)), &row(key, (0.15, 0.01))),
    ]
    .concat();
    assert_eq!(
        rows(&report("even", 2, &files).unwrap()),
        ["| flat_i8 | 0 | 0.200 → 0.125 ms | -25.0% (1/2) | 0.000 → 0.010 ms | — |"]
    );
}

#[test]
fn a_missing_line_round_or_number_is_an_error_and_never_half_a_table() {
    let (f32, i8) = (
        row("flat_f32 | 0", (0.1, 0.03)),
        row("flat_i8 | 0", (0.1, 0.03)),
    );
    let both = format!("{f32}{i8}");
    let comma = both.replace("0.100 ms /", "0,100 ms /");
    // A repetida vem no fim do último arquivo: com os dados completos, só o
    // registro da falha impede o END de imprimir a tabela.
    let cases = [
        (
            "line",
            1,
            round(1, &both, &f32),
            "a rodada 1 não tem a linha flat_i8 | 0",
        ),
        (
            "round",
            2,
            [round(1, &both, &both), vec![file("base-2.txt", &both)]].concat(),
            "a rodada 2",
        ),
        ("number", 1, round(1, &comma, &both), "número ilegível"),
        (
            "twice",
            1,
            round(1, &both, &format!("{both}{f32}")),
            "linha repetida",
        ),
        ("name", 1, vec![file("base.txt", &both)], "fora do padrão"),
        ("empty", 1, round(1, "", ""), "nenhuma linha"),
        ("usage", 0, round(1, &both, &both), "uso:"),
    ];
    for (case, expect, files, message) in cases {
        let err = report(case, expect, &files).unwrap_err();
        assert!(err.contains(message), "{case}: {err}");
    }
}
