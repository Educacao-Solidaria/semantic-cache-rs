//! Portão de regressão de desempenho do CI (`.github/workflows/bench-regression.yml`,
//! contrato da Fase 4, §3): os casos do caminho quente do LOOKUP medidos na base
//! e no PR, no mesmo job e com o MESMO harness (a base recebe este arquivo).
//!
//! - `count <base> <pr>`, o portão: instruções por iteração de cada caso nos dois
//!   binários, no cachegrind do valgrind; a rodada de 2N iterações menos a de N
//!   tira a partida do processo e a montagem do caso. Algum caso mais de 5% acima
//!   da base → saída 1. A conta não muda entre execuções (nos LOOKUPs, até 0,01%:
//!   a resposta leva o `latency_micros` medido); o relógio de um runner
//!   compartilhado varia mais que 5%.
//! - `clock save` (base) e `clock compare` (PR), o relatório: tempo de relógio
//!   pelo criterion contra a baseline `base` em `$CRITERION_HOME`, com intervalo
//!   de confiança de 95%. Não decide nada.
//! - `run <caso> <iterações>`: um caso sozinho (o que o valgrind executa).
//!
//! Local, com o valgrind no PATH: compile este binário na base e no PR com o
//! profile do servidor (`CARGO_PROFILE_RELEASE_LTO=fat`,
//! `CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1`, `cargo build --release --bin
//! bench-regression`) e rode `bench-regression count <base> <pr>`, com os dois
//! caminhos do mesmo tamanho: o `argv[0]` entra no heap e muda o alinhamento dos
//! dados (0,15% no `quantize`). Saída: 0 ok, 1 regressão, 2 erro.

use std::fmt::Write as _;
use std::hint::black_box;
use std::path::Path;
use std::process::{Command, ExitCode};
use std::{env, fs, process};

use semantic_cache_ipc::frame::{encode, kind, read_frame, Frame};
use semantic_cache_simd::{cosine, dot, Kernel, QuantizedVector};
use semantic_cache_testkit::bench::Precision;
use semantic_cache_testkit::fixtures::VectorGen;
use semantic_cache_testkit::ipc::{LookupReq, SearchCache};
use semantic_cache_testkit::stress::{FlatF32, FlatI8, VectorIndex};

/// Teto do portão: um caso pode ficar até 5% mais lento que a base.
const LIMIT_PERCENT: u64 = 5;
/// Dimensão dos vetores (`text-embedding-3-small`).
const DIM: usize = 1536;
/// Vetores guardados nos casos de LOOKUP.
const STORED: usize = 256;

/// Os casos e o N da rodada de instruções de cada um (a de 2N também roda; cada
/// N passa de um milhão de instruções).
const CASES: [(&str, u64); 8] = [
    ("dot_f32", 4_000),
    ("cosine_f32", 2_000),
    ("quantize", 200),
    ("dot_i8", 4_000),
    ("cosine_i8", 4_000),
    ("ipc_frame", 100),
    ("lookup_f32", 20),
    ("lookup_i8", 20),
];

/// A operação medida; o estado dela é montado antes, fora da medição.
type Op = Box<dyn FnMut()>;

/// `f` como operação medida, com o resultado no `black_box` (o compilador não
/// pode descartar o trabalho).
fn op<T>(mut f: impl FnMut() -> T + 'static) -> Op {
    Box::new(move || {
        black_box(f());
    })
}

/// O caso `name`, montado: dois vetores unitários de `DIM` dimensões, sempre os
/// mesmos, e o que cada caso faz com eles.
fn case(name: &str) -> Option<Op> {
    let mut gen = VectorGen::new(0x5EED);
    let (a, b) = (gen.unit(DIM).data, gen.unit(DIM).data);
    let quantize = |v: &[f32]| QuantizedVector::quantize(v).expect("valores finitos");
    let (qa, qb) = (quantize(&a), quantize(&b));
    Some(match name {
        "dot_f32" => op(move || dot(black_box(&a), black_box(&b))),
        "cosine_f32" => op(move || cosine(black_box(&a), black_box(&b))),
        "quantize" => op(move || QuantizedVector::quantize(black_box(&a))),
        "dot_i8" => op(move || black_box(&qa).dot(black_box(&qb))),
        "cosine_i8" => op(move || black_box(&qa).cosine(black_box(&qb))),
        // O frame de um LOOKUP_REQ (uns 17 KB): montar, com o CRC, e ler de volta.
        "ipc_frame" => {
            let payload = request(&a);
            let rt = tokio::runtime::Builder::new_current_thread().build();
            let rt = rt.expect("runtime de uma thread");
            op(move || {
                let frame = encode(kind::LOOKUP_REQ, 0, 1, black_box(&payload));
                let frame = frame.expect("cabe no frame");
                rt.block_on(read_frame(&mut frame.as_slice()))
            })
        }
        "lookup_f32" => lookup(FlatF32::default(), &mut gen),
        "lookup_i8" => lookup(FlatI8::default(), &mut gen),
        _ => return None,
    })
}

/// O payload de um `LOOKUP_REQ` com `embedding`.
fn request(embedding: &[f32]) -> Vec<u8> {
    let prompt = "Explique em poucas linhas como o gateway decide usar o cache.".to_owned();
    let (embedding, threshold) = (embedding.to_vec(), 0.9);
    let req = LookupReq {
        prompt,
        embedding,
        threshold,
    };
    serde_json::to_vec(&req).expect("JSON de um struct")
}

/// Um LOOKUP que acerta (cosseno 0,98 com o vetor guardado do meio) no servidor
/// de referência: JSON do pedido → busca → JSON da resposta → frame.
fn lookup<I: VectorIndex + 'static>(mut index: I, gen: &mut VectorGen) -> Op {
    let stored: Vec<_> = (0..STORED).map(|_| gen.unit(DIM)).collect();
    for vector in &stored {
        index.insert(&vector.data);
    }
    let texts = (0..STORED).map(|id| format!("resposta {id}")).collect();
    let cache = SearchCache::new(index, DIM, texts);
    let payload = request(&gen.with_cosine(&stored[STORED / 2], 0.98).data);
    let frame = Frame {
        kind: kind::LOOKUP_REQ,
        request_id: 1,
        payload,
    };
    // O caso mede um hit de verdade, não um ERROR nem um miss.
    let reply = String::from_utf8_lossy(&cache.answer(&frame)).into_owned();
    let hit = format!("\"resposta {}\"", STORED / 2);
    assert!(reply.contains(&hit), "o LOOKUP não acertou: {reply}");
    op(move || cache.answer(black_box(&frame)))
}

/// `run`: o caso `name`, `iters` vezes, e o kernel SIMD em uso (sob o valgrind,
/// o que a CPU emulada oferece).
fn run(name: &str, iters: u64) -> Result<(), String> {
    let mut op = case(name).ok_or_else(|| format!("caso desconhecido: {name}"))?;
    for _ in 0..iters {
        op();
    }
    println!("{}", Kernel::active());
    Ok(())
}

/// Instruções de um processo e o kernel que ele usou.
type Measured = Result<(u64, String), String>;

/// `bin run <name> <iters>` no cachegrind: as instruções do processo inteiro.
fn cachegrind(bin: &Path, name: &str, iters: u64) -> Measured {
    let file = env::temp_dir().join(format!("bench-regression-{}.cachegrind", process::id()));
    let out = Command::new("valgrind")
        .args(["--tool=cachegrind", "--cache-sim=no", "-q"])
        .arg(format!("--cachegrind-out-file={}", file.display()))
        .arg(bin)
        .args(["run", name, &iters.to_string()])
        .output()
        .map_err(|err| format!("valgrind: {err}"))?;
    let report = fs::read_to_string(&file);
    let _ = fs::remove_file(&file);
    let (bin, status) = (bin.display(), out.status);
    if !status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("{bin} run {name} {iters}: {status}\n{stderr}"));
    }
    let report = report.map_err(|err| format!("saída do cachegrind: {err}"))?;
    let total = summary(&report).ok_or("cachegrind sem a linha `summary:` de um evento")?;
    let kernel = String::from_utf8_lossy(&out.stdout).trim().to_owned();
    Ok((total, kernel))
}

/// O total da linha `summary:` de um arquivo do cachegrind só com o evento `Ir`.
fn summary(report: &str) -> Option<u64> {
    report
        .lines()
        .find_map(|line| line.strip_prefix("summary:")?.trim().parse().ok())
}

/// Mais de `LIMIT_PERCENT` acima da base, em inteiros (sem arredondar a borda).
fn regressed(base: u64, pr: u64) -> bool {
    u128::from(pr) * 100 > u128::from(base) * u128::from(100 + LIMIT_PERCENT)
}

/// O portão sobre `measure(binário, caso, iterações)`: a tabela em markdown e
/// `true` se nenhum caso passou do teto.
#[allow(clippy::cast_precision_loss)] // contagens muito abaixo de 2^53
fn gate(
    measure: &dyn Fn(&Path, &str, u64) -> Measured,
    base: &Path,
    pr: &Path,
) -> Result<(String, bool), String> {
    // Instruções de n iterações: a rodada de 2n menos a de n, que pagam a mesma
    // partida e a mesma montagem.
    let iterations = |bin: &Path, name: &str, n: u64| {
        let (once, kernel) = measure(bin, name, n)?;
        let (twice, _) = measure(bin, name, 2 * n)?;
        match twice.checked_sub(once) {
            Some(cost) if cost > 0 => Ok((cost, kernel)),
            _ => Err(format!("{name}: {twice} instruções com 2N, {once} com N")),
        }
    };
    let (mut rows, mut ok, mut kernel) = (String::new(), true, String::new());
    for (name, n) in CASES {
        let (before, _) = iterations(base, name, n)?;
        let (after, used) = iterations(pr, name, n)?;
        let worse = regressed(before, after);
        (ok, kernel) = (ok && !worse, used);
        let delta = (after as f64 - before as f64) / before as f64 * 100.0;
        let (before, after) = (before as f64 / n as f64, after as f64 / n as f64);
        let verdict = if worse { "**regressão**" } else { "ok" };
        let row = format!("| {name} | {n} | {before:.1} | {after:.1} | {delta:+.2}% | {verdict} |");
        writeln!(rows, "{row}").expect("String aceita qualquer texto");
    }
    let table = format!(
        "### Portão de instruções: até {LIMIT_PERCENT}% acima da base\n\n\
         Instruções por iteração no cachegrind (valgrind), kernel `{kernel}`: a rodada de 2N\n\
         iterações menos a de N. Δ = (PR − base) / base.\n\n\
         | caso | N | base | PR | Δ | |\n|---|---:|---:|---:|---:|---|\n{rows}"
    );
    Ok((table, ok))
}

/// `clock save|compare`: o tempo de relógio de cada caso pelo criterion, que
/// grava (`save`) ou compara com (o `compare` exige a de todos) a baseline
/// `base` em `$CRITERION_HOME`.
fn clock(save: bool) {
    let (criterion, baseline) = (Precision::from_env().criterion(), "base".to_owned());
    let mut criterion = if save {
        criterion.save_baseline(baseline)
    } else {
        criterion.retain_baseline(baseline, true)
    }
    .with_output_color(false);
    println!("kernel {}", Kernel::active());
    for (name, _) in CASES {
        let mut op = case(name).expect("caso da tabela");
        criterion.bench_function(name, |bench| bench.iter(&mut op));
    }
    criterion.final_summary();
}

/// Os argumentos → `Ok(false)` se o portão achou regressão.
fn dispatch(args: &[&str]) -> Result<bool, String> {
    match *args {
        ["run", name, iters] => match iters.parse() {
            Ok(iters) if iters > 0 => run(name, iters).map(|()| true),
            _ => Err(format!("iterações: inteiro positivo, não {iters:?}")),
        },
        ["count", base, pr] => {
            let (table, ok) = gate(&cachegrind, Path::new(base), Path::new(pr))?;
            print!("{table}");
            Ok(ok)
        }
        ["clock", mode @ ("save" | "compare")] => {
            clock(mode == "save");
            Ok(true)
        }
        _ => Err("uso: run <caso> <iterações> | count <base> <pr> | clock save|compare".into()),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match dispatch(&args.iter().map(String::as_str).collect::<Vec<_>>()) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(message) => {
            eprintln!("bench-regression: {message}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_case_runs_and_has_its_own_name() {
        for (name, n) in CASES {
            assert!(n > 0, "{name}");
            assert_eq!(dispatch(&["run", name, "2"]), Ok(true), "{name}");
        }
        let mut names = CASES.map(|(name, _)| name);
        names.sort_unstable();
        assert!(names.windows(2).all(|w| w[0] != w[1]), "{names:?}");
    }

    #[test]
    fn bad_arguments_are_refused() {
        let bad: [&[&str]; 6] = [
            &[],
            &["run", "dot_f32", "0"],
            &["run", "dot_f32", "-1"],
            &["run", "nenhum", "1"],
            &["count", "base"],
            &["clock", "replay"],
        ];
        for args in bad {
            assert!(dispatch(args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn the_summary_line_is_the_total_of_the_only_event() {
        let report = "cmd: x run dot_f32 8\nevents: Ir\nfl=a.rs\nfn=f\n3 120\nsummary: 193486\n";
        assert_eq!(summary(report), Some(193_486));
        assert_eq!(summary("events: Ir\nfl=a.rs\n"), None);
        // Com a simulação de cache ligada viriam vários eventos: não é o portão.
        assert_eq!(summary("summary: 10 2 1\n"), None);
    }

    #[test]
    fn up_to_the_limit_passes_and_above_it_fails() {
        assert!(!regressed(1_000, 1_050));
        assert!(regressed(1_000, 1_051));
        assert!(!regressed(1_000, 400));
        assert!(!regressed(u64::MAX, u64::MAX));
        assert!(regressed(u64::MAX / 2, u64::MAX));
    }

    /// Medição falsa: a iteração custa 1000, menos no caso `slow` do PR (`cost`);
    /// a partida e a montagem diferem entre os binários, e o 2N − N as cancela.
    fn fake(slow: &'static str, cost: u64) -> impl Fn(&Path, &str, u64) -> Measured {
        move |bin, name, n| {
            let pr = bin == Path::new("pr");
            let startup = if pr { 9_000 } else { 5_000 };
            let per = if pr && name == slow { cost } else { 1_000 };
            Ok((startup + per * n, "avx2".to_owned()))
        }
    }

    #[test]
    fn the_gate_flags_only_the_case_above_the_limit() {
        let (base, pr) = (Path::new("base"), Path::new("pr"));
        let (table, ok) = gate(&fake("lookup_i8", 1_051), base, pr).unwrap();
        assert!(!ok, "{table}");
        assert!(table.contains("kernel `avx2`"), "{table}");
        let row = "| lookup_i8 | 20 | 1000.0 | 1051.0 | +5.10% | **regressão** |";
        assert!(table.contains(row), "{table}");
        assert!(table.contains("| dot_f32 | 4000 | 1000.0 | 1000.0 | +0.00% | ok |"));
        assert_eq!(table.matches("**regressão**").count(), 1, "{table}");

        // Exatamente no teto passa; mais rápido também.
        assert!(gate(&fake("lookup_i8", 1_050), base, pr).unwrap().1);
        let (table, ok) = gate(&fake("quantize", 700), base, pr).unwrap();
        assert!(ok && table.contains("| 700.0 | -30.00% | ok |"), "{table}");
    }

    #[test]
    fn a_count_that_does_not_grow_with_the_iterations_is_an_error() {
        let flat = |_: &Path, _: &str, _: u64| Ok((5_000, "avx2".to_owned()));
        let err = gate(&flat, Path::new("base"), Path::new("pr")).unwrap_err();
        assert_eq!(err, "dot_f32: 5000 instruções com 2N, 5000 com N");
        let failing = |_: &Path, _: &str, _: u64| Err("valgrind: not found".to_owned());
        assert!(gate(&failing, Path::new("base"), Path::new("pr")).is_err());
    }
}
