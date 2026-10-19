//! Detector de degradação da precisão da quantização int8: o recall@k da busca
//! nos vetores quantizados ([`FlatI8`](crate::stress::FlatI8)) contra a busca
//! exata em `f32` ([`FlatF32`]), num corpus com a estrutura de um cache
//! semântico, e um piso ([`MIN_RECALL`]) que reprova a quantização quando ela
//! perde vizinhos.
//!
//! Vetores aleatórios em alta dimensão são quase ortogonais: nenhum é vizinho
//! de verdade de outro, e o recall deles mede ruído. [`Clusters`] monta grupos
//! em volta de tópicos, com as paráfrases amontoadas perto do centro, como
//! num tópico quente do cache. É ali que os escores ficam próximos e o erro do
//! int8 troca a ordem.

use semantic_cache_rs::Vector;

use crate::fixtures::VectorGen;
use crate::stress::{recall, FlatF32, Hit, VectorIndex};

/// Piso do recall@k do int8 na carga [`Clusters::CI`].
///
/// Medido: o int8 dá 0,983–0,994 nesta carga (4 sementes), e o pior caso do
/// projeto é 0,976 (vetores aleatórios, 100 mil, PR 37/100). A mesma carga com
/// 2 bits a menos (6 bits) cai para 0,954–0,957, e com 3 a menos, 0,89–0,91.
/// Abaixo de 0,97, o int8 está perdendo vizinhos como se tivesse perdido bits.
pub const MIN_RECALL: f64 = 0.97;

/// Corpus em grupos: `clusters` centros aleatórios, `per_cluster` vetores por
/// centro e `queries` consultas, cada uma perto de um centro. O cosseno ao
/// centro vai de 0,75 a 0,95, amontoado perto do topo (`0,95 − 0,2·u²`, com
/// `u` na sequência áurea): muitas paráfrases quase iguais e uma cauda do
/// mesmo assunto.
#[derive(Debug, Clone, Copy)]
pub struct Clusters {
    pub seed: u64,
    pub dim: usize,
    pub clusters: usize,
    pub per_cluster: usize,
    pub queries: usize,
    pub k: usize,
}

impl Clusters {
    /// A carga do teste: 1000 vetores de 1536 dimensões em 10 grupos e 100
    /// consultas top-10. Os outros grupos são quase ortogonais à consulta:
    /// quem decide o recall é a densidade dentro do grupo, não o tamanho.
    pub const CI: Self = Self {
        seed: 0x5EED,
        dim: 1536,
        clusters: 10,
        per_cluster: 100,
        queries: 100,
        k: 10,
    };

    /// O recall@k de cada índice (vazio, recebe os dados aqui) contra a busca
    /// exata em `f32` nos mesmos dados.
    ///
    /// # Panics
    ///
    /// Com `clusters`, `per_cluster`, `queries` ou `k` zero, ou `dim` < 2.
    #[must_use]
    pub fn recall(&self, indexes: &mut [&mut dyn VectorIndex]) -> Vec<f64> {
        let sizes = [self.clusters, self.per_cluster, self.queries, self.k];
        assert!(sizes.iter().all(|&n| n > 0), "carga vazia: {self:?}");
        let (corpus, queries) = self.generate();
        let mut exact = FlatF32::with_capacity(corpus.len(), self.dim);
        let truth = run(&mut exact, &corpus, &queries, self.k);
        indexes
            .iter_mut()
            .map(|index| recall(&truth, &run(*index, &corpus, &queries, self.k)))
            .collect()
    }

    fn generate(&self) -> (Vec<Vector>, Vec<Vector>) {
        let mut gen = VectorGen::new(self.seed);
        let centers: Vec<Vector> = (0..self.clusters).map(|_| gen.unit(self.dim)).collect();
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let cosine = |i: usize| {
            let u = (i as f64 * 0.618_033_988_749_895).fract();
            (0.95 - 0.2 * u * u) as f32
        };
        let size = self.clusters * self.per_cluster;
        let corpus = (0..size)
            .map(|i| gen.with_cosine(&centers[i / self.per_cluster], cosine(i)))
            .collect();
        let queries = (0..self.queries)
            .map(|i| gen.with_cosine(&centers[i % self.clusters], cosine(size + i)))
            .collect();
        (corpus, queries)
    }
}

/// Insere o corpus em `index` e devolve o top-`k` de cada consulta.
fn run(
    index: &mut dyn VectorIndex,
    corpus: &[Vector],
    queries: &[Vector],
    k: usize,
) -> Vec<Vec<Hit>> {
    for vector in corpus {
        index.insert(&vector.data);
    }
    queries.iter().map(|q| index.search(&q.data, k)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stress::FlatI8;

    /// Controle: cada componente arredondado para `levels` níveis de cada lado
    /// (a escala por vetor do int8, com menos bits) e renormalizado. Com 127,
    /// dá o mesmo recall do int8.
    struct Snapped {
        levels: f32,
        exact: FlatF32,
    }

    impl Snapped {
        fn new(levels: f32) -> Self {
            Self {
                levels,
                exact: FlatF32::default(),
            }
        }

        fn snap(&self, v: &[f32]) -> Vec<f32> {
            let step = v.iter().fold(0.0f32, |m, x| m.max(x.abs())) / self.levels;
            let q: Vec<f32> = v.iter().map(|x| (x / step).round() * step).collect();
            let norm = q.iter().map(|x| x * x).sum::<f32>().sqrt();
            q.iter().map(|x| x / norm).collect()
        }
    }

    impl VectorIndex for Snapped {
        fn insert(&mut self, vector: &[f32]) {
            let snapped = self.snap(vector);
            self.exact.insert(&snapped);
        }

        fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
            self.exact.search(&self.snap(query), k)
        }
    }

    #[test]
    fn int8_recall_stays_above_the_floor_and_fewer_bits_fall_below() {
        let (mut int8, mut five_bits) = (FlatI8::default(), Snapped::new(15.0));
        let got = Clusters::CI.recall(&mut [&mut int8, &mut five_bits]);
        let [int8, five_bits] = got[..] else {
            unreachable!("um recall por índice")
        };
        assert!(
            int8 >= MIN_RECALL,
            "int8 perdeu precisão: recall@10 {int8:.3} < {MIN_RECALL}"
        );
        // O detector reprova uma quantização pior: sem isso, o piso não prova nada.
        assert!(
            five_bits < MIN_RECALL,
            "5 bits passou no piso: {five_bits:.3}"
        );
    }

    /// A curva por bits, fora do CI:
    /// `cargo test --release --lib -- --ignored --nocapture recall_by_bits`.
    #[test]
    #[ignore = "medição"]
    fn recall_by_bits() {
        let bigger = Clusters {
            clusters: 50,
            per_cluster: 200,
            queries: 200,
            ..Clusters::CI
        };
        for load in [Clusters::CI, bigger] {
            for seed in [0x5EED, 1, 2, 3] {
                let mut indexes: Vec<Box<dyn VectorIndex>> = vec![Box::new(FlatI8::default())];
                indexes.extend([127.0, 31.0, 15.0, 7.0].map(|l| Box::new(Snapped::new(l)) as _));
                let mut refs: Vec<&mut dyn VectorIndex> =
                    indexes.iter_mut().map(|i| &mut **i as _).collect();
                let got = Clusters { seed, ..load }.recall(&mut refs);
                let n = load.clusters * load.per_cluster;
                println!("{n} vetores, semente {seed:#x}: int8, 127/31/15/7 níveis = {got:.3?}");
            }
        }
    }
}
