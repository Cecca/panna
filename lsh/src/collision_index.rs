use std::fmt::Debug;

use crate::types::*;

/// Permute the elements of the vector according to the given
/// permutation of the indices.
fn permute<T>(vec: &mut Vec<T>, permutation: &[usize], tmp: &mut Vec<T>) {
    assert!(permutation.len() == vec.len());
    assert!(permutation.len() == tmp.len());
    for (i, x) in tmp.iter_mut().enumerate() {
        std::mem::swap(&mut vec[permutation[i]], x);
    }
    std::mem::swap(tmp, vec);
}

struct TableBuilder<Hash: Ord + Eq> {
    pairs: Vec<(Hash, usize)>,
}

impl<Hash: Ord + Eq + Default + Clone + Copy + Debug> TableBuilder<Hash> {
    fn with_capacity(n: usize) -> Self {
        Self {
            pairs: Vec::with_capacity(n),
        }
    }

    fn push(&mut self, hash: Hash, index: usize) {
        self.pairs.push((hash, index));
    }

    fn build(mut self) -> Table<Hash> {
        self.pairs.sort();

        // now find boundaries of consecutive ranges
        let mut ranges = Vec::new();
        let mut hashes = Vec::new();
        let mut indices = Vec::new();
        let mut start = 0;
        hashes.push(self.pairs.first().unwrap().0);
        indices.push(self.pairs.first().unwrap().1);
        for (i, (h, idx)) in self.pairs.into_iter().enumerate().skip(1) {
            indices.push(idx);
            if h != *hashes.last().unwrap() {
                hashes.push(h);
                ranges.push((start, i));
                start = i;
            }
        }
        ranges.push((start, indices.len()));
        assert_eq!(hashes.len(), ranges.len());

        Table {
            hashes,
            indices,
            ranges,
        }
    }
}

struct Table<Hash: Ord + Eq> {
    /// The hash values, all distinct and sorted
    hashes: Vec<Hash>,
    /// The indices in the original vector
    indices: Vec<usize>,
    /// The endpoints of hash ranges
    ranges: Vec<(usize, usize)>,
}

impl<Hash: Ord + Eq> Table<Hash> {
    fn collisions(&self, h: Hash) -> &[usize] {
        match self.hashes.binary_search(&h) {
            Ok(i) => {
                let (s, e) = self.ranges[i];
                &self.indices[s..e]
            }
            Err(_) => &[],
        }
    }
}

pub struct CollisionIndex<
    'data,
    H: LSHFunction,
    Sim: SimilarityFunction<Point = H::Input>,
    D: Dataset<'data, H::Input>,
> {
    hashers: Vec<H>,
    data: &'data D,
    tables: Vec<Table<H::Output>>,
    /// A counter of collisions for each point, to be reset across queries
    collision_table: Vec<usize>,
    scratch: H::Scratch,
    similarity_function: Sim,
}

impl<
        'data,
        Sim: SimilarityFunction<Point = H::Input>,
        H: LSHFunction,
        D: Dataset<'data, H::Input>,
    > CollisionIndex<'data, H, Sim, D>
where
    H::Output: Clone + Copy + Ord + Eq + Default + Debug,
{
    pub fn new<B: LSHFunctionBuilder<LSH = H>>(
        similarity_function: Sim,
        data: &'data D,
        mut builder: B,
        max_repetitions: usize,
    ) -> Self {
        // Set up hashers
        let hashers = builder.build_vec(max_repetitions);

        // Build tables
        let mut scratch = hashers[0].allocate_scratch();
        let tables: Vec<Table<H::Output>> = hashers
            .iter()
            .map(|h| {
                let mut builder = TableBuilder::with_capacity(data.num_points());
                for i in 0..data.num_points() {
                    let v = data.get(i);
                    builder.push(h.hash(&v, &mut scratch), i);
                }
                builder.build()
            })
            .collect();

        let collision_table = vec![0; data.num_points()];
        let scratch = hashers[0].allocate_scratch();

        Self {
            similarity_function,
            hashers,
            data,
            tables,
            collision_table,
            scratch,
        }
    }

    pub fn query_range(&mut self, q: &H::Input, r: f32, delta: f32) -> Vec<usize> {
        self.collision_table.fill(0);

        for (hasher, table) in self.hashers.iter().zip(&self.tables) {
            let h = hasher.hash(q, &mut self.scratch);
            for idx in table.collisions(h) {
                debug_assert_eq!(hasher.hash(&self.data.get(*idx), &mut self.scratch), h);
                self.collision_table[*idx] += 1;
            }
        }

        let mut res = Vec::new();
        let samples = self.hashers.len();
        let p_r = self.hashers[0].collision_probability(r);
        let p_bound = p_r - (1.0 / (2.0 * samples as f32) * (2.0 / delta).ln()).sqrt();
        let threshold = (p_bound * samples as f32).ceil() as usize;
        let mut cnt_visited = 0;
        let mut cnt_fp = 0;
        for (idx, cnt) in self.collision_table.iter().enumerate() {
            // TODO incorporate confidence interval
            if *cnt >= threshold {
                cnt_visited += 1;
                if Sim::similarity(q, &self.data.get(idx)) >= r {
                    res.push(idx);
                } else {
                    cnt_fp += 1;
                }
            }
        }
        // eprintln!("visited {}, false positives {}", cnt_visited, cnt_fp);
        res
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::brute_force::brute_force_range_query;
    use crate::simhash::*;
    use crate::test::compute_recall;
    use ndarray::prelude::*;

    #[test]
    fn test_index() {
        let repetitions = 1000;
        let data = crate::test::load_glove25();
        let rng = ndarray_rand::rand::thread_rng();
        let builder = SimHashBuilder::<ArrayView1<f32>, _>::new(data.ncols(), 2, rng);
        let sim = CosineSimilarity::<ArrayView1<f32>>::default();
        let mut index = CollisionIndex::new(sim, &data, builder, repetitions);
        let q = data.row(0);
        let range = 0.8;
        let delta = 0.1;
        let ans = index.query_range(&q, 0.8, 0.1);
        let sim = CosineSimilarity::<ArrayView1<f32>>::default();
        let bf = brute_force_range_query(&data, &q, range, sim);
        let recall = compute_recall(bf, ans);
        dbg!(recall);
        assert!(recall >= 1.0 - delta);
    }
}
