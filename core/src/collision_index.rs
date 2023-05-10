use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

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

impl<H: Ord + Eq + Default + Clone + Copy + Debug + std::hash::Hash> TableBuilder<H> {
    fn with_capacity(n: usize) -> Self {
        Self {
            pairs: Vec::with_capacity(n),
        }
    }

    fn push(&mut self, hash: H, index: usize) {
        self.pairs.push((hash, index));
    }

    fn build(mut self) -> Table<H> {
        self.pairs.sort();

        // now find boundaries of consecutive ranges
        let mut map = HashMap::new();
        let mut indices = Vec::new();
        let mut start = 0;
        let mut cur_hash = self.pairs.first().unwrap().0;
        indices.push(self.pairs.first().unwrap().1);
        for (i, (h, idx)) in self.pairs.into_iter().enumerate().skip(1) {
            indices.push(idx);
            if h != cur_hash {
                map.insert(cur_hash, (start, i));
                start = i;
                cur_hash = h;
            }
        }
        map.insert(cur_hash, (start, indices.len()));

        Table { map, indices }
    }
}

struct Table<H: Ord + Eq + Hash> {
    /// The hash values, all distinct and sorted
    map: HashMap<H, (usize, usize)>,
    /// The indices in the original vector
    indices: Vec<usize>,
}

impl<H: Ord + Eq + Hash> Table<H> {
    fn collisions(&self, h: H) -> &[usize] {
        if let Some(&(s, e)) = self.map.get(&h) {
            &self.indices[s..e]
        } else {
            &[]
        }
    }
}

pub struct CollisionIndex<
    'data,
    H: LSHFunction,
    Sim: SimilarityFunction<Point = H::Input>,
    D: Dataset<'data, H::Input>,
> where
    H::Output: Hash + Send + Sync,
{
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
        D: Dataset<'data, H::Input> + Sync,
    > CollisionIndex<'data, H, Sim, D>
where
    H::Output: Clone + Copy + Ord + Eq + Default + Debug + Hash + Send + Sync,
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
        let tables: Vec<Table<H::Output>> = hashers
            .par_iter()
            .map(|h| {
                let mut scratch = hashers[0].allocate_scratch();
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

    pub fn query_range(
        &mut self,
        q: &H::Input,
        r: f32,
        delta: f32,
        result: &mut Vec<usize>,
        stats: &mut QueryStats,
    ) {
        result.clear();
        self.collision_table.fill(0);
        stats.total = self.data.num_points();

        // Reference local to the function. Benchmarking shows that
        // it makes this function faster!
        let collision_table = &mut self.collision_table;

        for (hasher, table) in self.hashers.iter().zip(&self.tables) {
            let h = hasher.hash(q, &mut self.scratch);
            let collisions = table.collisions(h);
            #[cfg(test)]
            debug_assert!(test::is_increasing(collisions));

            for idx in collisions {
                unsafe {
                    *collision_table.get_unchecked_mut(*idx) += 1;
                }
            }
        }

        let samples = self.hashers.len();
        let p_r = self.hashers[0].collision_probability(r);
        let p_bound = p_r - (1.0 / (2.0 * samples as f32) * (2.0 / delta).ln()).sqrt();
        stats.threshold_probability = p_r;
        stats.threshold_probability_bound = p_bound;
        let threshold = (p_bound * samples as f32).ceil() as usize;
        for (idx, cnt) in self.collision_table.iter().enumerate() {
            if *cnt >= threshold {
                stats.visited += 1;
                if Sim::similarity(q, &self.data.get(idx)) >= r {
                    stats.true_positives += 1;
                    result.push(idx);
                } else {
                    stats.false_positives += 1;
                }
            }
            if *cnt > 0 {
                stats.at_least_one_collision += 1;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::brute_force::brute_force_range_query;
    use crate::simhash::*;
    use crate::test::compute_recall;
    use ndarray::prelude::*;

    pub fn is_increasing(v: &[usize]) -> bool {
        for i in 1..v.len() {
            if v[i - 1] > v[i] {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_index() {
        let repetitions = 10000;
        let data = crate::test::load_glove25();
        let rng = ndarray_rand::rand::thread_rng();
        let builder = SimHashBuilder::<ArrayView1<f32>, _>::new(data.ncols(), 8, rng);
        let sim = CosineSimilarity::<ArrayView1<f32>>::default();
        let mut index = CollisionIndex::new(sim, &data, builder, repetitions);
        let q = data.row(0);
        let range = 0.8;
        let delta = 0.1;
        let mut stats = QueryStats::default();
        let mut ans = Vec::new();
        index.query_range(&q, 0.8, 0.1, &mut ans, &mut stats);
        dbg!(stats);
        let sim = CosineSimilarity::<ArrayView1<f32>>::default();
        let bf = brute_force_range_query(&data, &q, range, sim);
        let recall = compute_recall(bf, ans);
        dbg!(recall);
        assert!(recall >= 1.0 - delta);
    }
}
