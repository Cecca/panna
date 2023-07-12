use lsh::collision_index::*;
use lsh::cross_polytope::*;
use lsh::simhash::SimHash;
use lsh::types::*;
use ndarray::prelude::*;
use ndarray_rand::rand::prelude::*;
use progress_logger::*;
use std::collections::HashMap;
use std::io::prelude::*;
use std::time::Instant;

pub fn load_glove100() -> (Array2<f32>, Array2<f32>, Array2<usize>) {
    use std::io::BufWriter;
    use std::path::PathBuf;

    let local = PathBuf::from(".glove-100-angular.hdf5");
    if !local.is_file() {
        let mut remote = ureq::get("http://ann-benchmarks.com/glove-100-angular.hdf5")
            .call()
            .unwrap()
            .into_reader();
        let mut local_file = BufWriter::new(std::fs::File::create(&local).unwrap());
        std::io::copy(&mut remote, &mut local_file).unwrap();
    }
    let f = hdf5::File::open(&local).unwrap();
    let mut data = f.dataset("/train").unwrap().read_2d::<f32>().unwrap();
    let mut queries = f.dataset("/test").unwrap().read_2d::<f32>().unwrap();
    let ground = f.dataset("/neighbors").unwrap().read_2d::<usize>().unwrap();

    for mut row in data.rows_mut() {
        row /= norm2(&row);
    }
    for mut row in queries.rows_mut() {
        row /= norm2(&row);
    }
    (data, queries, ground)
}

fn sketch_dist(mut a: u128, mut b: u128, func_bits: usize, n_funcs: usize) -> usize {
    let mut d = 0;
    let mut mask = 0;
    for _ in 0..func_bits {
        mask = (mask << 1) | 1;
    }

    for _ in 0..n_funcs {
        if a & mask != b & mask {
            d += 1;
        }
        a >>= func_bits;
        b >>= func_bits;
    }
    
    d
}

#[derive(Clone, Copy)]
struct RunningStats {
    sum: f32,
    max: f32,
    min: f32,
    cnt: usize,
}
impl Default for RunningStats {
    fn default() -> Self {
        Self {
            sum: 0.0,
            max: f32::NEG_INFINITY,
            min: f32::INFINITY,
            cnt: 0,
        }
    }
}
impl RunningStats {
    fn push(&mut self, x: f32) {
        self.sum += x;
        self.max = self.max.max(x);
        self.min = self.min.min(x);
        self.cnt += 1;
    }
    fn avg(&self) -> f32 {
        self.sum / self.cnt as f32
    }
}

fn main() {
    env_logger::init();
    debug_assert!(false, "run only in release mode");
    let mut rng = StdRng::seed_from_u64(1234);

    let sim = CosineSimilarity::<ArrayView1<f32>>::default();

    let (data, queries, neighbors) = load_glove100();

    let K = std::env::args().nth(1).unwrap().parse::<usize>().unwrap();

    let hasher = SimHash::<ArrayView1<f32>>::new(data.num_dimensions(), K, &mut rng);
    let mut scratch = hasher.allocate_scratch();

    let mut hashes: HashMap<u128, Vec<usize>> = HashMap::new();
    for i in 0..data.num_points() {
        let h = hasher.hash(&data.row(i), &mut scratch);
        hashes
            .entry(h)
            .and_modify(|buck| buck.push(i))
            .or_insert_with(|| vec![i]);
    }

    let q_idx = 0;
    let q = queries.row(q_idx);
    let h = hasher.hash(&q, &mut scratch);
    let mut hist_hashes = vec![0; K+1];
    let mut hist_points = vec![0; K+1];
    let mut sims = vec![RunningStats::default(); K+1];
    for other in hashes.keys() {
        let d = sketch_dist(h, *other, 1, K);
        hist_hashes[d] += 1;
        hist_points[d] += hashes[other].len();
        for p_idx in hashes[other].iter() {
            let p = data.row(*p_idx);
            let s = CosineSimilarity::similarity(&q, &p);
            sims[d].push(s);
        }
    }
    println!("sketch_distance, hist_hashes, hist_points, max(sim), min(sim), avg(sim)");
    for d in 0..sims.len() {
        println!("{}, {}, {}, {}, {}, {}",
            d,
            hist_hashes[d],
            hist_points[d],
            sims[d].max,
            sims[d].min,
            sims[d].avg(),
        );
    }
}
