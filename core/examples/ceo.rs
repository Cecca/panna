use lsh::brute_force::brute_force_range_query;
use lsh::collision_index::*;
use lsh::cross_polytope::*;
use lsh::simhash::SimHash;
use lsh::simhash::SimHashBuilder;
use lsh::types::*;
use ndarray::linalg::*;
use ndarray::prelude::*;
use ndarray_rand::rand::prelude::*;
use ndarray_rand::*;
use progress_logger::*;
use std::collections::BTreeSet;
use std::io::prelude::*;
use std::iter::FromIterator;
use std::time::Instant;

fn recall(ground: ArrayView1<usize>, actual: &BTreeSet<usize>) -> f32 {
    let mut cnt = 0;
    for g in ground {
        if actual.contains(g) {
            cnt += 1;
        }
    }
    cnt as f32 / ground.shape()[0] as f32
}

fn main() {
    env_logger::init();
    debug_assert!(false, "run only in release mode");
    let mut rng = StdRng::seed_from_u64(1234);

    let (data, queries, distances, neighbors) = datasets::load_dense_dataset("glove-25-angular");

    let k = 1;
    let n = data.shape()[0];
    let d = data.shape()[1];
    let nq = 1; //queries.shape()[0];
    let D = 1024; // How many dimensions to project on
    let s0 = 1;
    let b = 1;
    let correction = 1.0 / (2.0 * (D as f32).ln()).sqrt();
    let samples = 10000;

    let normal = ndarray_rand::rand_distr::StandardNormal;
    let mut indices = Vec::from_iter(0..D);

    let mut estimates: Vec<f32> = Vec::new();

    let q_idx = 2;
    let q = queries.row(q_idx);
    let p = data.row(neighbors[[q_idx, 0]]);
    let true_ans = q.dot(&p);
    dbg!(true_ans);

    let mut pl = ProgressLogger::builder()
        .with_items_name("samples")
        .with_expected_updates(samples as u64)
        .start();
    for _ in 0..samples {
        let proj = Array2::<f32>::random_using((d, D), normal, &mut rng);
        let pq = q.dot(&proj);
        let pp = p.dot(&proj);
        indices.sort_by(|i, j| pq[[*i]].partial_cmp(&pq[[*j]]).unwrap().reverse());
        let max_idx = indices[0];
        let concomitant = pp[[max_idx]];
        let estimate = concomitant / (2.0 * (D as f32).ln()).sqrt();
        estimates.push(estimate);

        pl.update(1u64);
    }
    pl.stop();
    let estimates = Array1::from(estimates);
    dbg!(estimates
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap());
    dbg!(estimates.mean().unwrap());
    dbg!(estimates
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap());
}
