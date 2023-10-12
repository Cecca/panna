use lsh::brute_force::brute_force_range_query;
use lsh::collision_index::*;
use lsh::cross_polytope::*;
use lsh::simhash::SimHash;
use lsh::simhash::SimHashBuilder;
use lsh::types::*;
use ndarray::prelude::*;
use ndarray_rand::rand::prelude::*;
use progress_logger::*;
use std::io::prelude::*;
use std::time::Instant;

fn main() {
    env_logger::init();
    debug_assert!(false, "run only in release mode");
    let mut rng = StdRng::seed_from_u64(1234);

    let sim = CosineSimilarity::<ArrayView1<f32>>::default();

    let delta = 1.0;
    let reps = 256;
    let (data, queries, distances, neighbors) = datasets::load_dense_dataset("glove-25-angular");

    let builder =
        CrossPolytopeBuilder::<ArrayView1<f32>, _>::new(data.num_dimensions(), 1, &mut rng);
    // SimHashBuilder::<ArrayView1<f32>, _>::new(data.num_dimensions(), 16, &mut rng);
    eprintln!("Building CP index");
    let tstart = Instant::now();
    let mut index = CollisionIndex::new(sim, &data, builder, reps);
    let tend = Instant::now();
    eprintln!("Index built in {:?}", tend - tstart);

    eprintln!("Running queries");

    let all_queries = Vec::from_iter((0..queries.num_points()).map(|q_idx| {
        let query = queries.row(q_idx);
        let r =
            CosineSimilarity::similarity(&query, &data.row(*neighbors.get((q_idx, 9)).unwrap()));
        (query, r)
    }));

    let mut out = std::io::BufWriter::new(std::fs::File::create("res.csv").unwrap());
    writeln!(out, "q_idx,dimensions,range,visited,false_positives,true_positives,threshold_probability,threshold_probability_bound").unwrap();

    eprintln!("LSH queries");
    let mut pl = ProgressLogger::builder()
        .with_expected_updates(queries.num_points() as u64)
        .with_items_name("queries")
        .start();
    let mut res = Vec::new();
    for (q_idx, (query, r)) in all_queries.iter().enumerate() {
        let mut stats = QueryStats::default();
        index.query_range(query, *r, delta, &mut res, &mut stats);
        writeln!(
            out,
            "{},{},{},{},{},{},{},{}",
            q_idx,
            data.num_dimensions(),
            r,
            stats.visited,
            stats.false_positives,
            stats.true_positives,
            stats.threshold_probability,
            stats.threshold_probability_bound
        )
        .unwrap();
        pl.update(1u64);
    }
    pl.stop();
}
