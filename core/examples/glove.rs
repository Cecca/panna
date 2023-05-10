use lsh::collision_index::*;
use lsh::cross_polytope::*;
use lsh::types::*;
use ndarray::prelude::*;
use ndarray_rand::rand::prelude::*;
use progress_logger::*;
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

fn main() {
    env_logger::init();
    debug_assert!(false, "run only in release mode");
    let mut rng = StdRng::seed_from_u64(1234);

    let sim = CosineSimilarity::<ArrayView1<f32>>::default();

    let delta = 1.0;
    let reps = 32;
    let (data, queries, neighbors) = load_glove100();

    let builder =
        CrossPolytopeBuilder::<ArrayView1<f32>, _>::new(data.num_dimensions(), 1, &mut rng);
    eprintln!("Building CP index");
    let tstart = Instant::now();
    let mut index = CollisionIndex::new(sim, &data, builder, reps);
    let tend = Instant::now();
    eprintln!("Index built in {:?}", tend - tstart);


    eprintln!("Running queries");
    let mut pl = ProgressLogger::builder()
        .with_expected_updates(queries.num_points() as u64)
        .with_items_name("queries")
        .start();

    let all_queries = Vec::from_iter((0..queries.num_points()).map(|q_idx| {
        let query = queries.row(q_idx);
        let r =
            CosineSimilarity::similarity(&query, &data.row(*neighbors.get((q_idx, 9)).unwrap()));
        (query, r)
    }));

    let mut all_stats = Vec::new();
    for (query, r) in all_queries {
        let mut stats = QueryStats::default();
        index.query_range(&query, r, delta, &mut stats);
        all_stats.push(stats);
        pl.update(1u64);
    }
    pl.stop();

    // Report statistics
    let mut out = std::io::BufWriter::new(std::fs::File::create("res.csv").unwrap());
    writeln!(out, "q_idx,dimensions,range,visited,false_positives,true_positives,threshold_probability,threshold_probability_bound").unwrap();
    for (q_idx, stats) in all_stats.into_iter().enumerate() {
        let query = queries.row(q_idx);
        let r =
            CosineSimilarity::similarity(&query, &data.row(*neighbors.get((q_idx, 0)).unwrap()));
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
    }
}
