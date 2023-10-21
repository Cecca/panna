use std::time::Instant;

use ann_playground_core::brute_force::brute_force_knn;
use ann_playground_core::dataset::*;

fn main() {
    let path = ".fashion-mnist-784-euclidean.hdf5";
    let dataset = EuclideanDataset::from_hdf5(path);
    let queries = load_raw_queries(path);
    let qidx = 0;
    let k = 100;

    let q = queries.row(qidx);
    let mut prepared_query = dataset.default_prepared_query();
    dataset.prepare(&q, &mut prepared_query);
    let timer = Instant::now();
    let res = brute_force_knn(&dataset, &prepared_query, k);
    let elapsed = timer.elapsed();
    println!("Elapsed {:?}", elapsed);
    println!("{res:?}");
}
