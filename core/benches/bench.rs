use ann_playground_core::brute_force::brute_force_knn;
use ann_playground_core::dataset::*;

fn main() {
    divan::main()
}

#[divan::bench]
fn bench_brute_force_glove(bencher: divan::Bencher) {
    let path = ".glove-200-angular.hdf5";
    let dataset = AngularDataset::from_hdf5(path);
    let queries = load_raw_queries(path);
    let qidx = 0;
    let k = 100;

    bencher
        .with_inputs(|| {
            let q = queries.row(qidx);
            let mut prepared_query = dataset.default_prepared_query();
            dataset.prepare(&q, &mut prepared_query);
            prepared_query
        })
        .bench_refs(|query| brute_force_knn(&dataset, query, k));
}

#[divan::bench]
fn bench_brute_force_glove_padded(bencher: divan::Bencher) {
    let path = ".glove-200-angular.hdf5";
    let dataset = AngularDataset::from_hdf5(path);
    let queries = load_raw_queries(path);
    let qidx = 0;
    let k = 100;

    bencher
        .with_inputs(|| {
            let q = queries.row(qidx);
            let mut prepared_query = dataset.default_prepared_query();
            dataset.prepare(&q, &mut prepared_query);
            prepared_query
        })
        .bench_refs(|query| brute_force_knn(&dataset, query, k));
}

#[divan::bench]
fn bench_brute_force_fashion(bencher: divan::Bencher) {
    let path = ".fashion-mnist-784-euclidean.hdf5";
    let dataset = EuclideanDataset::from_hdf5(path);
    let queries = load_raw_queries(path);
    let qidx = 0;
    let k = 100;

    bencher
        .with_inputs(|| {
            let q = queries.row(qidx);
            let mut prepared_query = dataset.default_prepared_query();
            dataset.prepare(&q, &mut prepared_query);
            prepared_query
        })
        .bench_refs(|query| brute_force_knn(&dataset, query, k));
}
