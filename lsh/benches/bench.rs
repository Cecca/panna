use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lsh::collision_index::CollisionIndex;
use lsh::simhash::*;
use lsh::types::*;
use ndarray::prelude::*;
use ndarray_rand::rand::prelude::*;

pub fn load_glove25() -> (Array2<f32>, Array2<f32>) {
    use std::io::BufWriter;
    use std::path::PathBuf;

    let local = PathBuf::from(".glove-25-angular.hdf5");
    if !local.is_file() {
        let mut remote = ureq::get("http://ann-benchmarks.com/glove-25-angular.hdf5")
            .call()
            .unwrap()
            .into_reader();
        let mut local_file = BufWriter::new(std::fs::File::create(&local).unwrap());
        std::io::copy(&mut remote, &mut local_file).unwrap();
    }
    let f = hdf5::File::open(&local).unwrap();
    let mut data = f.dataset("/train").unwrap().read_2d::<f32>().unwrap();
    let mut queries = f.dataset("/train").unwrap().read_2d::<f32>().unwrap();

    for mut row in data.rows_mut() {
        row /= norm2(&row);
    }
    for mut row in queries.rows_mut() {
        row /= norm2(&row);
    }
    (data, queries)
}

pub fn bench_simhash_range_query(c: &mut Criterion) {
    let reps = 1000;
    let (data, queries) = load_glove25();
    let rng = StdRng::seed_from_u64(1234);
    let hash_per_sec = 20_000_000.0;
    let num_hashes = data.num_points() * reps;
    eprintln!(
        "Computing {} hashes, expected {} seconds time",
        num_hashes,
        num_hashes as f64 / hash_per_sec
    );

    let builder = SimHashBuilder::<ArrayView1<f32>, _>::new(data.ncols(), 8, rng);
    let sim = CosineSimilarity::<ArrayView1<f32>>::default();
    eprintln!("Building index");
    let mut index = CollisionIndex::new(sim, &data, builder, reps);
    eprintln!("Index built");

    let query = queries.row(0);

    let r = 0.8;
    let delta = 0.1;
    let mut group = c.benchmark_group("range query");
    let mut stats = QueryStats::default();
    group.bench_function("simhash", |b| {
        b.iter(|| black_box(index.query_range(&query, r, delta, &mut stats)))
    });
    drop(group);
}

pub fn bench_simhash(c: &mut Criterion) {
    let reps = 1000;
    let (data, _queries) = load_glove25();
    let rng = StdRng::seed_from_u64(1234);
    let hashers =
        SimHashBuilder::<ArrayView1<f32>, _>::new(data.shape()[1], 8, rng).build_vec(reps);
    let point = data.row(0);
    let mut scratch = hashers[0].allocate_scratch();

    let mut group = c.benchmark_group("one hash, one point");
    group.bench_function("simhash", |b| {
        b.iter(|| black_box(hashers[0].hash(&point, &mut scratch)))
    });
    drop(group);

    let mut group = c.benchmark_group("all hashes, one point");
    group.throughput(criterion::Throughput::Elements(reps as u64));
    group.bench_function("simhash", |b| {
        b.iter(|| {
            for h in hashers.iter() {
                black_box(h.hash(&point, &mut scratch));
            }
        })
    });
    drop(group);

    let mut group = c.benchmark_group("one hash, all points");
    group.throughput(criterion::Throughput::Elements(data.nrows() as u64));
    group.bench_function("simhash", |b| {
        b.iter(|| {
            for row in data.rows() {
                black_box(hashers[0].hash(&row, &mut scratch));
            }
        })
    });
    drop(group);
}

criterion_group!(benches, bench_simhash, bench_simhash_range_query);
criterion_main!(benches);
