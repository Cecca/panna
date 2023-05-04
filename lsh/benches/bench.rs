use std::time::Instant;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lsh::collision_index::CollisionIndex;
use lsh::simhash::*;
use lsh::types::*;
use ndarray::prelude::*;
use ndarray_rand::rand::prelude::*;

pub fn load_glove25() -> (Array2<f32>, Array2<f32>, Array2<usize>) {
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

pub fn bench_simhash_range_query(c: &mut Criterion) {
    let reps = 1000;
    let (data, queries, neighbors) = load_glove25();

    let query = queries.row(0);
    for i in 0..10 {
        let idx = *neighbors.get((0, i)).unwrap();
        let r = CosineSimilarity::similarity(&query, &data.row(idx));
        eprintln!("[{}] r_{} = {}", idx, i, r);
    }
    let r = CosineSimilarity::similarity(&query, &data.row(*neighbors.get((0, 10)).unwrap()));
    dbg!(r);

    let rng = StdRng::seed_from_u64(1234);

    let builder = SimHashBuilder::<ArrayView1<f32>, _>::new(data.num_dimensions(), 8, rng);
    let sim = CosineSimilarity::<ArrayView1<f32>>::default();
    eprintln!("Building index");
    let tstart = Instant::now();
    let mut index = CollisionIndex::new(sim, &data, builder, reps);
    let tend = Instant::now();
    eprintln!("Index built in {:?}", tend - tstart);

    let delta = 0.1;
    let mut group = c.benchmark_group("range query");
    let mut stats = QueryStats::default();
    index.query_range(&query, r, delta, &mut stats);
    eprintln!("{:?}", stats);
    group.bench_function("simhash", |b| {
        b.iter(|| black_box(index.query_range(&query, r, delta, &mut stats)))
    });
    drop(group);
}

pub fn bench_simhash(c: &mut Criterion) {
    let reps = 1000;
    let (data, _, _) = load_glove25();
    let rng = StdRng::seed_from_u64(1234);
    let hashers =
        SimHashBuilder::<ArrayView1<f32>, _>::new(data.shape()[1], 8, rng).build_vec(reps);
    let point = data.row(0);

    let mut group = c.benchmark_group("one hash, one point");
    group.bench_function("simhash", |b| {
        b.iter(|| black_box(hashers[0].hash(&point, &mut ())))
    });
    drop(group);

    let mut group = c.benchmark_group("all hashes, one point");
    group.throughput(criterion::Throughput::Elements(reps as u64));
    group.bench_function("simhash", |b| {
        b.iter(|| {
            for h in hashers.iter() {
                black_box(h.hash(&point, &mut ()));
            }
        })
    });
    drop(group);

    let mut group = c.benchmark_group("one hash, all points");
    group.throughput(criterion::Throughput::Elements(data.nrows() as u64));
    group.bench_function("simhash", |b| {
        b.iter(|| {
            for row in data.rows() {
                black_box(hashers[0].hash(&row, &mut ()));
            }
        })
    });
    drop(group);
}

criterion_group!(benches, bench_simhash, bench_simhash_range_query);
criterion_main!(benches);
