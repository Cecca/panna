use std::time::Instant;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lsh::brute_force::brute_force_range_query;
use lsh::collision_index::CollisionIndex;
use lsh::cross_polytope::CrossPolytopeBuilder;
use lsh::simhash::*;
use lsh::types::*;
use ndarray::prelude::*;
use ndarray_rand::rand::prelude::*;

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

pub fn bench_count(c: &mut Criterion) {
    use ndarray_rand::rand::seq::index::sample;
    let mut counters = vec![0; 1_000_000];

    let mut rng = StdRng::seed_from_u64(1234);

    let mut group = c.benchmark_group("counters");
    group.throughput(criterion::Throughput::Elements(counters.len() as u64));
    group.bench_function("update all", |b| {
        b.iter(|| {
            for i in 0..counters.len() {
                counters[i] += 1;
            }
        })
    });

    let mut idx = sample(&mut rng, counters.len(), counters.len() / 2).into_vec();
    idx.sort();
    group.throughput(criterion::Throughput::Elements(idx.len() as u64));
    group.bench_function("update 1/2", |b| {
        b.iter(|| {
            for i in &idx {
                counters[*i] += 1;
            }
        })
    });
    group.bench_function("update 1/2 unrolled", |b| {
        b.iter(|| {
            let chunks = idx.chunks_exact(4);
            for i in chunks.remainder() {
                counters[*i] += 1;
            }
            for chunk in chunks {
                for i in chunk {
                    counters[*i] += 1;
                }
            }
        })
    });

    let mut idx = sample(&mut rng, counters.len(), counters.len() / 4).into_vec();
    idx.sort();
    group.throughput(criterion::Throughput::Elements(idx.len() as u64));
    group.bench_function("update 1/4", |b| {
        b.iter(|| {
            for i in &idx {
                counters[*i] += 1;
            }
        })
    });
    group.bench_function("update 1/4 unrolled", |b| {
        b.iter(|| {
            let chunks = idx.chunks_exact(4);
            for i in chunks.remainder() {
                counters[*i] += 1;
            }
            for chunk in chunks {
                for i in chunk {
                    counters[*i] += 1;
                }
            }
        })
    });
}

pub fn bench_simhash_range_query(c: &mut Criterion) {
    let reps = 1000;
    let (data, queries, neighbors) = load_glove100();

    let q_idx = 0;
    let query = queries.row(q_idx);
    for i in 0..10 {
        let idx = *neighbors.get((q_idx, i)).unwrap();
        let r = CosineSimilarity::similarity(&query, &data.row(idx));
        eprintln!("[{}] r_{} = {}", idx, i, r);
    }
    let r = CosineSimilarity::similarity(&query, &data.row(*neighbors.get((q_idx, 10)).unwrap()));
    dbg!(r);

    let mut rng = StdRng::seed_from_u64(1234);

    let sim = CosineSimilarity::<ArrayView1<f32>>::default();

    let delta = 1.0;
    let mut group = c.benchmark_group("range query");
    group.bench_function("brute force", |b| {
        b.iter(|| black_box(brute_force_range_query(&data, &query, r, sim)))
    });

    let mut res = Vec::new();

    // Simhash
    let builder = SimHashBuilder::<ArrayView1<f32>, _>::new(data.num_dimensions(), 16, &mut rng);
    eprintln!("Building simhash index");
    let tstart = Instant::now();
    let mut index = CollisionIndex::new(sim, &data, builder, reps);
    let tend = Instant::now();
    eprintln!("Index built in {:?}", tend - tstart);
    let mut stats = QueryStats::default();
    index.query_range(&query, r, delta, &mut res, &mut stats);
    eprintln!("{:?}", stats);
    group.bench_function("simhash", |b| {
        b.iter(|| index.query_range(&query, r, delta, &mut res, &mut stats))
    });

    // cross polytope
    let builder =
        CrossPolytopeBuilder::<ArrayView1<f32>, _>::new(data.num_dimensions(), 1, &mut rng);
    eprintln!("Building CP index");
    let tstart = Instant::now();
    let mut index = CollisionIndex::new(sim, &data, builder, reps);
    let tend = Instant::now();
    eprintln!("Index built in {:?}", tend - tstart);
    let mut stats = QueryStats::default();
    index.query_range(&query, r, delta, &mut res, &mut stats);
    eprintln!("{:?}", stats);
    group.bench_function("crosspolytope", |b| {
        b.iter(|| index.query_range(&query, r, delta, &mut res, &mut stats))
    });

    // cross polytope 2
    let builder =
        CrossPolytopeBuilder::<ArrayView1<f32>, _>::new(data.num_dimensions(), 2, &mut rng);
    eprintln!("Building CP index");
    let tstart = Instant::now();
    let mut index = CollisionIndex::new(sim, &data, builder, reps);
    let tend = Instant::now();
    eprintln!("Index built in {:?}", tend - tstart);
    let mut stats = QueryStats::default();
    index.query_range(&query, r, delta, &mut res, &mut stats);
    eprintln!("{:?}", stats);
    group.bench_function("crosspolytope(x2)", |b| {
        b.iter(|| index.query_range(&query, r, delta, &mut res, &mut stats))
    });

    drop(group);
}

pub fn bench_simhash(c: &mut Criterion) {
    let reps = 200;
    let (data, _, _) = load_glove100();
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

criterion_group!(basic, bench_count);
criterion_group!(benches, bench_simhash, bench_simhash_range_query);
criterion_main!(basic, benches);
