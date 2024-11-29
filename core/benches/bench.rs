#![feature(portable_simd)]
use ann_playground_core::brute_force::brute_force_knn;
use ann_playground_core::dataset::*;
use std::arch::x86_64::*;
use std::simd::num::SimdFloat;
use std::simd::{f32x8, StdFloat};

fn main() {
    divan::main()
}

unsafe fn to_m256(v: &[f32]) -> Vec<__m256> {
    let mut v = v.to_owned();
    const PADDED_VECTORS_BASE: usize = 8;
    let padding = (PADDED_VECTORS_BASE - (v.len() % PADDED_VECTORS_BASE)) % PADDED_VECTORS_BASE;

    v.resize(v.len() + padding, 0.0);
    v.chunks_exact(8)
        .map(|c| _mm256_loadu_ps(c.as_ptr()))
        .collect()
}

#[inline(always)]
unsafe fn dot_fma_prechunked(a: &[__m256], b: &[__m256]) -> f32 {
    const LANES: usize = 8;
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len() % LANES, 0);

    let mut cc = _mm256_setzero_ps();

    for (aa, bb) in a.iter().zip(b) {
        cc = _mm256_fmadd_ps(*aa, *bb, cc);
    }

    let mut buf = [0.0f32; LANES];
    _mm256_storeu_ps(buf.as_mut_ptr(), cc);

    buf.into_iter().sum::<f32>()
}

fn to_simd(v: &[f32]) -> Vec<std::simd::f32x8> {
    let mut v = v.to_owned();
    const PADDED_VECTORS_BASE: usize = 8;
    let padding = (PADDED_VECTORS_BASE - (v.len() % PADDED_VECTORS_BASE)) % PADDED_VECTORS_BASE;

    v.resize(v.len() + padding, 0.0);
    v.chunks_exact(8)
        .map(|c| std::simd::f32x8::from_slice(c))
        .collect()
}

#[inline(always)]
fn dot_simd(v: &[f32x8], u: &[f32x8]) -> f32 {
    let mut sum = f32x8::splat(0.0);
    for (vv, uu) in v.iter().zip(u) {
        // sum += vv * uu;
        sum = vv.mul_add(*uu, sum);
    }
    // sum.reduce_sum()
    let vals = sum.to_array();
    vals.iter().sum::<f32>()
}

#[divan::bench]
fn bench_portable_simd_pre_chunked(bencher: divan::Bencher) {
    let path = ".glove-100-angular.hdf5";
    let dataset = load_raw_data(path);
    let queries = load_raw_queries(path);
    let qidx = 0;

    let v = dataset.row(qidx).to_owned();
    let v = v.as_slice().unwrap();
    let v = unsafe { to_m256(v) };
    let q = queries.row(qidx).to_owned();
    let q = q.as_slice().unwrap();
    let q = unsafe { to_m256(q) };

    bencher.bench(move || unsafe { dot_fma_prechunked(&v, &q) });
}

#[divan::bench]
fn bench_portable_simd_dotp(bencher: divan::Bencher) {
    let path = ".glove-100-angular.hdf5";
    let dataset = load_raw_data(path);
    let queries = load_raw_queries(path);
    let qidx = 50;

    let v = dataset.row(qidx).to_owned();
    let v = v.as_slice().unwrap();
    let v = to_simd(v);
    let q = queries.row(qidx).to_owned();
    let q = q.as_slice().unwrap();
    let q = to_simd(q);

    bencher.bench(move || dot_simd(&v, &q));
}

#[divan::bench]
fn bench_fma_dot_product(bencher: divan::Bencher) {
    let path = ".glove-100-angular.hdf5";
    let dataset = load_raw_data(path);
    let queries = load_raw_queries(path);
    let qidx = 100;

    let v = dataset.row(qidx).to_owned();
    let v = v.as_slice().unwrap();
    let q = queries.row(qidx).to_owned();
    let q = q.as_slice().unwrap();
    bencher.bench(move || dot(v, q));
}

// #[divan::bench]
fn bench_ndarray_dot_product(bencher: divan::Bencher) {
    let path = ".glove-100-angular.hdf5";
    let dataset = load_raw_data(path);
    let queries = load_raw_queries(path);
    let qidx = 0;

    let v = dataset.row(0);
    let q = queries.row(qidx);
    bencher.bench(|| v.dot(&q));
}

// #[divan::bench]
fn bench_ndarray_dot_product_batch(bencher: divan::Bencher) {
    let path = ".glove-100-angular.hdf5";
    let dataset = load_raw_data(path);
    let queries = load_raw_queries(path);
    let qidx = 0;

    bencher.bench(|| {
        let q = queries.row(qidx);
        dataset.dot(&q)
    });
}

// #[divan::bench]
fn bench_brute_force_glove(bencher: divan::Bencher) {
    let path = ".glove-100-angular.hdf5";
    let dataset = AngularDataset::from_hdf5(path);
    let queries = load_raw_queries(path);
    let qidx = 0;
    let k = 100;

    bencher
        .with_inputs(|| {
            let q = queries.row(qidx);
            let mut prepared_query = dataset.default_prepared_query();
            dataset.prepare(&q.as_slice().unwrap(), &mut prepared_query);
            prepared_query
        })
        .bench_refs(|query| brute_force_knn(&dataset, query, k));
}

// #[divan::bench]
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
            dataset.prepare(&q.as_slice().unwrap(), &mut prepared_query);
            prepared_query
        })
        .bench_refs(|query| brute_force_knn(&dataset, query, k));
}

// #[divan::bench]
fn bench_brute_force_sift(bencher: divan::Bencher) {
    let path = ".sift-128-euclidean.hdf5";
    let dataset = EuclideanDataset::from_hdf5(path);
    let queries = load_raw_queries(path);
    let qidx = 0;
    let k = 100;

    bencher
        .with_inputs(|| {
            let q = queries.row(qidx);
            let mut prepared_query = dataset.default_prepared_query();
            dataset.prepare(&q.as_slice().unwrap(), &mut prepared_query);
            prepared_query
        })
        .bench_refs(|query| brute_force_knn(&dataset, query, k));
}
