use std::time::Instant;

use ann_playground_core::{
    dataset::{load_distances, load_raw_queries, AngularDataset, Dataset, DistanceF32},
    lsh::LSHFunctionBuilder,
    puffinn::{Index, Stats},
    simhash::SimHashBuilder,
};

pub fn main() {
    let mut rng = ndarray_rand::rand::thread_rng();

    let path = ".glove-100-angular.hdf5";
    let dataset = AngularDataset::from_hdf5(&path);
    let queries = load_raw_queries(&path);
    let distances = load_distances(&path);

    let hashers =
        SimHashBuilder::<&[f32], _>::new(dataset.num_dimensions(), 32, &mut rng).build_vec(256);

    let index = Index::build(&dataset, hashers);

    let k = 10;
    let delta = 0.1;
    let mut out = vec![(DistanceF32::from(0.0f32), 0); k];
    let nqueries = 10.min(distances.nrows());
    let start = Instant::now();
    let answers = queries
        .rows()
        .into_iter()
        .zip(distances.rows().into_iter())
        .take(nqueries)
        .map(|(query, ground)| {
            let query = query.as_slice().unwrap();
            let stats = index.search(&query, delta, &mut out);
            (compute_recall(&out, &ground.as_slice().unwrap()), stats)
        })
        .collect::<Vec<(f32, Stats)>>();
    let recall = answers.iter().map(|pair| pair.0).sum::<f32>() / nqueries as f32;
    let elapsed = Instant::now() - start;
    let qps = nqueries as f64 / elapsed.as_secs_f64();

    for ans in answers.iter() {
        eprintln!("{:?}", ans);
    }

    dbg!(qps);
    dbg!(recall);
    assert!(recall >= 1.0 - delta);
}

fn compute_recall(actual: &[(DistanceF32, usize)], ground: &[f32]) -> f32 {
    let epsilon = 0.000001;
    let k = actual.len();
    assert!(k >= 1);
    let thresh = DistanceF32::from(ground[k - 1] + epsilon);
    let mut cnt = 0;
    for (d, _) in actual {
        if *d <= thresh {
            cnt += 1;
        }
    }
    cnt as f32 / k as f32
}
