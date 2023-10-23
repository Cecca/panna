use crate::dataset::*;

pub fn brute_force_range_query<'data, P, D>(
    dataset: &'data D,
    q: &P,
    range: D::Distance,
) -> Vec<usize>
where
    D: Dataset<'data, P>,
{
    let mut res = Vec::new();
    let mut prepared_query = dataset.default_prepared_query();
    dataset.prepare(q, &mut prepared_query);
    for i in 0..dataset.num_points() {
        if dataset.distance(i, &prepared_query) <= range {
            res.push(i);
        }
    }
    res
}

pub fn brute_force_knn<'data, P, D>(
    dataset: &'data D,
    prepared_query: &D::PreparedPoint,
    k: usize,
) -> Vec<(D::Distance, usize)>
where
    D: Dataset<'data, P>,
{
    let mut distances = Vec::with_capacity(dataset.num_points());
    for i in 0..dataset.num_points() {
        let d = dataset.distance(i, prepared_query);
        distances.push((d, i));
    }
    let knn = distances.select_nth_unstable(k).0;
    knn.sort_unstable();
    knn.to_vec()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::dataset::*;

    #[test]
    fn test_glove_k100_nn() {
        let path = ".glove-100-angular.hdf5";
        let dataset = AngularDataset::from_hdf5(&path);
        let queries = load_raw_queries(&path);
        let distances = load_distances(&path);
        let qidx = 0;
        let query = queries.row(qidx);
        let mut prepared_query = dataset.default_prepared_query();
        dataset.prepare(&query, &mut prepared_query);

        let k = 100;

        let expected = distances.row(qidx);
        let actual = brute_force_knn(&dataset, &prepared_query, k);
        for (idx, (d, _i)) in actual.into_iter().enumerate() {
            let d: f32 = d.into();
            let ed = expected[idx];
            assert!((d - ed).abs() <= 0.0001);
        }
    }

    #[test]
    fn test_glove_k100_nn_padded() {
        let path = ".glove-100-angular.hdf5";
        let dataset = AngularDatasetPadded::from_hdf5(&path);
        let queries = load_raw_queries(&path);
        let distances = load_distances(&path);
        let qidx = 0;
        let query = queries.row(qidx);
        let mut prepared_query: Vec<f32> = dataset.default_prepared_query();
        dataset.prepare(&query.as_slice().unwrap(), &mut prepared_query);

        let k = 100;

        let expected = distances.row(qidx);
        let actual = brute_force_knn(&dataset, &prepared_query, k);
        for (idx, (d, _i)) in actual.into_iter().enumerate() {
            let d: f32 = d.into();
            dbg!(d);
            let ed = expected[idx];
            dbg!(ed);
            assert!((d - ed).abs() <= 0.0001);
        }
    }

    #[test]
    fn test_fashion_k100_nn() {
        let path = ".fashion-mnist-784-euclidean.hdf5";
        let dataset = EuclideanDataset::from_hdf5(path);
        let queries = load_raw_queries(&path);
        let distances = load_distances(&path);
        let qidx = 0;
        let query = queries.row(qidx);
        let mut prepared_query = dataset.default_prepared_query();
        dataset.prepare(&query, &mut prepared_query);

        let k = 100;

        let expected = distances.row(qidx);
        let actual = brute_force_knn(&dataset, &prepared_query, k);
        for (idx, (d, _i)) in actual.into_iter().enumerate() {
            let d: f32 = d.into();
            let d = d.sqrt();
            let ed = expected[idx];
            assert!((d - ed).abs() <= 0.0001);
        }
    }
}
