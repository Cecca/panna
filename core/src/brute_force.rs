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

pub fn brute_force_knn<'data, P, D>(dataset: &'data D, q: &P, k: usize) -> Vec<(D::Distance, usize)>
where
    D: Dataset<'data, P>,
{
    use std::collections::BinaryHeap;
    let mut topk = BinaryHeap::new();
    let mut prepared_query = dataset.default_prepared_query();
    dataset.prepare(q, &mut prepared_query);
    for i in 0..dataset.num_points() {
        let d = dataset.distance(i, &prepared_query);
        topk.push((d, i));
        if topk.len() > k {
            topk.pop();
        }
    }
    let mut res: Vec<(D::Distance, usize)> = topk.into_iter().collect();
    res.sort();
    res
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

        let k = 100;

        let expected = distances.row(qidx);
        let actual = brute_force_knn(&dataset, &query, k);
        for (idx, (d, _i)) in actual.into_iter().enumerate() {
            let d: f32 = d.into();
            let ed = expected[idx];
            assert!((d - ed).abs() <= 0.0001);
        }
    }
}
