use crate::types::*;

pub fn brute_force_range_query<'data, P, D, Sim>(
    dataset: &'data D,
    q: &P,
    range: f32,
    _similarity: Sim,
) -> Vec<usize>
where
    D: Dataset<'data, P>,
    Sim: SimilarityFunction<Point = P>,
{
    let mut res = Vec::new();
    for i in 0..dataset.num_points() {
        if Sim::similarity(q, &dataset.get(i)) >= range {
            res.push(i);
        }
    }
    res
}
