use std::marker::PhantomData;

use ndarray::prelude::*;
use ndarray::Data;

pub fn norm2<S: Data<Elem = f32>>(v: &ArrayBase<S, Ix1>) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

pub fn cosine_similarity<S1, S2>(x: ArrayBase<S1, Ix1>, y: ArrayBase<S2, Ix1>) -> f32
where
    S1: Data<Elem = f32>,
    S2: Data<Elem = f32>,
{
    x.dot(&y)
}

pub trait SimilarityFunction {
    type Point;
    fn similarity(x: &Self::Point, y: &Self::Point) -> f32;
}
#[derive(Clone, Copy)]
pub struct CosineSimilarity<T> {
    _marker: PhantomData<T>,
}
impl<T> Default for CosineSimilarity<T> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}
impl<'a> SimilarityFunction for CosineSimilarity<ArrayView1<'a, f32>> {
    type Point = ArrayView1<'a, f32>;
    fn similarity(x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> f32 {
        x.dot(y)
    }
}

#[derive(Default, Debug)]
pub struct QueryStats {
    pub total: usize,
    pub visited: usize,
    pub false_positives: usize,
    pub true_positives: usize,
    pub at_least_one_collision: usize,
    pub threshold_probability: f32,
    pub threshold_probability_bound: f32,
}
