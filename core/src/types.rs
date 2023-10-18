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

pub trait LSHFunction: Send + Sync {
    type Input;
    type Output: Eq + Ord;
    type Scratch;

    fn allocate_scratch(&self) -> Self::Scratch;

    fn hash(&self, v: &Self::Input, scratch: &mut Self::Scratch) -> Self::Output;

    fn collision_probability(&self, similarity: f32) -> f32;
}

pub trait LSHFunctionBuilder {
    type LSH: LSHFunction;

    fn build(&mut self) -> Self::LSH;

    fn build_vec(&mut self, n: usize) -> Vec<Self::LSH> {
        let mut res = Vec::with_capacity(n);
        for _ in 0..n {
            res.push(self.build());
        }
        res
    }
}

// pub trait Dataset<'slf, T> {
//     fn num_dimensions(&self) -> usize;
//     fn num_points(&self) -> usize;
//     fn get(&'slf self, idx: usize) -> T;
// }
//
// pub struct UnitNormDataset {
//     points: Array2<f32>,
// }
//
// impl<'slf> Dataset<'slf, ArrayView1<'slf, f32>> for UnitNormDataset {
//     fn num_dimensions(&self) -> usize {
//         self.points.ncols()
//     }
//
//     fn num_points(&self) -> usize {
//         self.points.nrows()
//     }
//
//     fn get(&'slf self, idx: usize) -> ArrayView1<'slf, f32> {
//         self.points.row(idx)
//     }
// }
//
// impl UnitNormDataset {
//     fn from(mut points: Array2<f32>) -> Self {
//         for mut row in points.rows_mut() {
//             row /= norm2(&row);
//         }
//         Self { points }
//     }
// }
//
// impl<'slf> Dataset<'slf, ArrayView1<'slf, f32>> for Array2<f32> {
//     fn num_dimensions(&self) -> usize {
//         self.ncols()
//     }
//
//     fn num_points(&self) -> usize {
//         self.nrows()
//     }
//
//     fn get(&'slf self, idx: usize) -> ArrayView1<'slf, f32> {
//         self.row(idx)
//     }
// }

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
