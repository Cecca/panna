use ndarray_rand::rand::prelude::*;
use std::marker::PhantomData;

use crate::dataset::dot;
use crate::lsh::BitHash32;
use crate::lsh::LSHFunction;
use crate::lsh::LSHFunctionBuilder;

pub trait DotProduct {
    fn dot(&self, other: &[f32]) -> f32;
}

impl DotProduct for &[f32] {
    fn dot(&self, other: &[f32]) -> f32 {
        dot(self, other)
    }
}

pub struct SimHash<I> {
    /// The dimensionality of the input vectors
    dimensions: usize,
    /// the directions onto which the vectors are projected
    // directions: Array2<f32>,
    directions: Vec<f32>,
    _marker: PhantomData<I>,
}

impl<I> SimHash<I> {
    pub fn new<R: Rng>(dimensions: usize, num_functions: usize, rng: &mut R) -> Self {
        let distr = ndarray_rand::rand_distr::StandardNormal;
        // let directions = Array2::random_using((num_functions, dimensions), distr, rng);
        let directions = rng
            .sample_iter(distr)
            .take(num_functions * dimensions)
            .collect();

        Self {
            dimensions,
            directions,
            _marker: PhantomData::<I>,
        }
    }
}

impl<I: DotProduct> LSHFunction for SimHash<I> {
    type Input = I;
    type Output = BitHash32;
    type Scratch = ();

    fn allocate_scratch(&self) -> Self::Scratch {}

    fn hash(&self, v: &Self::Input, _scratch: &mut Self::Scratch) -> Self::Output {
        // assert_eq!(v.len(), self.dimensions);
        let mut h = Self::Output::default();
        for (i, dir) in self.directions.chunks_exact(self.dimensions).enumerate() {
            h.set(i, v.dot(&dir) >= 0.0);
        }
        h
    }

    fn collision_probability(&self, distance: f32) -> f32 {
        debug_assert!(-1.0 <= distance && distance <= 1.0);
        let angle = (1.0 - distance).acos();
        // 1.0 - similarity.acos() / std::f32::consts::PI
        1.0 - angle / std::f32::consts::PI
    }
}

pub struct SimHashBuilder<Input, R: Rng> {
    dimensions: usize,
    num_functions: usize,
    rng: R,
    _marker: PhantomData<Input>,
}

impl<Input, R: Rng> SimHashBuilder<Input, R> {
    pub fn new(dimensions: usize, num_functions: usize, rng: R) -> Self {
        assert!(num_functions > 0);
        assert!(dimensions > 0);
        Self {
            dimensions,
            num_functions,
            rng,
            _marker: PhantomData,
        }
    }
}

impl<I: DotProduct, R: Rng> LSHFunctionBuilder for SimHashBuilder<I, R> {
    type LSH = SimHash<I>;

    fn build(&mut self) -> Self::LSH {
        SimHash::new(self.dimensions, self.num_functions, &mut self.rng)
    }
}

// #[cfg(test)]
// mod test {
//     use super::*;

//     // #[test]
//     // fn simhash_collision_probability() {
//     //     let rng = StdRng::seed_from_u64(1234);
//     //     let dataset = crate::datasets::load_dense_dataset("glove-25-angular").0;
//     //     let builder = SimHashBuilder::<ArrayView1<f32>, _>::new(25, 8, rng);
//     //     crate::test::test_collision_probability(&dataset, builder, 1000000, 0.001);
//     // }
// }
