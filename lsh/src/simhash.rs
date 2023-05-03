use std::marker::PhantomData;

use ndarray::linalg::general_mat_vec_mul;
use ndarray::prelude::*;
use ndarray::Data;
use ndarray_rand::rand::prelude::*;
use ndarray_rand::RandomExt;

use crate::types::LSHFunction;
use crate::types::LSHFunctionBuilder;

pub struct SimHash<Input> {
    /// The dimensionality of the input vectors
    dimensions: usize,
    /// the directions onto which the vectors are projected
    directions: Array2<f32>,
    _marker: PhantomData<Input>,
}

impl<Input> SimHash<Input> {
    pub fn new<R: Rng>(dimensions: usize, num_functions: usize, rng: &mut R) -> Self {
        let distr = ndarray_rand::rand_distr::StandardNormal;
        let directions = Array2::random_using((num_functions, dimensions), distr, rng);

        Self {
            dimensions,
            directions,
            _marker: PhantomData,
        }
    }
}

impl<S: Data<Elem = f32>> LSHFunction for SimHash<ArrayBase<S, Ix1>> {
    type Input = ArrayBase<S, Ix1>;
    type Output = usize;
    type Scratch = Array1<f32>;

    fn allocate_scratch(&self) -> Self::Scratch {
        Array1::zeros(self.directions.shape()[0])
    }

    fn hash(&self, v: &Self::Input, scratch: &mut Self::Scratch) -> Self::Output {
        assert_eq!(v.len(), self.dimensions);
        let mut h = 0;
        general_mat_vec_mul(1.0, &self.directions, v, 0.0, scratch);
        for &dotp in scratch.iter() {
            h <<= 1;
            if dotp > 0.0 {
                h |= 1;
            }
        }
        h
    }

    fn collision_probability(&self, similarity: f32) -> f32 {
        let cos = 2.0 * similarity - 1.0;
        debug_assert!(-1.0 <= cos && cos <= 1.0);
        (1.0 - cos.acos() / std::f32::consts::PI).powi(self.directions.shape()[0] as i32)
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

impl<S: Data<Elem = f32>, R: Rng> LSHFunctionBuilder for SimHashBuilder<ArrayBase<S, Ix1>, R> {
    type LSH = SimHash<ArrayBase<S, Ix1>>;

    fn build(&mut self) -> Self::LSH {
        SimHash::new(self.dimensions, self.num_functions, &mut self.rng)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn simhash_collision_probability() {
        let rng = StdRng::seed_from_u64(1234);
        let dataset = crate::test::load_glove25();
        let builder = SimHashBuilder::<ArrayView1<f32>, _>::new(25, 8, rng);
        crate::test::test_collision_probability(&dataset, builder, 1000000, 0.001);
    }

    #[test]
    fn simhash_collision_ranking() {
        let rng = StdRng::seed_from_u64(1234);
        let dataset = crate::test::load_glove25();
        let builder = SimHashBuilder::<ArrayView1<f32>, _>::new(25, 8, rng);
        crate::test::test_collision_prob_ranking_cosine(&dataset, builder, 1000000, 0.001);
    }
}
