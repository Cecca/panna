use ndarray::prelude::*;
use ndarray::Data;
use ndarray_rand::rand::prelude::*;
use ndarray_rand::RandomExt;
use std::marker::PhantomData;

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

impl<S: Data<Elem = f32> + Send + Sync> LSHFunction for SimHash<ArrayBase<S, Ix1>> {
    type Input = ArrayBase<S, Ix1>;
    type Output = u128;
    type Scratch = ();

    fn allocate_scratch(&self) -> Self::Scratch {}

    fn hash(&self, v: &Self::Input, scratch: &mut Self::Scratch) -> Self::Output {
        assert_eq!(v.len(), self.dimensions);
        let mut h = 0;
        for x in self.directions.rows() {
            h <<= 1;
            if v.dot(&x) >= 0.0 {
                h |= 1;
            }
        }
        h
    }

    fn collision_probability(&self, similarity: f32) -> f32 {
        debug_assert!(-1.0 <= similarity && similarity <= 1.0);
        (1.0 - similarity.acos() / std::f32::consts::PI).powi(self.directions.shape()[0] as i32)
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

impl<S: Data<Elem = f32> + Send + Sync, R: Rng> LSHFunctionBuilder
    for SimHashBuilder<ArrayBase<S, Ix1>, R>
{
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
        let dataset = datasets::load_dense_dataset("glove-25-angular").0;
        let builder = SimHashBuilder::<ArrayView1<f32>, _>::new(25, 8, rng);
        crate::test::test_collision_probability(&dataset, builder, 1000000, 0.001);
    }
}
