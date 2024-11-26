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
        // NOTE: Here the assumption is that the distance is (1 - dotp)
        // where dotp is the dot product between two unit-norm vectors
        let angle = (1.0 - distance).acos();
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

#[cfg(test)]
mod test {
    use crate::dataset::{AngularDataset, Dataset};

    use super::*;

    #[test]
    fn simhash_collision_probability() {
        let rng = StdRng::seed_from_u64(1234);
        let dataset = AngularDataset::from_hdf5(".glove-100-angular.hdf5");
        let builder = SimHashBuilder::<&[f32], _>::new(dataset.num_dimensions(), 1, rng);
        test_collision_probability(&dataset, builder, 1000000, 0.05);
    }

    pub fn test_collision_probability<'data, P, D, F, B, O>(
        data: &'data D,
        mut builder: B,
        samples: usize,
        tolerance: f32,
    ) where
        D: Dataset<'data, P>,
        D::Distance: Into<f32>,
        F: LSHFunction<Input = P, Output = O>,
        B: LSHFunctionBuilder<LSH = F>,
        O: Eq + Copy,
    {
        let hashers = builder.build_vec(samples);

        let mut scratch = hashers[0].allocate_scratch();

        let n = 100;
        let mut hashes: Vec<Vec<F::Output>> = vec![Vec::new(); n];
        for i in 0..n {
            let x = data.get(i);
            hashes[i].extend(hashers.iter().map(|h| h.hash(&x, &mut scratch)));
        }

        let mut q = data.default_prepared_query();
        for i in 0..n {
            let x = data.get(i);
            data.prepare(&x, &mut q);
            let hx = &hashes[i];
            for j in (i + 1)..n {
                dbg!(j);
                let d_xy = data.distance(j, &q);
                let hy = &hashes[j];
                let p_xy =
                    hx.iter().zip(hy).filter(|(x, y)| x == y).count() as f32 / samples as f32;

                let p_expected = hashers[0].collision_probability(d_xy.into());
                dbg!(p_expected, p_xy, (p_xy - p_expected).abs());
                assert!(
                    (p_xy - p_expected).abs() <= tolerance,
                    "expected {}, got {} (distance={:?})",
                    p_expected,
                    p_xy,
                    d_xy,
                );
            }
        }
    }
}
