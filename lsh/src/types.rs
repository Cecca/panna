use rand::Rng;

/// Computes the dot product
pub trait DotProduct {
    fn dot_product(&self, other: &Self) -> f32;
}

/// Gets a random normal vector using the given random number generator
pub trait RandomNormal {
    type Output;
    fn random_normal<R: Rng>(dim: usize, rng: &mut R) -> Self::Output;
}

impl<A: AsRef<[f32]>> DotProduct for A {
    fn dot_product(&self, other: &Self) -> f32 {
        let mut s = 0.0;
        for (x, y) in self.as_ref().iter().zip(other.as_ref().iter()) {
            s += x*y;
        }
        s
    }
}

impl<A: AsRef<[f32]>> RandomNormal for A {
    type Output = Vec<f32>;
    fn random_normal<R: Rng>(dim: usize, rng: &mut R) -> Self::Output {
        let mut r = Vec::with_capacity(dim);
        let distr = rand_distr::StandardNormal;
        r.extend(rng.sample_iter::<f32, _>(distr).take(dim));
        r
    }
}


