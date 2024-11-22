use std::cell::Cell;
use std::collections::HashMap;
use std::marker::PhantomData;

use std::sync::RwLock;

use ffht::fht_f32;
use ndarray::prelude::*;
use ndarray::Data;
use ndarray_rand::rand::prelude::*;
use ndarray_rand::rand_distr::StandardNormal;
use once_cell::sync::Lazy;

use crate::lsh::LSHFunction;
use crate::lsh::LSHFunctionBuilder;

static ESTIMATES: Lazy<RwLock<HashMap<usize, CrossPolytopeProbabilities>>> =
    Lazy::new(|| RwLock::new(HashMap::default()));

const ROTATIONS: usize = 3;

pub struct CrossPolytopeLSH<Input> {
    /// The dimensionality of the input
    dimensions: usize,
    /// The closest power of two larger than `dimensions`
    power_dimensions: usize,
    /// How many bits the output of each function uses
    pub bits_per_function: usize,
    /// The number of functions to concatenate
    num_functions: usize,
    /// The diagonal matrices, each of size `dimensions`
    diagonals: Vec<f32>,
    _marker: PhantomData<Input>,
}

impl<Input> CrossPolytopeLSH<Input> {
    pub fn new<R: Rng>(dimensions: usize, num_functions: usize, rng: &mut R) -> Self {
        use ndarray_rand::rand_distr::Bernoulli;

        let power_dimensions = dimensions.next_power_of_two();
        let bits_per_function = power_dimensions.ilog2() as usize + 1; // the +1 is because we have
                                                                       // to accommodate twice as
                                                                       // many values than there
                                                                       // are dimensions (positive
                                                                       // and negative).
        assert!(bits_per_function * num_functions <= 8 * std::mem::size_of::<u128>());

        let distr = Bernoulli::new(0.5).unwrap();
        let diagonals: Vec<f32> = distr
            .sample_iter(rng)
            .map(|b| if b { 1.0 } else { -1.0 })
            .take(power_dimensions * ROTATIONS * num_functions)
            .collect();

        ESTIMATES
            .write()
            .unwrap()
            .entry(dimensions)
            .or_insert_with(|| {
                let p = CrossPolytopeProbabilities::get(dimensions, 0.001, 100000);
                p.write_csv("/tmp/cp.csv").unwrap();
                p
            });

        assert!(power_dimensions.is_power_of_two());
        Self {
            dimensions,
            power_dimensions,
            bits_per_function,
            num_functions,
            diagonals,
            _marker: PhantomData,
        }
    }

    #[inline]
    fn diagonal_multiply(&self, v: &mut [f32], diag_id: usize, func_id: usize) {
        assert_eq!(v.len(), self.power_dimensions);
        let start = ROTATIONS * self.power_dimensions * func_id + diag_id * self.power_dimensions;
        let end =
            ROTATIONS * self.power_dimensions * func_id + (diag_id + 1) * self.power_dimensions;
        let diag = &self.diagonals[start..end];
        assert_eq!(v.len(), diag.len());
        for i in 0..v.len() {
            v[i] *= diag[i];
        }
    }

    /// Gives the index of the axis of the space closest to the given vector
    fn closest_axis(&self, v: &mut [f32]) -> usize {
        assert_eq!(v.len(), self.power_dimensions);
        let mut pos = 0;
        let mut max_sim = v[0].abs();
        if v[0] < 0.0 {
            pos = v.len();
        }
        for (i, &val) in v.iter().take(self.dimensions).enumerate() {
            if val > max_sim {
                pos = i;
                max_sim = val;
            } else if -val > max_sim {
                pos = i + v.len();
                max_sim = -val;
            }
        }
        pos
    }
}

impl<S: Data<Elem = f32> + Send + Sync> LSHFunction for CrossPolytopeLSH<ArrayBase<S, Ix1>> {
    type Input = ArrayBase<S, Ix1>;
    type Output = u128;
    type Scratch = Vec<f32>;

    fn collision_probability(&self, similarity: f32) -> f32 {
        ESTIMATES.read().unwrap()[&self.dimensions].probability(similarity)
    }

    fn allocate_scratch(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.power_dimensions);
        v.resize(self.power_dimensions, 0.0);
        v
    }

    fn hash(&self, v: &Self::Input, scratch: &mut Self::Scratch) -> Self::Output {
        assert_eq!(v.len(), self.dimensions);
        assert_eq!(scratch.len(), self.power_dimensions);
        let v = v.as_slice().unwrap();

        let mut h = 0u128;
        for k in 0..self.num_functions {
            // Init the scratch space
            scratch.fill(0.0);
            scratch[..self.dimensions].copy_from_slice(v);

            for i in 0..ROTATIONS {
                self.diagonal_multiply(scratch, i, k);
                fht_f32(scratch);
            }

            h = (h << self.bits_per_function) | self.closest_axis(scratch) as u128;
        }

        h
    }
}

pub struct CrossPolytopeBuilder<Input, R: Rng> {
    dimensions: usize,
    num_functions: usize,
    rng: R,
    _marker: PhantomData<Input>,
}

impl<Input, R: Rng> CrossPolytopeBuilder<Input, R> {
    pub fn new(dimensions: usize, num_functions: usize, rng: R) -> Self {
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
    for CrossPolytopeBuilder<ArrayBase<S, Ix1>, R>
{
    type LSH = CrossPolytopeLSH<ArrayBase<S, Ix1>>;

    fn build(&mut self) -> Self::LSH {
        CrossPolytopeLSH::new(self.dimensions, self.num_functions, &mut self.rng)
    }
}

#[derive(serde::Serialize, serde::Deserialize, PartialEq)]
pub struct CrossPolytopeProbabilities {
    dimensions: usize,
    eps: f32,
    probs: Vec<f32>,
}

impl CrossPolytopeProbabilities {
    pub fn probability(&self, dot_product: f32) -> f32 {
        let lower = (dot_product / self.eps).floor() as usize;
        let upper = (dot_product / self.eps).ceil() as usize;
        let lower = self.probs[lower + self.probs.len() / 2];
        let upper = self.probs[upper + self.probs.len() / 2];
        // self.probs[idx]
        lower + (upper - lower) / 2.0
    }

    pub fn get(dimensions: usize, eps: f32, samples: usize) -> Self {
        use std::path::PathBuf;
        let fname = PathBuf::from(format!(".cp-{}-{}-{}.cache", dimensions, eps, samples));
        if !fname.is_file() {
            let p = Self::new(dimensions, eps, samples);
            let f = std::fs::File::create(&fname).unwrap();
            bincode::serialize_into(f, &p).unwrap();
            #[cfg(test)]
            {
                let f = std::fs::File::open(&fname).unwrap();
                let check: Self = bincode::deserialize_from(f).unwrap();
                assert!(check.eq(&p));
            }
        }
        let f = std::fs::File::open(&fname).unwrap();
        let s: Self = bincode::deserialize_from(f).unwrap();
        assert_eq!(s.dimensions, dimensions);
        assert_eq!(s.eps, eps);
        s
    }

    fn new(dimensions: usize, eps: f32, samples: usize) -> Self {
        use rayon::prelude::*;
        eprintln!(
            "Estimating cross polytope collision probabilities for dimensions {}",
            dimensions
        );
        let mut probs = Vec::new();

        // alpha is the inner product between the two vectors
        // p = e_1 and q = (alpha, sqrt(1 - alpha^2), 0, ....)
        let mut alpha = -1.0f32;

        let normal = StandardNormal;

        // we consider all alphas in increments of `eps`
        while alpha <= 1.0 {
            // now we are going to estimate the number of collisions
            let collisions = thread_local::ThreadLocal::new();

            (0..samples).into_par_iter().for_each(|i| {
                let collisions = collisions.get_or(|| Cell::new(0usize));
                let mut rng = StdRng::seed_from_u64(1234u64 + i as u64);
                let mut p_max_coord = 0.0;
                let mut p_hash = 0i32;
                let mut q_max_coord = 0.0;
                let mut q_hash = 0i32;
                for dim in 0i32..(dimensions as i32) {
                    // sample the only two entries of the rotation matrix that multiply non-zero
                    // elements in our vectors p and q
                    let r1: f32 = normal.sample(&mut rng);
                    let r2: f32 = normal.sample(&mut rng);

                    // now rotate p in the current dimension: 1 i r1
                    let p_rot = r1;
                    if p_rot.abs() > p_max_coord {
                        p_max_coord = p_rot.abs();
                        p_hash = if p_rot >= 0.0 { dim } else { -dim };
                    }
                    // and rotate q in the current dimension
                    let q_rot = alpha * r1 + (1.0 - alpha * alpha).sqrt() * r2;
                    if q_rot.abs() > q_max_coord {
                        q_max_coord = q_rot.abs();
                        q_hash = if q_rot >= 0.0 { dim } else { -dim };
                    }
                }
                // count collisions
                if p_hash == q_hash {
                    collisions.set(collisions.get() + 1);
                }
            });

            let collisions: usize = collisions.into_iter().fold(0, |c1, c2| c1 + c2.get());

            // compute the probabilities
            let prob = collisions as f32 / samples as f32;
            probs.push(prob);

            alpha += eps;
        }

        Self {
            dimensions,
            eps,
            probs,
        }
    }

    pub fn write_csv(&self, path: &str) -> std::io::Result<()> {
        use std::io::prelude::*;
        let mut f = std::fs::File::create(path).expect("cannot open file");
        writeln!(f, "dim,dotp,p")?;
        let mut alpha = -1.0;
        for p in self.probs.iter() {
            writeln!(f, "{},{},{}", self.dimensions, alpha, p)?;
            alpha += self.eps;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // #[test]
    // fn crosspolytope_collision_probability() {
    //     let mut rng = StdRng::seed_from_u64(1234);
    //     let dataset = datasets::load_dense_dataset("glove-25-angular").0;
    //     let builder = CrossPolytopeBuilder::<ArrayView1<f32>, _>::new(25, 1, &mut rng);
    //     crate::test::test_collision_probability(&dataset, builder, 1000000, 0.015);
    //
    //     let builder = CrossPolytopeBuilder::<ArrayView1<f32>, _>::new(25, 2, &mut rng);
    //     crate::test::test_collision_probability(&dataset, builder, 1000000, 0.015);
    // }
}
