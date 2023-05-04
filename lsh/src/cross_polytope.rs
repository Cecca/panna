use std::cell::{Cell, RefCell};

use ffht::fht_f32;
use ndarray_rand::rand::prelude::*;
use ndarray_rand::rand_distr::StandardNormal;

const ROTATIONS: usize = 3;

pub struct CrossPolytopeLSH {
    /// The dimensionality of the input
    dimensions: usize,
    /// The closest power of two larger than `dimensions`
    power_dimensions: usize,
    /// The three diagonal matrices, each of size `dimensions`
    diagonals: Vec<f32>,
}

impl CrossPolytopeLSH {
    pub fn new<R: Rng>(dimensions: usize, rng: &mut R) -> Self {
        use ndarray_rand::rand_distr::Bernoulli;

        let power_dimensions = dimensions.next_power_of_two();

        let distr = Bernoulli::new(0.5).unwrap();
        let diagonals: Vec<f32> = distr
            .sample_iter(rng)
            .map(|b| if b { 1.0 } else { -1.0 })
            .take(dimensions * ROTATIONS)
            .collect();

        Self {
            dimensions,
            power_dimensions,
            diagonals,
        }
    }

    pub fn allocate_scratch(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.power_dimensions);
        v.resize(self.power_dimensions, 0.0);
        v
    }

    pub fn hash(&self, v: &[f32], scratch: &mut [f32]) -> usize {
        assert_eq!(v.len(), self.dimensions);
        assert_eq!(scratch.len(), self.power_dimensions);

        // Init the scratch space
        scratch.fill(0.0);
        scratch[..self.dimensions].copy_from_slice(v);

        for i in 0..ROTATIONS {
            self.diagonal_multiply(scratch, i);
            fht_f32(scratch);
        }

        Self::closest_axis(scratch)
    }

    #[inline]
    fn diagonal_multiply(&self, v: &mut [f32], diag_id: usize) {
        for (x, sign) in v
            .iter_mut()
            .zip(&self.diagonals[diag_id * self.dimensions..(diag_id + 1) * self.dimensions])
        {
            *x *= sign;
        }
    }

    /// Gives the index of the axis of the space closest to the given vector
    fn closest_axis(v: &mut [f32]) -> usize {
        let mut pos = 0;
        let mut max_sim = v[0];
        if -v[0] > max_sim {
            max_sim = -v[0];
            pos = v.len();
        }
        for (i, &val) in v.iter().enumerate() {
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

pub struct CrossPolytopeProbabilities {
    dimensions: usize,
    eps: f32,
    probs: Vec<f32>,
}

impl CrossPolytopeProbabilities {
    pub fn probability(&self, dot_product: f32) -> f32 {
        let idx = (dot_product / self.eps).floor() as usize;
        let idx = idx + self.probs.len() / 2;
        self.probs[idx]
    }

    pub fn new(dimensions: usize, eps: f32, samples: usize) -> Self {
        use rayon::prelude::*;
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

        Self { dimensions, eps, probs }
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

    #[test]
    fn compute_probabilities() {
        let eps = 0.01;
        let dims = 100;
        let probs = CrossPolytopeProbabilities::new(dims, eps, 1000);
        probs.write_csv("cp.csv").unwrap();
        assert_eq!(probs.probability(0.5), 0.078);
        assert_eq!(probs.probability(0.96), 0.623);
    }
}
