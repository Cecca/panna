use ffht::fht_f32;
use rand::prelude::*;

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
        use rand::distributions::Bernoulli;

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

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Data, Ix1};
    use rand::prelude::*;
    use std::io::prelude::*;
    use std::io::BufWriter;
    use std::path::PathBuf;

    fn cosine_similarity(x: ArrayView1<f32>, y: ArrayView1<f32>) -> f32 {
        (x.dot(&y) + 1.0) / 2.0
    }

    fn norm2<S: Data<Elem = f32>>(v: &ArrayBase<S, Ix1>) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    fn load_glove25() -> Array2<f32> {
        let local = PathBuf::from(".glove-25-angular.hdf5");
        if !local.is_file() {
            let mut remote = ureq::get("http://ann-benchmarks.com/glove-25-angular.hdf5")
                .call()
                .unwrap()
                .into_reader();
            let mut local_file = BufWriter::new(std::fs::File::create(&local).unwrap());
            std::io::copy(&mut remote, &mut local_file).unwrap();
        }
        let f = hdf5::File::open(&local).unwrap();
        let mut data = f.dataset("/test").unwrap().read_2d::<f32>().unwrap();

        for mut row in data.rows_mut() {
            row /= norm2(&row);
        }
        data
    }

    #[test]
    fn cross_polytope_basic_collision_probability() {
        let dims = 25;
        let mut rng = StdRng::seed_from_u64(1234);
        let samples = 10000;
        let hashers: Vec<CrossPolytopeLSH> = (0..samples)
            .map(|_| CrossPolytopeLSH::new(dims, &mut rng))
            .collect();

        let mut scratch = hashers[0].allocate_scratch();

        let data = load_glove25();
        let n = 100;
        let mut pairs = Vec::new();
        for i in 0..n {
            let x = data.row(i);
            let hx: Vec<usize> = hashers
                .iter()
                .map(|h| h.hash(x.to_slice().unwrap(), &mut scratch))
                .collect();
            for j in (i + 1)..n {
                let y = data.row(j);
                let d_xy = cosine_similarity(x, y);
                let hy: Vec<usize> = hashers
                    .iter()
                    .map(|h| h.hash(y.to_slice().unwrap(), &mut scratch))
                    .collect();
                let p_xy =
                    hx.iter().zip(&hy).filter(|(x, y)| x == y).count() as f64 / samples as f64;
                pairs.push((d_xy, p_xy));
            }
        }
        pairs.sort_by(|p1, p2| p1.0.partial_cmp(&p2.0).unwrap().reverse());
        let tolerance = 0.001;
        for i in 1..pairs.len() {
            println!("{:?} {:?}", pairs[i-1], pairs[i]);
            assert!(pairs[i-1].1 >= pairs[i].1 + tolerance);
        }
    }
}
