use crate::types::*;
use rand::prelude::*;

pub struct SimHash {
    /// The dimensionality of the input vectors
    dimensions: usize,
    /// the directions onto which the vectors are projected
    directions: Vec<f32>,
}

impl SimHash {
    pub fn new<R: Rng>(dimensions: usize, num_functions: usize, rng: &mut R) -> Self {
        let distr = rand_distr::Normal::new(0.0, 1.0).unwrap();
        let directions: Vec<f32> = (0..dimensions * num_functions)
            .map(|_| distr.sample(rng))
            .collect();

        Self { dimensions, directions }
    }

    pub fn hash(&self, v: &[f32]) -> usize {
        assert_eq!(v.len(), self.dimensions);
        let mut h = 0;
        for direction in self.directions.chunks(self.dimensions) {
            let dotp = v.iter().zip(direction).map(|(x, y)| x*y).sum::<f32>();
            h <<= 1;
            if dotp > 0.0 {
                h |= 1;
            }
        }
        h
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::prelude::*;
    use ndarray::Data;
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
    fn simhash_basic_collision_probability() {
        let dims = 25;
        let mut rng = StdRng::seed_from_u64(1234);
        let samples = 10000;
        let hashers: Vec<SimHash> = (0..samples)
            .map(|_| SimHash::new(dims, 8, &mut rng))
            .collect();

        let data = load_glove25();
        let n = 100;
        for i in 0..n {
            let x = data.row(i);
            let hx: Vec<usize> = hashers
                .iter()
                .map(|h| h.hash(x.to_slice().unwrap()))
                .collect();
            for j in (i + 1)..n {
                let y = data.row(j);
                let d_xy = cosine_similarity(x, y);
                let hy: Vec<usize> = hashers
                    .iter()
                    .map(|h| h.hash(y.to_slice().unwrap()))
                    .collect();
                let p_xy =
                    hx.iter().zip(&hy).filter(|(x, y)| x == y).count() as f64 / samples as f64;
                for k in (j + 1)..n {
                    let z = data.row(k);
                    let d_xz = cosine_similarity(x, z);
                    if d_xz <= d_xy / 2.0 {
                        let hz: Vec<usize> = hashers
                            .iter()
                            .map(|h| h.hash(z.to_slice().unwrap()))
                            .collect();
                        let p_xz = hx.iter().zip(&hz).filter(|(x, z)| x == z).count() as f64
                            / samples as f64;

                        dbg!((i, j, k));
                        dbg!(d_xy);
                        dbg!(d_xz);
                        dbg!(p_xy);
                        dbg!(p_xz);
                        assert_eq!(
                            d_xy.partial_cmp(&d_xz).unwrap(),
                            p_xy.partial_cmp(&p_xz).unwrap()
                        );
                    }
                }
            }
        }
    }
}
