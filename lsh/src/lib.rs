pub mod brute_force;
pub mod collision_index;
pub mod cross_polytope;
pub mod simhash;
pub mod types;

#[cfg(test)]
pub mod test {
    use crate::types::*;
    use ndarray::prelude::*;

    pub fn load_glove25() -> Array2<f32> {
        use std::io::BufWriter;
        use std::path::PathBuf;

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

    pub fn test_collision_probability<'a, F, B>(
        data: &'a Array2<f32>,
        mut builder: B,
        samples: usize,
        tolerance: f32,
    ) where
        F: LSHFunction<Input = ArrayView1<'a, f32>, Output = usize>,
        B: LSHFunctionBuilder<LSH = F>,
    {
        let hashers = builder.build_vec(samples);

        let mut scratch = hashers[0].allocate_scratch();

        let n = 100;
        let mut hashes: Vec<Vec<F::Output>> = vec![Vec::new(); n];
        for i in 0..n {
            let x = data.row(i);
            hashes[i].extend(hashers.iter().map(|h| h.hash(&x, &mut scratch)));
        }

        for i in 0..n {
            let x = data.row(i);
            let hx = &hashes[i];
            for j in (i + 1)..n {
                dbg!(j);
                let y = data.row(j);
                let d_xy = cosine_similarity(x, y);
                if d_xy > 0.01 {
                    let hy = &hashes[j];
                    let p_xy =
                        hx.iter().zip(hy).filter(|(x, y)| x == y).count() as f32 / samples as f32;

                    let p_expected = hashers[0].collision_probability(d_xy);
                    assert!(
                        (p_xy - p_expected).abs() <= tolerance,
                        "expected {}, got {} (dot product={})",
                        p_expected,
                        p_xy,
                        d_xy,
                    );
                }
            }
        }
    }

    pub fn compute_recall<I1, I2>(ground: I1, answer: I2) -> f32
    where
        I1: IntoIterator<Item = usize>,
        I2: IntoIterator<Item = usize>,
    {
        use std::collections::HashSet;
        let ans: HashSet<usize> = answer.into_iter().collect();
        let mut present = 0;
        let mut total = 0;
        for x in ground.into_iter() {
            total += 1;
            if ans.contains(&x) {
                present += 1;
            }
        }
        present as f32 / total as f32
    }
}
