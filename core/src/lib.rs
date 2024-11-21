extern crate blas_src;

pub mod brute_force;
pub mod collision_index;
pub mod cross_polytope;
pub mod dataset;
pub mod lsh;
pub mod simhash;
pub mod types;

#[cfg(test)]
pub mod test {
    use crate::lsh::*;
    use crate::types::*;
    use ndarray::prelude::*;

    pub fn test_collision_probability<'a, F, B, O>(
        data: &'a Array2<f32>,
        mut builder: B,
        samples: usize,
        tolerance: f32,
    ) where
        F: LSHFunction<Input = ArrayView1<'a, f32>, Output = O>,
        B: LSHFunctionBuilder<LSH = F>,
        O: Eq + Copy,
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
