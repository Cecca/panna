pub mod simhash;
pub mod types;
//pub mod cross_polytope;

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

    pub fn test_collision_prob_ranking_cosine<'a, F, B>(
        data: &'a Array2<f32>,
        mut builder: B,
        samples: usize,
        tolerance: f64,
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

        let mut pairs = Vec::new();
        for i in 0..n {
            let x = data.row(i);
            let hx = &hashes[i];
            for j in (i + 1)..n {
                let y = data.row(j);
                let d_xy = cosine_similarity(x, y);
                let hy = &hashes[j];
                let p_xy =
                    hx.iter().zip(hy).filter(|(x, y)| x == y).count() as f64 / samples as f64;
                pairs.push((d_xy, p_xy));
            }
        }

        pairs.sort_by(|p1, p2| p1.0.partial_cmp(&p2.0).unwrap().reverse());
        for i in 1..pairs.len() {
            println!("{:?} {:?}", pairs[i - 1], pairs[i]);
            assert!(pairs[i - 1].1 >= pairs[i].1 - tolerance);
        }
    }
}
