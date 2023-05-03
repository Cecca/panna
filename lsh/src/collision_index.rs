use crate::types::{Dataset, LSHFunction, LSHFunctionBuilder};

pub struct CollisionIndex<'data, H: LSHFunction, D: Dataset<'data, H::Input>> {
    hashers: Vec<H>,
    data: &'data D,
    tables: Vec<Vec<(H::Output, usize)>>,
}

impl<'data, H: LSHFunction, D: Dataset<'data, H::Input>> CollisionIndex<'data, H, D>
where
    H::Output: Clone,
{
    pub fn new<B: LSHFunctionBuilder<LSH = H>>(
        data: &'data D,
        mut builder: B,
        max_repetitions: usize,
    ) -> Self {
        let hashers = builder.build_vec(max_repetitions);
        let mut tables = vec![Vec::new(); max_repetitions];
        let mut scratch = hashers[0].allocate_scratch();
        for (h, table) in hashers.iter().zip(tables.iter_mut()) {
            for i in 0..data.num_points() {
                let v = data.get(i);
                table.push((h.hash(&v, &mut scratch), i));
            }
            // Sort the table so that we can binary search in it.
            table.sort_unstable();
        }
        Self {
            hashers,
            data,
            tables,
        }
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use crate::types::*;
    use crate::simhash::*;
    use ndarray::prelude::*;

    #[test]
    fn test_index() {
        let repetitions = 128;
        let data = crate::test::load_glove25();
        let rng = ndarray_rand::rand::thread_rng();
        let builder = SimHashBuilder::<ArrayView1<f32>, _>::new(data.ncols(), 8, rng);
        let index = CollisionIndex::new(&data, builder, repetitions);
    }
}

