use ndarray::prelude::*;
use ndarray::Data;

pub fn norm2<S: Data<Elem = f32>>(v: &ArrayBase<S, Ix1>) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

pub fn cosine_similarity<S1, S2>(x: ArrayBase<S1, Ix1>, y: ArrayBase<S2, Ix1>) -> f32
where
    S1: Data<Elem = f32>,
    S2: Data<Elem = f32>,
{
    (x.dot(&y) + 1.0) / 2.0
}

pub trait LSHFunction {
    type Input;
    type Output: Eq + Ord;
    type Scratch;

    fn allocate_scratch(&self) -> Self::Scratch;

    fn hash(&self, v: &Self::Input, scratch: &mut Self::Scratch) -> Self::Output;
}

pub trait LSHFunctionBuilder {
    type LSH: LSHFunction;

    fn build(&mut self) -> Self::LSH;

    fn build_vec(&mut self, n: usize) -> Vec<Self::LSH> {
        let mut res = Vec::with_capacity(n);
        for _ in 0..n {
            res.push(self.build());
        }
        res
    }
}
