pub trait LSHFunction: Send + Sync {
    type Input;
    type Output: Eq + Ord;
    type Scratch;

    fn allocate_scratch(&self) -> Self::Scratch;

    fn hash(&self, v: &Self::Input, scratch: &mut Self::Scratch) -> Self::Output;

    fn collision_probability(&self, similarity: f32) -> f32;
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
