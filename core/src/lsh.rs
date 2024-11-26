/// The hash value obtained by concatenating up to 32 bit-wise hash functions.
/// The first concatenation picks the most significant bit: doing so the
/// lexicographic ordering of the hashes groups together hashes with the
/// same prefix
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct BitHash32(pub u32);
impl BitHash32 {
    pub fn set(&mut self, i: usize, v: bool) {
        let bit_pos = std::mem::size_of::<u32>() * 8 - i - 1;
        let mask = 1 << bit_pos;
        self.0 = (self.0 & !mask) | ((v as u32) << bit_pos);
    }
}
impl std::fmt::Debug for BitHash32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:032b}", self.0)
    }
}

#[test]
fn test_bithash() {
    let mut h = BitHash32::default();
    assert_eq!(h.0, 0);

    h.set(0, true);
    assert_eq!(h.0, 1 << 31);
    h.set(0, false);
    assert_eq!(h.0, 0);
    h.set(1, true);
    assert_eq!(h.0, 1 << 30);
    h.set(2, true);
    assert_eq!(h.0, (1 << 30) | (1 << 29));

    let mut h = BitHash32::default();
    h.set(31, true);
    assert_eq!(h.0, 1);

    let mut a = BitHash32::default();
    let mut b = BitHash32::default();
    a.set(0, true);
    a.set(1, false);
    b.set(0, false);
    b.set(1, true);
    assert!(b < a, "a={:?} b={:?}", a, b);
}

pub trait LSHFunction {
    type Input;
    type Output: Eq + Ord;
    type Scratch;

    fn allocate_scratch(&self) -> Self::Scratch;

    fn hash(&self, v: &Self::Input, scratch: &mut Self::Scratch) -> Self::Output;

    /// The probability of a single hash function to collide
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
