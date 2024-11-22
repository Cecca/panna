//! This is a reimplementation of PUFFINN

use std::ops::{Range, RangeBounds};

use crate::{
    dataset::Dataset,
    lsh::{BitHash32, LSHFunction},
};

pub trait PrefixCmp {
    fn max_prefix() -> usize;
    fn prefix_eq(&self, other: &Self, prefix: usize) -> bool;
    fn prefix_cmp(&self, other: &Self, prefix: usize) -> std::cmp::Ordering;
}

const BITHASH32_MASKS: [u32; 33] = [
    0b00000000000000000000000000000000,
    0b10000000000000000000000000000000,
    0b11000000000000000000000000000000,
    0b11100000000000000000000000000000,
    0b11110000000000000000000000000000,
    0b11111000000000000000000000000000,
    0b11111100000000000000000000000000,
    0b11111110000000000000000000000000,
    0b11111111000000000000000000000000,
    0b11111111100000000000000000000000,
    0b11111111110000000000000000000000,
    0b11111111111000000000000000000000,
    0b11111111111100000000000000000000,
    0b11111111111110000000000000000000,
    0b11111111111111000000000000000000,
    0b11111111111111100000000000000000,
    0b11111111111111110000000000000000,
    0b11111111111111111000000000000000,
    0b11111111111111111100000000000000,
    0b11111111111111111110000000000000,
    0b11111111111111111111000000000000,
    0b11111111111111111111100000000000,
    0b11111111111111111111110000000000,
    0b11111111111111111111111000000000,
    0b11111111111111111111111100000000,
    0b11111111111111111111111110000000,
    0b11111111111111111111111111000000,
    0b11111111111111111111111111100000,
    0b11111111111111111111111111110000,
    0b11111111111111111111111111111000,
    0b11111111111111111111111111111100,
    0b11111111111111111111111111111110,
    0b11111111111111111111111111111111,
];

impl PrefixCmp for BitHash32 {
    fn max_prefix() -> usize {
        32
    }
    fn prefix_eq(&self, other: &Self, prefix: usize) -> bool {
        debug_assert!(prefix <= 32);
        let mask = BITHASH32_MASKS[prefix];
        (self.0 & mask) == (other.0 & mask)
    }
    fn prefix_cmp(&self, other: &Self, prefix: usize) -> std::cmp::Ordering {
        debug_assert!(prefix <= 32);
        let mask = BITHASH32_MASKS[prefix];
        (self.0 & mask).cmp(&(other.0 & mask))
    }
}

pub struct Index<'data, P, D, H, LSH>
where
    H: PrefixCmp + Ord,
    D: Dataset<'data, P>,
    D::Distance: Into<f32>,
    LSH: LSHFunction<Input = P, Output = H>,
{
    data: &'data D,
    hashers: Vec<LSH>,
    repetitions: Vec<Repetition<H>>,
}

impl<'data, P, D, H, LSH> Index<'data, P, D, H, LSH>
where
    H: PrefixCmp + Ord,
    D: Dataset<'data, P>,
    D::Distance: Into<f32>,
    LSH: LSHFunction<Input = P, Output = H>,
{
    pub fn build(data: &'data D, hashers: Vec<LSH>) -> Self {
        // TODO: build in parallel
        let repetitions: Vec<Repetition<H>> =
            hashers.iter().map(|h| Repetition::build(data, h)).collect();
        Self {
            data,
            hashers,
            repetitions,
        }
    }

    pub fn search(&self, query: &P, delta: f32, out: &mut [(D::Distance, usize)]) {
        let k = out.len();
        let prepared = {
            let mut p = self.data.default_prepared_query();
            self.data.prepare(query, &mut p);
            p
        };
        // Initialize the cursors
        let mut scratch = self.hashers[0].allocate_scratch();
        let mut cursors: Vec<Cursor<H>> = self
            .repetitions
            .iter()
            .zip(self.hashers.iter())
            .map(|(rep, hasher)| {
                let h = hasher.hash(&query, &mut scratch);
                rep.cursor(h, H::max_prefix())
            })
            .collect();

        // OPTIMIZE: come up with a data structure that never allocates during the query
        let mut priority = std::collections::BinaryHeap::<(D::Distance, usize)>::new();

        for prefix in (0..=H::max_prefix()).rev() {
            dbg!(prefix);
            for (repetition_idx, repetition) in cursors.iter_mut().enumerate() {
                for i in repetition.collisions() {
                    let d = self.data.distance(i, &prepared);
                    if priority.len() < k || d < priority.peek().unwrap().0 {
                        let pair = (d, i);
                        if priority.iter().find(|x| **x == pair).is_none() {
                            priority.push((d, i));
                            while priority.len() > k {
                                priority.pop();
                            }
                        }
                    }
                }

                if priority.len() == k {
                    let max_d = priority.peek().unwrap().0;
                    let fp = self.failure_probability(max_d, prefix, repetition_idx + 1);
                    if fp < delta {
                        // copy the values in the output array
                        for i in (0..out.len()).rev() {
                            out[i] = priority.pop().unwrap();
                        }
                        assert!(priority.is_empty());
                        return;
                    }
                }
            }
        }
        unreachable!();
    }

    pub fn failure_probability(&self, d: D::Distance, prefix: usize, repetition: usize) -> f32 {
        let max_prefix = H::max_prefix();
        let p = self.hashers[0].collision_probability(d.into());
        if prefix < max_prefix {
            (1.0 - p.powi(prefix as i32)).powi(repetition as i32)
                * (1.0 - p.powi((prefix + 1) as i32))
                    .powi((self.repetitions.len() - repetition) as i32)
        } else {
            (1.0 - p.powi(prefix as i32)).powi(repetition as i32)
        }
    }
}

struct Repetition<H: PrefixCmp + Ord> {
    hashes: Vec<H>,
    indices: Vec<usize>,
}

impl<H: PrefixCmp + Ord> Repetition<H> {
    fn build<'data, D, P, LSH>(data: &'data D, hasher: &LSH) -> Self
    where
        D: Dataset<'data, P>,
        LSH: LSHFunction<Input = P, Output = H>,
    {
        let mut scratch = hasher.allocate_scratch();
        // FIXME: remove this memory allocation. Possibly a radix sort might do
        let mut tmp = Vec::with_capacity(data.num_points());
        for i in 0..data.num_points() {
            let v = data.get(i);
            let h = hasher.hash(&v, &mut scratch);
            tmp.push((h, i));
        }

        tmp.sort_unstable();
        let (hashes, indices) = tmp.into_iter().unzip();
        Self { hashes, indices }
    }

    fn cursor<'slf>(&'slf self, hash: H, prefix: usize) -> Cursor<'slf, H> {
        Cursor::new(self, hash, prefix)
    }
}

struct Cursor<'rep, H: PrefixCmp + Ord> {
    repetition: &'rep Repetition<H>,
    hash: H,
    prefix: usize,
    range: Range<usize>,
    prev_range: Option<Range<usize>>,
}

impl<'rep, H: PrefixCmp + Ord> Cursor<'rep, H> {
    fn new(repetition: &'rep Repetition<H>, hash: H, prefix: usize) -> Self {
        // Look for the starting index
        let start = match repetition
            .hashes
            .binary_search_by(|hprime| hprime.prefix_cmp(&hash, prefix))
        {
            Ok(i) => i,
            Err(i) => i,
        };
        let end = start
            + repetition.hashes[start..].partition_point(|hprime| hprime.prefix_eq(&hash, prefix));
        assert!(end >= start);
        let range = start..end;
        Self {
            repetition,
            hash,
            prefix,
            range,
            prev_range: None,
        }
    }

    /// shortens the prefix by one
    fn shorten_prefix(&mut self) {
        assert!(self.prefix >= 1);
        self.prev_range.replace(self.range.clone());
        self.prefix -= 1;
        // OPTIMIZE: possibly retrict the search to the unexplored part of the hashes
        let start = match self
            .repetition
            .hashes
            .binary_search_by(|hprime| hprime.prefix_cmp(&self.hash, self.prefix))
        {
            Ok(i) => i,
            Err(i) => i,
        };
        let end = start
            + self.repetition.hashes[start..]
                .partition_point(|hprime| hprime.prefix_eq(&self.hash, self.prefix));
        assert!(end >= start);
        self.range = start..end;
    }

    fn pair_collisions<'slf>(&'slf self) -> PairsIterator<'slf> {
        if let Some(prev_range) = self.prev_range.as_ref() {
            PairsIterator::nested(
                &self.repetition.indices,
                self.range.start..prev_range.start,
                prev_range.clone(),
                prev_range.end..self.range.end,
            )
        } else {
            PairsIterator::flat(&self.repetition.indices, self.range.clone())
        }
    }

    fn collisions<'slf>(&'slf self) -> RangesIterator<'slf> {
        if let Some(prev_range) = self.prev_range.as_ref() {
            RangesIterator::nested(
                &self.repetition.indices,
                prev_range.clone(),
                self.range.clone(),
            )
        } else {
            RangesIterator::flat(&self.repetition.indices, self.range.clone())
        }
    }
}

pub enum RangesIterator<'values> {
    Flat {
        values: &'values [usize],
        range: Range<usize>,
    },
    Nested {
        values: &'values [usize],
        first: Range<usize>,
        second: Range<usize>,
    },
}
impl<'values> RangesIterator<'values> {
    fn flat(values: &'values [usize], range: Range<usize>) -> Self {
        Self::Flat { values, range }
    }
    fn nested(values: &'values [usize], inner: Range<usize>, outer: Range<usize>) -> Self {
        Self::Nested {
            values,
            first: outer.start..inner.start,
            second: inner.end..outer.end,
        }
    }
}
impl<'values> Iterator for RangesIterator<'values> {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Flat { values, range } => range.next().map(|i| values[i]),
            Self::Nested {
                values,
                first,
                second,
            } => first.next().or_else(|| second.next()).map(|i| values[i]),
        }
    }
}

pub enum PairsIterator<'values> {
    /// Gets all pairs of values in the given range
    Flat {
        values: &'values [usize],
        range: Range<usize>,
        i: usize,
        j: usize,
    },
    Flat2 {
        values: &'values [usize],
        range1: Range<usize>,
        range2: Range<usize>,
        i: usize,
        j: usize,
    },
    Nested {
        iters: Box<[Self; 5]>,
        cur_iter: usize,
    },
}

impl<'values> PairsIterator<'values> {
    fn flat(values: &'values [usize], range: Range<usize>) -> Self {
        Self::Flat {
            values,
            i: range.start,
            j: range.start + 1,
            range,
        }
    }
    fn flat2(values: &'values [usize], range1: Range<usize>, range2: Range<usize>) -> Self {
        Self::Flat2 {
            values,
            i: range1.start,
            j: range2.start,
            range1,
            range2,
        }
    }
    fn nested(
        values: &'values [usize],
        pre: Range<usize>,
        mid: Range<usize>,
        post: Range<usize>,
    ) -> Self {
        Self::Nested {
            iters: Box::new([
                Self::flat(values, pre.clone()),
                Self::flat2(values, pre.clone(), mid.clone()),
                Self::flat2(values, pre.clone(), post.clone()),
                Self::flat2(values, mid.clone(), post.clone()),
                Self::flat(values, post.clone()),
            ]),
            cur_iter: 0,
        }
    }
}
impl<'values> Iterator for PairsIterator<'values> {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Flat {
                values,
                range,
                i,
                j,
            } => {
                while *i < range.end {
                    if *j >= range.end {
                        *i += 1;
                        *j = *i + 1;
                    } else {
                        let toret = (values[*i], values[*j]);
                        *j += 1;
                        return Some(toret);
                    }
                }
                None
            }
            Self::Flat2 {
                values,
                range1,
                range2,
                i,
                j,
            } => {
                while *i < range1.end {
                    if *j >= range2.end {
                        *i += 1;
                        *j = range2.start;
                    } else {
                        let toret = (values[*i], values[*j]);
                        *j += 1;
                        return Some(toret);
                    }
                }
                None
            }
            Self::Nested { iters, cur_iter } => {
                while *cur_iter < iters.len() {
                    let head = iters[*cur_iter].next();
                    if head.is_some() {
                        return head;
                    } else {
                        *cur_iter += 1;
                    }
                }
                None
            }
        }
    }
}

#[test]
fn test_pair_collision_iterator() {
    let values = vec![10, 11, 12, 13];
    let range = 0..values.len();
    assert_eq!(
        PairsIterator::flat(&values, range).collect::<Vec<(usize, usize)>>(),
        vec![(10, 11), (10, 12), (10, 13), (11, 12), (11, 13), (12, 13),]
    );

    let values = vec![10, 11, 12, 13, 14, 15, 16];
    assert_eq!(
        PairsIterator::flat2(&values, 0..3, 4..values.len()).collect::<Vec<(usize, usize)>>(),
        vec![
            (10, 14),
            (10, 15),
            (10, 16),
            (11, 14),
            (11, 15),
            (11, 16),
            (12, 14),
            (12, 15),
            (12, 16)
        ]
    );

    let values = vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    let pre = 0..4;
    let mid = 4..10;
    let post = 10..values.len();
    let mut expected = std::collections::BTreeSet::new();
    for i in 0..values.len() {
        for j in (i + 1)..values.len() {
            if !(mid.contains(&i) & mid.contains(&j)) {
                expected.insert((values[i], values[j]));
            }
        }
    }

    assert_eq!(
        PairsIterator::nested(&values, pre, mid, post)
            .collect::<std::collections::BTreeSet<(usize, usize)>>(),
        expected
    );
}

#[cfg(test)]
mod test {
    use ndarray_rand::rand::{thread_rng, Rng};

    use crate::{
        dataset::{load_distances, load_raw_queries, AngularDataset},
        lsh::*,
        simhash::SimHashBuilder,
    };

    use super::*;

    #[test]
    fn test_glove_k10_nn() {
        let mut rng = thread_rng();

        let path = ".glove-100-angular.hdf5";
        let dataset = AngularDataset::from_hdf5(&path);
        let queries = load_raw_queries(&path);
        let distances = load_distances(&path);

        let hashers =
            SimHashBuilder::<&[f32], _>::new(dataset.num_dimensions(), 32, &mut rng).build_vec(128);

        // let index = Index::build()

        // let qidx = 0;
        // let query = queries.row(qidx);

        // let k = 10;
        // let delta = 0.9;

        // let expected = distances.row(qidx);
        // let actual = brute_force_knn(&dataset, &prepared_query, k);
        // for (idx, (d, _i)) in actual.into_iter().enumerate() {
        //     let d: f32 = d.into();
        //     let ed = expected[idx];
        //     assert!((d - ed).abs() <= 0.0001);
        // }
    }
}
