//! This is a reimplementation of PUFFINN

use core::hash;
use std::{
    io::{BufReader, BufWriter},
    marker::PhantomData,
    ops::Range,
    path::Path,
    time::{Duration, Instant},
};

use ndarray_rand::{
    rand::thread_rng,
    rand_distr::{num_traits::ToBytes, Bernoulli, Distribution, Uniform},
};
use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone, Copy)]
pub struct Stats {
    elapsed: Duration,
    collisions: usize,
    prefix: usize,
    repetitions: usize,
}

pub struct StatsBuilder {
    start: Instant,
    collisions: usize,
}

impl Default for StatsBuilder {
    fn default() -> Self {
        Self {
            start: Instant::now(),
            collisions: 0,
        }
    }
}

impl StatsBuilder {
    #[inline]
    fn inc_collisions(&mut self, collisions: usize) {
        self.collisions += collisions;
    }
    fn build(self, prefix: usize, repetitions: usize) -> Stats {
        Stats {
            elapsed: self.start.elapsed(),
            collisions: self.collisions,
            prefix,
            repetitions,
        }
    }
}

#[derive(Eq, PartialEq)]
pub struct Index<'data, P, D, H, LSH>
where
    H: PrefixCmp + Ord,
    D: Dataset<'data, P>,
    D::Distance: Into<f32>,
    LSH: LSHFunction<Output = H>,
{
    data: &'data D,
    hashers: Vec<LSH>,
    repetitions: Vec<Repetition<H>>,
    _marker: PhantomData<P>,
}

impl<'data, P, D, H, LSH> Index<'data, P, D, H, LSH>
where
    H: PrefixCmp + Ord + Send + Sync,
    D: Dataset<'data, P> + Sync,
    D::Distance: Into<f32>,
    LSH: LSHFunction<Input = P, Output = H>,
{
    pub fn build(data: &'data D, hashers: Vec<LSH>) -> Self {
        use rayon::prelude::*;
        let repetitions: Vec<Repetition<H>> = hashers
            // FIXME: this does not seem to use all the cores as it should
            .par_iter()
            .map(|h| {
                eprintln!("build repetition");
                Repetition::build(data, h)
            })
            .collect();
        Self {
            data,
            hashers,
            repetitions,
            _marker: PhantomData::<P>,
        }
    }

    pub fn search(&self, query: &P, delta: f32, out: &mut [(D::Distance, usize)]) -> Stats {
        let mut stats = StatsBuilder::default();
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
                // OPTIMIZE: hash all the repetitions with a single call, to allow
                // for optimizations like using the Fast Hadamard Transform for
                // many dot products
                let h = hasher.hash(&query, &mut scratch);
                rep.cursor(h, H::max_prefix())
            })
            .collect();

        // OPTIMIZE: come up with a data structure that never allocates during the query
        let mut priority = std::collections::BinaryHeap::<(D::Distance, usize)>::new();

        for prefix in (0..=H::max_prefix()).rev() {
            for (repetition_idx, repetition) in cursors.iter_mut().enumerate() {
                for i in repetition.collisions() {
                    stats.inc_collisions(1);
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
                        return stats.build(prefix, repetition_idx + 1);
                    }
                }
                repetition.shorten_prefix()
            }
        }
        unreachable!();
    }

    pub fn failure_probability(&self, d: D::Distance, prefix: usize, repetition: usize) -> f32 {
        let max_prefix = H::max_prefix();
        // NOTE: the assumption is that all hash functions produce
        // the same collision probability
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

impl<'data, P, D, H, LSH> Index<'data, P, D, H, LSH>
where
    for<'de> H: PrefixCmp + Ord + Send + Sync + bytemuck::AnyBitPattern + bytemuck::NoUninit,
    D: Dataset<'data, P> + Sync,
    D::Distance: Into<f32>,
    for<'de> LSH: LSHFunction<Input = P, Output = H> + Serialize + Deserialize<'de>,
{
    fn serialize_into<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        let timer = Instant::now();
        let data_sha = self.data.sha2()?;
        eprintln!("data sha computed in {:?}", timer.elapsed());
        out.write_all(&data_sha)?;
        bincode::serialize_into(&mut *out, &self.hashers).unwrap();
        out.write_all(&(self.repetitions.len() as u64).to_be_bytes())?;
        for rep in &self.repetitions {
            rep.serialize_into(&mut *out)?;
        }
        eprintln!("Index serialized in {:?}", timer.elapsed());
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(input: &mut R, data: &'data D) -> std::io::Result<Self> {
        let timer = Instant::now();
        let data_sha = data.sha2()?;
        let mut expected_sha = [0u8; 32];
        input.read_exact(&mut expected_sha)?;
        if expected_sha != data_sha {
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "mismatch between data sha and stored sha: the index was created for another dataset"));
        }
        let hashers: Vec<LSH> = bincode::deserialize_from(&mut *input).unwrap();
        // let repetitions: Vec<Repetition<H>> = bincode::deserialize_from(&mut *input).unwrap();
        let mut n = [0u8; 8];
        input.read_exact(&mut n)?;
        let n = u64::from_be_bytes(n) as usize;
        let mut repetitions: Vec<Repetition<H>> = Vec::new();
        for _ in 0..n {
            repetitions.push(Repetition::deserialize_from(input)?);
        }

        eprintln!("Index de-serialized in {:?}", timer.elapsed());
        Ok(Self {
            data,
            hashers,
            repetitions,
            _marker: PhantomData::<P>,
        })
    }

    pub fn save<PP: AsRef<Path>>(&self, path: PP) -> std::io::Result<()> {
        use flate2::write::ZlibEncoder;
        use flate2::Compression;

        let file = std::fs::File::create(&path)?;
        let mut out = ZlibEncoder::new(BufWriter::new(file), Compression::default());
        self.serialize_into(&mut out)
    }

    pub fn load<PP: AsRef<Path>>(path: PP, data: &'data D) -> std::io::Result<Self> {
        use flate2::read::ZlibDecoder;
        let file = std::fs::File::open(&path)?;
        let mut input = ZlibDecoder::new(BufReader::new(file));
        Self::deserialize_from(&mut input, data)
    }

    /// First checks if the index with the given `hashers` on the given `data`
    /// has already been built and is stored in `cache_dir`.
    /// If so, it will return the cached version. Otherwise, the function will
    /// build the index, cache it, and return it.
    ///
    pub fn build_cached<PP: AsRef<Path>>(
        data: &'data D,
        hashers: Vec<LSH>,
        cache_dir: PP,
    ) -> std::io::Result<Self> {
        use sha2::Digest;
        let mut hashers_digest = sha2::Sha256::default();
        hashers_digest.update(&bincode::serialize(&hashers).unwrap());
        let data_sha: [u8; 32] = data.sha2()?.into();
        let hashers_sha: [u8; 32] = hashers_digest.finalize().into();
        let cache_dir = cache_dir.as_ref();
        if !cache_dir.is_dir() {
            std::fs::create_dir(&cache_dir)?;
        }
        let path = cache_dir.join(format!(
            "{}-{}.z",
            data_sha
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect::<String>(),
            hashers_sha
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect::<String>(),
        ));
        if !path.is_file() {
            eprintln!("index not in cache, building");
            let timer = Instant::now();
            let index = Self::build(data, hashers);
            eprintln!("time to build index: {:?}", timer.elapsed());
            index.save(path)?;
            Ok(index)
        } else {
            Self::load(path, data)
        }
    }
}

#[derive(Serialize, Deserialize, Eq, PartialEq)]
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

impl<H: PrefixCmp + Ord + bytemuck::NoUninit + bytemuck::AnyBitPattern> Repetition<H> {
    fn serialize_into<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        assert_eq!(self.hashes.len(), self.indices.len());
        let n: u64 = self.hashes.len() as u64;
        out.write(&n.to_be_bytes())?;
        out.write_all(bytemuck::cast_slice(&self.hashes))?;
        out.write_all(bytemuck::cast_slice(&self.indices))?;
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(input: &mut R) -> std::io::Result<Self> {
        let mut len_bytes = [0u8; 8];
        input.read_exact(&mut len_bytes)?;
        let n = u64::from_be_bytes(len_bytes) as usize;
        let mut hashes_bytes = vec![0u8; n * std::mem::size_of::<H>()];
        let mut indices_bytes = vec![0u8; n * std::mem::size_of::<usize>()];

        input.read_exact(&mut hashes_bytes)?;
        input.read_exact(&mut indices_bytes)?;

        let hashes: &[H] = bytemuck::cast_slice(&hashes_bytes);
        let indices: &[usize] = bytemuck::cast_slice(&indices_bytes);

        Ok(Self {
            hashes: hashes.to_vec(),
            indices: indices.to_vec(),
        })
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
        let start = repetition
            .hashes
            .partition_point(|hprime| hprime.prefix_cmp(&hash, prefix).is_lt());
        let end = start
            + repetition.hashes[start..].partition_point(|hprime| hprime.prefix_eq(&hash, prefix));
        assert!(end >= start);
        let range = start..end;
        let slf = Self {
            repetition,
            hash,
            prefix,
            range,
            prev_range: None,
        };
        #[cfg(test)]
        slf.check_invariant();
        slf
    }

    #[cfg(test)]
    fn check_invariant(&self) {
        let prefix = self.prefix;
        let hash = &self.hash;
        debug_assert!(
            self.repetition.hashes[..self.range.start]
                .iter()
                .all(|h| !hash.prefix_eq(h, prefix)),
            "all hashes before the Cursor's range should have a different prefix"
        );
        debug_assert!(
            self.repetition.hashes[self.range.end..]
                .iter()
                .all(|h| !hash.prefix_eq(h, prefix)),
            "all hashes after the Cursor's range should have a different prefix"
        );
        debug_assert!(
            self.repetition.hashes[self.range.clone()]
                .iter()
                .all(|h| hash.prefix_eq(h, prefix)),
            "all hashes in the Cursor's range should have the same prefix"
        );
    }

    /// shortens the prefix by one
    fn shorten_prefix(&mut self) {
        assert!(self.prefix >= 1);
        self.prev_range.replace(self.range.clone());
        self.prefix -= 1;
        // OPTIMIZE: possibly retrict the search to the unexplored part of the hashes
        let start = self
            .repetition
            .hashes
            .partition_point(|hprime| hprime.prefix_cmp(&self.hash, self.prefix).is_lt());
        let end = start
            + self.repetition.hashes[start..]
                .partition_point(|hprime| hprime.prefix_eq(&self.hash, self.prefix));
        assert!(end >= start);
        self.range = start..end;
        #[cfg(test)]
        self.check_invariant();
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

    fn collisions_naive<'slf>(&'slf self) -> impl Iterator<Item = usize> + 'slf {
        let hh = &self.hash;
        let prefix = self.prefix;
        self.repetition
            .indices
            .iter()
            .zip(self.repetition.hashes.iter())
            .filter_map(move |(i, h)| {
                if h.prefix_eq(&hh, prefix) {
                    Some(*i)
                } else {
                    None
                }
            })
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
    use std::{str::FromStr, time::Instant};

    use ndarray::s;
    use ndarray_rand::rand::{thread_rng, Rng};

    use crate::{
        dataset::{
            load_distances, load_raw_queries, AngularDataset, DistanceF32, EuclideanDataset,
        },
        lsh::*,
        simhash::{SimHash, SimHashBuilder},
    };

    use super::*;

    fn compute_recall(actual: &[(DistanceF32, usize)], ground: &[f32]) -> f32 {
        let epsilon = 0.000001;
        let k = actual.len();
        assert!(k >= 1);
        let thresh = DistanceF32::from(ground[k - 1] + epsilon);
        let mut cnt = 0;
        for (d, _) in actual {
            if *d <= thresh {
                cnt += 1;
            }
        }
        cnt as f32 / k as f32
    }

    #[test]
    fn test_glove_k10_nn_puffinn() {
        let mut rng = thread_rng();

        let path = ".glove-100-angular.hdf5";
        let dataset = AngularDataset::from_hdf5(&path);
        let queries = load_raw_queries(&path);
        let distances = load_distances(&path);

        dbg!();
        let hashers =
            SimHashBuilder::<&[f32], _>::new(dataset.num_dimensions(), 32, &mut rng).build_vec(64);

        let index = Index::build(&dataset, hashers);
        dbg!();

        let k = 10;
        let delta = 0.1;
        let mut out = vec![(DistanceF32::from(0.0f32), 0); k];
        let nqueries = 10.min(distances.nrows());
        let start = Instant::now();
        let mut answers = queries
            .rows()
            .into_iter()
            .zip(distances.rows().into_iter())
            .take(nqueries)
            .enumerate()
            .map(|(idx, (query, ground))| {
                dbg!();
                let query = query.as_slice().unwrap();
                let collisions = index.search(&query, delta, &mut out);
                (
                    compute_recall(&out, &ground.as_slice().unwrap()),
                    collisions,
                    idx,
                )
            })
            .collect::<Vec<(f32, Stats, usize)>>();
        let recall = answers.iter().map(|pair| pair.0).sum::<f32>() / nqueries as f32;
        let elapsed = Instant::now() - start;
        let qps = nqueries as f64 / elapsed.as_secs_f64();

        answers.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        dbg!(&answers[..10]);

        dbg!(qps);
        dbg!(recall);
        assert!(recall >= 1.0 - delta);
    }

    #[test]
    fn test_serialization() {
        use crate::puffinn::*;
        let mut rng = thread_rng();
        let path = ".glove-100-angular.hdf5";
        let dataset = AngularDataset::from_hdf5(&path);
        let hashers =
            SimHashBuilder::<&[f32], _>::new(dataset.num_dimensions(), 32, &mut rng).build_vec(64);
        let timer = Instant::now();
        let index = Index::build(&dataset, hashers);
        eprintln!("index built in {:?}", timer.elapsed());

        // check that the first repetition is serialized and deserialized correctly
        let mut buf = Vec::<u8>::new();
        index.repetitions[0].serialize_into(&mut buf).unwrap();
        let first_repetition =
            Repetition::<BitHash32>::deserialize_from(&mut std::io::Cursor::new(buf)).unwrap();
        assert!(index.repetitions[0] == first_repetition);

        // check that the index serializes and de-serializes correctly
        let test_path = "/tmp/panna-ser.z";
        index.save(&test_path).expect("error saving the index");
        let loaded_index = Index::<_, _, _, SimHash<&[f32]>>::load(&test_path, &dataset)
            .expect("problem loading dataset");

        assert!(index == loaded_index);

        // loading the same index for a different dataset should result in an error
        let dataset = EuclideanDataset::from_hdf5(&path);
        assert!(Index::<_, _, _, SimHash<&[f32]>>::load(&test_path, &dataset).is_err())
    }
}
