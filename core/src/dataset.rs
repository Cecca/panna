use ndarray::prelude::*;

#[derive(PartialEq, PartialOrd, Clone, Copy)]
pub struct DistanceF32(f32);
impl Eq for DistanceF32 {}
impl Ord for DistanceF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}
impl From<f32> for DistanceF32 {
    fn from(value: f32) -> Self {
        Self(value)
    }
}
impl Into<f32> for DistanceF32 {
    fn into(self) -> f32 {
        self.0
    }
}
impl std::fmt::Debug for DistanceF32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::fmt::Display for DistanceF32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

const PADDED_VECTORS_BASE: usize = 8;
struct PaddedVectors {
    stride: usize,
    dimensions: usize,
    num_vectors: usize,
    vectors: Vec<f32>,
}
impl PaddedVectors {
    fn new<A: AsRef<[f32]>, I: Iterator<Item = A>>(
        dimensions: usize,
        num_vectors: usize,
        input_vectors: I,
    ) -> Self {
        let padding =
            (PADDED_VECTORS_BASE - (dimensions % PADDED_VECTORS_BASE)) % PADDED_VECTORS_BASE;
        let stride = dimensions + padding;
        let num_elems = stride * num_vectors;
        let mut vectors = Vec::with_capacity(num_elems);
        vectors.resize(num_elems, 0.0);
        for (i, v) in input_vectors.enumerate() {
            let pos = i * stride;
            vectors[pos..pos + dimensions].copy_from_slice(v.as_ref());
        }
        Self {
            stride,
            dimensions,
            num_vectors,
            vectors,
        }
    }
}
impl std::ops::Index<usize> for PaddedVectors {
    type Output = [f32];

    fn index(&self, index: usize) -> &Self::Output {
        let offset = index * self.stride;
        &self.vectors[offset..offset + self.stride]
    }
}

/// Computes the dot product, using SIMD fused-multiply-add instructions,
/// assuming that the length of the arrays is _exactly_ a multiple of 8
#[cfg(all(target_feature = "fma"))]
#[inline(always)]
unsafe fn dot_fma_exact(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    const LANES: usize = 8;
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len() % LANES, 0);

    let ca = a.chunks_exact(LANES);
    let cb = b.chunks_exact(LANES);

    let mut cc = _mm256_setzero_ps();

    for (aa, bb) in ca.zip(cb) {
        let aa = _mm256_loadu_ps(aa.as_ptr());
        let bb = _mm256_loadu_ps(bb.as_ptr());

        cc = _mm256_fmadd_ps(aa, bb, cc);
    }

    let mut buf = [0.0f32; LANES];
    _mm256_storeu_ps(buf.as_mut_ptr(), cc);

    buf.into_iter().sum::<f32>()
}

#[cfg(all(target_feature = "fma"))]
#[inline(always)]
unsafe fn dot_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(a.len(), b.len());

    const LANES: usize = 8;

    let ca = a.chunks_exact(LANES);
    let cb = b.chunks_exact(LANES);

    let rem = ca
        .remainder()
        .into_iter()
        .zip(cb.remainder())
        .map(|(aa, bb)| aa * bb)
        .sum::<f32>();

    let mut cc = _mm256_setzero_ps();

    for (aa, bb) in ca.zip(cb) {
        let aa = _mm256_loadu_ps(aa.as_ptr());
        let bb = _mm256_loadu_ps(bb.as_ptr());

        cc = _mm256_fmadd_ps(aa, bb, cc);
    }

    let mut buf = [0.0f32; LANES];
    _mm256_storeu_ps(buf.as_mut_ptr(), cc);

    rem + buf.into_iter().sum::<f32>()
}

pub trait Dataset<'slf, Point> {
    /// The distance data type returned, e.g. `f64` of `u32`
    type Distance: Sized + Ord + Clone + Copy;
    /// A type to be used for prepared queries
    type PreparedPoint;

    /// Prepare a query to be in a format that allows to compute distances with
    /// points of this dataset
    fn prepare(&self, query: &Point, output: &mut Self::PreparedPoint);

    fn default_prepared_query(&self) -> Self::PreparedPoint;

    /// compute the distance between the given query and the i-th point
    fn distance(&self, i: usize, query: &Self::PreparedPoint) -> Self::Distance;

    /// Gets the i-th point
    fn get(&'slf self, i: usize) -> Point;

    /// How many dimensions the dataset has
    fn num_dimensions(&self) -> usize;

    /// How many points the dataset contains
    fn num_points(&self) -> usize;
}

/// A dataset using the angular distance
pub struct AngularDatasetPadded {
    /// The points, which are normalized upon construction, and padded
    /// to have a length multiple of 8
    points: PaddedVectors,
}
impl AngularDatasetPadded {
    pub fn new(mut raw: Array2<f32>) -> Self {
        let points = PaddedVectors::new(
            raw.shape()[1],
            raw.shape()[0],
            raw.rows_mut().into_iter().map(|mut row| {
                let norm = norm_squared(&row).sqrt();
                row /= norm;
                // FIXME: remove this clone
                row.to_vec()
            }),
        );
        Self { points }
    }

    pub fn from_hdf5<P: AsRef<std::path::Path>>(path: P) -> Self {
        let f = hdf5::File::open(path.as_ref()).unwrap();
        let raw = f.dataset("/train").unwrap().read_2d::<f32>().unwrap();
        Self::new(raw)
    }
}
impl<'slf> Dataset<'slf, &'slf [f32]> for AngularDatasetPadded {
    type Distance = DistanceF32;

    type PreparedPoint = Vec<f32>;

    fn default_prepared_query(&self) -> Self::PreparedPoint {
        let mut v = Vec::new();
        v.resize(self.points.stride, 0.0);
        v
    }

    fn prepare(&self, query: &&[f32], output: &mut Self::PreparedPoint) {
        let norm = norm_squared_arr(query).sqrt();
        for i in 0..query.len() {
            output[i] = query[i] / norm;
        }
    }

    fn distance(&self, i: usize, query: &Self::PreparedPoint) -> Self::Distance {
        let v = self.get(i);
        let dotp = unsafe { dot_fma_exact(v, query) };
        (1.0 - dotp).into()
    }

    fn get(&'slf self, i: usize) -> &[f32] {
        &self.points[i]
    }

    fn num_dimensions(&self) -> usize {
        self.points.dimensions
    }

    fn num_points(&self) -> usize {
        self.points.num_vectors
    }
}

/// A dataset using the angular distance
pub struct AngularDataset {
    /// The points, which are normalized upon construction
    points: Array2<f32>,
}

impl AngularDataset {
    pub fn new(mut raw: Array2<f32>) -> Self {
        for mut row in raw.rows_mut() {
            let norm = norm_squared(&row).sqrt();
            row /= norm;
        }
        Self { points: raw }
    }

    pub fn from_hdf5<P: AsRef<std::path::Path>>(path: P) -> Self {
        let f = hdf5::File::open(path.as_ref()).unwrap();
        let raw = f.dataset("/train").unwrap().read_2d::<f32>().unwrap();
        Self::new(raw)
    }
}

impl<'slf> Dataset<'slf, ArrayView1<'slf, f32>> for AngularDataset {
    type Distance = DistanceF32;

    type PreparedPoint = Array1<f32>;

    fn default_prepared_query(&self) -> Self::PreparedPoint {
        Array1::zeros(self.num_dimensions())
    }

    fn prepare(&self, query: &ArrayView1<f32>, output: &mut Self::PreparedPoint) {
        assert_eq!(query.shape(), output.shape());
        let norm = norm_squared(&query).sqrt();
        for i in 0..query.len() {
            output[i] = query[i] / norm;
        }
    }

    fn distance(&self, i: usize, query: &Self::PreparedPoint) -> Self::Distance {
        let v = self.points.row(i);
        let dotp = unsafe { dot_fma(v.as_slice().unwrap(), query.as_slice().unwrap()) };
        (1.0 - dotp).into()
    }

    fn get(&'slf self, i: usize) -> ArrayView1<'slf, f32> {
        self.points.row(i)
    }

    fn num_dimensions(&self) -> usize {
        self.points.ncols()
    }

    fn num_points(&self) -> usize {
        self.points.nrows()
    }
}

/// A dataset using the Euclidean Distance
pub struct EuclideanDataset {
    /// The points
    points: PaddedVectors,
    /// The squared norms of the points
    squared_norms: Vec<f32>,
}

impl EuclideanDataset {
    pub fn new(raw: Array2<f32>) -> Self {
        let squared_norms: Vec<f32> = raw
            .rows()
            .into_iter()
            .map(|row| norm_squared(&row))
            .collect();
        assert_eq!(squared_norms.len(), raw.nrows());
        let points = PaddedVectors::new(
            raw.shape()[1],
            raw.shape()[0],
            raw.rows().into_iter().map(|r| {
                let v = r.as_slice().unwrap();
                // FIXME: remove this clone
                v.to_vec()
            }),
        );
        Self {
            points,
            squared_norms,
        }
    }

    pub fn from_hdf5<P: AsRef<std::path::Path>>(path: P) -> Self {
        let f = hdf5::File::open(path.as_ref()).unwrap();
        let raw = f.dataset("/train").unwrap().read_2d::<f32>().unwrap();
        Self::new(raw)
    }
}

impl<'slf> Dataset<'slf, &'slf [f32]> for EuclideanDataset {
    type Distance = DistanceF32;

    /// A prepared point is the point itself along with its norm
    type PreparedPoint = (Vec<f32>, f32);

    fn default_prepared_query(&self) -> Self::PreparedPoint {
        let mut v = Vec::new();
        v.resize(self.points.stride, 0.0);
        (v, 0.0)
    }

    fn prepare(&self, query: &&[f32], output: &mut Self::PreparedPoint) {
        output.1 = norm_squared_arr(&query);
        for i in 0..query.len() {
            output.0[i] = query[i];
        }
    }

    fn distance(&self, i: usize, query: &Self::PreparedPoint) -> Self::Distance {
        let v = &self.points[i];
        let v_norm_squared = self.squared_norms[i];
        let dotp = unsafe { dot_fma_exact(v, &query.0) };
        // let dotp = v.dot(&query.0);
        (v_norm_squared + query.1 - 2.0 * dotp).into()
    }

    fn get(&'slf self, i: usize) -> &[f32] {
        &self.points[i]
    }

    fn num_dimensions(&self) -> usize {
        self.points.dimensions
    }

    fn num_points(&self) -> usize {
        self.points.num_vectors
    }
}

// FIXME: optimize with SIMD
fn norm_squared<S: ndarray::Data<Elem = f32>>(v: &ArrayBase<S, Ix1>) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>()
}
fn norm_squared_arr(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>()
}

pub fn load_distances<P: AsRef<std::path::Path>>(hdf5_path: P) -> Array2<f32> {
    let f = hdf5::File::open(hdf5_path.as_ref()).unwrap();
    f.dataset("/distances").unwrap().read_2d::<f32>().unwrap()
}

pub fn load_raw_queries<P: AsRef<std::path::Path>>(hdf5_path: P) -> Array2<f32> {
    let f = hdf5::File::open(hdf5_path.as_ref()).unwrap();
    f.dataset("/test").unwrap().read_2d::<f32>().unwrap()
}

pub fn load_raw_data<P: AsRef<std::path::Path>>(hdf5_path: P) -> Array2<f32> {
    let f = hdf5::File::open(hdf5_path.as_ref()).unwrap();
    f.dataset("/train").unwrap().read_2d::<f32>().unwrap()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_angular_padded_dataset() {
        let path = ".glove-100-angular.hdf5";
        let dataset = AngularDatasetPadded::from_hdf5(&path);
        let mut raw = load_raw_data(path);
        let dims = dataset.num_dimensions();

        assert_eq!(raw.nrows(), dataset.num_points());
        let n = raw.nrows();
        for i in 0..n {
            let mut row = raw.row_mut(i);
            row /= norm_squared(&row).sqrt();
            let row = row.as_slice().unwrap();
            assert_eq!(&dataset.get(i)[..dims], row);
        }
    }
}
