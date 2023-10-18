use ndarray::prelude::*;

#[derive(PartialEq, PartialOrd)]
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

pub trait Dataset<'slf, Point> {
    /// The distance data type returned, e.g. `f64` of `u32`
    type Distance: Sized + Ord;
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
        (1.0 - v.dot(query)).into()
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
    points: Array2<f32>,
    /// The squared norms of the points
    squared_norms: Array1<f32>,
}

impl EuclideanDataset {
    pub fn new(raw: Array2<f32>) -> Self {
        let squared_norms = raw.map_axis(Axis(0), |row| norm_squared(&row));
        assert_eq!(squared_norms.len(), raw.nrows());
        Self {
            points: raw,
            squared_norms,
        }
    }

    pub fn from_hdf5<P: AsRef<std::path::Path>>(path: P) -> Self {
        let f = hdf5::File::open(path.as_ref()).unwrap();
        let raw = f.dataset("/train").unwrap().read_2d::<f32>().unwrap();
        Self::new(raw)
    }
}

impl<'slf> Dataset<'slf, ArrayView1<'slf, f32>> for EuclideanDataset {
    type Distance = DistanceF32;

    /// A prepared point is the point itself along with its norm
    type PreparedPoint = (Array1<f32>, f32);

    fn default_prepared_query(&self) -> Self::PreparedPoint {
        (Array1::zeros(self.num_dimensions()), 0.0)
    }

    fn prepare(&self, query: &ArrayView1<f32>, output: &mut Self::PreparedPoint) {
        assert_eq!(query.shape(), output.0.shape());
        output.1 = norm_squared(&query);
        for i in 0..query.len() {
            output.0[i] = query[i];
        }
    }

    fn distance(&self, i: usize, query: &Self::PreparedPoint) -> Self::Distance {
        let v = self.points.row(i);
        let v_norm_squared = self.squared_norms[i];
        (v_norm_squared + query.1 - 2.0 * v.dot(&query.0)).into()
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

// FIXME: optimize with SIMD
fn norm_squared<S: ndarray::Data<Elem = f32>>(v: &ArrayBase<S, Ix1>) -> f32 {
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
