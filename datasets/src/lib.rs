use lazy_static::lazy_static;
use ndarray::prelude::*;
use std::collections::HashMap;
use std::io::BufWriter;
use std::path::PathBuf;

lazy_static! {
    pub static ref DATASETS: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        m.insert(
            "glove-25-angular",
            "http://ann-benchmarks.com/glove-25-angular.hdf5",
        );
        m.insert(
            "glove-100-angular",
            "http://ann-benchmarks.com/glove-100-angular.hdf5",
        );
        m.insert(
            "glove-200-angular",
            "http://ann-benchmarks.com/glove-200-angular.hdf5",
        );
        m
    };
}

fn norm2<S: ndarray::Data<Elem = f32>>(v: &ArrayBase<S, Ix1>) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn download_if_needed(name: &str) -> PathBuf {
    let url = DATASETS[name];
    let local = PathBuf::from(format!(".{name}.hdf5"));
    if !local.is_file() {
        let mut remote = ureq::get(url).call().unwrap().into_reader();
        let mut local_file = BufWriter::new(std::fs::File::create(&local).unwrap());
        std::io::copy(&mut remote, &mut local_file).unwrap();
    }
    local
}

pub fn load_dense_dataset(name: &str) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<usize>) {
    let local = download_if_needed(name);
    let f = hdf5::File::open(&local).unwrap();
    let mut data = f.dataset("/train").unwrap().read_2d::<f32>().unwrap();
    let mut queries = f.dataset("/test").unwrap().read_2d::<f32>().unwrap();
    let ground = f.dataset("/distances").unwrap().read_2d::<f32>().unwrap();
    let neighbors = f.dataset("/neighbors").unwrap().read_2d::<usize>().unwrap();

    let distance_metric = f
        .attr("distance")
        .unwrap()
        .read_scalar::<hdf5::types::VarLenUnicode>()
        .unwrap();
    if distance_metric.as_str() == "angular" {
        // normalize datasets with angular distance
        for mut row in data.rows_mut() {
            row /= norm2(&row);
        }
        for mut row in queries.rows_mut() {
            row /= norm2(&row);
        }
    }

    (data, queries, ground, neighbors)
}

#[test]
fn test_download() {
    load_dense_dataset("glove-25-angular");
}
