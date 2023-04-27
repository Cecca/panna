#[link(name = "fht", kind = "static")]
extern {
    fn fht_float(buf: *mut f32, log_n: ::std::os::raw::c_int) -> ::std::os::raw::c_int;
    fn fht_double(buf: *mut f64, log_n: ::std::os::raw::c_int) -> ::std::os::raw::c_int;
}

pub fn fht_f32(buf: &mut [f32]) {
    assert!(buf.len().is_power_of_two());
    let log_n = buf.len().ilog2();
    assert!(log_n <= 30);
    let log_n = log_n as ::std::os::raw::c_int;
    unsafe {
        let ptr = buf.as_mut_ptr();
        fht_float(ptr, log_n);
    }
}


pub fn fht_f64(buf: &mut [f64]) {
    assert!(buf.len().is_power_of_two());
    let log_n = buf.len().ilog2();
    assert!(log_n <= 30);
    let log_n = log_n as ::std::os::raw::c_int;
    unsafe {
        let ptr = buf.as_mut_ptr();
        fht_double(ptr, log_n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let mut x = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        fht_f32(&mut x);
        assert_eq!([4.0, 2.0, 0.0, -2.0, 0.0, 2.0, 0.0, 2.0], x);

        let mut x = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        fht_f64(&mut x);
        assert_eq!([4.0, 2.0, 0.0, -2.0, 0.0, 2.0, 0.0, 2.0], x);
    }
}
