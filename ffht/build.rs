fn main() {
    let mut builder = cc::Build::new();
    let build = builder
        .files(["ffht-c/fht.c"])
        .include("ffht-c")
        .flag("-O3")
        .flag("-march=native")
        .flag("-std=c99");

    build.compile("fht");
}
