{
  pkgs,
  python,
}: let
  # HSSL: approximate single-linkage clustering using graph-based
  # indexes (SISAP 2025). The importable Python package is the pyo3
  # extension module `hnswhsslrust`, built with maturin from the Rust
  # crate in the `HNSWhsslRust/` subdirectory of the repository.
  graphidx-src = pkgs.fetchFromGitHub {
    # Cargo path dependency of `hnswhsslrust` (referenced as
    # `../GraphIndexAPI`); dropped next to the crate before building.
    owner = "eth42";
    repo = "GraphIndexAPI";
    rev = "498eb45f80ee06da6fb82581588c7ae1e1daa5ee";
    hash = "sha256-KKdXRg/gtww9hpnLnk79d4FPZuSmjnGWC6ArFrxa96Q=";
  };

  # The Rust `hdf5` 0.8 crate does not support HDF5 1.14, so pin 1.10 and
  # join its split outputs into a single prefix that hdf5-sys can consume
  # via HDF5_DIR (it expects both `include/` and `lib/` under one root).
  hdf5-for-hssl = pkgs.symlinkJoin {
    name = "hdf5-for-hssl";
    paths = [pkgs.hdf5_1_10 pkgs.hdf5_1_10.dev];
  };
in
  python.pkgs.buildPythonPackage rec {
    pname = "hnswhsslrust";
    version = "0.1.0";
    pyproject = true;

    src = pkgs.fetchFromGitHub {
      owner = "CamillaOkkels";
      repo = "HSSL";
      rev = "db7fd8471b5e323e1186e5c22ea51cad46ea437e";
      hash = "sha256-gyoOJBYOCpX4AO4/Nsg0SgJkXlgkM43rh6pI3MmvTEw=";
    };

    # `hnswhsslrust` depends on the sibling `../GraphIndexAPI` via a Cargo
    # path dependency, so materialise GraphIndexAPI next to the crate.
    # graphidx pins `openblas-src` with a `rustls` feature that the locked
    # 0.10.x has never provided; switch it to `system` so it links the
    # OpenBLAS we supply through buildInputs instead of building/downloading.
    postUnpack = ''
      cp -r ${graphidx-src} "$sourceRoot/GraphIndexAPI"
      chmod -R u+w "$sourceRoot/GraphIndexAPI"
      substituteInPlace "$sourceRoot/GraphIndexAPI/Cargo.toml" \
        --replace-fail 'features = ["rustls"]' 'features = ["system"]'
    '';

    # The crate and its Cargo.lock live in the HNSWhsslRust subdirectory.
    cargoDeps = pkgs.rustPlatform.importCargoLock {
      lockFile = "${src}/HNSWhsslRust/Cargo.lock";
    };
    cargoRoot = "HNSWhsslRust";
    buildAndTestSubdir = "HNSWhsslRust";

    nativeBuildInputs = [
      pkgs.rustPlatform.cargoSetupHook
      pkgs.rustPlatform.maturinBuildHook
      pkgs.rustc
      pkgs.cargo
      pkgs.pkg-config
    ];
    buildInputs = [
      pkgs.openblas
      hdf5-for-hssl
      # openblas-src's downloader pulls in openssl-sys; provide OpenSSL so
      # it compiles (the download path itself is unused with `system`).
      pkgs.openssl
    ];

    env = {
      # graphidx's build.rs unconditionally `.unwrap()`s RUSTUP_TOOLCHAIN;
      # supply a stable-looking value so it doesn't panic outside rustup.
      RUSTUP_TOOLCHAIN = "stable";
      # hdf5-sys locates the C library through this prefix.
      HDF5_DIR = "${hdf5-for-hssl}";
    };

    doCheck = false;
    pythonImportsCheck = ["hnswhsslrust"];
  }
