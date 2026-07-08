# TODO: reintroduce the dbg macro somehow
{
  description = "PANNA: Playground for Approximate Nearest Neighbor Algorithms";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.hl = {
    url = "github:pamburus/hl";
    inputs.nixpkgs.follows = "nixpkgs";
  };
  inputs.flake-utils = {
    url = "github:numtide/flake-utils";
  };
  inputs.sigmod-hdbscan = {
    url = "github:FrancescoMonaco/hdbscan";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    self,
    nixpkgs,
    hl,
    flake-utils,
    sigmod-hdbscan,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {inherit system;};
        hl-bin = hl.packages.${system}.default;
        python = pkgs.python312.override {
          # make the override recursive so `python.pkgs` / `python.withPackages`
          # use the patched package set below.
          self = python;
          packageOverrides = pyfinal: pyprev: {
            # Patched pynndescent that counts the number of metric distance
            # evaluations performed during NNDescent graph construction and
            # querying (this is the subroutine fast_hdbscan uses for
            # non-euclidean / high-dimensional metrics). The patch exposes
            # `pynndescent.distance_count()` and `pynndescent.reset_distance_count()`.
            pynndescent = pyprev.pynndescent.overridePythonAttrs (old: {
              patches = (old.patches or []) ++ [./pynndescent-count-distances.patch];
            });
          };
        };

        libraryPath = with pkgs;
          lib.makeLibraryPath [
            # add other library packages here if needed
            stdenv.cc.cc
            stdenv.cc.libc
          ];

        fast-hdbscan = python.pkgs.buildPythonPackage rec {
          pname = "fast_hdbscan";
          version = "0.3.2";
          src = python.pkgs.fetchPypi {
            inherit pname version;
            hash = "sha256-JI4JIC7aBNpLhN2BnePMcURpkSIFu0R61McTbbWA8gU=";
          };
          # TODO: remove when we upgrade the flake
          # the upstream pyproject.toml uses the PEP 639 SPDX string form
          # (license = "BSD-2-Clause"), which the packaged setuptools rejects.
          # Rewrite it to the legacy table form so the build succeeds.
          postPatch = ''
            substituteInPlace pyproject.toml \
              --replace-fail 'license = "BSD-2-Clause"' 'license = {text = "BSD-2-Clause"}'
          '';
          # Local patch: add an `exact` boolean parameter (default False) to
          # `compute_minimum_spanning_tree`. When True it forces the exact
          # KD-tree MST for high-dimensional euclidean data instead of the
          # newer, approximate pynndescent (NNDescent) path.
          patches = [./fast-hdbscan-exact-mst.patch];
          # do not run tests
          doCheck = false;
          # specific to buildPythonPackage, see its reference
          pyproject = true;
          dependencies = with python.pkgs; [
            numba
            numpy
            pynndescent
            scikit-learn
          ];
          build-system = with python.pkgs; [
            setuptools
            wheel
          ];
        };

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

        hssl = python.pkgs.buildPythonPackage rec {
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
        };

        mlpack = pkgs.stdenv.mkDerivation rec {
          name = "mlpack";
          version = "4.6.2";
          src = pkgs.fetchurl {
            url = "https://www.mlpack.org/files/mlpack-${version}.tar.gz";
            hash = "sha256-L+dy2jg6k1ZFztB6B7UZQsoXjTgSnfO/aFiQvDwXUs8=";
          };
          installPhase = ''
            mkdir -p $out/include
            cp -r src/* $out/include
          '';
        };

        ensmallen = pkgs.stdenv.mkDerivation rec {
          name = "ensmallen";
          version = "3.10.0";
          src = pkgs.fetchurl {
            url = "https://ensmallen.org/files/ensmallen-${version}.tar.gz";
            hash = "sha256-JI4gNoVveqj6s0ygL6Onmyya8g9TsdJuPek50VDcuzo=";
          };
          installPhase = ''
            mkdir -p $out/include
            cp -r include/* $out/include
          '';
        };

        tree-similarity = pkgs.stdenv.mkDerivation rec {
          pname = "tree-similarity";
          version = "0.1.1";

          src = pkgs.fetchFromGitHub {
            owner = "DatabaseGroup";
            repo = "tree-similarity";
            rev = "0.1.1";
            hash = "sha256-bICwYyxXbZnMTfvkJvlrvm3NN4L+aRSFIl+kII5vSro=";
          };

          nativeBuildInputs = [
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
          ];

          buildInputs = [
            pkgs.llvmPackages.libcxx
          ];

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=Release"
            "-GNinja"
          ];
        };

        panna-python = python.pkgs.buildPythonPackage {
          pname = "panna";
          version = "0.0.1";
          pyproject = true;
          src = ./.;
          GIT_COMMIT_HASH = self.rev or "dirty";

          # as stated here, one should disable the cmake setup:
          # https://discourse.nixos.org/t/building-python-package-with-scikit-build-core-and-cmake-dependencies-die-python/69665/2
          dontUseCmakeConfigure = true;
          dontUseCmakeBuild = true;
          dontUseCmakeInstall = true;

          build-system = with python.pkgs; [
            scikit-build-core
            nanobind
          ];
          dependencies = with python.pkgs; [
            numpy
            h5py
          ];
          buildInputs = with pkgs; [
            python.pkgs.build
            catch2_3
            cereal
            hdf5
            highfive
          ];
          nativeBuildInputs = with pkgs; [
            cmake
            git
            ninja
            nanobench
          ];
        };

        # the Python interpreter with all the packages we need
        python-interpreter =
          python.withPackages
          (ppkgs:
            with ppkgs; [
              numpy
              pandas
              polars
              pyarrow
              h5py
              tqdm
              requests
              panna-python
              fast-hdbscan
              hssl
              icecream
              sigmod-hdbscan.packages.${system}.default
              scikit-learn
              scipy
              matplotlib
              certifi
              filelock
            ]);

        container = pkgs.singularity-tools.buildImage {
          name = "panna";
          runScript = "#!${pkgs.stdenv.shell}\npython $@";
          contents = [
            python-interpreter
            pkgs.cacert
            pkgs.coreutils-full
          ];
          diskSize = 1024 * 10; # necessary to fit the packages, otherwise the build fails
        };
      in {
        packages.default = panna-python;
        packages.container = container;
        packages.python = python-interpreter;
        packages.hssl = hssl;

        devShells.default = (pkgs.mkShell.override {stdenv = pkgs.clangStdenv;}) {
          venvDir = ".venv";

          packages = with pkgs; [
            gcc
            lldb
            clang-tools
            python.pkgs.venvShellHook
            (
              python.withPackages
              (ps:
                with ps; [
                  build
                  marimo
                  numpy
                  pandas
                  polars
                  pyarrow
                  matplotlib
                  seaborn
                  filelock
                  umap-learn
                  h5py
                  nanobind
                  icecream
                  great-tables
                  scikit-build-core
                  certifi
                  sigmod-hdbscan.packages.${system}.default
                ])
            )
            hdf5
            sqlite-interactive
            cmake
            just
            bear # To generate compile_commands.json files
            llvmPackages.openmp
            llvmPackages.libcxx
            rr
            gdbgui
            valgrind
            highfive
            samply
            boost
            cereal
            catch2_3
            fast-hdbscan
            hssl
            ensmallen
            mlpack
            armadillo
            hl-bin
            tree-similarity
            nanobench
          ];

          shellHook = ''
            #export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${libraryPath}"
          '';
          NIX_ENFORCE_NO_NATIVE = false;
        };
      }
    );
}
