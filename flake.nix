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
              panna-python
              fast-hdbscan
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
