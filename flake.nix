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

        fast-hdbscan = import ./nix/fast-hdbscan.nix {
          inherit python;
          patch = ./fast-hdbscan-exact-mst.patch;
        };

        hssl = import ./nix/hssl.nix {inherit pkgs python;};

        mlpack = import ./nix/mlpack.nix {inherit pkgs;};

        ensmallen = import ./nix/ensmallen.nix {inherit pkgs;};

        tree-similarity = import ./nix/tree-similarity.nix {inherit pkgs;};

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

          NIX_ENFORCE_NO_NATIVE = false;
        };
      }
    );
}
