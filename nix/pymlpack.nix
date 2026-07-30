{
  pkgs,
  python,
}: let
  inherit (pkgs) lib;

  version = "4.8.0";

  # The `mlpack` project publishes *no* source distribution on PyPI (the
  # bindings are generated and compiled as part of the mlpack CMake build),
  # so the only thing fetchPypi can grab is a prebuilt wheel. Wheels exist
  # for CPython 3.9-3.14 on linux-x86_64/i686 and macOS; we only map the
  # ones relevant here.
  pyTag = "cp${builtins.replaceStrings ["."] [""] python.pythonVersion}";

  wheels = {
    cp312 = {
      x86_64-linux = {
        platform = "manylinux_2_27_x86_64.manylinux_2_28_x86_64";
        hash = "sha256-+7+zhWlG90hf+Sju1Bigj4J+VcbeS5UoPJPRGkf1qWs=";
      };
      x86_64-darwin = {
        platform = "macosx_10_13_x86_64";
        hash = "sha256-xPZmhnXOgncESujJfDlfOnMJB3vpT6+e3ntFHu1ZywY=";
      };
      aarch64-darwin = {
        platform = "macosx_11_0_arm64";
        hash = "sha256-as9CoUuNZENB5HdE1kHqp9qNk4gULwXjMNuySQzQ/EE=";
      };
    };
  };

  system = pkgs.stdenv.hostPlatform.system;
  wheel =
    wheels.${pyTag}.${system}
    or (throw "nix/pymlpack.nix: no mlpack ${version} wheel recorded for ${pyTag} on ${system}");
in
  python.pkgs.buildPythonPackage {
    pname = "mlpack";
    inherit version;
    format = "wheel";

    src = python.pkgs.fetchPypi {
      pname = "mlpack";
      inherit version;
      format = "wheel";
      dist = pyTag;
      python = pyTag;
      abi = pyTag;
      inherit (wheel) platform hash;
    };

    # The manylinux wheel ships its vendored shared libraries (armadillo,
    # OpenBLAS, libgomp, libgfortran, ...) in `mlpack.libs/`; autoPatchelf
    # rewrites the RPATHs so those resolve against each other, and pulls the
    # few genuinely external ones (libstdc++, libz) out of this stdenv.
    nativeBuildInputs = lib.optionals pkgs.stdenv.hostPlatform.isLinux [
      pkgs.autoPatchelfHook
    ];
    buildInputs = lib.optionals pkgs.stdenv.hostPlatform.isLinux [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib # wanted by the vendored libgfortran
    ];

    dependencies = with python.pkgs; [
      cython
      numpy
      pandas
    ];

    doCheck = false;
    pythonImportsCheck = ["mlpack"];

    meta = {
      description = "mlpack: a fast, header-only C++ machine learning library (Python bindings)";
      homepage = "https://www.mlpack.org/";
      license = lib.licenses.bsd3;
      platforms = builtins.attrNames wheels.${pyTag} or [];
    };
  }
