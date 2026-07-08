{pkgs}:
pkgs.stdenv.mkDerivation rec {
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
}
