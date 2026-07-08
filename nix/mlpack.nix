{pkgs}:
pkgs.stdenv.mkDerivation rec {
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
}
