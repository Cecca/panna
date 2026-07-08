{pkgs}:
pkgs.stdenv.mkDerivation rec {
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
}
