{
  python,
  patch,
}:
python.pkgs.buildPythonPackage rec {
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
  patches = [patch];
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
}
