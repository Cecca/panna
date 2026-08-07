{python}:
# DENSIRED (https://github.com/PhilJahn/DENSIRED): synthetic density-based
# clustering dataset generator, used by panna.datasets' densired-hard
# entry to (re)generate that dataset if its cached .npz is missing. Not in
# nixpkgs; plain setup.py package (no pyproject.toml), pure Python.
python.pkgs.buildPythonPackage rec {
  pname = "densired";
  version = "1.2.0";
  pyproject = true;
  src = python.pkgs.fetchPypi {
    inherit pname version;
    hash = "sha256-xj7EUzYTEfAl68KsXputPCC4dnFki09qLbzVxNjcefA=";
  };
  # The PyPI sdist ships only setup.py/setup.cfg with no pyproject.toml;
  # add a minimal one so it builds through the standard PEP 517 setuptools
  # backend (pyproject = false silently produced an empty package output
  # on a recent nixpkgs revision).
  postPatch = ''
    cat > pyproject.toml <<'EOF'
    [build-system]
    requires = ["setuptools"]
    build-backend = "setuptools.build_meta"
    EOF
  '';
  build-system = with python.pkgs; [setuptools];
  dependencies = with python.pkgs; [matplotlib numpy scipy];
  doCheck = false;
}
