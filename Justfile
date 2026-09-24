minibench:
    pip install .
    python scripts/minibench.py

install-python-extension:
    pip install --no-build-isolation -ve .

example: build-example
    build/fashion

debug target:
    just build {{target}}
    rr record build/{{target}}

run target:
    just build {{target}}
    build/{{target}}

profile target:
    just build {{target}}
    samply record build/{{target}}

open-debugger:
    gdbgui --gdb-cmd 'rr replay --'

build target:
    cmake --build build --config RelWithDebugInfo -j --target {{target}}

profile-example: build-example
    samply record build/fashion

build-example:
    cmake --build build -j --target fashion

test: build-tests
    build/tests "angular distance trie index"

build-tests:
    cmake --config RelWithDebugInfo --build build -j --target tests

clean:
    cd build && make clean

setup-cmake:
    test -d build || mkdir build
    cd build && cmake -DCMAKE_BUILD_TYPE=Debug ..

generate-compile-commands:
    just clean
    bear -- cmake --build build -j

scan-build:
    just clean
    scan-build just build

# Build the apptainer container with the python package
container:
    nix build .#container

# Build the apptainer container and copy it to the remote target,
# that should be a valid rsync destination (e.g. ceccarello@login.dei.unipd.it:panna.sif)
deploy-container remote: container
    rsync --progress $(readlink result) {{remote}}

sync-results remote:
    rsync --progress {{remote}}/results/emst.json /tmp/new.json
    nix run .#python scripts/emst.py merge /tmp/new.json
