# wok-stir

Anytime Euclidean MST and mutual-reachability MST (HDBSCAN) with
provable bounds, built on LSH forests. The computation runs in a background
thread and publishes live **weight lower bound / total weight / confirmed
weight** curves; you can **pause** the run, inspect the current tree, then
**resume** or **accept-and-stop**.

## Install

```bash
pip install wok-stir            # one-shot APIs
pip install wok-stir[monitor]   # + anytime monitoring in notebooks
```

From source (needs a C++20 compiler, CMake, Python 3.12):

```bash
pip install .
```

Binaries target x86-64 with AVX2+FMA.

## Quick start

```python
import numpy as np
from panna.anytime import AnytimeEMST

data = np.random.default_rng(0).normal(size=(20000, 64)).astype(np.float32)

# k=0: Euclidean MST. k>0: mutual reachability MST with k-th neighbor cores
# (the tree HDBSCAN clusters from).
driver = AnytimeEMST(data, k=5, repetitions=512).start()

snap = driver.pause()   # freeze at an update boundary, no work lost
print(snap)             # total / confirmed / lower bound / gap / reps / prefix
driver.resume()         # ...or driver.accept() to keep the current tree

weights, edges = driver.wait()  # block until convergence
```

## Notebook monitor

```python
from panna.monitor import live_monitor

driver = AnytimeEMST(data, k=5).start()
fig, ax, stop = live_monitor(driver, interval=0.5)
# live plot of lower bound / total / confirmed + Pause/Resume/Accept buttons
weights, edges = driver.wait()
fig.savefig("bounds.png")
```

## One-shot API

```python
from panna import EMST
idx = EMST(data, repetitions=512)
weights, edges = idx.find_mst()          # Euclidean
tree, core, neigh = idx.find_mst_dbscan(5)  # mutual reachability + kNN
```

## License

AGPL-3.0. See the repository for details.
