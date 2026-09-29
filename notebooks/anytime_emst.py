import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import matplotlib

    matplotlib.use("Agg")  # marimo renders the figure; no GUI backend needed

    import marimo as mo
    import numpy as np

    from panna.anytime import AnytimeEMST

    # The Fashion-MNIST download lives outside the repo; keep the fallback so
    # the notebook also runs where PANNA_DATA_DIR is exported in the shell.
    DATA_DIR = os.environ.get("PANNA_DATA_DIR", "/data/matrices/ann")
    FASHION = os.path.join(DATA_DIR, "fashion-mnist-784-euclidean.hdf5")

    return AnytimeEMST, FASHION, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # wok-stir anytime EMST in action

    The whole live view is a single call:

    ```python
    AnytimeEMST(data, repetitions=1024, epsilon=0.0).start_live()
    ```

    The data is **Fashion-MNIST** (`60000 x 784`, loaded from
    `/data/matrices/ann/fashion-mnist-784-euclidean.hdf5`); the *n points*
    slider takes a prefix of it, so the knobs below still decide how long the
    run lives.

    `start_live()` starts the C++ anytime EMST in a background thread and
    returns the live UI: the axes, a status line and the controls.

    **Left axis — the weights.** *total weight* (orange, the tree found so
    far), *weight lower bound* (blue,
    `confirmed_weight + edges_to_confirm * heaviest_confirmed_edge`) and
    *confirmed weight* (green, the part of the tree already proven to be in
    the EMST).  They share one scale on purpose: with `epsilon = 0` the run
    stops only once every edge is confirmed, so the green curve climbs all the
    way to the orange one and all three curves converge.

    **Right axis — the confirmed share**, in percent (green dashes):
    `confirmed_weight / total_weight`.  It is visible from the first update,
    whereas the confirmed weight itself stays near zero for most of the run
    (it sums only the lightest edges, the ones certified so far).  It reaches
    100 % exactly when the confirmed weight equals the total weight.

    * `epsilon` is the approximation factor: the run stops as soon as
      `total_weight <= (1 + epsilon) * weight_lower_bound`.  A bigger epsilon
      returns sooner with a weaker guarantee; `epsilon = 0` waits for the
      exact tree, i.e. for the whole tree to be certified;
    * `delta` is the failure probability of that certification: a smaller
      delta needs more work to confirm the same edges;
    * the ticker refreshes 10 times per second by default (`0.1s`; pick
      another interval in the *refresh* dropdown);
    * it stops by itself when the run converges, and the buttons grey out
      with it — there is nothing left to pause at that point;
    * **Pause** freezes the collector at the next update boundary without
      losing work, **Resume** lets it continue, **Accept & stop** keeps the
      tree found so far;
    * afterwards, `weights, edges = driver.wait()` returns the final tree
      (`driver.accept()` the early one).

    A run only lives as long as the work does: the defaults below take ~10 s,
    so there is time to click.  Drop *n points* / *repetitions* and the run is
    over before the controls can be used (they grey out, they do not hang).

    The plot lives in the package (`panna.monitor.LiveMonitor`), so the very
    same call works in Jupyter and in a plain script.
    """)
    return


@app.cell
def _(mo):
    n_points = mo.ui.slider(1000, 60000, value=10000, step=1000, label="n points")
    repetitions = mo.ui.slider(2, 4096, value=1024, step=32, label="repetitions")
    epsilon = mo.ui.slider(0.0, 0.5, value=0.0, step=0.01, label="epsilon")
    delta = mo.ui.slider(0.001, 0.5, value=0.1, step=0.001, label="delta")
    k = mo.ui.slider(0, 15, value=0, label="k (0 = Euclidean, >0 = mutual reach.)")
    mo.hstack([n_points, repetitions, epsilon, delta, k], justify="start")
    return delta, epsilon, k, n_points, repetitions


@app.cell
def _(FASHION, mo, n_points, np):
    import h5py

    with h5py.File(FASHION, "r") as hfp:
        # A prefix of the real dataset: no shuffling, so the notebook is
        # reproducible and increasing *n points* extends the same subset.
        data = np.asarray(hfp["train"][: n_points.value], dtype=np.float32)
    mo.md(f"Fashion-MNIST: `{data.shape[0]}` points in `{data.shape[1]}` dims")
    return (data,)


@app.cell
def _(AnytimeEMST, data, delta, epsilon, k, repetitions):
    driver = AnytimeEMST(
        data,
        k=int(k.value),
        repetitions=int(repetitions.value),
        epsilon=float(epsilon.value),
        delta=float(delta.value),
    )
    driver.start_live()
    return


if __name__ == "__main__":
    app.run()
