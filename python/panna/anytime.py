"""Anytime driver for the EMST computation.

`AnytimeEMST` wraps the compiled `panna.EMST` object and runs the
computation in a background C++ thread. While it runs, `progress()` polls
the live bounds (weight lower bound, total weight, confirmed weight) and
`pause()` freezes the collector at the next update boundary without losing
work. `snapshot()` returns the current tree while paused (or running), and
the user then either calls `resume()` or `accept()` to keep the current
tree and stop early.

`epsilon` is the approximation factor: the run stops as soon as
`total_weight <= (1 + epsilon) * weight_lower_bound`, so a larger epsilon
returns sooner with a weaker guarantee.  With `epsilon=0` that condition
reduces to `total_weight == weight_lower_bound`, i.e. the exact tree.  `delta` (default 0.1) is the failure
probability of that per-edge certification.

For the live view there is one entry point::

    AnytimeEMST(data).start_live()

which starts the run *and* shows the figure with Pause / Resume / Accept &
stop controls, in a marimo notebook, in Jupyter or in a plain script (see
`LiveMonitor` in `panna.monitor` for the details).
"""

import numpy as np

# `monitor` imports numpy at module level and matplotlib only when a figure is
# built, so importing the defaults from it costs nothing at import time.
from .monitor import DEFAULT_MAX_HISTORY, DEFAULT_MAX_POINTS

__all__ = ["AnytimeEMST", "AnytimeResult"]


class AnytimeResult:
    """A point-in-time view of an anytime run, as returned by `snapshot()`."""

    def __init__(self, payload):
        self.tree_complete = bool(payload["tree_complete"])
        self.converged = bool(payload["converged"])
        self.done = bool(payload["done"])
        self.total_weight = float(payload["total_weight"])
        self.confirmed_weight = float(payload["confirmed_weight"])
        self.weight_lower_bound = float(payload["weight_lower_bound"])
        self.heaviest_confirmed_edge = float(payload["heaviest_confirmed_edge"])
        self.confirmed_edges = int(payload["confirmed_edges"])
        self.edges_to_confirm = int(payload["edges_to_confirm"])
        self.completed_repetitions = int(payload["completed_repetitions"])
        self.prefix = int(payload["prefix"])
        self.elapsed_ms = int(payload["elapsed_ms"])
        self.edges = np.asarray(payload["edges"])
        self.weights = np.asarray(payload["weights"])

    def gap(self):
        """Relative gap between the current total weight and its lower bound."""
        if self.weight_lower_bound <= 0:
            return float("inf")
        return self.total_weight / self.weight_lower_bound - 1.0

    def __repr__(self):
        return (
            f"AnytimeResult(total={self.total_weight:.3f}, "
            f"confirmed={self.confirmed_weight:.3f}, "
            f"lower_bound={self.weight_lower_bound:.3f}, gap={self.gap():.4f}, "
            f"edges={self.edges.shape[0]}, reps={self.completed_repetitions}, "
            f"prefix={self.prefix}, elapsed_ms={self.elapsed_ms})"
        )


class AnytimeEMST:
    """Anytime EMST computation with pause/resume and live bounds.

    Parameters mirror the compiled `panna.EMST` index; `k=0` runs the
    Euclidean MST path, `k>0` the mutual reachability (HDBSCAN) path.
    `epsilon`, `delta`, `repetitions`, `refine_iterations` and `family` are
    forwarded too.  `epsilon=0` runs until the whole tree is certified, so
    `confirmed_weight` climbs to `total_weight` (see the module docstring);
    `epsilon` and `delta` are kept on the instance for inspection.
    """

    def __init__(self, data, k=0, **kwargs):
        from ._panna_impl import EMST

        self._k = int(k)
        self._index = EMST(np.ascontiguousarray(data, dtype=np.float32), **kwargs)
        # defaults as in the compiled wrapper, so the effective values can be
        # read back from the driver (the figure title annotates epsilon)
        self.epsilon = float(kwargs.get("epsilon", 0.0))
        self.delta = float(kwargs.get("delta", 0.1))
        self._history = []
        self._accepted = None
        self._started = False
        self._monitor = None

    def start(self):
        """Start the background computation and return immediately."""
        self._history = []
        self._accepted = None
        self._index.start_anytime(self._k)
        self._started = True
        return self

    def start_live(
        self,
        interval=None,
        max_points=DEFAULT_MAX_POINTS,
        figsize=(8.0, 4.5),
        max_history=DEFAULT_MAX_HISTORY,
    ):
        """Start the run and show the live figure with Pause/Resume/Accept.

        The computation runs in the background
        thread, the figure appears immediately and the three curves (weight
        lower bound, total weight, confirmed weight) update in place.  What is
        returned depends on where the code runs:

        * **marimo** -- the live UI bundle (figure + Pause/Resume/Accept
          buttons + a refresh dropdown + a status line).  Leave the call as
          the cell's last expression and the view appears below it; the ticker
          stops by itself once the run converges.
        * **Jupyter** -- the `LiveMonitor`, with the figure refreshed from the
          kernel's event loop so the buttons stay clickable.
        * **plain Python** -- blocks until the run is done and returns the
          `LiveMonitor`, whose `.fig` holds the finished plot.

        `interval` is the refresh period, in seconds or as a label (`"0.25s"`);
        `None` uses `monitor.DEFAULT_INTERVAL` (0.25 s, which matches the
        collector's 100 ms publish throttle -- polling faster only repeats
        snapshots).  `monitor.stop()` stops the view, `monitor.close()`
        releases the figure, `monitor.driver.accept()` keeps the tree found so
        far.  Pass `epsilon=0` to the constructor to watch the confirmed weight
        climb all the way to the total weight.
        """
        from .monitor import DEFAULT_INTERVAL, LiveMonitor, start_view

        if not self._started:
            self.start()
        self._monitor = LiveMonitor(
            self,
            interval=DEFAULT_INTERVAL if interval is None else interval,
            max_points=max_points,
            figsize=figsize,
            max_history=max_history,
        )
        return start_view(self._monitor)

    def pause(self):
        """Freeze the run at the next update boundary; work is not lost."""
        self._index.pause()
        snap = self.snapshot()
        self._history.append(snap)
        return snap

    def resume(self):
        """Resume a paused run."""
        self._index.resume()
        return self

    def progress(self):
        """Lightweight poll of the live bounds (no tree copy)."""
        return dict(self._index.progress())

    def snapshot(self):
        """Full live view of the current tree and bounds."""
        return AnytimeResult(self._index.snapshot())

    def accept(self):
        """Stop early, keep the current tree, return `(weights, edges)`."""
        if self._accepted is not None:
            return self._accepted
        weights, edges = self._index.accept()
        self._accepted = (np.asarray(weights), np.asarray(edges))
        return self._accepted

    def wait(self):
        """Block until convergence, return `(weights, edges)`."""
        if self._accepted is not None:
            return self._accepted
        weights, edges = self._index.wait()
        self._accepted = (np.asarray(weights), np.asarray(edges))
        return self._accepted

    @property
    def is_running(self):
        return self._index.is_running()

    @property
    def is_paused(self):
        return self._index.is_paused()

    @property
    def is_done(self):
        return self._index.is_done()

    @property
    def history(self):
        """Snapshots taken at each `pause()` call, oldest first."""
        return list(self._history)

    @property
    def monitor(self):
        """The `LiveMonitor` created by `start_live()`, or None."""
        return self._monitor
