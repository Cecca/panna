"""Live view of an anytime EMST run: figure + Pause/Resume/Accept controls.

`AnytimeEMST.start_live()` (and the lower-level `live_monitor()`) give a live
picture of an anytime run: the three weights drawn in place on the shared
weight axis -- weight lower bound, total weight and confirmed weight (the
edges already proven to be in the tree) -- plus the *confirmed share* of the
total, in percent, on a right-hand axis.  The share is visible from the first
update, which the confirmed weight itself is not (it sums only the lightest
edges, so it is orders of magnitude below the total); with `epsilon = 0` the
run stops only once the whole tree is confirmed, so the confirmed weight
rises to meet the total weight and the share reaches 100 %.  There is also a
status line and **Pause** / **Resume** / **Accept & stop** buttons, which
grey out once there is nothing left to steer.  The environment is detected
automatically:

* **marimo** -- `start_live()` returns a UI bundle (refresh ticker, buttons,
  figure and status line).  The refresh callback resamples the run and pushes
  the new picture into the cell with `mo.output.replace`, so the cell is
  executed *once*: nothing blocks, no cell re-runs on every tick, the buttons
  repaint the moment they are clicked and the ticker stops by itself when the
  run converges.  Leave the call as the last expression of the cell and the
  view appears right below it.
* **Jupyter** -- the figure is published with an IPython display handle and
  refreshed from the kernel's own event loop (a daemon thread if no loop is
  running), so the kernel stays responsive and the ipywidgets buttons work
  while the run proceeds.  Sampling and painting are separate here too: a
  poll that finds the run unchanged costs nothing.
* **plain Python** -- the loop blocks until the run converges (or `stop()` is
  called) and the final figure is returned for saving.

The sampling/plotting state lives in `LiveMonitor`, which is
environment-agnostic: `sample()` polls the run once and records a point *only
if the run actually moved*, while `render()` is what puts the picture on
screen.  Keeping the two apart is what makes the view cheap: the C++ collector
publishes a snapshot at most every 100 ms, so polling faster than that yields
duplicate snapshots, and re-rendering an unchanged figure is pure overhead.
The view therefore skips a tick whose numbers did not change, and each
environment renders on its own schedule (marimo: the refresh ticker; Jupyter:
an asyncio task on the main loop, falling back to a thread when no loop is
running; plain Python: the blocking loop).

Two more details keep a long run honest and cheap: the full sample history is
kept (bounded by `max_history`) and *decimated* per pixel column for drawing
rather than truncated, so the beginning of the run -- where an anytime
algorithm makes most of its progress -- never scrolls off the figure; and
`close()` releases the matplotlib figure, which pyplot otherwise keeps alive
for the whole session.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections import deque

import numpy as np

__all__ = ["LiveMonitor", "live_monitor", "start_view"]

#: Default refresh period
DEFAULT_INTERVAL = 0.25
#: Points actually drawn per curve.  Above the figure's pixel width extra
#: points cost nothing but are also invisible.
DEFAULT_MAX_POINTS = 5000
#: Samples retained in memory.  The history is decimated for drawing.
DEFAULT_MAX_HISTORY = 20000
DEFAULT_FIGSIZE = (8.0, 4.5)

#: What the x axis measures.  `elapsed` is wall-clock seconds; `repetitions`
#: and `prefix` are the run's own units of work, which are reproducible where
#: elapsed time is not (a run spans all cores, and the monitor itself competes
#: for the GIL).
X_METRICS = ("elapsed", "repetitions")

#: Axis label per metric.
X_LABELS = {
    "elapsed": "elapsed (s)",
    "repetitions": "completed repetitions",
}

#: (progress key, curve label, axis) for the curves, in draw order.
#: The three weights share the left axis, so that the confirmed weight can be
#: seen meeting the total weight once the run converges.  `certified_share` is
#: derived, not a `progress()` key: it is the confirmed weight as a percentage
#: of the total, which stays visible on the right axis throughout the run.
CURVES = (
    ("weight_lower_bound", "weight lower bound", "left"),
    ("total_weight", "total weight", "left"),
    ("confirmed_weight", "confirmed weight", "left"),
    ("certified_share", "confirmed / total", "right"),
)

#: Curve colours; explicit because three curves share the left axis.
CURVE_COLORS = {
    "weight_lower_bound": "tab:blue",
    "total_weight": "tab:orange",
    "confirmed_weight": "tab:green",
    "certified_share": "tab:green",
}

#: Intervals offered by the marimo refresh dropdown.
MARIMO_INTERVALS = ("0.1s", "0.25s", "0.5s", "1s", "5s")

_UNIT_SECONDS = {
    "": 1.0,
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1e-3,
    "m": 60.0,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
}


def interval_label(interval):
    """Human label for `interval`: `0.1` -> `"0.1s"`, `"500ms"` unchanged."""
    if isinstance(interval, str):
        return interval.strip() or f"{DEFAULT_INTERVAL:g}s"
    return f"{float(interval):g}s"


def interval_seconds(interval):
    """Coerce `0.1`, `"100ms"`, `"0.1s"`, `"1m 30s"` to seconds (float)."""
    if isinstance(interval, (int, float)):
        return max(float(interval), 1e-3)
    total = 0.0
    for number, unit in re.findall(
        r"(\d+(?:\.\d+)?)\s*([a-z]*)", str(interval).lower()
    ):
        total += float(number) * _UNIT_SECONDS.get(unit, 1.0)
    return max(total, 1e-3) if total > 0 else DEFAULT_INTERVAL


def _finite(value):
    """`value` as a float, or NaN when it is missing/infinite.

    Until the collector publishes its first real snapshot the sink reports
    `total_weight = +inf`; a non-finite point would blank the whole axis, so
    it becomes a gap (matplotlib skips NaN).
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _curve_value(progress, key):
    """The value to plot for `key`, out of a `progress()` payload.

    `certified_share` is derived rather than published: the confirmed weight
    as a percentage of the total weight, i.e. how much of the tree found so
    far is already proven.  It reaches 100 % exactly when the whole tree is
    confirmed, which is what the `epsilon = 0` stopping condition waits for.
    """
    if key != "certified_share":
        return progress.get(key)
    total = _finite(progress.get("total_weight"))
    confirmed = _finite(progress.get("confirmed_weight"))
    if not math.isfinite(total) or total <= 0 or not math.isfinite(confirmed):
        return float("nan")
    return min(100.0 * confirmed / total, 100.0)


def _curve_fingerprint(progress, points):
    """Signature of a poll: unchanged means the run did not move.

    Deliberately excludes wall-clock time -- otherwise every poll would look
    new.  What is left is exactly what the figure can show: the plotted
    values, whether the run is over, and the run's own counters of work.  The
    C++ side publishes at most every 100 ms, so polling faster than that
    repeats this signature and the tick can be skipped entirely.
    """
    return (
        bool(progress.get("done", False)),
        int(progress.get("completed_repetitions", 0) or 0),
        int(progress.get("prefix", 0) or 0),
    ) + points


def _forward_fill(values):
    """`values` with every non-finite entry replaced by the last finite one."""
    values = np.asarray(values, dtype=float)
    if np.isfinite(values).all():
        return values
    index = np.where(np.isfinite(values), np.arange(values.size), 0)
    np.maximum.accumulate(index, out=index)
    return values[index]


def _decimate(xs, ys, max_points, how="minmax"):
    """Reduce `(xs, ys)` to at most ~`max_points` points.

    Anytime curves are step-like -- flat between decides, jumping at each one
    -- so keeping every Nth point would drop the jumps.  Two strategies, both
    costing time linear in the history and nothing per point drawn:

    * ``how="minmax"`` (default) gives each bucket its min and its max, which
      preserves the envelope exactly.  Use it for the raw curves, where a
      spike is information.
    * ``how="last"`` gives each bucket its last value plus both endpoints.
      Use it for a monotone series such as best-so-far, where min/max would
      draw a zigzag the data does not have.
    """
    n = len(ys)
    if n <= max_points:
        return xs, ys
    x = np.asarray(xs, dtype=float)
    y = _forward_fill(ys)
    buckets = max(int(max_points) // 2, 1)
    edges = np.linspace(0, y.size, buckets + 1).astype(int)
    if how == "last":
        index = np.unique(np.concatenate(([0], edges[1:] - 1, [y.size - 1])))
        return x[index], y[index]
    starts = edges[:-1]
    out_x = np.repeat(x[starts], 2)
    out_y = np.empty(starts.size * 2)
    out_y[0::2] = np.minimum.reduceat(y, starts)
    out_y[1::2] = np.maximum.reduceat(y, starts)
    # the true last sample, so the curve ends where the run actually is
    out_x = np.append(out_x, x[-1])
    out_y = np.append(out_y, y[-1])
    return out_x, out_y


def _apply_limits(ax, low, high):
    """Set `ax`'s y limits from the observed range, with a margin.

    Plain autoscaling puts a curve that sits at 0 exactly on the axis spine,
    where it disappears; the margin keeps it visible.
    """
    if not (math.isfinite(low) and math.isfinite(high)):
        ax.set_ylim(0.0, 1.0)
        return
    pad = 0.05 * (high - low) if high > low else max(abs(high) * 0.05, 0.5)
    ax.set_ylim(low - pad, high + pad)


class LiveMonitor:
    """Live figure of one anytime run, with in-place sampling.

    `sample()` is the only mutator: it polls `driver.progress()` and records a
    point, but only when the run actually moved.  The C++ collector publishes
    a snapshot at most every 100 ms, so a faster poll mostly repeats the
    previous one, and re-rendering an unchanged figure costs ~30 ms (draw plus
    PNG encode) for no new information.  Recording is therefore separate from
    rendering: `sample()` updates the curves' data and `render()` puts the
    figure on screen, and each environment calls the second on its own
    schedule.  Sampling stops either when the run converges or when `stop()` is
    called; the run itself is never touched by the monitor (use
    `driver.pause()`, `driver.resume()` or `driver.accept()` for that).

    Parameters
    ----------
    driver : AnytimeEMST
        The (started) run to watch.
    interval : float | str
        Sampling interval, in seconds (`0.25`) or as a label (`"0.25s"`); the
        label is what the marimo refresh dropdown shows.
    max_points : int
        Points *drawn* per curve.  The history is decimated (min/max per
        bucket) down to this, so the whole run stays on screen.
    max_history : int
        Samples kept in memory; bounds memory only, since the figure shows a
        decimated view of everything retained.
    figsize : tuple
        Matplotlib figure size.
    """

    def __init__(
        self,
        driver,
        interval=DEFAULT_INTERVAL,
        max_points=DEFAULT_MAX_POINTS,
        figsize=DEFAULT_FIGSIZE,
        max_history=DEFAULT_MAX_HISTORY,
    ):
        import matplotlib.pyplot as plt

        self.driver = driver
        self.interval = interval_seconds(interval)
        self.interval_label = interval_label(interval)
        self.max_points = max(int(max_points), 2)
        self.max_history = max(int(max_history), 2)

        # deques, not lists: appending to a bounded deque evicts in O(1),
        # where dropping the head of a list rebuilds it on every tick.
        keep = self.max_history
        self._xs = {metric: deque(maxlen=keep) for metric in X_METRICS}
        self.lower = deque(maxlen=keep)
        self.total = deque(maxlen=keep)
        self.confirmed = deque(maxlen=keep)
        self.certified = deque(maxlen=keep)
        self.best = deque(maxlen=keep)
        self.x_metric = "elapsed"
        self.last = {}
        self.done = False
        self.stopped = False
        self.closed = False
        self.samples = 0
        self.polls = 0
        self._fingerprint = None
        self._events = []
        self._dirty = True  # nothing painted yet
        self._t0 = time.time()
        self._ymin = math.inf
        self._ymax = -math.inf
        self._lock = threading.RLock()
        self._thread = None
        self._task = None
        self._error = None

        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax2 = self.ax.twinx()
        # the band goes down first so the curves stay on top of it
        self._band = self.ax.fill_between(
            [], [], [], color="tab:orange", alpha=0.12, linewidth=0, zorder=0
        )
        self._lines = tuple(
            (self.ax2 if axis == "right" else self.ax).plot(
                [],
                [],
                label=label,
                color=CURVE_COLORS[key],
                linestyle="--" if axis == "right" else "-",
                zorder=2,
            )[0]
            for key, label, axis in CURVES
        )
        # The published total weight is the *current* running tree, so it can
        # rise as well as fall; the anytime guarantee is on the best solution
        # found so far, which is the running minimum drawn here.  The raw
        # value stays behind it, faint, for context.
        self._lines[1].set_alpha(0.3)
        self._best_line = self.ax.plot(
            [],
            [],
            drawstyle="steps-post",
            color="tab:orange",
            linewidth=1.8,
            label="total weight (best so far)",
            zorder=3,
        )[0]
        self.ax.set_xlabel(X_LABELS[self.x_metric])
        self.ax.set_ylabel("weight")
        self.ax2.set_ylabel(
            "confirmed / total (%)", color=CURVE_COLORS["certified_share"]
        )
        self.ax2.tick_params(
            axis="y", labelcolor=CURVE_COLORS["certified_share"]
        )
        self.ax.set_title(self._title())
        self.ax.grid(alpha=0.25)
        handles = list(self._lines) + [self._best_line]
        self.ax.legend(
            handles, [handle.get_label() for handle in handles], loc="best"
        )
        self._rescale()

    # ------------------------------------------------------------------ state
    @property
    def x(self):
        """Recorded x values, for the metric currently on the axis."""
        return self._xs[self.x_metric]

    def set_x_metric(self, metric):
        """Show progress as `elapsed` seconds, `repetitions` or `prefix`.

        The two counters are the run's own units of work, so the picture does
        not shift when the machine is busier or the monitor takes a little GIL
        time.  All three series are recorded, so switching is instant and
        lossless.
        """
        if metric not in X_METRICS:
            raise ValueError(
                f"unknown x metric {metric!r}, expected one of {X_METRICS}"
            )
        with self._lock:
            self.x_metric = metric
            self.ax.set_xlabel(X_LABELS[metric])
            self._dirty = True
            self._redraw()
        self.render()

    @property
    def alive(self):
        """True until `stop()` is called (the view may still be sampling)."""
        return not self.stopped

    @property
    def interactive(self):
        """True while Pause/Resume/Accept still have something to do.

        A run that converged (or whose view was stopped) has nothing left to
        pause, so the notebook controls are disabled at that point instead of
        silently doing nothing.
        """
        return self.alive and not self.done

    def stop(self):
        """Stop sampling (and, in a notebook, the refresh loop)."""
        self.stopped = True
        task, self._task = self._task, None
        if task is not None:
            try:
                task.cancel()
            except Exception:
                pass

    def close(self):
        """Release the matplotlib figure.

        pyplot keeps every figure it ever created alive for the whole session,
        so without this a notebook cell that re-runs leaks one figure, and its
        history, per run.  The figure object itself stays usable (it can still
        be written with `fig.savefig(...)`); it is only deregistered.
        """
        with self._lock:
            if self.closed:
                return
            self.closed = True
        try:
            import matplotlib.pyplot as plt

            plt.close(self.fig)
        except Exception:
            pass

    # ---------------------------------------------------------------- sampling
    def sample(self):
        """Poll the run once; record a point only if the run has moved.

        Returns the `progress()` payload either way.  A poll that repeats the
        previous snapshot -- the common case when polling faster than the
        collector's 100 ms publish throttle, or for as long as the run is
        paused -- leaves the history and the artists untouched, so it costs
        one dict copy and nothing more.
        """
        with self._lock:
            progress = dict(self.driver.progress())
            self.polls += 1
            self.last = progress
            self.done = bool(progress.get("done", False))
            points = tuple(
                _finite(_curve_value(progress, key))
                for key, _label, _axis in CURVES
            )
            fingerprint = _curve_fingerprint(progress, points)
            if fingerprint == self._fingerprint:
                return progress
            self._fingerprint = fingerprint

            self._xs["elapsed"].append(time.time() - self._t0)
            self._xs["repetitions"].append(
                float(progress.get("completed_repetitions", 0) or 0)
            )
            self._xs["prefix"].append(
                float(progress.get("prefix", 0) or 0)
            )
            self.lower.append(points[0])
            self.total.append(points[1])
            self.confirmed.append(points[2])
            self.certified.append(points[3])
            self._append_best(points[1])
            for (_key, _label, axis), value in zip(CURVES, points):
                # the share axis is pinned to 0-100 %: only the weight curves
                # drive the autoscaler
                if axis == "right" or not math.isfinite(value):
                    continue
                self._ymin = min(self._ymin, value)
                self._ymax = max(self._ymax, value)
            self.samples += 1
            self._dirty = True
            self._redraw()
            return progress

    def _append_best(self, total):
        """Extend the best-so-far (running minimum) total weight."""
        if not math.isfinite(total):
            self.best.append(self.best[-1] if self.best else float("nan"))
            return
        previous = self.best[-1] if self.best else math.inf
        self.best.append(min(previous, total))

    def mark_event(self, label):
        """Mark where the user intervened, e.g. `"paused"` or `"accepted"`.

        Steering the run is the point of the view, so the picture records it:
        a vertical line at the current x -- which, for an accept, is also
        where the kept tree was taken.
        """
        with self._lock:
            if self.closed:
                return
            # a click before the first sample has no point to anchor to: the
            # start of the axis is the honest place for it
            x = float(self.x[-1]) if len(self.x) else 0.0
            self.ax.axvline(
                x, color="0.55", linewidth=0.9, linestyle=":", zorder=1
            )
            self._events.append((x, label))
            self._dirty = True
        self.render()

    # ----------------------------------------------------------------- drawing
    def invalidate(self):
        """Force the next `render()` to paint, even if nothing was sampled.

        Needed when something other than a sample changed what the figure
        should show -- the x axis, a marker, or the run's paused state, which
        no `progress()` payload carries.
        """
        self._dirty = True

    def render(self):
        """Draw a frame, but only if there is something new to draw.

        Returns whether a frame was painted.  This is where the cost of the
        view lives (~30 ms once a notebook re-encodes the figure), so it is
        spent on new information only: a poll that found the run unchanged
        leaves the picture already on screen, and a caller that publishes the
        figure can use the return value to skip that too.
        """
        if self.closed or not self._dirty:
            return False
        self._dirty = False
        self.fig.canvas.draw_idle()
        return True

    def _redraw(self):
        """Push the recorded history into the artists (no rendering)."""
        if not len(self.x):
            return
        xs = self.x
        for line, values in zip(
            self._lines,
            (self.lower, self.total, self.confirmed, self.certified),
        ):
            line.set_data(*_decimate(xs, values, self.max_points))
        self._best_line.set_data(
            *_decimate(xs, self.best, self.max_points, how="last")
        )
        # the band spans lower bound -> total: it pinching shut is the anytime
        # guarantee closing in
        low_x, low_y = _decimate(xs, self.lower, self.max_points)
        _, high_y = _decimate(xs, self.total, self.max_points)
        self._band.set_data(low_x, low_y, high_y)
        self._rescale()

    def _rescale(self):
        """Set the limits explicitly, from the sampled range, with a margin.
        """
        xs = self.x
        if len(xs):
            start, last = xs[0], xs[-1]
            pad = max((last - start) * 0.02, 1e-3)
            self.ax.set_xlim(start - 0.5 * pad, last + pad)
        else:
            self.ax.set_xlim(0.0, 1.0)
        # the twin axis shares x with `ax`
        _apply_limits(self.ax, self._ymin, self._ymax)
        self.ax2.set_ylim(0.0, 100.0)

    def _title(self):
        """Figure title, annotated with the run's epsilon when available."""
        epsilon = getattr(self.driver, "epsilon", None)
        if epsilon is None:
            return "WOK-STIR anytime run"
        return f"WOK-STIR anytime run — ε = {epsilon:g}"

    # ------------------------------------------------------------------ status
    def status_markdown(self):
        """One-line markdown log of the run (shown next to the figure)."""
        if self._error is not None:
            return f"**monitor stopped**: `{self._error}`"
        progress = self.last
        if not progress:
            return "_waiting for the first sample…_"
        if progress.get("done"):
            state = "**Done**"
            hint = "the run is over — Pause/Resume have nothing left to do"
        elif self.driver.is_paused:
            state = "_**Paused**_"
            hint = "Resume to continue, or Accept & stop to keep the tree found so far"
        elif self.stopped:
            state = "_view stopped_"
            hint = ""
        else:
            state = "**Running**"
            hint = ""
        total = _finite(progress.get("total_weight"))
        lower = _finite(progress.get("weight_lower_bound"))
        confirmed = _finite(progress.get("confirmed_weight"))
        edges = progress.get("confirmed_edges", 0)
        to_confirm = progress.get("edges_to_confirm", 0)
        bits = [
            state,
            f"`{progress.get('elapsed_ms', 0)}` ms",
            f"reps `{progress.get('completed_repetitions', 0)}`",
            f"confirmed edges `{edges}`/`{edges + to_confirm}`",
        ]
        # The confirmed weight is 0 until the first deciding prefix and stays
        # small afterwards, so it is worth printing as a number.
        if math.isfinite(confirmed):
            bits.append(f"confirmed weight `{confirmed:.3f}`")
        if math.isfinite(total):
            bits.append(f"total weight `{total:.3f}`")
            if self.best and math.isfinite(self.best[-1]):
                bits.append(f"best so far `{self.best[-1]:.3f}`")
        if math.isfinite(lower) and lower > 0:
            bits.append(f"lower bound `{lower:.3f}`")
            if math.isfinite(total):
                bits.append(f"gap `{total / lower - 1.0:.4f}`")
        line = " · ".join(bits)
        return f"{line}  \n_{hint}_" if hint else line

    def __repr__(self):
        state = "done" if self.done else ("stopped" if self.stopped else "live")
        skipped = self.polls - self.samples
        return (
            f"<LiveMonitor {state}, {self.samples} samples from {self.polls} "
            f"polls every {self.interval_label} ({skipped} unchanged), "
            f"x={self.x_metric}; .fig for the figure, .stop() to stop the "
            f"view, .close() to release it, .driver for the run>"
        )


# --------------------------------------------------------------------------- #
# marimo
# --------------------------------------------------------------------------- #
#: One view per marimo cell; a cell that re-runs supersedes its previous view.
_MARIMO_VIEWS = {}


def _marimo_module():
    """The `marimo` module when we run inside marimo, else None."""
    try:
        from marimo._runtime.context import get_context

        get_context()
    except Exception:  # marimo missing, or not inside a marimo runtime
        return None
    try:
        import marimo
    except Exception:
        return None
    return marimo


def _marimo_cell_key():
    """Id of the cell currently executing, used to supersede old views."""
    try:
        from marimo._runtime.context import get_context

        return str(get_context().execution_context.cell_id)
    except Exception:
        return None


class _MarimoView:
    """marimo UI for one monitor: refresh ticker, buttons, figure, status.

    The cell creating the view runs once.  Every tick calls `_on_tick`, which
    resamples the run and pushes the new picture with `mo.output.replace`:
    the figure is re-rendered in place, the cell body never runs again and
    nothing blocks the kernel.  Clicking a button calls `control()`, which
    steers the run and repaints at once -- no waiting for the next tick.  Once
    the run is done (or the view is stopped) the ticker is dropped from the
    output and the buttons switch to their disabled set, so the numbers freeze
    on the final state and nothing pretends to still be steerable.

    The widgets and the figure are created once and reused across ticks (only
    the status text and the figure's data change), so a tick allocates one
    container instead of a whole output subtree.
    """

    def __init__(self, monitor, mo):
        self.monitor = monitor
        self._mo = mo

        # A click runs the handler inside this cell's context (marimo does not
        # re-run the cell), and `control()` repaints straight away, so Pause /
        # Resume / Accept show up without waiting for the next tick.
        def _pause(_value):
            self.control("pause")

        def _resume(_value):
            self.control("resume")

        def _accept(_value):
            self.control("accept")

        def buttons(disabled):
            return mo.hstack(
                [
                    mo.ui.button(
                        label="Pause",
                        tooltip="Hold the collector",
                        disabled=disabled,
                        on_click=_pause,
                    ),
                    mo.ui.button(
                        label="Resume",
                        tooltip="Let the run continue",
                        disabled=disabled,
                        on_click=_resume,
                    ),
                    mo.ui.button(
                        label="Accept & stop",
                        kind="success",
                        tooltip="Keep the tree found so far",
                        disabled=disabled,
                        on_click=_accept,
                    ),
                ],
                justify="start",
                gap=0.5,
            )

        options = [monitor.interval_label]
        options += [i for i in MARIMO_INTERVALS if i != monitor.interval_label]
        # Two sets: the live buttons, and a disabled set shown once the run is
        # over -- a button that silently does nothing looks like a broken one.
        self._buttons = buttons(disabled=not monitor.interactive)
        self._buttons_done = buttons(disabled=True)
        self._ticker = mo.ui.refresh(
            options=options,
            default_interval=monitor.interval_label,
            label="refresh",
            on_change=self._on_tick,
        )
        self._x_metric = mo.ui.dropdown(
            options=list(X_METRICS),
            value=monitor.x_metric,
            label="x axis",
            on_change=self._on_x_metric,
        )
        # The leaves (ticker, buttons, figure, status) are built once and
        # reused: only their *contents* change.  Recreating them on every tick
        # would throw away the widgets' state and make marimo reconcile a
        # whole new subtree ~4 times a second for nothing.
        self._status = mo.md("")
        self._painted = False
        self._controls_live = mo.hstack(
            [self._ticker, self._x_metric, self._buttons], justify="start", gap=1.0
        )
        self._controls_done = self._buttons_done

    # ---------------------------------------------------------------- controls
    def control(self, action):
        """One of the buttons: steer the run, then repaint immediately.

        `action` is the `AnytimeEMST` method to call: `"pause"`, `"resume"`
        or `"accept"`.  Clicking when the run is already over does nothing;
        the buttons are disabled in that case anyway.
        """
        monitor = self.monitor
        if monitor.interactive:
            try:
                getattr(monitor.driver, action)()
            except Exception as exc:  # e.g. the run finished in between
                monitor._error = f"{type(exc).__name__}: {exc}"
        if action == "accept":
            # `accept()` joins the worker, so the run has published its final
            # snapshot by now: take it before switching the display off
            self._sample()
            monitor.stop()
        # the run's paused state travels outside `progress()`, so ask for a
        # repaint even when this click produced no new sample
        monitor.mark_event(f"{action}d")
        monitor.invalidate()
        self.refresh()

    # ---------------------------------------------------------------- ticking
    def _sample(self):
        """Sample once, reporting a failure in the status line."""
        monitor = self.monitor
        if not monitor.alive:
            return
        try:
            monitor.sample()
        except Exception as exc:  # never break marimo's event loop
            monitor.stop()
            monitor._error = f"{type(exc).__name__}: {exc}"

    def refresh(self):
        """Sample once and repaint; also called by the buttons on click.

        The whole output is only rebuilt when a frame was actually painted:
        re-encoding the figure and handing marimo a new tree is the expensive
        part of a tick, and when the run has published nothing new the picture
        on screen is already the right one.
        """
        monitor = self.monitor
        if monitor.interactive:
            self._sample()
        try:
            if monitor.render() or not self._painted:
                self._painted = True
                self._replace(keep_ticker=monitor.interactive)
        except Exception as exc:  # never kill marimo's event loop
            monitor._error = f"{type(exc).__name__}: {exc}"

    def _on_tick(self, _value):
        self.refresh()

    def _on_x_metric(self, value):
        try:
            self.monitor.set_x_metric(str(value))
        except Exception as exc:
            self.monitor._error = f"{type(exc).__name__}: {exc}"
        self.refresh()

    def _replace(self, keep_ticker=True):
        self._mo.output.replace(self.bundle(keep_ticker=keep_ticker))

    # --------------------------------------------------------------- outputs
    def bundle(self, keep_ticker=True):
        """The cell output: [ticker + buttons], figure, status line.

        The container is rebuilt on every tick -- so `mo.output.replace` has
        something new to show -- but the widgets and the figure inside it are
        the same objects as last time, and only the status text is reassigned.
        """
        mo = self._mo
        monitor = self.monitor
        self._status.value = monitor.status_markdown()
        controls = self._controls_live if keep_ticker else self._controls_done
        return mo.vstack([controls, monitor.fig, self._status], gap=0.5)


def _marimo_attach(monitor, mo):
    """Create the marimo view for `monitor` in the cell being executed."""
    view = _MarimoView(monitor, mo)
    key = _marimo_cell_key()
    if key is not None:
        previous = _MARIMO_VIEWS.get(key)
        if previous is not None and previous is not view:
            # A cell that re-runs (new parameters) supersedes its old view:
            # stop sampling it, let its run go -- nobody is watching it
            # anymore, so there is no point in burning cores on it -- and
            # release its figure, which pyplot would otherwise keep alive.
            previous.monitor.stop()
            try:
                previous.monitor.driver.accept()
            except Exception:
                pass
            previous.monitor.close()
        _MARIMO_VIEWS[key] = view
    monitor._view = view  # keep the UI elements alive for the cell's lifetime
    return view


# --------------------------------------------------------------------------- #
# Jupyter / IPython
# --------------------------------------------------------------------------- #
def _ipython_kernel():
    """The IPython shell when we run inside a Jupyter kernel, else None."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
    except Exception:
        return None
    # A terminal IPython (or plain `python`/pytest) cannot update an output
    # that was already displayed, so it takes the blocking path instead.
    if shell is None or type(shell).__name__ != "ZMQInteractiveShell":
        return None
    return shell


def _ipywidgets_controls(monitor):
    """Pause/Resume/Accept buttons, or None when ipywidgets is missing."""
    try:
        from ipywidgets import Button, HBox
    except Exception:
        return None

    def _control(action):
        def handler(_button=None):
            if monitor.interactive:
                try:
                    getattr(monitor.driver, action)()
                except Exception as exc:
                    monitor._error = f"{type(exc).__name__}: {exc}"
            if action == "accept":
                # `accept()` joined the worker: take the final state before
                # stopping the refresh loop
                if monitor.alive:
                    try:
                        monitor.sample()
                    except Exception as exc:
                        monitor._error = f"{type(exc).__name__}: {exc}"
                monitor.stop()
            if monitor.alive:
                # the marker is anchored to a recorded point, so record one;
                # the paused state is not in `progress()`, hence `invalidate`
                try:
                    monitor.sample()
                    monitor.mark_event(f"{action}d")
                    monitor.invalidate()
                except Exception as exc:
                    monitor._error = f"{type(exc).__name__}: {exc}"

        return handler

    live = monitor.interactive
    buttons = [
        Button(
            description="Pause",
            tooltip="Hold the collector",
            disabled=not live,
        ),
        Button(
            description="Resume",
            tooltip="Let the run continue",
            disabled=not live,
        ),
        Button(
            description="Accept & stop",
            button_style="success",
            tooltip="Keep the tree found so far",
            disabled=not live,
        ),
    ]
    for button, action in zip(buttons, ("pause", "resume", "accept")):
        button.on_click(_control(action))
    return HBox(buttons)


def _disable_controls(controls):
    """Grey the buttons out once the run is over: nothing left to steer."""
    for button in getattr(controls, "children", ()):
        try:
            button.disabled = True
        except Exception:
            pass


def _ipython_refresh(monitor, handle):
    """One Jupyter refresh: resample, draw, republish.  False on error.

    Republishing re-encodes the figure, so it is skipped when nothing was
    painted: the browser already shows this exact picture.
    """
    try:
        monitor.sample()
        if monitor.render():
            handle.update(monitor.fig)
    except Exception:
        return False
    return True


async def _ipython_loop(monitor, handle, controls):
    """Refresh the figure from the kernel's own event loop.

    Preferred over a thread: an IPython display handle and a matplotlib canvas
    are not meant to be driven from another thread, and awaiting between
    frames leaves the kernel free, so the buttons stay clickable and nothing
    is drawn behind the user's back.
    """
    import asyncio

    try:
        while monitor.alive and not monitor.done:
            await asyncio.sleep(monitor.interval)
            if not _ipython_refresh(monitor, handle):
                return
        # either the run published its final snapshot at `done`, or a control
        # stopped the view: repaint whatever it left behind
        if monitor.alive:
            _ipython_refresh(monitor, handle)
        else:
            try:
                handle.update(monitor.fig)
            except Exception:
                pass
    finally:
        _disable_controls(controls)


def _start_ipython_view(monitor):
    """Publish the figure and keep it up to date while the run proceeds.

    Returns the display handle, or None when the environment cannot update a
    figure in place (then the caller falls back to the blocking loop).  The
    refresh runs on the kernel's event loop when there is one and on a daemon
    thread otherwise (no loop running, e.g. a script under IPython); the
    kernel stays responsive either way, so the buttons remain clickable while
    the run proceeds.
    """
    try:
        from IPython.display import display

        monitor.sample()
        monitor.render()
        handle = display(monitor.fig, display_id=True)
    except Exception:
        return None
    if handle is None or not hasattr(handle, "update"):
        return None
    controls = _ipywidgets_controls(monitor)
    if controls is not None:
        try:
            display(controls)
        except Exception:
            pass

    try:
        import asyncio

        loop = asyncio.get_running_loop()
    except Exception:
        loop = None
    if loop is not None:
        monitor._task = loop.create_task(_ipython_loop(monitor, handle, controls))
        return handle

    def _thread_loop():
        while monitor.alive and not monitor.done:
            if not _ipython_refresh(monitor, handle):
                break
            time.sleep(monitor.interval)
        if monitor.alive and monitor.done:
            # one last sample: the run published its final snapshot at `done`
            _ipython_refresh(monitor, handle)
        else:
            # stopped from a control button: repaint what it left behind
            try:
                handle.update(monitor.fig)
            except Exception:
                pass
        _disable_controls(controls)

    monitor._thread = threading.Thread(
        target=_thread_loop, name="panna-live-monitor", daemon=True
    )
    monitor._thread.start()
    return handle


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def _run_blocking(monitor):
    """Sample until the run converges (or `stop()` is called).

    The schedule is deadline-based, so the period stays `interval` instead of
    drifting by the cost of each sample and render.
    """
    deadline = time.monotonic() + monitor.interval
    while monitor.alive and not monitor.done:
        monitor.sample()
        monitor.render()
        time.sleep(max(deadline - time.monotonic(), 0.0))
        deadline += monitor.interval
    if monitor.alive and monitor.done:
        monitor.sample()
        monitor.render()


def start_view(monitor):
    """Drive `monitor` in the running environment and return its display.

    * marimo: the live UI bundle (figure + controls + status line) to be
      returned by the cell that created it.
    * Jupyter: the monitor itself; the figure is refreshed in place from the
      kernel's event loop.
    * anything else: runs the sampling loop until the run is done and returns
      the monitor (whose `.fig` holds the finished plot).
    """
    mo = _marimo_module()
    if mo is not None:
        monitor.sample()
        view = _marimo_attach(monitor, mo)
        # a run that is already over needs no ticker: show the final state
        return view.bundle(keep_ticker=not monitor.done)
    if _ipython_kernel() is not None and _start_ipython_view(monitor) is not None:
        return monitor
    _run_blocking(monitor)
    return monitor


def live_monitor(
    driver,
    interval=DEFAULT_INTERVAL,
    max_points=DEFAULT_MAX_POINTS,
    figsize=DEFAULT_FIGSIZE,
    max_history=DEFAULT_MAX_HISTORY,
):
    """Watch a started run live; returns `(fig, ax, stop)`.

    `stop()` stops the *view* (the run keeps going; use `driver.accept()` to
    keep the tree found so far).  Outside a notebook the call blocks until the
    run is done, so the figure can be saved afterwards; inside a notebook it
    returns immediately and `fig` updates live (in marimo the live view is
    appended to the cell output, so prefer `AnytimeEMST.start_live()`, which
    is one call and returns the view for the cell).
    """
    monitor = LiveMonitor(
        driver,
        interval=interval,
        max_points=max_points,
        figsize=figsize,
        max_history=max_history,
    )
    view = start_view(monitor)
    mo = _marimo_module()
    if mo is not None:
        mo.output.append(view)
    return monitor.fig, monitor.ax, monitor.stop
