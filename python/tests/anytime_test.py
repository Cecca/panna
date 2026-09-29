"""Tests for the anytime driver: pause/resume/accept and live bounds."""

import time

import numpy as np
import pytest

from panna.anytime import AnytimeEMST


def _data(n=20000, d=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)).astype(np.float32)


def _wait_running(driver, timeout=120):
    deadline = time.time() + timeout
    while not driver.is_running and time.time() < deadline:
        time.sleep(0.01)
    return driver.is_running


def test_progress_keys():
    driver = AnytimeEMST(_data(), repetitions=512, family="e2lsh").start()
    try:
        assert _wait_running(driver)
        time.sleep(2.0)
        p = driver.progress()
        assert {
            "done", "tree_complete", "converged", "total_weight",
            "confirmed_weight", "weight_lower_bound", "confirmed_edges",
            "edges_to_confirm", "completed_repetitions", "prefix", "elapsed_ms",
        } <= set(p)
    finally:
        driver.wait()


def test_pause_resume_wait():
    data = _data()
    driver = AnytimeEMST(data, repetitions=512, family="e2lsh").start()
    assert _wait_running(driver)
    time.sleep(2.0)
    if not driver.is_done:
        driver.pause()
        assert driver.is_paused
        snap = driver.snapshot()
        assert snap.total_weight >= snap.weight_lower_bound >= 0
        driver.resume()
        assert not driver.is_paused
    weights, edges = driver.wait()
    assert np.asarray(edges).shape == (data.shape[0] - 1, 2)
    assert float(np.asarray(weights).sum()) > 0


def test_pause_accept_returns_spanning_tree():
    data = _data()
    driver = AnytimeEMST(data, repetitions=512, family="e2lsh").start()
    assert _wait_running(driver)
    time.sleep(2.0)
    if not driver.is_done:
        snap = driver.pause()
        assert snap.edges.shape[0] > 0
        weights, edges = driver.accept()
    else:
        weights, edges = driver.wait()
    assert np.asarray(edges).shape == (data.shape[0] - 1, 2)


def test_mutual_reachability_anytime_pause_accept():
    data = _data(n=5000, d=16)
    driver = AnytimeEMST(data, k=5, repetitions=64, family="e2lsh").start()
    assert _wait_running(driver)
    time.sleep(2.0)
    if not driver.is_done:
        snap = driver.pause()
        assert snap.total_weight >= snap.weight_lower_bound >= 0
        weights, edges = driver.accept()
    else:
        weights, edges = driver.wait()
    assert np.asarray(edges).shape == (data.shape[0] - 1, 2)
    assert float(np.asarray(weights).sum()) > 0
    # the snapshot seen at done must match the returned result exactly
    snap = driver.snapshot()
    total = float(np.asarray(weights, dtype=np.float64).sum())
    assert snap.done
    assert snap.edges.shape == (data.shape[0] - 1, 2)
    assert np.isclose(snap.total_weight, total, rtol=1e-3)


def test_live_monitor_runs_headless(tmp_path):
    from panna.monitor import CURVES, live_monitor

    data = _data(n=5000, d=16)
    driver = AnytimeEMST(data, repetitions=64, family="e2lsh").start()
    assert _wait_running(driver)
    fig, ax, stop = live_monitor(driver, interval=0.2)
    try:
        weights, edges = driver.wait()
    finally:
        stop()
    assert np.asarray(edges).shape == (data.shape[0] - 1, 2)
    # verify the curves and their axes
    assert len(CURVES) == 4
    assert [c[2] for c in CURVES] == ["left", "left", "left", "right"]
    out = tmp_path / "bounds.png"
    fig.savefig(out)
    assert out.is_file()


def test_monitor_curve_values():
    from panna.monitor import _curve_value

    progress = {
        "weight_lower_bound": 10.0,
        "total_weight": 20.0,
        "confirmed_weight": 5.0,
    }
    assert _curve_value(progress, "weight_lower_bound") == 10.0
    assert _curve_value(progress, "total_weight") == 20.0
    assert _curve_value(progress, "confirmed_weight") == 5.0
    assert np.isclose(_curve_value(progress, "certified_share"), 25.0)

    # edge cases: zero or negative total weight, NaNs
    assert np.isnan(_curve_value({"total_weight": 0.0, "confirmed_weight": 0.0}, "certified_share"))
    assert np.isnan(_curve_value({"total_weight": -5.0, "confirmed_weight": 1.0}, "certified_share"))
    assert np.isnan(_curve_value({}, "certified_share"))

    # capped at 100%
    assert _curve_value({"total_weight": 10.0, "confirmed_weight": 12.0}, "certified_share") == 100.0


def test_final_snapshot_matches_wait_result():
    """The snapshot observed at `done` must be the final state, not a stale
    mid-run one swallowed by the 100 ms publish throttle (regression test)."""
    data = _data(n=5000, d=16)
    driver = AnytimeEMST(data, repetitions=64, family="e2lsh").start()
    assert _wait_running(driver)
    weights, edges = driver.wait()
    snap = driver.snapshot()
    total = float(np.asarray(weights, dtype=np.float64).sum())
    assert snap.done
    assert snap.converged
    assert snap.tree_complete
    assert snap.edges.shape == (data.shape[0] - 1, 2)
    assert np.isclose(snap.total_weight, total, rtol=1e-3)
    # the last decide's lower bound stays a valid (non-trivial) bound
    assert 0.0 <= snap.weight_lower_bound <= snap.total_weight + 1e-3 * total


# --------------------------------------------------------------------------- #
# the live view, driven by a fake run (no C++ thread, so the view's own
# behaviour -- dedup, decimation, lifecycle -- is testable on its own)
# --------------------------------------------------------------------------- #
class _FakeDriver:
    """Stand-in for `AnytimeEMST`: replays a list of `progress()` payloads."""

    epsilon = 0.0

    def __init__(self, snapshots):
        self._snapshots = list(snapshots)
        self._index = 0
        self.is_paused = False
        self.calls = []

    def progress(self):
        index = min(self._index, len(self._snapshots) - 1)
        self._index += 1
        return dict(self._snapshots[index])

    def pause(self):
        self.calls.append("pause")
        self.is_paused = True

    def resume(self):
        self.calls.append("resume")
        self.is_paused = False

    def accept(self):
        self.calls.append("accept")


def _snapshot(total, lower=None, confirmed=0.0, reps=0, prefix=4, done=False):
    """One `progress()` payload; `lower` defaults to just under `total`."""
    if lower is None:
        lower = total * 0.99
    return {
        "done": done,
        "tree_complete": True,
        "converged": done,
        "total_weight": total,
        "confirmed_weight": confirmed,
        "weight_lower_bound": lower,
        "confirmed_edges": 0,
        "edges_to_confirm": 0,
        "completed_repetitions": reps,
        "prefix": prefix,
        "elapsed_ms": reps,
    }


def test_monitor_skips_unchanged_snapshots():
    """Polling faster than the collector publishes must not record or draw."""
    from panna.monitor import LiveMonitor

    driver = _FakeDriver([_snapshot(10.0, reps=1)])
    monitor = LiveMonitor(driver, interval=0.01)
    try:
        for _ in range(20):
            monitor.sample()
        assert monitor.polls == 20
        assert monitor.samples == 1
        assert len(monitor.x) == 1
    finally:
        monitor.close()


def test_sample_records_but_does_not_render():
    """`sample()` is cheap and side-effect free on the canvas; `render()` paints."""
    from panna.monitor import LiveMonitor

    driver = _FakeDriver([_snapshot(10.0, reps=1)])
    monitor = LiveMonitor(driver, interval=0.01)
    frames = []
    monitor.fig.canvas.draw_idle = lambda: frames.append(1)
    try:
        monitor.sample()
        assert frames == []
        monitor.render()
        assert frames == [1]
        monitor.sample()  # same snapshot: nothing new to draw
        assert frames == [1]
    finally:
        monitor.close()


def test_render_is_skipped_when_nothing_changed():
    """A poll with no new information must not cost a frame."""
    from panna.monitor import LiveMonitor

    driver = _FakeDriver([_snapshot(10.0, reps=1)])
    monitor = LiveMonitor(driver, interval=0.01)
    frames = []
    monitor.fig.canvas.draw_idle = lambda: frames.append(1)
    try:
        monitor.sample()
        assert monitor.render() is True  # first frame: something to show
        for _ in range(10):
            monitor.sample()  # the run has not moved since
            assert monitor.render() is False
        assert frames == [1]
        # ... and a non-sample change (x axis, paused state) repaints on demand
        monitor.set_x_metric("repetitions")
        assert frames == [1, 1]
        monitor.invalidate()
        assert monitor.render() is True
        assert frames == [1, 1, 1]
    finally:
        monitor.close()


def test_monitor_decimates_but_keeps_the_whole_run(tmp_path):
    """More samples than pixels: fewer points drawn, same envelope, t=0 kept."""
    from panna.monitor import LiveMonitor

    n = 500
    totals = [100.0 - i * 0.1 for i in range(n)]
    driver = _FakeDriver(
        [_snapshot(total, reps=i) for i, total in enumerate(totals)]
    )
    monitor = LiveMonitor(driver, interval=0.01, max_points=40)
    try:
        for _ in range(n):
            monitor.sample()
        assert monitor.samples == n
        xs, ys = monitor._best_line.get_data()
        assert len(xs) <= 40 + 2
        # the start of the run is still on the axis: decimation, not truncation
        assert xs[0] < 0.05
        # a monotone series keeps its direction and both endpoints ...
        assert (np.diff(ys) <= 1e-9).all()  # best-so-far never gets worse
        assert max(ys) == pytest.approx(max(totals))
        assert min(ys) == pytest.approx(min(totals))
        # ... while a raw curve keeps its envelope, spikes included
        _raw_x, raw_y = monitor._lines[1].get_data()
        assert len(_raw_x) <= 2 * 40 + 1
        assert max(raw_y) == pytest.approx(max(totals))
        assert min(raw_y) == pytest.approx(min(totals))
        out = tmp_path / "decimated.png"
        monitor.fig.savefig(out)
        assert out.is_file()
    finally:
        monitor.close()


def test_monitor_best_so_far_is_monotone():
    """The running tree may get worse; the best solution found may not."""
    from panna.monitor import LiveMonitor

    driver = _FakeDriver(
        [_snapshot(10.0, reps=1), _snapshot(12.0, reps=2), _snapshot(9.0, reps=3)]
    )
    monitor = LiveMonitor(driver, interval=0.01)
    try:
        for _ in range(3):
            monitor.sample()
        assert list(monitor.best) == [10.0, 10.0, 9.0]
    finally:
        monitor.close()


def test_monitor_x_metric_switch():
    """Progress can be read in the run's own units of work, losslessly."""
    from panna.monitor import X_LABELS, LiveMonitor

    driver = _FakeDriver(
        [_snapshot(10.0, reps=5, prefix=4), _snapshot(9.0, reps=7, prefix=5)]
    )
    monitor = LiveMonitor(driver, interval=0.01)
    try:
        monitor.sample()
        monitor.sample()
        monitor.set_x_metric("repetitions")
        assert list(monitor._lines[1].get_data()[0]) == [5.0, 7.0]
        assert monitor.ax.get_xlabel() == X_LABELS["repetitions"]
        monitor.set_x_metric("prefix")
        assert list(monitor._lines[1].get_data()[0]) == [4.0, 5.0]
        with pytest.raises(ValueError):
            monitor.set_x_metric("nope")
    finally:
        monitor.close()


def test_monitor_marks_user_events():
    from panna.monitor import LiveMonitor

    driver = _FakeDriver([_snapshot(10.0, reps=1), _snapshot(9.0, reps=2)])
    monitor = LiveMonitor(driver, interval=0.01)
    try:
        monitor.sample()
        monitor.sample()
        lines_before = len(monitor.ax.lines)
        monitor.mark_event("paused")
        assert [label for _x, label in monitor._events] == ["paused"]
        assert len(monitor.ax.lines) == lines_before + 1
    finally:
        monitor.close()


def test_monitor_close_releases_the_figure(tmp_path):
    """pyplot would otherwise keep one figure per run alive for the session."""
    import matplotlib.pyplot as plt

    from panna.monitor import LiveMonitor

    driver = _FakeDriver([_snapshot(10.0, reps=1)])
    monitor = LiveMonitor(driver, interval=0.01)
    monitor.sample()
    number = monitor.fig.number
    assert number in plt.get_fignums()
    monitor.close()
    assert number not in plt.get_fignums()
    assert monitor.closed
    monitor.close()  # idempotent
    # the figure object itself is still usable, e.g. to save the final plot
    out = tmp_path / "after_close.png"
    monitor.fig.savefig(out)
    assert out.is_file()


def test_run_blocking_samples_until_done():
    """The plain-Python loop ends on the final snapshot, without duplicating it."""
    from panna.monitor import LiveMonitor, _run_blocking

    final = _snapshot(8.0, reps=3, done=True)
    driver = _FakeDriver(
        [_snapshot(10.0, reps=1), _snapshot(9.0, reps=2), final, final]
    )
    monitor = LiveMonitor(driver, interval=0.001)
    try:
        _run_blocking(monitor)
        assert monitor.done
        assert monitor.samples == 3
        assert monitor.polls == 4  # the last one repeats `done`
    finally:
        monitor.close()


class _FakeElement:
    def __init__(self, name, children=(), **kwargs):
        self.name = name
        self.children = list(children)
        self.kwargs = kwargs


class _FakeMo:
    """Just enough of the `marimo` module to drive `_MarimoView`."""

    def __init__(self):
        self.log = []
        self.replaced = []
        self.output = self
        self.ui = self

    def _record(self, name, **kwargs):
        element = _FakeElement(name, **kwargs)
        self.log.append(element)
        return element

    def button(self, **kwargs):
        return self._record("button", **kwargs)

    def refresh(self, **kwargs):
        return self._record("refresh", **kwargs)

    def dropdown(self, **kwargs):
        return self._record("dropdown", **kwargs)

    def md(self, value=""):
        return self._record("md", value=value)

    def hstack(self, children, **kwargs):
        return _FakeElement("hstack", children, **kwargs)

    def vstack(self, children, **kwargs):
        return _FakeElement("vstack", children, **kwargs)

    def replace(self, obj):
        self.replaced.append(obj)


def test_marimo_view_reuses_its_widgets():
    """A tick must not rebuild the widgets -- only the status text changes."""
    from panna.monitor import LiveMonitor, _MarimoView

    driver = _FakeDriver(
        [
            _snapshot(10.0, reps=1),
            _snapshot(9.0, reps=2),
            _snapshot(8.0, reps=3),
        ]
    )
    monitor = LiveMonitor(driver, interval=0.01)
    mo = _FakeMo()
    view = _MarimoView(monitor, mo)
    try:
        first, second = view.bundle(), view.bundle()
        assert first is not second  # a fresh container for `output.replace`
        for a, b in zip(first.children, second.children):
            assert a is b  # ...wrapping the very same widgets and figure
        assert view._status.value == monitor.status_markdown()
        # a click steers the run, records where, and repaints at once
        view.control("pause")
        assert driver.calls == ["pause"]
        assert [label for _x, label in monitor._events] == ["paused"]
        assert len(mo.replaced) == 1
    finally:
        monitor.close()

