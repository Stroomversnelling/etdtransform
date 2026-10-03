"""Permanent phase checkpoints, so every run reports where its time went.

A stage wraps each phase in :meth:`Checkpoints.phase` and the recorder keeps
wall time and peak RSS for it. The result is written beside the stage's output
as ``<stage>_checkpoints.parquet``, so the cost breakdown of any past run can
be read without re-running or re-instrumenting it.

This exists because the calculated stage's ~58 minutes could not be accounted
for: planning, arithmetic and the combine were each measured in isolation and
none of them, nor their sum, explained the total. Isolated benchmarks answer
"how fast is this operation"; only a checkpointed run answers "where did this
run's time go".

Peak RSS is sampled, not exact. It polls the process in a daemon thread, so a
spike shorter than the interval can be missed -- enough to see which phase
owns the memory, not a substitute for a profiler.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import pandas as pd

#: How often the sampler reads process RSS. Short enough to catch a phase that
#: lasts a second, long enough to cost nothing over an hour.
POLL_SECONDS = 0.25


def _rss_bytes() -> int:
    """Resident set size of this process, or 0 when it cannot be read.

    psutil is optional: a stage must not fail because a measurement library is
    absent, so an unavailable reading is recorded as zero rather than raised.
    """
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return 0


@dataclass
class _Record:
    phase: str
    wall_s: float
    peak_rss_gb: float
    rows: int | None
    note: str


@dataclass
class Checkpoints:
    """Phase timings for one stage run.

    Parameters
    ----------
    stage : str
        Name used in the log lines and the output filename.
    """

    stage: str
    records: list = field(default_factory=list)
    #: Identifies this run in the cumulative log. Time-ordered and readable,
    #: so two runs on the same day are still told apart by eye.
    run_id: str = field(
        default_factory=lambda: time.strftime("%Y%m%d-%H%M%S", time.localtime())
    )
    started_at: float = field(default_factory=time.time)

    @contextmanager
    def phase(self, name: str, rows: int | None = None, note: str = ""):
        """Time *name*, sampling peak RSS while it runs.

        Yields a dict the body may update -- set ``rows`` or ``note`` on it to
        record what the phase actually processed, which the caller often only
        learns inside the block.
        """
        mutable = {"rows": rows, "note": note}
        peak = [_rss_bytes()]
        stop = threading.Event()

        def sample():
            while not stop.wait(POLL_SECONDS):
                current = _rss_bytes()
                if current > peak[0]:
                    peak[0] = current

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        started = time.time()
        try:
            yield mutable
        finally:
            wall = time.time() - started
            stop.set()
            sampler.join(timeout=1.0)
            record = _Record(
                phase=name,
                wall_s=wall,
                peak_rss_gb=peak[0] / 1e9,
                rows=mutable.get("rows"),
                note=str(mutable.get("note") or ""),
            )
            self.records.append(record)
            logging.info(
                "[%s] phase %-28s %8.2fs  peak RSS %5.2f GB%s",
                self.stage, name, record.wall_s, record.peak_rss_gb,
                f"  rows {record.rows:,}" if record.rows else "",
            )

    def frame(self) -> pd.DataFrame:
        """The records, with each phase's share of the total wall time."""
        if not self.records:
            return pd.DataFrame(
                columns=["stage", "phase", "wall_s", "peak_rss_gb", "rows",
                         "note", "share_pct"]
            )
        frame = pd.DataFrame([vars(r) for r in self.records])
        frame.insert(0, "stage", self.stage)
        total = frame["wall_s"].sum()
        frame["share_pct"] = (
            (100.0 * frame["wall_s"] / total).round(1) if total else 0.0
        )
        return frame.astype({
            "wall_s": "Float64", "peak_rss_gb": "Float64",
            "rows": "Int64", "share_pct": "Float64",
        })

    def totals(self) -> pd.DataFrame:
        """One row per phase NAME, for a stage that repeats a phase per group.

        Wall time sums, peak RSS takes the worst, because the phases ran one
        after another but the memory high-water mark is a single number for
        the process.
        """
        frame = self.frame()
        if frame.empty:
            return frame
        totals = frame.groupby("phase", as_index=False).agg(
            wall_s=("wall_s", "sum"),
            peak_rss_gb=("peak_rss_gb", "max"),
            rows=("rows", "sum"),
            times=("phase", "size"),
        )
        total = totals["wall_s"].sum()
        totals["share_pct"] = (
            (100.0 * totals["wall_s"] / total).round(1) if total else 0.0
        )
        return totals.sort_values("wall_s", ascending=False)

    def write(self, output_path) -> str | None:
        """Write this run's records, and append them to the run log.

        Two artifacts, because they answer different questions:

        - ``<stage>_checkpoints.parquet`` is THIS run, overwritten each time,
          for "where did that run's time go".
        - ``<stage>_run_log.parquet`` accumulates every run ever, for "has it
          got better or worse since we changed something". A single run cannot
          answer that, and re-deriving it later from scattered notes is how
          the 58-minute figure ended up unattributable for months.

        Never raises: losing a measurement must not fail a stage that
        otherwise succeeded.
        """
        try:
            directory = os.path.dirname(str(output_path)) or "."
            os.makedirs(directory, exist_ok=True)
            frame = self.frame()
            target = os.path.join(directory, f"{self.stage}_checkpoints.parquet")
            frame.to_parquet(target, engine="pyarrow", index=False)
            self.append_to_run_log(directory, frame)
            logging.info("[%s] checkpoints -> %s", self.stage, target)
            return target
        except Exception as problem:
            logging.warning(
                "[%s] could not write checkpoints: %s", self.stage, problem
            )
            return None

    def append_to_run_log(self, directory, frame=None) -> str | None:
        """Append this run's phases to the cumulative run log.

        Stamped with the run id and the start time so runs stay distinguishable
        and ordered. Reads the existing log and rewrites it: parquet has no
        append, and at one row per phase per run the file stays small for
        years.
        """
        try:
            frame = self.frame() if frame is None else frame
            if frame.empty:
                return None
            stamped = frame.copy()
            stamped.insert(0, "run_id", self.run_id)
            stamped.insert(1, "started_at", pd.Timestamp(self.started_at, unit="s", tz="UTC"))
            target = os.path.join(directory, f"{self.stage}_run_log.parquet")
            if os.path.isfile(target):
                previous = pd.read_parquet(target, dtype_backend="numpy_nullable")
                stamped = pd.concat([previous, stamped], ignore_index=True)
            stamped.to_parquet(target, engine="pyarrow", index=False)
            runs = stamped["run_id"].nunique()
            logging.info(
                "[%s] run log -> %s (%d run(s) recorded)",
                self.stage, target, runs,
            )
            return target
        except Exception as problem:
            logging.warning(
                "[%s] could not append to the run log: %s", self.stage, problem
            )
            return None

    def log_summary(self) -> None:
        """One block naming the phases in descending cost.

        Reports the totals rather than every record: a stage that repeats a
        phase per schema group would otherwise print a line per group and
        bury the answer to which phase owns the run.
        """
        totals = self.totals()
        if totals.empty:
            return
        logging.info(
            "[%s] total %.1fs across %d phase(s), %d record(s)",
            self.stage, float(totals["wall_s"].sum()), len(totals),
            len(self.records),
        )
        for row in totals.itertuples():
            logging.info(
                "[%s]   %-28s %8.2fs  %5.1f%%  peak %5.2f GB  x%d",
                self.stage, row.phase, float(row.wall_s),
                float(row.share_pct), float(row.peak_rss_gb), int(row.times),
            )
