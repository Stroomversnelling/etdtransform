"""Phase checkpoints: permanent instrumentation, so no run needs re-measuring.

The contract that matters is that a stage cannot be broken by its own
measurement -- an unavailable RSS reading or an unwritable directory must
degrade, never raise.
"""

import logging

import pandas as pd
import pytest

from etdtransform.checkpoints import Checkpoints


class TestRecording:
    def test_a_phase_is_recorded_with_wall_time(self):
        checkpoints = Checkpoints("teststage")
        with checkpoints.phase("one"):
            pass
        frame = checkpoints.frame()
        assert list(frame["phase"]) == ["one"]
        assert float(frame["wall_s"].iloc[0]) >= 0.0

    def test_phases_keep_their_order(self):
        checkpoints = Checkpoints("teststage")
        for name in ("first", "second", "third"):
            with checkpoints.phase(name):
                pass
        assert list(checkpoints.frame()["phase"]) == ["first", "second", "third"]

    def test_rows_and_note_can_be_set_inside_the_block(self):
        # The caller usually learns what a phase processed only while running
        # it, so the yielded mapping is writable.
        checkpoints = Checkpoints("teststage")
        with checkpoints.phase("one") as mark:
            mark["rows"] = 1234
            mark["note"] = "two groups"
        row = checkpoints.frame().iloc[0]
        assert int(row["rows"]) == 1234
        assert row["note"] == "two groups"

    def test_shares_sum_to_one_hundred(self):
        import time as _time

        checkpoints = Checkpoints("teststage")
        for name in ("a", "b"):
            with checkpoints.phase(name):
                _time.sleep(0.01)
        assert float(checkpoints.frame()["share_pct"].sum()) == pytest.approx(
            100.0, abs=0.2
        )

    def test_a_zero_total_apportions_zero_rather_than_dividing(self):
        # Instant phases give a zero total; a share of nothing is zero, not a
        # division by zero.
        checkpoints = Checkpoints("teststage")
        for name in ("a", "b"):
            with checkpoints.phase(name):
                pass
        frame = checkpoints.frame()
        if float(frame["wall_s"].sum()) == 0.0:
            assert float(frame["share_pct"].sum()) == 0.0

    def test_an_empty_recorder_gives_an_empty_frame_not_an_error(self):
        frame = Checkpoints("teststage").frame()
        assert frame.empty
        assert "phase" in frame.columns


class TestItNeverBreaksTheStage:
    def test_an_exception_in_the_body_still_records_the_phase(self):
        # A phase that raised is the one you most want timed.
        checkpoints = Checkpoints("teststage")
        with pytest.raises(ValueError):
            with checkpoints.phase("boom"):
                raise ValueError("inside")
        assert list(checkpoints.frame()["phase"]) == ["boom"]

    def test_write_failure_is_logged_not_raised(self, tmp_path, caplog):
        checkpoints = Checkpoints("teststage")
        with checkpoints.phase("one"):
            pass
        # A path whose parent cannot be a directory.
        blocker = tmp_path / "file"
        blocker.write_text("x", encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            assert checkpoints.write(blocker / "sub" / "out.parquet") is None

    def test_missing_rss_reading_records_zero(self, monkeypatch):
        import etdtransform.checkpoints as module

        monkeypatch.setattr(module, "_rss_bytes", lambda: 0)
        checkpoints = Checkpoints("teststage")
        with checkpoints.phase("one"):
            pass
        assert float(checkpoints.frame()["peak_rss_gb"].iloc[0]) == 0.0


class TestOutput:
    def test_it_writes_beside_the_stage_output(self, tmp_path):
        checkpoints = Checkpoints("household_calculated")
        with checkpoints.phase("one"):
            pass
        written = checkpoints.write(tmp_path / "household_calculated.parquet")
        assert written is not None
        target = tmp_path / "household_calculated_checkpoints.parquet"
        assert target.exists()
        back = pd.read_parquet(target, dtype_backend="numpy_nullable")
        assert list(back["phase"]) == ["one"]
        assert list(back["stage"]) == ["household_calculated"]
