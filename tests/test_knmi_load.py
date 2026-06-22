"""Tests for etdtransform.knmi.load_knmi_weather_data.

Covers: blank-string missing values in T/FH/U columns (KNMI API returns '     '
for missing readings instead of NaN), which caused TypeError on division before
the pd.to_numeric coercion fix.
"""

import os
import tempfile
import textwrap

import pandas as pd
import pytest

import ibis

from etdtransform.knmi import load_knmi_weather_data
from etdtransform.load_data import join_weather_data


KNMI_HEADER = textwrap.dedent("""\
    # SOURCE: TEST
    # STN,YYYYMMDD,HH,   DD,   FH,   FF,   FX,    T, T10N,   TD,   SQ,    Q,   DR,   RH,    P,   VV,    N,    U,   WW,   IX,    M,    R,    S,    O,    Y
""")

KNMI_DATA_NORMAL = """\
  215,20220101,    1,  200,   50,   50,   80,   80,     ,   60,    0,    0,    0,    0,10200,   50,    8,   80,     ,     ,    0,    0,    0,    0,    0
  215,20220101,    2,  210,   60,   60,   90,   82,     ,   55,    0,    0,    0,    0,10210,   55,    8,   78,     ,     ,    0,    0,    0,    0,    0
"""

KNMI_DATA_MISSING_T = """\
  323,20220422,   17,     ,     ,     ,     ,     ,     ,     ,     ,     ,     ,     ,     ,     ,     ,     ,     ,     ,    0,    0,    0,    0,    0
  323,20220422,   18,  200,   50,   50,   70,   90,     ,   60,    0,    0,    0,    0,10100,   60,    7,   75,     ,     ,    0,    0,    0,    0,    0
"""

KNMI_DATA_MISSING_DD = """\
  240,20231015,   10,     ,   80,   90,  120,   95,     ,   70,    5,   20,    1,    0,10250,   60,    6,   82,     ,     ,    0,    0,    0,    0,    0
  240,20231015,   11,  180,   75,   85,  110,   97,     ,   72,    6,   22,    0,    0,10255,   65,    6,   81,     ,     ,    0,    0,    0,    0,    0
"""


def _write_knmi_txt(folder, filename, data_rows):
    path = os.path.join(folder, filename)
    with open(path, "w") as f:
        f.write(KNMI_HEADER)
        f.write(data_rows)
    return path


def test_load_normal_file():
    """Standard numeric KNMI file loads without errors."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_knmi_txt(tmp, "normal.txt", KNMI_DATA_NORMAL)
        df = load_knmi_weather_data(tmp)
    assert not df.empty
    assert "Temperatuur" in df.columns
    assert df["Temperatuur"].dtype in (float, "float64")
    assert df["Temperatuur"].iloc[0] == pytest.approx(8.0)


def test_load_file_with_blank_t_values():
    """File with blank-string T values (KNMI missing readings) loads without TypeError."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_knmi_txt(tmp, "missing_t.txt", KNMI_DATA_MISSING_T)
        df = load_knmi_weather_data(tmp)
    assert not df.empty
    # Row with all blanks should have NaN Temperatuur
    missing_rows = df[df["Temperatuur"].isna()]
    assert len(missing_rows) >= 1
    # Row with valid T should parse correctly
    valid_rows = df[df["Temperatuur"].notna()]
    assert len(valid_rows) >= 1
    assert valid_rows["Temperatuur"].iloc[0] == pytest.approx(9.0)


def test_load_mixed_file():
    """File with both normal rows and blank-T rows combines correctly."""
    mixed = KNMI_DATA_NORMAL + KNMI_DATA_MISSING_T
    with tempfile.TemporaryDirectory() as tmp:
        _write_knmi_txt(tmp, "mixed.txt", mixed)
        df = load_knmi_weather_data(tmp)
    assert len(df) == 4
    assert df["Temperatuur"].notna().sum() == 3  # one row is all blanks


def test_join_weather_data_uurvak_alignment():
    """A reading at clock-hour h:00 UT must join to KNMI uurvak h (the obs at
    the end of vak h), not h+1, and a 00:00 reading must roll back to uurvak 24
    of the previous day rather than the non-existent HH=0.

    Each uurvak is given a sentinel temperature equal to its HH so the matched
    vak is identifiable; the cross-day vak 24 gets a distinct sentinel.
    """
    readings = pd.DataFrame({
        "ProjectIdBSV": [1, 1, 1, 1],
        "ReadingDate": pd.to_datetime([
            "2022-01-02 00:00",  # -> uurvak 24 of 2022-01-01
            "2022-01-02 01:00",  # -> uurvak 1  of 2022-01-02
            "2022-01-02 14:00",  # -> uurvak 14 of 2022-01-02
            "2022-01-02 23:00",  # -> uurvak 23 of 2022-01-02
        ]),
    })
    stations = pd.DataFrame({"ProjectIdBSV": [1], "Weerstation": ["TEST"], "STN": [215]})
    weather = pd.DataFrame({
        "STN": [215] * 25,
        "YYYYMMDD": [20220101] + [20220102] * 24,
        "HH": [24] + list(range(1, 25)),
        # vak 24 of Jan-01 gets sentinel 99.0; each Jan-02 vak h gets T=h.
        "Temperatuur": [99.0] + [float(h) for h in range(1, 25)],
    })

    out = join_weather_data(
        ibis.memtable(readings),
        weather_station_table=ibis.memtable(stations),
        weather_table=ibis.memtable(weather),
    ).execute()
    out = out.sort_values("ReadingDate").reset_index(drop=True)

    assert out["Temperatuur"].tolist() == [99.0, 1.0, 14.0, 23.0], (
        f"uurvak alignment wrong: {out[['ReadingDate', 'HH', 'Temperatuur']]}"
    )
    # The 00:00 reading must have rolled back to the previous calendar day.
    assert int(out.loc[0, "YYYYMMDD"]) == 20220101
    assert int(out.loc[0, "HH"]) == 24


def test_load_file_with_blank_dd_values():
    """Blank DD (wind direction) column also loads without ArrowTypeError."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_knmi_txt(tmp, "blank_dd.txt", KNMI_DATA_MISSING_DD)
        df = load_knmi_weather_data(tmp)
    assert not df.empty
    # Row with blank DD should have NaN, not a string
    assert pd.to_numeric(df["DD"], errors="coerce").notna().any()
    # All non-key columns must be numeric dtype (no object columns)
    non_key = [c for c in df.columns if c not in ("STN", "YYYYMMDD", "HH")]
    for col in non_key:
        assert df[col].dtype.kind in ("f", "i", "u"), (
            f"Column {col} has non-numeric dtype {df[col].dtype}"
        )
