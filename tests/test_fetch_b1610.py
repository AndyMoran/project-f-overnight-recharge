"""
Tests for src/fetchers/fetch_b1610.py.

We mock the Elexon API so tests run offline and fast. The mock returns
deterministic synthetic responses based on the (date, SP) queried, which
lets us verify:

  - BMU list loading handles various CSV column-name variants.
  - Preflight correctly identifies success, empty-response, missing-BMU,
    and schema-mismatch failure modes.
  - Monthly-chunk fetching writes parquet files, skips already-cached
    months, and combines chunks at the end.
  - The half_hour_end_time → start_time conversion is correct
    (30-minute subtraction, UTC preserved).
  - Retry logic engages on transient failure and gives up after N attempts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.fetchers.fetch_b1610 import (
    PreflightResult,
    _fetch_one_sp,
    _load_bmu_list,
    fetch_b1610,
    preflight,
)


# ===========================================================================
# Fixtures
# ===========================================================================

def _bmu_csv(tmp_path: Path, bmu_ids: list[str], col_name: str = "bmu_id") -> Path:
    """Write a minimal BMU list CSV and return its path."""
    csv_path = tmp_path / "bmus.csv"
    pd.DataFrame({col_name: bmu_ids}).to_csv(csv_path, index=False)
    return csv_path


def _fake_record(bmu_id: str, settlement_date: str, settlement_period: int) -> dict:
    """
    Construct a synthetic B1610 record mirroring Elexon's schema.
    `half_hour_end_time` is derived from (date, SP) using the simple
    (date + SP*30min) formula — this is the Elexon-server convention;
    the DST nuance applies on our side when we convert to start_time.
    """
    base = pd.Timestamp(settlement_date, tz="UTC")
    end_time = base + pd.Timedelta(minutes=30 * settlement_period)
    return {
        "dataset": "B1610",
        "psr_type": "Generation",
        "bm_unit": bmu_id,
        "national_grid_bm_unit_id": f"NG_{bmu_id}",
        "settlement_date": settlement_date,
        "settlement_period": settlement_period,
        "half_hour_end_time": end_time.to_pydatetime(),
        "quantity": 10.0 + settlement_period * 0.1,  # deterministic value
    }


def _make_fake_api(all_bmus: list[str], return_empty_for: set | None = None):
    """
    Build a fake DatasetsApi-like object. Its `datasets_b1610_get` method
    returns a synthetic response for any (date, SP) pair. If the requested
    (date, SP) is in `return_empty_for`, returns empty data.

    The `bm_unit` parameter is honoured — only rows for BMUs in the
    filter list are returned, mirroring the real API's behaviour.
    """
    return_empty_for = return_empty_for or set()
    fake_api = MagicMock()

    def _fake_get(settlement_date, settlement_period, bm_unit=None, format=None):
        key = (settlement_date, settlement_period)
        response = MagicMock()
        if key in return_empty_for:
            response.data = []
            return response

        # Generate one record per BMU in the filter (or per all_bmus if no filter)
        target_bmus = [b for b in all_bmus if (bm_unit is None or b in bm_unit)]
        records = [
            _fake_record(b, settlement_date, settlement_period)
            for b in target_bmus
        ]
        # Mock response.data as a list of objects with to_dict()
        mock_records = []
        for r in records:
            mock_row = MagicMock()
            mock_row.to_dict = lambda r=r: r
            mock_records.append(mock_row)
        response.data = mock_records
        return response

    fake_api.datasets_b1610_get.side_effect = _fake_get
    return fake_api


# ===========================================================================
# BMU list loading
# ===========================================================================

class TestLoadBmuList:
    def test_loads_bmu_id_column(self, tmp_path):
        path = _bmu_csv(tmp_path, ["T_UNIT1", "T_UNIT2"], col_name="bmu_id")
        ids = _load_bmu_list(path)
        assert ids == ["T_UNIT1", "T_UNIT2"]

    def test_loads_bm_unit_column(self, tmp_path):
        path = _bmu_csv(tmp_path, ["T_UNIT1", "T_UNIT2"], col_name="bm_unit")
        ids = _load_bmu_list(path)
        assert ids == ["T_UNIT1", "T_UNIT2"]

    def test_loads_elexon_bmu_id_column(self, tmp_path):
        path = _bmu_csv(tmp_path, ["T_X", "T_Y"], col_name="elexon_bmu_id")
        ids = _load_bmu_list(path)
        assert ids == ["T_X", "T_Y"]

    def test_dedupes_preserving_order(self, tmp_path):
        path = _bmu_csv(tmp_path, ["A", "B", "A", "C", "B"])
        ids = _load_bmu_list(path)
        assert ids == ["A", "B", "C"]

    def test_strips_whitespace_and_empty(self, tmp_path):
        path = _bmu_csv(tmp_path, ["  A  ", "B", "", "  ", "C"])
        ids = _load_bmu_list(path)
        assert ids == ["A", "B", "C"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="BMU list not found"):
            _load_bmu_list(tmp_path / "nope.csv")

    def test_bad_column_name_raises(self, tmp_path):
        path = _bmu_csv(tmp_path, ["A", "B"], col_name="something_else")
        with pytest.raises(ValueError, match="no plausible BMU-ID column"):
            _load_bmu_list(path)

    def test_empty_after_cleaning_raises(self, tmp_path):
        path = _bmu_csv(tmp_path, ["", "  ", ""])
        with pytest.raises(ValueError, match="zero IDs"):
            _load_bmu_list(path)


# ===========================================================================
# Preflight
# ===========================================================================

class TestPreflight:
    def test_all_checks_pass(self, tmp_path):
        path = _bmu_csv(tmp_path, ["T_X", "T_Y"])
        fake_api = _make_fake_api(["T_X", "T_Y"])

        with patch(
            "src.fetchers.fetch_b1610._make_api_client", return_value=fake_api
        ):
            result = preflight(path, sample_date="2026-01-15", sample_period=34)

        assert isinstance(result, PreflightResult)
        assert result.ok is True
        assert result.checks["bmu_list_loads"] is True
        assert result.checks["api_call_succeeds"] is True
        assert result.checks["at_least_one_matching_bmu"] is True
        assert result.checks["schema_has_required_fields"] is True
        assert result.n_bmus_loaded == 2
        assert result.sample_n_records == 2
        assert result.sample_n_matching_bmus == 2
        assert result.errors == []

    def test_bmu_list_missing(self, tmp_path):
        result = preflight(tmp_path / "nope.csv")
        assert result.ok is False
        assert result.checks["bmu_list_loads"] is False
        assert any("bmu_list_loads" in e for e in result.errors)

    def test_api_returns_empty(self, tmp_path):
        path = _bmu_csv(tmp_path, ["T_X"])
        fake_api = _make_fake_api(
            ["T_X"],
            return_empty_for={("2026-01-15", 34)},
        )
        with patch(
            "src.fetchers.fetch_b1610._make_api_client", return_value=fake_api
        ):
            result = preflight(path, sample_date="2026-01-15", sample_period=34)

        assert result.ok is False
        assert result.checks["api_call_succeeds"] is True  # the call itself succeeded
        assert result.sample_n_records == 0
        assert any("zero records" in e for e in result.errors)

    def test_api_raises_exception(self, tmp_path):
        path = _bmu_csv(tmp_path, ["T_X"])
        fake_api = MagicMock()
        fake_api.datasets_b1610_get.side_effect = RuntimeError("connection refused")

        with patch(
            "src.fetchers.fetch_b1610._make_api_client", return_value=fake_api
        ):
            result = preflight(path, sample_date="2026-01-15", sample_period=34)

        assert result.ok is False
        assert result.checks["api_call_succeeds"] is False
        assert any("connection refused" in e for e in result.errors)

    def test_schema_missing_field(self, tmp_path):
        """If the API returns records missing a required field (e.g.
        quantity), the schema check fails."""
        path = _bmu_csv(tmp_path, ["T_X"])
        fake_api = MagicMock()
        response = MagicMock()
        incomplete_record = MagicMock()
        incomplete_record.to_dict = lambda: {
            "bm_unit": "T_X",
            "settlement_date": "2026-01-15",
            "settlement_period": 34,
            # missing: half_hour_end_time, quantity
        }
        response.data = [incomplete_record]
        fake_api.datasets_b1610_get.return_value = response

        with patch(
            "src.fetchers.fetch_b1610._make_api_client", return_value=fake_api
        ):
            result = preflight(path, sample_date="2026-01-15", sample_period=34)

        assert result.ok is False
        assert result.checks["schema_has_required_fields"] is False


# ===========================================================================
# fetch_one_sp retry logic
# ===========================================================================

class TestFetchOneSp:
    def test_success_first_attempt(self):
        fake_api = _make_fake_api(["T_X"])
        rows = _fetch_one_sp(fake_api, "2024-06-15", 34, ["T_X"])
        assert len(rows) == 1
        assert rows[0]["bm_unit"] == "T_X"

    def test_retry_then_succeed(self):
        """First call fails, second succeeds."""
        fake_api = MagicMock()
        response = MagicMock()
        record = MagicMock()
        record.to_dict = lambda: {"bm_unit": "T_X", "quantity": 5.0}
        response.data = [record]

        fake_api.datasets_b1610_get.side_effect = [
            RuntimeError("transient"),
            response,
        ]
        # Use short wait for fast test
        rows = _fetch_one_sp(
            fake_api, "2024-06-15", 34, ["T_X"],
            retry_attempts=3, retry_base_wait=0,
        )
        assert len(rows) == 1
        assert fake_api.datasets_b1610_get.call_count == 2

    def test_all_attempts_fail_raises(self):
        fake_api = MagicMock()
        fake_api.datasets_b1610_get.side_effect = RuntimeError("persistent")
        with pytest.raises(RuntimeError, match="all 3 attempts failed"):
            _fetch_one_sp(
                fake_api, "2024-06-15", 34, ["T_X"],
                retry_attempts=3, retry_base_wait=0,
            )
        assert fake_api.datasets_b1610_get.call_count == 3


# ===========================================================================
# End-to-end (single-month, mocked)
# ===========================================================================

class TestFetchB1610EndToEnd:
    def test_short_fetch_produces_chunk_and_combined(self, tmp_path):
        """Fetch a 3-day period across a single month. Verify a chunk
        file is written, combined parquet exists, and start_time is
        30 minutes before half_hour_end_time."""
        bmu_path = _bmu_csv(tmp_path, ["T_X", "T_Y"])
        chunks_dir = tmp_path / "chunks"
        combined = tmp_path / "combined.parquet"

        fake_api = _make_fake_api(["T_X", "T_Y"])

        with patch(
            "src.fetchers.fetch_b1610._make_api_client", return_value=fake_api
        ):
            df = fetch_b1610(
                start_date="2024-06-15",
                end_date="2024-06-17",
                bmu_list_path=bmu_path,
                chunks_dir=chunks_dir,
                combined_path=combined,
                rate_limit_sleep=0,
                retry_attempts=1,
                retry_base_wait=0,
            )

        assert not df.empty
        # Expected: 2 BMUs * 3 days * 48 SPs = 288 rows
        assert len(df) == 288

        # Chunk file written
        assert (chunks_dir / "b1610_2024-06.parquet").exists()
        # Combined parquet written
        assert combined.exists()

        # Schema checks on the combined output
        for col in ["bm_unit", "settlement_date", "settlement_period",
                    "half_hour_end_time", "start_time", "quantity",
                    "fetched_at"]:
            assert col in df.columns, f"missing column {col}"

        # start_time = half_hour_end_time - 30 min
        row = df.iloc[0]
        delta = row["half_hour_end_time"] - row["start_time"]
        assert delta == pd.Timedelta(minutes=30)

        # start_time is tz-aware UTC
        assert str(df["start_time"].dt.tz) == "UTC"

    def test_cached_chunk_is_skipped(self, tmp_path):
        """If a chunk file already exists, the fetcher should not call
        the API for that month."""
        bmu_path = _bmu_csv(tmp_path, ["T_X"])
        chunks_dir = tmp_path / "chunks"
        combined = tmp_path / "combined.parquet"
        chunks_dir.mkdir()

        # Manually write a "pre-existing" chunk
        stub = pd.DataFrame([{
            "dataset": "B1610",
            "psr_type": "Generation",
            "bm_unit": "T_X",
            "national_grid_bm_unit_id": "NG_T_X",
            "settlement_date": "2024-06-01",
            "settlement_period": 1,
            "half_hour_end_time": pd.Timestamp("2024-06-01 00:30:00", tz="UTC"),
            "quantity": 5.0,
            "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }])
        stub.to_parquet(chunks_dir / "b1610_2024-06.parquet", index=False)

        fake_api = _make_fake_api(["T_X"])

        with patch(
            "src.fetchers.fetch_b1610._make_api_client", return_value=fake_api
        ):
            df = fetch_b1610(
                start_date="2024-06-15",
                end_date="2024-06-17",
                bmu_list_path=bmu_path,
                chunks_dir=chunks_dir,
                combined_path=combined,
                rate_limit_sleep=0,
                retry_attempts=1,
                retry_base_wait=0,
            )

        # API should NOT have been called for fetching (only the
        # preflight call). Preflight calls once; cached month is skipped.
        assert fake_api.datasets_b1610_get.call_count == 1
        # Combined should contain only the stub row.
        assert len(df) == 1

    def test_preflight_only_flag_skips_full_fetch(self, tmp_path):
        bmu_path = _bmu_csv(tmp_path, ["T_X"])
        fake_api = _make_fake_api(["T_X"])

        with patch(
            "src.fetchers.fetch_b1610._make_api_client", return_value=fake_api
        ):
            df = fetch_b1610(
                start_date="2024-06-15",
                end_date="2024-06-17",
                bmu_list_path=bmu_path,
                chunks_dir=tmp_path / "chunks",
                combined_path=tmp_path / "c.parquet",
                preflight_only=True,
                rate_limit_sleep=0,
                retry_attempts=1,
                retry_base_wait=0,
            )

        # Exactly one call (preflight), no fetch calls.
        assert fake_api.datasets_b1610_get.call_count == 1
        assert df.empty

    def test_preflight_failure_blocks_fetch(self, tmp_path):
        """If preflight fails, the full fetch must not start."""
        # Non-existent BMU list → preflight fails on bmu_list_loads.
        with pytest.raises(RuntimeError, match="preflight failed"):
            fetch_b1610(
                start_date="2024-06-15",
                end_date="2024-06-17",
                bmu_list_path=tmp_path / "nope.csv",
                chunks_dir=tmp_path / "chunks",
                combined_path=tmp_path / "c.parquet",
                rate_limit_sleep=0,
                retry_attempts=1,
                retry_base_wait=0,
            )
