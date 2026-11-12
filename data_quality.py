"""Validate the M5 input files and write a versioned data-quality report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "data" / "quality"
SALES_FILE = RAW_DIR / "sales_train_evaluation.csv"
CALENDAR_FILE = RAW_DIR / "calendar.csv"
PRICES_FILE = RAW_DIR / "sell_prices.csv"
SALES_METADATA = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _inspect_prices(
    prices_path: Path,
    price_columns: list[str],
    chunk_size: int = 500_000,
) -> tuple[int, int, int, set[tuple[str, str]]]:
    price_row_count = 0
    duplicate_price_keys = 0
    invalid_prices = 0
    price_pairs: set[tuple[str, str]] = set()
    with tempfile.TemporaryDirectory(prefix="m5_price_keys_") as temporary_dir:
        key_database = Path(temporary_dir) / "price_keys.sqlite"
        connection = sqlite3.connect(key_database)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE price_keys ("
                "store_id TEXT NOT NULL, item_id TEXT NOT NULL, wm_yr_wk INTEGER NOT NULL, "
                "PRIMARY KEY (store_id, item_id, wm_yr_wk)) WITHOUT ROWID"
            )
            for chunk in pd.read_csv(
                prices_path,
                usecols=price_columns,
                chunksize=chunk_size,
            ):
                price_row_count += len(chunk)
                price_keys = chunk[["store_id", "item_id", "wm_yr_wk"]].copy()
                price_keys["wm_yr_wk"] = pd.to_numeric(
                    price_keys["wm_yr_wk"],
                    errors="coerce",
                )
                if price_keys.isna().to_numpy().any() or (
                    price_keys["wm_yr_wk"] % 1 != 0
                ).any():
                    raise ValueError("Price keys must be present and wm_yr_wk must be an integer.")
                price_keys["wm_yr_wk"] = price_keys["wm_yr_wk"].astype("int64")
                numeric_prices = pd.to_numeric(chunk["sell_price"], errors="coerce")
                invalid_prices += int(
                    (~np.isfinite(numeric_prices) | numeric_prices.lt(0)).sum()
                )
                before_insert = connection.total_changes
                connection.executemany(
                    "INSERT OR IGNORE INTO price_keys VALUES (?, ?, ?)",
                    price_keys.itertuples(
                        index=False,
                        name=None,
                    ),
                )
                duplicate_price_keys += len(chunk) - (
                    connection.total_changes - before_insert
                )
                connection.commit()
                price_pairs.update(zip(chunk["store_id"], chunk["item_id"]))
        finally:
            connection.close()
    return price_row_count, duplicate_price_keys, invalid_prices, price_pairs


def validate_m5_inputs(
    sales_path: Path = SALES_FILE,
    calendar_path: Path = CALENDAR_FILE,
    prices_path: Path = PRICES_FILE,
) -> dict[str, Any]:
    """Validate schema, keys, values, date continuity, and join coverage."""
    for path in (sales_path, calendar_path, prices_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required M5 input file not found: {path}")

    sales = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path)
    _require_columns(sales, set(SALES_METADATA), sales_path.name)
    _require_columns(
        calendar,
        {
            "d",
            "date",
            "wm_yr_wk",
            "wday",
            "month",
            "year",
            "event_name_1",
            "event_name_2",
            "snap_CA",
            "snap_TX",
            "snap_WI",
        },
        calendar_path.name,
    )

    day_columns = [column for column in sales.columns if column.startswith("d_")]
    expected_day_columns = [
        f"d_{day}" for day in range(1, len(day_columns) + 1)
    ]
    if day_columns != expected_day_columns:
        raise ValueError("Sales day columns must be consecutive d_1...d_N in order.")
    if not day_columns:
        raise ValueError("Sales data does not contain daily sales columns.")

    duplicate_sales_ids = int(sales["id"].duplicated().sum())
    duplicate_calendar_days = int(calendar["d"].duplicated().sum())
    calendar_dates = pd.to_datetime(calendar["date"], errors="raise")
    date_gaps = int(
        calendar_dates.sort_values().diff().dropna().ne(pd.Timedelta(days=1)).sum()
    )
    if duplicate_sales_ids:
        raise ValueError(f"Sales data has {duplicate_sales_ids} duplicate series IDs.")
    if duplicate_calendar_days:
        raise ValueError(f"Calendar has {duplicate_calendar_days} duplicate day keys.")
    if date_gaps:
        raise ValueError(f"Calendar has {date_gaps} non-daily date intervals.")

    sales_values = sales[day_columns].apply(pd.to_numeric, errors="raise")
    negative_sales = int((sales_values < 0).to_numpy().sum())
    nonfinite_sales = int((~np.isfinite(sales_values.to_numpy(dtype=np.float64))).sum())
    if negative_sales:
        raise ValueError(f"Sales data has {negative_sales} negative values.")
    if nonfinite_sales:
        raise ValueError(f"Sales data has {nonfinite_sales} non-finite values.")

    parsed_ids = sales["id"].str.extract(
        r"^(?P<item>.+)_(?P<store>[A-Z]{2}_[0-9]+)_evaluation$"
    )
    inconsistent_ids = int(
        (
            parsed_ids["item"].ne(sales["item_id"])
            | parsed_ids["store"].ne(sales["store_id"])
            | parsed_ids.isna().any(axis=1)
        ).sum()
    )
    if inconsistent_ids:
        raise ValueError(f"{inconsistent_ids} sales IDs conflict with their metadata.")

    price_columns = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    _require_columns(pd.read_csv(prices_path, nrows=0), set(price_columns), prices_path.name)
    (
        price_row_count,
        duplicate_price_keys,
        invalid_prices,
        price_pairs,
    ) = _inspect_prices(prices_path, price_columns)
    if duplicate_price_keys:
        raise ValueError(f"Price data has {duplicate_price_keys} duplicate keys.")
    if invalid_prices:
        raise ValueError(f"Price data has {invalid_prices} invalid price values.")

    sales_pairs = set(zip(sales["store_id"], sales["item_id"]))
    missing_price_pairs = sales_pairs.difference(price_pairs)
    unknown_price_pairs = price_pairs.difference(sales_pairs)
    calendar_days = calendar["d"].astype(str).tolist()
    missing_calendar_days = sorted(set(day_columns).difference(calendar_days))
    if missing_calendar_days:
        raise ValueError(
            f"Calendar is missing {len(missing_calendar_days)} sales days."
        )

    data_version = hashlib.sha256(
        "".join(
            f"{path.name}:{_sha256(path)}\n"
            for path in (sales_path, calendar_path, prices_path)
        ).encode("utf-8")
    ).hexdigest()

    return {
        "report_schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_version": data_version,
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (sales_path, calendar_path, prices_path)
        },
        "sales": {
            "series_count": int(sales["id"].nunique()),
            "day_count": len(day_columns),
            "first_day": day_columns[0],
            "last_day": day_columns[-1],
            "negative_values": negative_sales,
            "non_finite_values": nonfinite_sales,
            "duplicate_series_ids": duplicate_sales_ids,
            "metadata_consistent_ids": inconsistent_ids == 0,
            "missing_cells": int(sales_values.isna().to_numpy().sum()),
        },
        "calendar": {
            "day_count": len(calendar),
            "duplicate_day_keys": duplicate_calendar_days,
            "date_gap_count": date_gaps,
            "first_date": calendar_dates.min().date().isoformat(),
            "last_date": calendar_dates.max().date().isoformat(),
            "sales_days_missing_from_calendar": len(missing_calendar_days),
        },
        "prices": {
            "row_count": price_row_count,
            "duplicate_price_keys": duplicate_price_keys,
            "invalid_prices": invalid_prices,
            "store_item_pairs_missing_all_prices": len(missing_price_pairs),
            "unknown_store_item_pairs": len(unknown_price_pairs),
        },
        "limitations": [
            "M5 contains observed sales, not confirmed unconstrained demand.",
            "M5 does not contain inventory, availability, or confirmed stock-out labels.",
        ],
    }


def write_quality_report(
    report: dict[str, Any],
    output_dir: Path = REPORT_DIR,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    version = report["data_version"][:12]
    versioned_path = output_dir / f"data_quality_{version}.json"
    latest_path = output_dir / "latest.json"
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    versioned_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")
    return versioned_path, latest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sales", type=Path, default=SALES_FILE)
    parser.add_argument("--calendar", type=Path, default=CALENDAR_FILE)
    parser.add_argument("--prices", type=Path, default=PRICES_FILE)
    args = parser.parse_args()
    report = validate_m5_inputs(args.sales, args.calendar, args.prices)
    paths = write_quality_report(report)
    print(json.dumps(report, indent=2))
    print(f"Versioned report: {paths[0]}")
    print(f"Latest report: {paths[1]}")


if __name__ == "__main__":
    main()
