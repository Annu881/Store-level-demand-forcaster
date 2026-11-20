"""Background worker for the Streamlit M5 train-and-save workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

import training_jobs
import forecasting_service


ROOT = training_jobs.ROOT
JOB_DIR = training_jobs.JOB_DIR


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_metadata(
    job: dict[str, Any],
    model_path: Path,
    forecast_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    forecast = pd.read_parquet(
        forecast_path,
        columns=["id", "d", "predicted_sales"],
    )
    expected_days = [f"d_{day}" for day in range(1942, 1970)]
    if forecast.empty or forecast["d"].nunique() != 28:
        raise ValueError("The forecast artifact must contain all 28 requested days.")
    if sorted(forecast["d"].unique().tolist()) != expected_days:
        raise ValueError("The saved forecast does not cover d_1942 through d_1969.")
    if forecast.duplicated(["id", "d"]).any():
        raise ValueError("The saved forecast contains duplicate series/day rows.")
    if not np.isfinite(forecast["predicted_sales"].to_numpy(dtype=np.float64)).all() or (
        forecast["predicted_sales"] < 0
    ).any():
        raise ValueError("Forecast predictions must be finite and non-negative.")
    series_count = int(forecast["id"].nunique())
    row_count = int(len(forecast))
    if row_count != series_count * 28:
        raise ValueError("Every series must have exactly one row for each of 28 days.")

    source_files = (
        training_jobs.SALES_FILE,
        training_jobs.CALENDAR_FILE,
        training_jobs.PRICES_FILE,
        training_jobs.FEATURE_FILE,
        ROOT / "data" / "model_ready" / "feature_config.json",
    )
    missing = [str(path) for path in source_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot record model provenance; missing inputs: " + ", ".join(missing)
        )
    source_hashes = {path.name: _sha256(path) for path in source_files}
    data_digest = hashlib.sha256(
        "".join(
            f"{name}:{source_hashes[name]}\n"
            for name in sorted(source_hashes)
        ).encode("utf-8")
    ).hexdigest()
    model_bundle_path = Path(
        job.get("artifacts", {}).get(
            "model_bundle_path",
            str(model_path.with_suffix(".joblib")),
        )
    )

    metadata = {
        "model_name": "recursive_xgboost",
        "model_version": _sha256(model_path)[:12],
        "model_sha256": _sha256(model_path),
        "data_version": data_digest,
        "source_file_sha256": source_hashes,
        "training_cutoff": "d_1941",
        "training_start_day": "d_1",
        "training_end_day": "d_1941",
        "training_day_count": 1941,
        "training_rounds": 150,
        "forecast_days": expected_days,
        "forecast_row_count": row_count,
        "series_count": series_count,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_job_id": job["job_id"],
        "model_path": str(model_path),
        "model_bundle_path": str(model_bundle_path),
        "forecast_path": str(forecast_path),
    }
    model = xgb.Booster()
    model.load_model(str(model_path))
    bundle_metadata = forecasting_service.save_model_bundle(
        model,
        model_bundle_path,
        training_date_utc=job.get("started_at_utc"),
        training_end_day="d_1941",
        data_source="M5 sales, calendar, and sell-price files",
        data_version=data_digest,
        training_data_end_date="2016-05-22",
        activate=True,
    )
    metadata["model_version"] = bundle_metadata["model_version"]
    metadata["model_bundle"] = bundle_metadata
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(metadata_path)
    return metadata


def run_job(job_id: str) -> int:
    job = training_jobs.read_job_status(job_id)
    if job is None:
        raise FileNotFoundError(f"Training job {job_id} was not registered.")
    artifacts = job["artifacts"]
    model_path = Path(artifacts["model_path"])
    model_bundle_path = Path(
        artifacts.get("model_bundle_path", model_path.with_suffix(".joblib"))
    )
    forecast_path = Path(artifacts["forecast_path"])
    metadata_path = Path(artifacts["metadata_path"])
    log_path = Path(job["log_path"])

    training_jobs.write_job_status({
        **job,
        "status": "running",
        "stage": "training_and_forecasting",
    })
    command = [
        sys.executable,
        "-u",
        str(training_jobs.TRAINING_SCRIPT),
        "future",
        "--model-output",
        str(model_path),
        "--forecast-output",
        str(forecast_path),
    ]
    boost_rounds = job.get("num_boost_round")
    if boost_rounds:
        command.extend(["--num-boost-round", str(boost_rounds)])

    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"Starting recursive XGBoost training job {job_id}.\n"
                "The existing default forecast and model files will not be overwritten.\n"
            )
            log_file.flush()
            creation_flags = 0
            if os.name == "nt":
                creation_flags = subprocess.CREATE_NO_WINDOW
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                creationflags=creation_flags,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"Training workflow exited with code {result.returncode}."
            )
        model_metadata = _write_metadata(
            job,
            model_path,
            forecast_path,
            metadata_path,
        )
        training_jobs.write_job_status({
            **job,
            "status": "completed",
            "stage": "completed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_metadata": model_metadata,
        })
        return 0
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("\nTraining job failed:\n")
            traceback.print_exc(file=log_file)
        training_jobs.write_job_status({
            **job,
            "status": "failed",
            "stage": "failed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        })
        return 1
    finally:
        if training_jobs.LOCK_PATH.is_file():
            try:
                lock = json.loads(
                    training_jobs.LOCK_PATH.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                lock = {}
            if lock.get("job_id") == job_id:
                training_jobs.LOCK_PATH.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    raise SystemExit(run_job(args.job_id))


if __name__ == "__main__":
    main()
