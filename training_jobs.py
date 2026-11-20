"""Start and inspect background recursive M5 training jobs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
JOB_DIR = ROOT / "data" / "model_ready" / "training_jobs"
JOBS_DIR = JOB_DIR / "history"
LOCK_PATH = JOB_DIR / "training.lock"
LATEST_STATUS_PATH = JOB_DIR / "latest.json"
TRAINING_SCRIPT = ROOT / "extended_workflows.py"
FEATURE_FILE = ROOT / "data" / "model_ready" / "m5_model_features.parquet"
FEATURE_CONFIG = ROOT / "data" / "model_ready" / "feature_config.json"
SALES_FILE = ROOT / "data" / "raw" / "sales_train_evaluation.csv"
CALENDAR_FILE = ROOT / "data" / "raw" / "calendar.csv"
PRICES_FILE = ROOT / "data" / "raw" / "sell_prices.csv"
MODEL_DIR = ROOT / "models"
FORECAST_DIR = ROOT / "data" / "model_ready"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def write_job_status(payload: dict[str, Any]) -> dict[str, Any]:
    updated = {**payload, "updated_at_utc": _now()}
    job_id = updated.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("Training job status must include a non-empty job_id.")
    _write_json(JOBS_DIR / f"{job_id}.json", updated)
    _write_json(LATEST_STATUS_PATH, updated)
    return updated


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            SYNCHRONIZE = 0x00100000
            STILL_ACTIVE = 259
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE,
                False,
                pid,
            )
            if not handle:
                return False
            try:
                exit_code = wintypes.DWORD()
                if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return exit_code.value == STILL_ACTIVE
                return False
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True


def check_and_update_job_liveness(status: dict[str, Any] | None) -> dict[str, Any] | None:
    """Check if the job worker is still alive; mark as failed and release lock if dead."""
    if not status or not isinstance(status, dict):
        return status
    job_status = status.get("status")
    if job_status not in {"starting", "running"}:
        return status

    worker_pid = status.get("worker_pid")
    is_alive = False
    if isinstance(worker_pid, int):
        is_alive = _pid_is_running(worker_pid)
    elif worker_pid is None:
        try:
            started = datetime.fromisoformat(status.get("started_at_utc", ""))
            if (datetime.now(timezone.utc) - started).total_seconds() < 30:
                is_alive = True
        except Exception:
            is_alive = False

    if not is_alive:
        updated = {
            **status,
            "status": "failed",
            "stage": "stopped",
            "error": "The background worker process stopped or was terminated before completing.",
            "finished_at_utc": _now(),
        }
        write_job_status(updated)
        if LOCK_PATH.is_file():
            try:
                lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
                if lock.get("job_id") == status.get("job_id") or not _pid_is_running(lock.get("worker_pid", -1)):
                    LOCK_PATH.unlink(missing_ok=True)
            except Exception:
                LOCK_PATH.unlink(missing_ok=True)
        return updated
    return status


def read_log_tail(path_string: str, max_lines: int = 24) -> list[str]:
    path = Path(path_string)
    if max_lines < 1:
        raise ValueError("max_lines must be at least one.")
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as log_file:
        return [line.rstrip("\r\n") for line in log_file.readlines()[-max_lines:]]


def read_job_status(job_id: str) -> dict[str, Any] | None:
    path = JOBS_DIR / f"{job_id}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return check_and_update_job_liveness(data)


def read_latest_job() -> dict[str, Any] | None:
    if not LATEST_STATUS_PATH.is_file():
        return None
    try:
        data = json.loads(LATEST_STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return check_and_update_job_liveness(data)


def cancel_active_job() -> bool:
    """Terminate any running training job and release its lock."""
    cancelled = False
    latest = None
    if LATEST_STATUS_PATH.is_file():
        try:
            latest = json.loads(LATEST_STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            latest = None

    pid_to_kill = None
    if LOCK_PATH.is_file():
        try:
            lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            pid_to_kill = lock.get("worker_pid")
        except Exception:
            pass
        LOCK_PATH.unlink(missing_ok=True)

    if not pid_to_kill and latest:
        pid_to_kill = latest.get("worker_pid")

    if isinstance(pid_to_kill, int) and _pid_is_running(pid_to_kill):
        if os.name == "nt":
            import subprocess
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid_to_kill)], capture_output=True)
        else:
            try:
                os.kill(pid_to_kill, 9)
            except OSError:
                pass
        cancelled = True

    if latest and latest.get("status") in {"starting", "running"}:
        write_job_status({
            **latest,
            "status": "failed",
            "stage": "cancelled",
            "error": "Training was cancelled by the user.",
            "finished_at_utc": _now(),
        })
        cancelled = True
    return cancelled


def _clear_stale_lock() -> None:
    if not LOCK_PATH.is_file():
        return
    try:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot inspect training lock {LOCK_PATH}; resolve it manually."
        ) from exc
    pid = lock.get("worker_pid")
    if isinstance(pid, int) and _pid_is_running(pid):
        raise RuntimeError(
            "A training job is already running. Wait for it to finish before starting another."
        )
    old_job_id = lock.get("job_id")
    if isinstance(old_job_id, str):
        old_status = read_job_status(old_job_id)
        if old_status and old_status.get("status") in {"starting", "running"}:
            write_job_status({
                **old_status,
                "status": "failed",
                "stage": "stopped",
                "error": "The background worker stopped before recording completion.",
            })
    LOCK_PATH.unlink(missing_ok=True)


def start_recursive_training(num_boost_round: int = 150) -> dict[str, Any]:
    """Launch a non-blocking M5 recursive model training and forecast job."""
    required = (
        TRAINING_SCRIPT,
        FEATURE_FILE,
        FEATURE_CONFIG,
        SALES_FILE,
        CALENDAR_FILE,
        PRICES_FILE,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Training requires these project files; prepare/download them first: "
            + ", ".join(missing)
        )

    JOB_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    _clear_stale_lock()

    job_id = uuid.uuid4().hex[:12]
    log_path = JOB_DIR / f"training_{job_id}.log"
    model_path = MODEL_DIR / f"recursive_xgboost_d1941_{job_id}.json"
    model_bundle_path = MODEL_DIR / f"recursive_xgboost_d1941_{job_id}.joblib"
    forecast_path = FORECAST_DIR / f"recursive_forecast_d1942_d1969_{job_id}.parquet"
    metadata_path = MODEL_DIR / f"recursive_xgboost_d1941_{job_id}.metadata.json"
    payload: dict[str, Any] = {
        "job_id": job_id,
        "status": "starting",
        "stage": "starting",
        "started_at_utc": _now(),
        "log_path": str(log_path),
        "num_boost_round": int(num_boost_round),
        "artifacts": {
            "model_path": str(model_path),
            "model_bundle_path": str(model_bundle_path),
            "forecast_path": str(forecast_path),
            "metadata_path": str(metadata_path),
        },
        "training_cutoff": "d_1941",
        "forecast_period": ["d_1942", "d_1969"],
    }

    descriptor = os.open(
        LOCK_PATH,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as lock_file:
        json.dump({"job_id": job_id, "worker_pid": None}, lock_file)

    write_job_status(payload)
    command = [
        sys.executable,
        "-u",
        str(ROOT / "training_job_runner.py"),
        "--job-id",
        job_id,
    ]
    creation_flags = 0
    if os.name == "nt":
        creation_flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        )

    try:
        with log_path.open("a", encoding="utf-8") as worker_log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=worker_log,
                stderr=subprocess.STDOUT,
                close_fds=True,
                creationflags=creation_flags,
            )
    except OSError as exc:
        payload.update(
            {
                "status": "failed",
                "stage": "start_failed",
                "error": str(exc),
                "finished_at_utc": _now(),
            }
        )
        write_job_status(payload)
        LOCK_PATH.unlink(missing_ok=True)
        raise

    with LOCK_PATH.open("w", encoding="utf-8") as lock_file:
        json.dump({"job_id": job_id, "worker_pid": process.pid}, lock_file)
    payload.update(
        {
            "status": "running",
            "stage": "preparing",
            "worker_pid": process.pid,
        }
    )
    return write_job_status(payload)
