"""Standalone background worker for queued training jobs."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from src.config import get_runtime_paths, load_config
from src.service.event_store import EventStore
from src.service.model_builder import TrainingJobCancelled, run_training_job, run_walk_forward_job
from src.service.model_registry import sync_models_registry


def _process_next_job(
    *,
    store: EventStore,
    config: dict[str, Any],
    root: Path,
    artifacts_root: Path,
) -> bool:
    job = store.claim_next_job(job_types=["train_model", "sweep_train", "wf_eval"])
    if not job:
        return False

    job_id = str(job["job_id"])
    job_type = str(job.get("type", "train_model"))

    try:
        if job_type == "wf_eval":
            result = run_walk_forward_job(
                job=job,
                store=store,
                base_config=config,
                root=root,
            )
        else:
            result = run_training_job(
                job=job,
                store=store,
                base_config=config,
                root=root,
                sync_models=lambda: sync_models_registry(store=store, artifacts_root=artifacts_root),
            )
        store.add_job_log(job_id, message="job succeeded", stage="done")
        store.update_job(
            job_id,
            status="succeeded",
            progress=1.0,
            stage="done",
            result=result,
            error_text=None,
            finished=True,
        )
    except TrainingJobCancelled as exc:
        store.add_job_log(job_id, message=str(exc), level="warning", stage="done")
        store.update_job(
            job_id,
            status="canceled",
            progress=1.0,
            stage="done",
            error_text=str(exc),
            finished=True,
        )
    except Exception as exc:
        store.add_job_log(job_id, message=str(exc), level="error", stage="done")
        store.add_alert(
            level="error",
            code="training_job_failed",
            message=str(exc),
            context={"job_id": job_id},
        )
        store.update_job(
            job_id,
            status="failed",
            progress=1.0,
            stage="done",
            error_text=str(exc),
            finished=True,
        )
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.worker")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--root", default=".")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_dir = str(Path(args.config_dir).resolve())
    root_dir = Path(args.root).resolve()
    config = load_config(config_dir)
    paths = get_runtime_paths(config, root=root_dir)
    poll_seconds = max(0.2, float(args.poll_seconds))
    store = EventStore(paths.artifacts_root / "db" / "events.sqlite3")
    try:
        sync_models_registry(store=store, artifacts_root=paths.artifacts_root)
        while True:
            processed = _process_next_job(
                store=store,
                config=config,
                root=paths.root,
                artifacts_root=paths.artifacts_root,
            )
            if processed:
                continue
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        return
    finally:
        store.close()


if __name__ == "__main__":
    main()
