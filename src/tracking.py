"""Optional MLflow tracking helpers for CLI workflows."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


DEFAULT_REPO_OWNER = "lytlong.pers"
DEFAULT_REPO_NAME = "news-sentiment-analysis"


@dataclass(frozen=True)
class TrackingConfig:
    """Configuration for optional MLflow tracking."""

    enabled: bool = False
    experiment_name: str | None = None
    run_name: str | None = None
    repo_owner: str = DEFAULT_REPO_OWNER
    repo_name: str = DEFAULT_REPO_NAME
    register_model: bool = False
    registered_model_name: str | None = None
    model_alias: str | None = None
    dagshub: bool = False
    tracking_uri: str | None = None
    dagshub_repo_owner: str | None = None
    dagshub_repo_name: str | None = None
    dagshub_token: str | None = None
    dagshub_host: str = "https://dagshub.com"


class TrackingSession:
    """Thin MLflow session wrapper with safe no-op behavior."""

    def __init__(self, *, enabled: bool, mlflow: Any | None = None) -> None:
        self.enabled = enabled
        self._mlflow = mlflow

    @contextmanager
    def start_run(
        self,
        *,
        run_name: str | None = None,
        nested: bool = False,
        tags: Mapping[str, Any] | None = None,
    ) -> Iterator[Any | None]:
        """Start an MLflow run when tracking is enabled."""
        if not self.enabled or self._mlflow is None:
            yield None
            return
        with self._mlflow.start_run(
            run_name=run_name,
            nested=nested,
            tags=_stringify_mapping(tags) if tags else None,
        ) as run:
            yield run

    def active_run_id(self) -> str | None:
        """Return the active MLflow run id when available."""
        if not self.enabled or self._mlflow is None:
            return None
        active = self._mlflow.active_run()
        if active is None:
            return None
        return str(active.info.run_id)

    def log_params(self, values: Mapping[str, Any]) -> None:
        """Log non-empty params as strings."""
        if not self.enabled or self._mlflow is None:
            return
        payload = {
            key: value
            for key, value in _stringify_mapping(values).items()
            if value not in {"", "null"}
        }
        if payload:
            self._mlflow.log_params(payload)

    def log_param(self, key: str, value: Any) -> None:
        """Log one param when it is set."""
        if value is None:
            return
        self.log_params({key: value})

    def log_metrics(
        self, values: Mapping[str, Any], *, prefix: str | None = None
    ) -> None:
        """Log numeric metrics, flattening nested dictionaries."""
        if not self.enabled or self._mlflow is None:
            return
        metrics = flatten_metrics(values, prefix=prefix)
        if metrics:
            self._mlflow.log_metrics(metrics)

    def set_tags(self, values: Mapping[str, Any]) -> None:
        """Set stringified tags when enabled."""
        if not self.enabled or self._mlflow is None:
            return
        payload = {
            key: value
            for key, value in _stringify_mapping(values).items()
            if value not in {"", "null"}
        }
        if payload:
            self._mlflow.set_tags(payload)

    def log_artifact(
        self, path: str | Path, *, artifact_path: str | None = None
    ) -> None:
        """Log a file or directory artifact when it exists."""
        if not self.enabled or self._mlflow is None:
            return
        artifact = Path(path)
        if not artifact.exists():
            return
        if artifact.is_dir():
            self._mlflow.log_artifacts(str(artifact), artifact_path=artifact_path)
        else:
            self._mlflow.log_artifact(str(artifact), artifact_path=artifact_path)

    def register_model(
        self,
        *,
        model_name: str,
        artifact_path: str,
        alias: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, str] | None:
        """Register a model artifact from the active or provided run."""
        if not self.enabled or self._mlflow is None:
            return None
        resolved_run_id = run_id or self.active_run_id()
        if not resolved_run_id:
            raise RuntimeError("Cannot register model without an active MLflow run id.")
        tracking_module = importlib.import_module("mlflow.tracking")
        client = tracking_module.MlflowClient()
        try:
            client.create_registered_model(model_name)
        except Exception as exc:  # pragma: no cover - backend-specific message text
            message = str(exc).lower()
            if (
                "already exists" not in message
                and "resource_already_exists" not in message
            ):
                raise
        source = f"runs:/{resolved_run_id}/{artifact_path}"
        version = client.create_model_version(
            name=model_name,
            source=source,
            run_id=resolved_run_id,
        )
        result = {
            "model_name": model_name,
            "model_version": str(version.version),
            "source": source,
            "run_id": resolved_run_id,
        }
        if alias:
            client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=str(version.version),
            )
            result["alias"] = alias
        return result


def add_tracking_arguments(
    parser: argparse.ArgumentParser, *, include_registry: bool = False
) -> None:
    """Add shared MLflow/DagsHub CLI flags to a parser."""
    parser.add_argument("--mlflow-enabled", action="store_true")
    parser.add_argument("--mlflow-experiment")
    parser.add_argument("--mlflow-run-name")
    parser.add_argument("--dagshub-repo-owner", default=DEFAULT_REPO_OWNER)
    parser.add_argument("--dagshub-repo-name", default=DEFAULT_REPO_NAME)
    if include_registry:
        parser.add_argument("--mlflow-register-model", action="store_true")
        parser.add_argument(
            "--mlflow-registered-model-name", default="phobert-sentiment"
        )
        parser.add_argument("--mlflow-model-alias")


def tracking_config_from_args(args: argparse.Namespace) -> TrackingConfig:
    """Build tracking config from parsed CLI arguments."""
    return TrackingConfig(
        enabled=bool(getattr(args, "mlflow_enabled", False)),
        experiment_name=getattr(args, "mlflow_experiment", None),
        run_name=getattr(args, "mlflow_run_name", None),
        repo_owner=str(getattr(args, "dagshub_repo_owner", DEFAULT_REPO_OWNER)),
        repo_name=str(getattr(args, "dagshub_repo_name", DEFAULT_REPO_NAME)),
        register_model=bool(getattr(args, "mlflow_register_model", False)),
        registered_model_name=getattr(args, "mlflow_registered_model_name", None),
        model_alias=getattr(args, "mlflow_model_alias", None),
        dagshub=bool(getattr(args, "enable_dagshub", False)),
        tracking_uri=getattr(args, "mlflow_tracking_uri", None),
        dagshub_repo_owner=getattr(args, "dagshub_repo_owner", None),
        dagshub_repo_name=getattr(args, "dagshub_repo_name", None),
        dagshub_token=getattr(args, "dagshub_token", None),
        dagshub_host=str(getattr(args, "dagshub_host", "https://dagshub.com")),
    )


def _import_mlflow() -> Any:
    try:
        return importlib.import_module("mlflow")
    except ImportError as exc:
        raise RuntimeError(
            "MLflow tracking requested but 'mlflow' is not installed. "
            "Install it with `pip install mlflow`."
        ) from exc


def _configure_dagshub_for_run(config: TrackingConfig) -> None:
    repo_owner = config.dagshub_repo_owner or config.repo_owner
    repo_name = config.dagshub_repo_name or config.repo_name
    if not repo_owner or not repo_name:
        raise ValueError(
            "DagsHub tracking requires --dagshub-repo-owner and --dagshub-repo-name."
        )
    try:
        dagshub = importlib.import_module("dagshub")
    except ImportError as exc:
        raise RuntimeError(
            "DagsHub tracking requested but 'dagshub' is not installed. "
            "Install it with `pip install dagshub`."
        ) from exc

    if config.dagshub_token:
        os.environ["DAGSHUB_TOKEN"] = config.dagshub_token
    dagshub.init(
        repo_owner=repo_owner,
        repo_name=repo_name,
        mlflow=True,
        host=config.dagshub_host,
    )


def start_tracking_run(
    config: TrackingConfig,
    *,
    run_name: str | None = None,
    nested: bool = False,
):
    """Return a context manager that starts an MLflow run when tracking is enabled."""
    if not config.enabled:
        return nullcontext()

    if config.dagshub:
        _configure_dagshub_for_run(config)

    mlflow = _import_mlflow()
    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)
    if config.experiment_name:
        mlflow.set_experiment(config.experiment_name)
    return mlflow.start_run(run_name=run_name or config.run_name, nested=nested)


def configure_tracking(config: TrackingConfig) -> TrackingSession:
    """Initialize DagsHub-backed MLflow when enabled."""
    if not config.enabled:
        return TrackingSession(enabled=False)

    try:
        mlflow = importlib.import_module("mlflow")
    except ImportError as exc:  # pragma: no cover - exercised in tests via monkeypatch
        raise RuntimeError(
            "MLflow tracking was requested, but the 'mlflow' package is not installed."
        ) from exc
    try:
        dagshub = importlib.import_module("dagshub")
    except ImportError as exc:  # pragma: no cover - exercised in tests via monkeypatch
        raise RuntimeError(
            "MLflow tracking was requested, but the 'dagshub' package is not installed."
        ) from exc

    token = bridge_dagshub_auth()
    if not token:
        raise RuntimeError(
            "MLflow tracking was requested, but DAGSHUB_USER_TOKEN is not set."
        )

    dagshub.init(
        repo_owner=config.repo_owner,
        repo_name=config.repo_name,
        mlflow=True,
    )
    if config.experiment_name:
        mlflow.set_experiment(config.experiment_name)
    return TrackingSession(enabled=True, mlflow=mlflow)


def bridge_dagshub_auth() -> str | None:
    """Bridge DagsHub auth env vars to MLflow auth env vars."""
    username = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_USER_TOKEN")
    if token and not os.getenv("MLFLOW_TRACKING_PASSWORD"):
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    if username and not os.getenv("MLFLOW_TRACKING_USERNAME"):
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
    return token


def collect_cli_params(
    args: argparse.Namespace, *, exclude: set[str] | None = None
) -> dict[str, Any]:
    """Collect parsed CLI params, excluding tracking-only keys when needed."""
    excluded = {
        "mlflow_enabled",
        "mlflow_experiment",
        "mlflow_run_name",
        "dagshub_repo_owner",
        "dagshub_repo_name",
        "mlflow_register_model",
        "mlflow_registered_model_name",
        "mlflow_model_alias",
    }
    if exclude:
        excluded.update(exclude)
    return {
        key: value
        for key, value in vars(args).items()
        if key not in excluded and value is not None
    }


def build_run_tags(
    *,
    stage: str,
    pipeline_mode: str,
    source_dataset: str | None = None,
    base_model: str | None = None,
) -> dict[str, str]:
    """Build standard run tags."""
    tags = {
        "stage": stage,
        "pipeline_mode": pipeline_mode,
    }
    if source_dataset:
        tags["source_dataset"] = source_dataset
    if base_model:
        tags["base_model"] = base_model
    return tags


def git_commit() -> str | None:
    """Return the current git commit hash when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def flatten_metrics(
    values: Mapping[str, Any], *, prefix: str | None = None
) -> dict[str, float]:
    """Flatten nested dictionaries into numeric MLflow metrics."""
    flattened: dict[str, float] = {}
    for key, value in values.items():
        metric_key = sanitize_key(f"{prefix}.{key}" if prefix else str(key))
        if isinstance(value, Mapping):
            flattened.update(flatten_metrics(value, prefix=metric_key))
        elif isinstance(value, bool):
            flattened[metric_key] = float(value)
        elif isinstance(value, int | float):
            flattened[metric_key] = float(value)
    return flattened


def sanitize_key(value: str) -> str:
    """Normalize metric and param subkeys."""
    return value.replace("@", "_at_").replace("/", "_").replace(" ", "_")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON from disk when the file exists, else return an empty dict."""
    target = Path(path)
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def _stringify_mapping(values: Mapping[str, Any]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for key, value in values.items():
        if value is None:
            continue
        payload[str(key)] = _stringify_value(value)
    return payload


def _stringify_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
