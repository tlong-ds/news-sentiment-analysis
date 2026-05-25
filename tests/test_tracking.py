from __future__ import annotations

import sys

import pytest

from src.tracking import TrackingConfig, start_tracking_run


def test_start_tracking_run_disabled_returns_nullcontext():
    config = TrackingConfig(enabled=False)
    with start_tracking_run(config):
        assert True


def test_start_tracking_run_requires_mlflow_when_enabled(monkeypatch):
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    config = TrackingConfig(enabled=True)
    with pytest.raises(RuntimeError, match="mlflow"):
        with start_tracking_run(config):
            pass


def test_start_tracking_run_requires_repo_fields_for_dagshub():
    config = TrackingConfig(
        enabled=True,
        dagshub=True,
        repo_owner="",
        repo_name="",
        dagshub_repo_owner="",
        dagshub_repo_name="",
    )
    with pytest.raises(ValueError, match="dagshub-repo-owner"):
        with start_tracking_run(config):
            pass


def test_start_tracking_run_configures_mlflow(monkeypatch):
    calls: list[tuple[str, object]] = []

    class FakeRunContext:
        def __enter__(self):
            calls.append(("enter", None))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", None))
            return False

    class FakeMlflow:
        def set_tracking_uri(self, uri):
            calls.append(("set_tracking_uri", uri))

        def set_experiment(self, name):
            calls.append(("set_experiment", name))

        def start_run(self, run_name=None, nested=False):
            calls.append(("start_run", (run_name, nested)))
            return FakeRunContext()

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())
    config = TrackingConfig(
        enabled=True,
        experiment_name="exp-a",
        run_name="run-a",
        tracking_uri="http://mlflow.local",
    )
    with start_tracking_run(config):
        calls.append(("inside", None))

    assert ("set_tracking_uri", "http://mlflow.local") in calls
    assert ("set_experiment", "exp-a") in calls
    assert ("start_run", ("run-a", False)) in calls
    assert ("inside", None) in calls


def test_start_tracking_run_configures_dagshub(monkeypatch):
    mlflow_calls: list[tuple[str, object]] = []
    dagshub_calls: list[tuple[str, object]] = []

    class FakeRunContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeMlflow:
        def set_tracking_uri(self, uri):
            mlflow_calls.append(("set_tracking_uri", uri))

        def set_experiment(self, name):
            mlflow_calls.append(("set_experiment", name))

        def start_run(self, run_name=None, nested=False):
            mlflow_calls.append(("start_run", (run_name, nested)))
            return FakeRunContext()

    class FakeDagsHub:
        @staticmethod
        def init(repo_owner, repo_name, mlflow, host):
            dagshub_calls.append(("init", (repo_owner, repo_name, mlflow, host)))

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())
    monkeypatch.setitem(sys.modules, "dagshub", FakeDagsHub())
    config = TrackingConfig(
        enabled=True,
        dagshub=True,
        run_name="tracking-run",
        dagshub_repo_owner="owner",
        dagshub_repo_name="repo",
        dagshub_host="https://dagshub.example",
    )
    with start_tracking_run(config):
        pass

    assert dagshub_calls == [
        ("init", ("owner", "repo", True, "https://dagshub.example"))
    ]
    assert ("start_run", ("tracking-run", False)) in mlflow_calls
