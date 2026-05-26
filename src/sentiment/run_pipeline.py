"""Orchestrate training and CafeF inference through the local CLIs."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.config import CAFEF_DATA_DIR, MODELS_DATA_DIR, PROCESSED_DATA_DIR
from src.sentiment.common import LABELED_REQUIRED_COLUMNS, default_model_dir
from src.tracking import (
    add_tracking_arguments,
    build_run_tags,
    collect_cli_params,
    configure_tracking,
    git_commit,
    load_json,
    tracking_config_from_args,
)
from src.utils.io import read_table


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local sentiment training and CafeF inference pipeline."
    )
    parser.add_argument("--mode", choices=["train", "infer", "full"], required=True)
    parser.add_argument("--training-input")
    parser.add_argument("--labeled-input")
    parser.add_argument("--reviewed-labeled-input")
    parser.add_argument("--training-cafef-input")
    parser.add_argument("--training-extra-input")
    parser.add_argument("--training-extra-source-name", default="full_data")
    parser.add_argument("--training-max-date")
    parser.add_argument("--model-dir", default=str(default_model_dir(MODELS_DATA_DIR)))
    parser.add_argument(
        "--training-output", default=f"{CAFEF_DATA_DIR}/training_corpus.parquet"
    )
    parser.add_argument(
        "--annotation-sample-output", default=f"{CAFEF_DATA_DIR}/annotation_sample.csv"
    )
    parser.add_argument(
        "--bootstrap-output",
        default=f"{CAFEF_DATA_DIR}/training_bootstrap_labels.parquet",
    )
    parser.add_argument(
        "--merged-labeled-output", default=f"{CAFEF_DATA_DIR}/training_labeled.parquet"
    )
    parser.add_argument(
        "--cafef-input", default=f"{PROCESSED_DATA_DIR}/articles_clean.parquet"
    )
    parser.add_argument("--prices-file", default="data/main/raw/prices_VN.csv")
    parser.add_argument(
        "--cafef-prepared-output", default=f"{CAFEF_DATA_DIR}/cafef_input.parquet"
    )
    parser.add_argument(
        "--sentiment-output",
        default=f"{PROCESSED_DATA_DIR}/article_sentiment_scores.parquet",
    )
    parser.add_argument(
        "--model-frame-output",
        default=f"{PROCESSED_DATA_DIR}/modeling_ready.parquet",
    )
    parser.add_argument("--daily-news-file")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-model", default="vinai/phobert-base-v2")
    parser.add_argument("--bootstrap-backend", default="ollama")
    parser.add_argument("--bootstrap-model", default="gemma4:latest")
    parser.add_argument(
        "--bootstrap-fallback-models", nargs="*", default=["nemotron-3-nano:4b"]
    )
    parser.add_argument("--bootstrap-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--bootstrap-concurrency", type=int, default=5)
    parser.add_argument("--annotation-sample-size", type=int, default=6000)
    add_tracking_arguments(parser, include_registry=True)
    return parser.parse_args()


def run_command(
    command: list[str],
    *,
    tracking: Any | None = None,
    stage_name: str | None = None,
    stage_tags: dict[str, str] | None = None,
    stage_params: dict[str, Any] | None = None,
    log_stage: Callable[[Any], None] | None = None,
) -> None:
    """Execute one CLI command, optionally wrapping it in a nested MLflow run."""
    logger.info("Running: %s", " ".join(command))
    if tracking and stage_name:
        with tracking.start_run(run_name=stage_name, nested=True, tags=stage_tags):
            tracking.log_params(stage_params or {})
            subprocess.run(command, check=True)
            if log_stage is not None:
                log_stage(tracking)
        return
    subprocess.run(command, check=True)


def resolve_labeled_corpus(
    args: argparse.Namespace, *, tracking: Any | None = None
) -> Path:
    """Prepare or assemble the labeled corpus used for classifier training."""
    if (
        not args.training_input
        and not args.training_cafef_input
        and not args.training_extra_input
    ):
        raise ValueError(
            "Provide --training-input, --training-cafef-input, or --training-extra-input for train/full mode."
        )

    command = [sys.executable, "-m", "src.sentiment.prepare_training_data"]
    source_dataset = "cafef"
    if args.training_input:
        command.extend(["--input-file", args.training_input])
        source_dataset = "training_input"
    elif args.training_cafef_input:
        command.extend(["--cafef-input", args.training_cafef_input])
        if args.training_extra_input:
            command.extend(
                [
                    "--extra-input",
                    args.training_extra_input,
                    "--extra-source-name",
                    args.training_extra_source_name,
                ]
            )
            source_dataset = f"cafef+{args.training_extra_source_name}"
        if args.training_max_date:
            command.extend(["--max-date", args.training_max_date])
    else:
        command.extend(
            [
                "--extra-input",
                args.training_extra_input,
                "--extra-source-name",
                args.training_extra_source_name,
            ]
        )
        source_dataset = args.training_extra_source_name
        if args.training_max_date:
            command.extend(["--max-date", args.training_max_date])
    command.extend(["--output-file", args.training_output])
    run_command(
        command,
        tracking=tracking,
        stage_name="prepare_training_data",
        stage_tags=build_run_tags(
            stage="prepare_training_data",
            pipeline_mode=args.mode,
            source_dataset=source_dataset,
        ),
        stage_params=_stage_params(
            args,
            command=command,
            output_path=args.training_output,
        ),
        log_stage=lambda session: _log_prepare_training_stage(
            session, Path(args.training_output)
        ),
    )
    prepared_df = read_table(args.training_output)
    if set(LABELED_REQUIRED_COLUMNS).issubset(prepared_df.columns):
        logger.info(
            "Prepared training corpus already contains labels and splits; skipping annotation merge."
        )
        Path(args.merged_labeled_output).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.training_output, args.merged_labeled_output)
        return Path(args.merged_labeled_output)

    sample_command = [
        sys.executable,
        "-m",
        "src.sentiment.sample_annotation",
        "--input-file",
        args.training_output,
        "--output-file",
        args.annotation_sample_output,
        "--sample-size",
        str(args.annotation_sample_size),
        "--source-balance",
        "--seed",
        str(args.seed),
    ]
    run_command(
        sample_command,
        tracking=tracking,
        stage_name="sample_annotation",
        stage_tags=build_run_tags(
            stage="sample_annotation",
            pipeline_mode=args.mode,
            source_dataset=source_dataset,
        ),
        stage_params=_stage_params(
            args,
            command=sample_command,
            input_path=args.training_output,
            output_path=args.annotation_sample_output,
        ),
        log_stage=lambda session: _log_sample_annotation_stage(
            session, Path(args.annotation_sample_output)
        ),
    )
    annotations_input = args.labeled_input
    if not annotations_input:
        bootstrap_command = [
            sys.executable,
            "-m",
            "src.sentiment.bootstrap_labels",
            "--input-file",
            args.training_output,
            "--output-file",
            args.bootstrap_output,
            "--backend",
            args.bootstrap_backend,
            "--model",
            args.bootstrap_model,
            "--confidence-threshold",
            str(args.bootstrap_confidence_threshold),
            "--fallback-models",
            *args.bootstrap_fallback_models,
            "--concurrency",
            str(args.bootstrap_concurrency),
        ]
        run_command(
            bootstrap_command,
            tracking=tracking,
            stage_name="bootstrap_labels",
            stage_tags=build_run_tags(
                stage="bootstrap_labels",
                pipeline_mode=args.mode,
                source_dataset=source_dataset,
                base_model=args.bootstrap_model,
            ),
            stage_params=_stage_params(
                args,
                command=bootstrap_command,
                input_path=args.training_output,
                output_path=args.bootstrap_output,
            ),
            log_stage=lambda session: _log_bootstrap_stage(
                session, Path(args.bootstrap_output)
            ),
        )
        annotations_input = args.bootstrap_output

    merge_command = [
        sys.executable,
        "-m",
        "src.sentiment.merge_annotations",
        "--corpus-file",
        args.training_output,
        "--annotations-file",
        annotations_input,
        "--output-file",
        args.merged_labeled_output,
        "--seed",
        str(args.seed),
        "--confidence-threshold",
        str(args.bootstrap_confidence_threshold),
    ] + (
        ["--reviewed-annotations-file", args.reviewed_labeled_input]
        if args.reviewed_labeled_input
        else []
    )
    run_command(
        merge_command,
        tracking=tracking,
        stage_name="merge_annotations",
        stage_tags=build_run_tags(
            stage="merge_annotations",
            pipeline_mode=args.mode,
            source_dataset=source_dataset,
        ),
        stage_params=_stage_params(
            args,
            command=merge_command,
            input_path=args.training_output,
            output_path=args.merged_labeled_output,
        ),
        log_stage=lambda session: _log_merge_annotations_stage(
            session, Path(args.merged_labeled_output)
        ),
    )
    return Path(args.merged_labeled_output)


def run_train_mode(args: argparse.Namespace, *, tracking: Any | None = None) -> None:
    """Execute the training branch of the pipeline."""
    labeled_corpus = resolve_labeled_corpus(args, tracking=tracking)
    train_command = [
        sys.executable,
        "-m",
        "src.sentiment.train_classifier",
        "--labeled-input",
        str(labeled_corpus),
        "--output-dir",
        args.model_dir,
        "--base-model",
        args.base_model,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--max-length",
        str(args.max_length),
        "--seed",
        str(args.seed),
    ]
    run_command(
        train_command,
        tracking=tracking,
        stage_name="train_classifier",
        stage_tags=build_run_tags(
            stage="train_classifier",
            pipeline_mode=args.mode,
            source_dataset="cafef",
            base_model=args.base_model,
        ),
        stage_params=_stage_params(
            args,
            command=train_command,
            input_path=labeled_corpus,
            output_path=args.model_dir,
        ),
        log_stage=lambda session: _log_train_classifier_stage(session, args),
    )


def run_infer_mode(args: argparse.Namespace, *, tracking: Any | None = None) -> None:
    """Execute the inference and modeling-frame branch of the pipeline."""
    sentiment_output = Path(args.sentiment_output)
    validation_report = sentiment_output.parent / "sentiment_inference_validation.json"
    aggregation_report = sentiment_output.parent / "daily_aggregation_validation.json"
    daily_news_file = args.daily_news_file or str(
        Path(args.cafef_input).parent / "daily_news_prices.parquet"
    )

    prepare_inputs_command = [
        sys.executable,
        "-m",
        "src.sentiment.prepare_inputs",
        "--cafef-input",
        args.cafef_input,
        "--cafef-output",
        args.cafef_prepared_output,
        "--max-tokens",
        str(args.max_length),
    ]
    run_command(
        prepare_inputs_command,
        tracking=tracking,
        stage_name="prepare_inputs",
        stage_tags=build_run_tags(
            stage="prepare_inputs",
            pipeline_mode=args.mode,
            source_dataset="cafef",
        ),
        stage_params=_stage_params(
            args,
            command=prepare_inputs_command,
            input_path=args.cafef_input,
            output_path=args.cafef_prepared_output,
        ),
        log_stage=lambda session: _log_prepare_inputs_stage(
            session, Path(args.cafef_prepared_output)
        ),
    )

    infer_command = [
        sys.executable,
        "-m",
        "src.sentiment.infer_cafef",
        "--model-dir",
        args.model_dir,
        "--input-file",
        args.cafef_prepared_output,
        "--output-file",
        args.sentiment_output,
        "--batch-size",
        str(args.batch_size),
        "--max-length",
        str(args.max_length),
    ]
    run_command(
        infer_command,
        tracking=tracking,
        stage_name="infer_cafef",
        stage_tags=build_run_tags(
            stage="infer_cafef",
            pipeline_mode=args.mode,
            source_dataset="cafef",
            base_model=args.base_model,
        ),
        stage_params=_stage_params(
            args,
            command=infer_command,
            input_path=args.cafef_prepared_output,
            output_path=args.sentiment_output,
        ),
        log_stage=lambda session: _log_infer_stage(session, sentiment_output),
    )

    validate_inference_command = [
        sys.executable,
        "-m",
        "src.sentiment.validate_inference",
        "--articles-file",
        args.cafef_input,
        "--sentiment-file",
        args.sentiment_output,
        "--daily-news-file",
        daily_news_file,
        "--report-file",
        str(validation_report),
        "--fail-on-validation",
    ]
    run_command(
        validate_inference_command,
        tracking=tracking,
        stage_name="validate_inference",
        stage_tags=build_run_tags(
            stage="validate_inference",
            pipeline_mode=args.mode,
            source_dataset="cafef",
        ),
        stage_params=_stage_params(
            args,
            command=validate_inference_command,
            input_path=args.sentiment_output,
            output_path=validation_report,
        ),
        log_stage=lambda session: _log_json_stage(
            session,
            validation_report,
            extra_artifacts=[sentiment_output],
        ),
    )

    validate_aggregation_command = [
        sys.executable,
        "-m",
        "src.sentiment.validate_daily_aggregation",
        "--sentiment-file",
        args.sentiment_output,
        "--output-file",
        str(aggregation_report),
    ]
    run_command(
        validate_aggregation_command,
        tracking=tracking,
        stage_name="validate_daily_aggregation",
        stage_tags=build_run_tags(
            stage="validate_daily_aggregation",
            pipeline_mode=args.mode,
            source_dataset="cafef",
        ),
        stage_params=_stage_params(
            args,
            command=validate_aggregation_command,
            input_path=args.sentiment_output,
            output_path=aggregation_report,
        ),
        log_stage=lambda session: _log_json_stage(
            session,
            aggregation_report,
            extra_artifacts=[sentiment_output],
        ),
    )

    prepare_model_frame_command = [
        sys.executable,
        "-m",
        "src.modeling.prepare_model_frame",
        "--prices",
        args.prices_file,
        "--daily-news",
        daily_news_file,
        "--sentiment",
        args.sentiment_output,
        "--articles-clean",
        args.cafef_input,
        "--output-file",
        args.model_frame_output,
    ]
    run_command(
        prepare_model_frame_command,
        tracking=tracking,
        stage_name="prepare_model_frame",
        stage_tags=build_run_tags(
            stage="prepare_model_frame",
            pipeline_mode=args.mode,
            source_dataset="cafef",
        ),
        stage_params=_stage_params(
            args,
            command=prepare_model_frame_command,
            input_path=args.sentiment_output,
            output_path=args.model_frame_output,
        ),
        log_stage=lambda session: _log_model_frame_stage(
            session, Path(args.model_frame_output)
        ),
    )


def main() -> None:
    """CLI entrypoint for the end-to-end sentiment pipeline."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    tracking_config = tracking_config_from_args(args)
    tracking = configure_tracking(tracking_config)
    run_name = tracking_config.run_name or f"sentiment_pipeline_{args.mode}"
    run_tags = build_run_tags(
        stage="run_pipeline",
        pipeline_mode=args.mode,
        source_dataset="cafef",
        base_model=args.base_model,
    )
    with tracking.start_run(run_name=run_name, tags=run_tags):
        tracking.log_params(collect_cli_params(args))
        tracking.log_params(
            {
                "invoked_at": _timestamp(),
                "git_commit": git_commit(),
            }
        )
        if args.mode in {"train", "full"}:
            run_train_mode(args, tracking=tracking)
        if args.mode in {"infer", "full"}:
            run_infer_mode(args, tracking=tracking)


def _stage_params(
    args: argparse.Namespace,
    *,
    command: list[str],
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "command": command,
        "input_path": str(input_path) if input_path is not None else None,
        "output_path": str(output_path) if output_path is not None else None,
        "seed": args.seed,
        "invoked_at": _timestamp(),
        "git_commit": git_commit(),
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_prepare_training_stage(tracking: Any, output_path: Path) -> None:
    prepared_df = read_table(output_path)
    tracking.log_metrics(
        {
            "rows": len(prepared_df),
            "source_count": prepared_df["source"].nunique(),
        }
    )
    tracking.log_params(
        {
            "date_min": str(prepared_df["published_at"].min()),
            "date_max": str(prepared_df["published_at"].max()),
            "sources": prepared_df["source"].value_counts().to_dict(),
        }
    )
    tracking.log_artifact(output_path)
    tracking.log_artifact(output_path.with_suffix(".report.json"))


def _log_sample_annotation_stage(tracking: Any, output_path: Path) -> None:
    sample_df = read_table(output_path)
    source_column = (
        "source_dataset" if "source_dataset" in sample_df.columns else "source"
    )
    tracking.log_metrics(
        {
            "sample_rows": len(sample_df),
            "source_count": sample_df[source_column].nunique() if len(sample_df) else 0,
        }
    )
    tracking.log_params(
        {
            "sample_source_distribution": sample_df[source_column]
            .value_counts()
            .to_dict()
            if len(sample_df)
            else {},
        }
    )
    tracking.log_artifact(output_path)
    tracking.log_artifact(output_path.with_suffix(".report.json"))


def _log_bootstrap_stage(tracking: Any, output_path: Path) -> None:
    labels_df = read_table(output_path)
    report_path = output_path.with_suffix(".report.json")
    raw_output_path = output_path.with_suffix(".raw.jsonl")
    report = load_json(report_path)
    rejected = int(len(labels_df) - report.get("auto_accept_rows", len(labels_df)))
    tracking.log_metrics(
        {
            "rows": len(labels_df),
            "accepted_rows": report.get("auto_accept_rows", len(labels_df)),
            "rejected_rows": rejected,
        }
    )
    tracking.log_params(
        {
            "label_distribution": labels_df["label"].value_counts().to_dict(),
            "models": report.get("models", {}),
            "prompt_versions": report.get("prompt_versions", {}),
        }
    )
    tracking.log_artifact(output_path)
    tracking.log_artifact(report_path)
    tracking.log_artifact(raw_output_path)


def _log_merge_annotations_stage(tracking: Any, output_path: Path) -> None:
    merged_df = read_table(output_path)
    report_path = output_path.with_suffix(".report.json")
    tracking.log_metrics(
        {
            "rows": len(merged_df),
            "reviewed_override_count": int(
                (merged_df.get("label_source") == "reviewed").sum()
            )
            if "label_source" in merged_df.columns
            else 0,
            "low_confidence_drops": 0,
        }
    )
    tracking.log_params(
        {
            "split_counts": merged_df["split"].value_counts().to_dict(),
            "label_distribution": merged_df["label"].value_counts().to_dict(),
            "source_distribution": (
                merged_df["source_dataset"].value_counts().to_dict()
                if "source_dataset" in merged_df.columns
                else merged_df["source"].value_counts().to_dict()
            ),
        }
    )
    tracking.log_artifact(output_path)
    tracking.log_artifact(report_path)


def _log_train_classifier_stage(tracking: Any, args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    evaluation_path = model_dir / "evaluation.json"
    training_report_path = model_dir / "training_report.json"
    evaluation = load_json(evaluation_path)
    report = load_json(training_report_path)
    tracking.log_metrics(evaluation)
    tracking.log_metrics(report.get("evaluation", {}), prefix="report")
    tracking.log_params(
        {
            "checkpoint_dir": str(model_dir),
            "checkpoint_files": sorted(path.name for path in model_dir.iterdir())
            if model_dir.exists()
            else [],
        }
    )
    tracking.log_artifact(model_dir, artifact_path="classifier_model")
    tracking.log_artifact(evaluation_path)
    tracking.log_artifact(training_report_path)
    if args.mlflow_register_model:
        tracking.register_model(
            model_name=args.mlflow_registered_model_name,
            artifact_path="classifier_model",
            alias=args.mlflow_model_alias,
        )


def _log_prepare_inputs_stage(tracking: Any, output_path: Path) -> None:
    prepared_df = read_table(output_path)
    tracking.log_metrics({"rows": len(prepared_df)})
    tracking.log_params({"columns": prepared_df.columns.tolist()})
    tracking.log_artifact(output_path)


def _log_infer_stage(tracking: Any, output_path: Path) -> None:
    inference_df = read_table(output_path)
    tracking.log_metrics(
        {
            "rows": len(inference_df),
            "sentiment_score_mean": float(inference_df["sentiment_score"].mean()),
            "sentiment_score_std": float(inference_df["sentiment_score"].std(ddof=0)),
        }
    )
    tracking.log_params(
        {
            "label_distribution": inference_df["sentiment_label"]
            .value_counts()
            .to_dict(),
        }
    )
    tracking.log_artifact(output_path)
    tracking.log_artifact(output_path.with_suffix(".report.json"))


def _log_json_stage(
    tracking: Any, report_path: Path, *, extra_artifacts: list[Path] | None = None
) -> None:
    report = load_json(report_path)
    tracking.log_metrics(report)
    tracking.log_params(
        {key: value for key, value in report.items() if not isinstance(value, dict)}
    )
    tracking.log_artifact(report_path)
    for artifact in extra_artifacts or []:
        tracking.log_artifact(artifact)


def _log_model_frame_stage(tracking: Any, output_path: Path) -> None:
    model_df = read_table(output_path)
    tracking.log_metrics(
        {
            "rows": len(model_df),
            "feature_count": len(model_df.columns),
        }
    )
    tracking.log_params(
        {
            "date_min": str(model_df["date"].min()),
            "date_max": str(model_df["date"].max()),
            "columns": model_df.columns.tolist(),
        }
    )
    tracking.log_artifact(output_path)
    tracking.log_artifact(output_path.with_suffix(".report.json"))


if __name__ == "__main__":
    main()
