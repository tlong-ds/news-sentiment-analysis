import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import (
    BertConfig,
    PreTrainedTokenizerFast,
    TFBertForMaskedLM,
    TFBertForSequenceClassification,
)

from src.sentiment.bootstrap_labels import (
    ModelSpec,
    bootstrap_labels_frame,
    parse_bootstrap_response,
)
from src.sentiment.common import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    validate_classifier_checkpoint,
)
from src.sentiment.merge_annotations import merge_annotation_frames
from src.sentiment.prepare_training_data import combine_training_sources
from src.sentiment.train_classifier import train_classifier


def _build_tiny_tokenizer(texts: list[str], output_dir: Path) -> int:
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=1,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    fast.save_pretrained(str(output_dir))
    return fast.vocab_size


def _tiny_sequence_classifier_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "tiny-seq-cls"
    model_dir.mkdir(parents=True, exist_ok=True)
    texts = [
        "thi truong tang manh",
        "doanh nghiep gap kho khan",
        "tin tuc trung lap",
        "co phieu giam nhe",
    ]
    vocab_size = _build_tiny_tokenizer(texts, model_dir)
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=256,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        architectures=["TFBertForSequenceClassification"],
    )
    model = TFBertForSequenceClassification(config)
    model(model.dummy_inputs)
    model.save_pretrained(str(model_dir))
    return model_dir


def _tiny_mlm_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "tiny-mlm"
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab_size = _build_tiny_tokenizer(["tin tuc thi truong"], model_dir)
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=128,
        architectures=["TFBertForMaskedLM"],
    )
    model = TFBertForMaskedLM(config)
    model(model.dummy_inputs)
    model.save_pretrained(str(model_dir))
    return model_dir


def _labeled_training_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "article_id": "a1",
                "source": "cafef",
                "category": "Chứng khoán",
                "published_at": "2024-01-01 09:00:00",
                "title": "Co phieu tang",
                "body_text": "thi truong tang manh va thanh khoan tot",
                "input_text": "Co phieu tang . thi truong tang manh va thanh khoan tot",
                "input_text_segmented": "Co phieu tang thi truong tang manh va thanh khoan tot",
                "label": "positive",
                "split": "train",
            },
            {
                "article_id": "a2",
                "source": "cafef",
                "category": "Kinh doanh",
                "published_at": "2024-01-02 09:00:00",
                "title": "Doanh nghiep kho khan",
                "body_text": "doanh nghiep gap kho khan va loi nhuan giam",
                "input_text": "Doanh nghiep kho khan . doanh nghiep gap kho khan va loi nhuan giam",
                "input_text_segmented": "Doanh nghiep kho khan doanh nghiep gap kho khan va loi nhuan giam",
                "label": "negative",
                "split": "train",
            },
            {
                "article_id": "a3",
                "source": "cafef",
                "category": "Vĩ mô",
                "published_at": "2024-01-03 09:00:00",
                "title": "Thong tin on dinh",
                "body_text": "thi truong van on dinh va khong co bien dong lon",
                "input_text": "Thong tin on dinh . thi truong van on dinh va khong co bien dong lon",
                "input_text_segmented": "Thong tin on dinh thi truong van on dinh va khong co bien dong lon",
                "label": "neutral",
                "split": "val",
            },
            {
                "article_id": "a4",
                "source": "cafef",
                "category": "Chứng khoán",
                "published_at": "2024-01-04 09:00:00",
                "title": "Co phieu hoi phuc",
                "body_text": "co phieu hoi phuc sau khi giam sau",
                "input_text": "Co phieu hoi phuc . co phieu hoi phuc sau khi giam sau",
                "input_text_segmented": "Co phieu hoi phuc co phieu hoi phuc sau khi giam sau",
                "label": "positive",
                "split": "test",
            },
        ]
    )


def _articles_clean_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "url": "u1",
                "category": "Chứng khoán",
                "date": "2024-01-02",
                "trading_date": "2024-01-02",
                "title": "Co phieu tang manh",
                "body_clean": "thi truong tang manh va dong tien cai thien ro ret hom nay",
            },
            {
                "url": "u2",
                "category": "Kinh doanh",
                "date": "2024-01-03",
                "trading_date": "2024-01-03",
                "title": "Doanh nghiep doi mat ap luc",
                "body_clean": "doanh nghiep doi mat ap luc chi phi va bien dong nhu cau ngan han",
            },
            {
                "url": "u3",
                "category": "Vĩ mô",
                "date": "2024-01-04",
                "trading_date": "2024-01-04",
                "title": "Kinh te giu nhip on dinh",
                "body_clean": "chi so kinh te giu nhip on dinh va tam ly thi truong can bang",
            },
        ]
    )


def _full_data_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "url": "f1",
                "category": "Tài chính",
                "time": "2024-01-05 10:00:00",
                "title": "Chi phi von ha nhiet",
                "content": "chi phi von giam giup doanh nghiep cai thien ky vong loi nhuan",
            },
            {
                "url": "f2",
                "category": "Vĩ mô",
                "time": "2025-01-05 10:00:00",
                "title": "Du lieu sau moc cat",
                "content": "dong nay phai bi loai bo vi sau ngay gioi han",
            },
        ]
    )


def _prices_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "TRDPRC_1": [100.0, 101.0, 100.5],
            "OPEN_PRC": [99.5, 100.0, 100.8],
            "HIGH_1": [101.0, 102.0, 101.0],
            "LOW_1": [99.0, 99.8, 99.7],
            "ACVOL_UNS": [1000, 1200, 1100],
        }
    )


def _daily_news_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "n_articles": [1, 1, 1],
            "n_categories": [1, 1, 1],
            "mean_body_len": [60.0, 66.0, 62.0],
        }
    )


def _mixed_labeled_training_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "article_id": "c1",
                "source": "cafef",
                "source_dataset": "cafef",
                "category": "Chứng khoán",
                "published_at": "2024-01-01 09:00:00",
                "title": "Co phieu tang",
                "body_text": "thi truong tang manh va thanh khoan tot",
                "input_text": "Co phieu tang . thi truong tang manh va thanh khoan tot",
                "input_text_segmented": "Co phieu tang thi truong tang manh va thanh khoan tot",
                "label": "positive",
                "split": "train",
            },
            {
                "article_id": "c2",
                "source": "cafef",
                "source_dataset": "cafef",
                "category": "Kinh doanh",
                "published_at": "2024-01-02 09:00:00",
                "title": "Doanh nghiep kho khan",
                "body_text": "doanh nghiep gap kho khan va loi nhuan giam",
                "input_text": "Doanh nghiep kho khan . doanh nghiep gap kho khan va loi nhuan giam",
                "input_text_segmented": "Doanh nghiep kho khan doanh nghiep gap kho khan va loi nhuan giam",
                "label": "negative",
                "split": "train",
            },
            {
                "article_id": "f1",
                "source": "full_data",
                "source_dataset": "full_data",
                "category": "Tài chính",
                "published_at": "2024-01-03 09:00:00",
                "title": "Thong tin trung lap",
                "body_text": "du lieu hien tai chua cho thay thay doi manh cua thi truong",
                "input_text": "Thong tin trung lap . du lieu hien tai chua cho thay thay doi manh cua thi truong",
                "input_text_segmented": "Thong tin trung lap du lieu hien tai chua cho thay thay doi manh cua thi truong",
                "label": "neutral",
                "split": "val",
            },
            {
                "article_id": "f2",
                "source": "full_data",
                "source_dataset": "full_data",
                "category": "Doanh nghiệp",
                "published_at": "2024-01-04 09:00:00",
                "title": "Tin xau ngan han",
                "body_text": "ap luc chi phi va don hang giam co the lam xau di ky vong nha dau tu",
                "input_text": "Tin xau ngan han . ap luc chi phi va don hang giam co the lam xau di ky vong nha dau tu",
                "input_text_segmented": "Tin xau ngan han ap luc chi phi va don hang giam co the lam xau di ky vong nha dau tu",
                "label": "negative",
                "split": "val",
            },
            {
                "article_id": "c3",
                "source": "cafef",
                "source_dataset": "cafef",
                "category": "Vĩ mô",
                "published_at": "2024-01-05 09:00:00",
                "title": "Ho tro thi truong",
                "body_text": "tin hieu no long giup cai thien tam ly thi truong",
                "input_text": "Ho tro thi truong . tin hieu no long giup cai thien tam ly thi truong",
                "input_text_segmented": "Ho tro thi truong tin hieu no long giup cai thien tam ly thi truong",
                "label": "positive",
                "split": "test",
            },
            {
                "article_id": "f3",
                "source": "full_data",
                "source_dataset": "full_data",
                "category": "Vĩ mô",
                "published_at": "2024-01-06 09:00:00",
                "title": "Thong tin can bang",
                "body_text": "so lieu moi khong thay doi dang ke ky vong cua nha dau tu",
                "input_text": "Thong tin can bang . so lieu moi khong thay doi dang ke ky vong cua nha dau tu",
                "input_text_segmented": "Thong tin can bang so lieu moi khong thay doi dang ke ky vong cua nha dau tu",
                "label": "neutral",
                "split": "test",
            },
        ]
    )


def test_infer_cafef_fails_for_missing_or_invalid_checkpoint(tmp_path: Path):
    articles_path = tmp_path / "cafef_input.parquet"
    pd.DataFrame(
        [
            {
                "url": "u1",
                "trading_date": "2024-01-02",
                "category": "Chứng khoán",
                "input_text_segmented": "thi truong tang manh",
            }
        ]
    ).to_parquet(articles_path, index=False)

    missing = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.sentiment.infer_cafef",
            "--model-dir",
            str(tmp_path / "missing"),
            "--input-file",
            str(articles_path),
            "--output-file",
            str(tmp_path / "out.parquet"),
        ],
        capture_output=True,
        text=True,
    )
    assert missing.returncode != 0
    assert "Missing classifier checkpoint" in (missing.stderr + missing.stdout)

    invalid_dir = _tiny_mlm_dir(tmp_path)
    invalid = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.sentiment.infer_cafef",
            "--model-dir",
            str(invalid_dir),
            "--input-file",
            str(articles_path),
            "--output-file",
            str(tmp_path / "out-invalid.parquet"),
        ],
        capture_output=True,
        text=True,
    )
    assert invalid.returncode != 0
    assert "not a sequence-classification model" in (invalid.stderr + invalid.stdout)


def test_merge_annotations_rejects_duplicate_missing_and_invalid_labels():
    corpus_df = pd.DataFrame(
        [
            {
                "article_id": "a1",
                "source": "cafef",
                "category": "Chứng khoán",
                "published_at": "2024-01-01",
                "title": "t1",
                "body_text": "b1",
                "input_text": "t1 . b1",
                "input_text_segmented": "t1 b1",
            }
        ]
    )

    with pytest.raises(ValueError, match="duplicate article_id"):
        merge_annotation_frames(
            corpus_df,
            pd.DataFrame(
                [
                    {"article_id": "a1", "label": "positive"},
                    {"article_id": "a1", "label": "negative"},
                ]
            ),
        )

    with pytest.raises(ValueError, match="missing labels"):
        merge_annotation_frames(
            corpus_df, pd.DataFrame([{"article_id": "a1", "label": ""}])
        )

    with pytest.raises(ValueError, match="Invalid sentiment label"):
        merge_annotation_frames(
            corpus_df, pd.DataFrame([{"article_id": "a1", "label": "bullish"}])
        )


def test_parse_bootstrap_response_rejects_malformed_and_invalid_labels():
    with pytest.raises(ValueError, match="valid JSON"):
        parse_bootstrap_response("not-json")
    with pytest.raises(ValueError, match="Invalid sentiment label"):
        parse_bootstrap_response('{"label":"bullish","confidence":0.9,"rationale":"x"}')


def test_bootstrap_labels_and_review_merge_path():
    corpus_df = combine_training_sources(
        _articles_clean_df(),
        extra_df=_full_data_df(),
        extra_source_name="full_data",
        max_date="2024-12-31",
    )

    def fake_runner(prompt: str, spec: ModelSpec) -> str:
        if "Doanh nghiep doi mat ap luc" in prompt:
            return '{"label":"negative","confidence":0.91,"rationale":"ap luc chi phi"}'
        return '{"label":"neutral","confidence":0.62,"rationale":"thong tin can bang"}'

    bootstrap_df, raw_records = bootstrap_labels_frame(
        corpus_df,
        model_specs=[ModelSpec("ollama", "gemma4:latest")],
        prompt_version="test-v1",
        model_runner=fake_runner,
    )
    assert set(bootstrap_df.columns) == {
        "article_id",
        "label",
        "confidence",
        "rationale",
        "model_name",
        "prompt_version",
    }
    assert raw_records

    reviewed_df = pd.DataFrame(
        [{"article_id": corpus_df.iloc[0]["article_id"], "label": "positive"}]
    )
    merged = merge_annotation_frames(
        corpus_df,
        bootstrap_df,
        reviewed_df=reviewed_df,
        confidence_threshold=0.8,
    )
    assert reviewed_df.iloc[0]["article_id"] in set(merged["article_id"])
    merged_row = merged.loc[
        merged["article_id"] == reviewed_df.iloc[0]["article_id"]
    ].iloc[0]
    assert merged_row["label"] == "positive"
    assert merged_row["label_source"] == "reviewed"
    assert merged["confidence"].min() >= 0.8


def test_train_classifier_output_passes_checkpoint_validation(tmp_path: Path):
    base_model_dir = _tiny_sequence_classifier_dir(tmp_path)
    labeled_df = _mixed_labeled_training_df()
    result = train_classifier(
        labeled_df,
        output_dir=tmp_path / "trained-model",
        base_model=str(base_model_dir),
        epochs=0,
        batch_size=2,
        learning_rate=5e-4,
        max_length=64,
        seed=7,
    )
    checkpoint = validate_classifier_checkpoint(result["output_dir"])
    assert checkpoint["num_labels"] == 3
    assert Path(result["output_dir"], "evaluation.json").exists()
    assert set(result["evaluation"]["test_by_source"]) == {"cafef", "full_data"}


def test_run_pipeline_infer_smoke(tmp_path: Path):
    model_dir = _tiny_sequence_classifier_dir(tmp_path)
    articles_path = tmp_path / "articles_clean.parquet"
    prepared_path = tmp_path / "cafef_input.parquet"
    sentiment_path = tmp_path / "article_sentiment_scores.parquet"
    prices_path = tmp_path / "prices_VN.csv"
    daily_news_path = tmp_path / "daily_news_prices.parquet"
    model_frame_path = tmp_path / "modeling_ready.parquet"
    _articles_clean_df().to_parquet(articles_path, index=False)
    _prices_df().to_csv(prices_path, index=False)
    _daily_news_df().to_parquet(daily_news_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.sentiment.run_pipeline",
            "--mode",
            "infer",
            "--model-dir",
            str(model_dir),
            "--cafef-input",
            str(articles_path),
            "--prices-file",
            str(prices_path),
            "--cafef-prepared-output",
            str(prepared_path),
            "--sentiment-output",
            str(sentiment_path),
            "--daily-news-file",
            str(daily_news_path),
            "--model-frame-output",
            str(model_frame_path),
            "--batch-size",
            "2",
            "--max-length",
            "64",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    output_df = pd.read_parquet(sentiment_path)
    assert list(output_df.columns) == [
        "url",
        "trading_date",
        "category",
        "sentiment_score",
        "sentiment_label",
        "prob_positive",
        "prob_negative",
        "prob_neutral",
    ]
    assert (tmp_path / "sentiment_inference_validation.json").exists()
    assert (tmp_path / "daily_aggregation_validation.json").exists()
    model_frame = pd.read_parquet(model_frame_path)
    assert "mean_sentiment" in model_frame.columns
    assert "target_next_vol" in model_frame.columns


def test_run_pipeline_full_smoke(tmp_path: Path):
    base_model_dir = _tiny_sequence_classifier_dir(tmp_path)
    training_input = tmp_path / "training_input.parquet"
    articles_path = tmp_path / "articles_clean.parquet"
    model_dir = tmp_path / "latest"
    sentiment_path = tmp_path / "article_sentiment_scores.parquet"
    prices_path = tmp_path / "prices_VN.csv"
    daily_news_path = tmp_path / "daily_news_prices.parquet"
    model_frame_path = tmp_path / "modeling_ready.parquet"
    _labeled_training_df().to_parquet(training_input, index=False)
    _articles_clean_df().to_parquet(articles_path, index=False)
    _prices_df().to_csv(prices_path, index=False)
    _daily_news_df().to_parquet(daily_news_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.sentiment.run_pipeline",
            "--mode",
            "full",
            "--training-input",
            str(training_input),
            "--base-model",
            str(base_model_dir),
            "--model-dir",
            str(model_dir),
            "--cafef-input",
            str(articles_path),
            "--prices-file",
            str(prices_path),
            "--daily-news-file",
            str(daily_news_path),
            "--cafef-prepared-output",
            str(tmp_path / "cafef_input.parquet"),
            "--training-output",
            str(tmp_path / "training_corpus.parquet"),
            "--merged-labeled-output",
            str(tmp_path / "training_labeled.parquet"),
            "--annotation-sample-output",
            str(tmp_path / "annotation_sample.csv"),
            "--sentiment-output",
            str(sentiment_path),
            "--model-frame-output",
            str(model_frame_path),
            "--epochs",
            "0",
            "--batch-size",
            "2",
            "--max-length",
            "64",
            "--learning-rate",
            "5e-4",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert sentiment_path.exists()
    assert model_frame_path.exists()
    checkpoint = validate_classifier_checkpoint(model_dir)
    assert checkpoint["num_labels"] == 3


def test_combined_prepare_bootstrap_merge_train_smoke(tmp_path: Path):
    combined = combine_training_sources(
        _articles_clean_df(),
        extra_df=_full_data_df(),
        extra_source_name="full_data",
        max_date="2024-12-31",
    )
    bootstrap_df = pd.DataFrame(
        [
            {
                "article_id": row.article_id,
                "label": "positive" if idx % 3 == 0 else "neutral",
                "confidence": 0.95,
                "rationale": "fixture",
                "model_name": "fixture",
                "prompt_version": "v1",
            }
            for idx, row in combined.reset_index(drop=True).iterrows()
        ]
    )
    merged = merge_annotation_frames(combined, bootstrap_df, confidence_threshold=0.8)
    split_map = {
        merged.loc[0, "article_id"]: "train",
        merged.loc[1, "article_id"]: "train",
        merged.loc[2, "article_id"]: "val",
        merged.loc[3, "article_id"]: "test",
    }
    merged["split"] = merged["article_id"].map(split_map).fillna("test")
    result = train_classifier(
        merged,
        output_dir=tmp_path / "smoke-model",
        base_model=str(_tiny_sequence_classifier_dir(tmp_path / "base")),
        epochs=0,
        batch_size=2,
        learning_rate=5e-4,
        max_length=64,
        seed=11,
    )
    assert Path(result["output_dir"], "training_report.json").exists()
