"""Annotate ViFiC samples with two independent Gemini models."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import ANNOTATION_DATA_DIR

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Bạn là chuyên gia phân tích tài chính Việt Nam với kinh nghiệm đánh giá tác động của tin tức lên thị trường chứng khoán.

Phân loại sentiment của bài báo tài chính sau theo góc nhìn của nhà đầu tư chứng khoán bán lẻ trên thị trường Việt Nam.

Nhãn và định nghĩa:
- positive: tin tức có lợi cho thị trường hoặc cổ phiếu liên quan.
- negative: tin tức bất lợi cho thị trường hoặc cổ phiếu liên quan.
- neutral: thông tin trung lập, không có hàm ý rõ ràng về chiều hướng thị trường.

Lưu ý quan trọng: Phân loại dựa trên hàm ý thị trường, không phải cảm xúc ngôn ngữ.

Bài báo:
Tiêu đề: {title}
Nội dung: {body_lead}

Trả lời chỉ bằng JSON:
{{
  "label": "positive|negative|neutral",
  "confidence": <float 0.0-1.0>,
  "reason": "<1-2 câu giải thích>"
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-LLM annotation for ViFiC sample rows.")
    parser.add_argument("--input-file", default=f"{ANNOTATION_DATA_DIR}/vific_annotation_sample.parquet")
    parser.add_argument("--llm-a-model", default="gemini-2.0-flash-lite")
    parser.add_argument("--llm-b-model", default="gemini-2.0-flash")

    parser.add_argument("--llm-a-output", default=f"{ANNOTATION_DATA_DIR}/llm_a_raw_responses.jsonl")
    parser.add_argument("--llm-b-output", default=f"{ANNOTATION_DATA_DIR}/llm_b_raw_responses.jsonl")
    parser.add_argument("--api-key-a", default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--api-key-b", default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--pilot-size", type=int, default=30)
    parser.add_argument("--pilot-report", default=f"{ANNOTATION_DATA_DIR}/annotation_pilot_report.json")
    parser.add_argument("--require-pilot-pass", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_prompt(row: pd.Series) -> str:
    return PROMPT_TEMPLATE.format(title=row["title"], body_lead=row["body_lead"])


def _endpoint(model_name: str, api_key: str) -> str:
    return (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )


def _extract_text(payload: dict) -> str:
    candidates = payload.get("candidates", [])
    parts: list[str] = []
    for candidate in candidates:
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _load_completed(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    completed: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            completed.add(json.loads(line)["article_id"])
    return completed


def annotate_rows(
    df: pd.DataFrame,
    *,
    model_name: str,
    api_key: str,
    output_path: str,
    timeout: int,
    dry_run: bool,
) -> None:
    completed = _load_completed(output_path)
    mode = "a" if completed else "w"
    with open(output_path, mode, encoding="utf-8") as handle:
        for _, row in df.iterrows():
            article_id = str(row["article_id"])
            if article_id in completed:
                continue

            prompt = build_prompt(row)
            record = {
                "article_id": article_id,
                "model": model_name,
                "prompt": prompt,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if dry_run:
                record["dry_run"] = True
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            response = requests.post(
                _endpoint(model_name, api_key),
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            record["response_text"] = _extract_text(payload)
            record["raw_response"] = payload
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_payload(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def validate_pilot_outputs(
    sample_df: pd.DataFrame,
    llm_a_output: str,
    llm_b_output: str,
    pilot_size: int,
) -> dict:
    pilot_ids = set(sample_df.head(pilot_size)["article_id"].astype(str))

    def collect(path: str) -> list[dict]:
        rows: list[dict] = []
        if not os.path.exists(path):
            return rows
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if str(record.get("article_id")) in pilot_ids:
                    rows.append(record)
        return rows

    raw_a = collect(llm_a_output)
    raw_b = collect(llm_b_output)

    def summarize(records: list[dict]) -> dict:
        parsed = [_parse_payload(record.get("response_text", "")) for record in records]
        parsed_ok = [payload for payload in parsed if payload is not None]
        labels = [str(payload.get("label", "")).lower() for payload in parsed_ok]
        confidences = [float(payload.get("confidence", 0.0)) for payload in parsed_ok if "confidence" in payload]
        reasons = [str(payload.get("reason", "")).strip() for payload in parsed_ok]
        label_distribution = pd.Series(labels).value_counts(normalize=True).to_dict() if labels else {}
        high_conf_share = (
            float(sum(conf >= 0.9 for conf in confidences) / len(confidences))
            if confidences else 0.0
        )
        return {
            "records": len(records),
            "parse_rate": float(len(parsed_ok) / len(records)) if records else 0.0,
            "label_distribution": label_distribution,
            "high_conf_share": high_conf_share,
            "mean_confidence": float(sum(confidences) / len(confidences)) if confidences else 0.0,
            "non_generic_reason_share": float(sum(len(reason.split()) >= 4 for reason in reasons) / len(reasons)) if reasons else 0.0,
        }

    report = {
        "pilot_size": int(min(pilot_size, len(sample_df))),
        "llm_a": summarize(raw_a),
        "llm_b": summarize(raw_b),
    }
    distributions = [report["llm_a"]["label_distribution"], report["llm_b"]["label_distribution"]]
    parse_ok = all(side["parse_rate"] >= 0.95 for side in [report["llm_a"], report["llm_b"]])
    not_degenerate = all(max(dist.values(), default=0.0) < 0.95 for dist in distributions)
    confidence_ok = all(side["high_conf_share"] < 0.95 for side in [report["llm_a"], report["llm_b"]])
    reason_ok = all(side["non_generic_reason_share"] >= 0.80 for side in [report["llm_a"], report["llm_b"]])
    report["checks"] = {
        "parseable_json": parse_ok,
        "non_degenerate_labels": not_degenerate,
        "confidence_not_collapsed": confidence_ok,
        "domain_specific_reasons": reason_ok,
    }
    report["pilot_passed"] = all(report["checks"].values())
    return report


def main() -> None:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    df = pd.read_parquet(args.input_file)
    if args.limit is not None:
        df = df.head(args.limit)

    if not args.dry_run and (not args.api_key_a or not args.api_key_b):
        raise ValueError("Gemini API keys are required unless --dry-run is set.")

    annotate_rows(
        df,
        model_name=args.llm_a_model,
        api_key=args.api_key_a,
        output_path=args.llm_a_output,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )
    annotate_rows(
        df,
        model_name=args.llm_b_model,
        api_key=args.api_key_b,
        output_path=args.llm_b_output,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )
    pilot_report = validate_pilot_outputs(df, args.llm_a_output, args.llm_b_output, args.pilot_size)
    with open(args.pilot_report, "w", encoding="utf-8") as handle:
        json.dump(pilot_report, handle, indent=2, ensure_ascii=False)
    if args.require_pilot_pass and not pilot_report["pilot_passed"]:
        raise RuntimeError(f"Pilot validation failed. See {args.pilot_report}")


if __name__ == "__main__":
    main()
