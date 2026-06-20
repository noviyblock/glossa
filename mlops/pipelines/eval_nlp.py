"""NLP and ASR evaluation pipeline — BLEU, ROUGE, WER.

Usage:
    python -m mlops.pipelines.eval_nlp
    python -m mlops.pipelines.eval_nlp --modality nlp
    python -m mlops.pipelines.eval_nlp --modality asr
"""

from __future__ import annotations

import argparse
import json
import time
from typing import TYPE_CHECKING, Any

from mlops.config import get_settings
from mlops.tracking.experiment_tracker import ExperimentTracker
from mlops.tracking.metrics import (
    compute_asr_metrics,
    compute_nlp_metrics,
    profile_latency,
)

if TYPE_CHECKING:
    from pathlib import Path


def _load_translation_pairs(data_path: Path) -> tuple[list[str], list[str]]:
    """Load (hypothesis, reference) pairs from test split."""
    test_file = data_path / "translations" / "test" / "pairs.jsonl"
    if not test_file.exists():
        raise FileNotFoundError(
            f"Translation test set not found: {test_file}\n"
            "Expected format: one JSON per line with 'hypothesis' and 'reference' keys."
        )
    hypotheses, references = [], []
    for line in test_file.read_text().splitlines():
        obj = json.loads(line.strip())
        hypotheses.append(obj["hypothesis"])
        references.append(obj["reference"])
    return hypotheses, references


def _load_asr_pairs(data_path: Path) -> tuple[list[str], list[str]]:
    """Load ASR (transcript, reference) pairs from test split."""
    test_file = data_path / "audio" / "test" / "transcripts.jsonl"
    if not test_file.exists():
        raise FileNotFoundError(f"ASR test set not found: {test_file}")
    hypotheses, references = [], []
    for line in test_file.read_text().splitlines():
        obj = json.loads(line.strip())
        hypotheses.append(obj["hypothesis"])
        references.append(obj["reference"])
    return hypotheses, references


def _eval_nlp_service(
    cfg: Any,
    tracker: ExperimentTracker,
    reports_path: Path,
) -> dict[str, Any]:
    """Evaluate NLP translation quality (BLEU/ROUGE) via the running service."""
    hypotheses, references = _load_translation_pairs(cfg.dataset_path)

    # Profile latency against the NLP HTTP service if available
    latency = None
    try:
        import httpx  # type: ignore[import]

        sample_gloss = "ПРИВЕТ КАК ДЕЛА"
        client = httpx.Client(base_url="http://localhost:8003", timeout=30.0)

        latency = profile_latency(
            fn=lambda: client.post("/translate", json={"gloss_sequence": sample_gloss}),
            n_samples=min(50, cfg.benchmark_n_samples),
            warmup_runs=5,
        )
        client.close()
    except Exception:
        pass

    metrics = compute_nlp_metrics(hypotheses, references, compute_meteor=False)

    with tracker.start_run(
        experiment=cfg.nlp_experiment,
        run_name=f"eval_nlp_{int(time.time())}",
        tags={"pipeline": "eval_nlp", "modality": "nlp"},
        dataset_path=cfg.dataset_path / "translations",
    ) as run:
        run_id = run.info.run_id
        tracker.log_metrics(metrics.to_mlflow_dict())
        if latency:
            tracker.log_metrics(latency.to_mlflow_dict())

        report: dict[str, Any] = {
            **metrics.to_mlflow_dict(),
            "n_samples": metrics.n_samples,
            "run_id": run_id,
        }
        if latency:
            report.update(latency.to_mlflow_dict())

    (reports_path / "nlp_eval_metrics.json").write_text(json.dumps(report, indent=2))

    print(
        f"NLP eval: BLEU-4={metrics.bleu_4:.4f} "
        f"ROUGE-L={metrics.rouge_l:.4f} "
        f"n={metrics.n_samples}"
    )
    return report


def _eval_asr_service(
    cfg: Any,
    tracker: ExperimentTracker,
    reports_path: Path,
) -> dict[str, Any]:
    """Evaluate ASR word-error-rate via the running service or offline."""
    hypotheses, references = _load_asr_pairs(cfg.dataset_path)
    metrics = compute_asr_metrics(hypotheses, references)

    with tracker.start_run(
        experiment=cfg.asr_experiment,
        run_name=f"eval_asr_{int(time.time())}",
        tags={"pipeline": "eval_nlp", "modality": "asr"},
        dataset_path=cfg.dataset_path / "audio",
    ) as run:
        run_id = run.info.run_id
        tracker.log_metrics(metrics.to_mlflow_dict())

        report: dict[str, Any] = {
            **metrics.to_mlflow_dict(),
            "n_samples": metrics.n_samples,
            "run_id": run_id,
        }

    (reports_path / "asr_eval_metrics.json").write_text(json.dumps(report, indent=2))

    print(f"ASR eval: WER={metrics.wer:.4f} CER={metrics.cer:.4f} n={metrics.n_samples}")
    return report


def run(modality: str = "nlp") -> dict[str, Any]:
    cfg = get_settings()
    tracker = ExperimentTracker(cfg)
    reports_path = cfg.reports_path
    reports_path.mkdir(parents=True, exist_ok=True)

    if modality == "asr":
        return _eval_asr_service(cfg, tracker, reports_path)
    elif modality == "nlp":
        return _eval_nlp_service(cfg, tracker, reports_path)
    else:
        # Run both
        nlp_report = _eval_nlp_service(cfg, tracker, reports_path)
        asr_report = _eval_asr_service(cfg, tracker, reports_path)
        combined = {"nlp": nlp_report, "asr": asr_report}
        (reports_path / "nlp_eval_metrics.json").write_text(json.dumps(combined, indent=2))
        return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NLP/ASR models")
    parser.add_argument("--modality", choices=["nlp", "asr", "all"], default="nlp")
    args = parser.parse_args()
    run(modality=args.modality)
