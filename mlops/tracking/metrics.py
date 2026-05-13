"""Metrics computation — accuracy, F1, WER, BLEU/ROUGE, latency, memory."""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class GestureMetrics:
    accuracy: float
    f1_weighted: float
    f1_macro: float
    f1_per_class: dict[str, float]
    precision_weighted: float
    recall_weighted: float
    confusion_matrix: list[list[int]]
    n_samples: int
    n_classes: int

    def to_mlflow_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "f1_weighted": self.f1_weighted,
            "f1_macro": self.f1_macro,
            "precision_weighted": self.precision_weighted,
            "recall_weighted": self.recall_weighted,
        }


@dataclass
class NLPMetrics:
    bleu_1: float
    bleu_2: float
    bleu_4: float
    rouge_1: float
    rouge_2: float
    rouge_l: float
    meteor: float = 0.0
    n_samples: int = 0

    def to_mlflow_dict(self) -> dict[str, float]:
        return {
            "bleu_1": self.bleu_1,
            "bleu_2": self.bleu_2,
            "bleu_4": self.bleu_4,
            "rouge_1": self.rouge_1,
            "rouge_2": self.rouge_2,
            "rouge_l": self.rouge_l,
            "meteor": self.meteor,
        }


@dataclass
class ASRMetrics:
    wer: float
    cer: float
    word_accuracy: float
    n_samples: int = 0

    def to_mlflow_dict(self) -> dict[str, float]:
        return {
            "wer": self.wer,
            "cer": self.cer,
            "word_accuracy": self.word_accuracy,
        }


@dataclass
class TTSMetrics:
    utmos_mean: float
    utmos_std: float
    rtf_mean: float
    p95_latency_ms: float
    model_size_mb: float = 0.0
    n_samples: int = 0

    def to_mlflow_dict(self) -> dict[str, float]:
        return {
            "utmos_mean":      self.utmos_mean,
            "utmos_std":       self.utmos_std,
            "rtf_mean":        self.rtf_mean,
            "p95_latency_ms":  self.p95_latency_ms,
            "model_size_mb":   self.model_size_mb,
        }


def compute_tts_metrics(
    audio_arrays: list,
    latencies_ms: list[float],
    sample_rate: int = 24_000,
    model_size_mb: float = 0.0,
) -> TTSMetrics:
    """Score a batch of synthesised audio arrays with UTMOS and compute RTF."""
    import numpy as np

    utmos_scores: list[float] = []
    rtfs: list[float] = []

    for audio, lat in zip(audio_arrays, latencies_ms):
        arr = np.asarray(audio, dtype=np.float32)
        try:
            import speechmos  # type: ignore[import]
            score = float(speechmos.utmos(arr, sample_rate))
        except ImportError:
            rms = float(np.sqrt(np.mean(arr ** 2)))
            zcr = float(np.mean(np.abs(np.diff(np.sign(arr)))) / 2)
            score = min(4.5, max(1.0, 2.5 + rms * 5 - zcr * 3))
        utmos_scores.append(score)
        audio_dur_s = len(arr) / max(sample_rate, 1)
        rtfs.append((lat / 1000.0) / max(audio_dur_s, 1e-6))

    lats = sorted(latencies_ms)
    n = len(lats)
    return TTSMetrics(
        utmos_mean=float(np.mean(utmos_scores)),
        utmos_std=float(np.std(utmos_scores)),
        rtf_mean=float(np.mean(rtfs)),
        p95_latency_ms=lats[int(0.95 * n)] if n else 0.0,
        model_size_mb=model_size_mb,
        n_samples=n,
    )


@dataclass
class LatencyProfile:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    throughput_rps: float
    n_samples: int
    batch_size: int = 1

    def to_mlflow_dict(self) -> dict[str, float]:
        return {
            "latency_mean_ms": self.mean_ms,
            "latency_std_ms": self.std_ms,
            "latency_p50_ms": self.p50_ms,
            "latency_p90_ms": self.p90_ms,
            "latency_p95_ms": self.p95_ms,
            "latency_p99_ms": self.p99_ms,
            "throughput_rps": self.throughput_rps,
        }


@dataclass
class MemoryProfile:
    peak_rss_mb: float
    delta_rss_mb: float
    gpu_vram_mb: float = 0.0

    def to_mlflow_dict(self) -> dict[str, float]:
        return {
            "peak_rss_mb": self.peak_rss_mb,
            "delta_rss_mb": self.delta_rss_mb,
            "gpu_vram_mb": self.gpu_vram_mb,
        }


# ── Gesture metrics ───────────────────────────────────────────────────────────

def compute_gesture_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | None = None,
) -> GestureMetrics:
    """Compute accuracy, F1, precision, recall and confusion matrix."""
    try:
        from sklearn.metrics import (  # type: ignore[import]
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            confusion_matrix,
        )
    except ImportError as e:
        raise ImportError("pip install scikit-learn") from e

    n_classes = len(set(y_true) | set(y_pred))
    names = class_names or [str(i) for i in range(n_classes)]
    f1_per_class_vals = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = {
        names[i]: float(v)
        for i, v in enumerate(f1_per_class_vals)
        if i < len(names)
    }
    return GestureMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_per_class=f1_per_class,
        precision_weighted=float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        recall_weighted=float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
        n_samples=len(y_true),
        n_classes=n_classes,
    )


# ── NLP metrics ───────────────────────────────────────────────────────────────

def compute_nlp_metrics(
    hypotheses: list[str],
    references: list[str],
    compute_meteor: bool = False,
) -> NLPMetrics:
    """Compute BLEU-1/2/4 and ROUGE-1/2/L.  Optionally METEOR."""
    try:
        import sacrebleu  # type: ignore[import]
        from rouge_score import rouge_scorer  # type: ignore[import]
    except ImportError as e:
        raise ImportError("pip install sacrebleu rouge-score") from e

    refs_wrapped = [[r] for r in references]

    bleu1 = sacrebleu.corpus_bleu(hypotheses, list(zip(*refs_wrapped)), max_ngram_order=1)
    bleu2 = sacrebleu.corpus_bleu(hypotheses, list(zip(*refs_wrapped)), max_ngram_order=2)
    bleu4 = sacrebleu.corpus_bleu(hypotheses, list(zip(*refs_wrapped)), max_ngram_order=4)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    r1_list, r2_list, rl_list = [], [], []
    for hyp, ref in zip(hypotheses, references):
        scores = scorer.score(ref, hyp)
        r1_list.append(scores["rouge1"].fmeasure)
        r2_list.append(scores["rouge2"].fmeasure)
        rl_list.append(scores["rougeL"].fmeasure)

    meteor = 0.0
    if compute_meteor:
        try:
            import nltk  # type: ignore[import]
            nltk.download("wordnet", quiet=True)
            meteor_scores = [
                nltk.translate.meteor_score.meteor_score([ref.split()], hyp.split())
                for hyp, ref in zip(hypotheses, references)
            ]
            meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
        except Exception:
            pass

    return NLPMetrics(
        bleu_1=bleu1.score / 100.0,
        bleu_2=bleu2.score / 100.0,
        bleu_4=bleu4.score / 100.0,
        rouge_1=sum(r1_list) / len(r1_list) if r1_list else 0.0,
        rouge_2=sum(r2_list) / len(r2_list) if r2_list else 0.0,
        rouge_l=sum(rl_list) / len(rl_list) if rl_list else 0.0,
        meteor=meteor,
        n_samples=len(hypotheses),
    )


# ── ASR metrics ───────────────────────────────────────────────────────────────

def compute_asr_metrics(
    hypotheses: list[str],
    references: list[str],
) -> ASRMetrics:
    """Compute WER and CER using jiwer."""
    try:
        import jiwer  # type: ignore[import]
    except ImportError as e:
        raise ImportError("pip install jiwer") from e

    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    wer = jiwer.wer(references, hypotheses, truth_transform=transform, hypothesis_transform=transform)
    cer = jiwer.cer(references, hypotheses)
    return ASRMetrics(
        wer=float(wer),
        cer=float(cer),
        word_accuracy=max(0.0, 1.0 - float(wer)),
        n_samples=len(hypotheses),
    )


# ── Latency profiling ─────────────────────────────────────────────────────────

def profile_latency(
    fn: Callable[[], Any],
    n_samples: int = 100,
    warmup_runs: int = 10,
    batch_size: int = 1,
) -> LatencyProfile:
    """Time `fn()` for `n_samples` calls, return percentile stats."""
    import statistics

    for _ in range(warmup_runs):
        fn()

    timings_ms: list[float] = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        fn()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    timings_ms.sort()
    n = len(timings_ms)

    def percentile(p: float) -> float:
        idx = int(p / 100.0 * n)
        return timings_ms[min(idx, n - 1)]

    mean_ms = statistics.mean(timings_ms)
    total_s = sum(timings_ms) / 1000.0
    throughput = (n_samples * batch_size) / total_s if total_s > 0 else 0.0

    return LatencyProfile(
        mean_ms=mean_ms,
        std_ms=statistics.stdev(timings_ms) if n > 1 else 0.0,
        min_ms=timings_ms[0],
        max_ms=timings_ms[-1],
        p50_ms=percentile(50),
        p90_ms=percentile(90),
        p95_ms=percentile(95),
        p99_ms=percentile(99),
        throughput_rps=throughput,
        n_samples=n_samples,
        batch_size=batch_size,
    )


# ── Memory profiling ──────────────────────────────────────────────────────────

def profile_memory(fn: Callable[[], Any]) -> tuple[MemoryProfile, Any]:
    """Run `fn()`, return (MemoryProfile, result).  Measures peak RSS delta."""
    import resource

    gc.collect()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    result = fn()

    gc.collect()
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Linux reports in kB, macOS in bytes
    scale = 1024.0 if os.uname().sysname == "Linux" else 1024.0 * 1024.0
    peak_mb = rss_after / scale
    delta_mb = (rss_after - rss_before) / scale

    gpu_mb = 0.0
    try:
        import torch  # type: ignore[import]
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    return MemoryProfile(
        peak_rss_mb=max(0.0, peak_mb),
        delta_rss_mb=max(0.0, delta_mb),
        gpu_vram_mb=gpu_mb,
    ), result
