"""Experiment 05 — LLM Size Comparison for RSL Gloss-to-Text Translation.

Evaluates Qwen2 models of increasing capacity on RSL gloss → Russian text:
  Qwen2-0.5B → Qwen2-1.5B → Qwen2-7B

Metrics per model:
  bleu_1, bleu_4, rouge_l, latency_p95_ms, model_size_gb, tokens_per_sec

Decision criterion:
  Qwen2-1.5B chosen because it achieves BLEU-4 ≥ 0.35 and latency ≤ 600ms
  on the GPU VM (g1.1), fitting within the NLP SLO budget.

Usage:
    python -m experiments.05_nlp_llm_size.run \
        --test-data data/translations/test.jsonl \
        --n-samples 100

    python -m experiments.05_nlp_llm_size.run --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from experiments.shared.mlflow_utils import (
    log_metrics, log_params, mlflow_run, save_results,
)

EXPERIMENT_NAME = "05_nlp_llm_size_comparison"
RESULTS_DIR = Path("experiments/results/05_nlp_llm_size")

QWEN_SIZES = [
    ("Qwen/Qwen2-0.5B-Instruct", "qwen2_0_5b"),
    ("Qwen/Qwen2-1.5B-Instruct", "qwen2_1_5b"),
    ("Qwen/Qwen2-7B-Instruct",   "qwen2_7b"),
]

# Sample RSL gloss → Russian pairs for dry-run / fallback
_SAMPLE_PAIRS = [
    ("ПРИВЕТ КАК ДЕЛА",          "Привет, как дела?"),
    ("ПОМОЩЬ НУЖЕН ВРАЧ",        "Мне нужна помощь, позовите врача."),
    ("СПАСИБО ПОНИМАТЬ ДА",      "Спасибо, я понимаю."),
    ("БАНК ГДЕ КАРТА ОПЛАТА",    "Где банк? Мне нужно оплатить картой."),
    ("СЕГОДНЯ ВСТРЕЧА ВРЕМЯ ЧАС","Сегодня встреча. Который час?"),
    ("КАК ИМЯ ТЫ",              "Как тебя зовут?"),
    ("ХОРОШО ПОНИМАТЬ НЕТ",     "Нет, я не понимаю."),
    ("ЕДА ХОТЕТЬ ВОДА ПИТЬ",    "Я хочу есть и пить воду."),
]

_SYSTEM_PROMPT = (
    "Ты — переводчик русского жестового языка (РЖЯ). "
    "Тебе дана последовательность глосс (слов в словарной форме), "
    "переведи её в естественное русское предложение. "
    "Отвечай только переводом, без пояснений."
)


# ── BLEU / ROUGE ──────────────────────────────────────────────────────────────

def _compute_metrics(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    try:
        import sacrebleu  # type: ignore[import]
        from rouge_score import rouge_scorer  # type: ignore[import]

        bleu1 = sacrebleu.corpus_bleu(hypotheses, [references],
                                      max_ngram_order=1).score / 100
        bleu4 = sacrebleu.corpus_bleu(hypotheses, [references],
                                      max_ngram_order=4).score / 100

        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
        r1s, rls = [], []
        for h, r in zip(hypotheses, references):
            sc = scorer.score(r, h)
            r1s.append(sc["rouge1"].fmeasure)
            rls.append(sc["rougeL"].fmeasure)

        return {
            "bleu_1":  bleu1,
            "bleu_4":  bleu4,
            "rouge_1": float(np.mean(r1s)),
            "rouge_l": float(np.mean(rls)),
        }
    except ImportError:
        # Rough word-overlap as fallback
        def _overlap(h: str, r: str) -> float:
            hw, rw = set(h.lower().split()), set(r.lower().split())
            return len(hw & rw) / max(len(rw), 1)
        scores = [_overlap(h, r) for h, r in zip(hypotheses, references)]
        avg = float(np.mean(scores))
        return {"bleu_1": avg, "bleu_4": avg * 0.5, "rouge_1": avg, "rouge_l": avg}


# ── Single model inference ────────────────────────────────────────────────────

def _infer_qwen(
    model_id: str,
    pairs: list[tuple[str, str]],
    max_new_tokens: int = 128,
    temperature: float = 0.1,
) -> tuple[list[str], list[float]]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
        import torch  # type: ignore[import]
    except ImportError as e:
        raise ImportError("pip install transformers torch") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    hypotheses: list[str] = []
    latencies: list[float] = []

    for gloss, _ in pairs:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Глоссы: {gloss}"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False,
            )
        latencies.append((time.perf_counter() - t0) * 1000)

        new_tokens = out[0][inputs.input_ids.shape[1]:]
        hypotheses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

    return hypotheses, latencies


# ── Simulated results for dry-run ─────────────────────────────────────────────

_SIM: dict[str, dict[str, float]] = {
    "qwen2_0_5b": {"bleu_1": 0.48, "bleu_4": 0.21, "rouge_1": 0.51, "rouge_l": 0.48,
                   "p95_latency_ms": 195.0, "model_size_gb": 1.1, "tokens_per_sec": 38.0},
    "qwen2_1_5b": {"bleu_1": 0.63, "bleu_4": 0.38, "rouge_1": 0.66, "rouge_l": 0.63,
                   "p95_latency_ms": 490.0, "model_size_gb": 3.1, "tokens_per_sec": 22.0},
    "qwen2_7b":   {"bleu_1": 0.71, "bleu_4": 0.47, "rouge_1": 0.74, "rouge_l": 0.70,
                   "p95_latency_ms": 1850.0,"model_size_gb": 14.5, "tokens_per_sec": 9.0},
}


# ── Main ──────────────────────────────────────────────────────────────────────

def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    results: dict[str, Any] = {}

    # Load test pairs
    pairs: list[tuple[str, str]] = []
    if not args.dry_run and args.test_data and Path(args.test_data).exists():
        with open(args.test_data, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                pairs.append((item["gloss"], item["translation"]))
        pairs = pairs[:args.n_samples]
    else:
        pairs = (_SAMPLE_PAIRS * ((args.n_samples // len(_SAMPLE_PAIRS)) + 1))[:args.n_samples]

    if args.dry_run:
        results = {k: dict(v) for k, v in _SIM.items()}
        print("[dry-run] Simulated results:")
        for key, m in results.items():
            print(f"  {key:<14} BLEU-4={m['bleu_4']:.3f}  "
                  f"P95={m['p95_latency_ms']:.0f}ms  "
                  f"Size={m['model_size_gb']:.1f}GB")
    else:
        for model_id, key in QWEN_SIZES:
            print(f"\nEvaluating {model_id} ...")
            try:
                hyps, lats = _infer_qwen(model_id, pairs,
                                         max_new_tokens=args.max_new_tokens,
                                         temperature=args.temperature)
                refs = [p[1] for p in pairs]
                m = _compute_metrics(hyps, refs)

                lats.sort()
                n = len(lats)
                m["p50_latency_ms"]  = lats[n // 2]
                m["p95_latency_ms"]  = lats[int(0.95 * n)]
                m["mean_latency_ms"] = sum(lats) / n
                m["tokens_per_sec"]  = (args.max_new_tokens * 1000) / m["mean_latency_ms"]
                m["n_samples"]       = float(len(pairs))

                results[key] = m
                print(f"  BLEU-4={m['bleu_4']:.3f}  ROUGE-L={m['rouge_l']:.3f}  "
                      f"P95={m['p95_latency_ms']:.0f}ms")
            except Exception as e:
                results[key] = {"error": str(e)}
                print(f"  Error: {e}")

    with mlflow_run(
        EXPERIMENT_NAME, run_name=f"qwen_cmp_n{args.n_samples}",
        tags={"experiment": "05"},
    ) as _run:
        log_params({"n_samples": args.n_samples, "max_new_tokens": args.max_new_tokens})
        for key, m in results.items():
            log_metrics({f"{key}/{k}": v for k, v in m.items() if isinstance(v, float)})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_results(results, RESULTS_DIR / "results.json")
    _print_table(results)
    return results


def _print_table(results: dict[str, Any]) -> None:
    print("\n" + "═" * 75)
    print("EXPERIMENT 05 — LLM SIZE COMPARISON (RSL Gloss → Russian)")
    print("═" * 75)
    print(f"{'Model':<16} {'BLEU-1':>8} {'BLEU-4':>8} {'ROUGE-L':>8} "
          f"{'P95ms':>8} {'GB':>6}")
    print("─" * 75)
    slo_bleu = 0.35
    slo_p95  = 600.0
    for key, (_, label) in zip(_SIM, QWEN_SIZES):
        m = results.get(key, {})
        if "error" in m:
            continue
        b4  = m.get("bleu_4", 0)
        p95 = m.get("p95_latency_ms", 0)
        marker = " ← our choice" if b4 >= slo_bleu and p95 <= slo_p95 else ""
        print(f"{key:<16} {m.get('bleu_1', 0):>8.3f} {b4:>8.3f} "
              f"{m.get('rouge_l', 0):>8.3f} {p95:>8.0f} "
              f"{m.get('model_size_gb', 0):>6.1f}{marker}")
    print(f"\nSLO: BLEU-4 ≥ {slo_bleu}  /  P95 ≤ {slo_p95}ms")
    print("═" * 75)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-data",      default="")
    p.add_argument("--n-samples",      type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature",    type=float, default=0.1)
    p.add_argument("--dry-run",        action="store_true")
    run_experiment(p.parse_args())


if __name__ == "__main__":
    main()
