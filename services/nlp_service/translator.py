from __future__ import annotations

import logging
import os
import re
import time

import torch
from prometheus_client import Histogram
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEVICE_MAP, MAX_NEW_TOKENS, MODEL_PATH

logger = logging.getLogger(__name__)

MODEL_INFERENCE_LATENCY = Histogram(
    "glossa_model_inference_latency_seconds", "Model inference latency per service",
    ["service", "model"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

SYSTEM_PROMPT = (
    "Ты — переводчик русского жестового языка (РЖЯ). \n"
    "Тебе даётся последовательность глосс РЖЯ \n"
    "(записи жестов заглавными буквами в порядке SOV: субъект → объект → глагол). \n"
    "Переведи её в грамматически правильное русское предложение. \n"
    "ВАЖНО: отвечай ТОЛЬКО на русском языке. \n"
    "Отвечай только переводом, без пояснений."
)

TOPK_SYSTEM_PROMPT = (
    "Ты — переводчик русского жестового языка (РЖЯ). \n"
    "Система распознавания предлагает варианты глосс с вероятностями — "
    "не всегда верен именно вариант с наибольшей вероятностью. \n"
    "Если дан контекст (предыдущие фразы диалога), используй его, чтобы "
    "выбрать вариант, наиболее связный по смыслу с предыдущим разговором, "
    "а не просто самый вероятный по отдельности. \n"
    "Выбери наиболее связный вариант и переведи в русское предложение. \n"
    "ВАЖНО: отвечай ТОЛЬКО на русском языке. \n"
    "Только перевод, без пояснений."
)

# Reverse: Russian text → RSL gloss sequence
REVERSE_SYSTEM_PROMPT = (
    "Ты — переводчик русского жестового языка (РЖЯ). "
    "Преобразуй русское предложение в последовательность глосс РЖЯ: "
    "ключевые слова ЗАГЛАВНЫМИ буквами через пробел, порядок SOV, служебные слова опускай.\n"
    "Примеры:\n"
    "Вход: Я хочу пить воду. Выход: Я ВОДА ПИТЬ\n"
    "Вход: Привет, как дела? Выход: ПРИВЕТ КАК ДЕЛА\n"
    "Вход: Мне нужна помощь. Выход: Я ПОМОЩЬ НУЖЕН\n"
    "Вход: Спасибо большое! Выход: СПАСИБО БОЛЬШОЙ\n"
    "Вход: Где туалет? Выход: ТУАЛЕТ ГДЕ\n"
    "Вход: Меня зовут Анна. Выход: Я ИМЯ АННА\n"
    "ВАЖНО: отвечай ТОЛЬКО глоссами заглавными буквами через пробел. "
    "Никаких пояснений, никаких предложений — только глоссы."
)

_CHINESE_RE = re.compile(r"[一-鿿]+")


class Translator:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        device_map: str = DEVICE_MAP,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> None:
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"NLP model directory not found: {model_path}\n"
                "Run `make models` or `dvc pull` and check the docker-compose "
                "volume mount for /models."
            )

        logger.info("Loading tokenizer from %s", model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )

        logger.info("Loading model (bfloat16, device_map=%s)", device_map)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=True,
        )
        self._model.eval()
        self._max_new_tokens = max_new_tokens
        logger.info("Model ready")

    # ------------------------------------------------------------------ #

    def generate(self, system_prompt: str, user_message: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        MODEL_INFERENCE_LATENCY.labels(service="nlp-service", model="qwen2-1.5b").observe(time.perf_counter() - start)

        new_tokens = outputs[0][input_len:]
        result = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return _CHINESE_RE.sub("", result).strip()

    def translate(self, gloss_sequence: str) -> str:
        return self.generate(SYSTEM_PROMPT, gloss_sequence)

    def translate_topk(self, hypotheses: list[dict], context: list[str] | None = None) -> str:
        lines = "\n".join(
            f"{h['gloss']} (вероятность: {h['prob']:.2f})" for h in hypotheses
        )
        if context:
            ctx = "\n".join(context[-2:])  # last 1-2 turns — enough for local coherence, keeps prompt short
            lines = f"Предыдущие фразы диалога:\n{ctx}\n\nВарианты глосс для текущего жеста:\n{lines}"
        return self.generate(TOPK_SYSTEM_PROMPT, lines)

    def translate_reverse(self, russian_text: str) -> str:
        """Russian sentence → RSL gloss sequence (SOV, uppercase)."""
        return self.generate(REVERSE_SYSTEM_PROMPT, russian_text)
