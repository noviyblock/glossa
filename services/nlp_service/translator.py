from __future__ import annotations

import logging
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEVICE_MAP, MAX_NEW_TOKENS, MODEL_PATH

logger = logging.getLogger(__name__)

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
    "Система распознавания предлагает варианты глосс с вероятностями. \n"
    "Выбери наиболее связный и переведи в русское предложение. \n"
    "ВАЖНО: отвечай ТОЛЬКО на русском языке. \n"
    "Только перевод, без пояснений."
)

# Reverse: Russian text → RSL gloss sequence
REVERSE_SYSTEM_PROMPT = (
    "Ты — лингвист русского жестового языка (РЖЯ). \n"
    "Тебе даётся русское предложение. \n"
    "Преобразуй его в последовательность глосс РЖЯ: \n"
    "ключевые слова заглавными буквами в порядке SOV (субъект → объект → глагол), \n"
    "служебные слова опускай. \n"
    "ВАЖНО: отвечай ТОЛЬКО глоссами заглавными буквами через пробел. \n"
    "Только глоссы, без пояснений."
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

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        result = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return _CHINESE_RE.sub("", result).strip()

    def translate(self, gloss_sequence: str) -> str:
        return self.generate(SYSTEM_PROMPT, gloss_sequence)

    def translate_topk(self, hypotheses: list[dict]) -> str:
        lines = "\n".join(
            f"{h['gloss']} (вероятность: {h['prob']:.2f})" for h in hypotheses
        )
        return self.generate(TOPK_SYSTEM_PROMPT, lines)

    def translate_reverse(self, russian_text: str) -> str:
        """Russian sentence → RSL gloss sequence (SOV, uppercase)."""
        return self.generate(REVERSE_SYSTEM_PROMPT, russian_text)
