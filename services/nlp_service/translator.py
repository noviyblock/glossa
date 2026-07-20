from __future__ import annotations

import logging
import os
import re
import time

import torch
from prometheus_client import Histogram
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CLASS_TO_IDX_PATH, DEVICE_MAP, MAX_NEW_TOKENS, MODEL_PATH
from vocabulary import GlossVocabulary

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

# Sequence-level ("T9-style") disambiguation: instead of picking the best
# candidate for one gesture in isolation, pick the most coherent combination
# across an entire buffered sentence of gestures at once.
SEQ_TOPK_SYSTEM_PROMPT = (
    "Ты — переводчик русского жестового языка (РЖЯ) в русский текст. \n"
    "Тебе даётся последовательность позиций — одна на каждый распознанный "
    "жест, по порядку. Для каждой позиции распознаватель предлагает "
    "несколько вариантов глоссы с вероятностями — распознавание не идеально: "
    "правильный вариант не всегда тот, что с наибольшей вероятностью, а "
    "иногда жест мог быть распознан ошибочно или пропущен. \n"
    "Выбери наиболее связную по смыслу последовательность вариантов "
    "(учитывая порядок SOV и общий смысл всего предложения, а не каждую "
    "позицию по отдельности) и переведи её в грамматически правильное "
    "русское предложение. Если какая-то позиция явно не вписывается по "
    "смыслу ни одним из вариантов, можешь её проигнорировать, если без неё "
    "предложение получается более естественным. \n"
    "Если дан контекст (предыдущие фразы диалога), используй его для "
    "связности с разговором. \n"
    "ВАЖНО: отвечай ТОЛЬКО на русском языке. \n"
    "Только перевод, без пояснений."
)

# Reverse: Russian text → RSL gloss sequence — constrained to the actual
# 200-word vocabulary (see vocabulary.py::GlossVocabulary), not free
# generation. The old version of this prompt taught the model (via
# few-shot examples) to invent whatever gloss-shaped text seemed to fit —
# fine for an unconstrained vocabulary, but this project's downstream
# consumers (SignVideoAssembler, SkeletonSequenceProvider) can only ever
# render the 200 words they actually have clips/keypoints for, so free
# invention just produces confident-looking words nothing can render and
# that were never actually said (see punch-list: "бабочка" ->
# "бабочка приближается", "с днем рождения" -> "верный добрый друг").
# {vocab} is filled in per-Translator-instance in __init__, once, from the
# real vocabulary file, not hand-maintained here.
REVERSE_SYSTEM_PROMPT_TEMPLATE = (
    "Ты — переводчик русского жестового языка (РЖЯ). "
    "У тебя есть строго ограниченный словарь из {n} глосс (жестов) РЖЯ — "
    "вот полный список, по одной глоссе на строку:\n"
    "{vocab}\n\n"
    "Определи, какие из ЭТИХ глосс (и только из них) по смыслу присутствуют "
    "во входном предложении. Выведи их через пробел, ЗАГЛАВНЫМИ буквами, "
    "в порядке SOV (субъект → объект → глагол), используя ТОЧНОЕ написание "
    "из списка выше — не изменяй его и не склоняй.\n"
    "ВАЖНО: если для какого-то смысла из предложения в списке НЕТ "
    "подходящей глоссы — просто пропусти его. Не придумывай глосс, "
    "которых нет в списке выше, даже если кажется, что похожее слово "
    "должно быть.\n"
    "Отвечай ТОЛЬКО глоссами через пробел, без пояснений. Если ни одна "
    "глосса из списка не подходит — верни пустую строку."
)

_CHINESE_RE = re.compile(r"[一-鿿]+")


class Translator:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        device_map: str = DEVICE_MAP,
        max_new_tokens: int = MAX_NEW_TOKENS,
        class_map_path: str = CLASS_TO_IDX_PATH,
    ) -> None:
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"NLP model directory not found: {model_path}\n"
                "Run `make models` or `dvc pull` and check the docker-compose "
                "volume mount for /models."
            )

        # Small file (a few KB), always expected to exist -- unlike
        # tts_service's SkeletonSequenceProvider (a large DVC dataset,
        # genuinely optional), there's no reasonable degraded mode for
        # translate_reverse without its own vocabulary: it would either
        # crash on every call or silently fall back to unconstrained free
        # generation, the exact hallucination problem this exists to fix.
        self._vocab = GlossVocabulary(class_map_path)
        self._reverse_system_prompt = REVERSE_SYSTEM_PROMPT_TEMPLATE.format(
            n=len(self._vocab.words), vocab=self._vocab.prompt_list(),
        )
        logger.info("GlossVocabulary loaded: %d glosses from %s", len(self._vocab.words), class_map_path)

        # bandit B615 (unpinned Hub revision) doesn't apply here: model_path
        # is required to already be a local directory (checked above), and
        # local_files_only=True means from_pretrained() never resolves a
        # revision against the Hub at all -- there's no network path for a
        # tampered/moved "main" ref to matter on.
        logger.info("Loading tokenizer from %s", model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            model_path, trust_remote_code=True, local_files_only=True
        )

        logger.info("Loading model (bfloat16, device_map=%s)", device_map)
        self._model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=True,
        )
        self._model.eval()
        self._max_new_tokens = max_new_tokens
        # `device_map` above is the REQUESTED placement — accelerate can
        # silently resolve "auto"/"cuda" to CPU (e.g. CUDA not visible
        # inside the container) with no exception. Log what actually
        # happened, not just what was asked for, so "is NLP really on GPU"
        # is answerable from `docker compose logs nlp-service` alone.
        resolved_placement = getattr(self._model, "hf_device_map", None) \
            or {"": str(next(self._model.parameters()).device)}
        logger.info(
            "Model ready — torch.cuda.is_available=%s cuda_device_count=%d resolved_placement=%s",
            torch.cuda.is_available(), torch.cuda.device_count(), resolved_placement,
        )

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
            # Last 4, not 2 -- history is now shared across both translation
            # directions (see orchestrator.py's process_frame/translate_sync),
            # tagged by speaker ("Слышащий: .../Жестикулирующий: ..."), so 2
            # entries could be just one side's last two turns with no reply
            # in view at all. 4 keeps roughly one full exchange visible.
            ctx = "\n".join(context[-4:])
            lines = f"Предыдущие фразы диалога:\n{ctx}\n\nВарианты глосс для текущего жеста:\n{lines}"
        return self.generate(TOPK_SYSTEM_PROMPT, lines)

    def translate_reverse(self, russian_text: str) -> str:
        """Russian sentence → RSL gloss sequence (SOV, uppercase), grounded
        in the actual 200-word vocabulary. The prompt (see
        REVERSE_SYSTEM_PROMPT_TEMPLATE, built once in __init__) instructs
        the model to only pick from that list, but constrain() is the real
        guarantee -- it deterministically drops anything the model output
        that isn't a literal vocabulary match, rather than trusting a
        small model to follow the instruction perfectly every time."""
        raw = self.generate(self._reverse_system_prompt, russian_text)
        constrained = self._vocab.constrain(raw)
        if raw.strip() and not constrained:
            logger.info("translate_reverse: model output %r matched no vocabulary entry", raw[:80])
        return constrained

    def translate_sequence_topk(
        self, positions: list[list[dict]], context: list[str] | None = None
    ) -> str:
        """Translate a whole buffered sentence of gestures at once.

        `positions` is a list of top-k candidate lists, one per recognised
        gesture in temporal order — generalizes translate_topk (single
        gesture) to a full sentence so the LLM can disambiguate using the
        whole sequence's coherence rather than one gesture in isolation.
        """
        lines = "\n".join(
            f"Позиция {i + 1}: " + ", ".join(f"{h['gloss']} ({h['prob']:.2f})" for h in pos)
            for i, pos in enumerate(positions)
        )
        if context:
            ctx = "\n".join(context[-4:])  # see translate_topk's comment on why 4, not 2
            lines = f"Предыдущие фразы диалога:\n{ctx}\n\nПозиции распознанных жестов:\n{lines}"
        return self.generate(SEQ_TOPK_SYSTEM_PROMPT, lines)
