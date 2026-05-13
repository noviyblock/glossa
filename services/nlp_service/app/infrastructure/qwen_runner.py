"""QwenRunner — production LLM backend with CTranslate2 + streaming.

Implements LLMPort.  CPU/GPU auto-detection, 4/8-bit quantization,
streaming token generation via TextIteratorStreamer (HuggingFace).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path

from glossa_common.logging import get_logger

from ..domain.interfaces import LLMPort

logger = get_logger(__name__)

RSL_SYSTEM_PROMPT = """Ты — специализированная система перевода русского жестового языка (РЖЯ).
Отвечай точно, кратко, по существу. Не добавляй пояснения если не просят."""


class QwenRunner(LLMPort):
    """Qwen2-1.5B-Instruct inference with streaming + quantization support."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        cpu_threads: int = 4,
    ) -> None:
        self._model_path = Path(model_path)
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._load_in_4bit = load_in_4bit
        self._load_in_8bit = load_in_8bit
        self._cpu_threads = cpu_threads
        self._model = None
        self._tokenizer = None

    async def ensure_loaded(self) -> None:
        if self._model is None:
            await asyncio.to_thread(self._load)

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
    ) -> tuple[str, int]:
        await self.ensure_loaded()
        max_t = max_tokens or self._max_new_tokens
        temp = temperature if temperature is not None else self._temperature
        return await asyncio.to_thread(self._generate_sync, messages, max_t, temp, stop)

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Yield decoded tokens as they are generated (using TextIteratorStreamer)."""
        await self.ensure_loaded()
        max_t = max_tokens or self._max_new_tokens
        temp = temperature if temperature is not None else self._temperature

        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=128)
        loop = asyncio.get_event_loop()

        def _stream_worker() -> None:
            try:
                for token in self._stream_sync(messages, max_t, temp):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.get_event_loop().run_in_executor(None, _stream_worker)

        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

    async def detect_language(self, text: str) -> str:
        """Lightweight language detection via short LLM probe."""
        messages = [{"role": "user", "content": f"Язык текста (только код BCP-47): «{text[:100]}»"}]
        result, _ = await self.chat(messages, max_tokens=8, temperature=0.0, stop=["\n"])
        return result.strip().lower() or "ru"

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if self._cpu_threads > 0:
            torch.set_num_threads(self._cpu_threads)

        quant_cfg = None
        if self._load_in_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self._load_in_8bit:
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self._model_path), trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self._model_path),
            device_map=self._device if self._device != "cpu" else None,
            torch_dtype="auto",
            trust_remote_code=True,
            quantization_config=quant_cfg,
        )
        if self._device == "cpu":
            self._model = self._model.to("cpu")

        logger.info(
            "qwen_loaded",
            model=self._model_path.name,
            device=self._device,
            quantized=self._load_in_4bit or self._load_in_8bit,
            cpu_threads=self._cpu_threads,
        )

    def _build_chat_text(self, messages: list[dict]) -> str:
        # Ensure system message is always first
        has_system = any(m["role"] == "system" for m in messages)
        if not has_system:
            messages = [{"role": "system", "content": RSL_SYSTEM_PROMPT}] + list(messages)
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generate_sync(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
    ) -> tuple[str, int]:
        import torch

        text = self._build_chat_text(messages)
        inputs = self._tokenizer(text, return_tensors="pt")
        if self._device != "cpu":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Build stop token ids
        stop_ids = []
        if stop:
            for s in stop:
                ids = self._tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    stop_ids.append(ids[0])

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-4),
                top_p=self._top_p,
                do_sample=temperature > 0.01,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=[self._tokenizer.eos_token_id] + stop_ids,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        result = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        return result, len(generated)

    def _stream_sync(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ):
        """Yield decoded string chunks using TextIteratorStreamer."""
        import torch
        from transformers import TextIteratorStreamer
        import threading

        text = self._build_chat_text(messages)
        inputs = self._tokenizer(text, return_tensors="pt")
        if self._device != "cpu":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "temperature": max(temperature, 1e-4),
            "top_p": self._top_p,
            "do_sample": temperature > 0.01,
            "pad_token_id": self._tokenizer.eos_token_id,
        }

        thread = threading.Thread(target=lambda: self._model.generate(**gen_kwargs))
        thread.start()

        for chunk in streamer:
            if chunk:
                yield chunk

        thread.join()
