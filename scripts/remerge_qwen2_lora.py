"""Re-merge the Qwen2-1.5B LoRA adapter into full-precision weights.

The existing models/qwen2_merged_qwen2_1.5b/ was saved while the base model
was still loaded via bitsandbytes 4-bit (model.save_pretrained() called on a
BitsAndBytesConfig-loaded model without dequantizing first). The resulting
safetensors file contains raw NF4-packed tensors under normal parameter
names, which is why transformers either demands bitsandbytes (with the stray
quantization_config) or raises a weight-shape mismatch once that config is
stripped. This script redoes the merge correctly: load the base model in
bfloat16 (no quantization), attach the LoRA adapter, merge, and save.

Usage:
    pip install transformers peft accelerate safetensors
    python scripts/remerge_qwen2_lora.py \
        --adapter models/qwen2_lora_qwen2_1.5b_slovo_synth \
        --out models/qwen2_merged_qwen2_1.5b
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"Loading base model {args.base} in bfloat16 (no quantization)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    print(f"Attaching LoRA adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("Merging adapter into base weights...")
    model = model.merge_and_unload()

    print(f"Saving merged full-precision model to {args.out}...")
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)

    print("Done. Verify with: python -c \"import json; "
          f"print(json.load(open('{args.out}/config.json')).get('quantization_config'))\"")
    print("Expected output: None")


if __name__ == "__main__":
    main()
