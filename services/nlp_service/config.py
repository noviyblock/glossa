import os

# MODEL_PATH (no prefix) or NLP_MODEL_PATH — mirrors docker-compose env
MODEL_PATH     = os.getenv("MODEL_PATH", os.getenv("NLP_MODEL_PATH", "/models/qwen2_merged_qwen2_1.5b"))
DEVICE_MAP     = os.getenv("NLP_DEVICE", "auto")
MAX_NEW_TOKENS = int(os.getenv("NLP_MAX_NEW_TOKENS", "128"))
CACHE_SIZE     = int(os.getenv("NLP_CACHE_SIZE", "500"))
SERVICE_PORT   = int(os.getenv("NLP_SERVICE_PORT", "8003"))
