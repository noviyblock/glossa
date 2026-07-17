import os

SILERO_REPO      = os.getenv("TTS_SILERO_REPO", "snakers4/silero-models")
SILERO_MODEL     = os.getenv("TTS_SILERO_MODEL", "silero_tts")
SAMPLE_RATE      = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
DEFAULT_SPEAKER  = os.getenv("TTS_DEFAULT_SPEAKER", "xenia")
DEVICE           = os.getenv("TTS_DEVICE", "cpu")
CACHE_SIZE       = int(os.getenv("TTS_CACHE_SIZE", "200"))
SERVICE_PORT     = int(os.getenv("TTS_SERVICE_PORT", "8004"))
TORCH_HOME       = os.getenv("TORCH_HOME", "/models/torch_cache")

# gloss -> видео (see video.py) — эталонные клипы жестов, собранные
# notebooks/colab_glossa_05_gloss_clips.ipynb (по одному на класс, имена
# файлов = числовой индекс класса, все в едином кодеке/разрешении/fps для
# concat demuxer без ре-энкода).
GLOSS_CLIPS_DIR   = os.getenv("GLOSS_CLIPS_DIR", "/models/gloss_clips")
CLASS_TO_IDX_PATH = os.getenv("CLASS_TO_IDX_PATH", "/data/class_to_idx.json")
