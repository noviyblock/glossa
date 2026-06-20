import os

WHISPER_MODEL    = os.getenv("ASR_WHISPER_MODEL", "base")
DEVICE           = os.getenv("ASR_DEVICE", "cpu")
COMPUTE_TYPE     = os.getenv("ASR_COMPUTE_TYPE", "int8")
LANGUAGE         = "ru"
BEAM_SIZE        = int(os.getenv("ASR_BEAM_SIZE", "5"))
VAD_FILTER       = os.getenv("ASR_VAD_FILTER", "true").lower() == "true"
SERVICE_PORT     = int(os.getenv("ASR_SERVICE_PORT", "8002"))
# Max audio chunk size accepted via REST (bytes)
MAX_AUDIO_BYTES  = int(os.getenv("ASR_MAX_AUDIO_BYTES", str(10 * 1024 * 1024)))
