import os

REDIS_URL       = os.getenv("REDIS_URL", "redis://redis:6379")
CV_SERVICE_URL  = os.getenv("CV_SERVICE_URL",  "http://cv-service:8001")
ASR_SERVICE_URL = os.getenv("ASR_SERVICE_URL", "http://asr-service:8002")
NLP_SERVICE_URL = os.getenv("NLP_SERVICE_URL", "http://nlp-service:8003")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://tts-service:8004")
SERVICE_PORT    = int(os.getenv("GATEWAY_PORT", "8000"))
SESSION_TTL     = int(os.getenv("GATEWAY_SESSION_TTL", "300"))   # 5 min
HTTP_TIMEOUT    = float(os.getenv("GATEWAY_HTTP_TIMEOUT", "30"))

# Sentence accumulation — buffer completed gestures per session and translate
# them as one sequence instead of calling the LLM after every single gesture.
# A sentence flushes when either fires: (a) no new gesture for
# SENTENCE_PAUSE_SECONDS, or (b) the buffer reaches MAX_SENTENCE_GLOSSES.
SENTENCE_PAUSE_SECONDS = float(os.getenv("SENTENCE_PAUSE_SECONDS", "2.5"))
MAX_SENTENCE_GLOSSES   = int(os.getenv("MAX_SENTENCE_GLOSSES", "8"))
