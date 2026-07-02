import os

WINDOW_SIZE      = int(os.getenv("WINDOW_SIZE", "64"))
NUM_JOINTS       = int(os.getenv("NUM_JOINTS", "75"))
IN_CHANNELS      = int(os.getenv("IN_CHANNELS", "3"))
NUM_CLASSES      = int(os.getenv("NUM_CLASSES", "200"))
WINDOW_STRIDE    = int(os.getenv("WINDOW_STRIDE", "5"))
EARLY_EMIT       = int(os.getenv("EARLY_EMIT", "10"))
CONF_THRESHOLD   = float(os.getenv("CONF_THRESHOLD", "0.7"))
ONNX_MOBILE_PATH = os.getenv("ONNX_MOBILE_PATH", "/models/gesture_classifier_mobile.onnx")
OV_XML_PATH      = os.getenv("OV_XML_PATH", "/models/stgcn_topk_int8/stgcn_topk_int8.xml")
NORM_STATS_PATH  = os.getenv("NORM_STATS_PATH", "/models/norm_stats.npz")
RTMLIB_MODE   = os.getenv("RTMLIB_MODE",   "lightweight")
RTMLIB_DEVICE = os.getenv("RTMLIB_DEVICE", "cpu")
CLASS_MAP_PATH   = os.getenv("CLASS_MAP_PATH", "/data/idx_to_class.json")

# Gesture segmentation (auto onset/offset from hand motion, replaces manual
# hold-to-sign button) — thresholds are best-effort defaults, tune empirically
# against the live camera using the periodic DIAG log's last_activity value.
GESTURE_ONSET_THRESHOLD    = float(os.getenv("GESTURE_ONSET_THRESHOLD", "0.06"))
GESTURE_OFFSET_THRESHOLD   = float(os.getenv("GESTURE_OFFSET_THRESHOLD", "0.03"))
GESTURE_ONSET_FRAMES       = int(os.getenv("GESTURE_ONSET_FRAMES", "3"))
GESTURE_OFFSET_FRAMES      = int(os.getenv("GESTURE_OFFSET_FRAMES", "8"))
GESTURE_HAND_PRESENCE_CONF = float(os.getenv("GESTURE_HAND_PRESENCE_CONF", "0.3"))
GESTURE_PREROLL_FRAMES     = int(os.getenv("GESTURE_PREROLL_FRAMES", "5"))
GESTURE_MIN_FRAMES         = int(os.getenv("GESTURE_MIN_FRAMES", "8"))
GESTURE_MAX_FRAMES         = int(os.getenv("GESTURE_MAX_FRAMES", "150"))

REDIS_URL        = os.getenv("REDIS_URL", "redis://redis:6379")
CV_STREAM_OUT    = os.getenv("CV_STREAM_OUT", "cv:results")
SERVICE_PORT     = int(os.getenv("CV_SERVICE_PORT", "8001"))
