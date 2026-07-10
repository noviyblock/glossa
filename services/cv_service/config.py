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
# rtmlib Wholebody mode — 'lightweight' (current default, RTMDet-nano +
# RTMPose-t, fastest/least accurate), 'balanced' (RTMDet-m + RTMPose-m,
# meaningfully more accurate hand/finger localization at moderate extra
# cost), 'performance' (heaviest/most accurate, likely too slow for
# real-time on CPU). NOT benchmarked in this environment (no network access
# to download the balanced/performance ONNX weights) — worth an empirical
# CPU-latency-vs-accuracy comparison on the actual deployment host before
# switching away from the default.
RTMLIB_MODE   = os.getenv("RTMLIB_MODE",   "lightweight")
RTMLIB_DEVICE = os.getenv("RTMLIB_DEVICE", "cpu")
CLASS_MAP_PATH   = os.getenv("CLASS_MAP_PATH", "/data/idx_to_class.json")

# Bbox tracking (keypoint_extractor.py) — run rtmlib on the full frame every
# DET_INTERVAL calls to (re)acquire the person, and on a padded crop around
# the last known keypoint bbox in between (cheaper: both detector and pose
# model cost scale with input resolution). TRACK_CROP_PADDING multiplies the
# last bbox's width/height before cropping, to give a moving person room to
# stay inside the crop until the next full-frame redetect.
DET_INTERVAL       = int(os.getenv("DET_INTERVAL", "5"))
TRACK_CROP_PADDING = float(os.getenv("TRACK_CROP_PADDING", "1.8"))

# Joints with confidence below this are hard-zeroed (x=y=score=0) to match
# DWPose's implicit "not detected = absent" semantics — see
# keypoint_extractor.py::_zero_low_confidence_joints. Matches the client's
# existing _SkeletonPainter._minScore rendering cutoff.
LOW_CONF_ZERO_THRESHOLD = float(os.getenv("LOW_CONF_ZERO_THRESHOLD", "0.3"))

# Hands get their OWN, lower threshold. Live production logs showed
# nonzero_kp dropping to ~15/75 mid-gesture — i.e. most of both hands zeroed
# out during actual signing motion. RTMW is a single-shot per-frame model
# with no temporal prior, so fast hand motion (motion blur) legitimately
# lowers its confidence even when the hand IS genuinely there and roughly
# correctly localized — unlike DWPose's separate hand detector, whose
# "undetected" case is closer to true absence. 0.3 was tuned for "is this
# joint absent", not "is this moving hand slightly less certain" — hands
# are the entire signal for RSL, so we tolerate more per-frame noise on them
# rather than lose them outright. Needs empirical tuning against real
# camera footage (see scripts/analyze_cv_logs.py for the nonzero_kp/region
# breakdown to calibrate this against).
HAND_LOW_CONF_ZERO_THRESHOLD = float(os.getenv("HAND_LOW_CONF_ZERO_THRESHOLD", "0.15"))

# One Euro Filter (keypoint_smoother.py) — adaptive temporal smoothing applied
# before keypoints reach the classifier. mincutoff: baseline smoothing at
# near-zero speed (lower = smoother when still). beta: how much fast motion
# cuts through unsmoothed (higher = less lag on real gestures, more jitter
# passthrough). Coordinates are normalised [0,1], not pixels — tuned an
# order of magnitude below typical mouse-cursor presets. Best-effort
# defaults, tune empirically.
SMOOTHING_ENABLED = os.getenv("SMOOTHING_ENABLED", "true").lower() == "true"
SMOOTH_MINCUTOFF  = float(os.getenv("SMOOTH_MINCUTOFF", "1.0"))
SMOOTH_BETA       = float(os.getenv("SMOOTH_BETA", "0.3"))
SMOOTH_DCUTOFF    = float(os.getenv("SMOOTH_DCUTOFF", "1.0"))

# Gesture segmentation (auto onset/offset from hand motion, replaces manual
# hold-to-sign button) — thresholds are best-effort defaults, tune empirically
# against the live camera using the periodic DIAG log's last_activity value.
GESTURE_ONSET_THRESHOLD    = float(os.getenv("GESTURE_ONSET_THRESHOLD", "0.06"))
GESTURE_OFFSET_THRESHOLD   = float(os.getenv("GESTURE_OFFSET_THRESHOLD", "0.03"))
GESTURE_ONSET_FRAMES       = int(os.getenv("GESTURE_ONSET_FRAMES", "3"))
# Stillness (in processed frames) required after a gesture before the FINAL
# classification fires — a straight latency/accuracy tradeoff: lower fires
# sooner but risks cutting off a gesture that has a brief internal pause;
# higher is safer but adds tail latency. Lowered from 8→6 (was ~0.7-1.3s of
# stillness, now ~0.5-1.0s at realistic frame cadence) now that
# GestureSegmenter also emits an early PREVIEW classification once the
# segment crosses GESTURE_MIN_FRAMES (see gesture_segmenter.py), which
# already gives the user fast feedback independent of this value — so this
# knob now only affects how soon the FINAL (corrected) result lands, not the
# perceived responsiveness.
GESTURE_OFFSET_FRAMES      = int(os.getenv("GESTURE_OFFSET_FRAMES", "6"))
# Matches HAND_LOW_CONF_ZERO_THRESHOLD, not the old body threshold — this
# checks mean confidence of the SAME (now less aggressively zeroed) hand
# keypoints; leaving this at the old 0.3 would make the segmenter think
# "no hands" even on frames the extractor now keeps as valid low-confidence
# hand data, causing spurious offset-triggering mid-gesture.
GESTURE_HAND_PRESENCE_CONF = float(os.getenv("GESTURE_HAND_PRESENCE_CONF", "0.15"))
GESTURE_PREROLL_FRAMES     = int(os.getenv("GESTURE_PREROLL_FRAMES", "5"))
GESTURE_MIN_FRAMES         = int(os.getenv("GESTURE_MIN_FRAMES", "8"))
GESTURE_MAX_FRAMES         = int(os.getenv("GESTURE_MAX_FRAMES", "150"))

# Test-time augmentation (gesture_classifier.py::predict_top3_tta) — averages
# softmax over the window plus jittered copies using the SAME sigma=0.02
# spatial_jitter the model was trained with (colab_glossa_01a). Applied only
# to FINAL classifications (not previews, to keep those fast) — see main.py.
# n=3 costs ~2 extra OpenVINO INT8 passes (~7-10ms each per prior benchmark,
# negligible next to the ~90ms/frame extraction cost).
TTA_ENABLED      = os.getenv("TTA_ENABLED", "true").lower() == "true"
TTA_VARIANTS     = int(os.getenv("TTA_VARIANTS", "3"))
TTA_JITTER_SIGMA = float(os.getenv("TTA_JITTER_SIGMA", "0.02"))

REDIS_URL        = os.getenv("REDIS_URL", "redis://redis:6379")
CV_STREAM_OUT    = os.getenv("CV_STREAM_OUT", "cv:results")
SERVICE_PORT     = int(os.getenv("CV_SERVICE_PORT", "8001"))
