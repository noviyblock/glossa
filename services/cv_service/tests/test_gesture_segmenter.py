"""Unit tests for GestureSegmenter's IDLE/ACTIVE state machine.

Pure numpy logic, no external services/models needed -- highest-ROI test
target in cv_service (see punch-list plan, item 6): onset/offset thresholds,
hysteresis, MAX_FRAMES force-flush, preview-once, force_reset/force_flush,
and the "hands disappear mid-gesture" edge case explicitly called out in
gesture_segmenter.py's docstring.

Run: pytest services/cv_service/tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# services/cv_service has no __init__.py (flat-file service) -- same
# sys.path pattern as services/api_gateway/tests/test_orchestrator_resilience.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Every flat-file service has its OWN config.py, imported as the bare name
# `config`. If pytest collects another service's tests in the same process
# first (default testpaths=["services","libs"] does exactly this), that
# service's config.py is already cached under sys.modules['config'] --
# without this pop, `from config import GESTURE_HAND_PRESENCE_CONF` below
# would silently resolve against the WRONG service's config and raise
# ImportError for a name that only exists in cv_service's.
sys.modules.pop("config", None)

from config import (  # noqa: E402
    GESTURE_HAND_PRESENCE_CONF,
    GESTURE_MAX_FRAMES,
    GESTURE_MIN_FRAMES,
    GESTURE_OFFSET_FRAMES,
    GESTURE_OFFSET_THRESHOLD,
    GESTURE_ONSET_FRAMES,
    GESTURE_ONSET_THRESHOLD,
    WINDOW_SIZE,
)
from gesture_segmenter import GestureSegmenter  # noqa: E402

_SHOULDER_R_IDX = 1
_SHOULDER_L_IDX = 4
_LEFT_HAND = slice(33, 54)
_RIGHT_HAND = slice(54, 75)

# Shoulder distance is the scale reference activity is divided by -- fixed
# at 1.0 so displacement deltas translate 1:1 into activity values.
_SHOULDER_DIST = 1.0
# Comfortably above/below the onset/offset thresholds so tests aren't
# sensitive to exact threshold values if config.py's defaults change.
_MOVE_DELTA = GESTURE_ONSET_THRESHOLD * 3.0
_STILL_DELTA = GESTURE_OFFSET_THRESHOLD * 0.1


def _make_frame(hand_x: float, conf: float = 1.0) -> np.ndarray:
    """One raw (75,3) extractor-output frame: fixed shoulders (scale ref),
    a single moving left hand at x=hand_x, right hand absent (tests the
    one-handed-sign path per the module's own per-hand-max rationale)."""
    kp = np.zeros((75, 3), dtype=np.float32)
    kp[_SHOULDER_R_IDX] = [0.0, 0.0, 1.0]
    kp[_SHOULDER_L_IDX] = [_SHOULDER_DIST, 0.0, 1.0]
    kp[_LEFT_HAND, 0] = hand_x
    kp[_LEFT_HAND, 2] = conf
    # right hand stays all-zero/zero-confidence -- absent, not just still
    return kp


def _push_still(seg: GestureSegmenter, n: int, x: float = 0.0):
    result = (None, False, False)
    for _ in range(n):
        result = seg.push(_make_frame(x))
    return result


def _push_moving(seg: GestureSegmenter, n: int, start_x: float = 0.0):
    """Push n frames, each _MOVE_DELTA further than the last."""
    result = (None, False, False)
    x = start_x
    for _ in range(n):
        x += _MOVE_DELTA
        result = seg.push(_make_frame(x))
    return result


def test_idle_stays_idle_without_motion():
    seg = GestureSegmenter("s1")
    for _ in range(20):
        window, active, preview = seg.push(_make_frame(0.0))
        assert window is None
        assert active is False
        assert preview is False
    assert seg.debug_state["state"] == "IDLE"


def test_onset_requires_debounce_streak():
    seg = GestureSegmenter("s2")
    _push_still(seg, 2)  # establish _prev_kp, activity=0 on the very first frame

    # Fewer than GESTURE_ONSET_FRAMES consecutive moving frames -> still IDLE.
    # Each frame must move further than the last (cumulative x), otherwise
    # displacement vs the previous (also-moved) frame is zero and the
    # onset streak never builds.
    x = 0.0
    for _ in range(GESTURE_ONSET_FRAMES - 1):
        x += _MOVE_DELTA
        window, active, _ = seg.push(_make_frame(x))
        assert active is False
    assert seg.debug_state["state"] == "IDLE"

    # One more moving frame reaches the streak -> transitions to ACTIVE.
    x += _MOVE_DELTA
    window, active, preview = seg.push(_make_frame(x))
    assert active is True
    assert window is None  # onset itself never returns a window


def test_offset_after_stillness_returns_window_of_correct_shape():
    seg = GestureSegmenter("s3")
    _push_still(seg, 2)
    _push_moving(seg, GESTURE_ONSET_FRAMES + GESTURE_MIN_FRAMES)
    assert seg.debug_state["state"] == "ACTIVE"

    last_x = seg._segment[-1][33, 0] if seg._segment else 0.0
    window = None
    active = True
    for _ in range(GESTURE_OFFSET_FRAMES):
        window, active, _ = seg.push(_make_frame(last_x + _STILL_DELTA))

    assert active is False
    assert window is not None
    assert window.shape == (WINDOW_SIZE, 75, 3)
    assert seg.debug_state["state"] == "IDLE"


def test_short_burst_is_discarded_not_classified():
    """force_flush() on a segment shorter than GESTURE_MIN_FRAMES must
    discard (return None), not hand a too-short/duplicated-by-resample
    window to the classifier."""
    seg = GestureSegmenter("s4")
    _push_still(seg, 2)
    _push_moving(seg, GESTURE_ONSET_FRAMES)  # segment now = preroll + onset frames only
    assert len(seg._segment) < GESTURE_MIN_FRAMES, (
        "test setup assumption broke: onset alone already reached MIN_FRAMES"
    )
    window = seg.force_flush()
    assert window is None
    assert seg.debug_state["state"] == "IDLE"


def test_preview_emitted_exactly_once():
    seg = GestureSegmenter("s5")
    _push_still(seg, 2)

    previews = []
    x = 0.0
    for _ in range(GESTURE_ONSET_FRAMES + GESTURE_MIN_FRAMES + 5):
        x += _MOVE_DELTA
        window, active, is_preview = seg.push(_make_frame(x))
        if is_preview:
            previews.append(window)

    assert len(previews) == 1
    assert previews[0] is not None
    assert previews[0].shape == (WINDOW_SIZE, 75, 3)
    # Preview does not reset accumulation -- still ACTIVE, segment kept growing.
    assert seg.debug_state["state"] == "ACTIVE"


def test_max_frames_force_flushes_without_leaving_active():
    seg = GestureSegmenter("s6")
    _push_still(seg, 2)
    x = 0.0
    for _ in range(GESTURE_ONSET_FRAMES):
        x += _MOVE_DELTA
        _, active, _ = seg.push(_make_frame(x))
    assert seg.debug_state["state"] == "ACTIVE"

    # Now already ACTIVE -- continuous motion, never stopping, must eventually
    # hit the GESTURE_MAX_FRAMES safety flush without ever dropping to IDLE.
    windows = []
    for _ in range(GESTURE_MAX_FRAMES + 10):
        x += _MOVE_DELTA
        window, active, _ = seg.push(_make_frame(x))
        assert active is True  # never drops out of ACTIVE from a safety flush
        if window is not None:
            windows.append(window)

    assert len(windows) >= 1  # at least the MAX_FRAMES safety flush fired
    assert seg.debug_state["state"] == "ACTIVE"  # still accumulating a fresh segment


def test_hands_disappearing_counts_as_offset_even_mid_motion():
    """A hand that vanishes from tracking (confidence -> 0) must count toward
    the offset streak even if the last known displacement was large -- this
    is the (not hands_present) branch in _push_active, distinct from the
    activity<=threshold branch."""
    seg = GestureSegmenter("s7")
    _push_still(seg, 2)
    _push_moving(seg, GESTURE_ONSET_FRAMES + 2)
    assert seg.debug_state["state"] == "ACTIVE"

    window = None
    active = True
    for i in range(GESTURE_OFFSET_FRAMES):
        # Large positional jump but confidence collapses to 0 -- hands_present
        # must be False regardless of the (meaningless, low-confidence) jump.
        frame = _make_frame(1000.0 * (i + 1), conf=0.0)
        window, active, _ = seg.push(frame)

    assert active is False
    assert seg.debug_state["state"] == "IDLE"


def test_force_reset_abandons_segment():
    seg = GestureSegmenter("s8")
    _push_still(seg, 2)
    _push_moving(seg, GESTURE_ONSET_FRAMES + 3)
    assert seg.debug_state["state"] == "ACTIVE"

    seg.force_reset()
    state = seg.debug_state
    assert state["state"] == "IDLE"
    assert state["segment_len"] == 0
    assert state["onset_streak"] == 0
    assert state["offset_streak"] == 0


def test_force_flush_classifies_pending_segment():
    seg = GestureSegmenter("s9")
    _push_still(seg, 2)
    _push_moving(seg, GESTURE_ONSET_FRAMES + GESTURE_MIN_FRAMES + 5)
    assert seg.debug_state["state"] == "ACTIVE"

    window = seg.force_flush()
    assert window is not None
    assert window.shape == (WINDOW_SIZE, 75, 3)
    assert seg.debug_state["state"] == "IDLE"


def test_force_flush_skips_when_hand_tracking_lost_at_the_end():
    """Real regression: a session ending right after hand tracking is lost
    (person stepped out of frame, occlusion) must not force-flush a
    degraded/zeroed tail into the classifier -- confirmed by a real DIAG
    log (rhand:0/21 in the frames right before eviction) followed by a
    spurious classification and a nonsense downstream LLM sentence."""
    seg = GestureSegmenter("s11")
    _push_still(seg, 2)
    _push_moving(seg, GESTURE_ONSET_FRAMES + GESTURE_MIN_FRAMES + 5)
    assert seg.debug_state["state"] == "ACTIVE"

    # Hands vanish for the last GESTURE_OFFSET_FRAMES frames (tracking
    # lost, not a genuine gesture ending).
    for _ in range(GESTURE_OFFSET_FRAMES):
        seg.push(_make_frame(0.0, conf=0.0))

    window = seg.force_flush()

    assert window is None
    assert seg.debug_state["state"] == "IDLE"  # still cleanly reset, not stuck ACTIVE


def test_force_flush_still_classifies_when_hands_return_before_flush():
    """A brief mid-gesture occlusion (hands come back before the session
    actually ends) must NOT block force_flush -- the gate only looks at
    the most recent frames, not the whole segment's history."""
    seg = GestureSegmenter("s12")
    _push_still(seg, 2)
    _push_moving(seg, GESTURE_ONSET_FRAMES + GESTURE_MIN_FRAMES + 5)
    for _ in range(2):
        seg.push(_make_frame(0.0, conf=0.0))  # brief occlusion, not the end
    _push_moving(seg, GESTURE_OFFSET_FRAMES + 2)  # hands return

    window = seg.force_flush()

    assert window is not None


def test_no_person_never_leaves_idle():
    seg = GestureSegmenter("s10")
    for _ in range(30):
        window, active, preview = seg.push(_make_frame(0.0, conf=0.0))
        assert active is False
        assert window is None
    assert seg.debug_state["state"] == "IDLE"
