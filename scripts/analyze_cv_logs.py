"""Summarize cv_service logs: keypoint-region dropout, segment lengths, and
gloss prediction frequency — instead of eyeballing single DIAG lines.

Usage:
    docker compose logs cv-service | python scripts/analyze_cv_logs.py
    # or, from a saved file:
    python scripts/analyze_cv_logs.py path/to/cv-service.log

Parses two log line shapes emitted by services/cv_service/main.py:
  DIAG session=... {'state': ..., 'segment_len': ..., ...} size=WxH
      person=True/False kp=body:B/33 lhand:L/21 rhand:R/21 total:T/75
  session=... top1=<gloss> conf=0.NN gesture_active=.. preview=..

Keep the regexes here in sync if those log lines change.
"""
from __future__ import annotations

import io
import re
import sys
from collections import Counter, defaultdict

# Force UTF-8 stdout regardless of console codepage — gloss names are
# Cyrillic, and Windows terminals often default to cp1252/cp866.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_DIAG_RE = re.compile(
    r"DIAG session=(?P<session>\S+) \{(?P<state_dict>[^}]*)\} "
    r"size=(?P<w>\d+)x(?P<h>\d+) person=(?P<person>True|False) "
    r"kp=body:(?P<body>\d+)/33 lhand:(?P<lhand>\d+)/21 rhand:(?P<rhand>\d+)/21 "
    r"total:(?P<total>\d+)/75"
)
_RESULT_RE = re.compile(
    r"session=(?P<session>\S+) top1=(?P<gloss>.+?) conf=(?P<conf>[\d.]+) "
    r"gesture_active=(?P<active>True|False) preview=(?P<preview>True|False)"
)
_STATE_RE = re.compile(r"'state': '(\w+)'")
_SEGLEN_RE = re.compile(r"'segment_len': (\d+)")


def _percentile(values: list[int], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _summarize(values: list[int], label: str, denom: int) -> None:
    if not values:
        print(f"  {label}: no samples")
        return
    n = len(values)
    mean = sum(values) / n
    print(f"  {label}: n={n} mean={mean:.1f}/{denom} "
          f"p10={_percentile(values, 0.10):.0f} median={_percentile(values, 0.5):.0f} "
          f"p90={_percentile(values, 0.90):.0f} min={min(values)} max={max(values)}")


def main() -> None:
    lines = (open(sys.argv[1], encoding="utf-8", errors="replace") if len(sys.argv) > 1 else sys.stdin)

    diag_total: list[int] = []
    diag_body: list[int] = []
    diag_lhand: list[int] = []
    diag_rhand: list[int] = []
    diag_by_state: dict[str, list[int]] = defaultdict(list)  # total kp, keyed by IDLE/ACTIVE
    seg_lens: list[int] = []
    near_empty_hand_frames = 0  # lhand<3 AND rhand<3 while ACTIVE — effectively no hand signal
    active_diag_count = 0

    gloss_counts: Counter[str] = Counter()
    conf_by_gloss: dict[str, list[float]] = defaultdict(list)
    preview_count = 0
    final_count = 0

    for line in lines:
        m = _DIAG_RE.search(line)
        if m:
            total, body, lhand, rhand = (int(m[k]) for k in ("total", "body", "lhand", "rhand"))
            diag_total.append(total)
            diag_body.append(body)
            diag_lhand.append(lhand)
            diag_rhand.append(rhand)
            state_m = _STATE_RE.search(m["state_dict"])
            state = state_m.group(1) if state_m else "?"
            diag_by_state[state].append(total)
            if state == "ACTIVE":
                active_diag_count += 1
                if lhand < 3 and rhand < 3:
                    near_empty_hand_frames += 1
            seglen_m = _SEGLEN_RE.search(m["state_dict"])
            if seglen_m and state == "ACTIVE":
                seg_lens.append(int(seglen_m.group(1)))
            continue

        m = _RESULT_RE.search(line)
        if m:
            is_preview = m["preview"] == "True"
            if is_preview:
                preview_count += 1
                continue  # only count FINAL predictions in the gloss histogram
            final_count += 1
            gloss = m["gloss"]
            gloss_counts[gloss] += 1
            conf_by_gloss[gloss].append(float(m["conf"]))

    print("=" * 70)
    print("KEYPOINT REGION DROPOUT")
    print("=" * 70)
    _summarize(diag_total, "total (0-75)", 75)
    _summarize(diag_body, "body  (0-33)", 33)
    _summarize(diag_lhand, "lhand (0-21)", 21)
    _summarize(diag_rhand, "rhand (0-21)", 21)
    print()
    for state, vals in sorted(diag_by_state.items()):
        _summarize(vals, f"total while state={state}", 75)
    if active_diag_count:
        pct = 100 * near_empty_hand_frames / active_diag_count
        print(f"\n  ACTIVE samples with BOTH hands near-empty (<3/21 each): "
              f"{near_empty_hand_frames}/{active_diag_count} ({pct:.0f}%)")
        if pct > 20:
            print("  -> hands are dropping out during real gestures often enough to hurt "
                  "classification; consider lowering HAND_LOW_CONF_ZERO_THRESHOLD further, "
                  "checking lighting/distance/framing, or trying rtmlib mode='balanced'.")

    print()
    print("=" * 70)
    print("SEGMENT LENGTH (ACTIVE gesture duration in processed frames)")
    print("=" * 70)
    if seg_lens:
        _summarize(seg_lens, "segment_len during ACTIVE", max(seg_lens))
        long_frac = sum(1 for s in seg_lens if s > 80) / len(seg_lens)
        print(f"  fraction of samples with segment_len > 80 (near GESTURE_MAX_FRAMES=150): "
              f"{100*long_frac:.0f}%")
        if long_frac > 0.15:
            print("  -> segments are frequently very long — either GESTURE_OFFSET_THRESHOLD is too "
                  "strict for real hand-tracking noise to ever settle, or gestures aren't being "
                  "isolated cleanly. Resampling a 100+ frame segment down to WINDOW_SIZE=64 "
                  "compresses/dilutes whatever real motion is in there.")
    else:
        print("  no ACTIVE-state samples found in this log")

    print()
    print("=" * 70)
    print(f"FINAL GLOSS PREDICTIONS ({final_count} final, {preview_count} preview, ignored)")
    print("=" * 70)
    if gloss_counts:
        top_n = min(15, len(gloss_counts))
        print(f"Top {top_n} predicted glosses:")
        for gloss, count in gloss_counts.most_common(top_n):
            pct = 100 * count / final_count
            mean_conf = sum(conf_by_gloss[gloss]) / len(conf_by_gloss[gloss])
            print(f"  {gloss!r:25s} count={count:4d} ({pct:4.1f}%)  mean_conf={mean_conf:.2f}")
        top3_share = sum(c for _, c in gloss_counts.most_common(3)) / final_count
        print(f"\nTop-3 most common glosses cover {100*top3_share:.0f}% of all final predictions.")
        if top3_share > 0.4 and len(gloss_counts) < final_count * 0.3:
            print("-> class collapse: a handful of classes dominate regardless of what gesture "
                  "was actually performed. This points to an input-quality problem (see keypoint "
                  "region dropout above), not the model confusing similar signs — a healthily "
                  "discriminating classifier's top-1 distribution should look much flatter across "
                  "attempts at genuinely different gestures.")
    else:
        print("  no final (non-preview) predictions found in this log")


if __name__ == "__main__":
    main()
