"""Calibration analysis for GestureSegmenter's onset/offset thresholds.

Groups cv_service log lines into per-session "episodes" (one IDLE->ACTIVE
through the next real ACTIVE->IDLE, i.e. one human-perceived gesture,
possibly containing intermediate force-flush splits) and reports:

  - segment_len, force-flush count, and the last_activity samples seen
    during each episode (works on ANY log, no ground truth needed) --
  - Optionally, if --ground-truth is given, joins episodes IN ORDER to a
    known sequence of (gloss, speed) and reports per-gesture correctness,
    for exactly the "segment_len vs verdict" table needed to decide
    whether GESTURE_ONSET/OFFSET_THRESHOLD need re-tuning.

IMPORTANT caveat baked into the join: episodes are matched to ground-truth
rows purely by ORDER. If the episode count doesn't match the ground-truth
row count (extra/missed onsets, e.g. a false trigger from jitter), the
script refuses to guess an alignment and prints the mismatch instead --
silently misaligning would produce a plausible-looking but meaningless
table.

Usage:
    # Preliminary read, no ground truth (works on any existing log):
    python scripts/analyze_segmenter_calibration.py path/to/cv-service.log

    # Full calibration table (needs a controlled run + this CSV):
    #   order,gloss,speed   (speed: fast|normal|slow, free text ok)
    #   1,"6 часов",normal
    #   2,"7:45",fast
    #   ...
    python scripts/analyze_segmenter_calibration.py path/to/cv-service.log \
        --ground-truth path/to/ground_truth.csv

For per-frame last_activity resolution during the controlled run itself,
set DIAG_LOG_INTERVAL=1 (or 2-3) on cv-service for that run only -- the
default (30) undersamples episodes shorter than ~30 frames, which is most
of them (see services/cv_service/config.py's DIAG_LOG_INTERVAL comment).
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from dataclasses import dataclass, field

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_ONSET_RE = re.compile(r"session=(?P<session>\S+) IDLE->ACTIVE activity=(?P<activity>[\d.]+) preroll_len=(?P<preroll>\d+)")
_OFFSET_RE = re.compile(r"session=(?P<session>\S+) ACTIVE->IDLE segment_len=(?P<seglen>\d+) discarded=(?P<discarded>True|False)")
_FORCEFLUSH_RE = re.compile(r"session=(?P<session>\S+) ACTIVE force-flush at MAX_FRAMES=(?P<max>\d+)")
_DIAG_RE = re.compile(
    r"DIAG session=(?P<session>\S+) \{(?P<state_dict>[^}]*)\}"
)
_STATE_RE = re.compile(r"'state': '(\w+)'")
_ACTIVITY_RE = re.compile(r"'last_activity': ([\d.]+)")
_OFFSET_STREAK_RE = re.compile(r"'offset_streak': (\d+)")
_RESULT_RE = re.compile(
    r"session=(?P<session>\S+) top1=(?P<gloss>.+?) conf=(?P<conf>[\d.]+) "
    r"gesture_active=(?P<active>True|False) preview=(?P<preview>True|False)"
)


@dataclass
class Episode:
    session: str
    segment_len: int = 0
    discarded: bool = False
    force_flushes: int = 0
    activity_samples: list[tuple[float, int]] = field(default_factory=list)  # (last_activity, offset_streak)
    final_gloss: str | None = None
    final_conf: float | None = None


def parse_log(lines) -> list[Episode]:
    episodes: list[Episode] = []
    open_ep: dict[str, Episode] = {}  # session -> in-progress episode
    pending_final: dict[str, bool] = {}  # session -> "next non-preview result belongs to just-closed episode"

    for line in lines:
        m = _ONSET_RE.search(line)
        if m:
            sess = m["session"][:8]
            open_ep[sess] = Episode(session=sess)
            continue

        m = _FORCEFLUSH_RE.search(line)
        if m:
            sess = m["session"][:8]
            if sess in open_ep:
                open_ep[sess].force_flushes += 1
                pending_final[sess] = True  # force-flush emits its own non-preview result
            continue

        m = _DIAG_RE.search(line)
        if m:
            sess = m["session"][:8]
            state_m = _STATE_RE.search(m["state_dict"])
            act_m = _ACTIVITY_RE.search(m["state_dict"])
            streak_m = _OFFSET_STREAK_RE.search(m["state_dict"])
            if sess in open_ep and state_m and state_m.group(1) == "ACTIVE" and act_m:
                streak = int(streak_m.group(1)) if streak_m else 0
                open_ep[sess].activity_samples.append((float(act_m.group(1)), streak))
            continue

        m = _OFFSET_RE.search(line)
        if m:
            sess = m["session"][:8]
            ep = open_ep.pop(sess, None)
            if ep is None:
                continue  # ACTIVE->IDLE with no matching onset in this log slice -- skip
            ep.segment_len = int(m["seglen"])
            ep.discarded = m["discarded"] == "True"
            episodes.append(ep)
            pending_final[sess] = not ep.discarded  # discarded segments log no top1 line
            continue

        m = _RESULT_RE.search(line)
        if m and m["preview"] == "False":
            sess = m["session"][:8]
            if pending_final.get(sess):
                # Attach to the most recently closed episode for this session.
                for ep in reversed(episodes):
                    if ep.session == sess and ep.final_gloss is None:
                        ep.final_gloss = m["gloss"]
                        ep.final_conf = float(m["conf"])
                        break
                pending_final[sess] = False
            continue

    return episodes


def load_ground_truth(path: str) -> list[tuple[str, str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["gloss"].strip(), row["speed"].strip()))
    return rows


def print_preliminary(episodes: list[Episode]) -> None:
    print("=" * 78)
    print(f"EPISODES (no ground truth) -- n={len(episodes)}")
    print("=" * 78)
    if not episodes:
        print("  no complete IDLE->ACTIVE->IDLE episodes found in this log")
        return
    print(f"{'#':>3} {'session':10} {'seg_len':>8} {'ff':>3} {'discarded':>10} {'pred':20} {'conf':>5}  activity samples (value@offset_streak)")
    for i, ep in enumerate(episodes, 1):
        samples = ", ".join(f"{a:.4f}@{s}" for a, s in ep.activity_samples) or "(none captured)"
        pred = ep.final_gloss or ("-" if ep.discarded else "?")
        conf = f"{ep.final_conf:.2f}" if ep.final_conf is not None else "-"
        print(f"{i:>3} {ep.session:10} {ep.segment_len:>8} {ep.force_flushes:>3} {str(ep.discarded):>10} {pred:20} {conf:>5}  {samples}")

    seg_lens = [e.segment_len for e in episodes]
    print(f"\nsegment_len: min={min(seg_lens)} max={max(seg_lens)} mean={sum(seg_lens)/len(seg_lens):.1f}")
    ff_total = sum(e.force_flushes for e in episodes)
    if ff_total:
        print(f"force-flushes: {ff_total} across {sum(1 for e in episodes if e.force_flushes)} episode(s) "
              f"-- these are cases GESTURE_OFFSET_THRESHOLD never triggered before MAX_FRAMES=150; "
              f"if frequent, offset_streak samples above show whether activity was genuinely still high "
              f"or hovering just above threshold (flapping).")
    n_diag = sum(len(e.activity_samples) for e in episodes)
    avg_diag_per_ep = n_diag / len(episodes)
    if avg_diag_per_ep < 2:
        print(f"\n⚠ avg {avg_diag_per_ep:.1f} DIAG last_activity samples per episode -- too sparse for a "
              f"real distribution-based threshold derivation. Re-run the controlled test with "
              f"DIAG_LOG_INTERVAL=1 or 2 (see services/cv_service/config.py) before trying to derive new "
              f"threshold values from this data.")


def print_ground_truth_table(episodes: list[Episode], gt: list[tuple[str, str]]) -> None:
    print("=" * 78)
    print(f"GROUND-TRUTH JOIN -- {len(episodes)} episodes vs {len(gt)} ground-truth rows")
    print("=" * 78)
    if len(episodes) != len(gt):
        print(f"⚠ MISMATCH: episode count ({len(episodes)}) != ground-truth row count ({len(gt)}).")
        print("Refusing to guess an order-based alignment -- extra/missing onsets (jitter false-triggers, "
              "a gesture that never crossed GESTURE_ONSET_THRESHOLD, etc.) would silently misalign every "
              "row after the first discrepancy. Fix the log/CSV (or trim to a matching sub-range) and re-run.")
        return

    print(f"{'#':>3} {'expected_gloss':20} {'speed':8} {'seg_len':>8} {'ff':>3} {'predicted':20} {'verdict':10}")
    correct = incorrect = discarded = 0
    by_verdict_seglen: dict[str, list[int]] = {"correct": [], "incorrect": [], "discarded": []}
    for i, ((expected_gloss, speed), ep) in enumerate(zip(gt, episodes), 1):
        if ep.discarded or ep.final_gloss is None:
            verdict = "discarded"
            discarded += 1
        elif ep.final_gloss == expected_gloss:
            verdict = "correct"
            correct += 1
        else:
            verdict = "incorrect"
            incorrect += 1
        by_verdict_seglen[verdict].append(ep.segment_len)
        pred = ep.final_gloss or "-"
        print(f"{i:>3} {expected_gloss:20} {speed:8} {ep.segment_len:>8} "
              f"{ep.force_flushes:>3} {pred:20} {verdict:10}")

    n = len(gt)
    print(f"\ncorrect={correct}/{n} ({100*correct/n:.0f}%)  incorrect={incorrect}/{n}  discarded={discarded}/{n}")
    for verdict, lens in by_verdict_seglen.items():
        if lens:
            print(f"  segment_len when {verdict}: min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.1f} (n={len(lens)})")

    if by_verdict_seglen["incorrect"] and by_verdict_seglen["correct"]:
        import statistics
        wrong_mean = statistics.mean(by_verdict_seglen["incorrect"])
        right_mean = statistics.mean(by_verdict_seglen["correct"])
        print(f"\nmean segment_len: correct={right_mean:.1f} vs incorrect={wrong_mean:.1f} "
              f"({'higher' if wrong_mean > right_mean else 'lower'} on wrong predictions)")
        print("-> eyeball only: run a proper significance check (e.g. Mann-Whitney) before citing this as "
              "a real effect if n is small.")
    else:
        print("\n(not enough of both classes to compare segment_len distributions)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logfile", help="saved cv-service log (or '-' for stdin)")
    ap.add_argument("--ground-truth", help="CSV with order,gloss,speed columns, in the order gestures were shown")
    args = ap.parse_args()

    lines = sys.stdin if args.logfile == "-" else open(args.logfile, encoding="utf-8", errors="replace")
    episodes = parse_log(lines)

    if args.ground_truth:
        gt = load_ground_truth(args.ground_truth)
        print_ground_truth_table(episodes, gt)
    else:
        print_preliminary(episodes)


if __name__ == "__main__":
    main()
