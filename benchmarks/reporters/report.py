"""Performance report generator — HTML with Chart.js + Markdown summary.

Combines results from all benchmark runners into a single report with:
  - SLO pass/fail table
  - Per-device latency projections (Poco M5, Realme X60, Poor 4G)
  - Throughput vs concurrency curves
  - Redis stream throughput
  - Mobile optimization recommendations

Usage:
    python -m benchmarks.reporters.report
    python -m benchmarks.reporters.report --title "Release 1.2.3"
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.config import DEVICE_PROFILES, SLOS, DeviceProfile, LatencySLO, get_config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def _slo_badge(passed: bool) -> str:
    return "✅ PASS" if passed else "❌ FAIL"


def _ms(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.1f}ms"


def _project_latency(base_ms: float, device: DeviceProfile) -> float:
    """Estimate perceived latency on a mobile device: add RTT and ~10% jitter budget."""
    return base_ms + device.rtt_ms + device.jitter_ms * 0.5


# ── Markdown report ───────────────────────────────────────────────────────────

def build_markdown(
    latency_results: list[dict],
    stress_results: list[dict],
    redis_results: list[dict],
    title: str = "Glossa Performance Report",
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append(f"_Generated: {now}_\n")

    # ── SLO table ──────────────────────────────────────────────────────────────
    lines.append("## Service Latency & SLO Status\n")
    lines.append("| Service | P50 | P95 | P99 | RPS | SLO |")
    lines.append("|---------|-----|-----|-----|-----|-----|")

    for r in latency_results:
        slo_key = r.get("slo", {}).get("description", "") if r.get("slo") else ""
        passed = r.get("slo_passed", True)
        lines.append(
            f"| {r['service']:<22} "
            f"| {_ms(r['p50_ms'])} "
            f"| {_ms(r['p95_ms'])} "
            f"| {_ms(r['p99_ms'])} "
            f"| {r.get('throughput_rps', 0):.1f} "
            f"| {_slo_badge(passed)} |"
        )

    # ── Device projections ─────────────────────────────────────────────────────
    lines.append("\n## Projected Latency on Mobile Devices\n")
    lines.append("_Base server latency + one-way RTT + jitter budget_\n")
    lines.append("| Service | " + " | ".join(d.name for d in DEVICE_PROFILES) + " |")
    lines.append("|---------|" + "|".join(["---" for _ in DEVICE_PROFILES]) + "|")

    for r in latency_results:
        svc = r["service"]
        p95 = r.get("p95_ms") or 0.0
        if "@" in svc:
            continue  # skip concurrency variants
        row = f"| {svc:<22} |"
        for dev in DEVICE_PROFILES:
            projected = _project_latency(p95, dev)
            # Check against SLO for that device's budget
            slo = None
            for k, v in SLOS.items():
                if k in svc.lower().replace("-", "_").replace(" ", "_"):
                    slo = v
                    break
            mark = ""
            if slo and projected > slo.p95_ms:
                mark = " ⚠"
            row += f" {projected:.0f}ms{mark} |"
        lines.append(row)

    # ── Stress / throughput ────────────────────────────────────────────────────
    if stress_results:
        lines.append("\n## Throughput vs Concurrency\n")
        for rep in stress_results:
            svc = rep.get("service", "unknown")
            sat = rep.get("saturation_concurrency")
            bp = rep.get("breaking_point_concurrency")
            max_rps = rep.get("max_rps", 0)
            lines.append(f"**{svc}** — max_rps={max_rps:.1f}  "
                         f"saturation@{sat}c  breaking@{bp}c\n")
            lines.append("| Concurrency | P50 | P95 | RPS | Errors |")
            lines.append("|-------------|-----|-----|-----|--------|")
            for pt in rep.get("points", []):
                lines.append(
                    f"| {pt['concurrency']:<11} "
                    f"| {_ms(pt['p50_ms'])} "
                    f"| {_ms(pt['p95_ms'])} "
                    f"| {pt['actual_rps']:.1f} "
                    f"| {pt['error_rate']:.1%} |"
                )
            lines.append("")

    # ── Redis ──────────────────────────────────────────────────────────────────
    if redis_results:
        lines.append("## Redis Stream Performance\n")
        lines.append("| Stream | Operation | Throughput | P95 | Lag |")
        lines.append("|--------|-----------|------------|-----|-----|")
        for r in redis_results:
            lines.append(
                f"| {r.get('stream',''):<38} "
                f"| {r.get('operation',''):<10} "
                f"| {r.get('throughput_eps', 0):.0f} eps "
                f"| {_ms(r.get('p95_ms'))} "
                f"| {r.get('consumer_lag', 0)} |"
            )

    # ── Recommendations ────────────────────────────────────────────────────────
    lines.append("\n## Mobile Optimization Recommendations\n")
    lines += _generate_recommendations(latency_results, stress_results)

    return "\n".join(lines)


def _generate_recommendations(
    latency_results: list[dict],
    stress_results: list[dict],
) -> list[str]:
    recs: list[str] = []
    latency_map = {r["service"]: r for r in latency_results}

    def check(svc_key: str, slo_key: str, rec: str) -> None:
        r = latency_map.get(svc_key)
        if r:
            slo = SLOS.get(slo_key)
            p95 = r.get("p95_ms") or 0.0
            if slo and p95 > slo.p95_ms:
                recs.append(f"- **{svc_key}**: P95={p95:.0f}ms exceeds {slo.p95_ms:.0f}ms SLO — {rec}")

    check("cv-service", "gesture_inference",
          "Enable OpenVINO backend or INT8 quantization for 2-4× speedup on CPU")
    check("asr-service", "asr_chunk",
          "Use `tiny` Whisper model for mobile use-cases; enable VAD to skip silence")
    check("nlp-service", "nlp_translate",
          "Enable 4-bit quantization (NLP_LOAD_IN_4BIT=true); consider model distillation")
    check("tts-service", "tts_synthesize",
          "Pre-warm TTS model; cache short phrases; stream audio chunks")
    check("websocket", "websocket_rtt",
          "Enable WebSocket compression; reduce ping interval to 30s on poor networks")

    # Redis
    recs.append(
        "- **Redis streams**: Use pipelined XADD in batches of 50-100 for 5-10× throughput improvement"
    )

    # Mobile network
    recs.append(
        "- **Poco M5 (4G)**: Add 65ms RTT to all latencies; "
        "total pipeline budget is ~2000ms → server budget ~1870ms at P95"
    )
    recs.append(
        "- **Poor 4G edge case**: Enable gzip WebSocket compression; "
        "use binary WS frames; batch keypoints per frame"
    )
    recs.append(
        "- **Video encoding**: Compress gesture keypoints to ~528 bytes (float32 × 33 × 4) instead of JPEG frames"
    )
    recs.append(
        "- **Connection keep-alive**: Reuse HTTP/1.1 keep-alive connections; "
        "use HTTP/2 multiplexing for concurrent streams"
    )

    return recs if recs else ["- All SLOs passing — no immediate action required."]


# ── HTML report ───────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 20px; background: #0f172a; color: #e2e8f0; }}
  h1 {{ color: #38bdf8; }} h2 {{ color: #7dd3fc; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th {{ background: #1e3a5f; color: #93c5fd; padding: 8px 12px; text-align: left; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #1e293b; }}
  tr:hover {{ background: #1e293b; }}
  .pass {{ color: #4ade80; }} .fail {{ color: #f87171; }}
  .card {{ background: #1e293b; border-radius: 8px; padding: 16px; margin: 12px 0; }}
  .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; }}
  .chart-box {{ background: #1e293b; border-radius: 8px; padding: 16px; }}
  .meta {{ color: #94a3b8; font-size: 0.85em; margin-bottom: 20px; }}
  .rec {{ background: #0c2340; border-left: 4px solid #38bdf8; padding: 10px 16px; margin: 8px 0; border-radius: 4px; }}
  .slo-ok {{ color: #4ade80; }} .slo-fail {{ color: #f87171; }}
</style>
</head>
<body>
<h1>🚀 {title}</h1>
<p class="meta">Generated: {timestamp} | Services: {n_services} | SLO pass rate: {slo_pass_rate}</p>

<h2>Service Latency & SLOs</h2>
<table>
<tr><th>Service</th><th>P50</th><th>P95</th><th>P99</th><th>RPS</th><th>Errors</th><th>SLO</th></tr>
{latency_rows}
</table>

<h2>Mobile Device Projections (P95 + RTT)</h2>
<table>
<tr><th>Service</th><th>Poco M5 (4G, 65ms RTT)</th><th>Realme X60 (5G, 30ms RTT)</th><th>Poor 4G (180ms RTT)</th></tr>
{device_rows}
</table>

<h2>Performance Charts</h2>
<div class="charts">
  <div class="chart-box"><canvas id="latencyChart"></canvas></div>
  <div class="chart-box"><canvas id="throughputChart"></canvas></div>
  <div class="chart-box"><canvas id="stressChart"></canvas></div>
  <div class="chart-box"><canvas id="redisChart"></canvas></div>
</div>

<h2>Recommendations for Mobile Optimization</h2>
<div class="card">
{recommendations}
</div>

<script>
const ctxLatency = document.getElementById('latencyChart');
new Chart(ctxLatency, {{
  type: 'bar',
  data: {{
    labels: {latency_labels},
    datasets: [
      {{ label: 'P50', data: {p50_data}, backgroundColor: '#38bdf8' }},
      {{ label: 'P95', data: {p95_data}, backgroundColor: '#f59e0b' }},
      {{ label: 'P99', data: {p99_data}, backgroundColor: '#f87171' }},
    ]
  }},
  options: {{
    plugins: {{ title: {{ display: true, text: 'Latency by Service (ms)', color: '#e2e8f0' }}, legend: {{ labels: {{ color: '#e2e8f0' }} }} }},
    scales: {{ x: {{ ticks: {{ color: '#94a3b8' }} }}, y: {{ ticks: {{ color: '#94a3b8' }}, title: {{ display: true, text: 'ms', color: '#94a3b8' }} }} }}
  }}
}});

const ctxTp = document.getElementById('throughputChart');
new Chart(ctxTp, {{
  type: 'bar',
  data: {{
    labels: {latency_labels},
    datasets: [{{ label: 'Throughput (RPS)', data: {rps_data}, backgroundColor: '#4ade80' }}]
  }},
  options: {{
    plugins: {{ title: {{ display: true, text: 'Throughput (RPS)', color: '#e2e8f0' }}, legend: {{ labels: {{ color: '#e2e8f0' }} }} }},
    scales: {{ x: {{ ticks: {{ color: '#94a3b8' }} }}, y: {{ ticks: {{ color: '#94a3b8' }} }} }}
  }}
}});

const ctxStress = document.getElementById('stressChart');
new Chart(ctxStress, {{
  type: 'line',
  data: {{
    labels: {stress_labels},
    datasets: {stress_datasets}
  }},
  options: {{
    plugins: {{ title: {{ display: true, text: 'P95 Latency vs Concurrency', color: '#e2e8f0' }}, legend: {{ labels: {{ color: '#e2e8f0' }} }} }},
    scales: {{ x: {{ ticks: {{ color: '#94a3b8' }}, title: {{ display: true, text: 'Concurrent Users', color: '#94a3b8' }} }}, y: {{ ticks: {{ color: '#94a3b8' }}, title: {{ display: true, text: 'P95 ms', color: '#94a3b8' }} }} }}
  }}
}});

const ctxRedis = document.getElementById('redisChart');
new Chart(ctxRedis, {{
  type: 'bar',
  data: {{
    labels: {redis_labels},
    datasets: [{{ label: 'Throughput (events/s)', data: {redis_data}, backgroundColor: '#a78bfa' }}]
  }},
  options: {{
    plugins: {{ title: {{ display: true, text: 'Redis Stream Throughput', color: '#e2e8f0' }}, legend: {{ labels: {{ color: '#e2e8f0' }} }} }},
    scales: {{ x: {{ ticks: {{ color: '#94a3b8' }} }}, y: {{ ticks: {{ color: '#94a3b8' }} }} }}
  }}
}});
</script>
</body>
</html>
"""


def build_html(
    latency_results: list[dict],
    stress_results: list[dict],
    redis_results: list[dict],
    title: str = "Glossa Performance Report",
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Latency table rows
    latency_rows = []
    base_services = [r for r in latency_results if "@" not in r["service"]]
    for r in base_services:
        passed = r.get("slo_passed", True)
        cls = "slo-ok" if passed else "slo-fail"
        badge = "✅" if passed else "❌"
        err = r.get("n_errors", 0)
        n = r.get("n_samples", 1) or 1
        latency_rows.append(
            f"<tr><td>{r['service']}</td>"
            f"<td>{_ms(r.get('p50_ms'))}</td>"
            f"<td>{_ms(r.get('p95_ms'))}</td>"
            f"<td>{_ms(r.get('p99_ms'))}</td>"
            f"<td>{r.get('throughput_rps', 0):.1f}</td>"
            f"<td>{err}/{n} ({err/n:.1%})</td>"
            f"<td class='{cls}'>{badge}</td></tr>"
        )

    # Device projection rows
    device_rows = []
    for r in base_services:
        p95 = r.get("p95_ms") or 0.0
        proj = [f"{_project_latency(p95, dev):.0f}ms" for dev in DEVICE_PROFILES]
        device_rows.append(
            f"<tr><td>{r['service']}</td>"
            + "".join(f"<td>{p}</td>" for p in proj)
            + "</tr>"
        )

    # Chart data
    chart_svcs = [r["service"] for r in base_services]
    p50s = [r.get("p50_ms") or 0 for r in base_services]
    p95s = [r.get("p95_ms") or 0 for r in base_services]
    p99s = [r.get("p99_ms") or 0 for r in base_services]
    rpss = [r.get("throughput_rps") or 0 for r in base_services]

    # Stress chart
    stress_labels: list[int] = []
    stress_datasets: list[dict] = []
    colors = ["#38bdf8", "#4ade80", "#f59e0b", "#f87171", "#a78bfa"]
    for i, rep in enumerate(stress_results[:5]):
        pts = rep.get("points", [])
        if pts:
            stress_labels = [p["concurrency"] for p in pts]
            stress_datasets.append({
                "label": rep["service"],
                "data": [p.get("p95_ms", 0) for p in pts],
                "borderColor": colors[i % len(colors)],
                "tension": 0.3,
                "fill": False,
            })

    # Redis chart
    redis_labels = [f"{r['operation']}\n{r['stream'][-20:]}" for r in redis_results]
    redis_data = [r.get("throughput_eps", 0) for r in redis_results]

    # Recommendations
    recs_md = _generate_recommendations(latency_results, stress_results)
    rec_html = "\n".join(f'<div class="rec">{r[2:]}</div>' for r in recs_md)

    n_pass = sum(1 for r in base_services if r.get("slo_passed", True))
    slo_rate = f"{n_pass}/{len(base_services)} ({n_pass/max(len(base_services),1):.0%})"

    return _HTML_TEMPLATE.format(
        title=title,
        timestamp=timestamp,
        n_services=len(base_services),
        slo_pass_rate=slo_rate,
        latency_rows="\n".join(latency_rows),
        device_rows="\n".join(device_rows),
        recommendations=rec_html,
        latency_labels=json.dumps(chart_svcs),
        p50_data=json.dumps(p50s),
        p95_data=json.dumps(p95s),
        p99_data=json.dumps(p99s),
        rps_data=json.dumps(rpss),
        stress_labels=json.dumps(stress_labels),
        stress_datasets=json.dumps(stress_datasets),
        redis_labels=json.dumps(redis_labels),
        redis_data=json.dumps(redis_data),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate performance report")
    parser.add_argument("--title", default="Glossa Performance Report")
    parser.add_argument("--output-html", default="benchmarks/reports/performance_report.html")
    parser.add_argument("--output-md", default="benchmarks/reports/performance_report.md")
    args = parser.parse_args()

    cfg = get_config()
    reports = cfg.reports_dir

    latency = _load_json(reports / "latency.json") or []
    stress = _load_json(reports / "stress_results.json") or []
    redis = _load_json(reports / "redis_results.json") or []

    html = build_html(latency, stress, redis, args.title)
    md = build_markdown(latency, stress, redis, args.title)

    Path(args.output_html).write_text(html)
    Path(args.output_md).write_text(md)
    print(f"HTML report: {args.output_html}")
    print(f"Markdown report: {args.output_md}")


if __name__ == "__main__":
    main()
