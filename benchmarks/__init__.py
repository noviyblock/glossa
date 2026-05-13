"""Glossa benchmarking and load testing suite.

Run individual runners:
    python -m benchmarks.runners.latency_runner
    python -m benchmarks.runners.stress_runner
    python -m benchmarks.runners.redis_runner

Run everything:
    bash scripts/run_benchmarks.sh

Locust load test:
    locust -f benchmarks/locustfile.py --host http://localhost:8000
"""
