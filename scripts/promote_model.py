"""Model promotion CLI — move versions between MLflow registry stages.

Usage:
    # Promote a specific version to staging
    python scripts/promote_model.py --model gesture --version 3 --stage staging

    # Promote to production (runs metric gates)
    python scripts/promote_model.py --model gesture --version 3 --stage production

    # Auto-promote best Staging model by a metric
    python scripts/promote_model.py --model gesture --auto --metric accuracy

    # Rollback production to previous archived version
    python scripts/promote_model.py --model gesture --rollback

    # Show current versions for a model
    python scripts/promote_model.py --model gesture --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mlops.config import get_settings
from mlops.registry.onnx_registry import ONNXModelRegistry
from mlops.registry.promotion import ModelPromoter


_MODEL_NAME_MAP = {
    "gesture": "gesture_model_name",
    "nlp": "nlp_model_name",
    "asr": "asr_model_name",
}


def _resolve_model_name(alias: str, cfg: object) -> str:
    attr = _MODEL_NAME_MAP.get(alias.lower())
    if attr:
        return getattr(cfg, attr)
    return alias  # treat as a full MLflow model name


def cmd_list(model_name: str, registry: ONNXModelRegistry) -> None:
    versions = registry.list_versions(model_name)
    if not versions:
        print(f"No versions found for: {model_name}")
        return
    print(f"\n{'Version':<10} {'Stage':<15} {'Run ID':<36}")
    print("-" * 65)
    for v in sorted(versions, key=lambda x: int(x.version)):
        print(f"{v.version:<10} {v.stage:<15} {v.run_id:<36}")
    print()


def cmd_compare(model_name: str, registry: ONNXModelRegistry) -> None:
    rows = registry.compare_versions(model_name)
    if not rows:
        print(f"No versions found for: {model_name}")
        return
    print(f"\nModel comparison: {model_name}")
    print(json.dumps(rows, indent=2, default=str))


def cmd_lineage(model_name: str, version: str, registry: ONNXModelRegistry) -> None:
    lineage = registry.get_lineage(model_name, version)
    if not lineage:
        print(f"No lineage found for {model_name} v{version}")
        return
    print(json.dumps(lineage, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Glossa model promotion")
    parser.add_argument("--model", required=True, help="Model alias (gesture/nlp/asr) or full MLflow name")
    parser.add_argument("--version", help="Model version to promote")
    parser.add_argument("--stage", choices=["staging", "production"], help="Target stage")
    parser.add_argument("--auto", action="store_true", help="Auto-promote best Staging version")
    parser.add_argument("--metric", default="accuracy", help="Metric for --auto selection")
    parser.add_argument("--lower-is-better", action="store_true", help="Metric direction for --auto")
    parser.add_argument("--rollback", action="store_true", help="Rollback production to archived version")
    parser.add_argument("--no-archive", dest="archive", action="store_false", default=True,
                        help="Skip archiving existing production on promote")
    parser.add_argument("--list", action="store_true", help="List all versions")
    parser.add_argument("--compare", action="store_true", help="Show metric comparison table")
    parser.add_argument("--lineage", action="store_true", help="Show provenance for --version")
    args = parser.parse_args()

    cfg = get_settings()
    model_name = _resolve_model_name(args.model, cfg)
    registry = ONNXModelRegistry(cfg)
    promoter = ModelPromoter(registry=registry, settings=cfg)

    if args.list:
        cmd_list(model_name, registry)
        return

    if args.compare:
        cmd_compare(model_name, registry)
        return

    if args.lineage:
        if not args.version:
            print("--lineage requires --version")
            sys.exit(1)
        cmd_lineage(model_name, args.version, registry)
        return

    if args.rollback:
        result = promoter.rollback_production(model_name)
        _print_result(result)
        sys.exit(0 if result.success else 1)

    if args.auto:
        result = promoter.auto_promote(
            model_name,
            metric=args.metric,
            lower_is_better=args.lower_is_better,
        )
        _print_result(result)
        sys.exit(0 if result.success else 1)

    if not args.version:
        print("--version is required (or use --auto / --rollback / --list)")
        sys.exit(1)
    if not args.stage:
        print("--stage is required (or use --auto / --rollback / --list)")
        sys.exit(1)

    if args.stage == "staging":
        result = promoter.promote_to_staging(model_name, args.version)
    else:
        result = promoter.promote_to_production(
            model_name, args.version, archive_existing=args.archive
        )

    _print_result(result)
    sys.exit(0 if result.success else 1)


def _print_result(result: object) -> None:
    from mlops.registry.promotion import PromotionResult
    assert isinstance(result, PromotionResult)
    status = "SUCCESS" if result.success else "FAILED"
    print(f"\n[{status}] {result.model_name} v{result.version}: "
          f"{result.from_stage} → {result.to_stage}")
    print(f"  {result.reason}")
    if result.failed_checks:
        print("  Failed checks:")
        for check in result.failed_checks:
            print(f"    ✗ {check}")


if __name__ == "__main__":
    main()
