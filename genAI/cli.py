from __future__ import annotations

import argparse

from genAI.providers.registry import build_provider, list_provider_factories
from genAI.runners.evaluator import EvaluationRunner
from genAI.suites.registry import build_suite, list_suites


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local GenAI benchmarks.")
    parser.add_argument("--provider", required=True, choices=list_provider_factories())
    parser.add_argument("--model", required=True, help="Model name for the provider backend.")
    parser.add_argument("--suite", required=True, choices=list_suites())
    return parser


def main() -> None:
    args = build_parser().parse_args()
    provider = build_provider(args.provider, args.model)
    suite = build_suite(args.suite)
    results = suite.run(provider)
    aggregate = EvaluationRunner().aggregate(results)
    print(
        f"suite={aggregate.benchmark_name} provider={aggregate.provider_name} "
        f"model={aggregate.model_name} modality={aggregate.modality} "
        f"avg_score={aggregate.average_score:.2f} avg_latency={aggregate.average_latency_seconds:.4f}s"
    )


if __name__ == "__main__":
    main()

