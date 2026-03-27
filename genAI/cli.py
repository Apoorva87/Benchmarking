from __future__ import annotations

import argparse

from genAI.providers.registry import build_provider, list_provider_factories
from genAI.runners.evaluator import EvaluationRunner
from genAI.suites.registry import build_suite, list_suites


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local GenAI benchmarks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a benchmark suite.")
    run_parser.add_argument("--provider", required=True, choices=list_provider_factories())
    run_parser.add_argument("--model", required=True, help="Model name for the provider backend.")
    run_parser.add_argument("--suite", required=True, choices=list_suites())
    run_parser.add_argument("--max-new-tokens", type=int, default=None, help="Optional override for generation length.")

    info_parser = subparsers.add_parser("provider-info", help="Show setup information for a provider.")
    info_parser.add_argument("--provider", required=True, choices=list_provider_factories())
    info_parser.add_argument("--model", required=True, help="Model name for the provider backend.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    provider = build_provider(args.provider, args.model)
    if args.command == "provider-info":
        print(provider.setup_message())
        return

    suite = build_suite(args.suite)
    if args.command == "run" and args.max_new_tokens is not None:
        original_samples = list(suite.samples())
        for sample in original_samples:
            if hasattr(sample, "max_new_tokens"):
                sample.max_new_tokens = args.max_new_tokens
    results = suite.run(provider)
    aggregate = EvaluationRunner().aggregate(results)
    print(
        f"suite={aggregate.benchmark_name} provider={aggregate.provider_name} "
        f"model={aggregate.model_name} modality={aggregate.modality} "
        f"avg_score={aggregate.average_score:.2f} avg_latency={aggregate.average_latency_seconds:.4f}s"
    )
    for result in results:
        if "token_generation_rate" in result.metadata:
            print(
                f"sample={result.sample_id} size={result.metadata.get('prompt_size')} "
                f"ttft={result.metadata.get('time_to_first_token_seconds'):.4f}s "
                f"tok_s={result.metadata.get('token_generation_rate'):.2f} "
                f"total={result.metadata.get('total_duration_seconds'):.4f}s "
                f"tokens={result.metadata.get('output_token_count')}"
            )


if __name__ == "__main__":
    main()
