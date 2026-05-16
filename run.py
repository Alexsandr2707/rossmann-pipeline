from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming ML pipeline CLI")
    parser.add_argument(
        "-mode",
        required=True,
        choices=("inference", "update", "pretrain", "summary", "reset", "evaluate"),
        help="Pipeline mode to run.",
    )
    parser.add_argument(
        "-file",
        default=None,
        help="Input file for inference mode.",
    )
    parser.add_argument(
        "-config",
        default="config/config.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from app.config import load_config
    from app.logging_utils import configure_logging
    from app.pipeline import Pipeline

    config = load_config(args.config)
    configure_logging(config.paths.pipeline_log_path)

    pipeline = Pipeline(config)

    try:
        if args.mode == "inference":
            if not args.file:
                print("Inference mode requires -file.", file=sys.stderr)
                return 2
            pipeline.inference(Path(args.file))
        elif args.mode == "update":
            pipeline.update()
        elif args.mode == "pretrain":
            pipeline.pretrain()
        elif args.mode == "reset":
            pipeline.reset()
        elif args.mode == "evaluate":
            pipeline.evaluate()
        else:
            pipeline.summary()
    except NotImplementedError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
