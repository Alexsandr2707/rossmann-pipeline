from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
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
        "update_count",
        nargs="?",
        type=int,
        help="Number of update executions for update mode.",
    )
    parser.add_argument(
        "-config",
        default="config/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "-open",
        action="store_true",
        help="Open the generated HTML dashboard after summary mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.open and args.mode != "summary":
        print("-open can only be used with -mode summary.", file=sys.stderr)
        return 2
    if args.update_count is not None and args.mode != "update":
        print("Update count can only be used with -mode update.", file=sys.stderr)
        return 2
    if args.update_count is not None and args.update_count < 1:
        print("Update count must be a positive integer.", file=sys.stderr)
        return 2

    from app.core.config import load_config
    from app.core.logging_utils import configure_logging
    from app.core.pipeline import Pipeline

    config = load_config(args.config)
    configure_logging(config.paths.pipeline_log_path)

    pipeline = Pipeline(config)

    try:
        if args.mode == "inference":
            if not args.file:
                print("Inference mode requires -file.", file=sys.stderr)
                return 2
            print(pipeline.inference(Path(args.file)))
        elif args.mode == "update":
            update_count = args.update_count or 1
            for _ in range(update_count):
                updated = bool(pipeline.update())
                print(updated)
                print(pipeline.summary())
                if not updated:
                    break
        elif args.mode == "pretrain":
            print(pipeline.pretrain())
        elif args.mode == "reset":
            print(pipeline.reset())
        elif args.mode == "evaluate":
            print(pipeline.evaluate())
        else:
            report_path = pipeline.summary()
            print(report_path)
            dashboard_path = config.paths.reports_dir / "index.html"
            if args.open:
                open_report(dashboard_path)
    except (FileExistsError, NotImplementedError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


def open_report(path: Path) -> None:
    full_path = path.resolve()
    try:
        # open the file in the default application
        if is_wsl():
            windows_path = subprocess.check_output(
                ["wslpath", "-w", str(full_path)],
                text=True,
            ).strip()
            subprocess.Popen(
                ["cmd.exe", "/c", "start", "", windows_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        webbrowser.open(full_path.as_uri())
    except OSError as error:
        print(f"Could not open browser automatically: {error}", file=sys.stderr)
        print(f"Open this file manually: {full_path}", file=sys.stderr)


def is_wsl() -> bool:
    version_path = Path("/proc/version")
    if not version_path.exists():
        return False
    return "microsoft" in version_path.read_text(encoding="utf-8").lower()


if __name__ == "__main__":
    raise SystemExit(main())
