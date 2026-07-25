"""CLI entry point for YOLO ground-truth small-object scale analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from .small_object import analyse_yolo_dataset, write_analysis_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse YOLO labels for P2/TAL decisions")
    parser.add_argument("--data", required=True, help="YOLO data.yaml path")
    parser.add_argument("--output", default="reports/small_object")
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    report = analyse_yolo_dataset(Path(args.data), args.clusters, args.imgsz)
    json_path, markdown_path = write_analysis_report(report, Path(args.output))
    print(f"Wrote scale report: {json_path}")
    print(f"Wrote scale appendix: {markdown_path}")


if __name__ == "__main__":
    main()
