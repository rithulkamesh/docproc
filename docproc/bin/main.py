import argparse
import logging
import sys
from pathlib import Path

from docproc.doc.analyzer import DocumentAnalyzer
from docproc.writer import CSVWriter, SQLiteWriter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and analyze document regions")
    parser.add_argument("input_file", type=str, help="Path to input document (PDF)")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file path (default: input_file.csv or input_file.db)",
        default=None,
    )
    parser.add_argument(
        "-w",
        "--writer",
        type=str,
        choices=["csv", "sqlite"],
        default="csv",
        help="Output writer type (default: csv)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    try:
        # Select writer based on argument
        writer_class = CSVWriter if args.writer == "csv" else SQLiteWriter
        default_suffix = ".csv" if args.writer == "csv" else ".db"

        output_path = (
            args.output if args.output else str(input_path.stem + default_suffix)
        )

        logger.info(f"Processing document: {input_path}")
        logger.info(f"Using {args.writer} writer")

        with DocumentAnalyzer(
            str(input_path),
            writer_class,
            output_path=output_path,
            exclude_fields=["bbox"],
        ) as analyzer:
            regions = analyzer.detect_regions()
            logger.info(f"Detected {len(regions)} regions")
            analyzer.export_regions()
            logger.info("Export complete")
        return 0

    except Exception as e:
        logger.error(f"Failed to process document: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
