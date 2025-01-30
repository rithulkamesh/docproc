import argparse
import logging
import sys
from pathlib import Path

from docproc.doc.analyzer import DocumentAnalyzer
from docproc.writer.csv import CSVWriter

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
        help="Output CSV file path (default: input_file.csv)",
        default=None,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    return parser.parse_args()


def main():
    """Main entry point for the document processing application.

    This function processes a PDF document, detects regions within it, and exports the results to a CSV file.
    It handles command-line arguments and logging configuration.

    Returns:
        int: 0 for successful execution, 1 for errors (file not found or processing failure)

    Raises:
        Exception: Any exception that occurs during document processing will be caught,
                  logged, and converted to a return value of 1

    Note:
        - Input file path must exist and be accessible
        - Output path defaults to input filename with .csv extension if not specified
        - Verbose flag enables debug logging and detailed error information
    """
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    try:
        output_path = (
            args.output
            if args.output
            else str(input_path.split(".pdf")[0].with_suffix(".csv"))
        )
        logger.info(f"Processing document: {input_path}")
        with DocumentAnalyzer(
            str(input_path), CSVWriter, output_path=output_path
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
