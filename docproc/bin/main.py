import argparse
import itertools
import logging
import sys
from pathlib import Path

from docproc.doc.analyzer import DocumentAnalyzer
from docproc.writer import CSVWriter, SQLiteWriter, JSONWriter
from docproc.doc.regions import RegionType

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
        choices=["csv", "sqlite", "json"],
        default="csv",
        help="Output writer type (default: csv)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "-r",
        "--regions",
        nargs="+",
        choices=[rt.name.lower() for rt in RegionType],
        help="Specify which region types to extract (e.g. -r text equation)",
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
        writer_map = {
            "csv": (CSVWriter, ".csv"),
            "sqlite": (SQLiteWriter, ".db"),
            "json": (JSONWriter, ".json"),
        }
        writer_class, default_suffix = writer_map[args.writer]

        output_path = (
            args.output if args.output else str(input_path.stem + default_suffix)
        )

        region_types = None
        if args.regions:
            region_types = [RegionType[r.upper()] for r in args.regions]

        logger.info(f"Processing document: {input_path}")
        logger.info(f"Using {args.writer} writer")

        with DocumentAnalyzer(
            str(input_path),
            output_path=output_path,
            writer=writer_class,
            exclude_fields=["bbox"],
            region_types=region_types,
        ) as analyzer:
            # Tee the generated regions so we can count without consuming the iterator
            regions_gen = analyzer.detect_regions()
            regions_for_count, regions_for_export = itertools.tee(regions_gen, 2)
            count = sum(1 for _ in regions_for_count)
            logger.info(f"Detected {count} regions")
            analyzer.export_regions(regions_for_export)
            logger.info("Export complete")
        return 0

    except Exception as e:
        logger.error(f"Failed to process document: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
