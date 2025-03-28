import argparse
import itertools
import logging
import sys
import yaml
import glob
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from docproc.doc.analyzer import DocumentAnalyzer
from docproc.writer import CSVWriter, SQLiteWriter, JSONWriter
from docproc.doc.regions import RegionType

# Keep existing logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and analyze document regions")

    # Input options with mutually exclusive group for single/batch
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", type=str, help="Path to input document (PDF)")
    input_group.add_argument(
        "--batch", action="store_true", help="Enable batch processing mode"
    )

    # Batch processing options
    batch_group = parser.add_argument_group("Batch processing options")
    batch_group.add_argument(
        "--input-dir", type=str, help="Directory containing documents to process"
    )
    batch_group.add_argument(
        "--pattern",
        type=str,
        default="*.pdf",
        help="File pattern for batch processing (default: *.pdf)",
    )
    batch_group.add_argument(
        "--output-dir", type=str, help="Output directory for batch results"
    )

    # Config file
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")

    # Existing arguments
    parser.add_argument(
        "-o", "--output", type=str, help="Output file path for single file mode"
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
    parser.add_argument(
        "--enable-handwriting",
        action="store_true",
        help="Enable handwriting detection (more resource intensive)",
    )
    parser.add_argument(
        "--convert-handwriting",
        action="store_true",
        help="Convert detected handwriting to LaTeX equations (requires pix2tex)",
    )

    return parser.parse_args()


def process_file(
    input_path: Path,
    output_path: str,
    writer_class: type,
    region_types: Optional[List[RegionType]] = None,
    enable_handwriting: bool = False,
    convert_handwriting: bool = False,
    verbose: bool = False,
) -> int:
    """Process a single file with the given parameters."""
    try:
        logger.info(f"Processing document: {input_path}")

        with DocumentAnalyzer(
            str(input_path),
            writer=writer_class,
            output_path=output_path,
            exclude_fields=["bbox"],
            region_types=region_types,
            enable_handwriting_detection=enable_handwriting,
            convert_handwriting_to_latex=convert_handwriting,
        ) as analyzer:
            regions_gen = analyzer.detect_regions()
            regions_for_count, regions_for_export = itertools.tee(regions_gen, 2)
            count = sum(1 for _ in regions_for_count)
            logger.info(f"Detected {count} regions")
            analyzer.export_regions(regions_for_export)
        return 0
    except Exception as e:
        logger.error(f"Failed to process document: {e}", exc_info=verbose)
        return 1


def main():
    args = parse_args()
    config = {}

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration file if specified
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

    # Command line args override config file
    writer_type = args.writer or config.get("writer", "csv")
    enable_handwriting = args.enable_handwriting or config.get(
        "enable_handwriting", False
    )
    convert_handwriting = args.convert_handwriting or config.get(
        "convert_handwriting", False
    )

    # Set up region types
    region_types = None
    if args.regions:
        region_types = [RegionType[r.upper()] for r in args.regions]
    elif "regions" in config:
        region_types = [RegionType[r.upper()] for r in config["regions"]]

    # Set up writer class
    writer_map = {
        "csv": (CSVWriter, ".csv"),
        "sqlite": (SQLiteWriter, ".db"),
        "json": (JSONWriter, ".json"),
    }
    writer_class, default_suffix = writer_map[writer_type]

    # Handle single file processing
    if args.file:
        input_path = Path(args.file)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1

        output_path = (
            args.output if args.output else str(input_path.stem + default_suffix)
        )
        return process_file(
            input_path,
            output_path,
            writer_class,
            region_types,
            enable_handwriting,
            convert_handwriting,
            args.verbose,
        )

    # Handle batch processing
    elif args.batch:
        if not args.input_dir and not args.file:
            logger.error("Batch processing requires --input-dir or --file")
            return 1

        input_dir = args.input_dir or os.path.dirname(args.file)
        output_dir = args.output_dir or input_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Find all matching files
        pattern = os.path.join(input_dir, args.pattern)
        pdf_files = glob.glob(pattern)

        if not pdf_files:
            logger.error(f"No files found matching pattern: {pattern}")
            return 1

        logger.info(f"Found {len(pdf_files)} files to process")

        # Process each file
        success_count = 0
        fail_count = 0

        for pdf_file in pdf_files:
            input_path = Path(pdf_file)
            output_path = os.path.join(output_dir, input_path.stem + default_suffix)

            logger.info(
                f"Processing {input_path.name} -> {os.path.basename(output_path)}"
            )

            result = process_file(
                input_path,
                output_path,
                writer_class,
                region_types,
                enable_handwriting,
                convert_handwriting,
                args.verbose,
            )

            if result == 0:
                success_count += 1
            else:
                fail_count += 1

        logger.info(
            f"Batch processing complete. Success: {success_count}, Failed: {fail_count}"
        )
        return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
