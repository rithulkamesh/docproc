import argparse
import itertools
import logging
import sys
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and analyze document content including text, equations, and visual elements"
    )

    # Main input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "input", nargs="?", help="Path to input document (PDF or image file)"
    )
    input_group.add_argument(
        "-b", "--batch", action="store_true", help="Enable batch processing mode"
    )

    # Basic options
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file path (default: input filename with appropriate extension)",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["csv", "sqlite", "json"],
        default="csv",
        help="Output format (default: csv)",
    )

    # Content options
    parser.add_argument(
        "-c",
        "--content",
        nargs="+",
        choices=["text", "equation", "image", "figure", "all"],
        default=["all"],
        help="Content types to extract (default: all)",
    )

    # Enable visual content detection
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Enable advanced visual content detection (more accurate but slower)",
    )

    # For extracting all content from a single file
    parser.add_argument(
        "--extract-all",
        action="store_true",
        help="Extract all data into a structured format including content and metadata",
    )

    # Batch options
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing documents to process (for batch mode)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdf",
        help="File pattern for batch processing (default: *.pdf)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for batch results (default: same as input)",
    )

    # Debugging and verbosity
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


def process_file(
    input_path: Path,
    output_path: str,
    writer_class: type,
    region_types: Optional[List[RegionType]] = None,
    enable_visual_detection: bool = False,
    extract_all: bool = False,
    verbose: bool = False,
) -> int:
    """Process a single file with the given parameters."""
    try:
        logger.info(f"Processing document: {input_path}")

        with DocumentAnalyzer(
            str(input_path),
            writer=writer_class,
            output_path=output_path,
            exclude_fields=["bbox"] if not extract_all else None,
            region_types=region_types,
            enable_visual_detection=enable_visual_detection,
        ) as analyzer:
            if extract_all:
                # Extract all data as structured content
                data = analyzer.extract_all_data()
                logger.info(f"Extracted {len(data)} regions")
                # Write to output file using the writer
                with writer_class(output_path) as writer:
                    writer.init_tables()
                    writer.write_data(data)
            else:
                # Use standard region detection and export
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

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine writer type
    writer_map = {
        "csv": (CSVWriter, ".csv"),
        "sqlite": (SQLiteWriter, ".db"),
        "json": (JSONWriter, ".json"),
    }
    writer_class, default_suffix = writer_map[args.format]

    # Set up region types
    if "all" in args.content:
        region_types = None  # None means all types
    else:
        region_types = []
        # Map from CLI arguments to RegionType enum
        content_map = {
            "text": RegionType.TEXT,
            "equation": RegionType.EQUATION,
            "image": RegionType.IMAGE,
            "figure": RegionType.FIGURE,
        }
        for content_type in args.content:
            if content_type in content_map:
                region_types.append(content_map[content_type])

    # Handle single file processing
    if not args.batch:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1

        output_path = (
            args.output if args.output else str(input_path.with_suffix(default_suffix))
        )

        return process_file(
            input_path,
            output_path,
            writer_class,
            region_types,
            args.visual,
            args.extract_all,
            args.verbose,
        )

    # Handle batch processing
    else:
        if not args.input_dir:
            logger.error("Batch processing requires --input-dir")
            return 1

        input_dir = args.input_dir
        output_dir = args.output_dir or input_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Find all matching files
        pattern = os.path.join(input_dir, args.pattern)
        files = list(Path(input_dir).glob(args.pattern.replace("*", "**/*")))

        if not files:
            logger.error(f"No files found matching pattern: {pattern}")
            return 1

        logger.info(f"Found {len(files)} files to process")

        # Process each file
        success_count = 0
        fail_count = 0

        for file_path in files:
            output_path = os.path.join(output_dir, file_path.stem + default_suffix)

            logger.info(
                f"Processing {file_path.name} -> {os.path.basename(output_path)}"
            )

            result = process_file(
                file_path,
                output_path,
                writer_class,
                region_types,
                args.visual,
                args.extract_all,
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
