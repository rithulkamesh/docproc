"""CLI: extract document to markdown (docproc --file input.pdf -o output.md)."""

import argparse
import logging
import sys
from pathlib import Path

from docproc.doc.loaders import get_supported_extensions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract document to markdown (vision + optional LLM refine)"
    )
    parser.add_argument("--file", type=str, required=True, help="Path to input document")
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output markdown path (.md)"
    )
    parser.add_argument("--config", type=str, help="Path to docproc config (docproc.yaml)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = Path(args.file)
    output_path = args.output

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    if not output_path.lower().endswith(".md"):
        logger.error("Output must be a .md file (-o output.md)")
        return 1

    supported = get_supported_extensions()
    if input_path.suffix.lower() not in supported:
        logger.error(
            "Unsupported format: %s. Supported: %s",
            input_path.suffix,
            ", ".join(sorted(supported)),
        )
        return 1

    try:
        from docproc.config import load_config
        from docproc.pipeline import extract_document_to_text

        if args.config:
            load_config(args.config)
        else:
            load_config()

        def progress(page: int, total: int, message: str):
            logger.info("%s (%d/%d)", message, page, total)

        full_text = extract_document_to_text(
            input_path, progress_callback=progress
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(full_text, encoding="utf-8")
        logger.info("Wrote %s", output_path)
        return 0
    except Exception as e:
        logger.error("Failed to extract: %s", e, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
