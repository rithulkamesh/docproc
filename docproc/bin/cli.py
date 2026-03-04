"""CLI: extract document to markdown (docproc --file input.pdf -o output.md)."""

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

import shtab
from tqdm import tqdm

from docproc.doc.loaders import get_page_count, get_supported_extensions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _run_init_config():
    """Populate ~/.config/docproc/docproc.yml from .env (one-time)."""
    import os
    import yaml
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(prog="docproc init-config")
    parser.add_argument("--env", type=str, default=".env", help="Path to .env file")
    args = parser.parse_args(sys.argv[2:])
    load_dotenv(args.env)
    cfg_dir = Path.home() / ".config" / "docproc"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg_dir / "docproc.yml"
    raw = {}
    if os.getenv("AZURE_OPENAI_API_KEY"):
        raw["ai_providers"] = [
            {
                "provider": "azure",
                "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "default_model": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
                "default_vision_model": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
                "extra": {
                    "azure_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                    "azure_embedding_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
                    "azure_vision_endpoint": os.getenv("AZURE_VISION_ENDPOINT"),
                },
            }
        ]
        raw["primary_ai"] = "azure"
    elif os.getenv("OPENAI_API_KEY"):
        raw["ai_providers"] = [{"provider": "openai", "api_key": os.getenv("OPENAI_API_KEY")}]
        raw["primary_ai"] = "openai"
    elif os.getenv("ANTHROPIC_API_KEY"):
        raw["ai_providers"] = [{"provider": "anthropic", "api_key": os.getenv("ANTHROPIC_API_KEY")}]
        raw["primary_ai"] = "anthropic"
    else:
        raw["ai_providers"] = [
            {"provider": "ollama", "base_url": "http://localhost:11434", "default_vision_model": "llava"}
        ]
        raw["primary_ai"] = "ollama"
    if os.getenv("DATABASE_URL"):
        raw["database"] = {"provider": "pgvector", "connection_string": os.getenv("DATABASE_URL")}
    else:
        raw["database"] = {"provider": "memory"}
    raw["rag"] = {"backend": "clara", "top_k": 5}
    raw["ingest"] = {"use_vision": True, "use_llm_refine": True}
    with open(out_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
    logger.info("Wrote %s", out_path)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract document to markdown (vision + optional LLM refine)"
    )
    parser.add_argument("--file", type=str, required=True, help="Path to input document")
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output markdown path (.md)"
    )
    parser.add_argument("--config", type=str, help="Path to docproc config (docproc.yaml)")
    parser.add_argument(
        "--progress-file",
        type=str,
        default=None,
        help="If set, append JSON lines {page, total, message} for each progress update (for demo worker)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def _get_completion_parser():
    """Parser used for shtab completion generation (matches main extract args)."""
    parser = argparse.ArgumentParser(prog="docproc")
    parser.add_argument("--file", "-f", help="Input document").complete = shtab.FILE
    parser.add_argument("-o", "--output", help="Output .md path").complete = shtab.FILE
    parser.add_argument("--config", help="Config file path").complete = shtab.FILE
    parser.add_argument("--progress-file", help="Progress JSON-lines file path").complete = shtab.FILE
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _run_completions():
    """Print shell completion script. Usage: docproc completions [bash|zsh]."""
    parser = _get_completion_parser()
    shell = sys.argv[2] if len(sys.argv) > 2 else "bash"
    if shell not in ("bash", "zsh"):
        shell = "bash"
    print(shtab.complete(parser, shell=shell))
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "init-config":
        return _run_init_config()
    if len(sys.argv) > 1 and sys.argv[1] == "completions":
        return _run_completions()
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

        # Single-line UX: suppress all logs during extraction
        _log = logging.getLogger
        _quiet = [_log("httpx"), _log("httpcore"), _log("openai"), _log("docproc.extractors.vision_llm")]
        _saved = [g.level for g in _quiet]
        for g in _quiet:
            g.setLevel(logging.WARNING)

        pbar: tqdm | None = None
        spin_idx = [0]
        SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        _C = "\033[36m"
        _G = "\033[32m"
        _Y = "\033[33m"
        _R = "\033[0m"
        stop_spinner = threading.Event()
        spinner_thread: threading.Thread | None = None

        def spinner_loop():
            while not stop_spinner.wait(0.08):
                if pbar is None:
                    continue
                spin_idx[0] = (spin_idx[0] + 1) % len(SPINNER)
                pbar.set_description_str(f"{_C}docproc {SPINNER[spin_idx[0]]}{_R}")
                pbar.refresh()

        progress_file_path = getattr(args, "progress_file", None)

        def progress(page: int, total: int, message: str):
            nonlocal pbar, spinner_thread
            if progress_file_path:
                try:
                    with open(progress_file_path, "a", encoding="utf-8") as pf:
                        pf.write(json.dumps({"page": page, "total": total, "message": message}) + "\n")
                        pf.flush()
                except OSError:
                    pass
            if total == 1 and "Refining" in message:
                if pbar is not None:
                    pbar.n = pbar.total
                    pbar.set_postfix_str("refining…", refresh=False)
                    pbar.refresh()
                return
            if pbar is None:
                pbar = tqdm(
                    total=max(1, total),
                    unit="",
                    desc=f"{_C}docproc {SPINNER[0]}{_R}",
                    bar_format=f"{{desc}} {_G}{{bar}}{_R} {_Y}{{n_fmt}}/{{total_fmt}}{_R} {{postfix}}",
                    dynamic_ncols=True,
                    leave=False,
                    mininterval=0.2,
                    maxinterval=0.5,
                )
                spinner_thread = threading.Thread(target=spinner_loop, daemon=True)
                spinner_thread.start()
            # Only advance; parallel batches complete out of order
            new_n = min(page, pbar.total - 1) if pbar.total else page
            if new_n > pbar.n:
                pbar.n = new_n
            pbar.set_postfix_str(message[:40].strip(), refresh=False)
            pbar.refresh()

        try:
            full_text = extract_document_to_text(
                input_path, progress_callback=progress
            )
        finally:
            stop_spinner.set()
            if spinner_thread is not None:
                spinner_thread.join(timeout=0.5)
            for g, level in zip(_quiet, _saved):
                g.setLevel(level)
            if pbar is not None:
                pbar.close()
        try:
            num_pages = get_page_count(input_path)
        except Exception:
            num_pages = 0
        if num_pages > 0:
            full_text = f"<!-- PAGES: {num_pages} -->\n" + full_text
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(full_text, encoding="utf-8")
        full_path = str(out.resolve())
        sys.stderr.write(f"\r\033[K{_G}✓ Wrote{_R} {full_path}\n")
        return 0
    except Exception as e:
        logger.error("Failed to extract: %s", e, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
