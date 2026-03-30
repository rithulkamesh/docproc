"""Configuration module — single source of truth for docproc.

Load from YAML/JSON config file. One database provider, multiple AI providers.
"""

from docproc.config.loader import get_config, load_config, parse_config
from docproc.config.schema import docprocConfig

__all__ = ["load_config", "get_config", "parse_config", "docprocConfig"]
