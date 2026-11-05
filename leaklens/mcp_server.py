"""Read-only MCP integration for the LeakLens Python/notebook scanner.

Install the optional mcp extra and run python -m leaklens.mcp_server.
The default stdio transport keeps the server local to its MCP host.
"""

from __future__ import annotations

import os
from pathlib import Path

from .leakage import score
from .scanner import scan_file, scan_source

MAX_INPUT_BYTES = 512 * 1024
MAX_FILE_BYTES = 2 * 1024 * 1024
ALLOWED_SUFFIXES = {".py", ".ipynb"}


def _summary(findings) -> dict:
    return {
        "trust_score": score(findings),
        "finding_count": len(findings),
        "findings": [f.to_dict() for f in findings],
    }


def scan_python_code(code: str, filename: str = "input.py") -> dict:
    """Find data leaks in a Python source snippet, without executing it.

    Args:
        code: Python code to analyze.
        filename: Display label used in findings, not a filesystem path.
    """
    if len(code.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ValueError("Python source exceeds the 512 KiB input limit")
    if not filename or len(filename) > 256:
        raise ValueError("filename must be 1-256 characters")
    return _summary(scan_source(code, filename))


def scan_local_file(path: str) -> dict:
    """Scan one .py or .ipynb file within the configured local root.

    Relative paths are resolved inside LEAKLENS_MCP_ROOT (default: the
    server's current directory). Paths escaping that directory, including
    through symlinks, are rejected. Never executes code in the file.
    """
    root = Path(os.environ.get("LEAKLENS_MCP_ROOT") or os.getcwd()).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("LEAKLENS_MCP_ROOT must be an existing directory")
    if not path:
        raise ValueError("path cannot be empty")
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        candidate = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("File does not exist or cannot be accessed") from exc
    if not candidate.is_relative_to(root):
        raise ValueError("File is outside LEAKLENS_MCP_ROOT")
    if not candidate.is_file() or candidate.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError("Only existing .py and .ipynb files are supported")
    if candidate.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("File exceeds the 2 MiB limit")
    return _summary(scan_file(candidate))


def create_server():
    """Initialize the optional MCP SDK and register the scanner tools."""
    try:
        from mcp.server import MCPServer
    except ImportError as exc:
        raise RuntimeError(
            "MCP support needs Python 3.11+ and 'pip install leaklens-ml[mcp]'"
        ) from exc

    server = MCPServer("LeakLens")
    server.tool()(scan_python_code)
    server.tool()(scan_local_file)
    return server


def main() -> None:
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
