from __future__ import annotations

import importlib.util

import pandas as pd

# suffix -> (pandas reader, module needed for that format, extra that provides it)
_FORMATS = {
    ".csv": (pd.read_csv, None, None),
    ".json": (pd.read_json, None, None),
    ".parquet": (pd.read_parquet, "pyarrow", "parquet"),
    ".feather": (pd.read_feather, "pyarrow", "feather"),
    ".xlsx": (pd.read_excel, "openpyxl", "excel"),
}


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _suffix(source, name) -> str:
    text = name if name is not None else str(source)
    dot = text.rfind(".")
    slash = max(text.rfind("/"), text.rfind("\\"))
    return text[dot:].lower() if dot > slash else ""


def read_dataset(source, name=None):
    """Read a dataset from a path or an in-memory buffer, dispatching on suffix.

    An unknown or missing suffix falls back to CSV, which is what the CLI has
    always done for extensionless input. Optional formats raise ImportError
    naming the extra that provides them instead of a bare pandas error.
    """
    ext = _suffix(source, name)
    reader, module, extra = _FORMATS.get(ext, _FORMATS[".csv"])
    if module and not _available(module):
        raise ImportError(f'{ext} input needs {module}: pip install "leaklens-ml[{extra}]"')
    return reader(source)
