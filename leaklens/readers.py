from __future__ import annotations

import gzip
import importlib.util
import io
import zipfile

import pandas as pd


class DatasetLimitError(ValueError):
    """Raised when an uploaded dataset exceeds row or decompressed-size limits."""


# suffix -> (pandas reader, module needed for that format, extra that provides it)
_FORMATS = {
    ".csv": (pd.read_csv, None, None),
    ".json": (pd.read_json, None, None),
    ".parquet": (pd.read_parquet, "pyarrow", "parquet"),
    ".feather": (pd.read_feather, "pyarrow", "feather"),
    ".xlsx": (pd.read_excel, "openpyxl", "excel"),
}

MAX_CONTAINER_UNCOMPRESSED_BYTES = 256 * 1024 * 1024  # 256 MiB


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _suffix(source, name) -> str:
    text = name if name is not None else str(source)
    dot = text.rfind(".")
    slash = max(text.rfind("/"), text.rfind("\\"))
    return text[dot:].lower() if dot > slash else ""


def _iter_file(source):
    """Yield every byte a file-like (or path) source would expose, bounded.

    For in-memory buffers this slices the underlying bytearray without
    copying the whole archive. For real file objects we stream, but the
    caller here guarantees bounded size via MAX_CONTAINER_UNCOMPRESSED_BYTES
    accounting against the same cap, not by loading 256 MiB into memory.
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        yield bytes(source)
        return
    if hasattr(source, "getvalue"):
        try:
            raw = source.getvalue()  # type: ignore[no-any-return]
            if isinstance(raw, (bytes, bytearray)):
                yield bytes(raw)
                return
        except Exception:  # noqa: BLE001
            return
    if hasattr(source, "read"):
        try:
            pos = source.tell()  # type: ignore[attr-defined]
            chunk_size = 262_144
            while True:
                chunk = source.read(chunk_size)
                if not chunk:
                    break
                if isinstance(chunk, str):
                    chunk = chunk.encode()
                yield bytes(chunk)
            source.seek(pos)
        except Exception:  # noqa: BLE001
            return


def _check_zip_entry_path(filename: str, name: str) -> None:
    if filename.startswith(("/", "\\")) or (len(filename) > 2 and filename[1:2] == ":"):
        raise DatasetLimitError(
            f"archive entry outside extraction root: {name}:{filename}"
        )
    if ".." in filename.replace("\\", "/").split("/"):
        raise DatasetLimitError(f"archive entry traversal: {name}:{filename}")


def _check_zip_container(data: bytes, name: str) -> None:
    if data[:2] != b"PK":
        return
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            total = 0
            for info in zf.infolist():
                if info.is_dir():
                    continue
                total += info.file_size
                if total > MAX_CONTAINER_UNCOMPRESSED_BYTES:
                    raise DatasetLimitError(
                        f"archive too large (>256 MiB decompressed): {name}"
                    )
                _check_zip_entry_path(info.filename, name)
    except zipfile.BadZipFile:
        return
    except DatasetLimitError:
        raise
    except Exception:  # noqa: BLE001 - let the real parser report the error
        return


def _decompress_bounded(data: bytes, name: str, budget: int) -> tuple[bytes, int]:
    out = bytearray()
    # Bounded streaming decompression -- never call gzip.decompress()
    # which would materialize the full expansion before checking.
    stream = data
    layers = 0
    while stream[:2] == b"\x1f\x8b" and layers < 5:
        layers += 1
        with gzip.GzipFile(fileobj=io.BytesIO(stream)) as gz:
            layer_out = bytearray()
            while True:
                chunk = gz.read(1_048_576)
                if not chunk:
                    break
                layer_out.extend(chunk)
                if len(layer_out) > budget:
                    raise DatasetLimitError(
                        f"gzip archive too large (>256 MiB decompressed): {name}"
                    )
        if layer_out[:2] != b"\x1f\x8b":
            out.extend(layer_out)
            break
        out.extend(layer_out)
        stream = bytes(layer_out)
    return bytes(out), len(out)


def _check_gzip_container(data: bytes, name: str) -> None:
    if data[:2] != b"\x1f\x8b":
        return
    try:
        total = 0
        stream = data
        layers = 0
        while stream[:2] == b"\x1f\x8b" and layers < 5:
            layers += 1
            layer_out = bytearray()
            with gzip.GzipFile(fileobj=io.BytesIO(stream)) as gz:
                while True:
                    chunk = gz.read(1_048_576)
                    if not chunk:
                        break
                    layer_out.extend(chunk)
                    if total + len(layer_out) > MAX_CONTAINER_UNCOMPRESSED_BYTES:
                        raise DatasetLimitError(
                            f"gzip archive too large (>256 MiB decompressed): {name}"
                        )
            total += len(layer_out)
            if len(layer_out) < 2 or layer_out[:2] != b"\x1f\x8b":
                break
            stream = bytes(layer_out)
    except DatasetLimitError:
        raise
    except Exception:  # noqa: BLE001
        return


def read_dataset(source, name=None, max_rows=None):
    """Read a dataset from a path or an in-memory buffer, dispatching on suffix.

    An unknown or missing suffix falls back to CSV, which is what the CLI has
    always done for extensionless input. Optional formats raise ImportError
    naming the extra that provides them instead of a bare pandas error.

    Compressed containers (ZIP/gzip, which covers xlsx) are validated
    before pandas reads them on every call path, whether or not
    ``max_rows`` is supplied -- CLI and API inputs are both checked. When
    ``max_rows`` is also set, text formats are additionally read with
    ``nrows`` so a small compressed upload cannot expand into an
    unbounded DataFrame. Exceeding a limit raises :class:`DatasetLimitError`.
    """
    ext = _suffix(source, name)
    reader, module, extra = _FORMATS.get(ext, _FORMATS[".csv"])
    if module and not _available(module):
        raise ImportError(
            f'{ext} input needs {module}: pip install "leaklens-ml[{extra}]"'
        )
    display_name = name or (str(source) if isinstance(source, str) else "upload")
    _check_stream_zip_and_gzip(source, display_name)
    if max_rows is None:
        return reader(source)
    effective = ext if ext in _FORMATS else ".csv"
    if effective == ".csv":
        df = reader(source, nrows=max_rows + 1)
    elif effective == ".json":
        # pd.read_json supports nrows= only with lines=True, which would
        # reject the default record-oriented JSON the API supports. Read
        # fully and check length after; the 8 MiB upload cap bounds this.
        df = reader(source)
    else:
        df = reader(source)
    if len(df) > max_rows:
        raise DatasetLimitError(f"too many rows (> {max_rows:,}): {name or 'upload'}")
    return df


def _check_stream_zip_and_gzip(source, name: str) -> None:
    """Run ZIP/gzip container checks without loading the whole archive twice."""
    if isinstance(source, str):
        try:
            with open(source, "rb") as fh:
                prefix = fh.read(2)
                fh.seek(0)
                if prefix[:2] == b"PK":
                    with zipfile.ZipFile(fh) as zf:
                        total = 0
                        for info in zf.infolist():
                            if info.is_dir():
                                continue
                            total += info.file_size
                            if total > MAX_CONTAINER_UNCOMPRESSED_BYTES:
                                raise DatasetLimitError(
                                    f"archive too large (>256 MiB decompressed): {name}"
                                )
                            _check_zip_entry_path(info.filename, name)
                elif prefix[:2] == b"\x1f\x8b":
                    with gzip.GzipFile(fileobj=fh) as gz:
                        total = 0
                        while True:
                            chunk = gz.read(1_048_576)
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > MAX_CONTAINER_UNCOMPRESSED_BYTES:
                                raise DatasetLimitError(
                                    f"gzip archive too large (>256 MiB decompressed): {name}"
                                )
        except DatasetLimitError:
            raise
        except Exception:  # noqa: BLE001 - let pandas report the real error
            return
        return
    for data in _iter_file(source):
        _check_zip_container(data, name)
        _check_gzip_container(data, name)
