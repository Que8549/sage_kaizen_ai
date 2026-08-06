"""
evals/gguf_meta.py

Minimal GGUF metadata reader — header key/value pairs only, no tensor data.

Why hand-rolled rather than the `gguf` package: the gates need exactly one thing
(does this mmproj carry an audio encoder, a vision encoder, or both?) and adding
a dependency to answer it is disproportionate. The header format is small,
stable, and fully specified, and reading only the KV block means a 20 GB model
file is inspected with a few kilobytes of I/O.

Format (little-endian throughout):

    magic       4 bytes   b"GGUF"
    version     uint32
    n_tensors   uint64
    n_kv        uint64
    kv pairs    n_kv x (key:string, value_type:uint32, value)

Strings are ``uint64`` length followed by that many UTF-8 bytes. Arrays are
``elem_type:uint32``, ``count:uint64``, then the elements.

Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

__all__ = ["GgufMetadata", "GgufError", "read_gguf_metadata"]

_MAGIC = b"GGUF"

# Value type enum -> (struct format, byte width). Arrays (9) are handled apart.
_SCALAR: dict[int, tuple[str, int]] = {
    0:  ("<B", 1),   # UINT8
    1:  ("<b", 1),   # INT8
    2:  ("<H", 2),   # UINT16
    3:  ("<h", 2),   # INT16
    4:  ("<I", 4),   # UINT32
    5:  ("<i", 4),   # INT32
    6:  ("<f", 4),   # FLOAT32
    7:  ("<?", 1),   # BOOL
    10: ("<Q", 8),   # UINT64
    11: ("<q", 8),   # INT64
    12: ("<d", 8),   # FLOAT64
}
_TYPE_STRING = 8
_TYPE_ARRAY = 9

# Refuse absurd lengths rather than trying to allocate them — a truncated or
# non-GGUF file otherwise reads a garbage length and asks for gigabytes.
_MAX_STRING_BYTES = 64 * 1024 * 1024
_MAX_ARRAY_ITEMS = 16 * 1024 * 1024
_MAX_KV_PAIRS = 1_000_000


class GgufError(ValueError):
    """Raised when a file is not valid GGUF or its header cannot be parsed."""


@dataclass(frozen=True)
class GgufMetadata:
    """Parsed GGUF header. `kv` holds every metadata key/value pair."""

    path: Path
    version: int
    tensor_count: int
    kv: dict[str, Any]

    # ── Convenience accessors used by the gates ───────────────────────────── #

    @property
    def architecture(self) -> str:
        return str(self.kv.get("general.architecture", ""))

    @property
    def name(self) -> str:
        return str(self.kv.get("general.name", ""))

    @property
    def has_vision_encoder(self) -> bool:
        """True when this file carries a vision projector (an mmproj)."""
        return bool(self.kv.get("clip.has_vision_encoder", False))

    @property
    def has_audio_encoder(self) -> bool:
        """True when this file carries an audio projector."""
        return bool(self.kv.get("clip.has_audio_encoder", False))

    @property
    def is_multimodal_projector(self) -> bool:
        return self.has_vision_encoder or self.has_audio_encoder

    @property
    def projector_type(self) -> str:
        return str(self.kv.get("clip.projector_type", ""))

    @property
    def block_count(self) -> int | None:
        """
        Transformer block count, if declared.

        For Qwen3.x MTP models this exceeds the transformer layer count by the
        number of MTP prediction heads — that difference is how brains.yaml
        derives `spec_draft_n_max`.
        """
        for key, value in self.kv.items():
            if key.endswith(".block_count"):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
        return None

    def get(self, key: str, default: Any = None) -> Any:
        return self.kv.get(key, default)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def _read_exact(fh: BinaryIO, n: int) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise GgufError(f"unexpected end of file (wanted {n} bytes, got {len(data)})")
    return data


def _read_scalar(fh: BinaryIO, value_type: int) -> Any:
    fmt, width = _SCALAR[value_type]
    return struct.unpack(fmt, _read_exact(fh, width))[0]


def _read_string(fh: BinaryIO) -> str:
    (length,) = struct.unpack("<Q", _read_exact(fh, 8))
    if length > _MAX_STRING_BYTES:
        raise GgufError(f"implausible string length {length} — file is probably not GGUF")
    return _read_exact(fh, length).decode("utf-8", errors="replace")


def _read_value(fh: BinaryIO, value_type: int) -> Any:
    if value_type in _SCALAR:
        return _read_scalar(fh, value_type)
    if value_type == _TYPE_STRING:
        return _read_string(fh)
    if value_type == _TYPE_ARRAY:
        (elem_type,) = struct.unpack("<I", _read_exact(fh, 4))
        (count,) = struct.unpack("<Q", _read_exact(fh, 8))
        if count > _MAX_ARRAY_ITEMS:
            raise GgufError(f"implausible array length {count} — file is probably not GGUF")
        return [_read_value(fh, elem_type) for _ in range(count)]
    raise GgufError(f"unknown GGUF value type {value_type}")


def read_gguf_metadata(path: str | Path, *, max_kv: int = _MAX_KV_PAIRS) -> GgufMetadata:
    """
    Read a GGUF file's header metadata.

    Only the header is read — tensor data is never touched, so this is fast even
    on a 20 GB model file.

    Raises
    ------
    FileNotFoundError : the path does not exist
    GgufError         : not a GGUF file, or the header is malformed
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"GGUF file not found: {p}")

    with p.open("rb") as fh:
        magic = _read_exact(fh, 4)
        if magic != _MAGIC:
            raise GgufError(f"not a GGUF file: magic is {magic!r}, expected {_MAGIC!r}")

        version, tensor_count, kv_count = struct.unpack("<IQQ", _read_exact(fh, 20))
        if kv_count > max_kv:
            raise GgufError(f"implausible metadata count {kv_count}")

        kv: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_string(fh)
            (value_type,) = struct.unpack("<I", _read_exact(fh, 4))
            kv[key] = _read_value(fh, value_type)

    return GgufMetadata(path=p, version=version, tensor_count=tensor_count, kv=kv)
