"""EBML (Extensible Binary Meta Language) primitives.

Implements variable-length integer (VINT) encoding/decoding and element
ID/size reading/writing per the EBML RFC 8794 specification.

VINT layout:
  Width 1: 1xxxxxxx                       (7 data bits)
  Width 2: 01xxxxxx xxxxxxxx              (14 data bits)
  Width 3: 001xxxxx xxxxxxxx xxxxxxxx     (21 data bits)
  Width 4: 0001xxxx xxxxxxxx^3            (28 data bits)
  ...up to width 8.

The leading zero count determines byte width.  The first '1' bit is
the VINT_MARKER.  For element sizes, the marker is stripped (data only).
For element IDs, the marker is kept (it's part of the ID value).
"""

from __future__ import annotations

import struct
from typing import BinaryIO, Tuple

# Sentinel: EBML "unknown size" (all data bits set after marker).
UNKNOWN_SIZE = -1

# VINT_MARKER masks indexed by width (1-8).
# _MARKER_MASKS[w] = the marker bit position for width w.
_MARKER_MASKS = [
    0,  # unused index 0
    0x80,
    0x4000,
    0x200000,
    0x10000000,
    0x0800000000,
    0x040000000000,
    0x02000000000000,
    0x0100000000000000,
]

# "All ones" values (unknown size) for each width.
_ALL_ONES = [
    0,
    0x7F,
    0x3FFF,
    0x1FFFFF,
    0x0FFFFFFF,
    0x07FFFFFFFF,
    0x03FFFFFFFFFF,
    0x01FFFFFFFFFFFF,
    0x00FFFFFFFFFFFFFF,
]


def read_vint_raw(f: BinaryIO) -> Tuple[int, int]:
    """Read a raw VINT from *f* (marker bit included in value).

    Returns (raw_value, byte_width).
    Raises ``EOFError`` if no bytes available, ``ValueError`` on invalid
    leading byte (0x00).
    """
    first = f.read(1)
    if not first:
        raise EOFError("unexpected end of stream reading VINT")
    b0 = first[0]
    if b0 == 0:
        raise ValueError("invalid VINT leading byte 0x00")

    # Count leading zeros to determine width.
    width = 1
    mask = 0x80
    while (b0 & mask) == 0:
        width += 1
        mask >>= 1

    if width == 1:
        return b0, 1

    remaining = f.read(width - 1)
    if len(remaining) < width - 1:
        raise EOFError(f"truncated VINT: expected {width} bytes total")

    value = b0
    for byte in remaining:
        value = (value << 8) | byte
    return value, width


def read_element_id(f: BinaryIO) -> Tuple[int, int]:
    """Read an EBML Element ID (VINT with marker bit kept).

    Returns (element_id, byte_width).
    """
    return read_vint_raw(f)


def read_element_size(f: BinaryIO) -> Tuple[int, int]:
    """Read an EBML Element Size (VINT with marker bit stripped).

    Returns (size, byte_width).  Returns ``UNKNOWN_SIZE`` (-1) when all
    data bits are set (streaming / unknown length).
    """
    raw, width = read_vint_raw(f)
    # Strip the marker bit.
    value = raw & ~_MARKER_MASKS[width]
    # Check for "all ones" = unknown size.
    if value == _ALL_ONES[width]:
        return UNKNOWN_SIZE, width
    return value, width


def read_vint_value(f: BinaryIO) -> Tuple[int, int]:
    """Read a VINT used as a data value (marker stripped), e.g. track number
    inside a SimpleBlock header.

    Returns (value, byte_width).
    """
    raw, width = read_vint_raw(f)
    return raw & ~_MARKER_MASKS[width], width


# ---------------------------------------------------------------------------
# Writing helpers
# ---------------------------------------------------------------------------

def _byte_width_for_id(element_id: int) -> int:
    """Determine the byte width of an encoded element ID."""
    if element_id <= 0xFF:
        return 1
    elif element_id <= 0xFFFF:
        return 2
    elif element_id <= 0xFFFFFF:
        return 3
    else:
        return 4


def encode_element_id(element_id: int) -> bytes:
    """Encode an Element ID to bytes (marker bit already embedded in the ID value)."""
    width = _byte_width_for_id(element_id)
    return element_id.to_bytes(width, "big")


def _min_size_width(value: int) -> int:
    """Minimum VINT width needed to represent *value* as an element size."""
    for w in range(1, 9):
        if value < _ALL_ONES[w]:
            return w
    raise OverflowError(f"size {value} too large for EBML VINT")


def encode_element_size(size: int, *, width: int = 0) -> bytes:
    """Encode an element size as an EBML VINT.

    If *width* is 0, the minimum width is chosen.  Use width=8 to force
    8-byte encoding (useful for unknown-size Segment elements).
    """
    if size == UNKNOWN_SIZE:
        # Unknown size: all bits set.
        w = width or 1
        return (_MARKER_MASKS[w] | _ALL_ONES[w]).to_bytes(w, "big")

    w = width or _min_size_width(size)
    raw = _MARKER_MASKS[w] | size
    return raw.to_bytes(w, "big")


def encode_vint_value(value: int, *, width: int = 0) -> bytes:
    """Encode a data-value VINT (e.g. track number in a block header)."""
    if width == 0:
        for w in range(1, 9):
            if value < _ALL_ONES[w]:
                width = w
                break
        else:
            raise OverflowError(f"value {value} too large for EBML VINT")
    raw = _MARKER_MASKS[width] | value
    return raw.to_bytes(width, "big")


def peek_element_id(f: BinaryIO) -> Tuple[int, int]:
    """Peek at the next element ID without consuming it.

    Returns (element_id, byte_width).  Restores the file position.
    """
    pos = f.tell()
    try:
        return read_element_id(f)
    finally:
        f.seek(pos)
