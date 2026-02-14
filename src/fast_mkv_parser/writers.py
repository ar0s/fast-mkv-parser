"""Output writers for extracted MKV track data.

MkvWriter:  Produces a valid Matroska container with only the selected tracks.
SupWriter:  Reconstructs Blu-ray SUP format from MKV PGS subtitle blocks.
"""

from __future__ import annotations

import struct
from typing import BinaryIO, List

from .ebml import (
    UNKNOWN_SIZE,
    encode_element_id,
    encode_element_size,
)
from . import matroska as mkv


class MkvWriter:
    """Write a valid Matroska container with selected tracks only.

    Output structure:
      EBML Header (copied from source)
      Segment (unknown size — streaming style)
        SegmentInfo (copied from source)
        Tracks (only entries for desired tracks)
        Cluster*  (known size, buffered)
          Timestamp
          SimpleBlock* / BlockGroup* (only desired tracks)
    """

    def __init__(self, f: BinaryIO):
        self._f = f
        self._cluster_buf: List[bytes] = []  # buffered cluster body chunks
        self._cluster_ts: int = 0

    def write_header(
        self,
        ebml_header_bytes: bytes,
        segment_info_bytes: bytes,
        track_entries: List[bytes],
    ) -> None:
        """Write the file header: EBML Header + Segment + SegmentInfo + Tracks.

        Args:
            ebml_header_bytes: Complete EBML Header element (ID + size + body).
            segment_info_bytes: Raw body of the SegmentInfo element.
            track_entries: List of raw TrackEntry element bodies (one per desired track).
        """
        f = self._f

        # 1. EBML Header — copy verbatim
        f.write(ebml_header_bytes)

        # 2. Segment — unknown size (streaming)
        f.write(encode_element_id(mkv.SEGMENT))
        f.write(encode_element_size(UNKNOWN_SIZE, width=8))

        # 3. SegmentInfo
        f.write(encode_element_id(mkv.SEGMENT_INFO))
        f.write(encode_element_size(len(segment_info_bytes)))
        f.write(segment_info_bytes)

        # 4. Tracks
        tracks_body = b""
        for entry_body in track_entries:
            tracks_body += encode_element_id(mkv.TRACK_ENTRY)
            tracks_body += encode_element_size(len(entry_body))
            tracks_body += entry_body
        f.write(encode_element_id(mkv.TRACKS))
        f.write(encode_element_size(len(tracks_body)))
        f.write(tracks_body)

    def begin_cluster(self, timestamp: int) -> None:
        """Start buffering a new Cluster."""
        self._flush_cluster()
        self._cluster_ts = timestamp
        self._cluster_buf = []

        # Buffer the Cluster Timestamp element.
        ts_bytes = _encode_uint(timestamp)
        self._cluster_buf.append(
            encode_element_id(mkv.CLUSTER_TIMESTAMP)
            + encode_element_size(len(ts_bytes))
            + ts_bytes
        )

    def write_simple_block(self, block_data: bytes) -> None:
        """Buffer a SimpleBlock element."""
        self._cluster_buf.append(
            encode_element_id(mkv.SIMPLE_BLOCK)
            + encode_element_size(len(block_data))
            + block_data
        )

    def write_block_group(self, block_group_data: bytes) -> None:
        """Buffer a BlockGroup element."""
        self._cluster_buf.append(
            encode_element_id(mkv.BLOCK_GROUP)
            + encode_element_size(len(block_group_data))
            + block_group_data
        )

    def _flush_cluster(self) -> None:
        """Write the buffered Cluster with a known size."""
        if not self._cluster_buf:
            return
        cluster_body = b"".join(self._cluster_buf)
        self._f.write(encode_element_id(mkv.CLUSTER))
        self._f.write(encode_element_size(len(cluster_body)))
        self._f.write(cluster_body)
        self._cluster_buf = []

    def finalize(self) -> None:
        """Flush any remaining buffered Cluster and finalize."""
        self._flush_cluster()
        self._f.flush()


class SupWriter:
    """Write PGS (Presentation Graphic Stream) subtitles in Blu-ray SUP format.

    SUP segment format (repeated):
      "PG"            2 bytes   magic
      PTS             4 bytes   presentation timestamp (90 kHz clock, big-endian)
      DTS             4 bytes   decoding timestamp (big-endian, usually 0)
      segment_type    1 byte    PGS segment type
      segment_size    2 bytes   payload length (big-endian)
      payload         N bytes

    MKV stores PGS blocks without the "PG" + PTS + DTS header.  Each block
    payload may contain one or more raw PGS segments (type + size + data).
    We reconstruct the SUP header using the block's MKV timestamp.
    """

    def __init__(self, f: BinaryIO, timecode_scale_ns: int = 1_000_000):
        self._f = f
        self._timecode_scale_ns = timecode_scale_ns

    def write_block(
        self,
        cluster_timestamp: int,
        relative_timestamp: int,
        payload: bytes,
    ) -> None:
        """Write one MKV block's PGS data as SUP segments.

        Args:
            cluster_timestamp: Cluster-level timestamp (in TimecodeScale units).
            relative_timestamp: Block-relative timestamp (signed int16).
            payload: Raw block payload (after track number + timestamp + flags).
        """
        # Compute absolute timestamp in TimecodeScale units, then convert to 90 kHz.
        abs_ts = cluster_timestamp + relative_timestamp
        # TimecodeScale is in nanoseconds.  90 kHz = 90000 ticks/second.
        pts_90khz = int(abs_ts * self._timecode_scale_ns / 1_000_000_000 * 90_000)
        # Clamp to 32-bit unsigned for SUP format.
        pts_90khz = pts_90khz & 0xFFFFFFFF

        # Parse PGS segments from the block payload.
        offset = 0
        while offset < len(payload):
            if offset + 3 > len(payload):
                break
            seg_type = payload[offset]
            seg_size = struct.unpack(">H", payload[offset + 1 : offset + 3])[0]
            seg_data = payload[offset + 3 : offset + 3 + seg_size]

            # Write SUP segment: PG + PTS + DTS + type + size + data
            self._f.write(b"PG")
            self._f.write(struct.pack(">I", pts_90khz))
            self._f.write(struct.pack(">I", 0))  # DTS = 0
            self._f.write(bytes([seg_type]))
            self._f.write(struct.pack(">H", seg_size))
            self._f.write(seg_data)

            offset += 3 + seg_size

    def finalize(self) -> None:
        """Finalize the SUP output."""
        self._f.flush()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_uint(value: int) -> bytes:
    """Encode an unsigned integer in the minimum number of bytes (big-endian)."""
    if value == 0:
        return b"\x00"
    byte_length = (value.bit_length() + 7) // 8
    return value.to_bytes(byte_length, "big")
