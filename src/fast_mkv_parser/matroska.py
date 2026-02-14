"""Matroska (MKV/MKA/MKS) element IDs, track parsing, and block header decoding.

Reference: https://www.matroska.org/technical/elements.html
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from io import BytesIO
from typing import BinaryIO, Dict, List, Optional, Tuple

from .ebml import (
    UNKNOWN_SIZE,
    read_element_id,
    read_element_size,
    read_vint_value,
)

# ---------------------------------------------------------------------------
# Element ID constants (hex values match the Matroska spec)
# ---------------------------------------------------------------------------

# Top-level
EBML_HEADER = 0x1A45DFA3
SEGMENT = 0x18538067

# Segment children (Level 1)
SEEK_HEAD = 0x114D9B74
SEGMENT_INFO = 0x1549A966
TRACKS = 0x1654AE6B
CUES = 0x1C53BB6B
CLUSTER = 0x1F43B675
CHAPTERS = 0x1043A770
TAGS = 0x1254C367
ATTACHMENTS = 0x1941A469

# SeekHead children
SEEK = 0x4DBB
SEEK_ID = 0x53AB
SEEK_POSITION = 0x53AC

# SegmentInfo children
TIMECODE_SCALE = 0x2AD7B1
DURATION = 0x4489
MUXING_APP = 0x4D80
WRITING_APP = 0x5741
SEGMENT_UID = 0x73A4

# Tracks children
TRACK_ENTRY = 0xAE
TRACK_NUMBER = 0xD7
TRACK_UID = 0x73C5
TRACK_TYPE = 0x83
CODEC_ID = 0x86
CODEC_PRIVATE = 0x63A2
LANGUAGE = 0x22B59C
NAME = 0x536E
DEFAULT_DURATION = 0x23E383
AUDIO = 0xE1
VIDEO = 0xE0

# Audio sub-elements
SAMPLING_FREQUENCY = 0xB5
CHANNELS = 0x9F
BIT_DEPTH = 0x6264

# Cluster children
CLUSTER_TIMESTAMP = 0xE7
SIMPLE_BLOCK = 0xA3
BLOCK_GROUP = 0xA0
BLOCK = 0xA1
BLOCK_DURATION = 0x9B

# Cues children
CUE_POINT = 0xBB
CUE_TIME = 0xB3
CUE_TRACK_POSITIONS = 0xB7
CUE_TRACK = 0xF7
CUE_CLUSTER_POSITION = 0xF1

# Track type values
TRACK_TYPE_VIDEO = 1
TRACK_TYPE_AUDIO = 2
TRACK_TYPE_COMPLEX = 3
TRACK_TYPE_LOGO = 0x10
TRACK_TYPE_SUBTITLE = 0x11
TRACK_TYPE_BUTTONS = 0x12
TRACK_TYPE_CONTROL = 0x20
TRACK_TYPE_METADATA = 0x21

_TRACK_TYPE_NAMES = {
    TRACK_TYPE_VIDEO: "video",
    TRACK_TYPE_AUDIO: "audio",
    TRACK_TYPE_SUBTITLE: "subtitle",
    TRACK_TYPE_COMPLEX: "complex",
    TRACK_TYPE_LOGO: "logo",
    TRACK_TYPE_BUTTONS: "buttons",
    TRACK_TYPE_CONTROL: "control",
    TRACK_TYPE_METADATA: "metadata",
}

# Known container (master) elements whose children we may need to descend into.
_MASTER_ELEMENTS = {
    EBML_HEADER, SEGMENT, SEEK_HEAD, SEEK, SEGMENT_INFO, TRACKS, TRACK_ENTRY,
    CUES, CUE_POINT, CUE_TRACK_POSITIONS, CLUSTER, BLOCK_GROUP, AUDIO, VIDEO,
    CHAPTERS, TAGS, ATTACHMENTS,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrackInfo:
    """Metadata for a single track in the MKV file."""
    number: int
    uid: int
    type: str                  # "video", "audio", "subtitle", etc.
    type_id: int               # Raw TrackType integer
    codec_id: str              # e.g. "V_MPEGH/ISO/HEVC", "A_TRUEHD", "S_HDMV/PGS"
    language: str = "und"
    name: str = ""
    raw_entry: bytes = b""     # Original TrackEntry bytes for passthrough copying

    def __repr__(self) -> str:
        parts = [f"Track {self.number}: {self.type} ({self.codec_id})"]
        if self.language != "und":
            parts.append(f"lang={self.language}")
        if self.name:
            parts.append(f'name="{self.name}"')
        return " ".join(parts)


@dataclass
class CueEntry:
    """A single cue point mapping a timestamp to a cluster byte offset."""
    time: int                  # Cluster-level timestamp (in TimecodeScale units)
    track: int                 # Track number this cue refers to
    cluster_position: int      # Byte offset relative to Segment data start


@dataclass
class SegmentLayout:
    """Byte offsets of key top-level elements within the Segment."""
    segment_data_offset: int = 0    # Byte offset where Segment payload begins
    segment_info_offset: int = 0
    segment_info_size: int = 0
    tracks_offset: int = 0
    tracks_size: int = 0
    cues_offset: int = 0
    cues_size: int = 0
    first_cluster_offset: int = 0   # Absolute file offset of first Cluster


# ---------------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------------

def _read_uint(f: BinaryIO, size: int) -> int:
    """Read an unsigned integer of *size* bytes (big-endian)."""
    data = f.read(size)
    if len(data) < size:
        raise EOFError(f"expected {size} bytes, got {len(data)}")
    val = 0
    for b in data:
        val = (val << 8) | b
    return val


def _read_float(f: BinaryIO, size: int) -> float:
    """Read a big-endian float (4 or 8 bytes)."""
    data = f.read(size)
    if size == 4:
        return struct.unpack(">f", data)[0]
    return struct.unpack(">d", data)[0]


def _read_string(f: BinaryIO, size: int) -> str:
    """Read a UTF-8 string, stripping trailing NULs."""
    data = f.read(size)
    return data.rstrip(b"\x00").decode("utf-8", errors="replace")


def parse_segment_info(f: BinaryIO, info_size: int) -> Dict:
    """Parse a SegmentInfo element body.

    Returns dict with at least 'timecode_scale' (default 1_000_000 ns = 1 ms).
    """
    result: Dict = {"timecode_scale": 1_000_000}
    end = f.tell() + info_size
    while f.tell() < end:
        eid, _ = read_element_id(f)
        esize, _ = read_element_size(f)
        if eid == TIMECODE_SCALE:
            result["timecode_scale"] = _read_uint(f, esize)
        elif eid == DURATION:
            result["duration"] = _read_float(f, esize)
        elif eid == MUXING_APP:
            result["muxing_app"] = _read_string(f, esize)
        elif eid == WRITING_APP:
            result["writing_app"] = _read_string(f, esize)
        elif eid == SEGMENT_UID:
            result["segment_uid"] = f.read(esize)
        else:
            f.seek(esize, 1)
    return result


def _parse_single_track_entry(data: bytes) -> TrackInfo:
    """Parse a single TrackEntry element body from raw bytes."""
    f = BytesIO(data)
    number = 0
    uid = 0
    type_id = 0
    codec_id = ""
    language = "und"
    name = ""
    end = len(data)

    while f.tell() < end:
        eid, _ = read_element_id(f)
        esize, _ = read_element_size(f)
        pos = f.tell()
        if eid == TRACK_NUMBER:
            number = _read_uint(f, esize)
        elif eid == TRACK_UID:
            uid = _read_uint(f, esize)
        elif eid == TRACK_TYPE:
            type_id = _read_uint(f, esize)
        elif eid == CODEC_ID:
            codec_id = _read_string(f, esize)
        elif eid == LANGUAGE:
            language = _read_string(f, esize)
        elif eid == NAME:
            name = _read_string(f, esize)
        else:
            pass  # Skip unknown/unneeded sub-elements
        f.seek(pos + esize)

    type_name = _TRACK_TYPE_NAMES.get(type_id, f"unknown({type_id})")
    return TrackInfo(
        number=number,
        uid=uid,
        type=type_name,
        type_id=type_id,
        codec_id=codec_id,
        language=language,
        name=name,
        raw_entry=data,
    )


def parse_tracks(f: BinaryIO, tracks_size: int) -> List[TrackInfo]:
    """Parse a Tracks element body, returning a list of TrackInfo objects."""
    tracks: List[TrackInfo] = []
    end = f.tell() + tracks_size
    while f.tell() < end:
        eid, _ = read_element_id(f)
        esize, _ = read_element_size(f)
        if eid == TRACK_ENTRY:
            entry_data = f.read(esize)
            tracks.append(_parse_single_track_entry(entry_data))
        else:
            f.seek(esize, 1)
    return tracks


def parse_cues(f: BinaryIO, cues_size: int) -> List[CueEntry]:
    """Parse a Cues element body, returning a list of CueEntry objects."""
    cues: List[CueEntry] = []
    end = f.tell() + cues_size
    while f.tell() < end:
        eid, _ = read_element_id(f)
        esize, _ = read_element_size(f)
        if eid == CUE_POINT:
            cue_end = f.tell() + esize
            time = 0
            track = 0
            cluster_pos = 0
            while f.tell() < cue_end:
                ceid, _ = read_element_id(f)
                cesize, _ = read_element_size(f)
                if ceid == CUE_TIME:
                    time = _read_uint(f, cesize)
                elif ceid == CUE_TRACK_POSITIONS:
                    tp_end = f.tell() + cesize
                    while f.tell() < tp_end:
                        tpid, _ = read_element_id(f)
                        tpsize, _ = read_element_size(f)
                        if tpid == CUE_TRACK:
                            track = _read_uint(f, tpsize)
                        elif tpid == CUE_CLUSTER_POSITION:
                            cluster_pos = _read_uint(f, tpsize)
                        else:
                            f.seek(tpsize, 1)
                else:
                    f.seek(cesize, 1)
            cues.append(CueEntry(time=time, track=track, cluster_position=cluster_pos))
        else:
            f.seek(esize, 1)
    return cues


def parse_block_header(data: bytes) -> Tuple[int, int, int, int]:
    """Parse the header of a SimpleBlock or Block.

    Args:
        data: The first N bytes of the block payload (at least 4 bytes needed).

    Returns:
        (track_number, relative_timestamp, flags, header_size)
        where header_size is the number of bytes consumed by the header.
    """
    f = BytesIO(data)
    track_number, vint_width = read_vint_value(f)
    # Relative timestamp: signed 16-bit big-endian
    ts_bytes = f.read(2)
    relative_timestamp = struct.unpack(">h", ts_bytes)[0]
    # Flags: 1 byte
    flags = f.read(1)[0]
    header_size = vint_width + 3  # VINT + 2 timestamp + 1 flags
    return track_number, relative_timestamp, flags, header_size
