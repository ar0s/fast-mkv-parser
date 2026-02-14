"""Unit tests for Matroska element parsing."""

import io
import struct
import pytest

from fast_mkv_parser.ebml import encode_element_id, encode_element_size, encode_vint_value
from fast_mkv_parser.matroska import (
    TRACK_ENTRY, TRACK_NUMBER, TRACK_TYPE, TRACK_UID, CODEC_ID, LANGUAGE,
    TRACK_TYPE_VIDEO, TRACK_TYPE_AUDIO, TRACK_TYPE_SUBTITLE,
    TIMECODE_SCALE, SEGMENT_INFO,
    parse_tracks, parse_block_header, parse_segment_info,
    _read_uint,
)


def _make_uint_element(eid: int, value: int) -> bytes:
    """Build a complete EBML element with a uint payload."""
    if value == 0:
        payload = b"\x00"
    else:
        byte_len = (value.bit_length() + 7) // 8
        payload = value.to_bytes(byte_len, "big")
    return encode_element_id(eid) + encode_element_size(len(payload)) + payload


def _make_string_element(eid: int, value: str) -> bytes:
    """Build a complete EBML element with a string payload."""
    payload = value.encode("utf-8")
    return encode_element_id(eid) + encode_element_size(len(payload)) + payload


def _make_track_entry(number: int, type_id: int, codec_id: str,
                      language: str = "eng", uid: int = 12345) -> bytes:
    """Build a complete TrackEntry element body."""
    body = b""
    body += _make_uint_element(TRACK_NUMBER, number)
    body += _make_uint_element(TRACK_UID, uid)
    body += _make_uint_element(TRACK_TYPE, type_id)
    body += _make_string_element(CODEC_ID, codec_id)
    body += _make_string_element(LANGUAGE, language)
    return body


class TestParseTracks:
    def test_single_video_track(self):
        entry_body = _make_track_entry(1, TRACK_TYPE_VIDEO, "V_MPEGH/ISO/HEVC")
        # Wrap in TrackEntry element
        tracks_body = (
            encode_element_id(TRACK_ENTRY)
            + encode_element_size(len(entry_body))
            + entry_body
        )
        f = io.BytesIO(tracks_body)
        tracks = parse_tracks(f, len(tracks_body))
        assert len(tracks) == 1
        assert tracks[0].number == 1
        assert tracks[0].type == "video"
        assert tracks[0].codec_id == "V_MPEGH/ISO/HEVC"
        assert tracks[0].language == "eng"

    def test_multiple_tracks(self):
        entries = []
        for num, type_id, codec in [
            (1, TRACK_TYPE_VIDEO, "V_MPEGH/ISO/HEVC"),
            (2, TRACK_TYPE_AUDIO, "A_TRUEHD"),
            (3, TRACK_TYPE_AUDIO, "A_AAC"),
            (5, TRACK_TYPE_SUBTITLE, "S_HDMV/PGS"),
        ]:
            body = _make_track_entry(num, type_id, codec, uid=num * 1000)
            entries.append(
                encode_element_id(TRACK_ENTRY)
                + encode_element_size(len(body))
                + body
            )
        tracks_body = b"".join(entries)
        f = io.BytesIO(tracks_body)
        tracks = parse_tracks(f, len(tracks_body))
        assert len(tracks) == 4
        assert tracks[0].type == "video"
        assert tracks[1].type == "audio"
        assert tracks[1].codec_id == "A_TRUEHD"
        assert tracks[2].codec_id == "A_AAC"
        assert tracks[3].type == "subtitle"
        assert tracks[3].number == 5


class TestParseBlockHeader:
    def test_track_1_positive_ts(self):
        # Track 1 (VINT: 0x81), timestamp +100 (0x0064), flags 0x00
        data = b"\x81" + struct.pack(">h", 100) + b"\x00"
        track, ts, flags, hdr_size = parse_block_header(data)
        assert track == 1
        assert ts == 100
        assert flags == 0
        assert hdr_size == 4  # 1 + 2 + 1

    def test_track_2_negative_ts(self):
        # Track 2 (VINT: 0x82), timestamp -50
        data = b"\x82" + struct.pack(">h", -50) + b"\x80"  # keyframe flag
        track, ts, flags, hdr_size = parse_block_header(data)
        assert track == 2
        assert ts == -50
        assert flags == 0x80

    def test_track_5_zero_ts(self):
        data = b"\x85" + struct.pack(">h", 0) + b"\x00"
        track, ts, flags, hdr_size = parse_block_header(data)
        assert track == 5
        assert ts == 0

    def test_large_track_number(self):
        # Track 200 needs width-2 VINT: 0x40 | 200 = 0x40C8
        data = b"\x40\xC8" + struct.pack(">h", 0) + b"\x00"
        track, ts, flags, hdr_size = parse_block_header(data)
        assert track == 200
        assert hdr_size == 5  # 2 + 2 + 1


class TestParseSegmentInfo:
    def test_default_timecode_scale(self):
        """Empty body should yield default timecode scale."""
        f = io.BytesIO(b"")
        result = parse_segment_info(f, 0)
        assert result["timecode_scale"] == 1_000_000

    def test_custom_timecode_scale(self):
        body = _make_uint_element(TIMECODE_SCALE, 500_000)
        f = io.BytesIO(body)
        result = parse_segment_info(f, len(body))
        assert result["timecode_scale"] == 500_000
