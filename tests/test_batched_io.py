"""Tests for two-phase batched I/O extraction.

Covers:
- _merge_regions logic (gap merging, region splitting)
- _split_region helper
- End-to-end: batched vs single-pass produces byte-identical output
"""

import io
import struct
import tempfile
import os

import pytest

from fast_mkv_parser.ebml import (
    encode_element_id,
    encode_element_size,
    encode_vint_value,
    UNKNOWN_SIZE,
)
from fast_mkv_parser import matroska as mkv
from fast_mkv_parser.extractor import (
    BlockPlan,
    ReadRegion,
    MkvParser,
    _split_region,
    _scan_block_range,
    _MAX_REGION_READ,
)


# ---------------------------------------------------------------------------
# Helpers to construct synthetic MKV data
# ---------------------------------------------------------------------------

def _make_uint_element(eid: int, value: int) -> bytes:
    if value == 0:
        payload = b"\x00"
    else:
        byte_len = (value.bit_length() + 7) // 8
        payload = value.to_bytes(byte_len, "big")
    return encode_element_id(eid) + encode_element_size(len(payload)) + payload


def _make_string_element(eid: int, value: str) -> bytes:
    payload = value.encode("utf-8")
    return encode_element_id(eid) + encode_element_size(len(payload)) + payload


def _make_track_entry_body(number: int, type_id: int, codec_id: str,
                           language: str = "eng", uid: int = 12345) -> bytes:
    body = b""
    body += _make_uint_element(mkv.TRACK_NUMBER, number)
    body += _make_uint_element(mkv.TRACK_UID, uid)
    body += _make_uint_element(mkv.TRACK_TYPE, type_id)
    body += _make_string_element(mkv.CODEC_ID, codec_id)
    body += _make_string_element(mkv.LANGUAGE, language)
    return body


def _make_simple_block(track: int, rel_ts: int, flags: int, payload: bytes) -> bytes:
    """Build a SimpleBlock body (track VINT + timestamp + flags + payload)."""
    body = encode_vint_value(track) + struct.pack(">h", rel_ts) + bytes([flags]) + payload
    return body


def _make_block_group(track: int, rel_ts: int, flags: int, payload: bytes,
                      duration: int = None) -> bytes:
    """Build a BlockGroup body containing a Block element and optional BlockDuration."""
    # Block element
    block_body = encode_vint_value(track) + struct.pack(">h", rel_ts) + bytes([flags]) + payload
    block_elem = (
        encode_element_id(mkv.BLOCK)
        + encode_element_size(len(block_body))
        + block_body
    )
    group_body = block_elem
    if duration is not None:
        group_body += _make_uint_element(mkv.BLOCK_DURATION, duration)
    return group_body


def _make_cluster(timestamp: int, blocks: list) -> bytes:
    """Build a Cluster element.

    *blocks* is a list of (element_id, body_bytes) tuples — either
    (SIMPLE_BLOCK, body) or (BLOCK_GROUP, body).
    """
    body = _make_uint_element(mkv.CLUSTER_TIMESTAMP, timestamp)
    for eid, block_body in blocks:
        body += encode_element_id(eid) + encode_element_size(len(block_body)) + block_body
    return encode_element_id(mkv.CLUSTER) + encode_element_size(len(body)) + body


def _make_cue_point(time: int, track: int, cluster_position: int) -> bytes:
    """Build a CuePoint element."""
    # CueTrackPositions
    tp_body = (
        _make_uint_element(mkv.CUE_TRACK, track)
        + _make_uint_element(mkv.CUE_CLUSTER_POSITION, cluster_position)
    )
    tp_elem = (
        encode_element_id(mkv.CUE_TRACK_POSITIONS)
        + encode_element_size(len(tp_body))
        + tp_body
    )
    # CuePoint
    cp_body = _make_uint_element(mkv.CUE_TIME, time) + tp_elem
    return (
        encode_element_id(mkv.CUE_POINT)
        + encode_element_size(len(cp_body))
        + cp_body
    )


def _make_cues_element(cue_points: list) -> bytes:
    """Build a Cues element from a list of (time, track, cluster_position) tuples."""
    body = b""
    for time, track, cluster_pos in cue_points:
        body += _make_cue_point(time, track, cluster_pos)
    return (
        encode_element_id(mkv.CUES)
        + encode_element_size(len(body))
        + body
    )


def _make_seek_entry(seek_id: int, seek_position: int) -> bytes:
    """Build a Seek element (child of SeekHead)."""
    id_bytes = encode_element_id(seek_id)
    body = (
        encode_element_id(mkv.SEEK_ID)
        + encode_element_size(len(id_bytes))
        + id_bytes
        + _make_uint_element(mkv.SEEK_POSITION, seek_position)
    )
    return encode_element_id(mkv.SEEK) + encode_element_size(len(body)) + body


def _make_seekhead(entries: list) -> bytes:
    """Build a SeekHead element from (element_id, seek_position) tuples."""
    body = b""
    for eid, pos in entries:
        body += _make_seek_entry(eid, pos)
    return encode_element_id(mkv.SEEK_HEAD) + encode_element_size(len(body)) + body


def _make_minimal_mkv(
    tracks: list,
    clusters_bytes: bytes,
    timecode_scale: int = 1_000_000,
    cues: list = None,
) -> bytes:
    """Build a minimal MKV file with EBML Header + Segment.

    *tracks* is a list of (number, type_id, codec_id) tuples.
    *clusters_bytes* is pre-built Cluster element bytes.
    *cues* is an optional list of (time, track, cluster_position) tuples.
    cluster_position values are relative to the start of the Segment body.
    """
    # EBML Header
    ebml_body = b""
    ebml_body += _make_uint_element(0x4286, 1)      # EBMLVersion
    ebml_body += _make_uint_element(0x42F7, 1)      # EBMLReadVersion
    ebml_body += _make_uint_element(0x42F2, 4)      # EBMLMaxIDLength
    ebml_body += _make_uint_element(0x42F3, 8)      # EBMLMaxSizeLength
    ebml_body += _make_string_element(0x4282, "matroska")  # DocType
    ebml_body += _make_uint_element(0x4287, 4)      # DocTypeVersion
    ebml_body += _make_uint_element(0x4285, 2)      # DocTypeReadVersion
    ebml_header = (
        encode_element_id(mkv.EBML_HEADER)
        + encode_element_size(len(ebml_body))
        + ebml_body
    )

    # SegmentInfo
    seg_info_body = _make_uint_element(mkv.TIMECODE_SCALE, timecode_scale)
    seg_info = (
        encode_element_id(mkv.SEGMENT_INFO)
        + encode_element_size(len(seg_info_body))
        + seg_info_body
    )

    # Tracks
    tracks_body = b""
    for num, type_id, codec_id in tracks:
        entry_body = _make_track_entry_body(num, type_id, codec_id, uid=num * 1000)
        tracks_body += (
            encode_element_id(mkv.TRACK_ENTRY)
            + encode_element_size(len(entry_body))
            + entry_body
        )
    tracks_elem = (
        encode_element_id(mkv.TRACKS)
        + encode_element_size(len(tracks_body))
        + tracks_body
    )

    # Cues (optional, placed before clusters)
    cues_elem = _make_cues_element(cues) if cues else b""

    # Segment (unknown size)
    segment_payload = seg_info + tracks_elem + cues_elem + clusters_bytes
    segment = (
        encode_element_id(mkv.SEGMENT)
        + encode_element_size(UNKNOWN_SIZE, width=8)
        + segment_payload
    )

    return ebml_header + segment


# ---------------------------------------------------------------------------
# Unit tests: _merge_regions
# ---------------------------------------------------------------------------

class TestMergeRegions:
    """Test the static _merge_regions method."""

    def test_empty_plan(self):
        assert MkvParser._merge_regions([]) == []

    def test_single_block(self):
        bp = BlockPlan(
            file_offset=1000, size=500, track_number=2,
            cluster_ts=0, rel_ts=0, flags=0, hdr_size=4,
            is_simple_block=True,
        )
        regions = MkvParser._merge_regions([bp])
        assert len(regions) == 1
        assert regions[0].offset == 1000
        assert regions[0].length == 500
        assert regions[0].blocks == [bp]

    def test_adjacent_blocks_merged(self):
        """Two blocks within gap_threshold should be merged."""
        bp1 = BlockPlan(
            file_offset=1000, size=100, track_number=2,
            cluster_ts=0, rel_ts=0, flags=0, hdr_size=4,
            is_simple_block=True,
        )
        bp2 = BlockPlan(
            file_offset=1200, size=100, track_number=2,
            cluster_ts=0, rel_ts=0, flags=0, hdr_size=4,
            is_simple_block=True,
        )
        regions = MkvParser._merge_regions([bp1, bp2], gap_threshold=200)
        assert len(regions) == 1
        assert regions[0].offset == 1000
        assert regions[0].length == 300  # 1200 + 100 - 1000
        assert len(regions[0].blocks) == 2

    def test_distant_blocks_separate(self):
        """Two blocks with a gap exceeding threshold should be separate regions."""
        bp1 = BlockPlan(
            file_offset=1000, size=100, track_number=2,
            cluster_ts=0, rel_ts=0, flags=0, hdr_size=4,
            is_simple_block=True,
        )
        bp2 = BlockPlan(
            file_offset=200_000, size=100, track_number=2,
            cluster_ts=0, rel_ts=0, flags=0, hdr_size=4,
            is_simple_block=True,
        )
        regions = MkvParser._merge_regions([bp1, bp2], gap_threshold=1000)
        assert len(regions) == 2
        assert regions[0].offset == 1000
        assert regions[0].length == 100
        assert regions[1].offset == 200_000
        assert regions[1].length == 100

    def test_three_blocks_partial_merge(self):
        """Blocks 1 and 2 merge, block 3 is separate."""
        bp1 = BlockPlan(
            file_offset=100, size=50, track_number=2,
            cluster_ts=0, rel_ts=0, flags=0, hdr_size=4,
            is_simple_block=True,
        )
        bp2 = BlockPlan(
            file_offset=200, size=50, track_number=2,
            cluster_ts=0, rel_ts=0, flags=0, hdr_size=4,
            is_simple_block=True,
        )
        bp3 = BlockPlan(
            file_offset=100_000, size=50, track_number=2,
            cluster_ts=0, rel_ts=0, flags=0, hdr_size=4,
            is_simple_block=True,
        )
        regions = MkvParser._merge_regions([bp1, bp2, bp3], gap_threshold=200)
        assert len(regions) == 2
        assert len(regions[0].blocks) == 2
        assert len(regions[1].blocks) == 1

    def test_zero_gap_threshold(self):
        """With gap_threshold=0, only truly overlapping/contiguous blocks merge."""
        bp1 = BlockPlan(
            file_offset=100, size=50, track_number=2,
            cluster_ts=0, rel_ts=0, flags=0, hdr_size=4,
            is_simple_block=True,
        )
        bp2 = BlockPlan(
            file_offset=150, size=50, track_number=2,
            cluster_ts=0, rel_ts=0, flags=0, hdr_size=4,
            is_simple_block=True,
        )
        bp3 = BlockPlan(
            file_offset=201, size=50, track_number=2,
            cluster_ts=0, rel_ts=0, flags=0, hdr_size=4,
            is_simple_block=True,
        )
        regions = MkvParser._merge_regions([bp1, bp2, bp3], gap_threshold=0)
        assert len(regions) == 2
        # bp1 and bp2 are contiguous (gap = 150 - (100+50) = 0)
        assert regions[0].length == 100
        assert len(regions[0].blocks) == 2


class TestSplitRegion:
    """Test the _split_region helper."""

    def test_small_region_not_split(self):
        bp = BlockPlan(
            file_offset=0, size=1000, track_number=2,
            cluster_ts=0, rel_ts=0, flags=0, hdr_size=4,
            is_simple_block=True,
        )
        regions = _split_region(0, 1000, [bp])
        assert len(regions) == 1

    def test_large_region_split(self):
        """A region exceeding _MAX_REGION_READ should be split."""
        blocks = []
        for i in range(32):
            blocks.append(BlockPlan(
                file_offset=i * 1_000_000,
                size=500_000,
                track_number=2,
                cluster_ts=0, rel_ts=0, flags=0, hdr_size=4,
                is_simple_block=True,
            ))
        total_length = blocks[-1].file_offset + blocks[-1].size - blocks[0].file_offset
        regions = _split_region(blocks[0].file_offset, total_length, blocks)

        # Should be split into multiple regions
        assert len(regions) > 1
        # Every region should be <= _MAX_REGION_READ
        for r in regions:
            assert r.length <= _MAX_REGION_READ
        # All blocks should be accounted for
        all_blocks = []
        for r in regions:
            all_blocks.extend(r.blocks)
        assert len(all_blocks) == 32


# ---------------------------------------------------------------------------
# Integration test: batched vs single-pass byte-identical output
# ---------------------------------------------------------------------------

class TestBatchedVsSinglePass:
    """End-to-end tests: both strategies must produce identical output."""

    def _build_test_mkv(self) -> bytes:
        """Build a small MKV with video (track 1), audio (track 2), subtitle (track 5)."""
        clusters = b""
        for cluster_ts in range(0, 3000, 1000):
            blocks = []
            # Video blocks (track 1) — large payload, will be skipped
            for i in range(5):
                video_payload = bytes(range(256)) * 4  # 1024 bytes
                sb = _make_simple_block(1, i * 40, 0x80, video_payload)
                blocks.append((mkv.SIMPLE_BLOCK, sb))
            # Audio blocks (track 2) — small payload
            for i in range(3):
                audio_payload = bytes([0xAA, 0xBB, 0xCC] * 20)  # 60 bytes
                sb = _make_simple_block(2, i * 30, 0x80, audio_payload)
                blocks.append((mkv.SIMPLE_BLOCK, sb))
            # Subtitle block (track 5) as BlockGroup
            sub_payload = bytes([0x16, 0x00, 0x04, 0xDE, 0xAD, 0xBE, 0xEF])  # PGS-like
            bg = _make_block_group(5, 0, 0x00, sub_payload, duration=1000)
            blocks.append((mkv.BLOCK_GROUP, bg))

            clusters += _make_cluster(cluster_ts, blocks)

        tracks = [
            (1, mkv.TRACK_TYPE_VIDEO, "V_MPEGH/ISO/HEVC"),
            (2, mkv.TRACK_TYPE_AUDIO, "A_TRUEHD"),
            (5, mkv.TRACK_TYPE_SUBTITLE, "S_HDMV/PGS"),
        ]
        return _make_minimal_mkv(tracks, clusters)

    def test_mkv_audio_extraction_identical(self):
        """Extracting audio via batched and single-pass should be byte-identical."""
        mkv_data = self._build_test_mkv()

        with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as f:
            f.write(mkv_data)
            src_path = f.name

        try:
            parser = MkvParser(src_path)
            assert len(parser.tracks) == 3

            # Single-pass extraction
            with tempfile.NamedTemporaryFile(suffix=".mka", delete=False) as f:
                single_path = f.name
            parser.extract(
                track_types=["audio"],
                output=single_path,
                format="mkv",
                strategy="single-pass",
            )

            # Batched extraction
            with tempfile.NamedTemporaryFile(suffix=".mka", delete=False) as f:
                batched_path = f.name
            parser.extract(
                track_types=["audio"],
                output=batched_path,
                format="mkv",
                strategy="batched",
            )

            with open(single_path, "rb") as a, open(batched_path, "rb") as b:
                single_data = a.read()
                batched_data = b.read()

            assert len(single_data) > 0
            assert single_data == batched_data
        finally:
            os.unlink(src_path)
            os.unlink(single_path)
            os.unlink(batched_path)

    def test_mkv_subtitle_extraction_identical(self):
        """Extracting subtitles via batched and single-pass should be byte-identical."""
        mkv_data = self._build_test_mkv()

        with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as f:
            f.write(mkv_data)
            src_path = f.name

        try:
            parser = MkvParser(src_path)

            # Single-pass
            with tempfile.NamedTemporaryFile(suffix=".mks", delete=False) as f:
                single_path = f.name
            parser.extract(
                track_types=["subtitle"],
                output=single_path,
                format="mkv",
                strategy="single-pass",
            )

            # Batched
            with tempfile.NamedTemporaryFile(suffix=".mks", delete=False) as f:
                batched_path = f.name
            parser.extract(
                track_types=["subtitle"],
                output=batched_path,
                format="mkv",
                strategy="batched",
            )

            with open(single_path, "rb") as a, open(batched_path, "rb") as b:
                assert a.read() == b.read()
        finally:
            os.unlink(src_path)
            os.unlink(single_path)
            os.unlink(batched_path)

    def test_sup_extraction_identical(self):
        """Extracting PGS subtitles as SUP via both strategies should be identical."""
        mkv_data = self._build_test_mkv()

        with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as f:
            f.write(mkv_data)
            src_path = f.name

        try:
            parser = MkvParser(src_path)

            # Single-pass
            with tempfile.NamedTemporaryFile(suffix=".sup", delete=False) as f:
                single_path = f.name
            parser.extract(
                track_types=["subtitle"],
                output=single_path,
                format="sup",
                strategy="single-pass",
            )

            # Batched
            with tempfile.NamedTemporaryFile(suffix=".sup", delete=False) as f:
                batched_path = f.name
            parser.extract(
                track_types=["subtitle"],
                output=batched_path,
                format="sup",
                strategy="batched",
            )

            with open(single_path, "rb") as a, open(batched_path, "rb") as b:
                single_data = a.read()
                batched_data = b.read()

            assert len(single_data) > 0
            assert single_data == batched_data
        finally:
            os.unlink(src_path)
            os.unlink(single_path)
            os.unlink(batched_path)

    def test_multi_track_extraction_identical(self):
        """Extracting audio+subtitle via both strategies should be byte-identical."""
        mkv_data = self._build_test_mkv()

        with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as f:
            f.write(mkv_data)
            src_path = f.name

        try:
            parser = MkvParser(src_path)

            # Single-pass (default: all non-video)
            with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as f:
                single_path = f.name
            parser.extract(
                output=single_path,
                format="mkv",
                strategy="single-pass",
            )

            # Batched
            with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as f:
                batched_path = f.name
            parser.extract(
                output=batched_path,
                format="mkv",
                strategy="batched",
            )

            with open(single_path, "rb") as a, open(batched_path, "rb") as b:
                assert a.read() == b.read()
        finally:
            os.unlink(src_path)
            os.unlink(single_path)
            os.unlink(batched_path)

    def test_auto_strategy_selects_single_pass_for_small_files(self):
        """auto strategy should use single-pass for files < 1 GB (our test files)."""
        mkv_data = self._build_test_mkv()

        with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as f:
            f.write(mkv_data)
            src_path = f.name

        try:
            parser = MkvParser(src_path)
            assert parser._file_size < 1_000_000_000

            # auto and single-pass should produce identical output
            with tempfile.NamedTemporaryFile(suffix=".mka", delete=False) as f:
                auto_path = f.name
            with tempfile.NamedTemporaryFile(suffix=".mka", delete=False) as f:
                sp_path = f.name

            parser.extract(
                track_types=["audio"], output=auto_path, format="mkv",
                strategy="auto",
            )
            parser.extract(
                track_types=["audio"], output=sp_path, format="mkv",
                strategy="single-pass",
            )

            with open(auto_path, "rb") as a, open(sp_path, "rb") as b:
                assert a.read() == b.read()
        finally:
            os.unlink(src_path)
            os.unlink(auto_path)
            os.unlink(sp_path)


class TestScanBlocks:
    """Test Phase 1 scan produces correct BlockPlan entries."""

    def _build_and_parse(self):
        """Build test MKV and return (parser, src_path)."""
        clusters = b""
        # Cluster at ts=0: 2 video (track 1), 1 audio (track 2)
        blocks = [
            (mkv.SIMPLE_BLOCK, _make_simple_block(1, 0, 0x80, b"\x00" * 100)),
            (mkv.SIMPLE_BLOCK, _make_simple_block(2, 0, 0x80, b"\xAA" * 20)),
            (mkv.SIMPLE_BLOCK, _make_simple_block(1, 40, 0x00, b"\x00" * 100)),
        ]
        clusters += _make_cluster(0, blocks)

        # Cluster at ts=1000: 1 video, 1 audio
        blocks = [
            (mkv.SIMPLE_BLOCK, _make_simple_block(1, 0, 0x80, b"\x00" * 100)),
            (mkv.SIMPLE_BLOCK, _make_simple_block(2, 0, 0x80, b"\xBB" * 20)),
        ]
        clusters += _make_cluster(1000, blocks)

        tracks = [
            (1, mkv.TRACK_TYPE_VIDEO, "V_MPEGH/ISO/HEVC"),
            (2, mkv.TRACK_TYPE_AUDIO, "A_TRUEHD"),
        ]
        mkv_data = _make_minimal_mkv(tracks, clusters)

        with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as f:
            f.write(mkv_data)
            path = f.name
        return MkvParser(path), path

    def test_scan_filters_to_wanted_tracks(self):
        parser, path = self._build_and_parse()
        try:
            with open(path, "rb") as src:
                plan = parser._scan_blocks(src, {2})  # only audio

            # Should have exactly 2 audio blocks
            assert len(plan) == 2
            assert all(bp.track_number == 2 for bp in plan)
            assert plan[0].cluster_ts == 0
            assert plan[1].cluster_ts == 1000
        finally:
            os.unlink(path)

    def test_scan_preserves_file_order(self):
        parser, path = self._build_and_parse()
        try:
            with open(path, "rb") as src:
                plan = parser._scan_blocks(src, {1, 2})  # all tracks

            # Should be in file offset order
            offsets = [bp.file_offset for bp in plan]
            assert offsets == sorted(offsets)
            # 5 total blocks: 3 in cluster 0, 2 in cluster 1
            assert len(plan) == 5
        finally:
            os.unlink(path)

    def test_scan_records_correct_metadata(self):
        parser, path = self._build_and_parse()
        try:
            with open(path, "rb") as src:
                plan = parser._scan_blocks(src, {2})

            bp = plan[0]
            assert bp.track_number == 2
            assert bp.cluster_ts == 0
            assert bp.rel_ts == 0
            assert bp.is_simple_block is True
            assert bp.hdr_size == 4  # track VINT(1) + ts(2) + flags(1)
            assert bp.size > bp.hdr_size  # has payload beyond header
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Tests: SeekHead → Cues-at-EOF discovery
# ---------------------------------------------------------------------------

class TestSeekHeadCues:
    """Verify Cues at end-of-file are found via SeekHead pointer."""

    def _build_mkv_with_cues_at_eof(self):
        """Build an MKV with SeekHead → Cues at end (typical mkvmerge layout).

        Layout: EBML Header | Segment [ SeekHead | SegmentInfo | Tracks | Clusters | Cues ]
        The SeekHead contains a pointer to the Cues position.
        """
        tracks_spec = [
            (1, mkv.TRACK_TYPE_VIDEO, "V_MPEGH/ISO/HEVC"),
            (2, mkv.TRACK_TYPE_AUDIO, "A_TRUEHD"),
        ]

        # Build clusters.
        cluster_data_list = []
        for cluster_ts in range(0, 3000, 500):
            blocks = [
                (mkv.SIMPLE_BLOCK, _make_simple_block(1, 0, 0x80, b"\x00" * 100)),
                (mkv.SIMPLE_BLOCK, _make_simple_block(2, 0, 0x80, b"\xAA" * 20)),
            ]
            cluster_data_list.append((cluster_ts, _make_cluster(cluster_ts, blocks)))
        clusters_bytes = b"".join(data for _, data in cluster_data_list)

        # Build segment children WITHOUT SeekHead/Cues first to measure sizes.
        seg_info_body = _make_uint_element(mkv.TIMECODE_SCALE, 1_000_000)
        seg_info = (
            encode_element_id(mkv.SEGMENT_INFO)
            + encode_element_size(len(seg_info_body))
            + seg_info_body
        )
        tracks_body = b""
        for num, type_id, codec_id in tracks_spec:
            entry_body = _make_track_entry_body(num, type_id, codec_id, uid=num * 1000)
            tracks_body += (
                encode_element_id(mkv.TRACK_ENTRY)
                + encode_element_size(len(entry_body))
                + entry_body
            )
        tracks_elem = (
            encode_element_id(mkv.TRACKS)
            + encode_element_size(len(tracks_body))
            + tracks_body
        )

        # EBML Header (fixed).
        ebml_body = b""
        ebml_body += _make_uint_element(0x4286, 1)
        ebml_body += _make_uint_element(0x42F7, 1)
        ebml_body += _make_uint_element(0x42F2, 4)
        ebml_body += _make_uint_element(0x42F3, 8)
        ebml_body += _make_string_element(0x4282, "matroska")
        ebml_body += _make_uint_element(0x4287, 4)
        ebml_body += _make_uint_element(0x4285, 2)
        ebml_header = (
            encode_element_id(mkv.EBML_HEADER)
            + encode_element_size(len(ebml_body))
            + ebml_body
        )

        # Segment header: ID (4) + size VINT (8 for unknown-size).
        segment_hdr = (
            encode_element_id(mkv.SEGMENT)
            + encode_element_size(UNKNOWN_SIZE, width=8)
        )

        # segment_data_offset = len(ebml_header) + len(segment_hdr)
        segment_data_offset = len(ebml_header) + len(segment_hdr)

        # Build SeekHead with placeholder Cues position (iterate to stabilize).
        # Cues position = len(seekhead) + len(seg_info) + len(tracks_elem) + len(clusters)
        # But seekhead size depends on the Cues position value encoding...
        # Use 2-pass: estimate seekhead size, then compute exact.
        approx_seekhead = _make_seekhead([(mkv.CUES, 99999)])
        seekhead_size = len(approx_seekhead)

        cues_pos_rel = seekhead_size + len(seg_info) + len(tracks_elem) + len(clusters_bytes)

        # Rebuild seekhead with correct position.
        seekhead = _make_seekhead([(mkv.CUES, cues_pos_rel)])
        # If size changed, iterate once more.
        if len(seekhead) != seekhead_size:
            seekhead_size = len(seekhead)
            cues_pos_rel = seekhead_size + len(seg_info) + len(tracks_elem) + len(clusters_bytes)
            seekhead = _make_seekhead([(mkv.CUES, cues_pos_rel)])

        # Build Cues with correct cluster positions.
        cue_tuples = []
        cluster_offset_in_segment = seekhead_size + len(seg_info) + len(tracks_elem)
        pos = 0
        for cluster_ts, data in cluster_data_list:
            cue_tuples.append((cluster_ts, 1, cluster_offset_in_segment + pos))
            pos += len(data)
        cues_elem = _make_cues_element(cue_tuples)

        # Assemble: SeekHead | SegmentInfo | Tracks | Clusters | Cues
        segment_payload = seekhead + seg_info + tracks_elem + clusters_bytes + cues_elem
        mkv_data = ebml_header + segment_hdr + segment_payload
        return mkv_data, len(cue_tuples)

    def test_cues_found_via_seekhead(self):
        """Parser should find Cues at EOF by following SeekHead pointer."""
        mkv_data, expected_cue_count = self._build_mkv_with_cues_at_eof()

        with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as f:
            f.write(mkv_data)
            src_path = f.name

        try:
            parser = MkvParser(src_path)
            assert len(parser._cues) == expected_cue_count, (
                f"Expected {expected_cue_count} cues, found {len(parser._cues)}"
            )
            # Verify cue timestamps are sensible.
            times = [c.time for c in parser._cues]
            assert times == sorted(times)
            assert times[0] == 0
        finally:
            os.unlink(src_path)

    def test_parallel_scan_with_eof_cues(self):
        """Parallel scan should work when Cues are discovered via SeekHead."""
        mkv_data, _ = self._build_mkv_with_cues_at_eof()

        with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as f:
            f.write(mkv_data)
            src_path = f.name

        try:
            parser = MkvParser(src_path)
            assert len(parser._cues) > 0

            # Single-threaded baseline.
            with open(src_path, "rb") as src:
                plan_single = parser._scan_blocks(src, {2})

            # Parallel scan.
            plan_parallel = parser._scan_blocks_parallel({2}, n_workers=2)

            assert len(plan_parallel) == len(plan_single)
            for bp_s, bp_p in zip(plan_single, plan_parallel):
                assert bp_s.file_offset == bp_p.file_offset
                assert bp_s.size == bp_p.size
                assert bp_s.track_number == bp_p.track_number
                assert bp_s.cluster_ts == bp_p.cluster_ts
        finally:
            os.unlink(src_path)

    def test_extraction_identical_with_eof_cues(self):
        """Batched extraction with EOF Cues should match single-pass."""
        mkv_data, _ = self._build_mkv_with_cues_at_eof()

        with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as f:
            f.write(mkv_data)
            src_path = f.name

        try:
            parser = MkvParser(src_path)

            with tempfile.NamedTemporaryFile(suffix=".mka", delete=False) as f:
                single_path = f.name
            parser.extract(
                track_types=["audio"], output=single_path,
                format="mkv", strategy="single-pass",
            )

            with tempfile.NamedTemporaryFile(suffix=".mka", delete=False) as f:
                parallel_path = f.name
            parser.extract(
                track_types=["audio"], output=parallel_path,
                format="mkv", strategy="batched", scan_workers=2,
            )

            with open(single_path, "rb") as a, open(parallel_path, "rb") as b:
                single_data = a.read()
                parallel_data = b.read()

            assert len(single_data) > 0
            assert single_data == parallel_data
        finally:
            os.unlink(src_path)
            os.unlink(single_path)
            os.unlink(parallel_path)


# ---------------------------------------------------------------------------
# Tests: _build_scan_ranges
# ---------------------------------------------------------------------------

class TestBuildScanRanges:
    """Test Cue-based range partitioning for parallel scanning."""

    def test_many_cues_produces_n_ranges(self):
        """With many evenly-spaced cues, should produce n_workers ranges."""
        # Simulate cues at every 1000 bytes, segment starts at offset 100.
        cues = [
            mkv.CueEntry(time=i, track=1, cluster_position=i * 1000)
            for i in range(20)
        ]
        ranges = MkvParser._build_scan_ranges(
            cues,
            segment_data_offset=100,
            first_cluster_offset=100,
            file_size=20100,
            n_workers=4,
        )
        assert len(ranges) == 4
        # Ranges should cover the full span.
        assert ranges[0][0] == 100
        assert ranges[-1][1] == 20100
        # Ranges should be non-overlapping and contiguous.
        for i in range(len(ranges) - 1):
            assert ranges[i][1] == ranges[i + 1][0]

    def test_fewer_cues_than_workers(self):
        """With fewer cues than workers, should produce fewer ranges."""
        cues = [
            mkv.CueEntry(time=0, track=1, cluster_position=0),
            mkv.CueEntry(time=1000, track=1, cluster_position=5000),
        ]
        ranges = MkvParser._build_scan_ranges(
            cues,
            segment_data_offset=100,
            first_cluster_offset=100,
            file_size=10000,
            n_workers=8,
        )
        assert len(ranges) == 2
        assert ranges[0][0] == 100
        assert ranges[-1][1] == 10000

    def test_single_cue_returns_single_range(self):
        """A single cue point can't be split — return one range."""
        cues = [mkv.CueEntry(time=0, track=1, cluster_position=0)]
        ranges = MkvParser._build_scan_ranges(
            cues,
            segment_data_offset=100,
            first_cluster_offset=100,
            file_size=50000,
            n_workers=4,
        )
        assert len(ranges) == 1
        assert ranges[0] == (100, 50000)

    def test_empty_cues_returns_single_range(self):
        """No cues — return single range covering whole cluster region."""
        ranges = MkvParser._build_scan_ranges(
            cues=[],
            segment_data_offset=100,
            first_cluster_offset=100,
            file_size=50000,
            n_workers=4,
        )
        assert len(ranges) == 1
        assert ranges[0] == (100, 50000)

    def test_duplicate_cluster_positions_deduplicated(self):
        """Cues with duplicate cluster_position should be deduplicated."""
        cues = [
            mkv.CueEntry(time=0, track=1, cluster_position=0),
            mkv.CueEntry(time=0, track=2, cluster_position=0),  # same pos
            mkv.CueEntry(time=1000, track=1, cluster_position=5000),
            mkv.CueEntry(time=1000, track=2, cluster_position=5000),  # same
            mkv.CueEntry(time=2000, track=1, cluster_position=10000),
        ]
        ranges = MkvParser._build_scan_ranges(
            cues,
            segment_data_offset=100,
            first_cluster_offset=100,
            file_size=20000,
            n_workers=3,
        )
        # 3 unique offsets, 3 workers — should get 3 ranges.
        assert len(ranges) == 3
        assert ranges[0][0] == 100
        assert ranges[-1][1] == 20000

    def test_n_workers_1_returns_single_range(self):
        """n_workers=1 should always return a single range."""
        cues = [
            mkv.CueEntry(time=i, track=1, cluster_position=i * 1000)
            for i in range(20)
        ]
        ranges = MkvParser._build_scan_ranges(
            cues,
            segment_data_offset=100,
            first_cluster_offset=100,
            file_size=20100,
            n_workers=1,
        )
        assert len(ranges) == 1
        assert ranges[0] == (100, 20100)


# ---------------------------------------------------------------------------
# Tests: parallel scan correctness
# ---------------------------------------------------------------------------

class TestParallelScan:
    """Verify _scan_blocks_parallel produces identical results to _scan_blocks."""

    def _build_mkv_and_inject_cues(self):
        """Build a synthetic MKV, scan for real cluster offsets, inject Cues.

        Returns (src_path, parser) with parser._cues populated with correct
        CueEntry objects pointing to actual cluster positions.
        """
        from fast_mkv_parser.ebml import read_element_id, read_element_size

        tracks_spec = [
            (1, mkv.TRACK_TYPE_VIDEO, "V_MPEGH/ISO/HEVC"),
            (2, mkv.TRACK_TYPE_AUDIO, "A_TRUEHD"),
            (5, mkv.TRACK_TYPE_SUBTITLE, "S_HDMV/PGS"),
        ]

        # Build 10 clusters with mixed track types.
        clusters = b""
        for cluster_ts in range(0, 5000, 500):
            blocks = []
            # Video (track 1) — large payloads
            for i in range(3):
                payload = bytes(range(256)) * 2  # 512 bytes
                sb = _make_simple_block(1, i * 40, 0x80, payload)
                blocks.append((mkv.SIMPLE_BLOCK, sb))
            # Audio (track 2)
            for i in range(2):
                payload = bytes([0xAA, 0xBB] * 10)
                sb = _make_simple_block(2, i * 30, 0x80, payload)
                blocks.append((mkv.SIMPLE_BLOCK, sb))
            # Subtitle (track 5) as BlockGroup
            sub_payload = bytes([0xDE, 0xAD, 0xBE, 0xEF])
            bg = _make_block_group(5, 0, 0x00, sub_payload, duration=500)
            blocks.append((mkv.BLOCK_GROUP, bg))
            clusters += _make_cluster(cluster_ts, blocks)

        mkv_data = _make_minimal_mkv(tracks_spec, clusters)

        with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as f:
            f.write(mkv_data)
            src_path = f.name

        parser = MkvParser(src_path)

        # Scan the file to find actual cluster byte offsets.
        cues = []
        with open(src_path, "rb") as f:
            f.seek(parser._layout.first_cluster_offset)
            while f.tell() < parser._file_size:
                cluster_pos = f.tell()
                try:
                    eid, _ = read_element_id(f)
                    esize, _ = read_element_size(f)
                except EOFError:
                    break
                if eid == mkv.CLUSTER:
                    body_start = f.tell()
                    # Read ClusterTimestamp (first child).
                    cid, _ = read_element_id(f)
                    csize, _ = read_element_size(f)
                    ts = mkv._read_uint(f, csize) if cid == mkv.CLUSTER_TIMESTAMP else 0
                    rel_pos = cluster_pos - parser._layout.segment_data_offset
                    cues.append(mkv.CueEntry(time=ts, track=1, cluster_position=rel_pos))
                    # Skip to end of cluster.
                    if esize != UNKNOWN_SIZE:
                        f.seek(body_start + esize)
                    else:
                        break
                else:
                    if esize == UNKNOWN_SIZE:
                        break
                    f.seek(f.tell() + esize)

        parser._cues = cues
        return src_path, parser

    def test_parallel_matches_single_threaded(self):
        """Parallel scan with 2 workers should produce identical BlockPlan."""
        src_path, parser = self._build_mkv_and_inject_cues()

        try:
            assert len(parser._cues) > 0, "Parser should have injected Cues"

            # Single-threaded scan.
            with open(src_path, "rb") as src:
                plan_single = parser._scan_blocks(src, {2, 5})

            # Parallel scan with 2 workers.
            plan_parallel = parser._scan_blocks_parallel({2, 5}, n_workers=2)

            # Must produce identical results.
            assert len(plan_parallel) == len(plan_single)
            for bp_s, bp_p in zip(plan_single, plan_parallel):
                assert bp_s.file_offset == bp_p.file_offset
                assert bp_s.size == bp_p.size
                assert bp_s.track_number == bp_p.track_number
                assert bp_s.cluster_ts == bp_p.cluster_ts
                assert bp_s.rel_ts == bp_p.rel_ts
                assert bp_s.flags == bp_p.flags
                assert bp_s.hdr_size == bp_p.hdr_size
                assert bp_s.is_simple_block == bp_p.is_simple_block
        finally:
            os.unlink(src_path)

    def test_parallel_scan_all_tracks(self):
        """Parallel scan for all tracks should match single-threaded."""
        src_path, parser = self._build_mkv_and_inject_cues()

        try:
            with open(src_path, "rb") as src:
                plan_single = parser._scan_blocks(src, {1, 2, 5})

            plan_parallel = parser._scan_blocks_parallel({1, 2, 5}, n_workers=4)

            assert len(plan_parallel) == len(plan_single)
            for bp_s, bp_p in zip(plan_single, plan_parallel):
                assert bp_s.file_offset == bp_p.file_offset
                assert bp_s.size == bp_p.size
                assert bp_s.track_number == bp_p.track_number
                assert bp_s.cluster_ts == bp_p.cluster_ts
        finally:
            os.unlink(src_path)

    def test_parallel_fallback_no_cues(self):
        """Without Cues, parallel scan should fall back to single-threaded."""
        # Build MKV without Cues.
        clusters = b""
        for cluster_ts in range(0, 2000, 1000):
            blocks = [
                (mkv.SIMPLE_BLOCK, _make_simple_block(2, 0, 0x80, b"\xAA" * 20)),
            ]
            clusters += _make_cluster(cluster_ts, blocks)

        tracks = [(2, mkv.TRACK_TYPE_AUDIO, "A_TRUEHD")]
        mkv_data = _make_minimal_mkv(tracks, clusters)

        with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as f:
            f.write(mkv_data)
            src_path = f.name

        try:
            parser = MkvParser(src_path)
            assert len(parser._cues) == 0

            with open(src_path, "rb") as src:
                plan_single = parser._scan_blocks(src, {2})

            plan_parallel = parser._scan_blocks_parallel({2}, n_workers=4)

            assert len(plan_parallel) == len(plan_single)
            for bp_s, bp_p in zip(plan_single, plan_parallel):
                assert bp_s.file_offset == bp_p.file_offset
                assert bp_s.size == bp_p.size
        finally:
            os.unlink(src_path)

    def test_batched_extraction_with_workers(self):
        """Batched extraction with scan_workers=2 should produce byte-identical output."""
        src_path, parser = self._build_mkv_and_inject_cues()

        try:
            # Single-pass extraction (baseline).
            with tempfile.NamedTemporaryFile(suffix=".mka", delete=False) as f:
                single_path = f.name
            parser.extract(
                track_types=["audio"],
                output=single_path,
                format="mkv",
                strategy="single-pass",
            )

            # Batched with parallel scan.
            with tempfile.NamedTemporaryFile(suffix=".mka", delete=False) as f:
                parallel_path = f.name
            parser.extract(
                track_types=["audio"],
                output=parallel_path,
                format="mkv",
                strategy="batched",
                scan_workers=2,
            )

            with open(single_path, "rb") as a, open(parallel_path, "rb") as b:
                single_data = a.read()
                parallel_data = b.read()

            assert len(single_data) > 0
            assert single_data == parallel_data
        finally:
            os.unlink(src_path)
            os.unlink(single_path)
            os.unlink(parallel_path)

    def test_sup_extraction_with_workers(self):
        """SUP extraction with parallel scan should match single-pass."""
        src_path, parser = self._build_mkv_and_inject_cues()

        try:
            with tempfile.NamedTemporaryFile(suffix=".sup", delete=False) as f:
                single_path = f.name
            parser.extract(
                track_types=["subtitle"],
                output=single_path,
                format="sup",
                strategy="single-pass",
            )

            with tempfile.NamedTemporaryFile(suffix=".sup", delete=False) as f:
                parallel_path = f.name
            parser.extract(
                track_types=["subtitle"],
                output=parallel_path,
                format="sup",
                strategy="batched",
                scan_workers=2,
            )

            with open(single_path, "rb") as a, open(parallel_path, "rb") as b:
                assert a.read() == b.read()
        finally:
            os.unlink(src_path)
            os.unlink(single_path)
            os.unlink(parallel_path)
