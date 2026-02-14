"""Sparse MKV track extraction.

Parses MKV container headers, then walks Clusters reading only block
headers.  For unwanted tracks (e.g. video), the block payload is skipped
via file.seek() — no I/O on NFS for those bytes.  Wanted track payloads
are read and written to the output file.

NFS optimization:  Uses posix_fadvise(FADV_RANDOM) to disable kernel
read-ahead.  Without this, each seek past a video block triggers ~1 MB
of wasted NFS read-ahead.  With ~30,000 video blocks, that's ~30 GB of
unnecessary NFS traffic.
"""

from __future__ import annotations

import os
import struct
import sys
from io import BytesIO
from typing import BinaryIO, Callable, Dict, List, Optional, Set, Tuple

from .ebml import (
    UNKNOWN_SIZE,
    read_element_id,
    read_element_size,
    read_vint_value,
    encode_element_id,
    encode_element_size,
)
from . import matroska as mkv
from .writers import MkvWriter, SupWriter

# posix_fadvise constants (may not be available on all platforms).
_FADV_RANDOM = getattr(os, "POSIX_FADV_RANDOM", 1)
_HAS_FADVISE = hasattr(os, "posix_fadvise")


def _open_for_sparse_read(path: str) -> BinaryIO:
    """Open a file optimized for sparse random access (NFS-friendly).

    Disables kernel read-ahead via posix_fadvise(FADV_RANDOM) to prevent
    wasted NFS traffic when seeking past large unwanted block payloads.
    """
    fd = os.open(path, os.O_RDONLY)
    if _HAS_FADVISE:
        try:
            os.posix_fadvise(fd, 0, 0, _FADV_RANDOM)
        except OSError:
            pass  # Not all filesystems support fadvise; proceed anyway.
    return os.fdopen(fd, "rb", buffering=8192)


class MkvParser:
    """Parse an MKV file and extract selected tracks with minimal I/O.

    Usage::

        parser = MkvParser("/path/to/movie.mkv")
        for t in parser.tracks:
            print(t)
        parser.extract(track_types=["audio", "subtitle"], output="stripped.mkv")
    """

    def __init__(self, path: str):
        self.path = path
        self._file_size = os.path.getsize(path)

        self._ebml_header_bytes: bytes = b""
        self._segment_info_bytes: bytes = b""
        self._segment_info: Dict = {}
        self._tracks: List[mkv.TrackInfo] = []
        self._cues: List[mkv.CueEntry] = []
        self._layout = mkv.SegmentLayout()

        with open(path, "rb") as f:
            self._parse_header(f)

    @property
    def tracks(self) -> List[mkv.TrackInfo]:
        return list(self._tracks)

    @property
    def timecode_scale(self) -> int:
        """TimecodeScale in nanoseconds (default 1,000,000 = 1 ms)."""
        return self._segment_info.get("timecode_scale", 1_000_000)

    def extract(
        self,
        track_numbers: Optional[List[int]] = None,
        track_types: Optional[List[str]] = None,
        output: str = "output.mkv",
        format: str = "mkv",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Extract selected tracks to *output* file.

        Args:
            track_numbers: Explicit track numbers to extract.
            track_types: Track types to extract (e.g. ["audio", "subtitle"]).
                         Ignored if track_numbers is provided.
            output: Output file path.
            format: Output format: "mkv" (Matroska container) or "sup" (PGS SUP).
            progress_callback: Called with (bytes_processed, total_file_size).
        """
        wanted = self._resolve_wanted_tracks(track_numbers, track_types)
        if not wanted:
            raise ValueError("no matching tracks found for the given filter")

        with _open_for_sparse_read(self.path) as src, open(output, "wb") as dst:
            if format == "sup":
                writer = SupWriter(dst, timecode_scale_ns=self.timecode_scale)
            else:
                writer = MkvWriter(dst)
                self._write_mkv_header(writer, wanted)

            self._walk_clusters(src, writer, wanted, format, progress_callback)

            writer.finalize()

    def extract_audio(self, output: str = "audio.mka", **kwargs) -> None:
        """Convenience: extract all audio tracks to MKA."""
        self.extract(track_types=["audio"], output=output, format="mkv", **kwargs)

    def extract_subtitles(
        self,
        output: str = "subs.sup",
        format: str = "sup",
        **kwargs,
    ) -> None:
        """Convenience: extract all subtitle tracks."""
        self.extract(track_types=["subtitle"], output=output, format=format, **kwargs)

    # ------------------------------------------------------------------
    # Header parsing
    # ------------------------------------------------------------------

    def _parse_header(self, f: BinaryIO) -> None:
        """Parse EBML Header, then locate and parse Segment children:
        SegmentInfo, Tracks, and optionally Cues.
        """
        # --- EBML Header ---
        start = f.tell()
        eid, _ = read_element_id(f)
        if eid != mkv.EBML_HEADER:
            raise ValueError(f"not an EBML file (first element ID: 0x{eid:X})")
        esize, _ = read_element_size(f)
        body = f.read(esize)
        # Save complete EBML Header (ID + size + body) for passthrough.
        end = f.tell()
        f.seek(start)
        self._ebml_header_bytes = f.read(end - start)

        # --- Segment ---
        seg_id, _ = read_element_id(f)
        if seg_id != mkv.SEGMENT:
            raise ValueError(f"expected Segment, got 0x{seg_id:X}")
        seg_size, _ = read_element_size(f)
        self._layout.segment_data_offset = f.tell()

        # Scan Level-1 children until we find Tracks and SegmentInfo.
        # Stop at the first Cluster (we don't need to read beyond that for init).
        found_info = False
        found_tracks = False

        while f.tell() < self._file_size:
            child_pos = f.tell()
            try:
                child_id, id_width = read_element_id(f)
                child_size, size_width = read_element_size(f)
            except EOFError:
                break

            body_pos = f.tell()

            if child_id == mkv.SEGMENT_INFO:
                self._layout.segment_info_offset = child_pos
                self._layout.segment_info_size = child_size
                self._segment_info_bytes = f.read(child_size)
                self._segment_info = mkv.parse_segment_info(
                    _BytesReader(self._segment_info_bytes),
                    child_size,
                )
                found_info = True

            elif child_id == mkv.TRACKS:
                self._layout.tracks_offset = child_pos
                self._layout.tracks_size = child_size
                self._tracks = mkv.parse_tracks(f, child_size)
                found_tracks = True

            elif child_id == mkv.CUES:
                self._layout.cues_offset = child_pos
                self._layout.cues_size = child_size
                self._cues = mkv.parse_cues(f, child_size)

            elif child_id == mkv.CLUSTER:
                # First Cluster found — record offset and stop scanning.
                self._layout.first_cluster_offset = child_pos
                break

            else:
                # Skip SeekHead, Chapters, Tags, Attachments, etc.
                if child_size == UNKNOWN_SIZE:
                    break
                f.seek(body_pos + child_size)
                continue

            # Ensure we're past this element.
            f.seek(body_pos + child_size)

        if not found_tracks:
            raise ValueError("MKV file has no Tracks element")
        if not found_info:
            self._segment_info = {"timecode_scale": 1_000_000}

    # ------------------------------------------------------------------
    # Cluster walking (sparse extraction core)
    # ------------------------------------------------------------------

    def _walk_clusters(
        self,
        src: BinaryIO,
        writer,
        wanted: Set[int],
        format: str,
        progress_callback: Optional[Callable],
    ) -> None:
        """Walk all Clusters, reading only headers + wanted block payloads."""
        src.seek(self._layout.first_cluster_offset)

        while src.tell() < self._file_size:
            try:
                elem_pos = src.tell()
                eid, _ = read_element_id(src)
                esize, _ = read_element_size(src)
            except EOFError:
                break

            if eid != mkv.CLUSTER:
                # Non-Cluster Level 1 element (e.g. Cues at end of file).
                if esize == UNKNOWN_SIZE:
                    break
                src.seek(src.tell() + esize)
                continue

            # Parse Cluster children.
            cluster_body_start = src.tell()
            cluster_end = (
                cluster_body_start + esize if esize != UNKNOWN_SIZE
                else self._file_size
            )

            self._process_cluster(
                src, cluster_body_start, cluster_end,
                writer, wanted, format,
            )

            if progress_callback:
                progress_callback(src.tell(), self._file_size)

    def _process_cluster(
        self,
        src: BinaryIO,
        body_start: int,
        cluster_end: int,
        writer,
        wanted: Set[int],
        format: str,
    ) -> None:
        """Process a single Cluster's children."""
        cluster_ts = 0
        cluster_has_output = False

        while src.tell() < cluster_end:
            try:
                child_pos = src.tell()
                child_id, _ = read_element_id(src)
                child_size, _ = read_element_size(src)
            except EOFError:
                break

            child_body_pos = src.tell()

            if child_id == mkv.CLUSTER_TIMESTAMP:
                cluster_ts = mkv._read_uint(src, child_size)
                continue

            if child_id == mkv.SIMPLE_BLOCK:
                had_output = self._handle_simple_block(
                    src, child_size, child_body_pos,
                    writer, wanted, format, cluster_ts,
                    cluster_has_output,
                )
                cluster_has_output = cluster_has_output or had_output
                continue

            if child_id == mkv.BLOCK_GROUP:
                had_output = self._handle_block_group(
                    src, child_size, child_body_pos,
                    writer, wanted, format, cluster_ts,
                    cluster_has_output,
                )
                cluster_has_output = cluster_has_output or had_output
                continue

            if child_id == mkv.CLUSTER:
                # Encountered next Cluster inside unknown-size Cluster.
                src.seek(child_pos)
                return

            # Unknown child — skip.
            if child_size == UNKNOWN_SIZE:
                return
            src.seek(child_body_pos + child_size)

    def _handle_simple_block(
        self,
        src: BinaryIO,
        block_size: int,
        body_pos: int,
        writer,
        wanted: Set[int],
        format: str,
        cluster_ts: int,
        cluster_has_output: bool,
    ) -> bool:
        """Handle a SimpleBlock: read header, skip or copy payload.

        Returns True if data was written to the output.
        """
        # Read just enough for the block header.
        header_peek = src.read(min(12, block_size))
        track_num, rel_ts, flags, hdr_size = mkv.parse_block_header(header_peek)

        if track_num not in wanted:
            # Skip the rest of this block — no NFS reads thanks to FADV_RANDOM.
            remaining = block_size - len(header_peek)
            if remaining > 0:
                src.seek(remaining, 1)
            return False

        # Read the remaining payload.
        already_read = len(header_peek)
        remaining = block_size - already_read
        if remaining > 0:
            rest = src.read(remaining)
            full_data = header_peek + rest
        else:
            full_data = header_peek[:block_size]

        if format == "sup":
            writer.write_block(cluster_ts, rel_ts, full_data[hdr_size:])
        else:
            if not cluster_has_output:
                writer.begin_cluster(cluster_ts)
            writer.write_simple_block(full_data)
        return True

    def _handle_block_group(
        self,
        src: BinaryIO,
        group_size: int,
        body_pos: int,
        writer,
        wanted: Set[int],
        format: str,
        cluster_ts: int,
        cluster_has_output: bool,
    ) -> bool:
        """Handle a BlockGroup: peek at the Block header to check track number.

        Returns True if data was written to the output.
        """
        group_end = body_pos + group_size
        scan_start = src.tell()

        # Scan for the Block element to check its track number.
        track_num = None
        block_rel_ts = 0

        while src.tell() < group_end:
            try:
                eid, _ = read_element_id(src)
                esize, _ = read_element_size(src)
            except EOFError:
                break

            if eid == mkv.BLOCK:
                header_peek = src.read(min(12, esize))
                track_num, block_rel_ts, _, hdr_size = mkv.parse_block_header(header_peek)
                break
            else:
                if esize == UNKNOWN_SIZE:
                    break
                src.seek(esize, 1)

        if track_num is None or track_num not in wanted:
            src.seek(group_end)
            return False

        # Track is wanted — read the entire BlockGroup body.
        src.seek(scan_start)
        group_body = src.read(group_end - scan_start)

        if format == "sup":
            gf = BytesIO(group_body)
            while gf.tell() < len(group_body):
                try:
                    geid, _ = read_element_id(gf)
                    gesize, _ = read_element_size(gf)
                except EOFError:
                    break
                if geid == mkv.BLOCK:
                    block_data = gf.read(gesize)
                    _, _, _, bhs = mkv.parse_block_header(block_data)
                    writer.write_block(cluster_ts, block_rel_ts, block_data[bhs:])
                    break
                else:
                    gf.seek(gesize, 1)
        else:
            if not cluster_has_output:
                writer.begin_cluster(cluster_ts)
            writer.write_block_group(group_body)
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_wanted_tracks(
        self,
        track_numbers: Optional[List[int]],
        track_types: Optional[List[str]],
    ) -> Set[int]:
        """Resolve track filters to a set of track numbers."""
        if track_numbers:
            return set(track_numbers)
        if track_types:
            type_set = set(track_types)
            return {t.number for t in self._tracks if t.type in type_set}
        # Default: all non-video tracks.
        return {t.number for t in self._tracks if t.type != "video"}

    def _write_mkv_header(self, writer: MkvWriter, wanted: Set[int]) -> None:
        """Write MKV header with only the wanted track entries."""
        track_entries = [
            t.raw_entry for t in self._tracks if t.number in wanted
        ]
        writer.write_header(
            self._ebml_header_bytes,
            self._segment_info_bytes,
            track_entries,
        )


class _BytesReader:
    """Minimal file-like wrapper around bytes for reusing parse functions."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            result = self._data[self._pos:]
            self._pos = len(self._data)
        else:
            result = self._data[self._pos : self._pos + n]
            self._pos += len(result)
        return result

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = len(self._data) + offset
        return self._pos
