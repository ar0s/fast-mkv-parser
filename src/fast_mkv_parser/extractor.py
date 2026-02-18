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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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
_FADV_SEQUENTIAL = getattr(os, "POSIX_FADV_SEQUENTIAL", 2)
_HAS_FADVISE = hasattr(os, "posix_fadvise")

# Maximum size of a single merged region read (16 MB).
_MAX_REGION_READ = 16 * 1024 * 1024


@dataclass
class BlockPlan:
    """One block to be extracted, discovered during Phase 1 header scan."""
    file_offset: int      # Absolute offset of block body in source file
    size: int             # Block body size in bytes
    track_number: int     # Track number (from block header)
    cluster_ts: int       # Cluster timestamp for this block
    rel_ts: int           # Block-relative timestamp (signed int16)
    flags: int            # Block flags (keyframe etc.)
    hdr_size: int         # Size of block header within body (track + ts + flags)
    is_simple_block: bool # SimpleBlock vs BlockGroup


@dataclass
class ReadRegion:
    """A merged contiguous region to read from the source file."""
    offset: int
    length: int
    blocks: List[BlockPlan] = field(default_factory=list)


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


def _split_region(
    offset: int, length: int, blocks: List[BlockPlan]
) -> List[ReadRegion]:
    """Split a merged region into sub-regions if it exceeds _MAX_REGION_READ.

    Each sub-region contains the blocks whose file_offset falls within it.
    Sub-region boundaries are chosen so that no block is split across regions.
    """
    if length <= _MAX_REGION_READ:
        return [ReadRegion(offset=offset, length=length, blocks=blocks)]

    regions: List[ReadRegion] = []
    sub_start = offset
    sub_blocks: List[BlockPlan] = []

    for bp in blocks:
        bp_end = bp.file_offset + bp.size
        # Would adding this block push the sub-region over the limit?
        if sub_blocks and (bp_end - sub_start) > _MAX_REGION_READ:
            # Finalize current sub-region up to the end of the last block.
            last_end = sub_blocks[-1].file_offset + sub_blocks[-1].size
            regions.append(ReadRegion(
                offset=sub_start,
                length=last_end - sub_start,
                blocks=sub_blocks,
            ))
            sub_start = bp.file_offset
            sub_blocks = [bp]
        else:
            sub_blocks.append(bp)

    # Finalize last sub-region.
    if sub_blocks:
        last_end = sub_blocks[-1].file_offset + sub_blocks[-1].size
        regions.append(ReadRegion(
            offset=sub_start,
            length=last_end - sub_start,
            blocks=sub_blocks,
        ))

    return regions


def _scan_block_range(
    path: str,
    start_offset: int,
    end_offset: int,
    wanted: Set[int],
) -> List[BlockPlan]:
    """Scan blocks in [start_offset, end_offset) of an MKV file.

    Opens its own file descriptor with FADV_RANDOM.  Returns BlockPlan
    entries for wanted tracks, in file-offset order.

    Designed for ThreadPoolExecutor: module-level, no shared state.
    """
    plan: List[BlockPlan] = []
    fd = os.open(path, os.O_RDONLY)
    if _HAS_FADVISE:
        try:
            length = end_offset - start_offset
            os.posix_fadvise(fd, start_offset, length, _FADV_RANDOM)
        except OSError:
            pass
    src = os.fdopen(fd, "rb", buffering=8192)
    try:
        src.seek(start_offset)

        # Scan for Cluster elements in our range.
        while src.tell() < end_offset:
            try:
                elem_pos = src.tell()
                eid, _ = read_element_id(src)
                esize, _ = read_element_size(src)
            except EOFError:
                break

            if eid != mkv.CLUSTER:
                # Non-Cluster Level 1 element — skip it.
                if esize == UNKNOWN_SIZE:
                    break
                src.seek(src.tell() + esize)
                continue

            # Parse Cluster children.
            cluster_body_start = src.tell()
            cluster_end = (
                cluster_body_start + esize if esize != UNKNOWN_SIZE
                else end_offset
            )
            # Don't scan past our assigned range.
            cluster_end = min(cluster_end, end_offset)
            cluster_ts = 0

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
                    header_peek = src.read(min(12, child_size))
                    track_num, rel_ts, flags, hdr_size = mkv.parse_block_header(
                        header_peek
                    )

                    if track_num in wanted:
                        plan.append(BlockPlan(
                            file_offset=child_body_pos,
                            size=child_size,
                            track_number=track_num,
                            cluster_ts=cluster_ts,
                            rel_ts=rel_ts,
                            flags=flags,
                            hdr_size=hdr_size,
                            is_simple_block=True,
                        ))

                    remaining = child_size - len(header_peek)
                    if remaining > 0:
                        src.seek(remaining, 1)
                    continue

                if child_id == mkv.BLOCK_GROUP:
                    group_end = child_body_pos + child_size
                    track_num = None
                    rel_ts = 0
                    flags = 0
                    hdr_size = 0

                    while src.tell() < group_end:
                        try:
                            bgeid, _ = read_element_id(src)
                            bgesize, _ = read_element_size(src)
                        except EOFError:
                            break

                        if bgeid == mkv.BLOCK:
                            header_peek = src.read(min(12, bgesize))
                            track_num, rel_ts, flags, hdr_size = (
                                mkv.parse_block_header(header_peek)
                            )
                            break
                        else:
                            if bgesize == UNKNOWN_SIZE:
                                break
                            src.seek(bgesize, 1)

                    if track_num is not None and track_num in wanted:
                        plan.append(BlockPlan(
                            file_offset=child_body_pos,
                            size=child_size,
                            track_number=track_num,
                            cluster_ts=cluster_ts,
                            rel_ts=rel_ts,
                            flags=flags,
                            hdr_size=hdr_size,
                            is_simple_block=False,
                        ))

                    src.seek(group_end)
                    continue

                if child_id == mkv.CLUSTER:
                    # Encountered next Cluster inside unknown-size Cluster.
                    src.seek(child_pos)
                    break

                # Unknown child — skip.
                if child_size == UNKNOWN_SIZE:
                    break
                src.seek(child_body_pos + child_size)

    finally:
        src.close()

    return plan


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
        strategy: str = "auto",
        scan_workers: int = 0,
    ) -> None:
        """Extract selected tracks to *output* file.

        Args:
            track_numbers: Explicit track numbers to extract.
            track_types: Track types to extract (e.g. ["audio", "subtitle"]).
                         Ignored if track_numbers is provided.
            output: Output file path.
            format: Output format: "mkv" (Matroska container) or "sup" (PGS SUP).
            progress_callback: Called with (bytes_processed, total_file_size).
            strategy: Extraction strategy:
                "auto" — batched for files >1 GB, single-pass otherwise.
                "batched" — two-phase scan+fetch (optimized for NFS).
                "single-pass" — original interleaved read/skip.
            scan_workers: Number of parallel Phase 1 scan workers.
                0 = auto (4 for batched strategy, 1 otherwise).
                1 = single-threaded.
        """
        wanted = self._resolve_wanted_tracks(track_numbers, track_types)
        if not wanted:
            raise ValueError("no matching tracks found for the given filter")

        if strategy == "auto":
            use_batched = self._file_size > 1_000_000_000  # >1 GB
        else:
            use_batched = strategy == "batched"

        # Resolve worker count.
        workers = scan_workers
        if workers == 0:
            workers = 4 if use_batched else 1

        with open(output, "wb") as dst:
            if format == "sup":
                writer = SupWriter(dst, timecode_scale_ns=self.timecode_scale)
            else:
                writer = MkvWriter(dst)
                self._write_mkv_header(writer, wanted)

            if use_batched:
                # Phase 1: scan (parallel or single-threaded).
                plan = self._scan_blocks_parallel(wanted, n_workers=workers)
                regions = self._merge_regions(plan)
                # Phase 2: fetch regions with a single fd.
                with _open_for_sparse_read(self.path) as src:
                    self._fetch_regions(
                        src, regions, writer, format, progress_callback
                    )
            else:
                with _open_for_sparse_read(self.path) as src:
                    self._walk_clusters(
                        src, writer, wanted, format, progress_callback
                    )

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
    # Batched I/O: two-phase extraction for NFS performance
    # ------------------------------------------------------------------

    def _scan_blocks(self, src: BinaryIO, wanted: Set[int]) -> List[BlockPlan]:
        """Phase 1: Scan all Clusters, recording metadata for wanted blocks.

        Uses FADV_SEQUENTIAL for efficient NFS read-ahead during the scan.
        Does NOT read block payloads — only element headers + track numbers.
        """
        plan: List[BlockPlan] = []
        src.seek(self._layout.first_cluster_offset)

        # Use FADV_RANDOM to avoid wasted NFS read-ahead during header scan.
        # Each header peek is ~12 bytes; FADV_SEQUENTIAL triggers 128KB+
        # read-ahead that is discarded on the next seek (833:1 waste ratio).
        if _HAS_FADVISE:
            try:
                fd = src.fileno()
                os.posix_fadvise(fd, self._layout.first_cluster_offset, 0,
                                 _FADV_RANDOM)
            except OSError:
                pass

        while src.tell() < self._file_size:
            try:
                elem_pos = src.tell()
                eid, _ = read_element_id(src)
                esize, _ = read_element_size(src)
            except EOFError:
                break

            if eid != mkv.CLUSTER:
                if esize == UNKNOWN_SIZE:
                    break
                src.seek(src.tell() + esize)
                continue

            cluster_body_start = src.tell()
            cluster_end = (
                cluster_body_start + esize if esize != UNKNOWN_SIZE
                else self._file_size
            )
            cluster_ts = 0

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
                    # Read just the header peek (track + ts + flags).
                    header_peek = src.read(min(12, child_size))
                    track_num, rel_ts, flags, hdr_size = mkv.parse_block_header(
                        header_peek
                    )

                    if track_num in wanted:
                        plan.append(BlockPlan(
                            file_offset=child_body_pos,
                            size=child_size,
                            track_number=track_num,
                            cluster_ts=cluster_ts,
                            rel_ts=rel_ts,
                            flags=flags,
                            hdr_size=hdr_size,
                            is_simple_block=True,
                        ))

                    # Skip remaining payload.
                    remaining = child_size - len(header_peek)
                    if remaining > 0:
                        src.seek(remaining, 1)
                    continue

                if child_id == mkv.BLOCK_GROUP:
                    group_end = child_body_pos + child_size
                    track_num = None
                    rel_ts = 0
                    flags = 0
                    hdr_size = 0

                    # Scan children to find the Block element and its track number.
                    while src.tell() < group_end:
                        try:
                            bgeid, _ = read_element_id(src)
                            bgesize, _ = read_element_size(src)
                        except EOFError:
                            break

                        if bgeid == mkv.BLOCK:
                            header_peek = src.read(min(12, bgesize))
                            track_num, rel_ts, flags, hdr_size = (
                                mkv.parse_block_header(header_peek)
                            )
                            break
                        else:
                            if bgesize == UNKNOWN_SIZE:
                                break
                            src.seek(bgesize, 1)

                    if track_num is not None and track_num in wanted:
                        plan.append(BlockPlan(
                            file_offset=child_body_pos,
                            size=child_size,
                            track_number=track_num,
                            cluster_ts=cluster_ts,
                            rel_ts=rel_ts,
                            flags=flags,
                            hdr_size=hdr_size,
                            is_simple_block=False,
                        ))

                    src.seek(group_end)
                    continue

                if child_id == mkv.CLUSTER:
                    # Encountered next Cluster inside unknown-size Cluster.
                    src.seek(child_pos)
                    break

                # Unknown child — skip.
                if child_size == UNKNOWN_SIZE:
                    break
                src.seek(child_body_pos + child_size)

        return plan

    @staticmethod
    def _build_scan_ranges(
        cues: List[mkv.CueEntry],
        segment_data_offset: int,
        first_cluster_offset: int,
        file_size: int,
        n_workers: int,
    ) -> List[Tuple[int, int]]:
        """Partition the cluster region into N byte ranges using Cue positions.

        Returns list of (start_offset, end_offset) tuples.  Ranges are
        non-overlapping and cover the entire cluster region from
        first_cluster_offset to file_size.
        """
        # Deduplicate cue cluster positions and convert to absolute offsets.
        abs_offsets = sorted({
            segment_data_offset + c.cluster_position for c in cues
        })
        # Filter to offsets within the cluster region.
        abs_offsets = [o for o in abs_offsets if first_cluster_offset <= o < file_size]

        if len(abs_offsets) < 2 or n_workers <= 1:
            return [(first_cluster_offset, file_size)]

        # Pick N-1 evenly-spaced split points by byte position.
        total_span = abs_offsets[-1] - abs_offsets[0]
        if total_span == 0:
            return [(first_cluster_offset, file_size)]

        ranges: List[Tuple[int, int]] = []
        n = min(n_workers, len(abs_offsets))
        chunk_size = total_span / n

        prev_start = first_cluster_offset
        for i in range(1, n):
            target_byte = abs_offsets[0] + chunk_size * i
            # Find the closest cue offset at or past the target.
            best = abs_offsets[-1]
            for o in abs_offsets:
                if o >= target_byte:
                    best = o
                    break
            if best <= prev_start:
                continue
            ranges.append((prev_start, best))
            prev_start = best

        # Final range to end of file.
        ranges.append((prev_start, file_size))
        return ranges

    def _scan_blocks_parallel(
        self,
        wanted: Set[int],
        n_workers: int = 4,
    ) -> List[BlockPlan]:
        """Phase 1 with parallel scanning using Cue-based range partitioning.

        Falls back to single-threaded _scan_blocks() if no Cues available
        or n_workers <= 1.
        """
        if n_workers <= 1 or not self._cues:
            with _open_for_sparse_read(self.path) as src:
                return self._scan_blocks(src, wanted)

        ranges = self._build_scan_ranges(
            self._cues,
            self._layout.segment_data_offset,
            self._layout.first_cluster_offset,
            self._file_size,
            n_workers,
        )

        if len(ranges) <= 1:
            with _open_for_sparse_read(self.path) as src:
                return self._scan_blocks(src, wanted)

        # Submit parallel scan tasks — each opens its own fd.
        all_plans: List[Tuple[int, List[BlockPlan]]] = []
        with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
            futures = {}
            for idx, (start, end) in enumerate(ranges):
                fut = pool.submit(_scan_block_range, self.path, start, end, wanted)
                futures[fut] = idx

            for fut in as_completed(futures):
                idx = futures[fut]
                all_plans.append((idx, fut.result()))

        # Concatenate in range order (each sub-list is already in file-offset order).
        all_plans.sort(key=lambda x: x[0])
        merged: List[BlockPlan] = []
        for _, sub_plan in all_plans:
            merged.extend(sub_plan)
        return merged

    @staticmethod
    def _merge_regions(
        plan: List[BlockPlan],
        gap_threshold: int = 65536,
    ) -> List[ReadRegion]:
        """Merge adjacent wanted blocks into contiguous read regions.

        If two blocks are within *gap_threshold* bytes of each other, they are
        merged into one region.  This dramatically reduces NFS round trips for
        audio tracks where blocks are close together in the file.

        Regions exceeding ``_MAX_REGION_READ`` are split into sub-regions.
        """
        if not plan:
            return []

        # Plan is already in file-offset order (from sequential scan).
        regions: List[ReadRegion] = []
        cur_start = plan[0].file_offset
        cur_end = plan[0].file_offset + plan[0].size
        cur_blocks: List[BlockPlan] = [plan[0]]

        for bp in plan[1:]:
            bp_end = bp.file_offset + bp.size
            if bp.file_offset - cur_end <= gap_threshold:
                # Merge: extend current region.
                cur_end = max(cur_end, bp_end)
                cur_blocks.append(bp)
            else:
                # Gap too large: finalize current region, start new one.
                regions.extend(
                    _split_region(cur_start, cur_end - cur_start, cur_blocks)
                )
                cur_start = bp.file_offset
                cur_end = bp_end
                cur_blocks = [bp]

        # Finalize last region.
        regions.extend(_split_region(cur_start, cur_end - cur_start, cur_blocks))
        return regions

    def _fetch_regions(
        self,
        src: BinaryIO,
        regions: List[ReadRegion],
        writer,
        format: str,
        progress_callback: Optional[Callable],
    ) -> None:
        """Phase 2: Read merged regions sequentially and write wanted blocks."""
        # Switch to sequential read-ahead for payload fetch.
        if _HAS_FADVISE:
            try:
                fd = src.fileno()
                os.posix_fadvise(fd, 0, 0, _FADV_SEQUENTIAL)
            except OSError:
                pass

        prev_cluster_ts = None

        for region in regions:
            src.seek(region.offset)
            region_data = src.read(region.length)

            for bp in region.blocks:
                # Extract this block's data from the region buffer.
                local_offset = bp.file_offset - region.offset
                block_data = region_data[local_offset : local_offset + bp.size]

                if format == "sup":
                    if bp.is_simple_block:
                        writer.write_block(
                            bp.cluster_ts, bp.rel_ts, block_data[bp.hdr_size:]
                        )
                    else:
                        # BlockGroup: scan the in-memory body to find the Block
                        # element, then extract its payload (matches single-pass
                        # _handle_block_group SUP logic).
                        gf = BytesIO(block_data)
                        while gf.tell() < len(block_data):
                            try:
                                geid, _ = read_element_id(gf)
                                gesize, _ = read_element_size(gf)
                            except EOFError:
                                break
                            if geid == mkv.BLOCK:
                                inner = gf.read(gesize)
                                _, _, _, bhs = mkv.parse_block_header(inner)
                                writer.write_block(
                                    bp.cluster_ts, bp.rel_ts, inner[bhs:]
                                )
                                break
                            else:
                                gf.seek(gesize, 1)
                else:
                    if bp.cluster_ts != prev_cluster_ts:
                        writer.begin_cluster(bp.cluster_ts)
                        prev_cluster_ts = bp.cluster_ts

                    if bp.is_simple_block:
                        writer.write_simple_block(block_data)
                    else:
                        writer.write_block_group(block_data)

            if progress_callback:
                progress_callback(region.offset + region.length, self._file_size)

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
