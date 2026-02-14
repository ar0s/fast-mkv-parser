"""Command-line interface for fast-mkv-parser.

Usage:
    fast-mkv-parser info <input>
    fast-mkv-parser extract <input> -o <output> [-t audio,subtitle] [-f mkv|sup] [--progress]
"""

from __future__ import annotations

import argparse
import sys
import time

from .extractor import MkvParser


def _format_size(n: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def cmd_info(args: argparse.Namespace) -> None:
    """Show track listing for an MKV file."""
    parser = MkvParser(args.input)

    tcs = parser.timecode_scale
    duration = parser._segment_info.get("duration")
    muxing = parser._segment_info.get("muxing_app", "")
    writing = parser._segment_info.get("writing_app", "")

    print(f"File: {args.input}")
    if duration:
        dur_s = duration * tcs / 1_000_000_000
        h, rem = divmod(int(dur_s), 3600)
        m, s = divmod(rem, 60)
        print(f"Duration: {h}:{m:02d}:{s:02d} ({dur_s:.1f}s)")
    print(f"TimecodeScale: {tcs} ns")
    if muxing:
        print(f"Muxing app: {muxing}")
    if writing:
        print(f"Writing app: {writing}")
    print(f"Cue points: {len(parser._cues)}")
    print()

    print(f"{'#':<4} {'Type':<10} {'Codec':<25} {'Lang':<6} {'Name'}")
    print("-" * 70)
    for t in parser.tracks:
        print(f"{t.number:<4} {t.type:<10} {t.codec_id:<25} {t.language:<6} {t.name}")


def cmd_extract(args: argparse.Namespace) -> None:
    """Extract selected tracks from an MKV file."""
    parser = MkvParser(args.input)

    # Resolve track filter.
    track_numbers = None
    track_types = None
    if args.track_numbers:
        track_numbers = [int(x.strip()) for x in args.track_numbers.split(",")]
    elif args.tracks:
        track_types = [x.strip() for x in args.tracks.split(",")]

    # Resolve format from flag or output extension.
    fmt = args.format
    if fmt is None:
        if args.output.endswith(".sup"):
            fmt = "sup"
        else:
            fmt = "mkv"

    # Describe what we're extracting.
    wanted = parser._resolve_wanted_tracks(track_numbers, track_types)
    wanted_tracks = [t for t in parser.tracks if t.number in wanted]
    if not wanted_tracks:
        print("Error: no tracks match the given filter.", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting {len(wanted_tracks)} track(s) from {args.input}:")
    for t in wanted_tracks:
        print(f"  Track {t.number}: {t.type} ({t.codec_id})")
    print(f"Output: {args.output} (format: {fmt})")
    print()

    # Progress callback.
    start_time = time.monotonic()
    last_report = [0.0]

    def progress(bytes_done: int, total: int) -> None:
        now = time.monotonic()
        if now - last_report[0] < 0.5:
            return
        last_report[0] = now
        pct = bytes_done / total * 100 if total else 0
        elapsed = now - start_time
        rate = bytes_done / elapsed if elapsed > 0 else 0
        eta = (total - bytes_done) / rate if rate > 0 else 0
        sys.stdout.write(
            f"\r  {pct:5.1f}%  {_format_size(bytes_done)}/{_format_size(total)}"
            f"  {_format_size(int(rate))}/s  ETA {eta:.0f}s   "
        )
        sys.stdout.flush()

    cb = progress if args.progress else None

    parser.extract(
        track_numbers=track_numbers,
        track_types=track_types,
        output=args.output,
        format=fmt,
        progress_callback=cb,
        strategy=args.strategy,
    )

    elapsed = time.monotonic() - start_time
    if args.progress:
        sys.stdout.write("\r" + " " * 70 + "\r")
    import os
    out_size = os.path.getsize(args.output)
    print(f"Done in {elapsed:.1f}s. Output size: {_format_size(out_size)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fast-mkv-parser",
        description="Sparse MKV track extraction — reads only wanted track data, "
                    "skipping video payloads via lseek.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # info
    info_p = subparsers.add_parser("info", help="Show track listing")
    info_p.add_argument("input", help="MKV file path")

    # extract
    ext_p = subparsers.add_parser("extract", help="Extract selected tracks")
    ext_p.add_argument("input", help="MKV file path")
    ext_p.add_argument("-o", "--output", required=True, help="Output file path")
    ext_p.add_argument(
        "-t", "--tracks",
        help="Track types to extract (comma-separated): audio,subtitle,video",
    )
    ext_p.add_argument(
        "--track-numbers",
        help="Specific track numbers to extract (comma-separated): 2,5",
    )
    ext_p.add_argument(
        "-f", "--format",
        choices=["mkv", "sup"],
        default=None,
        help="Output format (default: inferred from extension)",
    )
    ext_p.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bar",
    )
    ext_p.add_argument(
        "--strategy",
        choices=["auto", "batched", "single-pass"],
        default="auto",
        help="Extraction strategy: auto (batched for >1 GB, single-pass otherwise), "
             "batched (two-phase scan+fetch, optimized for NFS), "
             "single-pass (original interleaved read/skip). Default: auto.",
    )

    args = parser.parse_args()

    if args.command == "info":
        cmd_info(args)
    elif args.command == "extract":
        cmd_extract(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
