# fast-mkv-parser

Sparse MKV track extraction library and CLI. Reads only block headers and desired track payloads from Matroska containers, skipping unwanted data (e.g. video) via `lseek`. Designed for large files on network storage where reading every byte is prohibitively slow.

## The Problem

MKV (Matroska) containers interleave all tracks within Clusters — video, audio, and subtitle blocks are stored sequentially in the same data stream. Every existing tool (ffmpeg, mkvextract, all known Python/Rust/Go libraries) reads **every block payload** before checking the track number and discarding unwanted data.

For an 80 GB UHD Blu-ray rip where you only need the audio track (~4.5 GB) and subtitles (~40 MB), conventional tools transfer the entire 80 GB from storage — even though 95% of it is video data you don't need.

## How It Works

`fast-mkv-parser` exploits the EBML block header structure: each SimpleBlock begins with a track number (1–4 bytes) followed by the payload. By reading just the header, the parser determines whether the block belongs to a wanted track. If not, it calls `lseek()` to skip the payload — **no I/O occurs for skipped data**, even on NFS.

Additionally, `posix_fadvise(FADV_RANDOM)` is used to disable kernel read-ahead, preventing the OS from speculatively fetching video data that will never be read.

```
Source MKV Cluster layout:

[Cluster Header] [Video 2MB] [Audio 2KB] [Audio 2KB] [Video 1.5MB] [Audio 2KB] [Sub 500B] ...
                  ^^^^^^^^^^                          ^^^^^^^^^^^^^
                  lseek (skip)                        lseek (skip)
```

## Performance

Benchmarks on an 80 GB UHD MKV over NFS on a 1 Gbps LAN:

| Method | Time | Data Transferred | Notes |
|--------|------|-----------------|-------|
| **ffmpeg** (full demux) | ~7 min | ~80 GB (entire file) | Reads all blocks, discards video in userspace |
| **fast-mkv-parser** v0.1 (no fadvise) | 3m 36s | ~27 GB (~34% of file) | Sparse seeking, but NFS read-ahead wastes bandwidth |
| **fast-mkv-parser** v0.1 (with fadvise) | **57s** | ~6 GB (~8% of file) | FADV_RANDOM disables read-ahead; only wanted data is transferred |
| Theoretical minimum | ~48s | ~4.5 GB (audio only) | Limited by 1 Gbps wire speed |

**Subtitle-only extraction** (40 MB PGS track from the same 80 GB file): **16 seconds**.

### Summary

- **~7x faster** than ffmpeg for audio extraction from large MKV files over NFS
- **~13x less I/O** than conventional tools
- Approaches wire-speed theoretical minimum on 1 Gbps networks

On local storage (SSD/NVMe), the improvement is smaller since local I/O is fast regardless, but sparse reading still reduces unnecessary wear and memory pressure.

### Extraction Strategies

For files >1 GB, the default `auto` strategy uses **two-phase batched extraction**:

1. **Phase 1 (scan)**: Parallel threads walk the file's Cluster headers, recording the offset and size of every wanted block. Each thread opens its own file descriptor with `FADV_RANDOM` to avoid read-ahead waste. Cue-based range partitioning ensures threads scan non-overlapping byte ranges.
2. **Phase 2 (fetch)**: A single thread reads the wanted block payloads sequentially, merging adjacent blocks into contiguous reads to minimize NFS round trips.

For files ≤1 GB, the simpler single-pass strategy is used (interleaved read/skip in one pass).

### NFS Tuning

Two parameters significantly affect NFS extraction speed: **Python I/O buffer size** and **parallel scan workers**. Both are tunable via the CLI (`--workers`) and library API (`scan_workers`).

**Buffer size** (compile-time, default 32 KB): Controls the Python `buffering` parameter on file descriptors, which determines how much data each NFS READ RPC fetches. A larger buffer batches multiple EBML element reads into fewer NFS round trips. Measured impact on a Synology NAS over 1 Gbps LAN:

| Buffer | Phase 1 kB/op | Phase 1 MB/s | Phase 2 MB/s |
|--------|--------------|-------------|-------------|
| 8 KB | ~12 | ~26 | ~10 |
| 32 KB | ~36 | ~60–70 | ~23 |

The 8 KB→32 KB jump is large because it captures the typical EBML read sequence (cluster header + timestamp + block header ≈ 26 bytes) in a single NFS RPC instead of 2–3. Going to 64 KB shows diminishing returns: Phase 1 seeks past video payloads (50 KB–2 MB) after each header peek, invalidating the buffer regardless of size.

**Worker count** (default 8 for batched strategy): Controls Phase 1 parallelism. More workers issue concurrent NFS RPCs (the GIL is released during I/O), but push up NFS server RTT as the disk queue depth increases:

| Workers | RTT | Phase 1 ops/s | Phase 1 MB/s |
|---------|-----|--------------|-------------|
| 4 | 2.4 ms | 1,645 | ~59 |
| 8 | 3.9 ms | 1,950 | ~70 |

8 workers is ~19% faster than 4 despite doubling RTT, because the additional concurrency more than compensates. Beyond 8, the NAS disk queue saturates with diminishing returns (~5–10% estimated for 16 workers). The optimal value depends on your NFS server's storage backend — SSDs tolerate more workers than spinning disks.

Phase 2 (payload fetch) is always single-threaded since it reads large contiguous regions where bandwidth, not latency, is the bottleneck.

## Installation

```bash
pip install -e /path/to/fast-mkv-parser
```

No external dependencies — pure Python stdlib (`struct`, `os`, `io`, `argparse`).

Requires Python 3.10+.

## CLI Usage

### List tracks

```bash
fast-mkv-parser info media.mkv
```

```
File: media.mkv
Duration: 2:55:28 (10528.1s)
TimecodeScale: 1000000 ns

#    Type       Codec                     Lang   Name
----------------------------------------------------------------------
1    video      V_MPEGH/ISO/HEVC          und
2    audio      A_TRUEHD                  eng    Surround 5.1
3    audio      A_AC3                     eng    Surround 5.1
6    subtitle   S_HDMV/PGS                eng
```

### Extract tracks by type

```bash
# Audio + subtitles to a minimal MKV
fast-mkv-parser extract media.mkv -t audio,subtitle -o stripped.mkv --progress

# Audio only to MKA container
fast-mkv-parser extract media.mkv -t audio -o audio.mka --progress

# Subtitles only
fast-mkv-parser extract media.mkv -t subtitle -o subs.mkv
```

### Extract specific track numbers

```bash
fast-mkv-parser extract media.mkv --track-numbers 2,6 -o selected.mkv
```

### Extract PGS subtitles as Blu-ray SUP format

```bash
fast-mkv-parser extract media.mkv -t subtitle -f sup -o subs.sup
```

The SUP output is compatible with any tool that reads Blu-ray PGS subtitle files (e.g. BDSup2Sub, Tesseract-based OCR pipelines).

### Extraction strategy and worker count

```bash
# Force two-phase batched extraction (default for >1 GB files)
fast-mkv-parser extract media.mkv -t audio -o audio.mka --strategy batched --progress

# Override parallel scan workers (default: 8)
fast-mkv-parser extract media.mkv -t audio -o audio.mka --workers 4 --progress

# Single-threaded single-pass (no parallel scan)
fast-mkv-parser extract media.mkv -t audio -o audio.mka --strategy single-pass --progress
```

### Default behavior

If no `-t` or `--track-numbers` is specified, all non-video tracks are extracted.

## Library API

```python
from fast_mkv_parser import MkvParser

parser = MkvParser("/path/to/media.mkv")

# Inspect tracks
for track in parser.tracks:
    print(f"Track {track.number}: {track.type} ({track.codec_id})")

# Extract audio + subtitles to minimal MKV
parser.extract(track_types=["audio", "subtitle"], output="stripped.mkv")

# Extract with progress reporting
def on_progress(bytes_done, total_bytes):
    print(f"{bytes_done / total_bytes * 100:.1f}%")

parser.extract(track_types=["audio"], output="audio.mka", progress_callback=on_progress)

# Override scan workers (default: 8 for batched, 1 for single-pass)
parser.extract(track_types=["audio"], output="audio.mka", scan_workers=4)

# Convenience methods
parser.extract_audio("audio.mka")
parser.extract_subtitles("subs.sup", format="sup")
```

### Track filtering options

| Parameter | Example | Description |
|-----------|---------|-------------|
| `track_types` | `["audio", "subtitle"]` | Extract all tracks of these types |
| `track_numbers` | `[2, 6]` | Extract specific track numbers |
| Neither | — | Extracts all non-video tracks (default) |

### Output formats

| Format | Extension | Description |
|--------|-----------|-------------|
| `"mkv"` | `.mkv`, `.mka`, `.mks` | Valid Matroska container with selected tracks only |
| `"sup"` | `.sup` | Blu-ray PGS subtitle format (reconstructed from MKV timestamps) |

## How the Output Compares to ffmpeg

The extracted MKA/MKV output is a valid Matroska container that ffmpeg, mkvtoolnix, and any standards-compliant player can read. The audio codec data is copied byte-for-byte — no transcoding or quality loss.

```bash
# Verify output with ffprobe
ffprobe stripped.mkv

# Decode extracted audio to WAV (if needed for processing)
ffmpeg -i audio.mka -acodec pcm_s16le -ac 1 -ar 16000 audio.wav
```

## Limitations

- **Read-only**: Extracts tracks from existing MKV files. Does not mux or remux.
- **Sequential Cluster walking**: Processes Clusters in file order. MKV files without Cues (seek index) still require walking through the entire file's block headers, though video payloads are skipped.
- **No video codec awareness**: Does not parse codec-specific data (NAL units, frame types). Treats all block payloads as opaque byte sequences.
- **Python performance**: The EBML header parsing is pure Python. For files with millions of blocks, the CPU overhead is ~3-4 seconds. The I/O savings far outweigh this cost for network storage.
